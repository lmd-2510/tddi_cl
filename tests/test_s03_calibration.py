from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.eval.calibration_metrics import (
    TemperatureFitResult,
    compute_calibration_metrics,
    fit_temperature,
    map_raw_labels_to_indices,
    softmax_probabilities,
)
from src.eval.cil_evaluation import PredictionOutputs
from src.eval.run_s03_calibration import calibrate_task, run, validate_manifest
from src.eval.s02_artifacts import (
    DRUG_ID_A_COLUMN,
    DRUG_ID_B_COLUMN,
    S02ExportContext,
    export_s02_artifacts,
)


class CalibrationMetricsTest(unittest.TestCase):
    def test_maps_noncontiguous_raw_labels_in_logit_order(self) -> None:
        mapped = map_raw_labels_to_indices(
            np.asarray([30, 10, 70, 30]),
            np.asarray([10, 70, 30]),
        )
        np.testing.assert_array_equal(mapped, [2, 0, 1, 2])
        with self.assertRaisesRegex(ValueError, "absent from class_ids"):
            map_raw_labels_to_indices(np.asarray([99]), np.asarray([10, 30]))

    def test_metrics_match_direct_probability_computation(self) -> None:
        logits = np.asarray(
            [[3.0, 0.0, -1.0], [0.2, 0.1, 0.0], [-1.0, 1.0, 2.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1, 2], dtype=np.int64)
        probabilities = softmax_probabilities(logits)
        metrics = compute_calibration_metrics(
            logits,
            labels,
            num_bins=5,
            high_confidence_threshold=0.8,
            batch_size=2,
        )
        rows = np.arange(labels.shape[0])
        expected_nll = -np.log(probabilities[rows, labels]).mean()
        expected_brier = np.mean(
            np.sum(probabilities * probabilities, axis=1)
            - 2.0 * probabilities[rows, labels]
            + 1.0
        )
        self.assertAlmostEqual(metrics.negative_log_likelihood, expected_nll)
        self.assertAlmostEqual(metrics.brier_score, expected_brier)
        self.assertAlmostEqual(metrics.mean_confidence, probabilities.max(axis=1).mean())
        self.assertEqual(metrics.num_samples, 3)
        self.assertEqual(metrics.num_classes, 3)

        no_high_confidence = compute_calibration_metrics(
            np.zeros((4, 2), dtype=np.float32),
            np.asarray([0, 1, 0, 1]),
            high_confidence_threshold=0.9,
        )
        self.assertEqual(no_high_confidence.high_confidence_count, 0)
        self.assertIsNone(no_high_confidence.high_confidence_error_rate)

    def test_temperature_fit_recovers_known_softening_and_reduces_nll(self) -> None:
        rng = np.random.default_rng(812)
        logits = rng.normal(size=(12000, 4)).astype(np.float32) * 2.0
        generating_temperature = 2.5
        probabilities = softmax_probabilities(logits, generating_temperature)
        labels = np.asarray(
            [rng.choice(4, p=row) for row in probabilities],
            dtype=np.int64,
        )
        fit = fit_temperature(logits, labels, tolerance=1e-8, batch_size=1024)
        raw = compute_calibration_metrics(logits, labels, batch_size=1024)
        scaled = compute_calibration_metrics(
            logits, labels, temperature=fit.temperature, batch_size=1024
        )
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.temperature, generating_temperature, delta=0.2)
        self.assertLess(scaled.negative_log_likelihood, raw.negative_log_likelihood)
        self.assertEqual(scaled.accuracy, raw.accuracy)


class S03RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.run_dir = self.root / "runs" / "seed0_replay"
        self.checkpoint = self.run_dir / "checkpoints" / "task_0_model.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"checkpoint")
        self.run_config = self.run_dir / "run_config.json"
        self.run_config.write_text(
            json.dumps(
                {
                    "run_id": "run-s03-test",
                    "arguments": {"seed": 0, "method": "replay"},
                    "resolved": {"method_protocol": "replay_balanced_per_class_cap50"},
                }
            ),
            encoding="utf-8",
        )
        self._export("validation", label_offset=0)
        self._export("test", label_offset=1)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _export(self, split: str, *, label_offset: int) -> None:
        logits = np.asarray(
            [[3.0, 0.0], [0.0, 3.0], [2.0, -1.0], [-1.0, 2.0]],
            dtype=np.float32,
        )
        probabilities = softmax_probabilities(logits).astype(np.float32)
        class_ids = np.asarray([10, 30], dtype=np.int64)
        local_labels = (np.arange(4) + label_offset) % 2
        outputs = PredictionOutputs(
            logits=logits,
            probabilities=probabilities,
            predictions=class_ids[probabilities.argmax(axis=1)],
            labels=class_ids[local_labels],
            latent_features=np.arange(12, dtype=np.float32).reshape(4, 3),
            class_ids=class_ids,
        )
        metadata = {
            DRUG_ID_A_COLUMN: np.asarray([f"A-{split}-{index}" for index in range(4)]),
            DRUG_ID_B_COLUMN: np.asarray([f"B-{split}-{index}" for index in range(4)]),
        }
        export_s02_artifacts(
            outputs,
            metadata,
            S02ExportContext(
                run_id="run-s03-test",
                seed=0,
                method="replay",
                method_protocol="replay_balanced_per_class_cap50",
                train_task=0,
                split=split,
                checkpoint_path=self.checkpoint,
                run_config_path=self.run_config,
            ),
            self.run_dir / "s02",
        )

    def test_calibration_fit_receives_validation_only(self) -> None:
        manifest_path = self.run_dir / "s02" / "manifest.json"
        manifest, export_map = validate_manifest(manifest_path)
        observed: dict[str, np.ndarray] = {}

        def capture_fit(logits: np.ndarray, labels: np.ndarray, **_: object) -> TemperatureFitResult:
            observed["logits"] = logits.copy()
            observed["labels"] = labels.copy()
            return TemperatureFitResult(1.0, 1.0, 0.0, 1, True, "none")

        with patch("src.eval.run_s03_calibration.fit_temperature", side_effect=capture_fit):
            rows = calibrate_task(
                manifest_path,
                manifest,
                export_map,
                0,
                num_bins=5,
                high_confidence_threshold=0.9,
                batch_size=2,
                min_temperature=0.05,
                max_temperature=100.0,
                max_fit_iterations=10,
                fit_tolerance=1e-7,
            )
        np.testing.assert_array_equal(observed["labels"], [0, 1, 0, 1])
        self.assertEqual({row["temperature_fit_split"] for row in rows}, {"validation"})
        self.assertEqual(len(rows), 4)

    def test_full_runner_writes_four_rows_and_provenance_manifest(self) -> None:
        outdir = self.root / "s03"
        output = run(
            argparse.Namespace(
                runs_root=self.root / "runs",
                outdir=outdir,
                num_bins=5,
                high_confidence_threshold=0.9,
                batch_size=2,
                min_temperature=0.05,
                max_temperature=100.0,
                max_fit_iterations=10,
                fit_tolerance=1e-7,
                overwrite=False,
            )
        )
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["evaluation_split"] for row in rows}, {"validation", "test"})
        self.assertEqual(
            {row["calibration_stage"] for row in rows},
            {"raw", "temperature_scaled"},
        )
        manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["temperature_fit_split"], "validation")
        self.assertEqual(manifest["test_usage"], "reporting_only")
        self.assertEqual(
            manifest["undefined_metric_encoding"],
            "empty_csv_field_with_zero_support_count",
        )
        self.assertEqual(manifest["row_count"], 4)


if __name__ == "__main__":
    unittest.main()
