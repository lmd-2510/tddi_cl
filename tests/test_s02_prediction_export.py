from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.eval.cil_evaluation import PredictionOutputs, evaluate_model
from src.eval.s02_artifacts import (
    DRUG_ID_A_COLUMN,
    DRUG_ID_B_COLUMN,
    S02ExportContext,
    export_s02_artifacts,
    validate_s02_artifacts,
)
from src.models.mlp import MLP, MLPConfig


def _prediction_outputs() -> PredictionOutputs:
    logits = np.asarray(
        [[2.0, 0.0], [0.0, 2.0], [1.5, -0.5]],
        dtype=np.float32,
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    class_ids = np.asarray([10, 30], dtype=np.int64)
    return PredictionOutputs(
        logits=logits,
        probabilities=probabilities.astype(np.float32, copy=False),
        predictions=class_ids[probabilities.argmax(axis=1)],
        labels=np.asarray([10, 30, 10], dtype=np.int64),
        latent_features=np.asarray(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            dtype=np.float32,
        ),
        class_ids=class_ids,
    )


def _metadata() -> dict[str, np.ndarray]:
    return {
        DRUG_ID_A_COLUMN: np.asarray(["DB001", "DB002", "DB003"]),
        DRUG_ID_B_COLUMN: np.asarray(["DB101", "DB102", "DB103"]),
    }


class MLPExportInterfaceTest(unittest.TestCase):
    def test_latent_api_preserves_forward_and_checkpoint_keys(self) -> None:
        torch.manual_seed(4)
        config = MLPConfig(
            input_dim=5,
            hidden_dims=(7, 4),
            num_classes=3,
            dropout=0.0,
            norm="none",
        )
        model = MLP(config).eval()
        features = torch.randn(6, 5)

        with torch.no_grad():
            forward_logits = model(features)
            logits, latent = model.forward_with_latent(features)

        torch.testing.assert_close(forward_logits, logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(model.classify(model.encode(features)), logits)
        self.assertEqual(tuple(latent.shape), (6, 4))
        self.assertTrue(all(key.startswith(("backbone.", "head.")) for key in model.state_dict()))

        restored = MLP(config)
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored.eval()(features), forward_logits)


class EvaluatorOutputTest(unittest.TestCase):
    def test_collects_aligned_outputs_for_noncontiguous_raw_classes(self) -> None:
        torch.manual_seed(9)
        model = MLP(
            MLPConfig(
                input_dim=4,
                hidden_dims=(6,),
                num_classes=3,
                dropout=0.0,
                norm="none",
            )
        )
        features = torch.randn(7, 4)
        local_labels = torch.tensor([0, 2, 1, 0, 1, 2, 2], dtype=torch.long)
        loader = DataLoader(TensorDataset(features, local_labels), batch_size=3, shuffle=False)
        result = evaluate_model(
            model,
            loader,
            nn.CrossEntropyLoss(),
            "cpu",
            {0: 10, 1: 30, 2: 70},
            evaluation_class_indices=[0, 1, 2],
            include_classwise=True,
            collect_outputs=True,
        )

        self.assertIsNotNone(result.classwise_metrics)
        self.assertIsNotNone(result.outputs)
        outputs = result.outputs
        assert outputs is not None
        np.testing.assert_array_equal(outputs.class_ids, [10, 30, 70])
        np.testing.assert_array_equal(outputs.labels, [10, 70, 30, 10, 30, 70, 70])
        self.assertEqual(outputs.logits.shape, (7, 3))
        self.assertEqual(outputs.latent_features.shape, (7, 6))
        self.assertEqual(outputs.logits.dtype, np.float32)
        np.testing.assert_allclose(outputs.probabilities.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(
            outputs.predictions,
            outputs.class_ids[outputs.probabilities.argmax(axis=1)],
        )

    def test_default_evaluation_does_not_collect_raw_arrays(self) -> None:
        model = MLP(MLPConfig(input_dim=2, hidden_dims=(3,), num_classes=2, dropout=0.0))
        loader = DataLoader(
            TensorDataset(torch.zeros(2, 2), torch.tensor([0, 1])),
            batch_size=2,
        )
        result = evaluate_model(model, loader, nn.CrossEntropyLoss(), "cpu", {0: 4, 1: 9})
        self.assertIsNone(result.outputs)

    def test_evaluation_does_not_advance_training_rng(self) -> None:
        model = MLP(MLPConfig(input_dim=2, hidden_dims=(3,), num_classes=2, dropout=0.0))
        loader = DataLoader(
            TensorDataset(torch.zeros(2, 2), torch.tensor([0, 1])),
            batch_size=2,
            shuffle=False,
        )
        torch.manual_seed(123)
        state_before = torch.random.get_rng_state().clone()
        evaluate_model(
            model,
            loader,
            nn.CrossEntropyLoss(),
            "cpu",
            {0: 4, 1: 9},
            collect_outputs=True,
        )
        torch.testing.assert_close(torch.random.get_rng_state(), state_before)


class S02ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.checkpoint = self.root / "task_0_model.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.run_config = self.root / "run_config.json"
        self.run_config.write_text(
            json.dumps(
                {
                    "run_id": "run-abc",
                    "arguments": {"seed": 17, "method": "replay_distill"},
                    "resolved": {"method_protocol": "replay_distill_balanced_per_class_cap50"},
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def context(self, *, split: str = "validation", run_id: str = "run-abc") -> S02ExportContext:
        return S02ExportContext(
            run_id=run_id,
            seed=17,
            method="replay_distill",
            method_protocol="replay_distill_balanced_per_class_cap50",
            train_task=0,
            split=split,
            checkpoint_path=self.checkpoint,
            run_config_path=self.run_config,
        )

    def test_parquet_npz_and_manifest_round_trip(self) -> None:
        paths = export_s02_artifacts(
            _prediction_outputs(),
            _metadata(),
            self.context(),
            self.root / "s02",
        )
        summary = validate_s02_artifacts(
            paths.predictions_path,
            paths.latent_features_path,
            expected_context=self.context(),
        )
        self.assertEqual((summary.row_count, summary.class_count, summary.latent_dim), (3, 2, 4))

        table = pq.read_table(paths.predictions_path)
        self.assertEqual(table["confidence"].type, pa.float32())
        self.assertEqual(table["label"].type, pa.int64())
        self.assertTrue(pa.types.is_fixed_size_list(table["logits"].type))
        self.assertEqual(table["logits"].type.list_size, 2)
        with np.load(paths.latent_features_path, allow_pickle=False) as payload:
            self.assertEqual(payload["latent_features"].dtype, np.float32)
            self.assertEqual(payload["labels"].dtype, np.int64)
            self.assertEqual(payload["run_id"].item(), "run-abc")

        manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["run_id"], "run-abc")
        self.assertEqual(len(manifest["exports"]), 1)
        self.assertEqual(manifest["exports"][0]["class_ids"], [10, 30])
        self.assertEqual(manifest["exports"][0]["row_count"], 3)

    def test_manifest_accumulates_independent_shards_and_rejects_overwrite(self) -> None:
        output_root = self.root / "s02"
        export_s02_artifacts(_prediction_outputs(), _metadata(), self.context(), output_root)
        export_s02_artifacts(
            _prediction_outputs(),
            _metadata(),
            self.context(split="test"),
            output_root,
        )
        manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [(row["train_task"], row["split"]) for row in manifest["exports"]],
            [(0, "test"), (0, "validation")],
        )
        with self.assertRaises(FileExistsError):
            export_s02_artifacts(_prediction_outputs(), _metadata(), self.context(), output_root)

    def test_rejects_duplicate_pair_and_provenance_mismatch(self) -> None:
        duplicate_metadata = _metadata()
        duplicate_metadata[DRUG_ID_A_COLUMN][1] = duplicate_metadata[DRUG_ID_A_COLUMN][0]
        duplicate_metadata[DRUG_ID_B_COLUMN][1] = duplicate_metadata[DRUG_ID_B_COLUMN][0]
        with self.assertRaisesRegex(ValueError, "unique"):
            export_s02_artifacts(
                _prediction_outputs(),
                duplicate_metadata,
                self.context(),
                self.root / "duplicate",
            )
        with self.assertRaisesRegex(ValueError, "run_id"):
            export_s02_artifacts(
                _prediction_outputs(),
                _metadata(),
                self.context(run_id="wrong-run"),
                self.root / "wrong-run",
            )

    def test_rejects_bad_dtype_shape_alignment_and_nonfinite_values(self) -> None:
        valid = _prediction_outputs()
        invalid_cases = [
            replace(valid, logits=valid.logits.astype(np.float64)),
            replace(valid, class_ids=np.asarray([10], dtype=np.int64)),
            replace(valid, labels=np.asarray([10, 30], dtype=np.int64)),
            replace(
                valid,
                latent_features=np.asarray(
                    [[np.nan, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    dtype=np.float32,
                ),
            ),
            replace(
                valid,
                probabilities=np.asarray(
                    [[np.inf, 0], [0, 1], [1, 0]],
                    dtype=np.float32,
                ),
            ),
        ]
        for index, outputs in enumerate(invalid_cases):
            with self.subTest(index=index), self.assertRaises((TypeError, ValueError)):
                export_s02_artifacts(
                    outputs,
                    _metadata(),
                    self.context(),
                    self.root / f"invalid-{index}",
                )

    def test_validation_detects_o03_o04_label_mismatch(self) -> None:
        paths = export_s02_artifacts(
            _prediction_outputs(),
            _metadata(),
            self.context(),
            self.root / "s02",
        )
        with np.load(paths.latent_features_path, allow_pickle=False) as payload:
            rewritten = {key: payload[key] for key in payload.files}
        rewritten["labels"] = np.asarray([30, 30, 10], dtype=np.int64)
        with paths.latent_features_path.open("wb") as handle:
            np.savez_compressed(handle, **rewritten)

        with self.assertRaisesRegex(ValueError, "labels are not aligned"):
            validate_s02_artifacts(paths.predictions_path, paths.latent_features_path)


if __name__ == "__main__":
    unittest.main()
