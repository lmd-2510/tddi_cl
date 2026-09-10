from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.train_cil import (
    build_student_old_indices,
    ordered_raw_classes,
)
from src.utils.logging import RunLogger, ensure_run_paths
from tests.tiny_tddi import expand_tiny_tddi


class ClassAlignmentRegressionTest(unittest.TestCase):
    def test_expansion_preserves_old_logits_after_indices_move(self) -> None:
        old_map = {6: 0, 11: 1, 41: 2}
        current_map = {1: 0, 6: 1, 11: 2, 30: 3, 41: 4}
        torch.manual_seed(7)
        previous_model = expand_tiny_tddi(
            previous_model=None,
            previous_seen_map=None,
            current_seen_map=old_map,
            variant="tddi_paper_member",
            input_dim=5,
            dropout=0.0,
            activation="gelu",
            norm="none",
        )
        expanded_model = expand_tiny_tddi(
            previous_model=previous_model,
            previous_seen_map=old_map,
            current_seen_map=current_map,
            variant="tddi_paper_member",
            input_dim=5,
            dropout=0.0,
            activation="gelu",
            norm="none",
        )
        previous_model.eval()
        expanded_model.eval()
        features = torch.randn(8, 5)

        teacher_raw_classes = ordered_raw_classes(old_map)
        student_old_indices = build_student_old_indices(
            teacher_raw_classes,
            current_map,
        )
        with torch.no_grad():
            teacher_logits = previous_model(features)
            aligned_student_logits = expanded_model(features)[:, student_old_indices]

        self.assertEqual(teacher_raw_classes, [6, 11, 41])
        self.assertEqual(student_old_indices, [1, 2, 4])
        # Different classifier widths may select different GEMM kernels; the
        # copied rows remain equivalent up to normal float32 round-off.
        torch.testing.assert_close(
            teacher_logits,
            aligned_student_logits,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_rejects_invalid_teacher_class_order_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            build_student_old_indices([6, 6], {6: 0})
        with self.assertRaisesRegex(ValueError, "missing"):
            build_student_old_indices([6, 11], {6: 0})
        with self.assertRaisesRegex(ValueError, "dense"):
            ordered_raw_classes({6: 0, 11: 2})


class RunProvenanceTest(unittest.TestCase):
    def test_nonempty_run_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            outdir = Path(tempdir) / "run"
            outdir.mkdir()
            (outdir / "events.csv").write_text("old run\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "must be empty"):
                ensure_run_paths(outdir, require_empty=True)

    def test_empty_run_directory_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            outdir = Path(tempdir) / "run"
            outdir.mkdir()
            paths = ensure_run_paths(outdir, require_empty=True)

            self.assertEqual(paths["outdir"], outdir)
            self.assertEqual(paths["run_config_json"], outdir / "run_config.json")
            self.assertEqual(paths["training_audit_csv"], outdir / "training_audit.csv")

    def test_events_are_bound_to_one_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            outdir = Path(tempdir)
            events_path = outdir / "events.csv"
            logger = RunLogger(
                log_path=outdir / "train.log",
                events_path=events_path,
                run_id="run-123",
            )
            logger.event("run_started", "started")
            logger.event("run_completed", "completed")

            with events_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([row["run_id"] for row in rows], ["run-123", "run-123"])
        self.assertEqual(
            [row["event_type"] for row in rows],
            ["run_started", "run_completed"],
        )


if __name__ == "__main__":
    unittest.main()
