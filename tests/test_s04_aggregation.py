from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.eval.run_s04_aggregation import run_aggregation
from src.training.train_cil import FIXED_BUDGET_METHOD


class S04AggregationTest(unittest.TestCase):
    def _write_run(self, root: Path) -> Path:
        run = root / f"random_seed0_{FIXED_BUDGET_METHOD}_mlpbase"
        (run / "s02").mkdir(parents=True)
        (run / "run_config.json").write_text(
            json.dumps(
                {
                    "run_id": "run-0",
                    "arguments": {"seed": 0, "method": FIXED_BUDGET_METHOD},
                    "resolved": {
                        "method_protocol": FIXED_BUDGET_METHOD,
                        "num_tasks": 2,
                        "total_memory_budget": 4,
                        "replay_draws_per_epoch": 4,
                    },
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"event_type": "run_started"},
                {"event_type": "run_completed"},
            ]
        ).to_csv(run / "events.csv", index=False)
        (run / "s02/manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "run-0",
                    "seed": 0,
                    "method": FIXED_BUDGET_METHOD,
                    "exports": [
                        {"train_task": task, "split": split}
                        for task in range(2)
                        for split in ("validation", "test")
                    ],
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "train_task_id": task,
                    "eval_task_id": "seen_all",
                    "accuracy": 0.7 + task / 10,
                    "macro_f1": 0.6 + task / 10,
                    "weighted_f1": 0.65 + task / 10,
                    "balanced_accuracy": 0.62 + task / 10,
                }
                for task in range(2)
            ]
        ).to_csv(run / "metrics.csv", index=False)
        pd.DataFrame(
            [
                {"task_id": 0, "forgetting": 0.1},
                {"task_id": "mean_old_tasks", "forgetting": 0.1},
            ]
        ).to_csv(run / "forgetting.csv", index=False)
        pd.DataFrame(
            [
                {"train_task": 1, "class_id": 1, "current_f1": 0.0, "forgetting": 0.4},
                {"train_task": 1, "class_id": 2, "current_f1": 0.8, "forgetting": 0.0},
            ]
        ).to_csv(run / "class_forgetting.csv", index=False)
        pd.DataFrame(
            [
                {
                    "run_id": "run-0",
                    "task": 0,
                    "raw_class_id": 1,
                    "available_samples": 3,
                    "total_memory_after": 3,
                    "actual_replay_draws_per_epoch": 0,
                    "epochs_trained": 2,
                    "replay_draws_total": 0,
                },
                {
                    "run_id": "run-0",
                    "task": 1,
                    "raw_class_id": 1,
                    "available_samples": 3,
                    "total_memory_after": 4,
                    "actual_replay_draws_per_epoch": 4,
                    "epochs_trained": 2,
                    "replay_draws_total": 8,
                },
                {
                    "run_id": "run-0",
                    "task": 1,
                    "raw_class_id": 2,
                    "available_samples": 3,
                    "total_memory_after": 4,
                    "actual_replay_draws_per_epoch": 4,
                    "epochs_trained": 2,
                    "replay_draws_total": 0,
                },
            ]
        ).to_csv(run / "replay_budget_audit.csv", index=False)
        return run

    def test_writes_one_o07_row_with_final_metrics_and_forgetting(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            runs = root / "runs"
            runs.mkdir()
            self._write_run(runs)
            output, manifest = run_aggregation(
                argparse.Namespace(
                    runs_root=runs,
                    outdir=root / "s04",
                    expected_seeds=[0],
                    expected_tasks=2,
                    total_memory_budget=4,
                    replay_draws_per_epoch=4,
                    allow_partial=False,
                    overwrite=False,
                )
            )
            frame = pd.read_csv(output)
            payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(frame.shape[0], 1)
        self.assertAlmostEqual(float(frame.iloc[0]["macro_f1"]), 0.7)
        self.assertAlmostEqual(float(frame.iloc[0]["class_forgetting"]), 0.2)
        self.assertEqual(int(frame.iloc[0]["zero_f1_classes"]), 1)
        self.assertEqual(payload["completed_seeds"], [0])


if __name__ == "__main__":
    unittest.main()
