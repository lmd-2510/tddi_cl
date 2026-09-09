from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ddi_dataset import load_class_counts
from src.eval.classwise_metrics import (
    CLASS_METRIC_COLUMNS,
    FORGETTING_COLUMNS,
    TRAJECTORY_COLUMNS,
    ClasswiseTracker,
    compute_classwise_metrics,
)


def metric_frame(rows: list[dict[str, float | int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CLASS_METRIC_COLUMNS)


class ComputeClasswiseMetricsTest(unittest.TestCase):
    def test_metrics_confidence_entropy_mapping_and_zero_support(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array(
            [
                [0.80, 0.15, 0.05],
                [0.30, 0.60, 0.10],
                [0.10, 0.80, 0.10],
                [0.20, 0.70, 0.10],
            ]
        )
        y_pred = probabilities.argmax(axis=1)

        result = compute_classwise_metrics(
            y_true,
            y_pred,
            probabilities,
            class_indices=[0, 1, 2],
            inverse_class_map={0: 10, 1: 20, 2: 30},
        ).set_index("class_id")

        self.assertEqual(result.loc[10, "test_count"], 2)
        self.assertAlmostEqual(result.loc[10, "precision"], 1.0)
        self.assertAlmostEqual(result.loc[10, "recall"], 0.5)
        self.assertAlmostEqual(result.loc[10, "f1"], 2.0 / 3.0)
        self.assertAlmostEqual(result.loc[10, "confidence"], 0.7)

        self.assertEqual(result.loc[20, "test_count"], 2)
        self.assertAlmostEqual(result.loc[20, "precision"], 2.0 / 3.0)
        self.assertAlmostEqual(result.loc[20, "recall"], 1.0)
        self.assertAlmostEqual(result.loc[20, "f1"], 0.8)
        self.assertAlmostEqual(result.loc[20, "confidence"], 0.75)

        expected_entropy_class_10 = np.mean(
            -np.sum(probabilities[:2] * np.log(probabilities[:2]), axis=1)
        )
        self.assertAlmostEqual(result.loc[10, "entropy"], expected_entropy_class_10)

        self.assertEqual(result.loc[30, "test_count"], 0)
        self.assertEqual(result.loc[30, "precision"], 0.0)
        self.assertEqual(result.loc[30, "recall"], 0.0)
        self.assertEqual(result.loc[30, "f1"], 0.0)
        self.assertTrue(np.isnan(result.loc[30, "confidence"]))
        self.assertTrue(np.isnan(result.loc[30, "entropy"]))

    def test_rejects_missing_class_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing raw class IDs"):
            compute_classwise_metrics(
                np.array([0]),
                np.array([0]),
                np.array([[1.0, 0.0]]),
                class_indices=[0, 1],
                inverse_class_map={0: 10},
            )


class ClasswiseTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = ClasswiseTracker(
            seed=7,
            method="replay_distill_fixed_budget_uniform",
            first_task_by_class={10: 0, 20: 1},
            train_count_by_class={10: 100, 20: 5},
        )

    def add_example_tasks(self) -> None:
        self.tracker.add_task(
            0,
            metric_frame(
                [
                    {
                        "class_id": 10,
                        "test_count": 8,
                        "precision": 0.6,
                        "recall": 0.6,
                        "f1": 0.6,
                        "confidence": 0.7,
                        "entropy": 0.5,
                    }
                ]
            ),
        )
        self.tracker.add_task(
            1,
            metric_frame(
                [
                    {
                        "class_id": 10,
                        "test_count": 8,
                        "precision": 0.4,
                        "recall": 0.4,
                        "f1": 0.4,
                        "confidence": 0.6,
                        "entropy": 0.6,
                    },
                    {
                        "class_id": 20,
                        "test_count": 3,
                        "precision": 0.8,
                        "recall": 0.8,
                        "f1": 0.8,
                        "confidence": 0.85,
                        "entropy": 0.3,
                    },
                ]
            ),
        )
        self.tracker.add_task(
            2,
            metric_frame(
                [
                    {
                        "class_id": 10,
                        "test_count": 8,
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                        "confidence": 0.65,
                        "entropy": 0.55,
                    },
                    {
                        "class_id": 20,
                        "test_count": 3,
                        "precision": 0.3,
                        "recall": 0.3,
                        "f1": 0.3,
                        "confidence": 0.7,
                        "entropy": 0.7,
                    },
                ]
            ),
        )

    def test_adds_metadata_and_computes_cumulative_forgetting(self) -> None:
        self.add_example_tasks()
        trajectory = self.tracker.trajectory_frame()
        forgetting = self.tracker.forgetting_frame()

        self.assertEqual(trajectory.columns.tolist(), TRAJECTORY_COLUMNS)
        self.assertEqual(forgetting.columns.tolist(), FORGETTING_COLUMNS)
        self.assertEqual(trajectory[["train_task", "class_id"]].values.tolist(), [
            [0, 10],
            [1, 10],
            [1, 20],
            [2, 10],
            [2, 20],
        ])
        self.assertEqual(trajectory.loc[trajectory["class_id"] == 20, "first_task"].unique().tolist(), [1])
        self.assertEqual(trajectory.loc[trajectory["class_id"] == 20, "train_count"].unique().tolist(), [5])

        class_10_final = forgetting[
            (forgetting["class_id"] == 10) & (forgetting["train_task"] == 2)
        ].iloc[0]
        self.assertAlmostEqual(class_10_final["best_f1"], 0.6)
        self.assertAlmostEqual(class_10_final["current_f1"], 0.5)
        self.assertAlmostEqual(class_10_final["forgetting"], 0.1)

        class_20_first = forgetting[
            (forgetting["class_id"] == 20) & (forgetting["train_task"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(class_20_first["forgetting"], 0.0)
        class_20_final = forgetting[
            (forgetting["class_id"] == 20) & (forgetting["train_task"] == 2)
        ].iloc[0]
        self.assertAlmostEqual(class_20_final["forgetting"], 0.5)

    def test_rejects_duplicate_keys_and_missing_metadata(self) -> None:
        first_task_metrics = metric_frame(
            [
                {
                    "class_id": 10,
                    "test_count": 1,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "confidence": 1.0,
                    "entropy": 0.0,
                }
            ]
        )
        self.tracker.add_task(0, first_task_metrics)
        with self.assertRaisesRegex(ValueError, "Duplicate class trajectory keys"):
            self.tracker.add_task(0, first_task_metrics)

        missing_metadata_tracker = ClasswiseTracker(
            seed=0,
            method="replay_distill_fixed_budget_uniform",
            first_task_by_class={10: 0},
            train_count_by_class={},
        )
        with self.assertRaisesRegex(ValueError, "Missing train counts"):
            missing_metadata_tracker.add_task(0, first_task_metrics)

    def test_save_uses_exact_csv_schema_without_index(self) -> None:
        self.add_example_tasks()
        with tempfile.TemporaryDirectory() as tempdir:
            trajectory_path, forgetting_path = self.tracker.save(tempdir)
            trajectory = pd.read_csv(trajectory_path)
            forgetting = pd.read_csv(forgetting_path)

        self.assertEqual(trajectory.columns.tolist(), TRAJECTORY_COLUMNS)
        self.assertEqual(forgetting.columns.tolist(), FORGETTING_COLUMNS)
        self.assertNotIn("Unnamed: 0", trajectory.columns)
        self.assertNotIn("Unnamed: 0", forgetting.columns)


class LoadClassCountsTest(unittest.TestCase):
    def test_counts_full_parquet_label_column(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "train.parquet"
            pd.DataFrame(
                {
                    "class": [10, 10, 20, 30, 30, 30],
                    "unused_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                }
            ).to_parquet(path, index=False)
            counts = load_class_counts(path)

        self.assertEqual(counts, {10: 2, 20: 1, 30: 3})


if __name__ == "__main__":
    unittest.main()
