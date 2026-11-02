from __future__ import annotations

import unittest
import warnings

import numpy as np

from src.eval.metrics import (
    compute_aurc,
    compute_backward_transfer,
    compute_classification_metrics,
)


class ClassificationMetricsTest(unittest.TestCase):
    def test_backward_transfer_uses_final_minus_learning_boundary(self) -> None:
        matrix = np.array(
            [
                [0.80, np.nan, np.nan],
                [0.70, 0.60, np.nan],
                [0.50, 0.65, 0.90],
            ]
        )

        result = compute_backward_transfer(matrix)

        self.assertEqual(result["task_id"].astype(str).tolist(), ["0", "1", "mean_old_tasks"])
        self.assertAlmostEqual(result.iloc[0]["backward_transfer"], -0.30)
        self.assertAlmostEqual(result.iloc[1]["backward_transfer"], 0.05)
        self.assertAlmostEqual(result.iloc[2]["backward_transfer"], -0.125)

    def test_backward_transfer_rejects_non_square_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty square"):
            compute_backward_transfer(np.zeros((2, 3)))

    def test_explicit_labels_exclude_prediction_only_classes_from_macro_average(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 2, 1, 2])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = compute_classification_metrics(
                y_true,
                y_pred,
                labels=[0, 1],
            )

        self.assertEqual(caught, [])
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["weighted_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["macro_precision"], 1.0)
        self.assertAlmostEqual(metrics["macro_recall"], 0.5)
        self.assertAlmostEqual(metrics["weighted_precision"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

    def test_aurc_uses_entropy_confidence_order(self) -> None:
        value = compute_aurc(
            np.array([0, 0, 1]),
            np.array([0, 1, 1]),
            np.array([0.9, 0.1, 0.8]),
        )
        self.assertAlmostEqual(value, (0.0 + 0.0 + 1.0 / 3.0) / 3.0)

    def test_default_labels_are_ground_truth_classes(self) -> None:
        metrics = compute_classification_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 2, 1, 2]),
        )
        self.assertAlmostEqual(metrics["macro_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

    def test_rejects_empty_or_duplicate_evaluation_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            compute_classification_metrics(
                np.array([0]),
                np.array([0]),
                labels=[],
            )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            compute_classification_metrics(
                np.array([0]),
                np.array([0]),
                labels=[0, 0],
            )


if __name__ == "__main__":
    unittest.main()
