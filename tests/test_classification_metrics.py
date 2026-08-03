from __future__ import annotations

import unittest
import warnings

import numpy as np

from src.eval.classification_metrics import compute_classification_metrics


class ClassificationMetricsTest(unittest.TestCase):
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
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

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
