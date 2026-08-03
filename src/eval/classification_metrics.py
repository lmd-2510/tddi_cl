"""Classification metrics for DDI2025-CIL."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[int] | None = None,
) -> dict[str, float]:
    """Compute classification metrics over an explicit evaluation label set."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    evaluation_labels = (
        sorted(np.unique(y_true).astype(int).tolist())
        if labels is None
        else [int(label) for label in labels]
    )
    if not evaluation_labels:
        raise ValueError("At least one evaluation label is required.")
    if len(set(evaluation_labels)) != len(evaluation_labels):
        raise ValueError("Evaluation labels must not contain duplicates.")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=evaluation_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=evaluation_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            recall_score(
                y_true,
                y_pred,
                labels=evaluation_labels,
                average="macro",
                zero_division=0,
            )
        ),
    }
