"""Classification metrics for DDI2025-CIL."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def compute_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_indices: list[int],
    raw_class_ids: list[int],
) -> pd.DataFrame:
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=class_indices,
        average=None,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "raw_class_id": raw_class_ids,
            "class_index": class_indices,
            "f1": per_class_f1,
        }
    )
