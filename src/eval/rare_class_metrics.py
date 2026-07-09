"""Rare-class grouped metrics for DDI2025-CIL."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def build_rare_class_groups(
    train_class_counts: pd.DataFrame,
    thresholds: Iterable[int] = (5, 10, 20),
) -> dict[str, set[int]]:
    counts = train_class_counts[["class_id", "count"]].copy()
    groups: dict[str, set[int]] = {}
    for threshold in thresholds:
        groups[f"le_{threshold}"] = set(
            counts.loc[counts["count"] <= threshold, "class_id"].astype(int).tolist()
        )
    return groups


def compute_group_macro_f1(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    *,
    class_group: set[int],
) -> float:
    if not class_group:
        return float("nan")
    y_true_raw = np.asarray(y_true_raw, dtype=np.int64)
    y_pred_raw = np.asarray(y_pred_raw, dtype=np.int64)
    mask = np.isin(y_true_raw, list(class_group))
    if not np.any(mask):
        return float("nan")
    labels = sorted(class_group)
    return float(
        f1_score(
            y_true_raw[mask],
            y_pred_raw[mask],
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def rare_class_summary(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    train_class_counts: pd.DataFrame,
    thresholds: Iterable[int] = (5, 10, 20),
) -> pd.DataFrame:
    groups = build_rare_class_groups(train_class_counts, thresholds=thresholds)
    rows = []
    for group_name, class_group in groups.items():
        rows.append(
            {
                "group": group_name,
                "num_classes": len(class_group),
                "macro_f1": compute_group_macro_f1(
                    y_true_raw,
                    y_pred_raw,
                    class_group=class_group,
                ),
            }
        )
    return pd.DataFrame(rows)
