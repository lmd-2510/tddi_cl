"""Per-class evaluation trajectories for class-incremental runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


CLASS_METRIC_COLUMNS = [
    "class_id",
    "test_count",
    "precision",
    "recall",
    "f1",
    "confidence",
    "entropy",
]

TRAJECTORY_COLUMNS = [
    "seed",
    "method",
    "train_task",
    "class_id",
    "first_task",
    "train_count",
    "test_count",
    "precision",
    "recall",
    "f1",
    "confidence",
    "entropy",
]

FORGETTING_COLUMNS = [
    "seed",
    "method",
    "train_task",
    "class_id",
    "first_task",
    "train_count",
    "test_count",
    "best_f1",
    "current_f1",
    "forgetting",
]


def compute_classwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_indices: list[int],
    inverse_class_map: Mapping[int, int],
) -> pd.DataFrame:
    """Compute metrics for every requested class, including zero-support classes."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_indices = [int(class_index) for class_index in class_indices]

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays.")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array.")
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("Labels, predictions, and probabilities must have the same row count.")
    if len(set(class_indices)) != len(class_indices):
        raise ValueError("class_indices must not contain duplicates.")

    missing_map = sorted(set(class_indices) - set(inverse_class_map))
    if missing_map:
        raise ValueError(f"Missing raw class IDs for local indices: {missing_map}")
    invalid_indices = [
        class_index
        for class_index in class_indices
        if class_index < 0 or class_index >= probabilities.shape[1]
    ]
    if invalid_indices:
        raise ValueError(f"Class indices outside probability columns: {invalid_indices}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_indices,
        zero_division=0,
    )
    confidence_by_sample = probabilities.max(axis=1)
    safe_probabilities = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    entropy_by_sample = -np.sum(probabilities * np.log(safe_probabilities), axis=1)

    rows: list[dict[str, float | int]] = []
    for position, class_index in enumerate(class_indices):
        class_mask = y_true == class_index
        rows.append(
            {
                "class_id": int(inverse_class_map[class_index]),
                "test_count": int(support[position]),
                "precision": float(precision[position]),
                "recall": float(recall[position]),
                "f1": float(f1[position]),
                "confidence": (
                    float(confidence_by_sample[class_mask].mean())
                    if np.any(class_mask)
                    else float("nan")
                ),
                "entropy": (
                    float(entropy_by_sample[class_mask].mean())
                    if np.any(class_mask)
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows, columns=CLASS_METRIC_COLUMNS).sort_values(
        "class_id", ignore_index=True
    )


@dataclass
class ClasswiseTracker:
    """Accumulate one run's class metrics and write trajectory artifacts."""

    seed: int
    method: str
    first_task_by_class: Mapping[int, int]
    train_count_by_class: Mapping[int, int]
    _frames: list[pd.DataFrame] = field(default_factory=list, init=False, repr=False)

    def add_task(self, train_task: int, class_metrics: pd.DataFrame) -> None:
        missing_columns = sorted(set(CLASS_METRIC_COLUMNS) - set(class_metrics.columns))
        if missing_columns:
            raise ValueError(f"Missing class metric columns: {missing_columns}")

        frame = class_metrics[CLASS_METRIC_COLUMNS].copy()
        frame["class_id"] = frame["class_id"].astype(int)
        if frame["class_id"].duplicated().any():
            duplicates = sorted(frame.loc[frame["class_id"].duplicated(), "class_id"].tolist())
            raise ValueError(f"Duplicate class IDs for train task {train_task}: {duplicates}")

        class_ids = frame["class_id"].tolist()
        missing_first_task = sorted(set(class_ids) - set(self.first_task_by_class))
        if missing_first_task:
            raise ValueError(f"Missing first-task metadata for classes: {missing_first_task}")
        missing_train_count = sorted(set(class_ids) - set(self.train_count_by_class))
        if missing_train_count:
            raise ValueError(f"Missing train counts for classes: {missing_train_count}")

        future_classes = [
            class_id
            for class_id in class_ids
            if int(self.first_task_by_class[class_id]) > int(train_task)
        ]
        if future_classes:
            raise ValueError(
                f"Classes are evaluated before their first task {train_task}: {sorted(future_classes)}"
            )

        frame.insert(0, "seed", int(self.seed))
        frame.insert(1, "method", self.method)
        frame.insert(2, "train_task", int(train_task))
        frame.insert(
            4,
            "first_task",
            frame["class_id"].map(self.first_task_by_class).astype(int),
        )
        frame.insert(
            5,
            "train_count",
            frame["class_id"].map(self.train_count_by_class).astype(int),
        )
        frame = frame[TRAJECTORY_COLUMNS]

        candidate = pd.concat([*self._frames, frame], ignore_index=True)
        key_columns = ["seed", "method", "train_task", "class_id"]
        duplicate_keys = candidate.duplicated(key_columns, keep=False)
        if duplicate_keys.any():
            duplicate_rows = candidate.loc[duplicate_keys, key_columns].to_dict("records")
            raise ValueError(f"Duplicate class trajectory keys: {duplicate_rows}")
        self._frames.append(frame)

    def trajectory_frame(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame(columns=TRAJECTORY_COLUMNS)
        return (
            pd.concat(self._frames, ignore_index=True)
            .sort_values(["train_task", "class_id"], ignore_index=True)
            [TRAJECTORY_COLUMNS]
        )

    def forgetting_frame(self) -> pd.DataFrame:
        trajectory = self.trajectory_frame()
        if trajectory.empty:
            return pd.DataFrame(columns=FORGETTING_COLUMNS)

        forgetting = trajectory.copy()
        group_columns = ["seed", "method", "class_id"]
        forgetting["best_f1"] = forgetting.groupby(group_columns, sort=False)["f1"].cummax()
        forgetting["current_f1"] = forgetting["f1"]
        forgetting["forgetting"] = forgetting["best_f1"] - forgetting["current_f1"]
        return forgetting[FORGETTING_COLUMNS]

    def save(self, outdir: str | Path) -> tuple[Path, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        trajectory_path = outdir / "class_trajectory.csv"
        forgetting_path = outdir / "class_forgetting.csv"
        self.trajectory_frame().to_csv(trajectory_path, index=False)
        self.forgetting_frame().to_csv(forgetting_path, index=False)
        return trajectory_path, forgetting_path
