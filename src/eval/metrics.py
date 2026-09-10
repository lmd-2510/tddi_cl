"""Classification, class-wise, and continual-learning metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


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

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=evaluation_labels,
            average="macro",
            zero_division=0,
        )
    )
    return {
        "accuracy": accuracy,
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=evaluation_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": macro_recall,
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
        "weighted_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=evaluation_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        # Compatibility alias: for explicit multiclass labels this is macro recall.
        "balanced_accuracy": macro_recall,
    }


def compute_aurc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> float:
    """Area under the selective risk-coverage curve; lower is better."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or confidence.shape != y_true.shape:
        raise ValueError("y_true, y_pred and confidence must be equally sized vectors.")
    if y_true.size == 0 or not np.isfinite(confidence).all():
        raise ValueError("AURC requires non-empty finite confidence values.")
    order = np.argsort(-confidence, kind="stable")
    errors = (y_pred[order] != y_true[order]).astype(np.float64)
    selective_risk = np.cumsum(errors) / np.arange(1, errors.size + 1)
    return float(np.mean(selective_risk))


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

    def restore_trajectory(self, trajectory: pd.DataFrame) -> None:
        """Restore completed task rows from a validated task-boundary checkpoint."""

        if self._frames:
            raise RuntimeError("Cannot restore a class trajectory into a non-empty tracker.")
        trajectory = trajectory.copy()
        if trajectory.empty:
            return
        missing_columns = sorted(set(TRAJECTORY_COLUMNS) - set(trajectory.columns))
        if missing_columns:
            raise ValueError(f"Restored class trajectory is missing columns: {missing_columns}")
        trajectory = trajectory[TRAJECTORY_COLUMNS]
        if set(trajectory["seed"].astype(int)) != {int(self.seed)}:
            raise ValueError("Restored class trajectory seed does not match the run.")
        if set(trajectory["method"].astype(str)) != {self.method}:
            raise ValueError("Restored class trajectory method does not match the run.")
        key_columns = ["seed", "method", "train_task", "class_id"]
        if trajectory.duplicated(key_columns).any():
            raise ValueError("Restored class trajectory contains duplicate keys.")
        task_ids = sorted(trajectory["train_task"].astype(int).unique().tolist())
        if task_ids != list(range(task_ids[-1] + 1)):
            raise ValueError("Restored class trajectory tasks must be contiguous from zero.")
        for class_id, first_task in zip(
            trajectory["class_id"].astype(int),
            trajectory["first_task"].astype(int),
        ):
            if self.first_task_by_class.get(class_id) != first_task:
                raise ValueError("Restored class trajectory first-task metadata does not match.")
        for class_id, train_count in zip(
            trajectory["class_id"].astype(int),
            trajectory["train_count"].astype(int),
        ):
            if self.train_count_by_class.get(class_id) != train_count:
                raise ValueError("Restored class trajectory train-count metadata does not match.")
        self._frames = [
            frame.reset_index(drop=True)
            for _, frame in trajectory.groupby("train_task", sort=True)
        ]

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


def init_result_matrix(num_tasks: int) -> np.ndarray:
    return np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)


def compute_forgetting(result_matrix: np.ndarray) -> pd.DataFrame:
    num_tasks = result_matrix.shape[0]
    final_row = num_tasks - 1
    rows: list[dict[str, Any]] = []
    forgetting_values: list[float] = []

    for task_id in range(num_tasks):
        observed = result_matrix[task_id:, task_id]
        observed = observed[~np.isnan(observed)]
        if observed.size == 0:
            continue
        best = float(np.max(observed))
        final = (
            float(result_matrix[final_row, task_id])
            if not np.isnan(result_matrix[final_row, task_id])
            else np.nan
        )
        forgetting = best - final if not np.isnan(final) else np.nan
        rows.append(
            {
                "task_id": task_id,
                "best_macro_f1": best,
                "final_macro_f1": final,
                "forgetting": forgetting,
            }
        )
        if task_id < final_row and not np.isnan(forgetting):
            forgetting_values.append(forgetting)

    mean_forgetting = float(np.mean(forgetting_values)) if forgetting_values else 0.0
    rows.append(
        {
            "task_id": "mean_old_tasks",
            "best_macro_f1": np.nan,
            "final_macro_f1": np.nan,
            "forgetting": mean_forgetting,
        }
    )
    return pd.DataFrame(rows)


def result_matrix_to_frame(result_matrix: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for train_task in range(result_matrix.shape[0]):
        row: dict[str, Any] = {"train_task_id": train_task}
        for test_task in range(result_matrix.shape[1]):
            row[f"test_task_{test_task}"] = result_matrix[train_task, test_task]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_average_incremental_macro_f1(result_matrix: np.ndarray) -> float:
    """Mean performance on all seen task groups after each training task."""

    matrix = np.asarray(result_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("result_matrix must be a non-empty square matrix.")
    per_step = []
    for train_task in range(matrix.shape[0]):
        observed = matrix[train_task, : train_task + 1]
        observed = observed[np.isfinite(observed)]
        if observed.size:
            per_step.append(float(np.mean(observed)))
    if not per_step:
        raise ValueError("result_matrix contains no finite observed task results.")
    return float(np.mean(per_step))
