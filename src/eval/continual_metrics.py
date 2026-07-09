"""Continual-learning metrics and result matrix helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
        final = float(result_matrix[final_row, task_id]) if not np.isnan(result_matrix[final_row, task_id]) else np.nan
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
