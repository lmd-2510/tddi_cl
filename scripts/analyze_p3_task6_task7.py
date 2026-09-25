#!/usr/bin/env python3
"""Derive aligned task-6/task-7 and classwise forgetting diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ensemble_ue import load_offline_ensemble_artifact
from src.eval.metrics import compute_classification_metrics
from src.eval.predictions import load_member_prediction_artifact


MEMBERS = (0, 1, 2)
BOUNDARIES = (6, 7)


def _load_tasks(path: Path) -> list[list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "tail_to_head":
        raise ValueError(f"Expected P3 tail_to_head task file: {path}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 8:
        raise ValueError("P3 diagnostics require all eight frozen tasks.")
    result = [[int(value) for value in task["classes"]] for task in tasks]
    if any(not group for group in result) or len({raw for group in result for raw in group}) != 178:
        raise ValueError("P3 task file must define 178 unique raw class IDs.")
    return result


def _load_prediction(root: Path, scope: str, task_id: int, split: str) -> dict[str, Any]:
    if scope == "ensemble":
        path = root / "offline_evaluation" / f"task_{task_id}" / (
            "oof.npz" if split == "validation" else "test.npz"
        )
        artifact = load_offline_ensemble_artifact(path)
        context = artifact.context
        if context.task_id != task_id or context.split != split:
            raise ValueError(f"Offline ensemble context mismatch: {path}")
        return {
            "sample_ids": np.asarray(artifact.sample_ids).astype(np.str_),
            "labels": np.asarray(artifact.labels, dtype=np.int64),
            "predictions": np.asarray(artifact.predictions, dtype=np.int64),
            "class_ids": np.asarray(artifact.raw_class_ids, dtype=np.int64),
        }

    member_id = int(scope.removeprefix("member"))
    path = root / f"member_{member_id}" / "member_predictions" / f"task_{task_id}" / f"{split}.npz"
    artifact = load_member_prediction_artifact(path)
    context = artifact.context
    if context.member_id != member_id or context.task_id != task_id or context.split != split:
        raise ValueError(f"Member prediction context mismatch: {path}")
    predictions = np.asarray(artifact.raw_class_ids, dtype=np.int64)[
        np.asarray(artifact.probabilities).argmax(axis=1)
    ]
    return {
        "sample_ids": np.asarray(artifact.sample_ids).astype(np.str_),
        "labels": np.asarray(artifact.labels, dtype=np.int64),
        "predictions": predictions.astype(np.int64, copy=False),
        "class_ids": np.asarray(artifact.raw_class_ids, dtype=np.int64),
    }


def _metric_row(
    prediction: dict[str, Any], class_ids: list[int], *, scope: str,
    split: str, boundary: int, group: str,
) -> dict[str, Any]:
    labels = prediction["labels"]
    predicted = prediction["predictions"]
    keep = np.isin(labels, class_ids)
    if not keep.any():
        raise ValueError(f"No {split} rows for {scope}, boundary={boundary}, group={group}.")
    return {
        "scope": scope,
        "split": split,
        "boundary": boundary,
        "group": group,
        "class_count": len(class_ids),
        "sample_count": int(keep.sum()),
        **compute_classification_metrics(labels[keep], predicted[keep], labels=class_ids),
    }


def _markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".6f") -> str:
    """Render a small Markdown table without requiring optional `tabulate`."""
    columns = [str(column) for column in frame.columns]
    rows = [columns]
    for values in frame.itertuples(index=False, name=None):
        row = []
        for value in values:
            if pd.isna(value):
                row.append("")
            elif isinstance(value, (float, np.floating)):
                row.append(format(float(value), floatfmt))
            else:
                row.append(str(value))
        rows.append(row)
    widths = [max(len(row[index]) for row in rows) for index in range(len(columns))]
    lines = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])) + " |"]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend(
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows[1:]
    )
    return "\n".join(lines)


def _ordered_common_old_rows(
    before: dict[str, Any], after: dict[str, Any], old_classes: list[int], *,
    scope: str, split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    before_keep = np.isin(before["labels"], old_classes)
    after_keep = np.isin(after["labels"], old_classes)
    before_ids = before["sample_ids"][before_keep]
    after_ids = after["sample_ids"][after_keep]
    before_index = {sample_id: i for i, sample_id in enumerate(before_ids.tolist())}
    after_index = {sample_id: i for i, sample_id in enumerate(after_ids.tolist())}
    if len(before_index) != len(before_ids) or len(after_index) != len(after_ids):
        raise ValueError(f"Duplicate sample IDs in {scope} {split} predictions.")
    if set(before_index) != set(after_index):
        raise ValueError(f"Task-6/task-7 old-class sample IDs differ for {scope} {split}.")
    order = sorted(before_index)
    ib = np.asarray([before_index[sample_id] for sample_id in order], dtype=np.int64)
    ia = np.asarray([after_index[sample_id] for sample_id in order], dtype=np.int64)
    labels_before = before["labels"][before_keep][ib]
    labels_after = after["labels"][after_keep][ia]
    if not np.array_equal(labels_before, labels_after):
        raise ValueError(f"Task-6/task-7 labels differ for aligned {scope} {split} sample IDs.")
    return (
        labels_before,
        before["predictions"][before_keep][ib],
        after["predictions"][after_keep][ia],
        np.asarray(order, dtype=np.str_),
    )


def analyze(full_root: Path, task_file: Path, outdir: Path) -> dict[str, Path]:
    tasks = _load_tasks(task_file)
    outdir.mkdir(parents=True, exist_ok=True)
    boundary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    scopes = [*(f"member{member}" for member in MEMBERS), "ensemble"]

    for scope in scopes:
        for split in ("validation", "test"):
            boundary_predictions = {
                task_id: _load_prediction(full_root, scope, task_id, split)
                for task_id in BOUNDARIES
            }
            for task_id in BOUNDARIES:
                prediction = boundary_predictions[task_id]
                seen = [raw for group in tasks[: task_id + 1] for raw in group]
                groups: list[tuple[str, list[int]]] = [
                    ("seen_all", seen),
                    ("current_task", tasks[task_id]),
                ]
                if task_id > 0:
                    groups.append(("previous_tasks", [raw for group in tasks[:task_id] for raw in group]))
                groups.extend((f"task_{index}", tasks[index]) for index in range(task_id + 1))
                for group_name, group_classes in groups:
                    boundary_rows.append(_metric_row(
                        prediction, group_classes, scope=scope, split=split,
                        boundary=task_id, group=group_name,
                    ))

            before = boundary_predictions[6]
            after = boundary_predictions[7]
            old_classes = [raw for group in tasks[:7] for raw in group]
            new_classes = tasks[7]
            old_labels, pred6, pred7, _ = _ordered_common_old_rows(
                before, after, old_classes, scope=scope, split=split,
            )
            confusion_rows.append({
                "scope": scope,
                "split": split,
                "comparison": "old_true_predicted_as_task7_class",
                "sample_count": int(len(old_labels)),
                "error_count": int(np.isin(pred7, new_classes).sum()),
                "error_rate": float(np.isin(pred7, new_classes).mean()),
            })
            new_mask = np.isin(after["labels"], new_classes)
            if not new_mask.any():
                raise ValueError(f"No task-7 rows in {scope} {split} predictions.")
            confusion_rows.append({
                "scope": scope,
                "split": split,
                "comparison": "task7_true_predicted_as_old_class",
                "sample_count": int(new_mask.sum()),
                "error_count": int(np.isin(after["predictions"][new_mask], old_classes).sum()),
                "error_rate": float(np.isin(after["predictions"][new_mask], old_classes).mean()),
            })

            for raw_class in old_classes:
                class_mask = old_labels == raw_class
                support = int(class_mask.sum())
                p6, r6, f6, _ = precision_recall_fscore_support(
                    before["labels"], before["predictions"], labels=[raw_class],
                    average=None, zero_division=0,
                )
                p7, r7, f7, _ = precision_recall_fscore_support(
                    after["labels"], after["predictions"], labels=[raw_class],
                    average=None, zero_division=0,
                )
                class_rows.append({
                    "scope": scope,
                    "split": split,
                    "raw_class_id": raw_class,
                    "origin_task": next(index for index, group in enumerate(tasks) if raw_class in group),
                    "support": support,
                    "task6_precision": float(p6[0]),
                    "task6_recall": float(r6[0]),
                    "task6_f1": float(f6[0]),
                    "task7_precision": float(p7[0]),
                    "task7_recall": float(r7[0]),
                    "task7_f1": float(f7[0]),
                    "f1_change_task7_minus_task6": float(f7[0] - f6[0]),
                    "task7_old_true_predicted_as_new_rate": float(np.isin(pred7[class_mask], new_classes).mean()),
                })
            for raw_class in new_classes:
                class_mask = after["labels"] == raw_class
                p, r, f, supports = precision_recall_fscore_support(
                    after["labels"], after["predictions"],
                    labels=[raw_class], average=None, zero_division=0,
                )
                class_rows.append({
                    "scope": scope,
                    "split": split,
                    "raw_class_id": raw_class,
                    "origin_task": 7,
                    "support": int(supports[0]),
                    "task6_precision": np.nan,
                    "task6_recall": np.nan,
                    "task6_f1": np.nan,
                    "task7_precision": float(p[0]),
                    "task7_recall": float(r[0]),
                    "task7_f1": float(f[0]),
                    "f1_change_task7_minus_task6": np.nan,
                    "task7_old_true_predicted_as_new_rate": np.nan,
                })

    boundary_frame = pd.DataFrame(boundary_rows)
    class_frame = pd.DataFrame(class_rows)
    confusion_frame = pd.DataFrame(confusion_rows)
    boundary_path = outdir / "task6_task7_boundary_metrics.csv"
    class_path = outdir / "task6_task7_classwise_comparison.csv"
    confusion_path = outdir / "task6_task7_cross_group_confusion.csv"
    boundary_frame.to_csv(boundary_path, index=False)
    class_frame.to_csv(class_path, index=False)
    confusion_frame.to_csv(confusion_path, index=False)

    ensemble = boundary_frame[boundary_frame["scope"] == "ensemble"]
    report_lines = [
        "# P3 task 6 → task 7 diagnostic",
        "",
        "Compares task-6 and task-7 models on aligned sample IDs. Validation is the development diagnostic; test is a final descriptive check and must not be used to tune the next run.",
        "Macro-F1 for `seen_all` changes its class set from task 6 to 7. The fixed old-class comparison is task-6 `seen_all` against task-7 `previous_tasks`.",
        "",
        "## Three-member offline ensemble",
        "",
    ]
    columns = ["split", "boundary", "group", "class_count", "sample_count", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    report_lines.append(_markdown_table(ensemble[columns]))
    report_lines.extend(["", "## Old→new/current cross-group errors", ""])
    report_lines.append(_markdown_table(confusion_frame[confusion_frame["scope"] == "ensemble"]))
    report_lines.extend([
        "",
        "## How to read the files",
        "",
        "- `task6_task7_boundary_metrics.csv`: seen-all, previous-task, current-task, and task-group metrics for each member and the offline ensemble.",
        "- `task6_task7_classwise_comparison.csv`: per-class F1/precision/recall change from boundary 6 to 7 on the same old-class sample IDs; task-7 classes have task-7 metrics only.",
        "- `task6_task7_cross_group_confusion.csv`: fractions of old true samples predicted as task-7 classes and task-7 true samples predicted as old classes.",
        "",
        "A decline in old-class metrics plus old→new errors supports a classifier/interference diagnosis; weak task-7 class metrics with stable old classes instead suggests new-class learning difficulty. These are diagnostics, not causal proof.",
        "",
    ])
    report_path = outdir / "task6_task7_diagnostic.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {"boundary_metrics": boundary_path, "classwise": class_path, "cross_group": confusion_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", required=True, type=Path)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    outputs = analyze(args.full_root, args.task_file, args.outdir)
    for name, path in outputs.items():
        print(f"[DONE] {name}: {path}")


if __name__ == "__main__":
    main()
