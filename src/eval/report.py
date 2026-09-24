#!/usr/bin/env python3
"""Build a final P3/P4 ensemble report from completed offline artifacts.

This entrypoint is report-only: it never loads a model checkpoint and never trains or
runs inference. It derives classification, continual-learning, and ensemble-diversity
metrics from artifacts already exported by the full study.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import compute_classification_metrics  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    compute_forgetting,
    result_matrix_to_frame,
)
from src.eval.ensemble_ue import (  # noqa: E402
    OfflineEnsembleArtifact,
    load_offline_ensemble_artifact,
)
from src.eval.predictions import load_member_prediction_artifact  # noqa: E402


DEFAULT_FULL_ROOT = Path(
    "outputs/full/tddi_ensemble3_replay_distill_p3_seed0_8tasks_v1"
)
DEFAULT_TASK_FILE = Path("study_assets/task_protocols/tail_to_head_tasks.json")
SUPPORTED_PROTOCOLS = {
    "tail_to_head": ("P3", "ensemble3_p3"),
    "constrained_mass_balanced": ("P4", "ensemble3_p4"),
}
EXPECTED_MEMBER_IDS = (0, 1, 2)


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_task_groups(task_file: Path) -> tuple[list[list[int]], Mapping[str, Any]]:
    payload = _read_json(task_file)
    protocol = payload.get("protocol")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"Unsupported final-report protocol {protocol!r}: {task_file}. "
            f"Expected one of {sorted(SUPPORTED_PROTOCOLS)}."
        )
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(f"Task file has no tasks: {task_file}")
    task_groups: list[list[int]] = []
    for expected_id, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict) or int(raw_task.get("task_id", -1)) != expected_id:
            raise ValueError("Task IDs must be contiguous from zero.")
        classes = [int(value) for value in raw_task.get("classes", [])]
        if not classes or len(set(classes)) != len(classes):
            raise ValueError(f"Task {expected_id} has empty or duplicate class IDs.")
        task_groups.append(classes)
    flattened = [class_id for group in task_groups for class_id in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("Raw class IDs must not occur in more than one task.")
    if "num_classes" in payload and int(payload["num_classes"]) != len(flattened):
        raise ValueError("Task-file num_classes does not match its task contents.")
    return task_groups, payload


def _ensemble_path(full_root: Path, task_id: int) -> Path:
    candidates = (
        full_root / "offline_ensemble" / f"task_{task_id}" / "test.npz",
        full_root / "offline_evaluation" / f"task_{task_id}" / "test.npz",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _threshold_report_path(full_root: Path, task_id: int) -> Path:
    candidates = (
        full_root / "threshold" / f"task_{task_id}" / "test_report.json",
        full_root / "offline_evaluation" / f"task_{task_id}" / "test_threshold_report.json",
        full_root / "offline_ensemble" / f"task_{task_id}" / "test_threshold_report.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _validate_ensemble_context(
    artifact: OfflineEnsembleArtifact,
    *,
    task_id: int,
    expected_seen_classes: set[int],
    shared_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    context = artifact.context
    if context.task_id != task_id or context.split.casefold() != "test":
        raise ValueError(
            f"Unexpected ensemble context at task {task_id}: "
            f"task={context.task_id}, split={context.split!r}."
        )
    if tuple(context.member_ids) != EXPECTED_MEMBER_IDS or context.member_count != 3:
        raise ValueError(f"Task {task_id} is not an aligned three-member ensemble.")
    if set(artifact.raw_class_ids.astype(int).tolist()) != expected_seen_classes:
        raise ValueError(f"Task {task_id} raw class IDs do not match the task schedule.")
    current = {
        "method": context.method,
        "method_protocol": context.method_protocol,
        "experiment_seed": context.experiment_seed,
        "member_ids": tuple(context.member_ids),
        "member_seeds": tuple(context.member_seeds),
    }
    if shared_context is not None and current != dict(shared_context):
        raise ValueError(f"Ensemble provenance changed at task {task_id}.")
    return current


def _threshold_mask(
    artifact: OfflineEnsembleArtifact,
    *,
    score_name: str,
    threshold_value: float,
) -> np.ndarray:
    if score_name not in {"entropy_confidence", "max_probability"}:
        raise ValueError(f"Unsupported threshold score: {score_name!r}")
    scores = np.asarray(getattr(artifact, score_name), dtype=np.float64)
    mask = scores >= threshold_value
    if not mask.any():
        raise ValueError("Threshold selected no test samples; metrics are undefined.")
    return mask


def _close(actual: float, expected: object, *, name: str, task_id: int) -> None:
    if expected is None or not np.isclose(actual, float(expected), rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"Task {task_id} {name} does not match the frozen-threshold report: "
            f"{actual} != {expected}."
        )


def _build_task_row(
    artifact: OfflineEnsembleArtifact,
    report: Mapping[str, Any],
    *,
    task_id: int,
    task_classes: Sequence[int] | None = None,
    protocol_name: str = "tail_to_head",
) -> dict[str, Any]:
    if (
        str(report.get("evaluation_split", "")).casefold() != "test"
        or int(report.get("task_id", -1)) != task_id
        or int(report.get("experiment_seed", -1)) != artifact.context.experiment_seed
    ):
        raise ValueError(f"Threshold report context mismatch at task {task_id}.")
    report_classes = [int(value) for value in report.get("raw_class_ids", [])]
    if report_classes != artifact.raw_class_ids.astype(int).tolist():
        raise ValueError(f"Threshold report class order mismatch at task {task_id}.")

    threshold = report.get("threshold")
    full = report.get("full_set")
    selected = report.get("threshold_score_selective_metrics", report.get("high_confidence"))
    if not isinstance(threshold, dict) or not isinstance(full, dict) or not isinstance(selected, dict):
        raise ValueError(f"Threshold report is missing metric sections at task {task_id}.")
    if threshold.get("source_split") not in {"validation", "oof"}:
        raise ValueError(
            f"Task {task_id} threshold did not originate from validation or OOF."
        )
    if threshold.get("probability_source", "raw") != "raw":
        raise ValueError("This report builder expects raw offline ensemble probabilities.")

    labels = artifact.raw_class_ids.astype(int).tolist()
    full_metrics = compute_classification_metrics(
        artifact.labels,
        artifact.predictions,
        labels=labels,
    )
    score_name = str(threshold.get("score_name", ""))
    selection_applied = bool(threshold.get("selection_applied", True))
    raw_threshold_value = threshold.get("value")
    if selection_applied:
        threshold_value = float(raw_threshold_value)
        if not np.isfinite(threshold_value):
            raise ValueError(f"Task {task_id} threshold value is not finite.")
        mask = _threshold_mask(
            artifact,
            score_name=score_name,
            threshold_value=threshold_value,
        )
    else:
        if raw_threshold_value is not None:
            raise ValueError(f"Task {task_id} no-selection report invented a threshold.")
        threshold_value = float("nan")
        mask = np.ones(artifact.row_count, dtype=bool)
    threshold_metrics = compute_classification_metrics(
        artifact.labels[mask],
        artifact.predictions[mask],
        labels=labels,
    )
    selected_count = int(mask.sum())
    total_count = artifact.row_count
    coverage = float(selected_count / total_count)

    _close(full_metrics["accuracy"], full.get("accuracy"), name="full accuracy", task_id=task_id)
    _close(full_metrics["macro_f1"], full.get("macro_f1"), name="full Macro-F1", task_id=task_id)
    _close(
        threshold_metrics["accuracy"],
        selected.get("accuracy"),
        name="threshold accuracy",
        task_id=task_id,
    )
    _close(
        threshold_metrics["macro_f1"],
        selected.get("macro_f1"),
        name="threshold Macro-F1",
        task_id=task_id,
    )
    _close(coverage, selected.get("coverage"), name="coverage", task_id=task_id)
    if int(selected.get("selected_count", -1)) != selected_count:
        raise ValueError(f"Task {task_id} selected_count mismatch.")
    if int(selected.get("total_count", -1)) != total_count:
        raise ValueError(f"Task {task_id} total_count mismatch.")

    task_classes = [int(value) for value in (task_classes or [])]
    new_class_mask = np.isin(artifact.labels, task_classes) if task_classes else np.zeros(artifact.row_count, dtype=bool)
    stage = (
        "tail" if protocol_name == "tail_to_head" and task_id == 0 else
        "head" if protocol_name == "tail_to_head" and task_id == 7 else
        "balanced" if protocol_name == "constrained_mass_balanced" else "mid"
    )
    return {
        "task_id": task_id,
        "task_label": f"task_{task_id}_{stage}",
        "new_class_count": len(task_classes),
        "new_class_test_samples": int(new_class_mask.sum()),
        "full_accuracy": full_metrics["accuracy"],
        "full_balanced_accuracy": full_metrics["balanced_accuracy"],
        "full_macro_f1": full_metrics["macro_f1"],
        "full_weighted_f1": full_metrics["weighted_f1"],
        "full_samples": total_count,
        "threshold_value": threshold_value,
        "threshold_score": score_name,
        "threshold_accuracy": threshold_metrics["accuracy"],
        "threshold_balanced_accuracy": threshold_metrics["balanced_accuracy"],
        "threshold_macro_f1": threshold_metrics["macro_f1"],
        "threshold_weighted_f1": threshold_metrics["weighted_f1"],
        "coverage": coverage,
        "selected_count": selected_count,
        "total_count": total_count,
        "member_count": artifact.context.member_count,
        "mean_pairwise_disagreement": float(np.mean(artifact.pairwise_disagreement)),
        "mean_mutual_information": float(np.mean(artifact.mutual_information)),
        "mean_member_normalized_mi": float(np.mean(artifact.member_normalized_mi)),
        "mean_total_probability_variance": float(
            np.mean(artifact.total_probability_variance)
        ),
    }


def _build_ensemble_continual_metrics(
    artifacts: Sequence[OfflineEnsembleArtifact],
    task_groups: Sequence[Sequence[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_count = len(task_groups)
    matrix = np.full((task_count, task_count), np.nan, dtype=np.float64)
    for train_task, artifact in enumerate(artifacts):
        for eval_task in range(train_task + 1):
            class_ids = [int(value) for value in task_groups[eval_task]]
            mask = np.isin(artifact.labels, class_ids)
            if not mask.any():
                raise ValueError(
                    f"Task {train_task} ensemble has no samples for task group {eval_task}."
                )
            metrics = compute_classification_metrics(
                artifact.labels[mask],
                artifact.predictions[mask],
                labels=class_ids,
            )
            matrix[train_task, eval_task] = metrics["macro_f1"]
    return result_matrix_to_frame(matrix), compute_forgetting(matrix)


def _load_member_forgetting(
    full_root: Path,
    *,
    member_ids: Sequence[int],
    task_groups: Sequence[Sequence[int]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for member_id in member_ids:
        member_root = full_root / f"member_{member_id}"
        forgetting_path = member_root / "forgetting.csv"
        class_path = member_root / "class_forgetting.csv"
        if not forgetting_path.is_file() or not class_path.is_file():
            rows.append(
                _derive_member_forgetting(
                    member_root,
                    member_id=member_id,
                    task_groups=task_groups,
                )
            )
            continue
        forgetting = pd.read_csv(forgetting_path)
        summary = forgetting[forgetting["task_id"].astype(str) == "mean_old_tasks"]
        if len(summary) != 1:
            raise ValueError(f"Member {member_id} has no unique mean_old_tasks row.")
        class_forgetting = pd.read_csv(class_path)
        final_classes = class_forgetting[
            class_forgetting["train_task"].astype(int) == len(task_groups) - 1
        ].copy()
        if final_classes.empty or final_classes["class_id"].duplicated().any():
            raise ValueError(f"Member {member_id} final class-forgetting rows are invalid.")
        values = final_classes["forgetting"].astype(float).to_numpy()
        current_f1 = final_classes["current_f1"].astype(float).to_numpy()
        if not np.isfinite(values).all() or not np.isfinite(current_f1).all():
            raise ValueError(f"Member {member_id} forgetting contains non-finite values.")
        rows.append(
            {
                "member_id": member_id,
                "task_forgetting_mean_old_tasks": float(summary.iloc[0]["forgetting"]),
                "class_forgetting_mean_final": float(np.mean(values)),
                "class_forgetting_median_final": float(np.median(values)),
                "zero_f1_classes_final": int(np.sum(current_f1 == 0.0)),
                "evaluated_classes_final": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def _derive_member_forgetting(
    member_root: Path,
    *,
    member_id: int,
    task_groups: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Derive member forgetting from immutable per-boundary test predictions."""

    task_count = len(task_groups)
    matrix = np.full((task_count, task_count), np.nan, dtype=np.float64)
    best_class_f1: dict[int, float] = {}
    final_class_f1: dict[int, float] = {}
    for train_task in range(task_count):
        path = member_root / "member_predictions" / f"task_{train_task}" / "test.npz"
        artifact = load_member_prediction_artifact(path)
        expected_classes = sorted(
            int(raw)
            for group in task_groups[: train_task + 1]
            for raw in group
        )
        if (
            artifact.context.member_id != member_id
            or artifact.context.task_id != train_task
            or artifact.context.split.casefold() != "test"
            or artifact.raw_class_ids.astype(int).tolist() != expected_classes
        ):
            raise ValueError(f"Member prediction context mismatch: {path}")
        predictions = artifact.raw_class_ids[
            np.asarray(artifact.probabilities).argmax(axis=1)
        ].astype(np.int64, copy=False)
        for eval_task in range(train_task + 1):
            class_ids = [int(raw) for raw in task_groups[eval_task]]
            mask = np.isin(artifact.labels, class_ids)
            if not mask.any():
                raise ValueError(
                    f"Member {member_id} task {train_task} has no test rows for group {eval_task}."
                )
            matrix[train_task, eval_task] = compute_classification_metrics(
                artifact.labels[mask], predictions[mask], labels=class_ids
            )["macro_f1"]
        per_class = f1_score(
            artifact.labels,
            predictions,
            labels=expected_classes,
            average=None,
            zero_division=0,
        )
        for class_id, value in zip(expected_classes, per_class):
            value = float(value)
            best_class_f1[class_id] = max(best_class_f1.get(class_id, value), value)
            if train_task == task_count - 1:
                final_class_f1[class_id] = value

    forgetting = compute_forgetting(matrix)
    summary = forgetting[forgetting["task_id"].astype(str) == "mean_old_tasks"]
    if len(summary) != 1 or len(final_class_f1) != len(best_class_f1):
        raise ValueError(f"Could not derive complete forgetting for member {member_id}.")
    class_forgetting = np.asarray(
        [best_class_f1[raw] - final_class_f1[raw] for raw in sorted(final_class_f1)],
        dtype=np.float64,
    )
    final_values = np.asarray(
        [final_class_f1[raw] for raw in sorted(final_class_f1)], dtype=np.float64
    )
    return {
        "member_id": member_id,
        "task_forgetting_mean_old_tasks": float(summary.iloc[0]["forgetting"]),
        "class_forgetting_mean_final": float(class_forgetting.mean()),
        "class_forgetting_median_final": float(np.median(class_forgetting)),
        "zero_f1_classes_final": int(np.sum(final_values == 0.0)),
        "evaluated_classes_final": int(final_values.size),
    }


def _format_markdown_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n.a."
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def _build_markdown(
    task_summary: pd.DataFrame,
    paper_table: pd.DataFrame,
    ensemble_forgetting: pd.DataFrame,
    member_forgetting: pd.DataFrame,
    diversity: pd.DataFrame,
    *,
    shared_context: Mapping[str, Any],
    protocol_id: str,
    protocol_name: str,
) -> str:
    metric_columns = {
        "Full Accuracy": "full_accuracy",
        "Full Balanced Accuracy": "full_balanced_accuracy",
        "Full Macro-F1": "full_macro_f1",
        "Full Weighted F1": "full_weighted_f1",
        "Threshold Accuracy": "threshold_accuracy",
        "Threshold Balanced Accuracy": "threshold_balanced_accuracy",
        "Threshold Macro-F1": "threshold_macro_f1",
        "Threshold Weighted F1": "threshold_weighted_f1",
        "Average Coverage": "coverage",
    }
    average_rows = [
        {"Metric": label, "Mean across tasks": float(task_summary[column].mean())}
        for label, column in metric_columns.items()
    ]
    average_table = pd.DataFrame(average_rows)
    ensemble_mean_row = ensemble_forgetting[
        ensemble_forgetting["task_id"].astype(str) == "mean_old_tasks"
    ]
    ensemble_mean_forgetting = float(ensemble_mean_row.iloc[0]["forgetting"])
    member_task_mean = float(member_forgetting["task_forgetting_mean_old_tasks"].mean())
    member_task_std = float(
        member_forgetting["task_forgetting_mean_old_tasks"].std(ddof=0)
    )
    member_class_mean = float(member_forgetting["class_forgetting_mean_final"].mean())
    member_class_std = float(
        member_forgetting["class_forgetting_mean_final"].std(ddof=0)
    )
    pooled_disagreement = float(
        np.average(
            diversity["mean_pairwise_disagreement"],
            weights=diversity["sample_count"],
        )
    )

    return "\n".join(
        [
            f"# Ensemble3 Replay-Distill {protocol_id} Final Report",
            "",
            "## Experiment",
            "",
            f"- Protocol: {protocol_id} `{protocol_name}`",
            f"- Method: `{shared_context['method']}`",
            "- Backbone: `tddi_paper_member`",
            f"- Experiment seed: `{shared_context['experiment_seed']}`",
            f"- Ensemble members: `{list(shared_context['member_ids'])}`",
            f"- Member seeds: `{list(shared_context['member_seeds'])}`",
            f"- Tasks: `{len(task_summary)}`",
            "",
            "## Average metrics",
            "",
            "Mỗi task có trọng số bằng nhau trong bảng trung bình này.",
            "",
            _markdown_table(average_table),
            "",
            "## Per-task results",
            "",
            "`new_class_test_samples` là số test mẫu thuộc các class mới của task; "
            "`full_samples` là toàn bộ test mẫu của các class đã thấy đến boundary đó.",
            "Tên `task_0_tail`/`task_7_head` là nhãn diễn giải; raw class ID vẫn là khóa chính.",
            "",
            _markdown_table(paper_table),
            "",
            "## Forgetting summary",
            "",
            f"- Ensemble mean forgetting trên các old task: `{ensemble_mean_forgetting:.6f}`.",
            (
                "- Member task forgetting, mean ± population std: "
                f"`{member_task_mean:.6f} ± {member_task_std:.6f}`."
            ),
            (
                "- Member class forgetting ở task cuối, mean ± population std: "
                f"`{member_class_mean:.6f} ± {member_class_std:.6f}`."
            ),
            "",
            "### Member forgetting",
            "",
            _markdown_table(member_forgetting),
            "",
            "### Ensemble task-group forgetting",
            "",
            _markdown_table(ensemble_forgetting),
            "",
            "Forgetting của task là best Macro-F1 từng đạt trừ Macro-F1 tại task cuối. "
            "`mean_old_tasks` chỉ lấy các task cũ, không tính task cuối vừa học.",
            "",
            "## Ensemble diversity",
            "",
            _markdown_table(diversity),
            "",
            (
                "Pooled mean pairwise disagreement, có trọng số theo số mẫu: "
                f"`{pooled_disagreement:.6f}`."
            ),
            "",
            "Pairwise disagreement đo tỷ lệ các cặp member dự đoán khác nhau. Mutual "
            "information và probability variance đo mức khác biệt giữa phân phối xác "
            "suất của ba member. Diversity cao hơn không tự động đồng nghĩa accuracy tốt "
            "hơn; nó phải được đọc cùng classification metrics và UE error detection.",
            "",
            "## Metric definitions",
            "",
            "- Balanced Accuracy là macro recall, nên mỗi class có trọng số bằng nhau.",
            "- Weighted F1 lấy F1 từng class và đặt trọng số theo số mẫu thật của class.",
            "- Metric `full_*` dùng toàn bộ test samples của các class đã thấy.",
            "- Metric `threshold_*` chỉ dùng samples vượt frozen validation/OOF threshold và "
            "phải được báo cáo cùng coverage.",
            "- Diversity được tổng hợp từ offline ensemble artifact, không chạy lại model.",
            "",
        ]
    )


def _prepare_outputs(
    outdir: Path,
    *,
    output_prefix: str,
    overwrite: bool,
) -> dict[str, Path]:
    paths = {
        "task_summary": outdir / f"{output_prefix}_task_summary.csv",
        "paper_table": outdir / f"{output_prefix}_paper_table.csv",
        "ensemble_matrix": outdir / f"{output_prefix}_task_matrix.csv",
        "ensemble_forgetting": outdir / f"{output_prefix}_forgetting_summary.csv",
        "member_forgetting": outdir / f"{output_prefix}_member_forgetting_summary.csv",
        "diversity": outdir / f"{output_prefix}_diversity_summary.csv",
        "report": outdir / f"{output_prefix}_final_report.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Final report outputs already exist; pass --overwrite to replace only these "
            f"derived files: {existing}"
        )
    return paths


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_final_report(
    *,
    full_root: str | Path,
    task_file: str | Path,
    outdir: str | Path | None = None,
    member_ids: Sequence[int] = EXPECTED_MEMBER_IDS,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Build all derived tables and the Markdown report from completed artifacts."""

    full_root = Path(full_root)
    task_file = Path(task_file)
    outdir = full_root / "final_results" if outdir is None else Path(outdir)
    if tuple(member_ids) != EXPECTED_MEMBER_IDS:
        raise ValueError("This Ensemble3 report requires member IDs exactly [0, 1, 2].")
    task_groups, task_metadata = _load_task_groups(task_file)
    protocol_name = str(task_metadata["protocol"])
    protocol_id, output_prefix = SUPPORTED_PROTOCOLS[protocol_name]
    paths = _prepare_outputs(
        outdir,
        output_prefix=output_prefix,
        overwrite=overwrite,
    )

    artifacts: list[OfflineEnsembleArtifact] = []
    rows: list[dict[str, Any]] = []
    shared_context: Mapping[str, Any] | None = None
    seen_classes: set[int] = set()
    for task_id, task_classes in enumerate(task_groups):
        seen_classes.update(task_classes)
        artifact = load_offline_ensemble_artifact(_ensemble_path(full_root, task_id))
        shared_context = _validate_ensemble_context(
            artifact,
            task_id=task_id,
            expected_seen_classes=set(seen_classes),
            shared_context=shared_context,
        )
        artifacts.append(artifact)
        rows.append(
            _build_task_row(
                artifact,
                _read_json(_threshold_report_path(full_root, task_id)),
                task_id=task_id,
                task_classes=task_classes,
                protocol_name=protocol_name,
            )
        )
    if shared_context is None:
        raise RuntimeError("No ensemble artifacts were loaded.")

    task_summary = pd.DataFrame(rows)
    paper_columns = [
        "task_id",
        "task_label",
        "new_class_count",
        "new_class_test_samples",
        "full_samples",
        "full_accuracy",
        "full_balanced_accuracy",
        "full_macro_f1",
        "full_weighted_f1",
        "threshold_accuracy",
        "threshold_balanced_accuracy",
        "threshold_macro_f1",
        "threshold_weighted_f1",
        "coverage",
        "threshold_value",
        "mean_pairwise_disagreement",
        "mean_member_normalized_mi",
    ]
    paper_table = task_summary[paper_columns].copy()
    diversity = task_summary[
        [
            "task_id",
            "full_samples",
            "member_count",
            "mean_pairwise_disagreement",
            "mean_mutual_information",
            "mean_member_normalized_mi",
            "mean_total_probability_variance",
        ]
    ].rename(columns={"full_samples": "sample_count"})
    ensemble_matrix, ensemble_forgetting = _build_ensemble_continual_metrics(
        artifacts,
        task_groups,
    )
    member_forgetting = _load_member_forgetting(
        full_root,
        member_ids=member_ids,
        task_groups=task_groups,
    )
    markdown = _build_markdown(
        task_summary,
        paper_table,
        ensemble_forgetting,
        member_forgetting,
        diversity,
        shared_context=shared_context,
        protocol_id=protocol_id,
        protocol_name=protocol_name,
    )

    _atomic_write_csv(task_summary, paths["task_summary"])
    _atomic_write_csv(paper_table, paths["paper_table"])
    _atomic_write_csv(ensemble_matrix, paths["ensemble_matrix"])
    _atomic_write_csv(ensemble_forgetting, paths["ensemble_forgetting"])
    _atomic_write_csv(member_forgetting, paths["member_forgetting"])
    _atomic_write_csv(diversity, paths["diversity"])
    _atomic_write_text(markdown, paths["report"])
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an expanded P3/P4 final report from existing ensemble/threshold/"
            "forgetting artifacts; no training or inference is performed."
        )
    )
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace only the derived final-report files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    paths = build_final_report(
        full_root=args.full_root,
        task_file=args.task_file,
        outdir=args.outdir,
        overwrite=args.overwrite,
    )
    print("[DONE] Final report rebuilt without training or inference.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
