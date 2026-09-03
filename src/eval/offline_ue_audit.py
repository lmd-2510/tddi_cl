#!/usr/bin/env python3
"""Report-only uncertainty audit for an existing offline ensemble artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.classification_metrics import compute_classification_metrics  # noqa: E402
from src.eval.offline_ensemble import (  # noqa: E402
    OfflineEnsembleArtifact,
    load_offline_ensemble_artifact,
)


OFFLINE_UE_AUDIT_SCHEMA_VERSION = 1
OFFLINE_UE_AUDIT_KIND = "ddi_cil_offline_ue_audit"
SUPPORTED_PROTOCOL_NAMES = {
    "p3",
    "tail_to_head",
    "p4",
    "constrained_mass_balanced",
}
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
SELECTIVE_COVERAGE_TARGETS = (0.25, 0.5, 0.75, 0.9, 1.0)
RISK_CURVE_COVERAGE_GRID = tuple(float(value) / 100.0 for value in range(1, 101))
RARITY_RULES = {
    "ultra_tail": "train_count <= 20",
    "tail": "21 <= train_count <= 100",
    "medium": "101 <= train_count <= 1000",
    "head": "train_count > 1000",
}
SCORE_NAMES = (
    "predictive_entropy",
    "normalized_entropy",
    "mutual_information",
    "normalized_mi",
    "member_normalized_mi",
    "mean_probability_variance",
    "total_probability_variance",
    "pairwise_disagreement",
    "one_minus_max_probability",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_config() -> dict[str, Any]:
    return {
        "score_names": list(SCORE_NAMES),
        "score_direction": "higher_means_more_uncertain",
        "error_positive_class": True,
        "quantiles": list(QUANTILES),
        "selective_order": "ascending_uncertainty_stable_source_order",
        "selective_coverage_targets": list(SELECTIVE_COVERAGE_TARGETS),
        "risk_curve_coverage_grid": list(RISK_CURVE_COVERAGE_GRID),
        "aurc_definition": "mean_prefix_error_risk_over_all_nonempty_coverages",
        "macro_f1_label_scope": "all_seen_raw_class_ids",
        "breakdown_assignment": "true_raw_label",
        "breakdown_coverage": "group_sample_count_divided_by_all_samples",
        "rarity_source": "train_counts_only",
        "rarity_rules": dict(RARITY_RULES),
        "test_policy": "report_only_fixed_rules_no_threshold_or_score_selection",
    }


def _require_json_object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def load_cil_task_file(
    path: str | Path,
    *,
    ensemble: OfflineEnsembleArtifact,
) -> tuple[Mapping[str, Any], set[int], set[int]]:
    """Load a supported CIL task schedule and return old/current classes."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing CIL task file: {path}")
    try:
        payload = _require_json_object(
            json.loads(path.read_text(encoding="utf-8")),
            "CIL task file",
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid CIL task-file JSON: {path}") from error

    protocol = str(payload.get("protocol", "")).casefold()
    if protocol not in SUPPORTED_PROTOCOL_NAMES:
        raise ValueError(
            "UE audit requires a supported P3/P4 CIL task file; "
            f"got protocol={protocol!r}."
        )
    task_seed = payload.get("seed")
    if task_seed is not None and int(task_seed) != ensemble.context.experiment_seed:
        raise ValueError("Task-file seed does not match ensemble experiment seed.")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("CIL task file must contain a non-empty tasks list.")

    tasks_by_id: dict[int, set[int]] = {}
    all_classes: set[int] = set()
    for raw_task in raw_tasks:
        task = _require_json_object(raw_task, "CIL task")
        task_id = int(task.get("task_id", -1))
        raw_classes = task.get("classes")
        if task_id < 0 or task_id in tasks_by_id:
            raise ValueError("CIL task IDs must be unique non-negative integers.")
        if not isinstance(raw_classes, list) or not raw_classes:
            raise ValueError(f"CIL task {task_id} must contain a non-empty classes list.")
        classes = {int(class_id) for class_id in raw_classes}
        if len(classes) != len(raw_classes):
            raise ValueError(f"CIL task {task_id} contains duplicate class IDs.")
        overlap = all_classes.intersection(classes)
        if overlap:
            raise ValueError(f"CIL class IDs occur in multiple tasks: {sorted(overlap)}")
        tasks_by_id[task_id] = classes
        all_classes.update(classes)

    expected_ids = list(range(max(tasks_by_id) + 1))
    if sorted(tasks_by_id) != expected_ids:
        raise ValueError("CIL task IDs must be contiguous from zero.")
    task_id = ensemble.context.task_id
    if task_id not in tasks_by_id:
        raise ValueError(f"Ensemble task_id={task_id} is absent from the CIL task file.")
    old_classes = set().union(*(tasks_by_id[index] for index in range(task_id)))
    current_classes = tasks_by_id[task_id]
    seen_classes = old_classes | current_classes
    artifact_classes = set(ensemble.raw_class_ids.astype(int).tolist())
    if artifact_classes != seen_classes:
        raise ValueError(
            "Ensemble raw_class_ids do not match CIL classes seen through task "
            f"{task_id}."
        )
    return payload, old_classes, current_classes


def _validate_count_mapping(counts: Mapping[int, int]) -> dict[int, int]:
    normalized = {int(class_id): int(count) for class_id, count in counts.items()}
    if not normalized:
        raise ValueError("Train class counts cannot be empty.")
    if any(count <= 0 for count in normalized.values()):
        raise ValueError("Every train class count must be positive.")
    return normalized


def _counts_from_frame(frame: pd.DataFrame) -> dict[int, int]:
    class_column = next(
        (name for name in ("class_id", "raw_class_id") if name in frame.columns),
        None,
    )
    count_column = next(
        (name for name in ("count", "train_count") if name in frame.columns),
        None,
    )
    if class_column is None or count_column is None:
        raise ValueError(
            "Train class-count table requires class_id/raw_class_id and count/train_count."
        )
    selected = frame[[class_column, count_column]].copy()
    if selected.isna().any().any():
        raise ValueError("Train class-count table contains missing values.")
    class_ids = selected[class_column].astype(np.int64)
    counts = selected[count_column].astype(np.int64)
    if class_ids.duplicated().any():
        raise ValueError("Train class-count table contains duplicate class IDs.")
    return _validate_count_mapping(
        dict(zip(class_ids.astype(int), counts.astype(int), strict=True))
    )


def load_train_class_count_artifact(path: str | Path) -> dict[int, int]:
    """Load train-only class counts from CSV or JSON."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing train class-count artifact: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return _counts_from_frame(pd.read_csv(path))
    if suffix == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid train class-count JSON: {path}") from error
        if isinstance(raw, dict) and "class_counts" in raw:
            raw = raw["class_counts"]
        if isinstance(raw, dict):
            return _validate_count_mapping(
                {int(class_id): int(count) for class_id, count in raw.items()}
            )
        if isinstance(raw, list):
            return _counts_from_frame(pd.DataFrame(raw))
        raise ValueError("Train class-count JSON must be a mapping or row list.")
    raise ValueError("Train class-count artifact must use .csv or .json.")


def count_train_classes_from_data(
    path: str | Path,
    *,
    label_column: str = "class",
) -> dict[int, int]:
    """Count train labels without loading descriptor columns into memory."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing train data: {path}")
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        labels = pq.read_table(path, columns=[label_column])[label_column].to_numpy()
        class_ids, counts = np.unique(labels.astype(np.int64, copy=False), return_counts=True)
        return _validate_count_mapping(
            dict(zip(class_ids.astype(int), counts.astype(int), strict=True))
        )
    if suffix == ".csv":
        totals: dict[int, int] = {}
        for chunk in pd.read_csv(path, usecols=[label_column], chunksize=250_000):
            values = chunk[label_column].value_counts()
            for class_id, count in values.items():
                normalized_id = int(class_id)
                totals[normalized_id] = totals.get(normalized_id, 0) + int(count)
        return _validate_count_mapping(totals)
    raise ValueError("Train data must use .parquet, .pq, or .csv.")


def _score_arrays(ensemble: OfflineEnsembleArtifact) -> dict[str, np.ndarray]:
    scores = {
        name: np.asarray(getattr(ensemble, name), dtype=np.float64)
        for name in SCORE_NAMES
        if name != "one_minus_max_probability"
    }
    scores["one_minus_max_probability"] = 1.0 - np.asarray(
        ensemble.max_probability,
        dtype=np.float64,
    )
    for name, values in scores.items():
        if values.shape != (ensemble.row_count,) or not np.isfinite(values).all():
            raise ValueError(f"Uncertainty score {name} must be a finite row vector.")
    return scores


def _optional_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def _score_summary(
    values: np.ndarray,
    errors: np.ndarray,
) -> dict[str, Any]:
    correct = ~errors
    quantile_values = np.quantile(values, QUANTILES)
    error_count = int(errors.sum())
    correct_count = int(correct.sum())
    if error_count == 0 or correct_count == 0:
        auroc: float | None = None
        auprc: float | None = None
        status = "undefined_single_error_class"
    else:
        binary_errors = errors.astype(np.int8)
        auroc = float(roc_auc_score(binary_errors, values))
        auprc = float(average_precision_score(binary_errors, values))
        status = "ok"
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "quantiles": {
            f"q{int(round(quantile * 100)):02d}": float(value)
            for quantile, value in zip(QUANTILES, quantile_values, strict=True)
        },
        "mean_correct": _optional_mean(values[correct]),
        "mean_error": _optional_mean(values[errors]),
        "error_detection": {
            "positive_class": "prediction_error",
            "status": status,
            "error_count": error_count,
            "correct_count": correct_count,
            "auroc": auroc,
            "auprc": auprc,
        },
    }


def _nearest_prefix_count(coverage: float, row_count: int) -> int:
    return min(row_count, max(1, int(np.floor(coverage * row_count + 0.5))))


def _classification_at_indices(
    ensemble: OfflineEnsembleArtifact,
    indices: np.ndarray,
) -> dict[str, float]:
    metrics = compute_classification_metrics(
        ensemble.labels[indices],
        ensemble.predictions[indices],
        labels=ensemble.raw_class_ids.astype(int).tolist(),
    )
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def _selective_prediction(
    ensemble: OfflineEnsembleArtifact,
    values: np.ndarray,
    errors: np.ndarray,
) -> dict[str, Any]:
    # Source sample order is stable and is the predeclared tie breaker.
    order = np.argsort(values, kind="stable")
    ordered_errors = errors[order].astype(np.float64)
    prefix_counts = np.arange(1, ensemble.row_count + 1, dtype=np.float64)
    prefix_risk = np.cumsum(ordered_errors) / prefix_counts
    aurc = float(prefix_risk.mean())

    curve = []
    used_counts: set[int] = set()
    for target in RISK_CURVE_COVERAGE_GRID:
        count = _nearest_prefix_count(target, ensemble.row_count)
        if count in used_counts:
            continue
        used_counts.add(count)
        curve.append(
            {
                "coverage": float(count / ensemble.row_count),
                "risk": float(prefix_risk[count - 1]),
                "selected_count": count,
                "uncertainty_cutoff": float(values[order[count - 1]]),
            }
        )

    target_metrics = []
    for target in SELECTIVE_COVERAGE_TARGETS:
        count = _nearest_prefix_count(target, ensemble.row_count)
        selected = order[:count]
        metrics = _classification_at_indices(ensemble, selected)
        target_metrics.append(
            {
                "target_coverage": float(target),
                "actual_coverage": float(count / ensemble.row_count),
                "selected_count": count,
                "risk": float(prefix_risk[count - 1]),
                **metrics,
            }
        )
    return {
        "selection_rule": "retain_lowest_uncertainty_first",
        "tie_breaker": "stable_source_sample_order",
        "aurc": aurc,
        "risk_coverage_curve": curve,
        "coverage_metrics": target_metrics,
    }


def _rarity_group(class_id: int, counts: Mapping[int, int]) -> str:
    count = counts[class_id]
    if count <= 20:
        return "ultra_tail"
    if count <= 100:
        return "tail"
    if count <= 1000:
        return "medium"
    return "head"


def _group_report(
    ensemble: OfflineEnsembleArtifact,
    scores: Mapping[str, np.ndarray],
    *,
    class_ids: set[int],
) -> dict[str, Any]:
    ordered_classes = sorted(class_ids)
    mask = np.isin(ensemble.labels, ordered_classes)
    count = int(mask.sum())
    if count == 0:
        accuracy: float | None = None
        macro_f1: float | None = None
    else:
        metrics = compute_classification_metrics(
            ensemble.labels[mask],
            ensemble.predictions[mask],
            labels=ordered_classes,
        )
        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
    return {
        "class_ids": ordered_classes,
        "class_count": len(ordered_classes),
        "sample_count": count,
        "coverage": float(count / ensemble.row_count),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_uncertainty": {
            name: _optional_mean(values[mask]) for name, values in scores.items()
        },
    }


def build_offline_ue_audit(
    ensemble: OfflineEnsembleArtifact,
    *,
    task_payload: Mapping[str, Any],
    old_classes: set[int],
    current_classes: set[int],
    train_counts: Mapping[int, int],
    source_ensemble_sha256: str,
    task_file_sha256: str,
    train_count_source_sha256: str,
    train_count_source_kind: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a deterministic report; this function performs no model selection."""

    del task_payload  # Validation happens in load_cil_task_file; hashes preserve provenance.
    seen_classes = set(ensemble.raw_class_ids.astype(int).tolist())
    missing_counts = sorted(seen_classes - set(train_counts))
    if missing_counts:
        raise ValueError(f"Train counts are missing seen classes: {missing_counts}")

    scores = _score_arrays(ensemble)
    errors = ensemble.predictions != ensemble.labels
    score_reports: dict[str, Any] = {}
    for name, values in scores.items():
        score_reports[name] = {
            **_score_summary(values, errors),
            "selective_prediction": _selective_prediction(
                ensemble,
                values,
                errors,
            ),
        }

    rarity_classes = {name: set() for name in RARITY_RULES}
    for class_id in seen_classes:
        rarity_classes[_rarity_group(class_id, train_counts)].add(class_id)
    old_current = {
        "old": _group_report(
            ensemble,
            scores,
            class_ids=old_classes,
        ),
        "current_task": _group_report(
            ensemble,
            scores,
            class_ids=current_classes,
        ),
    }
    rarity = {
        name: _group_report(ensemble, scores, class_ids=class_ids)
        for name, class_ids in rarity_classes.items()
    }

    config = _audit_config()
    config_sha256 = _canonical_sha256(config)
    report = {
        "schema_version": OFFLINE_UE_AUDIT_SCHEMA_VERSION,
        "artifact_kind": OFFLINE_UE_AUDIT_KIND,
        "source_ensemble_sha256": source_ensemble_sha256,
        "task_file_sha256": task_file_sha256,
        "train_count_source_sha256": train_count_source_sha256,
        "train_count_source_kind": train_count_source_kind,
        "config_sha256": config_sha256,
        "split": ensemble.context.split,
        "task_id": ensemble.context.task_id,
        "experiment_seed": ensemble.context.experiment_seed,
        "member_ids": list(ensemble.context.member_ids),
        "member_count": ensemble.context.member_count,
        "method": ensemble.context.method,
        "method_protocol": ensemble.context.method_protocol,
        "policy": config["test_policy"],
        "config": config,
        "dataset": {
            "sample_count": ensemble.row_count,
            "correct_count": int((~errors).sum()),
            "error_count": int(errors.sum()),
            "accuracy": float((~errors).mean()),
        },
        "uncertainty_scores": score_reports,
        "breakdowns": {
            "old_vs_current": old_current,
            "rarity": rarity,
        },
    }
    return report, _csv_rows(report)


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score, score_report in report["uncertainty_scores"].items():
        detection = score_report["error_detection"]
        summary_row = {
            "record_type": "score_summary",
            "score": score,
            "mean": score_report["mean"],
            "std": score_report["std"],
            "mean_correct": score_report["mean_correct"],
            "mean_error": score_report["mean_error"],
            "auroc": detection["auroc"],
            "auprc": detection["auprc"],
            "status": detection["status"],
            "aurc": score_report["selective_prediction"]["aurc"],
        }
        summary_row.update(score_report["quantiles"])
        rows.append(summary_row)
        for point in score_report["selective_prediction"]["risk_coverage_curve"]:
            rows.append(
                {
                    "record_type": "risk_coverage",
                    "score": score,
                    **point,
                }
            )
        for point in score_report["selective_prediction"]["coverage_metrics"]:
            rows.append(
                {
                    "record_type": "selective_target",
                    "score": score,
                    **point,
                }
            )
    for dimension, groups in report["breakdowns"].items():
        for group_name, group in groups.items():
            base = {
                "record_type": "breakdown",
                "dimension": dimension,
                "group": group_name,
                "class_count": group["class_count"],
                "sample_count": group["sample_count"],
                "coverage": group["coverage"],
                "accuracy": group["accuracy"],
                "macro_f1": group["macro_f1"],
            }
            for score, mean_uncertainty in group["mean_uncertainty"].items():
                rows.append({**base, "score": score, "mean": mean_uncertainty})
    return rows


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"UE audit JSON already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"UE audit CSV already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fieldnames = sorted({key for row in rows for key in row})
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_offline_ue_audit(
    *,
    ensemble_path: str | Path,
    task_file: str | Path,
    output_json: str | Path,
    output_csv: str | Path | None = None,
    train_data: str | Path | None = None,
    train_class_counts: str | Path | None = None,
    label_column: str = "class",
) -> dict[str, Any]:
    """Load sources, build the report, and atomically write requested outputs."""

    if (train_data is None) == (train_class_counts is None):
        raise ValueError("Provide exactly one of train_data or train_class_counts.")
    ensemble_path = Path(ensemble_path)
    task_file = Path(task_file)
    output_json = Path(output_json)
    output_csv_path = None if output_csv is None else Path(output_csv)
    if output_json.exists():
        raise FileExistsError(f"UE audit JSON already exists: {output_json}")
    if output_csv_path is not None and output_csv_path.exists():
        raise FileExistsError(f"UE audit CSV already exists: {output_csv_path}")
    ensemble = load_offline_ensemble_artifact(ensemble_path)
    task_payload, old_classes, current_classes = load_cil_task_file(
        task_file,
        ensemble=ensemble,
    )
    if train_class_counts is not None:
        count_source = Path(train_class_counts)
        counts = load_train_class_count_artifact(count_source)
        count_source_kind = "train_class_count_artifact"
    else:
        count_source = Path(train_data)  # type: ignore[arg-type]
        counts = count_train_classes_from_data(count_source, label_column=label_column)
        count_source_kind = "train_data"
    report, csv_rows = build_offline_ue_audit(
        ensemble,
        task_payload=task_payload,
        old_classes=old_classes,
        current_classes=current_classes,
        train_counts=counts,
        source_ensemble_sha256=_sha256_file(ensemble_path),
        task_file_sha256=_sha256_file(task_file),
        train_count_source_sha256=_sha256_file(count_source),
        train_count_source_kind=count_source_kind,
    )
    _atomic_write_json(report, output_json)
    if output_csv_path is not None:
        _atomic_write_csv(csv_rows, output_csv_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report-only UE audit for one offline ensemble artifact."
    )
    parser.add_argument("--ensemble", required=True, type=Path)
    parser.add_argument("--task-file", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-data", type=Path)
    source.add_argument("--train-class-counts", type=Path)
    parser.add_argument("--label-column", default="class")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_offline_ue_audit(
        ensemble_path=args.ensemble,
        task_file=args.task_file,
        train_data=args.train_data,
        train_class_counts=args.train_class_counts,
        label_column=args.label_column,
        output_json=args.out_json,
        output_csv=args.out_csv,
    )
    print(
        f"exported={args.out_json} split={report['split']} task={report['task_id']} "
        f"rows={report['dataset']['sample_count']} errors={report['dataset']['error_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
