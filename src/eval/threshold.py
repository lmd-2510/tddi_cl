#!/usr/bin/env python3
"""Select an OOF threshold using raw normalized-entropy confidence and Macro-F1."""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Mapping, Sequence
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval.ensemble_ue import OfflineEnsembleArtifact, load_offline_ensemble_artifact
from src.eval.metrics import compute_aurc, compute_classification_metrics

THRESHOLD_CONFIG_SCHEMA_VERSION = 1
FROZEN_THRESHOLD_SCHEMA_VERSION = 5
FROZEN_THRESHOLD_KIND = "ddi_cil_frozen_confidence_threshold"
THRESHOLD_REPORT_SCHEMA_VERSION = 4
THRESHOLD_REPORT_KIND = "ddi_cil_confidence_threshold_report"
MACRO_F1_SELECTION_RULE = "max_macro_f1_subject_to_min_coverage"
SUPPORTED_TIE_BREAKERS = ("accuracy", "coverage", "lower_threshold")


@dataclass(frozen=True)
class ThresholdSelectionConfig:
    candidate_grid: tuple[float, ...]
    selection_rule: str
    minimum_coverage: float
    tie_breakers: tuple[str, ...]
    calibration_bins: int
    confidence_score: str
    probability_source: str
    low_threshold: float
    selection_source: str
    sha256: str
    source_path: Path = field(compare=False, repr=False)


@dataclass(frozen=True)
class FrozenThresholdArtifact:
    candidate_grid: tuple[float, ...]
    selection_rule: str
    minimum_coverage: float
    tie_breakers: tuple[str, ...]
    selected_threshold: float | None
    validation_accuracy: float | None
    validation_macro_f1: float | None
    validation_coverage: float | None
    validation_selected_count: int
    validation_total_count: int
    timestamp_utc: str
    config_sha256: str
    source_split: str
    source_ensemble_sha256: str
    method: str
    method_protocol: str
    task_id: int
    experiment_seed: int
    raw_class_ids: tuple[int, ...]
    calibration_bins: int
    confidence_score: str
    probability_source: str
    candidate_results: tuple[Mapping[str, Any], ...]
    selection_status: str = "macro_f1_selected"
    assignment_sha256: str | None = None
    fold_manifest_sha256: str | None = None
    task_file_sha256: str | None = None
    study_contract_sha256: str | None = None
    source_ensemble_mode: str = "stratified_3fold"
    source_member_ids: tuple[int, ...] = ()
    low_threshold: float = 0.5
    loaded_from_path: Path | None = field(default=None, compare=False, repr=False)

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _require_object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value

def load_threshold_selection_config(path):
    path = Path(path)
    payload = _require_object(json.loads(path.read_text(encoding="utf-8")), "Threshold config")
    allowed = {"schema_version", "confidence_score", "probability_source", "selection_source",
               "candidate_grid", "selection_rule", "low_threshold", "calibration_bins"}
    if set(payload) - allowed:
        raise ValueError("Unknown threshold config fields.")
    if payload.get("schema_version") != THRESHOLD_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported threshold config schema.")
    if (payload.get("confidence_score") != "entropy_confidence"
            or payload.get("probability_source") != "raw" or payload.get("selection_source") != "oof"):
        raise ValueError("Threshold requires raw entropy_confidence and OOF selection.")
    rule = _require_object(payload.get("selection_rule"), "selection_rule")
    if set(rule) != {"name", "minimum_coverage", "tie_breakers"} or rule["name"] != MACRO_F1_SELECTION_RULE:
        raise ValueError("Only Macro-F1 selection is supported.")
    if tuple(rule["tie_breakers"]) != SUPPORTED_TIE_BREAKERS:
        raise ValueError("Unsupported tie breakers.")
    grid = payload.get("candidate_grid")
    if not isinstance(grid, list) or not grid or any(type(v) not in (int, float) for v in grid):
        raise ValueError("candidate_grid must be a non-empty numeric list.")
    grid = tuple(float(v) for v in grid)
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in grid) or grid != tuple(sorted(set(grid))):
        raise ValueError("candidate_grid must be strictly increasing in [0,1].")
    coverage, low = float(rule["minimum_coverage"]), float(payload.get("low_threshold", 0.5))
    if not 0 <= coverage <= 1 or not 0 <= low <= min(grid):
        raise ValueError("Invalid coverage/low threshold.")
    bins = payload.get("calibration_bins", 15)
    if type(bins) is not int or bins <= 0:
        raise ValueError("calibration_bins must be a positive integer.")
    return ThresholdSelectionConfig(grid, rule["name"], coverage, SUPPORTED_TIE_BREAKERS,
        bins, "entropy_confidence", "raw", low, "oof", _sha256_file(path), path.resolve())


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _confidence_values(
    ensemble: OfflineEnsembleArtifact,
    confidence_score: str,
) -> np.ndarray:
    if confidence_score != "entropy_confidence":
        raise ValueError(f"Unsupported confidence score: {confidence_score!r}.")
    values = np.asarray(getattr(ensemble, confidence_score), dtype=np.float64)
    if values.shape != (ensemble.row_count,) or not np.isfinite(values).all():
        raise ValueError(f"Confidence score {confidence_score} must be a finite row vector.")
    return values

def _selected_metrics(
    ensemble: OfflineEnsembleArtifact,
    threshold: float,
    *,
    confidence_score: str,
) -> dict[str, Any]:
    mask = _confidence_values(ensemble, confidence_score) >= threshold
    count = int(mask.sum())
    total = ensemble.row_count
    coverage = float(count / total)
    if count == 0:
        accuracy: float | None = None
        macro_f1: float | None = None
        balanced_accuracy: float | None = None
    else:
        metrics = compute_classification_metrics(
            ensemble.labels[mask],
            ensemble.predictions[mask],
            labels=ensemble.raw_class_ids.astype(int).tolist(),
        )
        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        balanced_accuracy = metrics["balanced_accuracy"]
    return {
        "confidence_score": confidence_score,
        "threshold": float(threshold),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "coverage": coverage,
        "selected_count": count,
        "total_count": total,
    }

def _choose(candidates, minimum_coverage):
    eligible = [row for row in candidates if row["selected_count"] > 0 and row["coverage"] >= minimum_coverage]
    return max(eligible, key=lambda r: (r["macro_f1"], r["accuracy"], r["coverage"], -r["threshold"])) if eligible else None


def select_confidence_threshold(
    validation_ensemble: OfflineEnsembleArtifact,
    config: ThresholdSelectionConfig,
    *,
    source_ensemble_path: str | Path,
) -> FrozenThresholdArtifact:
    """Select from the mode-appropriate OOF/validation source."""

    expected_source = "oof"
    if validation_ensemble.context.ensemble_mode != "stratified_3fold" or validation_ensemble.context.split != "oof":
        raise ValueError("Threshold selection is OOF-only for stratified_3fold, never test.")
    if config.selection_source != "oof" or config.confidence_score != "entropy_confidence" or config.probability_source != "raw":
        raise ValueError("Only raw OOF normalized-entropy confidence is supported.")
    source_path = Path(source_ensemble_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source ensemble artifact: {source_path}")

    candidates = tuple(
        _selected_metrics(
            validation_ensemble,
            threshold,
            confidence_score=config.confidence_score,
        )
        for threshold in config.candidate_grid
    )
    selected = _choose(candidates, config.minimum_coverage)
    selection_status = "macro_f1_selected" if selected is not None else "no_selection"
    provenance = validation_ensemble.member_provenance
    shared = provenance[0] if provenance is not None else None
    return FrozenThresholdArtifact(
        candidate_grid=config.candidate_grid,
        selection_rule=config.selection_rule,
        minimum_coverage=config.minimum_coverage,
        tie_breakers=config.tie_breakers,
        selected_threshold=None if selected is None else float(selected["threshold"]),
        validation_accuracy=None if selected is None else float(selected["accuracy"]),
        validation_macro_f1=None if selected is None else float(selected["macro_f1"]),
        validation_coverage=None if selected is None else float(selected["coverage"]),
        validation_selected_count=0 if selected is None else int(selected["selected_count"]),
        validation_total_count=validation_ensemble.row_count,
        timestamp_utc=_utc_timestamp(),
        config_sha256=config.sha256,
        source_split=expected_source,
        source_ensemble_sha256=_sha256_file(source_path),
        method=validation_ensemble.context.method,
        method_protocol=validation_ensemble.context.method_protocol,
        task_id=validation_ensemble.context.task_id,
        experiment_seed=validation_ensemble.context.experiment_seed,
        raw_class_ids=tuple(int(value) for value in validation_ensemble.raw_class_ids),
        calibration_bins=config.calibration_bins,
        confidence_score=config.confidence_score,
        probability_source=config.probability_source,
        candidate_results=candidates,
        selection_status=selection_status,
        assignment_sha256=None if shared is None else shared.assignment_sha256,
        fold_manifest_sha256=None if shared is None else shared.fold_manifest_sha256,
        task_file_sha256=None if shared is None else shared.task_file_sha256,
        study_contract_sha256=None if shared is None else shared.study_contract_sha256,
        source_ensemble_mode=validation_ensemble.context.ensemble_mode,
        source_member_ids=tuple(
            int(value) for value in validation_ensemble.context.member_ids
        ),
        low_threshold=config.low_threshold,
    )

def _frozen_threshold_to_dict(artifact: FrozenThresholdArtifact) -> dict[str, Any]:
    payload = {
        "schema_version": FROZEN_THRESHOLD_SCHEMA_VERSION,
        "artifact_kind": FROZEN_THRESHOLD_KIND,
        "frozen": True,
        "timestamp_utc": artifact.timestamp_utc,
        "config_sha256": artifact.config_sha256,
        "confidence_score": artifact.confidence_score,
        "probability_source": artifact.probability_source,
        "candidate_grid": list(artifact.candidate_grid),
        "selection_rule": {
            "name": artifact.selection_rule,
            "minimum_coverage": artifact.minimum_coverage,
            "tie_breakers": list(artifact.tie_breakers),
        },
        "selection_status": artifact.selection_status,
        "low_threshold": artifact.low_threshold,
        "selected_threshold": artifact.selected_threshold,
        "validation_metrics": {
            "accuracy": artifact.validation_accuracy,
            "macro_f1": artifact.validation_macro_f1,
            "coverage": artifact.validation_coverage,
            "selected_count": artifact.validation_selected_count,
            "total_count": artifact.validation_total_count,
        },
        "source": {
            "split": artifact.source_split,
            "ensemble_sha256": artifact.source_ensemble_sha256,
            "method": artifact.method,
            "method_protocol": artifact.method_protocol,
            "task_id": artifact.task_id,
            "experiment_seed": artifact.experiment_seed,
            "raw_class_ids": list(artifact.raw_class_ids),
            "ensemble_mode": artifact.source_ensemble_mode,
            "member_ids": list(artifact.source_member_ids),
            "study_provenance": {
                "assignment_sha256": artifact.assignment_sha256,
                "fold_manifest_sha256": artifact.fold_manifest_sha256,
                "task_file_sha256": artifact.task_file_sha256,
                "study_contract_sha256": artifact.study_contract_sha256,
            },
        },
        "calibration_bins": artifact.calibration_bins,
        "candidate_results": [dict(candidate) for candidate in artifact.candidate_results],
    }
    return payload

def _atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path

def export_frozen_threshold_artifact(
    artifact: FrozenThresholdArtifact,
    path: str | Path,
) -> Path:
    """Persist the selected OOF/validation threshold as an immutable JSON artifact."""

    if artifact.source_split not in {"validation", "oof"}:
        raise ValueError("A frozen threshold must originate from validation or OOF.")
    return _atomic_write_json(_frozen_threshold_to_dict(artifact), path)

def load_frozen_threshold_artifact(path, *, config_path=None):
    path = Path(path)
    payload = _require_object(json.loads(path.read_text(encoding="utf-8")), "Frozen threshold")
    if payload.get("schema_version") != FROZEN_THRESHOLD_SCHEMA_VERSION:
        raise ValueError("Unsupported frozen threshold schema; recreate the threshold from OOF.")
    if payload.get("artifact_kind") != FROZEN_THRESHOLD_KIND or payload.get("frozen") is not True:
        raise ValueError("Expected a frozen threshold artifact.")
    source, rule, metrics = payload["source"], payload["selection_rule"], payload["validation_metrics"]
    provenance = source["study_provenance"]
    artifact = FrozenThresholdArtifact(
        candidate_grid=tuple(payload["candidate_grid"]), selection_rule=rule["name"],
        minimum_coverage=rule["minimum_coverage"], tie_breakers=tuple(rule["tie_breakers"]),
        selected_threshold=payload["selected_threshold"], validation_accuracy=metrics["accuracy"],
        validation_macro_f1=metrics["macro_f1"], validation_coverage=metrics["coverage"],
        validation_selected_count=metrics["selected_count"], validation_total_count=metrics["total_count"],
        timestamp_utc=payload["timestamp_utc"], config_sha256=payload["config_sha256"],
        source_split=source["split"], source_ensemble_sha256=source["ensemble_sha256"],
        method=source["method"], method_protocol=source["method_protocol"], task_id=source["task_id"],
        experiment_seed=source["experiment_seed"], raw_class_ids=tuple(source["raw_class_ids"]),
        calibration_bins=payload["calibration_bins"], confidence_score=payload["confidence_score"],
        probability_source=payload["probability_source"], candidate_results=tuple(payload["candidate_results"]),
        selection_status=payload["selection_status"], assignment_sha256=provenance["assignment_sha256"],
        fold_manifest_sha256=provenance["fold_manifest_sha256"], task_file_sha256=provenance["task_file_sha256"],
        study_contract_sha256=provenance["study_contract_sha256"], source_ensemble_mode=source["ensemble_mode"],
        source_member_ids=tuple(source["member_ids"]), low_threshold=payload["low_threshold"],
        loaded_from_path=path.resolve())
    _validate_frozen_threshold(artifact)
    if config_path is not None:
        config = load_threshold_selection_config(config_path)
        fields = ("candidate_grid", "selection_rule", "minimum_coverage", "tie_breakers",
                  "calibration_bins", "confidence_score", "probability_source", "low_threshold")
        if artifact.config_sha256 != config.sha256 or any(getattr(artifact, k) != getattr(config, k) for k in fields):
            raise ValueError("Frozen threshold config hash/metadata mismatch.")
    return artifact


def _validate_frozen_threshold(artifact):
    if artifact.selection_rule != MACRO_F1_SELECTION_RULE or artifact.tie_breakers != SUPPORTED_TIE_BREAKERS:
        raise ValueError("Only Macro-F1 threshold selection is supported.")
    if artifact.confidence_score != "entropy_confidence" or artifact.probability_source != "raw":
        raise ValueError("Only raw entropy_confidence is supported.")
    if artifact.source_split != "oof" or artifact.source_ensemble_mode != "stratified_3fold":
        raise ValueError("Frozen threshold source must be OOF.")
    grid = artifact.candidate_grid
    if not grid or tuple(sorted(set(grid))) != grid or any(not np.isfinite(v) or not 0 <= v <= 1 for v in grid):
        raise ValueError("Invalid frozen threshold grid.")
    if not 0 <= artifact.minimum_coverage <= 1 or not 0 <= artifact.low_threshold <= min(grid):
        raise ValueError("Invalid frozen coverage/low threshold.")
    if artifact.validation_total_count <= 0 or artifact.calibration_bins <= 0:
        raise ValueError("Invalid threshold counts/bins.")
    if tuple(r["threshold"] for r in artifact.candidate_results) != grid:
        raise ValueError("Frozen candidates/grid mismatch.")
    for row in artifact.candidate_results:
        n = row["selected_count"]
        if type(n) is not int or not 0 <= n <= artifact.validation_total_count:
            raise ValueError("Invalid candidate count.")
        if row["total_count"] != artifact.validation_total_count or not np.isclose(row["coverage"], n / artifact.validation_total_count):
            raise ValueError("Invalid candidate coverage.")
        if row["confidence_score"] != "entropy_confidence":
            raise ValueError("Invalid candidate score.")
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            v = row[metric]
            if n and (v is None or not np.isfinite(v) or not 0 <= v <= 1):
                raise ValueError("Invalid candidate metric.")
            if not n and v is not None:
                raise ValueError("Empty candidate must have null metrics.")
    selected = _choose(artifact.candidate_results, artifact.minimum_coverage)
    if selected is None:
        if artifact.selection_status != "no_selection" or artifact.selected_threshold is not None or artifact.validation_selected_count != 0 or any(
            v is not None for v in (artifact.validation_accuracy, artifact.validation_macro_f1, artifact.validation_coverage)):
            raise ValueError("Invalid no_selection state.")
    else:
        if artifact.selection_status != "macro_f1_selected" or artifact.selected_threshold != selected["threshold"]:
            raise ValueError("Frozen selection does not maximize Macro-F1.")
        for attr, key in (("validation_accuracy","accuracy"), ("validation_macro_f1","macro_f1"),
                          ("validation_coverage","coverage"), ("validation_selected_count","selected_count")):
            if getattr(artifact, attr) != selected[key]:
                raise ValueError("Frozen selected metrics mismatch.")
    for digest in (artifact.config_sha256, artifact.source_ensemble_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid threshold source/config hash.")
    if artifact.source_member_ids != (0, 1, 2) or not artifact.raw_class_ids or len(set(artifact.raw_class_ids)) != len(artifact.raw_class_ids):
        raise ValueError("Invalid frozen member/class provenance.")
    if not artifact.method or not artifact.method_protocol or artifact.task_id < 0 or not artifact.timestamp_utc:
        raise ValueError("Missing threshold provenance.")
    hashes = (artifact.assignment_sha256, artifact.fold_manifest_sha256, artifact.task_file_sha256, artifact.study_contract_sha256)
    if any(v is not None for v in hashes) and not all(isinstance(v, str) and len(v) == 64 for v in hashes):
        raise ValueError("Incomplete study provenance.")


def _validate_threshold_context(
    ensemble: OfflineEnsembleArtifact,
    threshold: FrozenThresholdArtifact,
) -> None:
    expected = {
        "method": threshold.method,
        "method_protocol": threshold.method_protocol,
        "task_id": threshold.task_id,
        "experiment_seed": threshold.experiment_seed,
    }
    actual = {
        "method": ensemble.context.method,
        "method_protocol": ensemble.context.method_protocol,
        "task_id": ensemble.context.task_id,
        "experiment_seed": ensemble.context.experiment_seed,
    }
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if tuple(int(value) for value in ensemble.raw_class_ids) != threshold.raw_class_ids:
        mismatches.append("raw_class_ids")
    expected_study = (
        threshold.assignment_sha256, threshold.fold_manifest_sha256,
        threshold.task_file_sha256, threshold.study_contract_sha256,
    )
    if any(value is not None for value in expected_study):
        if ensemble.member_provenance is None:
            mismatches.append("study_provenance")
        else:
            source = ensemble.member_provenance[0]
            actual_study = (
                source.assignment_sha256, source.fold_manifest_sha256,
                source.task_file_sha256, source.study_contract_sha256,
            )
            if actual_study != expected_study:
                mismatches.append("study_provenance")
    if threshold.source_ensemble_mode != ensemble.context.ensemble_mode:
        mismatches.append("ensemble_mode")
    if threshold.source_member_ids and (
        tuple(int(value) for value in ensemble.context.member_ids)
        != threshold.source_member_ids
    ):
        mismatches.append("member_ids")
    if mismatches:
        raise ValueError(
            "Ensemble does not match frozen threshold provenance: "
            + ", ".join(mismatches)
        )

def _probability_metrics(
    ensemble: OfflineEnsembleArtifact,
    *,
    calibration_bins: int,
    mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute max-probability ECE, multiclass NLL and Brier score."""

    if mask is None:
        mask = np.ones(ensemble.row_count, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (ensemble.row_count,):
        raise ValueError("Probability metric mask row count mismatch.")
    if not mask.any():
        return {"ece": None, "negative_log_likelihood": None, "brier_score": None}
    probabilities = np.asarray(ensemble.probabilities[mask], dtype=np.float64)
    labels = ensemble.labels[mask]
    predictions = ensemble.predictions[mask]
    class_to_column = {
        int(raw): column for column, raw in enumerate(ensemble.raw_class_ids.tolist())
    }
    columns = np.asarray([class_to_column[int(raw)] for raw in labels], dtype=np.int64)
    row_indices = np.arange(labels.shape[0])
    true_probability = np.clip(
        probabilities[row_indices, columns], np.finfo(np.float64).tiny, 1.0
    )
    nll = float(-np.log(true_probability).mean())
    targets = np.zeros_like(probabilities)
    targets[row_indices, columns] = 1.0
    brier = float(np.square(probabilities - targets).sum(axis=1).mean())
    max_probability = probabilities.max(axis=1)
    correct = (predictions == labels).astype(np.float64)
    ece = 0.0
    for index in range(calibration_bins):
        lower, upper = index / calibration_bins, (index + 1) / calibration_bins
        in_bin = (max_probability >= lower) & (
            max_probability <= upper if index == calibration_bins - 1
            else max_probability < upper
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(max_probability[in_bin].mean())
            )
    return {"ece": float(ece), "negative_log_likelihood": nll, "brier_score": brier}

def evaluate_with_frozen_threshold(
    ensemble: OfflineEnsembleArtifact,
    threshold: FrozenThresholdArtifact,
) -> dict[str, Any]:
    """Report full and selected-subset metrics using a disk-loaded threshold."""

    if ensemble.context.split.casefold() == "test" and threshold.loaded_from_path is None:
        raise ValueError(
            "Test evaluation requires a frozen threshold loaded from its artifact file."
        )
    _validate_frozen_threshold(threshold)
    if ensemble.context.ensemble_mode != "stratified_3fold" or ensemble.context.split not in {"oof", "test"}:
        raise ValueError("Threshold evaluation requires stratified_3fold OOF or test.")
    _validate_threshold_context(ensemble, threshold)

    class_ids = ensemble.raw_class_ids.astype(int).tolist()
    full_classification = compute_classification_metrics(
        ensemble.labels,
        ensemble.predictions,
        labels=class_ids,
    )
    selection_applied = threshold.selected_threshold is not None
    if selection_applied:
        selected = _selected_metrics(
            ensemble, threshold.selected_threshold,
            confidence_score=threshold.confidence_score,
        )
        selected_mask = _confidence_values(
            ensemble, threshold.confidence_score
        ) >= threshold.selected_threshold
    else:
        # Explicit no-selection fallback: do not silently invent a threshold or
        # discard rows. The report carries full-set values with applied=False.
        selected = {
            "selected_count": ensemble.row_count, "total_count": ensemble.row_count,
            "accuracy": full_classification["accuracy"],
            "macro_f1": full_classification["macro_f1"], "coverage": 1.0,
        }
        selected_mask = np.ones(ensemble.row_count, dtype=bool)
    confidence = np.asarray(ensemble.entropy_confidence, dtype=np.float64)

    def tier_metrics(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        payload: dict[str, Any] = {
            "sample_count": count,
            "coverage": float(count / ensemble.row_count),
        }
        if count == 0:
            payload.update({"accuracy": None, "macro_f1": None, "weighted_f1": None})
        else:
            payload.update(
                compute_classification_metrics(
                    ensemble.labels[mask],
                    ensemble.predictions[mask],
                    labels=class_ids,
                )
            )
        return payload

    high_mask = (
        confidence >= threshold.selected_threshold
        if selection_applied else np.zeros(ensemble.row_count, dtype=bool)
    )
    medium_mask = (confidence >= threshold.low_threshold) & ~high_mask
    low_mask = confidence < threshold.low_threshold
    selected_metrics = {
        "selected_count": selected["selected_count"],
        "total_count": selected["total_count"],
        "accuracy": selected["accuracy"],
        "macro_f1": selected["macro_f1"],
        "coverage": selected["coverage"],
    }
    selected_metrics["balanced_accuracy"] = selected.get("balanced_accuracy", full_classification["balanced_accuracy"])
    selected_probability_metrics = _probability_metrics(
        ensemble,
        calibration_bins=threshold.calibration_bins,
        mask=selected_mask,
    )
    return {
        "schema_version": THRESHOLD_REPORT_SCHEMA_VERSION,
        "artifact_kind": THRESHOLD_REPORT_KIND,
        "timestamp_utc": _utc_timestamp(),
        "evaluation_split": ensemble.context.split,
        "method": ensemble.context.method,
        "method_protocol": ensemble.context.method_protocol,
        "task_id": ensemble.context.task_id,
        "experiment_seed": ensemble.context.experiment_seed,
        "raw_class_ids": class_ids,
        "threshold_score_name": threshold.confidence_score,
        "threshold_value": threshold.selected_threshold,
        "threshold_probability_source": threshold.probability_source,
        "selection_status": threshold.selection_status,
        "score_semantics": {
            "entropy_confidence": (
                "1 - predictive_entropy/log(class_count); it is not max probability."
            ),
            "selection_interpretation": (
                "Maximizing selected-subset OOF Macro-F1 does not guarantee the same score "
                "on test data."
            ),
        },
        "threshold": {
            "score_name": threshold.confidence_score,
            "probability_source": threshold.probability_source,
            "value": threshold.selected_threshold,
            "selection_status": threshold.selection_status,
                "selection_applied": selection_applied,
            "source_split": threshold.source_split,
            "config_sha256": threshold.config_sha256,
            "loaded_from": str(threshold.loaded_from_path) if threshold.loaded_from_path else None,
        },
        "selection_candidates": {
            "source_split": threshold.source_split,
            "confidence_score": threshold.confidence_score,
            "probability_source": threshold.probability_source,
            "candidate_grid": list(threshold.candidate_grid),
            "results": [dict(candidate) for candidate in threshold.candidate_results],
        },
        "full_set": {
            "sample_count": ensemble.row_count,
            **full_classification,
            **_probability_metrics(
                ensemble, calibration_bins=threshold.calibration_bins
            ),
            "aurc": compute_aurc(
                ensemble.labels,
                ensemble.predictions,
                ensemble.entropy_confidence,
            ),
        },
        "threshold_score_selective_metrics": {
            "threshold_score_name": threshold.confidence_score,
            "probability_source": threshold.probability_source,
            "threshold_value": threshold.selected_threshold,
            **selected_metrics,
            "selection_applied": selection_applied,
            **selected_probability_metrics,
        },
        "confidence_tiers": {
            "low_threshold": threshold.low_threshold,
            "high_threshold": threshold.selected_threshold,
            "selection_applied": selection_applied,
            "high": tier_metrics(high_mask),
            "medium": tier_metrics(medium_mask),
            "low": tier_metrics(low_mask),
        },
    }

def export_threshold_report(report: Mapping[str, Any], path: str | Path) -> Path:
    if report.get("artifact_kind") != THRESHOLD_REPORT_KIND:
        raise ValueError("Invalid confidence-threshold report payload.")
    return _atomic_write_json(report, path)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select/freeze an OOF or validation entropy threshold and evaluate it."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select", help="Select using OOF/validation and freeze.")
    select.add_argument("--ensemble", required=True, type=Path)
    select.add_argument("--config", required=True, type=Path)
    select.add_argument("--threshold-out", required=True, type=Path)
    select.add_argument("--report-out", required=True, type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Load frozen threshold and evaluate.")
    evaluate.add_argument("--ensemble", required=True, type=Path)
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--threshold-artifact", required=True, type=Path)
    evaluate.add_argument("--report-out", required=True, type=Path)
    return parser

def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    ensemble = load_offline_ensemble_artifact(args.ensemble)
    if args.command == "select":
        config = load_threshold_selection_config(args.config)
        selected = select_confidence_threshold(
            ensemble,
            config,
            source_ensemble_path=args.ensemble,
        )
        threshold_path = export_frozen_threshold_artifact(selected, args.threshold_out)
        frozen = load_frozen_threshold_artifact(threshold_path, config_path=args.config)
        report = evaluate_with_frozen_threshold(ensemble, frozen)
    else:
        frozen = load_frozen_threshold_artifact(
            args.threshold_artifact,
            config_path=args.config,
        )
        report = evaluate_with_frozen_threshold(ensemble, frozen)
        threshold_path = args.threshold_artifact
    report_path = export_threshold_report(report, args.report_out)
    threshold_value = (
        "none"
        if frozen.selected_threshold is None
        else f"{frozen.selected_threshold:.6g}"
    )
    print(
        f"threshold_score={frozen.confidence_score} "
        f"threshold_value={threshold_value} "
        f"threshold_artifact={threshold_path} "
        f"split={ensemble.context.split} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
