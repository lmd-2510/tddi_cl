#!/usr/bin/env python3
"""Paper-style entropy threshold selection from OOF or seeded validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import compute_aurc, compute_classification_metrics  # noqa: E402
from src.eval.ensemble_ue import (  # noqa: E402
    OfflineEnsembleArtifact,
    load_offline_ensemble_artifact,
)


THRESHOLD_CONFIG_SCHEMA_VERSION = 1
FROZEN_THRESHOLD_SCHEMA_VERSION = 3
LEGACY_FROZEN_THRESHOLD_SCHEMA_VERSIONS = (1, 2)
FROZEN_THRESHOLD_KIND = "ddi_cil_frozen_confidence_threshold"
THRESHOLD_REPORT_SCHEMA_VERSION = 2
THRESHOLD_REPORT_KIND = "ddi_cil_confidence_threshold_report"
PAPER_SELECTION_RULE = "smallest_threshold_meeting_target_accuracy"
LEGACY_SELECTION_RULE = "max_macro_f1_subject_to_min_coverage"
SUPPORTED_SELECTION_RULES = (PAPER_SELECTION_RULE, LEGACY_SELECTION_RULE)
SUPPORTED_TIE_BREAKERS = ("accuracy", "coverage", "lower_threshold")
SUPPORTED_CONFIDENCE_SCORES = ("entropy_confidence", "max_probability")
LEGACY_CONFIDENCE_SCORE = "entropy_confidence"
SUPPORTED_PROBABILITY_SOURCES = ("raw", "calibrated")
LEGACY_PROBABILITY_SOURCE = "raw"


@dataclass(frozen=True)
class ThresholdSelectionConfig:
    candidate_grid: tuple[float, ...]
    selection_rule: str
    minimum_coverage: float
    tie_breakers: tuple[str, ...]
    calibration_bins: int
    confidence_score: str
    probability_source: str
    target_accuracy: float | None
    low_threshold: float
    sha256: str
    source_path: Path = field(compare=False, repr=False)


@dataclass(frozen=True)
class FrozenThresholdArtifact:
    candidate_grid: tuple[float, ...]
    selection_rule: str
    minimum_coverage: float
    tie_breakers: tuple[str, ...]
    selected_threshold: float
    validation_accuracy: float
    validation_macro_f1: float
    validation_coverage: float
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
    target_accuracy: float | None = None
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


def load_threshold_selection_config(path: str | Path) -> ThresholdSelectionConfig:
    """Load and strictly validate a predeclared candidate grid and rule."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing threshold config: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid threshold config JSON: {path}") from error
    payload = _require_object(payload, "Threshold config")
    if int(payload.get("schema_version", -1)) != THRESHOLD_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported threshold config schema_version.")

    raw_grid = payload.get("candidate_grid")
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError("candidate_grid must be a non-empty JSON list.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_grid):
        raise ValueError("candidate_grid values must be numeric.")
    grid = tuple(float(value) for value in raw_grid)
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("candidate_grid values must be finite and lie in [0, 1].")
    if tuple(sorted(set(grid))) != grid:
        raise ValueError("candidate_grid must be strictly increasing with no duplicates.")

    rule_payload = _require_object(payload.get("selection_rule"), "selection_rule")
    rule_name = str(rule_payload.get("name", ""))
    if rule_name not in SUPPORTED_SELECTION_RULES:
        raise ValueError(f"Unsupported threshold selection rule: {rule_name!r}.")
    minimum_coverage = float(rule_payload.get("minimum_coverage", 0.0))
    if not np.isfinite(minimum_coverage) or not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("selection_rule.minimum_coverage must lie in [0, 1].")
    raw_tie_breakers = rule_payload.get("tie_breakers")
    if not isinstance(raw_tie_breakers, list):
        raise ValueError("selection_rule.tie_breakers must be a JSON list.")
    tie_breakers = tuple(str(value) for value in raw_tie_breakers)
    if rule_name == LEGACY_SELECTION_RULE and tie_breakers != SUPPORTED_TIE_BREAKERS:
        raise ValueError(
            "selection_rule.tie_breakers must be exactly "
            f"{list(SUPPORTED_TIE_BREAKERS)}."
        )
    if rule_name == PAPER_SELECTION_RULE and tie_breakers not in {(), ("lower_threshold",)}:
        raise ValueError("Paper threshold rule only permits the lower_threshold tie breaker.")
    target_accuracy_raw = rule_payload.get("target_accuracy")
    target_accuracy = (
        float(target_accuracy_raw) if target_accuracy_raw is not None else None
    )
    if rule_name == PAPER_SELECTION_RULE:
        if target_accuracy is None or not 0.0 < target_accuracy <= 1.0:
            raise ValueError("Paper threshold rule requires target_accuracy in (0, 1].")
        if confidence_score := payload.get("confidence_score"):
            if str(confidence_score) != "entropy_confidence":
                raise ValueError("Paper threshold rule only supports entropy_confidence.")
    low_threshold = float(payload.get("low_threshold", 0.5))
    if not 0.0 <= low_threshold <= 1.0:
        raise ValueError("low_threshold must lie in [0, 1].")
    # Retained only to load legacy frozen artifacts; current reports do not
    # calculate calibration metrics.
    calibration_bins = int(payload.get("calibration_bins", 15))
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive.")
    # Schema-1 configs predate the explicit field. Their historical behavior
    # was entropy-derived confidence, so omission remains a compatibility path.
    confidence_score = str(payload.get("confidence_score", LEGACY_CONFIDENCE_SCORE))
    if confidence_score not in SUPPORTED_CONFIDENCE_SCORES:
        raise ValueError(
            "confidence_score must be one of "
            f"{list(SUPPORTED_CONFIDENCE_SCORES)}, got {confidence_score!r}."
        )
    probability_source = str(
        payload.get("probability_source", LEGACY_PROBABILITY_SOURCE)
    )
    if probability_source not in SUPPORTED_PROBABILITY_SOURCES:
        raise ValueError(
            "probability_source must be one of "
            f"{list(SUPPORTED_PROBABILITY_SOURCES)}, got {probability_source!r}."
        )

    return ThresholdSelectionConfig(
        candidate_grid=grid,
        selection_rule=rule_name,
        minimum_coverage=minimum_coverage,
        tie_breakers=tie_breakers,
        calibration_bins=calibration_bins,
        confidence_score=confidence_score,
        probability_source=probability_source,
        target_accuracy=target_accuracy,
        low_threshold=low_threshold,
        sha256=_sha256_file(path),
        source_path=path.resolve(),
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _confidence_values(
    ensemble: OfflineEnsembleArtifact,
    confidence_score: str,
) -> np.ndarray:
    if confidence_score not in SUPPORTED_CONFIDENCE_SCORES:
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
    else:
        metrics = compute_classification_metrics(
            ensemble.labels[mask],
            ensemble.predictions[mask],
            labels=ensemble.raw_class_ids.astype(int).tolist(),
        )
        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
    return {
        "confidence_score": confidence_score,
        "threshold": float(threshold),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "coverage": coverage,
        "selected_count": count,
        "total_count": total,
    }


def select_confidence_threshold(
    validation_ensemble: OfflineEnsembleArtifact,
    config: ThresholdSelectionConfig,
    *,
    source_ensemble_path: str | Path,
) -> FrozenThresholdArtifact:
    """Select from the mode-appropriate OOF/validation source."""

    expected_source = (
        "oof"
        if validation_ensemble.context.ensemble_mode == "stratified_3fold"
        else "validation"
    )
    if validation_ensemble.context.split.casefold() != expected_source:
        raise ValueError(
            "Threshold selection is validation-only/OOF-only; "
            f"{validation_ensemble.context.ensemble_mode} mode requires "
            f"{expected_source!r}; got {validation_ensemble.context.split!r}."
        )
    if config.probability_source != "raw":
        raise ValueError(
            "This threshold entrypoint received a raw offline ensemble but config "
            "probability_source is calibrated; provide a calibration-aware probability "
            "artifact rather than silently thresholding raw probabilities."
        )
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
    eligible = [candidate for candidate in candidates if candidate["selected_count"] > 0]
    if config.selection_rule == PAPER_SELECTION_RULE:
        eligible = [
            candidate
            for candidate in eligible
            if float(candidate["accuracy"]) >= float(config.target_accuracy)
        ]
    else:
        eligible = [
            candidate
            for candidate in eligible
            if candidate["coverage"] >= config.minimum_coverage
        ]
    if not eligible:
        raise ValueError(
            "No candidate threshold satisfies the configured selection target."
        )
    if config.selection_rule == PAPER_SELECTION_RULE:
        selected = min(eligible, key=lambda candidate: float(candidate["threshold"]))
    else:
        selected = max(
            eligible,
            key=lambda candidate: (
                float(candidate["macro_f1"]),
                float(candidate["accuracy"]),
                float(candidate["coverage"]),
                -float(candidate["threshold"]),
            ),
        )
    return FrozenThresholdArtifact(
        candidate_grid=config.candidate_grid,
        selection_rule=config.selection_rule,
        minimum_coverage=config.minimum_coverage,
        tie_breakers=config.tie_breakers,
        selected_threshold=float(selected["threshold"]),
        validation_accuracy=float(selected["accuracy"]),
        validation_macro_f1=float(selected["macro_f1"]),
        validation_coverage=float(selected["coverage"]),
        validation_selected_count=int(selected["selected_count"]),
        validation_total_count=int(selected["total_count"]),
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
        target_accuracy=config.target_accuracy,
        low_threshold=config.low_threshold,
    )


def _frozen_threshold_to_dict(artifact: FrozenThresholdArtifact) -> dict[str, Any]:
    return {
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
            "target_accuracy": artifact.target_accuracy,
        },
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
        },
        "calibration_bins": artifact.calibration_bins,
        "candidate_results": [dict(candidate) for candidate in artifact.candidate_results],
    }


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


def load_frozen_threshold_artifact(
    path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> FrozenThresholdArtifact:
    """Load a frozen threshold; optionally verify it against the current config."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen threshold artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid frozen threshold JSON: {path}") from error
    payload = _require_object(payload, "Frozen threshold artifact")
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in {
        *LEGACY_FROZEN_THRESHOLD_SCHEMA_VERSIONS,
        FROZEN_THRESHOLD_SCHEMA_VERSION,
    }:
        raise ValueError("Unsupported frozen threshold schema_version.")
    if schema_version >= 2 and "confidence_score" not in payload:
        raise ValueError("Frozen threshold is missing confidence_score.")
    if schema_version == FROZEN_THRESHOLD_SCHEMA_VERSION and "probability_source" not in payload:
        raise ValueError("Schema-3 frozen threshold is missing probability_source.")
    if payload.get("artifact_kind") != FROZEN_THRESHOLD_KIND or payload.get("frozen") is not True:
        raise ValueError("File is not a frozen DDI-CIL confidence threshold artifact.")
    source = _require_object(payload.get("source"), "source")
    if source.get("split") not in {"validation", "oof"}:
        raise ValueError("Frozen threshold source split must be validation or OOF, never test.")
    rule = _require_object(payload.get("selection_rule"), "selection_rule")
    metrics = _require_object(payload.get("validation_metrics"), "validation_metrics")
    candidates = payload.get("candidate_results")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Frozen threshold candidate_results must be non-empty.")

    artifact = FrozenThresholdArtifact(
        candidate_grid=tuple(float(value) for value in payload.get("candidate_grid", [])),
        selection_rule=str(rule.get("name", "")),
        minimum_coverage=float(rule.get("minimum_coverage", -1.0)),
        tie_breakers=tuple(str(value) for value in rule.get("tie_breakers", [])),
        selected_threshold=float(payload.get("selected_threshold", -1.0)),
        validation_accuracy=float(metrics.get("accuracy", -1.0)),
        validation_macro_f1=float(metrics.get("macro_f1", -1.0)),
        validation_coverage=float(metrics.get("coverage", -1.0)),
        validation_selected_count=int(metrics.get("selected_count", -1)),
        validation_total_count=int(metrics.get("total_count", -1)),
        timestamp_utc=str(payload.get("timestamp_utc", "")),
        config_sha256=str(payload.get("config_sha256", "")),
        source_split=str(source.get("split", "")),
        source_ensemble_sha256=str(source.get("ensemble_sha256", "")),
        method=str(source.get("method", "")),
        method_protocol=str(source.get("method_protocol", "")),
        task_id=int(source.get("task_id", -1)),
        experiment_seed=int(source.get("experiment_seed", -1)),
        raw_class_ids=tuple(int(value) for value in source.get("raw_class_ids", [])),
        calibration_bins=int(payload.get("calibration_bins", 0)),
        confidence_score=str(
            payload.get("confidence_score", LEGACY_CONFIDENCE_SCORE)
        ),
        probability_source=str(
            payload.get("probability_source", LEGACY_PROBABILITY_SOURCE)
        ),
        candidate_results=tuple(_require_object(value, "candidate result") for value in candidates),
        target_accuracy=(
            None
            if rule.get("target_accuracy") is None
            else float(rule.get("target_accuracy"))
        ),
        low_threshold=float(payload.get("low_threshold", 0.5)),
        loaded_from_path=path.resolve(),
    )
    _validate_frozen_threshold(artifact)
    if config_path is not None:
        config = load_threshold_selection_config(config_path)
        if artifact.config_sha256 != config.sha256:
            raise ValueError("Frozen threshold config hash does not match the supplied config.")
        if (
            artifact.candidate_grid != config.candidate_grid
            or artifact.selection_rule != config.selection_rule
            or artifact.minimum_coverage != config.minimum_coverage
            or artifact.tie_breakers != config.tie_breakers
            or artifact.calibration_bins != config.calibration_bins
            or artifact.confidence_score != config.confidence_score
            or artifact.probability_source != config.probability_source
            or artifact.target_accuracy != config.target_accuracy
            or artifact.low_threshold != config.low_threshold
        ):
            raise ValueError("Frozen threshold metadata does not match the supplied config.")
    return artifact


def _validate_frozen_threshold(artifact: FrozenThresholdArtifact) -> None:
    if not artifact.candidate_grid or tuple(sorted(set(artifact.candidate_grid))) != artifact.candidate_grid:
        raise ValueError("Frozen threshold grid must be strictly increasing and non-empty.")
    if any(value < 0.0 or value > 1.0 for value in artifact.candidate_grid):
        raise ValueError("Frozen threshold grid values must lie in [0, 1].")
    if artifact.selection_rule not in SUPPORTED_SELECTION_RULES:
        raise ValueError("Unsupported frozen threshold selection rule.")
    if (
        artifact.selection_rule == LEGACY_SELECTION_RULE
        and artifact.tie_breakers != SUPPORTED_TIE_BREAKERS
    ):
        raise ValueError("Unsupported frozen threshold tie breakers.")
    if (
        artifact.selection_rule == PAPER_SELECTION_RULE
        and artifact.tie_breakers not in {(), ("lower_threshold",)}
    ):
        raise ValueError("Unsupported paper-rule tie breakers.")
    if artifact.confidence_score not in SUPPORTED_CONFIDENCE_SCORES:
        raise ValueError("Unsupported frozen threshold confidence_score.")
    if artifact.probability_source not in SUPPORTED_PROBABILITY_SOURCES:
        raise ValueError("Unsupported frozen threshold probability_source.")
    if artifact.selected_threshold not in artifact.candidate_grid:
        raise ValueError("Selected threshold is absent from the frozen candidate grid.")
    if not 0.0 <= artifact.minimum_coverage <= 1.0:
        raise ValueError("Frozen minimum coverage must lie in [0, 1].")
    for name, value in (
        ("validation_accuracy", artifact.validation_accuracy),
        ("validation_macro_f1", artifact.validation_macro_f1),
        ("validation_coverage", artifact.validation_coverage),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Frozen {name} must lie in [0, 1].")
    if (
        artifact.validation_total_count <= 0
        or artifact.validation_selected_count <= 0
        or artifact.validation_selected_count > artifact.validation_total_count
    ):
        raise ValueError("Frozen validation counts are invalid.")
    if not artifact.timestamp_utc or len(artifact.config_sha256) != 64:
        raise ValueError("Frozen threshold timestamp/config hash is invalid.")
    if artifact.source_split not in {"validation", "oof"}:
        raise ValueError("Frozen threshold source split must be validation or OOF.")
    if not 0.0 <= artifact.low_threshold <= 1.0:
        raise ValueError("Frozen low_threshold must lie in [0, 1].")
    if artifact.low_threshold > artifact.selected_threshold:
        raise ValueError("Frozen low_threshold cannot exceed selected_threshold.")
    if artifact.selection_rule == PAPER_SELECTION_RULE:
        if artifact.target_accuracy is None or not 0.0 < artifact.target_accuracy <= 1.0:
            raise ValueError("Frozen paper threshold requires target_accuracy in (0, 1].")
        if artifact.confidence_score != "entropy_confidence":
            raise ValueError("Frozen paper threshold requires entropy_confidence.")
    if not artifact.source_ensemble_sha256 or len(artifact.source_ensemble_sha256) != 64:
        raise ValueError("Frozen source ensemble hash is invalid.")
    if not artifact.method or not artifact.method_protocol:
        raise ValueError("Frozen threshold method provenance is missing.")
    if artifact.task_id < 0 or artifact.experiment_seed < 0 or not artifact.raw_class_ids:
        raise ValueError("Frozen threshold task/seed/class provenance is invalid.")
    if artifact.calibration_bins <= 0:
        raise ValueError("Frozen calibration_bins must be positive.")
    if len(artifact.candidate_results) != len(artifact.candidate_grid):
        raise ValueError("Frozen threshold must report the full candidate grid.")
    result_thresholds = tuple(
        float(candidate.get("threshold", -1.0))
        for candidate in artifact.candidate_results
    )
    if result_thresholds != artifact.candidate_grid:
        raise ValueError("Frozen candidate_results do not match candidate_grid order.")
    for candidate in artifact.candidate_results:
        candidate_score = str(
            candidate.get("confidence_score", LEGACY_CONFIDENCE_SCORE)
        )
        if candidate_score != artifact.confidence_score:
            raise ValueError("Frozen candidate result confidence_score mismatch.")


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
    if mismatches:
        raise ValueError(
            "Ensemble does not match frozen threshold provenance: "
            + ", ".join(mismatches)
        )


def evaluate_with_frozen_threshold(
    ensemble: OfflineEnsembleArtifact,
    threshold: FrozenThresholdArtifact,
) -> dict[str, Any]:
    """Report full and selected-subset metrics using a disk-loaded threshold."""

    if ensemble.context.split.casefold() == "test" and threshold.loaded_from_path is None:
        raise ValueError(
            "Test evaluation requires a frozen threshold loaded from its artifact file."
        )
    expected_source = (
        "oof" if ensemble.context.ensemble_mode == "stratified_3fold" else "validation"
    )
    if threshold.source_split != expected_source:
        raise ValueError(
            f"{ensemble.context.ensemble_mode} evaluation requires a threshold from "
            f"{expected_source}, got {threshold.source_split}."
        )
    if threshold.probability_source != "raw":
        raise ValueError(
            "Frozen threshold expects calibrated probabilities; refusing to apply it "
            "to the raw offline ensemble."
        )
    _validate_threshold_context(ensemble, threshold)

    class_ids = ensemble.raw_class_ids.astype(int).tolist()
    full_classification = compute_classification_metrics(
        ensemble.labels,
        ensemble.predictions,
        labels=class_ids,
    )
    selected = _selected_metrics(
        ensemble,
        threshold.selected_threshold,
        confidence_score=threshold.confidence_score,
    )
    entropy_selected = _selected_metrics(
        ensemble,
        threshold.selected_threshold,
        confidence_score="entropy_confidence",
    )
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

    high_mask = confidence >= threshold.selected_threshold
    medium_mask = (confidence >= threshold.low_threshold) & ~high_mask
    low_mask = confidence < threshold.low_threshold
    selected_metrics = {
        "selected_count": selected["selected_count"],
        "total_count": selected["total_count"],
        "accuracy": selected["accuracy"],
        "macro_f1": selected["macro_f1"],
        "coverage": selected["coverage"],
    }
    entropy_selective_metrics = {
        "threshold_score_name": "entropy_confidence",
        "probability_source": threshold.probability_source,
        "threshold_value": threshold.selected_threshold,
        "selected_count": entropy_selected["selected_count"],
        "total_count": entropy_selected["total_count"],
        "accuracy": entropy_selected["accuracy"],
        "macro_f1": entropy_selected["macro_f1"],
        "coverage": entropy_selected["coverage"],
    }
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
        "threshold": {
            "score_name": threshold.confidence_score,
            "probability_source": threshold.probability_source,
            "value": threshold.selected_threshold,
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
        },
        "entropy_confidence_selective_metrics": entropy_selective_metrics,
        "confidence_tiers": {
            "low_threshold": threshold.low_threshold,
            "high_threshold": threshold.selected_threshold,
            "high": tier_metrics(high_mask),
            "medium": tier_metrics(medium_mask),
            "low": tier_metrics(low_mask),
        },
        # Compatibility alias. Its score semantics are stated explicitly by
        # threshold_score_name above and must not be described as probability.
        "high_confidence": selected_metrics,
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
    print(
        f"threshold_score={frozen.confidence_score} "
        f"threshold_value={frozen.selected_threshold:.6g} "
        f"threshold_artifact={threshold_path} "
        f"split={ensemble.context.split} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
