#!/usr/bin/env python3
"""Validation-fitted scalar temperature calibration for offline ensembles."""

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

from src.eval.calibration_metrics import (  # noqa: E402
    expected_calibration_error,
    fit_temperature,
    map_raw_labels_to_indices,
    multiclass_brier_score,
    negative_log_likelihood,
    softmax_probabilities,
)
from src.eval.offline_ensemble import (  # noqa: E402
    OfflineEnsembleArtifact,
    load_offline_ensemble_artifact,
)


TEMPERATURE_CONFIG_SCHEMA_VERSION = 1
FROZEN_TEMPERATURE_SCHEMA_VERSION = 1
FROZEN_TEMPERATURE_KIND = "ddi_cil_frozen_offline_ensemble_temperature"
CALIBRATION_REPORT_SCHEMA_VERSION = 1
CALIBRATION_REPORT_KIND = "ddi_cil_offline_ensemble_calibration_report"
CALIBRATED_PROBABILITY_SCHEMA_VERSION = 1
CALIBRATED_PROBABILITY_KIND = "ddi_cil_calibrated_ensemble_probabilities"


@dataclass(frozen=True)
class TemperatureCalibrationConfig:
    min_temperature: float
    max_temperature: float
    max_iterations: int
    tolerance: float
    batch_size: int
    calibration_bins: int
    probability_epsilon: float
    sha256: str
    source_path: Path = field(compare=False, repr=False)


@dataclass(frozen=True)
class FrozenTemperatureArtifact:
    temperature: float
    timestamp_utc: str
    config_sha256: str
    source_split: str
    source_ensemble_sha256: str
    method: str
    method_protocol: str
    task_id: int
    experiment_seed: int
    member_ids: tuple[int, ...]
    raw_class_ids: tuple[int, ...]
    validation_sample_count: int
    optimization_diagnostics: Mapping[str, Any]
    loaded_from_path: Path | None = field(default=None, compare=False, repr=False)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def load_temperature_calibration_config(
    path: str | Path,
) -> TemperatureCalibrationConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing temperature calibration config: {path}")
    try:
        payload = _require_object(
            json.loads(path.read_text(encoding="utf-8")),
            "Temperature calibration config",
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid temperature calibration config JSON: {path}") from error
    if int(payload.get("schema_version", -1)) != TEMPERATURE_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported temperature calibration config schema_version.")
    optimizer = _require_object(payload.get("optimizer"), "optimizer")
    min_temperature = float(optimizer.get("min_temperature", -1.0))
    max_temperature = float(optimizer.get("max_temperature", -1.0))
    max_iterations = int(optimizer.get("max_iterations", 0))
    tolerance = float(optimizer.get("tolerance", 0.0))
    batch_size = int(optimizer.get("batch_size", 0))
    calibration_bins = int(payload.get("calibration_bins", 0))
    probability_epsilon = float(payload.get("probability_epsilon", 0.0))
    if not 0.0 < min_temperature < max_temperature:
        raise ValueError("Temperature bounds must satisfy 0 < min < max.")
    if max_iterations <= 0 or tolerance <= 0.0 or batch_size <= 0:
        raise ValueError("Temperature optimizer controls must be positive.")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive.")
    if not 0.0 < probability_epsilon < 1.0:
        raise ValueError("probability_epsilon must lie strictly between zero and one.")
    return TemperatureCalibrationConfig(
        min_temperature=min_temperature,
        max_temperature=max_temperature,
        max_iterations=max_iterations,
        tolerance=tolerance,
        batch_size=batch_size,
        calibration_bins=calibration_bins,
        probability_epsilon=probability_epsilon,
        sha256=_sha256_file(path),
        source_path=path.resolve(),
    )


def _mean_probability_logits(
    probabilities: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("Mean ensemble probabilities must be a finite matrix.")
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("Mean ensemble probability rows must be valid distributions.")
    # These are log mean probabilities, not member logits and not mean logits.
    return np.log(np.clip(probabilities, epsilon, 1.0))


def calibrate_mean_probabilities(
    probabilities: np.ndarray,
    temperature: float,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    logits = _mean_probability_logits(probabilities, epsilon=epsilon)
    return softmax_probabilities(logits, temperature=temperature)


def _metrics(
    probabilities: np.ndarray,
    dense_labels: np.ndarray,
    *,
    calibration_bins: int,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    return {
        "sample_count": int(dense_labels.shape[0]),
        "accuracy": float(np.mean(predictions == dense_labels)),
        "max_probability_ece": expected_calibration_error(
            probabilities,
            dense_labels,
            num_bins=calibration_bins,
        ),
        "negative_log_likelihood": negative_log_likelihood(
            probabilities,
            dense_labels,
        ),
        "brier_score": multiclass_brier_score(probabilities, dense_labels),
    }


def fit_offline_temperature(
    validation_ensemble: OfflineEnsembleArtifact,
    config: TemperatureCalibrationConfig,
    *,
    source_ensemble_path: str | Path,
) -> FrozenTemperatureArtifact:
    """Fit one scalar on validation NLL from log(mean ensemble probability)."""

    if validation_ensemble.context.split.casefold() != "validation":
        raise ValueError(
            "Temperature fitting is validation-only; refusing source split "
            f"{validation_ensemble.context.split!r}."
        )
    source_path = Path(source_ensemble_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source ensemble artifact: {source_path}")
    probabilities = np.asarray(validation_ensemble.probabilities, dtype=np.float64)
    logits = _mean_probability_logits(
        probabilities,
        epsilon=config.probability_epsilon,
    )
    dense_labels = map_raw_labels_to_indices(
        validation_ensemble.labels,
        validation_ensemble.raw_class_ids,
    )
    raw_nll = negative_log_likelihood(probabilities, dense_labels)
    fit = fit_temperature(
        logits,
        dense_labels,
        min_temperature=config.min_temperature,
        max_temperature=config.max_temperature,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
        batch_size=config.batch_size,
    )
    if not fit.converged:
        raise RuntimeError("Temperature optimization did not converge.")
    calibrated = calibrate_mean_probabilities(
        probabilities,
        fit.temperature,
        epsilon=config.probability_epsilon,
    )
    calibrated_nll = negative_log_likelihood(calibrated, dense_labels)
    if calibrated_nll > raw_nll + 1e-10:
        raise RuntimeError("Optimized temperature increased validation NLL.")
    raw_argmax = probabilities.argmax(axis=1)
    calibrated_argmax = calibrated.argmax(axis=1)
    if not np.array_equal(raw_argmax, calibrated_argmax):
        raise RuntimeError("Positive scalar temperature changed validation argmax.")
    diagnostics = {
        "optimizer": "safeguarded_newton_inverse_temperature",
        "objective": "validation_negative_log_likelihood",
        "input": "log_mean_ensemble_probability",
        "member_logits_averaged": False,
        "initial_temperature": 1.0,
        "initial_nll": raw_nll,
        "optimized_nll": calibrated_nll,
        "nll_improvement": raw_nll - calibrated_nll,
        "inverse_temperature": fit.inverse_temperature,
        "iterations": fit.iterations,
        "converged": fit.converged,
        "boundary": fit.boundary,
        "temperature_bounds": [config.min_temperature, config.max_temperature],
        "max_iterations": config.max_iterations,
        "tolerance": config.tolerance,
        "batch_size": config.batch_size,
        "probability_epsilon": config.probability_epsilon,
        "argmax_invariant": True,
    }
    return FrozenTemperatureArtifact(
        temperature=fit.temperature,
        timestamp_utc=_utc_timestamp(),
        config_sha256=config.sha256,
        source_split="validation",
        source_ensemble_sha256=_sha256_file(source_path),
        method=validation_ensemble.context.method,
        method_protocol=validation_ensemble.context.method_protocol,
        task_id=validation_ensemble.context.task_id,
        experiment_seed=validation_ensemble.context.experiment_seed,
        member_ids=tuple(validation_ensemble.context.member_ids),
        raw_class_ids=tuple(int(value) for value in validation_ensemble.raw_class_ids),
        validation_sample_count=validation_ensemble.row_count,
        optimization_diagnostics=diagnostics,
    )


def _frozen_to_dict(artifact: FrozenTemperatureArtifact) -> dict[str, Any]:
    return {
        "schema_version": FROZEN_TEMPERATURE_SCHEMA_VERSION,
        "artifact_kind": FROZEN_TEMPERATURE_KIND,
        "frozen": True,
        "temperature": artifact.temperature,
        "timestamp_utc": artifact.timestamp_utc,
        "config_sha256": artifact.config_sha256,
        "source": {
            "split": artifact.source_split,
            "ensemble_sha256": artifact.source_ensemble_sha256,
            "method": artifact.method,
            "method_protocol": artifact.method_protocol,
            "task_id": artifact.task_id,
            "experiment_seed": artifact.experiment_seed,
            "member_ids": list(artifact.member_ids),
            "raw_class_ids": list(artifact.raw_class_ids),
            "sample_count": artifact.validation_sample_count,
        },
        "optimization_diagnostics": dict(artifact.optimization_diagnostics),
    }


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Calibration artifact already exists: {path}")
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
    return path


def export_frozen_temperature(
    artifact: FrozenTemperatureArtifact,
    path: str | Path,
) -> Path:
    if artifact.source_split != "validation":
        raise ValueError("Frozen temperature must originate from validation.")
    _validate_frozen_temperature(artifact)
    return _atomic_write_json(_frozen_to_dict(artifact), Path(path))


def _validate_frozen_temperature(artifact: FrozenTemperatureArtifact) -> None:
    if not np.isfinite(artifact.temperature) or artifact.temperature <= 0.0:
        raise ValueError("Frozen temperature must be finite and positive.")
    if artifact.source_split != "validation":
        raise ValueError("Frozen temperature source split must be validation, never test.")
    if len(artifact.config_sha256) != 64 or len(artifact.source_ensemble_sha256) != 64:
        raise ValueError("Frozen temperature source/config hash is invalid.")
    if not artifact.timestamp_utc or not artifact.method or not artifact.method_protocol:
        raise ValueError("Frozen temperature provenance is incomplete.")
    if artifact.task_id < 0 or artifact.experiment_seed < 0:
        raise ValueError("Frozen temperature task/seed metadata is invalid.")
    if not artifact.member_ids or len(set(artifact.member_ids)) != len(artifact.member_ids):
        raise ValueError("Frozen temperature member IDs are invalid.")
    if not artifact.raw_class_ids or len(set(artifact.raw_class_ids)) != len(
        artifact.raw_class_ids
    ):
        raise ValueError("Frozen temperature class order is invalid.")
    if artifact.validation_sample_count <= 0:
        raise ValueError("Frozen temperature validation sample count must be positive.")
    diagnostics = artifact.optimization_diagnostics
    if diagnostics.get("objective") != "validation_negative_log_likelihood":
        raise ValueError("Frozen temperature objective must be validation NLL.")
    if diagnostics.get("member_logits_averaged") is not False:
        raise ValueError("Frozen temperature must not average member logits.")
    if diagnostics.get("argmax_invariant") is not True:
        raise ValueError("Frozen temperature must record argmax invariance.")
    if diagnostics.get("converged") is not True:
        raise ValueError("Frozen temperature optimizer must have converged.")


def load_frozen_temperature(
    path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> FrozenTemperatureArtifact:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen temperature artifact: {path}")
    try:
        payload = _require_object(
            json.loads(path.read_text(encoding="utf-8")),
            "Frozen temperature artifact",
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid frozen temperature JSON: {path}") from error
    if int(payload.get("schema_version", -1)) != FROZEN_TEMPERATURE_SCHEMA_VERSION:
        raise ValueError("Unsupported frozen temperature schema_version.")
    if payload.get("artifact_kind") != FROZEN_TEMPERATURE_KIND or payload.get("frozen") is not True:
        raise ValueError("File is not a frozen offline ensemble temperature artifact.")
    source = _require_object(payload.get("source"), "source")
    diagnostics = _require_object(
        payload.get("optimization_diagnostics"),
        "optimization_diagnostics",
    )
    artifact = FrozenTemperatureArtifact(
        temperature=float(payload.get("temperature", -1.0)),
        timestamp_utc=str(payload.get("timestamp_utc", "")),
        config_sha256=str(payload.get("config_sha256", "")),
        source_split=str(source.get("split", "")),
        source_ensemble_sha256=str(source.get("ensemble_sha256", "")),
        method=str(source.get("method", "")),
        method_protocol=str(source.get("method_protocol", "")),
        task_id=int(source.get("task_id", -1)),
        experiment_seed=int(source.get("experiment_seed", -1)),
        member_ids=tuple(int(value) for value in source.get("member_ids", [])),
        raw_class_ids=tuple(int(value) for value in source.get("raw_class_ids", [])),
        validation_sample_count=int(source.get("sample_count", 0)),
        optimization_diagnostics=diagnostics,
        loaded_from_path=path.resolve(),
    )
    _validate_frozen_temperature(artifact)
    if config_path is not None:
        config = load_temperature_calibration_config(config_path)
        if artifact.config_sha256 != config.sha256:
            raise ValueError("Frozen temperature config hash does not match supplied config.")
    return artifact


def _validate_application_context(
    ensemble: OfflineEnsembleArtifact,
    artifact: FrozenTemperatureArtifact,
) -> None:
    expected = {
        "method": artifact.method,
        "method_protocol": artifact.method_protocol,
        "task_id": artifact.task_id,
        "experiment_seed": artifact.experiment_seed,
        "member_ids": artifact.member_ids,
        "raw_class_ids": artifact.raw_class_ids,
    }
    actual = {
        "method": ensemble.context.method,
        "method_protocol": ensemble.context.method_protocol,
        "task_id": ensemble.context.task_id,
        "experiment_seed": ensemble.context.experiment_seed,
        "member_ids": tuple(ensemble.context.member_ids),
        "raw_class_ids": tuple(int(value) for value in ensemble.raw_class_ids),
    }
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        raise ValueError(
            "Ensemble does not match frozen temperature metadata/class order: "
            + ", ".join(mismatches)
        )


def evaluate_with_frozen_temperature(
    ensemble: OfflineEnsembleArtifact,
    artifact: FrozenTemperatureArtifact,
    config: TemperatureCalibrationConfig,
    *,
    source_ensemble_path: str | Path,
) -> tuple[dict[str, Any], np.ndarray]:
    """Apply a disk-loaded validation temperature and report raw/calibrated metrics."""

    if ensemble.context.split.casefold() == "test" and artifact.loaded_from_path is None:
        raise ValueError("Test calibration requires a frozen temperature loaded from disk.")
    if artifact.source_split != "validation":
        raise ValueError("Temperature must have been fitted on validation.")
    if artifact.config_sha256 != config.sha256:
        raise ValueError("Frozen temperature config hash does not match supplied config.")
    _validate_application_context(ensemble, artifact)
    source_path = Path(source_ensemble_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing evaluation ensemble artifact: {source_path}")
    if (
        ensemble.context.split.casefold() == "validation"
        and _sha256_file(source_path) != artifact.source_ensemble_sha256
    ):
        raise ValueError("Validation ensemble hash does not match temperature fit source.")

    raw = np.asarray(ensemble.probabilities, dtype=np.float64)
    calibrated = calibrate_mean_probabilities(
        raw,
        artifact.temperature,
        epsilon=config.probability_epsilon,
    )
    if not np.array_equal(raw.argmax(axis=1), calibrated.argmax(axis=1)):
        raise RuntimeError("Positive scalar temperature changed prediction argmax.")
    dense_labels = map_raw_labels_to_indices(ensemble.labels, ensemble.raw_class_ids)
    raw_metrics = _metrics(raw, dense_labels, calibration_bins=config.calibration_bins)
    calibrated_metrics = _metrics(
        calibrated,
        dense_labels,
        calibration_bins=config.calibration_bins,
    )
    if raw_metrics["accuracy"] != calibrated_metrics["accuracy"]:
        raise RuntimeError("Temperature calibration changed accuracy.")
    report = {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "artifact_kind": CALIBRATION_REPORT_KIND,
        "timestamp_utc": _utc_timestamp(),
        "evaluation_split": ensemble.context.split,
        "temperature_fit_split": artifact.source_split,
        "temperature": artifact.temperature,
        "config_sha256": config.sha256,
        "source_ensemble_sha256": _sha256_file(source_path),
        "temperature_artifact_sha256": (
            _sha256_file(artifact.loaded_from_path)
            if artifact.loaded_from_path is not None
            else None
        ),
        "method": ensemble.context.method,
        "method_protocol": ensemble.context.method_protocol,
        "task_id": ensemble.context.task_id,
        "experiment_seed": ensemble.context.experiment_seed,
        "member_ids": list(ensemble.context.member_ids),
        "raw_class_ids": ensemble.raw_class_ids.astype(int).tolist(),
        "formula": "softmax(log(mean_probability) / temperature)",
        "member_logits_averaged": False,
        "argmax_invariant": True,
        "accuracy_invariant": True,
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
    }
    return report, calibrated


def export_calibrated_probabilities(
    ensemble: OfflineEnsembleArtifact,
    calibrated_probabilities: np.ndarray,
    artifact: FrozenTemperatureArtifact,
    *,
    source_ensemble_path: str | Path,
    path: str | Path,
) -> Path:
    """Export a separate calibrated view without mutating the raw ensemble."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Calibrated probability artifact already exists: {path}")
    probabilities = np.asarray(calibrated_probabilities, dtype=np.float64)
    if probabilities.shape != ensemble.probabilities.shape:
        raise ValueError("Calibrated probability shape does not match raw ensemble.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("Calibrated probability rows must sum to one.")
    raw_argmax = ensemble.probabilities.argmax(axis=1)
    calibrated_argmax = probabilities.argmax(axis=1)
    if not np.array_equal(raw_argmax, calibrated_argmax):
        raise ValueError("Calibrated probabilities changed argmax.")
    logarithms = np.zeros_like(probabilities)
    np.log(probabilities, out=logarithms, where=probabilities > 0.0)
    predictive_entropy = -np.sum(probabilities * logarithms, axis=1)
    class_count = probabilities.shape[1]
    normalized_entropy = (
        np.zeros_like(predictive_entropy)
        if class_count <= 1
        else np.clip(predictive_entropy / np.log(class_count), 0.0, 1.0)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(
                    CALIBRATED_PROBABILITY_SCHEMA_VERSION,
                    dtype=np.int32,
                ),
                artifact_kind=np.asarray(CALIBRATED_PROBABILITY_KIND),
                source_ensemble_sha256=np.asarray(
                    _sha256_file(Path(source_ensemble_path))
                ),
                temperature_artifact_sha256=np.asarray(
                    _sha256_file(artifact.loaded_from_path)
                    if artifact.loaded_from_path is not None
                    else ""
                ),
                temperature=np.asarray(artifact.temperature, dtype=np.float64),
                split=np.asarray(ensemble.context.split),
                task_id=np.asarray(ensemble.context.task_id, dtype=np.int32),
                experiment_seed=np.asarray(
                    ensemble.context.experiment_seed,
                    dtype=np.int64,
                ),
                member_ids=np.asarray(ensemble.context.member_ids, dtype=np.int32),
                sample_ids=ensemble.sample_ids,
                labels=ensemble.labels,
                raw_class_ids=ensemble.raw_class_ids,
                calibrated_probabilities=probabilities.astype(np.float32),
                calibrated_predictions=ensemble.raw_class_ids[calibrated_argmax],
                calibrated_predictive_entropy=predictive_entropy,
                calibrated_normalized_entropy=normalized_entropy,
                calibrated_entropy_confidence=1.0 - normalized_entropy,
                calibrated_max_probability=probabilities.max(axis=1),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit/apply scalar temperature calibration to offline ensemble probabilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="Fit on validation and freeze temperature.")
    fit.add_argument("--ensemble", required=True, type=Path)
    fit.add_argument("--config", required=True, type=Path)
    fit.add_argument("--temperature-out", required=True, type=Path)
    fit.add_argument("--report-out", required=True, type=Path)
    fit.add_argument("--calibrated-out", type=Path)
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Apply a disk-loaded validation temperature.",
    )
    evaluate.add_argument("--ensemble", required=True, type=Path)
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--temperature-artifact", required=True, type=Path)
    evaluate.add_argument("--report-out", required=True, type=Path)
    evaluate.add_argument("--calibrated-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    ensemble = load_offline_ensemble_artifact(args.ensemble)
    config = load_temperature_calibration_config(args.config)
    if args.command == "fit":
        fitted = fit_offline_temperature(
            ensemble,
            config,
            source_ensemble_path=args.ensemble,
        )
        temperature_path = export_frozen_temperature(fitted, args.temperature_out)
        frozen = load_frozen_temperature(temperature_path, config_path=args.config)
    else:
        temperature_path = args.temperature_artifact
        frozen = load_frozen_temperature(temperature_path, config_path=args.config)
    report, calibrated = evaluate_with_frozen_temperature(
        ensemble,
        frozen,
        config,
        source_ensemble_path=args.ensemble,
    )
    report_path = _atomic_write_json(report, args.report_out)
    if args.calibrated_out is not None:
        export_calibrated_probabilities(
            ensemble,
            calibrated,
            frozen,
            source_ensemble_path=args.ensemble,
            path=args.calibrated_out,
        )
    print(
        f"temperature={frozen.temperature:.8g} fit_split=validation "
        f"evaluation_split={ensemble.context.split} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
