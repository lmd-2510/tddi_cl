"""Versioned per-member prediction artifacts for offline ensemble evaluation."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.eval.cil_evaluation import PredictionOutputs
from src.eval.s02_artifacts import SUPPORTED_SPLITS, build_stable_sample_ids


MEMBER_PREDICTION_SCHEMA_VERSION = 1
MEMBER_PREDICTION_KIND = "ddi_cil_member_predictions"


@dataclass(frozen=True)
class MemberPredictionContext:
    run_id: str
    method: str
    method_protocol: str
    task_id: int
    split: str
    member_id: int
    experiment_seed: int
    member_seed: int


@dataclass(frozen=True)
class MemberPredictionArtifact:
    context: MemberPredictionContext
    sample_ids: np.ndarray
    labels: np.ndarray
    raw_class_ids: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def class_count(self) -> int:
        return int(self.raw_class_ids.shape[0])


def _validate_context(context: MemberPredictionContext) -> None:
    if not context.run_id:
        raise ValueError("Member prediction run_id must be non-empty.")
    if not context.method or not context.method_protocol:
        raise ValueError("Member prediction method provenance must be non-empty.")
    if context.task_id < 0:
        raise ValueError("Member prediction task_id must be non-negative.")
    if context.split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported member prediction split: {context.split}")
    if context.member_id < 0:
        raise ValueError("Member prediction member_id must be non-negative.")
    if context.experiment_seed < 0 or context.member_seed < 0:
        raise ValueError("Member prediction seeds must be non-negative.")


def _validate_arrays(
    *,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    raw_class_ids: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    if sample_ids.ndim != 1 or sample_ids.shape[0] == 0:
        raise ValueError("Member prediction sample_ids must be a non-empty vector.")
    if sample_ids.dtype.kind not in {"U", "S"}:
        raise TypeError("Member prediction sample_ids must be strings.")
    if np.any(sample_ids.astype(np.str_) == ""):
        raise ValueError("Member prediction sample_ids must be non-empty strings.")
    if np.unique(sample_ids).shape[0] != sample_ids.shape[0]:
        raise ValueError("Member prediction artifact contains duplicate sample IDs.")
    if labels.dtype != np.dtype(np.int64) or labels.shape != (sample_ids.shape[0],):
        raise ValueError("Member prediction labels must be int64 and match the row count.")
    if raw_class_ids.dtype != np.dtype(np.int64):
        raise TypeError("Member prediction raw_class_ids must use int64.")
    if raw_class_ids.ndim != 1 or raw_class_ids.shape[0] == 0:
        raise ValueError("Member prediction raw_class_ids metadata must be a non-empty vector.")
    if np.unique(raw_class_ids).shape[0] != raw_class_ids.shape[0]:
        raise ValueError("Member prediction raw_class_ids must be unique.")
    if logits.dtype != np.dtype(np.float32) or probabilities.dtype != np.dtype(np.float32):
        raise TypeError("Member prediction logits and probabilities must use float32.")
    if logits.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("Member prediction logits and probabilities must be two-dimensional.")
    if logits.shape != probabilities.shape:
        raise ValueError("Member prediction logits/probabilities shapes do not match.")
    expected_shape = (sample_ids.shape[0], raw_class_ids.shape[0])
    if probabilities.shape != expected_shape:
        raise ValueError(
            "Member prediction probability width or row count does not match "
            f"metadata: {probabilities.shape} != {expected_shape}."
        )
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("Member prediction logits/probabilities contain non-finite values.")
    if np.any(probabilities < -1e-7) or np.any(probabilities > 1.0 + 1e-7):
        raise ValueError("Member prediction probabilities must lie in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Member prediction probability rows must sum to one.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected_probabilities = np.exp(shifted)
    expected_probabilities /= expected_probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities, expected_probabilities, rtol=1e-5, atol=1e-6):
        raise ValueError("Member probabilities do not match softmax(logits).")
    if not np.isin(labels, raw_class_ids).all():
        raise ValueError("Member prediction labels must belong to raw_class_ids.")


def _artifact_from_arrays(
    context: MemberPredictionContext,
    *,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    raw_class_ids: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> MemberPredictionArtifact:
    _validate_context(context)
    _validate_arrays(
        sample_ids=sample_ids,
        labels=labels,
        raw_class_ids=raw_class_ids,
        logits=logits,
        probabilities=probabilities,
    )
    return MemberPredictionArtifact(
        context=context,
        sample_ids=sample_ids,
        labels=labels,
        raw_class_ids=raw_class_ids,
        logits=logits,
        probabilities=probabilities,
    )


def export_member_prediction_artifact(
    outputs: PredictionOutputs,
    metadata: Mapping[str, np.ndarray],
    context: MemberPredictionContext,
    path: str | Path,
) -> Path:
    """Validate and atomically publish one self-contained member artifact."""

    _validate_context(context)
    logits = np.asarray(outputs.logits)
    probabilities = np.asarray(outputs.probabilities)
    labels = np.asarray(outputs.labels)
    raw_class_ids = np.asarray(outputs.class_ids)
    row_count = int(logits.shape[0]) if logits.ndim >= 1 else 0
    sample_ids = build_stable_sample_ids(metadata, row_count)
    _artifact_from_arrays(
        context,
        sample_ids=sample_ids,
        labels=labels,
        raw_class_ids=raw_class_ids,
        logits=logits,
        probabilities=probabilities,
    )

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Member prediction artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(MEMBER_PREDICTION_SCHEMA_VERSION, dtype=np.int32),
                artifact_kind=np.asarray(MEMBER_PREDICTION_KIND),
                run_id=np.asarray(context.run_id),
                method=np.asarray(context.method),
                method_protocol=np.asarray(context.method_protocol),
                task_id=np.asarray(context.task_id, dtype=np.int32),
                split=np.asarray(context.split),
                member_id=np.asarray(context.member_id, dtype=np.int32),
                experiment_seed=np.asarray(context.experiment_seed, dtype=np.int64),
                member_seed=np.asarray(context.member_seed, dtype=np.int64),
                sample_ids=sample_ids,
                labels=labels,
                raw_class_ids=raw_class_ids,
                logits=logits,
                probabilities=probabilities,
            )
            handle.flush()
            os.fsync(handle.fileno())
        load_member_prediction_artifact(temporary_path, expected_context=context)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _read_scalar(payload: Mapping[str, np.ndarray], key: str) -> object:
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError(f"Member prediction {key} must be a scalar.")
    return value.item()


def load_member_prediction_artifact(
    path: str | Path,
    *,
    expected_context: MemberPredictionContext | None = None,
) -> MemberPredictionArtifact:
    """Load and fully validate one per-member prediction artifact."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing member prediction artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "schema_version",
            "artifact_kind",
            "run_id",
            "method",
            "method_protocol",
            "task_id",
            "split",
            "member_id",
            "experiment_seed",
            "member_seed",
            "sample_ids",
            "labels",
            "raw_class_ids",
            "logits",
            "probabilities",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Member prediction artifact is missing keys: {missing}")
        schema_version = int(_read_scalar(payload, "schema_version"))
        artifact_kind = str(_read_scalar(payload, "artifact_kind"))
        if schema_version != MEMBER_PREDICTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported member prediction schema version: {schema_version}.")
        if artifact_kind != MEMBER_PREDICTION_KIND:
            raise ValueError(f"Unexpected member prediction artifact kind: {artifact_kind!r}.")
        context = MemberPredictionContext(
            run_id=str(_read_scalar(payload, "run_id")),
            method=str(_read_scalar(payload, "method")),
            method_protocol=str(_read_scalar(payload, "method_protocol")),
            task_id=int(_read_scalar(payload, "task_id")),
            split=str(_read_scalar(payload, "split")),
            member_id=int(_read_scalar(payload, "member_id")),
            experiment_seed=int(_read_scalar(payload, "experiment_seed")),
            member_seed=int(_read_scalar(payload, "member_seed")),
        )
        artifact = _artifact_from_arrays(
            context,
            sample_ids=np.asarray(payload["sample_ids"]),
            labels=np.asarray(payload["labels"]),
            raw_class_ids=np.asarray(payload["raw_class_ids"]),
            logits=np.asarray(payload["logits"]),
            probabilities=np.asarray(payload["probabilities"]),
        )

    if expected_context is not None and artifact.context != expected_context:
        raise ValueError("Member prediction provenance does not match the expected context.")
    return artifact
