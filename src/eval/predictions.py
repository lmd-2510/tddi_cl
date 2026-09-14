"""Versioned per-member prediction artifacts for offline ensemble evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np

from src.data.sample_identity import SUPPORTED_PREDICTION_SPLITS, build_stable_sample_ids
from src.eval.evaluation import PredictionOutputs


MEMBER_PREDICTION_SCHEMA_VERSION = 3
LEGACY_MEMBER_PREDICTION_SCHEMA_VERSIONS = frozenset({1, 2})
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
    ensemble_mode: str = "seeded"
    fold_id: int | None = None
    fold_count: int | None = None
    fold_seed: int | None = None


@dataclass(frozen=True)
class PredictionProvenance:
    """Frozen study contract carried by schema-3 predictions.

    The preprocessing hash belongs to one member. It is not expected to be
    equal across the three fold members.
    """

    assignment_sha256: str
    fold_manifest_sha256: str
    task_file_sha256: str
    study_contract_sha256: str
    preprocessing_policy: str
    preprocessing_sha256: str
    preprocessing_member_id: int
    ranking_policy: str
    buffer_policy: str
    member_memory_budget: int
    global_memory_budget: int
    expected_partition_rows: int
    expected_partition_sha256: str
    expected_oof_rows: int | None = None
    expected_oof_sha256: str | None = None


@dataclass(frozen=True)
class MemberPredictionArtifact:
    context: MemberPredictionContext
    sample_ids: np.ndarray
    labels: np.ndarray
    raw_class_ids: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    fold_ids: np.ndarray | None = None
    provenance: PredictionProvenance | None = None

    @property
    def row_count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def class_count(self) -> int:
        return int(self.raw_class_ids.shape[0])


def partition_identity_sha256(
    sample_ids: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray,
) -> str:
    """Hash aligned ID/label/fold triples in canonical sample-ID order."""

    sample_ids = np.asarray(sample_ids).astype(np.str_, copy=False)
    labels = np.asarray(labels, dtype=np.int64)
    fold_ids = np.asarray(fold_ids, dtype=np.int16)
    if sample_ids.ndim != 1 or labels.shape != sample_ids.shape or fold_ids.shape != sample_ids.shape:
        raise ValueError("Partition identity arrays must be aligned one-dimensional vectors.")
    order = np.argsort(sample_ids, kind="stable")
    digest = hashlib.sha256()
    for sample_id, label, fold_id in zip(
        sample_ids[order], labels[order], fold_ids[order], strict=True
    ):
        encoded = str(sample_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(int(label).to_bytes(8, "big", signed=True))
        digest.update(int(fold_id).to_bytes(2, "big", signed=True))
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_context(context: MemberPredictionContext) -> None:
    if not context.run_id or not context.method or not context.method_protocol:
        raise ValueError("Member prediction run/method provenance must be non-empty.")
    if context.task_id < 0 or context.member_id < 0:
        raise ValueError("Member prediction task_id/member_id must be non-negative.")
    if context.split not in SUPPORTED_PREDICTION_SPLITS:
        raise ValueError(f"Unsupported member prediction split: {context.split}")
    if context.experiment_seed < 0 or context.member_seed < 0:
        raise ValueError("Member prediction seeds must be non-negative.")
    if context.ensemble_mode not in {"seeded", "stratified_3fold"}:
        raise ValueError(f"Unsupported ensemble mode: {context.ensemble_mode!r}.")
    if context.ensemble_mode == "stratified_3fold":
        if context.fold_count != 3:
            raise ValueError("stratified_3fold member artifacts require fold_count=3.")
        if context.fold_id is None or not 0 <= context.fold_id < context.fold_count:
            raise ValueError("stratified_3fold member artifacts require fold_id in [0, 2].")
        if context.fold_id != context.member_id:
            raise ValueError("stratified_3fold requires member_id == held-out fold_id.")
        if context.fold_seed is None or context.fold_seed < 0:
            raise ValueError("stratified_3fold requires a non-negative fold_seed.")
    elif any(value is not None for value in (context.fold_id, context.fold_count, context.fold_seed)):
        raise ValueError("seeded member artifacts must not contain fold metadata.")


def _validate_provenance(
    provenance: PredictionProvenance,
    context: MemberPredictionContext,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
) -> None:
    for name in (
        "assignment_sha256", "fold_manifest_sha256", "task_file_sha256",
        "study_contract_sha256", "preprocessing_sha256", "expected_partition_sha256",
    ):
        if not _valid_sha256(getattr(provenance, name)):
            raise ValueError(f"Prediction provenance {name} must be a lowercase SHA256.")
    if not all((provenance.preprocessing_policy, provenance.ranking_policy, provenance.buffer_policy)):
        raise ValueError("Prediction preprocessing/ranking/buffer provenance must be non-empty.")
    if provenance.preprocessing_member_id != context.member_id:
        raise ValueError("Prediction preprocessing artifact belongs to the wrong member.")
    if not 0 < provenance.member_memory_budget <= provenance.global_memory_budget:
        raise ValueError("Prediction memory budgets are invalid.")
    if provenance.expected_partition_rows != sample_ids.shape[0]:
        raise ValueError("Prediction row coverage does not match expected partition rows.")
    if partition_identity_sha256(sample_ids, labels, fold_ids) != provenance.expected_partition_sha256:
        raise ValueError("Prediction IDs/labels/folds do not match assignment provenance.")
    if (provenance.expected_oof_rows is None) != (provenance.expected_oof_sha256 is None):
        raise ValueError("Expected OOF rows and SHA256 must be present together.")
    if provenance.expected_oof_sha256 is not None:
        if not _valid_sha256(provenance.expected_oof_sha256):
            raise ValueError("Prediction expected_oof_sha256 must be a lowercase SHA256.")
        if provenance.expected_oof_rows is None or provenance.expected_oof_rows < sample_ids.shape[0]:
            raise ValueError("Expected OOF coverage cannot be smaller than one held-out fold.")


def _validate_arrays(
    *, sample_ids: np.ndarray, labels: np.ndarray, raw_class_ids: np.ndarray,
    logits: np.ndarray, probabilities: np.ndarray, fold_ids: np.ndarray,
) -> None:
    if sample_ids.ndim != 1 or sample_ids.shape[0] == 0 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("Member prediction sample_ids must be a non-empty string vector.")
    if np.any(sample_ids.astype(np.str_) == "") or np.unique(sample_ids).shape[0] != sample_ids.shape[0]:
        raise ValueError("Member prediction sample IDs must be non-empty and unique.")
    if labels.dtype != np.dtype(np.int64) or labels.shape != sample_ids.shape:
        raise ValueError("Member prediction labels must be int64 and match the row count.")
    if fold_ids.dtype != np.dtype(np.int16) or fold_ids.shape != sample_ids.shape:
        raise ValueError("Member prediction fold_ids must be int16 and match the row count.")
    if raw_class_ids.dtype != np.dtype(np.int64) or raw_class_ids.ndim != 1 or not raw_class_ids.size:
        raise TypeError("Member prediction raw_class_ids must be a non-empty int64 vector.")
    if np.unique(raw_class_ids).shape[0] != raw_class_ids.shape[0]:
        raise ValueError("Member prediction raw_class_ids must be unique.")
    if logits.dtype != np.dtype(np.float32) or probabilities.dtype != np.dtype(np.float32):
        raise TypeError("Member prediction logits and probabilities must use float32.")
    if logits.ndim != 2 or logits.shape != probabilities.shape:
        raise ValueError("Member prediction logits/probabilities shapes do not match.")
    expected_shape = (sample_ids.shape[0], raw_class_ids.shape[0])
    if probabilities.shape != expected_shape:
        raise ValueError(f"Member prediction probability width or row count mismatch: {probabilities.shape} != {expected_shape}.")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("Member prediction logits/probabilities contain non-finite values.")
    if np.any(probabilities < -1e-7) or np.any(probabilities > 1.0 + 1e-7):
        raise ValueError("Member prediction probabilities must lie in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Member prediction probability rows must sum to one.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    expected = np.exp(shifted)
    expected /= expected.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities, expected, rtol=1e-5, atol=1e-6):
        raise ValueError("Member probabilities do not match softmax(logits).")
    if not np.isin(labels, raw_class_ids).all():
        raise ValueError("Member prediction labels must belong to raw_class_ids.")


def _artifact_from_arrays(
    context: MemberPredictionContext, *, sample_ids: np.ndarray, labels: np.ndarray,
    raw_class_ids: np.ndarray, logits: np.ndarray, probabilities: np.ndarray,
    fold_ids: np.ndarray | None = None, provenance: PredictionProvenance | None = None,
) -> MemberPredictionArtifact:
    _validate_context(context)
    if fold_ids is None:
        implied = context.fold_id if context.split == "validation" and context.fold_id is not None else -1
        fold_ids = np.full(sample_ids.shape[0], implied, dtype=np.int16)
    fold_ids = np.asarray(fold_ids)
    _validate_arrays(sample_ids=sample_ids, labels=labels, raw_class_ids=raw_class_ids,
                     logits=logits, probabilities=probabilities, fold_ids=fold_ids)
    if context.ensemble_mode == "stratified_3fold":
        expected_fold = context.fold_id if context.split == "validation" else -1
        if not np.all(fold_ids == expected_fold):
            raise ValueError("Prediction rows belong to the wrong held-out/test partition.")
    elif np.any(fold_ids != -1):
        raise ValueError("Seeded member predictions must use fold_id=-1 per row.")
    if provenance is not None:
        _validate_provenance(provenance, context, sample_ids, labels, fold_ids)
    return MemberPredictionArtifact(context, sample_ids, labels, raw_class_ids, logits,
                                    probabilities, fold_ids, provenance)


def export_member_prediction_artifact(
    outputs: PredictionOutputs, metadata: Mapping[str, np.ndarray],
    context: MemberPredictionContext, path: str | Path, *,
    provenance: PredictionProvenance | None = None,
) -> Path:
    """Validate and atomically publish one self-contained member artifact."""

    if context.ensemble_mode == "stratified_3fold" and provenance is None:
        raise ValueError("Schema-3 stratified_3fold export requires assignment provenance.")
    logits = np.asarray(outputs.logits)
    row_count = int(logits.shape[0]) if logits.ndim >= 1 else 0
    if "sample_id" in metadata:
        sample_ids = np.asarray(metadata["sample_id"]).astype(np.str_, copy=False)
        if sample_ids.shape != (row_count,):
            raise ValueError("sample_id metadata row count does not match prediction outputs.")
    else:
        sample_ids = build_stable_sample_ids(metadata, row_count)
    implied = context.fold_id if context.split == "validation" and context.fold_id is not None else -1
    fold_ids = np.asarray(metadata.get("fold_id", np.full(row_count, implied)), dtype=np.int16)
    artifact = _artifact_from_arrays(
        context, sample_ids=sample_ids, labels=np.asarray(outputs.labels),
        raw_class_ids=np.asarray(outputs.class_ids), logits=logits,
        probabilities=np.asarray(outputs.probabilities), fold_ids=fold_ids,
        provenance=provenance,
    )
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Member prediction artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle, schema_version=np.asarray(MEMBER_PREDICTION_SCHEMA_VERSION, dtype=np.int32),
                artifact_kind=np.asarray(MEMBER_PREDICTION_KIND), run_id=np.asarray(context.run_id),
                method=np.asarray(context.method), method_protocol=np.asarray(context.method_protocol),
                task_id=np.asarray(context.task_id, dtype=np.int32), split=np.asarray(context.split),
                member_id=np.asarray(context.member_id, dtype=np.int32),
                experiment_seed=np.asarray(context.experiment_seed, dtype=np.int64),
                member_seed=np.asarray(context.member_seed, dtype=np.int64),
                ensemble_mode=np.asarray(context.ensemble_mode),
                fold_id=np.asarray(-1 if context.fold_id is None else context.fold_id, dtype=np.int32),
                fold_count=np.asarray(-1 if context.fold_count is None else context.fold_count, dtype=np.int32),
                fold_seed=np.asarray(-1 if context.fold_seed is None else context.fold_seed, dtype=np.int64),
                provenance_json=np.asarray("" if provenance is None else json.dumps(asdict(provenance), sort_keys=True)),
                sample_ids=artifact.sample_ids, labels=artifact.labels,
                raw_class_ids=artifact.raw_class_ids, fold_ids=artifact.fold_ids,
                logits=artifact.logits, probabilities=artifact.probabilities,
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
    path: str | Path, *, expected_context: MemberPredictionContext | None = None,
) -> MemberPredictionArtifact:
    """Load schema 1--3 artifacts; legacy schemas remain readable."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing member prediction artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "schema_version" not in payload.files:
            raise ValueError("Member prediction artifact is missing keys: ['schema_version']")
        schema = int(np.asarray(payload["schema_version"]).item())
        required = {"schema_version", "artifact_kind", "run_id", "method", "method_protocol",
                    "task_id", "split", "member_id", "experiment_seed", "member_seed",
                    "sample_ids", "labels", "raw_class_ids", "logits", "probabilities"}
        if schema >= 2:
            required.update({"ensemble_mode", "fold_id", "fold_count", "fold_seed"})
        if schema >= 3:
            required.update({"fold_ids", "provenance_json"})
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Member prediction artifact is missing keys: {missing}")
        if schema not in LEGACY_MEMBER_PREDICTION_SCHEMA_VERSIONS | {MEMBER_PREDICTION_SCHEMA_VERSION}:
            raise ValueError(f"Unsupported member prediction schema version: {schema}.")
        if str(_read_scalar(payload, "artifact_kind")) != MEMBER_PREDICTION_KIND:
            raise ValueError("Unexpected member prediction artifact kind.")
        mode = str(_read_scalar(payload, "ensemble_mode")) if schema >= 2 else "seeded"
        raw_fold_id = int(_read_scalar(payload, "fold_id")) if schema >= 2 else -1
        raw_fold_count = int(_read_scalar(payload, "fold_count")) if schema >= 2 else -1
        raw_fold_seed = int(_read_scalar(payload, "fold_seed")) if schema >= 2 else -1
        context = MemberPredictionContext(
            run_id=str(_read_scalar(payload, "run_id")), method=str(_read_scalar(payload, "method")),
            method_protocol=str(_read_scalar(payload, "method_protocol")),
            task_id=int(_read_scalar(payload, "task_id")), split=str(_read_scalar(payload, "split")),
            member_id=int(_read_scalar(payload, "member_id")),
            experiment_seed=int(_read_scalar(payload, "experiment_seed")),
            member_seed=int(_read_scalar(payload, "member_seed")), ensemble_mode=mode,
            fold_id=None if raw_fold_id < 0 else raw_fold_id,
            fold_count=None if raw_fold_count < 0 else raw_fold_count,
            fold_seed=None if raw_fold_seed < 0 else raw_fold_seed,
        )
        provenance = None
        if schema >= 3:
            raw = str(_read_scalar(payload, "provenance_json"))
            if raw:
                try:
                    provenance = PredictionProvenance(**json.loads(raw))
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError("Invalid prediction provenance JSON.") from error
            if mode == "stratified_3fold" and provenance is None:
                raise ValueError("Schema-3 stratified_3fold artifact lacks assignment provenance.")
        artifact = _artifact_from_arrays(
            context, sample_ids=np.asarray(payload["sample_ids"]), labels=np.asarray(payload["labels"]),
            raw_class_ids=np.asarray(payload["raw_class_ids"]), logits=np.asarray(payload["logits"]),
            probabilities=np.asarray(payload["probabilities"]),
            fold_ids=np.asarray(payload["fold_ids"]) if schema >= 3 else None,
            provenance=provenance,
        )
    if expected_context is not None and artifact.context != expected_context:
        raise ValueError("Member prediction provenance does not match the expected context.")
    return artifact
