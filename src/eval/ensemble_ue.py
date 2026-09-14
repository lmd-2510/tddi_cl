#!/usr/bin/env python3
"""Offline probability ensemble and leak-free held-out-fold OOF assembly."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence
import uuid

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.predictions import (  # noqa: E402
    MemberPredictionArtifact,
    PredictionProvenance,
    load_member_prediction_artifact,
    partition_identity_sha256,
)


OFFLINE_ENSEMBLE_SCHEMA_VERSION = 4
LEGACY_OFFLINE_ENSEMBLE_SCHEMA_VERSIONS = frozenset({1, 2, 3})
OFFLINE_ENSEMBLE_KIND = "ddi_cil_offline_probability_ensemble"
EXPECTED_MEMBER_COUNT = 3
ENSEMBLE_ONLY_METRICS = (
    "expected_member_entropy",
    "mutual_information",
    "normalized_mi",
    "mean_probability_variance",
    "total_probability_variance",
    "member_normalized_mi",
    "pairwise_disagreement",
)
ALL_METRICS = (
    "predictive_entropy", *ENSEMBLE_ONLY_METRICS[:2], "normalized_entropy",
    *ENSEMBLE_ONLY_METRICS[2:6], "entropy_confidence", "max_probability",
    "confidence", ENSEMBLE_ONLY_METRICS[-1],
)


@dataclass(frozen=True)
class OfflineEnsembleContext:
    method: str
    method_protocol: str
    task_id: int
    split: str
    experiment_seed: int
    member_ids: tuple[int, int, int]
    member_seeds: tuple[int, int, int]
    source_run_ids: tuple[str, str, str]
    member_count: int = EXPECTED_MEMBER_COUNT
    ensemble_mode: str = "seeded"


@dataclass(frozen=True)
class OfflineEnsembleArtifact:
    context: OfflineEnsembleContext
    sample_ids: np.ndarray
    labels: np.ndarray
    raw_class_ids: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    predictive_entropy: np.ndarray
    expected_member_entropy: np.ndarray
    mutual_information: np.ndarray
    normalized_entropy: np.ndarray
    normalized_mi: np.ndarray
    mean_probability_variance: np.ndarray
    total_probability_variance: np.ndarray
    member_normalized_mi: np.ndarray
    entropy_confidence: np.ndarray
    max_probability: np.ndarray
    confidence: np.ndarray
    pairwise_disagreement: np.ndarray
    source_artifact_paths: tuple[str, str, str]
    prediction_count: np.ndarray
    unavailable_metrics: tuple[str, ...] = ()
    member_provenance: tuple[PredictionProvenance, PredictionProvenance, PredictionProvenance] | None = None

    @property
    def row_count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def class_count(self) -> int:
        return int(self.raw_class_ids.shape[0])

    def metric_available(self, name: str) -> bool:
        return name not in self.unavailable_metrics


def _require_equal_scalar(artifacts: Sequence[MemberPredictionArtifact], field: str) -> object:
    values = [getattr(artifact.context, field) for artifact in artifacts]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Member artifact {field} mismatch: {values}")
    return values[0]


def _require_equal_array(artifacts: Sequence[MemberPredictionArtifact], field: str) -> np.ndarray:
    reference = np.asarray(getattr(artifacts[0], field))
    for artifact in artifacts[1:]:
        if not np.array_equal(np.asarray(getattr(artifact, field)), reference):
            raise ValueError(
                f"Member artifact {field} mismatch or permutation for member_id={artifact.context.member_id}."
            )
    return reference.copy()


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    logarithms = np.zeros_like(probabilities, dtype=np.float64)
    np.log(probabilities, out=logarithms, where=probabilities > 0.0)
    return -np.sum(probabilities * logarithms, axis=-1)


def _normalize_mi_by_member_count(mutual_information: np.ndarray, member_count: int) -> np.ndarray:
    if member_count <= 1:
        return np.zeros_like(mutual_information, dtype=np.float64)
    return np.clip(np.asarray(mutual_information, dtype=np.float64) / np.log(member_count), 0.0, 1.0)


def _shared_provenance(provenance: PredictionProvenance) -> tuple[object, ...]:
    return (
        provenance.assignment_sha256,
        provenance.fold_manifest_sha256,
        provenance.task_file_sha256,
        provenance.study_contract_sha256,
        provenance.preprocessing_policy,
        provenance.ranking_policy,
        provenance.buffer_policy,
        provenance.global_memory_budget,
    )


def _validate_member_provenance(
    artifacts: Sequence[MemberPredictionArtifact], *, is_oof: bool,
) -> tuple[PredictionProvenance, PredictionProvenance, PredictionProvenance] | None:
    values = [artifact.provenance for artifact in artifacts]
    if all(value is None for value in values):
        # Schema 1/2 remains usable, but cannot claim assignment-level coverage.
        return None
    if any(value is None for value in values):
        raise ValueError("Member prediction provenance is missing for part of the study.")
    provenance = tuple(values)  # type: ignore[arg-type]
    reference = _shared_provenance(provenance[0])
    if any(_shared_provenance(value) != reference for value in provenance[1:]):
        raise ValueError("Member artifacts do not share the same partition/study contract.")
    for artifact, value in zip(artifacts, provenance, strict=True):
        if value.preprocessing_member_id != artifact.context.member_id:
            raise ValueError("Member preprocessing provenance belongs to the wrong member.")
        actual = partition_identity_sha256(artifact.sample_ids, artifact.labels, artifact.fold_ids)
        if actual != value.expected_partition_sha256 or artifact.row_count != value.expected_partition_rows:
            raise ValueError("Member IDs/labels do not match its recorded assignment partition.")
    if is_oof:
        expected = {(value.expected_oof_rows, value.expected_oof_sha256) for value in provenance}
        if len(expected) != 1 or next(iter(expected))[0] is None:
            raise ValueError("OOF artifacts disagree on expected assignment coverage.")
    # Preprocessing hashes may legitimately differ because each scaler is fit on
    # that member's own two-fold task-0 training partition.
    return provenance  # type: ignore[return-value]


def _row_fold_ids(artifact: MemberPredictionArtifact) -> np.ndarray:
    """Derive row folds for legacy in-memory/schema-2 artifacts."""

    if artifact.fold_ids is not None:
        return np.asarray(artifact.fold_ids, dtype=np.int16)
    implied = (
        artifact.context.fold_id
        if artifact.context.split == "validation" and artifact.context.fold_id is not None
        else -1
    )
    return np.full(artifact.row_count, implied, dtype=np.int16)


def aggregate_member_predictions(
    artifacts: Sequence[MemberPredictionArtifact], *,
    source_artifact_paths: Sequence[str | Path] | None = None,
    ensemble_mode: str | None = None,
) -> OfflineEnsembleArtifact:
    """Mean common-test probabilities or concatenate disjoint held-out folds."""

    if len(artifacts) != EXPECTED_MEMBER_COUNT:
        raise ValueError(f"Offline ensemble requires exactly {EXPECTED_MEMBER_COUNT} member artifacts.")
    method = str(_require_equal_scalar(artifacts, "method"))
    method_protocol = str(_require_equal_scalar(artifacts, "method_protocol"))
    task_id = int(_require_equal_scalar(artifacts, "task_id"))
    split = str(_require_equal_scalar(artifacts, "split"))
    experiment_seed = int(_require_equal_scalar(artifacts, "experiment_seed"))
    member_ids = tuple(int(artifact.context.member_id) for artifact in artifacts)
    if len(set(member_ids)) != EXPECTED_MEMBER_COUNT:
        raise ValueError(f"Member IDs must be distinct, got {member_ids}.")
    modes = {artifact.context.ensemble_mode for artifact in artifacts}
    if len(modes) != 1:
        raise ValueError(f"Member ensemble modes do not match: {sorted(modes)}")
    resolved_mode = ensemble_mode or next(iter(modes))
    if resolved_mode not in {"seeded", "stratified_3fold"} or modes != {resolved_mode}:
        raise ValueError("Requested ensemble mode does not match member provenance.")
    raw_class_ids = _require_equal_array(artifacts, "raw_class_ids")
    is_oof = resolved_mode == "stratified_3fold" and split == "validation"

    if is_oof:
        fold_ids = tuple(artifact.context.fold_id for artifact in artifacts)
        if set(fold_ids) != {0, 1, 2}:
            raise ValueError(f"OOF member artifacts must cover folds 0, 1, 2; got {fold_ids}.")
        if {artifact.context.fold_count for artifact in artifacts} != {3} or len(
            {artifact.context.fold_seed for artifact in artifacts}
        ) != 1:
            raise ValueError("OOF member artifacts disagree on fold count or fold seed.")
        for artifact in artifacts:
            if not np.all(_row_fold_ids(artifact) == artifact.context.fold_id):
                raise ValueError("OOF prediction rows belong to the wrong held-out fold.")
        provenance = _validate_member_provenance(artifacts, is_oof=True)
        sample_ids = np.concatenate([artifact.sample_ids for artifact in artifacts])
        labels = np.concatenate([artifact.labels for artifact in artifacts])
        row_folds = np.concatenate([_row_fold_ids(artifact) for artifact in artifacts])
        probabilities64 = np.concatenate(
            [np.asarray(artifact.probabilities, dtype=np.float64) for artifact in artifacts], axis=0
        )
        if np.unique(sample_ids).shape[0] != sample_ids.shape[0]:
            raise ValueError("OOF member validation folds overlap in sample IDs.")
        if provenance is not None:
            expected_rows = provenance[0].expected_oof_rows
            expected_sha = provenance[0].expected_oof_sha256
            if sample_ids.shape[0] != expected_rows or partition_identity_sha256(
                sample_ids, labels, row_folds
            ) != expected_sha:
                raise ValueError("OOF coverage is incomplete or misaligned with frozen assignments.")
        order = np.argsort(sample_ids, kind="stable")
        sample_ids, labels, probabilities64 = sample_ids[order], labels[order], probabilities64[order]
        member_probabilities = None
        output_split = "oof"
        prediction_count = np.ones(sample_ids.shape[0], dtype=np.int16)
        unavailable = ENSEMBLE_ONLY_METRICS
    else:
        provenance = _validate_member_provenance(artifacts, is_oof=False)
        sample_ids = _require_equal_array(artifacts, "sample_ids")
        labels = _require_equal_array(artifacts, "labels")
        if resolved_mode == "stratified_3fold" and split != "test":
            raise ValueError("stratified_3fold aggregation supports held-out validation or common test only.")
        member_probabilities = np.stack(
            [np.asarray(artifact.probabilities, dtype=np.float64) for artifact in artifacts], axis=0
        )
        expected_shape = (3, sample_ids.shape[0], raw_class_ids.shape[0])
        if member_probabilities.shape != expected_shape:
            raise ValueError(f"Member probability row/class shape mismatch: {member_probabilities.shape} != {expected_shape}.")
        probabilities64 = member_probabilities.mean(axis=0)
        output_split = split
        prediction_count = np.full(sample_ids.shape[0], 3, dtype=np.int16)
        unavailable = ()

    if not np.isfinite(probabilities64).all() or not np.allclose(probabilities64.sum(1), 1.0, atol=1e-8):
        raise ValueError("Member probability rows must be finite and sum to one.")
    predictive_entropy = _entropy(probabilities64)
    class_count = raw_class_ids.shape[0]
    normalized_entropy = (
        np.zeros_like(predictive_entropy) if class_count <= 1
        else np.clip(predictive_entropy / np.log(class_count), 0.0, 1.0)
    )
    entropy_confidence = 1.0 - normalized_entropy
    max_probability = probabilities64.max(axis=1)
    if member_probabilities is None:
        missing = np.full(sample_ids.shape[0], np.nan, dtype=np.float64)
        expected_member_entropy = mutual_information = normalized_mi = missing.copy()
        mean_variance = total_variance = member_normalized_mi = missing.copy()
        disagreement = missing.copy()
    else:
        expected_member_entropy = _entropy(member_probabilities).mean(axis=0)
        mutual_information = np.maximum(predictive_entropy - expected_member_entropy, 0.0)
        normalized_mi = (
            np.zeros_like(mutual_information) if class_count <= 1
            else np.clip(mutual_information / np.log(class_count), 0.0, 1.0)
        )
        variance = member_probabilities.var(axis=0)
        mean_variance, total_variance = variance.mean(axis=1), variance.sum(axis=1)
        member_normalized_mi = _normalize_mi_by_member_count(mutual_information, 3)
        predictions_by_member = member_probabilities.argmax(axis=2)
        pairs = [predictions_by_member[a] != predictions_by_member[b]
                 for a in range(3) for b in range(a + 1, 3)]
        disagreement = np.mean(np.stack(pairs), axis=0)
    predictions = raw_class_ids[probabilities64.argmax(axis=1)]
    if source_artifact_paths is None:
        sources = ("", "", "")
    elif len(source_artifact_paths) != 3:
        raise ValueError("Exactly three source artifact paths are required.")
    else:
        sources = tuple(str(Path(path)) for path in source_artifact_paths)
    artifact = OfflineEnsembleArtifact(
        context=OfflineEnsembleContext(
            method, method_protocol, task_id, output_split, experiment_seed,
            member_ids, tuple(int(item.context.member_seed) for item in artifacts),
            tuple(item.context.run_id for item in artifacts), 3, resolved_mode,
        ),
        sample_ids=sample_ids, labels=labels, raw_class_ids=raw_class_ids,
        probabilities=probabilities64.astype(np.float32),
        predictions=predictions.astype(np.int64), predictive_entropy=predictive_entropy,
        expected_member_entropy=expected_member_entropy, mutual_information=mutual_information,
        normalized_entropy=normalized_entropy, normalized_mi=normalized_mi,
        mean_probability_variance=mean_variance, total_probability_variance=total_variance,
        member_normalized_mi=member_normalized_mi,
        entropy_confidence=entropy_confidence, max_probability=max_probability,
        confidence=entropy_confidence.copy(), pairwise_disagreement=disagreement,
        source_artifact_paths=sources, prediction_count=prediction_count,
        unavailable_metrics=tuple(unavailable), member_provenance=provenance,
    )
    _validate_ensemble_artifact(artifact)
    return artifact


def _validate_ensemble_artifact(artifact: OfflineEnsembleArtifact) -> None:
    context, rows, classes = artifact.context, artifact.row_count, artifact.class_count
    if not context.method or not context.method_protocol or context.task_id < 0 or context.experiment_seed < 0:
        raise ValueError("Offline ensemble context is invalid.")
    if len(set(context.member_ids)) != 3 or len(context.member_seeds) != 3 or context.member_count != 3:
        raise ValueError("Offline ensemble provenance must contain three distinct members.")
    if context.ensemble_mode not in {"seeded", "stratified_3fold"}:
        raise ValueError("Offline ensemble has an unsupported ensemble_mode.")
    if context.split == "oof" and context.ensemble_mode != "stratified_3fold":
        raise ValueError("Only stratified_3fold may produce an OOF artifact.")
    if rows <= 0 or classes <= 0 or artifact.sample_ids.ndim != 1:
        raise ValueError("Offline ensemble cannot be empty.")
    if artifact.sample_ids.dtype.kind not in {"U", "S"} or np.unique(artifact.sample_ids).size != rows:
        raise ValueError("Offline ensemble sample IDs must be unique strings.")
    if artifact.labels.dtype != np.int64 or artifact.predictions.dtype != np.int64:
        raise TypeError("Offline ensemble labels/predictions must use int64.")
    if artifact.raw_class_ids.dtype != np.int64 or np.unique(artifact.raw_class_ids).size != classes:
        raise TypeError("Offline ensemble raw_class_ids must be unique int64 values.")
    if artifact.labels.shape != (rows,) or artifact.predictions.shape != (rows,):
        raise ValueError("Offline ensemble labels/predictions row count mismatch.")
    if artifact.probabilities.dtype != np.float32 or artifact.probabilities.shape != (rows, classes):
        raise ValueError("Offline ensemble probability width/dtype is invalid.")
    if not np.isfinite(artifact.probabilities).all() or not np.allclose(artifact.probabilities.sum(1), 1.0, atol=1e-7):
        raise ValueError("Offline ensemble probability rows must be finite and sum to one.")
    if not np.isin(artifact.labels, artifact.raw_class_ids).all():
        raise ValueError("Offline ensemble labels are absent from raw_class_ids.")
    if not np.array_equal(artifact.predictions, artifact.raw_class_ids[artifact.probabilities.argmax(1)]):
        raise ValueError("Offline ensemble predictions do not match probabilities.")
    expected_count = 1 if context.split == "oof" else 3
    if artifact.prediction_count.dtype != np.int16 or artifact.prediction_count.shape != (rows,):
        raise ValueError("Offline ensemble prediction_count must be an int16 row vector.")
    if not np.all(artifact.prediction_count == expected_count):
        raise ValueError("OOF/common-test per-sample prediction_count semantics are invalid.")
    unknown = set(artifact.unavailable_metrics) - set(ENSEMBLE_ONLY_METRICS)
    if unknown:
        raise ValueError(f"Unknown unavailable UE metrics: {sorted(unknown)}")
    for name in ALL_METRICS:
        values = np.asarray(getattr(artifact, name))
        if values.shape != (rows,):
            raise ValueError(f"Offline ensemble {name} row count mismatch.")
        if name in artifact.unavailable_metrics:
            if not np.isnan(values).all():
                raise ValueError(f"Unavailable OOF metric {name} must be represented as NaN, not zero.")
        elif not np.isfinite(values).all():
            raise ValueError(f"Available metric {name} must be finite.")
    expected_entropy = _entropy(artifact.probabilities.astype(np.float64))
    if not np.allclose(artifact.predictive_entropy, expected_entropy, atol=1e-7):
        raise ValueError("Predictive entropy does not match probabilities.")
    if not np.array_equal(artifact.confidence, artifact.entropy_confidence):
        raise ValueError("confidence must remain an alias of entropy_confidence.")
    if not np.allclose(artifact.entropy_confidence, 1.0 - artifact.normalized_entropy):
        raise ValueError("entropy_confidence must equal 1-normalized_entropy.")
    if not np.allclose(artifact.max_probability, artifact.probabilities.max(1), atol=1e-7):
        raise ValueError("max_probability does not match probabilities.")
    if not artifact.unavailable_metrics:
        expected_mi = np.maximum(artifact.predictive_entropy - artifact.expected_member_entropy, 0.0)
        if not np.allclose(artifact.mutual_information, expected_mi, atol=1e-12):
            raise ValueError("Mutual information entropy decomposition is invalid.")
        if not np.allclose(artifact.total_probability_variance,
                           artifact.mean_probability_variance * classes, atol=1e-12):
            raise ValueError("Total/mean probability variance are inconsistent.")


def export_offline_ensemble_artifact(artifact: OfflineEnsembleArtifact, path: str | Path) -> Path:
    _validate_ensemble_artifact(artifact)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Offline ensemble artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    provenance_json = "" if artifact.member_provenance is None else json.dumps(
        [asdict(value) for value in artifact.member_provenance], sort_keys=True
    )
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, schema_version=np.asarray(OFFLINE_ENSEMBLE_SCHEMA_VERSION, dtype=np.int32),
                artifact_kind=np.asarray(OFFLINE_ENSEMBLE_KIND),
                method=np.asarray(artifact.context.method), method_protocol=np.asarray(artifact.context.method_protocol),
                task_id=np.asarray(artifact.context.task_id, dtype=np.int32), split=np.asarray(artifact.context.split),
                experiment_seed=np.asarray(artifact.context.experiment_seed, dtype=np.int64),
                member_ids=np.asarray(artifact.context.member_ids, dtype=np.int32),
                member_seeds=np.asarray(artifact.context.member_seeds, dtype=np.int64),
                member_count=np.asarray(artifact.context.member_count, dtype=np.int32),
                ensemble_mode=np.asarray(artifact.context.ensemble_mode),
                source_run_ids=np.asarray(artifact.context.source_run_ids),
                source_artifact_paths=np.asarray(artifact.source_artifact_paths),
                prediction_count=artifact.prediction_count,
                unavailable_metrics=np.asarray(artifact.unavailable_metrics),
                member_provenance_json=np.asarray(provenance_json),
                sample_ids=artifact.sample_ids, labels=artifact.labels,
                raw_class_ids=artifact.raw_class_ids, probabilities=artifact.probabilities,
                predictions=artifact.predictions,
                **{name: getattr(artifact, name) for name in ALL_METRICS},
            )
            handle.flush()
            os.fsync(handle.fileno())
        load_offline_ensemble_artifact(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_scalar(payload: Mapping[str, np.ndarray], key: str) -> object:
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError(f"Offline ensemble {key} must be a scalar.")
    return value.item()


def load_offline_ensemble_artifact(path: str | Path) -> OfflineEnsembleArtifact:
    """Read schemas 1--4; old OOF fake-zero UE is marked unavailable."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing offline ensemble artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "schema_version" not in payload.files:
            raise ValueError("Offline ensemble artifact is missing schema_version.")
        schema = int(_read_scalar(payload, "schema_version"))
        if schema not in LEGACY_OFFLINE_ENSEMBLE_SCHEMA_VERSIONS | {OFFLINE_ENSEMBLE_SCHEMA_VERSION}:
            raise ValueError(f"Unsupported offline ensemble schema version: {schema}.")
        required = {"artifact_kind", "method", "method_protocol", "task_id", "split",
                    "experiment_seed", "member_ids", "member_seeds", "source_run_ids",
                    "source_artifact_paths", "sample_ids", "labels", "raw_class_ids",
                    "probabilities", "predictions", "predictive_entropy",
                    "expected_member_entropy", "mutual_information", "normalized_entropy",
                    "normalized_mi", "mean_probability_variance", "confidence",
                    "pairwise_disagreement"}
        if schema >= 2:
            required.update({"member_count", "total_probability_variance", "member_normalized_mi",
                             "entropy_confidence", "max_probability"})
        if schema >= 3:
            required.add("ensemble_mode")
        if schema >= 4:
            required.update({"prediction_count", "unavailable_metrics", "member_provenance_json"})
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Offline ensemble artifact is missing keys: {missing}")
        if str(_read_scalar(payload, "artifact_kind")) != OFFLINE_ENSEMBLE_KIND:
            raise ValueError("Unexpected offline ensemble artifact kind.")
        member_ids = tuple(int(v) for v in payload["member_ids"].tolist())
        member_seeds = tuple(int(v) for v in payload["member_seeds"].tolist())
        run_ids = tuple(str(v) for v in payload["source_run_ids"].tolist())
        sources = tuple(str(v) for v in payload["source_artifact_paths"].tolist())
        if not all(len(v) == 3 for v in (member_ids, member_seeds, run_ids, sources)):
            raise ValueError("Offline ensemble provenance must contain exactly three members.")
        member_count = int(_read_scalar(payload, "member_count")) if schema >= 2 else 3
        mode = str(_read_scalar(payload, "ensemble_mode")) if schema >= 3 else "seeded"
        split = str(_read_scalar(payload, "split"))
        probabilities = np.asarray(payload["probabilities"])
        raw_classes = np.asarray(payload["raw_class_ids"])
        normalized_entropy = np.asarray(payload["normalized_entropy"])
        mutual_information = np.asarray(payload["mutual_information"])
        mean_variance = np.asarray(payload["mean_probability_variance"])
        if schema == 1:
            entropy_confidence = 1.0 - normalized_entropy
            max_probability = probabilities.max(1)
            total_variance = mean_variance * raw_classes.shape[0]
            member_normalized_mi = _normalize_mi_by_member_count(mutual_information, member_count)
        else:
            entropy_confidence = np.asarray(payload["entropy_confidence"])
            max_probability = np.asarray(payload["max_probability"])
            total_variance = np.asarray(payload["total_probability_variance"])
            member_normalized_mi = np.asarray(payload["member_normalized_mi"])
        provenance = None
        if schema >= 4:
            raw = str(_read_scalar(payload, "member_provenance_json"))
            if raw:
                values = json.loads(raw)
                provenance = tuple(PredictionProvenance(**value) for value in values)
            unavailable = tuple(str(value) for value in payload["unavailable_metrics"].tolist())
            prediction_count = np.asarray(payload["prediction_count"])
        else:
            unavailable = ENSEMBLE_ONLY_METRICS if split == "oof" else ()
            prediction_count = np.full(probabilities.shape[0], 1 if split == "oof" else member_count, dtype=np.int16)
        metric_values = {name: np.asarray(payload[name]) for name in ALL_METRICS if name in payload.files}
        if unavailable:
            for name in unavailable:
                metric_values[name] = np.full(probabilities.shape[0], np.nan)
        artifact = OfflineEnsembleArtifact(
            OfflineEnsembleContext(
                str(_read_scalar(payload, "method")), str(_read_scalar(payload, "method_protocol")),
                int(_read_scalar(payload, "task_id")), split,
                int(_read_scalar(payload, "experiment_seed")), member_ids, member_seeds,
                run_ids, member_count, mode,
            ), np.asarray(payload["sample_ids"]), np.asarray(payload["labels"]), raw_classes,
            probabilities, np.asarray(payload["predictions"]),
            metric_values["predictive_entropy"], metric_values["expected_member_entropy"],
            metric_values["mutual_information"], metric_values["normalized_entropy"],
            metric_values["normalized_mi"], metric_values["mean_probability_variance"],
            total_variance if "total_probability_variance" not in unavailable else metric_values["total_probability_variance"],
            member_normalized_mi if "member_normalized_mi" not in unavailable else metric_values["member_normalized_mi"],
            entropy_confidence, max_probability, np.asarray(payload["confidence"]),
            metric_values["pairwise_disagreement"], sources, prediction_count,
            unavailable, provenance,  # type: ignore[arg-type]
        )
    _validate_ensemble_artifact(artifact)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate exactly three member prediction artifacts.")
    parser.add_argument("--member-artifacts", nargs=3, required=True, type=Path,
                        metavar=("MEMBER_0", "MEMBER_1", "MEMBER_2"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=["seeded", "stratified_3fold"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members = [load_member_prediction_artifact(path) for path in args.member_artifacts]
    artifact = aggregate_member_predictions(
        members, source_artifact_paths=args.member_artifacts, ensemble_mode=args.mode
    )
    output = export_offline_ensemble_artifact(artifact, args.out)
    print(
        f"exported={output} rows={artifact.row_count} classes={artifact.class_count} "
        f"task={artifact.context.task_id} split={artifact.context.split} "
        f"per_sample_predictions={int(artifact.prediction_count[0])} "
        f"unavailable={artifact.unavailable_metrics}", flush=True,
    )


if __name__ == "__main__":
    main()
