#!/usr/bin/env python3
"""Strict offline probability ensemble and uncertainty estimation for three members."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.predictions import (  # noqa: E402
    MemberPredictionArtifact,
    load_member_prediction_artifact,
)


OFFLINE_ENSEMBLE_SCHEMA_VERSION = 3
LEGACY_OFFLINE_ENSEMBLE_SCHEMA_VERSION = 1
LEGACY_OFFLINE_ENSEMBLE_SCHEMA_VERSION_2 = 2
OFFLINE_ENSEMBLE_KIND = "ddi_cil_offline_probability_ensemble"
EXPECTED_MEMBER_COUNT = 3


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

    @property
    def row_count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def class_count(self) -> int:
        return int(self.raw_class_ids.shape[0])


def _require_equal_scalar(
    artifacts: Sequence[MemberPredictionArtifact],
    field: str,
) -> object:
    values = [getattr(artifact.context, field) for artifact in artifacts]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Member artifact {field} mismatch: {values}")
    return values[0]


def _require_equal_array(
    artifacts: Sequence[MemberPredictionArtifact],
    field: str,
) -> np.ndarray:
    reference = np.asarray(getattr(artifacts[0], field))
    for artifact in artifacts[1:]:
        candidate = np.asarray(getattr(artifact, field))
        if not np.array_equal(candidate, reference):
            raise ValueError(
                f"Member artifact {field} mismatch or permutation for "
                f"member_id={artifact.context.member_id}."
            )
    return reference.copy()


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    logarithms = np.zeros_like(probabilities, dtype=np.float64)
    np.log(probabilities, out=logarithms, where=probabilities > 0.0)
    return -np.sum(probabilities * logarithms, axis=-1)


def _normalize_mi_by_member_count(
    mutual_information: np.ndarray,
    member_count: int,
) -> np.ndarray:
    """Normalize epistemic uncertainty by the ensemble-size entropy ceiling."""

    if member_count <= 1:
        return np.zeros_like(mutual_information, dtype=np.float64)
    return np.clip(
        np.asarray(mutual_information, dtype=np.float64) / float(np.log(member_count)),
        0.0,
        1.0,
    )


def aggregate_member_predictions(
    artifacts: Sequence[MemberPredictionArtifact],
    *,
    source_artifact_paths: Sequence[str | Path] | None = None,
    ensemble_mode: str | None = None,
) -> OfflineEnsembleArtifact:
    """Average seeded members or merge three leak-free held-out fold predictions."""

    if len(artifacts) != EXPECTED_MEMBER_COUNT:
        raise ValueError(
            f"Offline ensemble requires exactly {EXPECTED_MEMBER_COUNT} member artifacts."
        )
    method = str(_require_equal_scalar(artifacts, "method"))
    method_protocol = str(_require_equal_scalar(artifacts, "method_protocol"))
    task_id = int(_require_equal_scalar(artifacts, "task_id"))
    split = str(_require_equal_scalar(artifacts, "split"))
    experiment_seed = int(_require_equal_scalar(artifacts, "experiment_seed"))
    member_ids = tuple(int(artifact.context.member_id) for artifact in artifacts)
    if len(set(member_ids)) != EXPECTED_MEMBER_COUNT:
        raise ValueError(f"Member IDs must be distinct, got {member_ids}.")

    artifact_modes = {artifact.context.ensemble_mode for artifact in artifacts}
    if len(artifact_modes) != 1:
        raise ValueError(f"Member ensemble modes do not match: {sorted(artifact_modes)}")
    resolved_mode = ensemble_mode or next(iter(artifact_modes))
    if resolved_mode not in {"seeded", "stratified_3fold"}:
        raise ValueError(f"Unsupported ensemble mode: {resolved_mode!r}.")
    if artifact_modes != {resolved_mode}:
        raise ValueError("Requested ensemble mode does not match member provenance.")

    raw_class_ids = _require_equal_array(artifacts, "raw_class_ids")
    is_oof = resolved_mode == "stratified_3fold" and split.casefold() == "validation"
    if is_oof:
        fold_ids = tuple(artifact.context.fold_id for artifact in artifacts)
        fold_counts = {artifact.context.fold_count for artifact in artifacts}
        fold_seeds = {artifact.context.fold_seed for artifact in artifacts}
        if set(fold_ids) != set(range(EXPECTED_MEMBER_COUNT)):
            raise ValueError(f"OOF member artifacts must cover folds 0, 1, 2; got {fold_ids}.")
        if fold_counts != {EXPECTED_MEMBER_COUNT} or len(fold_seeds) != 1:
            raise ValueError("OOF member artifacts disagree on fold count or fold seed.")
        sample_ids = np.concatenate([artifact.sample_ids for artifact in artifacts])
        labels = np.concatenate([artifact.labels for artifact in artifacts])
        mean_probabilities64 = np.concatenate(
            [np.asarray(artifact.probabilities, dtype=np.float64) for artifact in artifacts],
            axis=0,
        )
        if np.unique(sample_ids).shape[0] != sample_ids.shape[0]:
            raise ValueError("OOF member validation folds overlap in sample IDs.")
        order = np.argsort(sample_ids, kind="stable")
        sample_ids = sample_ids[order]
        labels = labels[order]
        mean_probabilities64 = mean_probabilities64[order]
        member_probabilities = None
        output_split = "oof"
    else:
        sample_ids = _require_equal_array(artifacts, "sample_ids")
        labels = _require_equal_array(artifacts, "labels")
        member_probabilities = np.stack(
            [np.asarray(artifact.probabilities, dtype=np.float64) for artifact in artifacts],
            axis=0,
        )
        expected_shape = (EXPECTED_MEMBER_COUNT, sample_ids.shape[0], raw_class_ids.shape[0])
        if member_probabilities.shape != expected_shape:
            raise ValueError(
                "Member probability row/class shape mismatch: "
                f"{member_probabilities.shape} != {expected_shape}."
            )
        mean_probabilities64 = member_probabilities.mean(axis=0)
        output_split = split
    if not np.isfinite(mean_probabilities64).all():
        raise ValueError("Member probabilities contain non-finite values.")
    if not np.allclose(mean_probabilities64.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("Member probability rows must sum to one before aggregation.")

    predictive_entropy = _entropy(mean_probabilities64)
    if member_probabilities is None:
        expected_member_entropy = predictive_entropy.copy()
        mutual_information = np.zeros_like(predictive_entropy)
    else:
        member_entropy = _entropy(member_probabilities)
        expected_member_entropy = member_entropy.mean(axis=0)
        mutual_information = np.maximum(predictive_entropy - expected_member_entropy, 0.0)
    class_count = raw_class_ids.shape[0]
    if class_count <= 1:
        normalized_entropy = np.zeros_like(predictive_entropy)
        normalized_mi = np.zeros_like(mutual_information)
    else:
        normalization = float(np.log(class_count))
        normalized_entropy = np.clip(predictive_entropy / normalization, 0.0, 1.0)
        normalized_mi = np.clip(mutual_information / normalization, 0.0, 1.0)
    entropy_confidence = 1.0 - normalized_entropy
    # Backward-compatible alias. This is entropy-derived confidence, not the
    # maximum softmax/ensemble probability.
    confidence = entropy_confidence.copy()
    max_probability = mean_probabilities64.max(axis=1)
    probability_variance = (
        np.zeros_like(mean_probabilities64)
        if member_probabilities is None
        else member_probabilities.var(axis=0)
    )
    mean_probability_variance = probability_variance.mean(axis=1)
    total_probability_variance = probability_variance.sum(axis=1)
    member_count = len(artifacts)
    member_normalized_mi = _normalize_mi_by_member_count(
        mutual_information,
        member_count,
    )
    if member_probabilities is None:
        pairwise_disagreement = np.zeros(sample_ids.shape[0], dtype=np.float64)
    else:
        member_predictions = member_probabilities.argmax(axis=2)
        disagreements = []
        for left in range(EXPECTED_MEMBER_COUNT):
            for right in range(left + 1, EXPECTED_MEMBER_COUNT):
                disagreements.append(member_predictions[left] != member_predictions[right])
        pairwise_disagreement = np.mean(np.stack(disagreements, axis=0), axis=0)
    predictions = raw_class_ids[mean_probabilities64.argmax(axis=1)]

    if source_artifact_paths is None:
        sources = ("", "", "")
    else:
        if len(source_artifact_paths) != EXPECTED_MEMBER_COUNT:
            raise ValueError("Exactly three source artifact paths are required.")
        sources = tuple(str(Path(path)) for path in source_artifact_paths)

    artifact = OfflineEnsembleArtifact(
        context=OfflineEnsembleContext(
            method=method,
            method_protocol=method_protocol,
            task_id=task_id,
            split=output_split,
            experiment_seed=experiment_seed,
            member_ids=member_ids,  # type: ignore[arg-type]
            member_seeds=tuple(  # type: ignore[arg-type]
                int(member.context.member_seed) for member in artifacts
            ),
            source_run_ids=tuple(  # type: ignore[arg-type]
                member.context.run_id for member in artifacts
            ),
            member_count=member_count,
            ensemble_mode=resolved_mode,
        ),
        sample_ids=sample_ids,
        labels=labels,
        raw_class_ids=raw_class_ids,
        probabilities=mean_probabilities64.astype(np.float32),
        predictions=predictions.astype(np.int64, copy=False),
        predictive_entropy=predictive_entropy,
        expected_member_entropy=expected_member_entropy,
        mutual_information=mutual_information,
        normalized_entropy=normalized_entropy,
        normalized_mi=normalized_mi,
        mean_probability_variance=mean_probability_variance,
        total_probability_variance=total_probability_variance,
        member_normalized_mi=member_normalized_mi,
        entropy_confidence=entropy_confidence,
        max_probability=max_probability,
        confidence=confidence,
        pairwise_disagreement=pairwise_disagreement,
        source_artifact_paths=sources,  # type: ignore[arg-type]
    )
    _validate_ensemble_artifact(artifact)
    return artifact


def _validate_ensemble_artifact(artifact: OfflineEnsembleArtifact) -> None:
    context = artifact.context
    if not context.method or not context.method_protocol:
        raise ValueError("Offline ensemble method provenance must be non-empty.")
    if context.task_id < 0 or context.experiment_seed < 0:
        raise ValueError("Offline ensemble task/seed metadata must be non-negative.")
    if len(set(context.member_ids)) != EXPECTED_MEMBER_COUNT:
        raise ValueError("Offline ensemble member IDs must be distinct.")
    if len(context.member_seeds) != EXPECTED_MEMBER_COUNT:
        raise ValueError("Offline ensemble must record three member seeds.")
    if context.member_count != len(context.member_ids):
        raise ValueError("Offline ensemble member_count does not match member_ids.")
    if context.ensemble_mode not in {"seeded", "stratified_3fold"}:
        raise ValueError("Offline ensemble has an unsupported ensemble_mode.")
    if context.split == "oof" and context.ensemble_mode != "stratified_3fold":
        raise ValueError("Only stratified_3fold may produce an OOF artifact.")
    rows = artifact.sample_ids.shape[0]
    classes = artifact.raw_class_ids.shape[0]
    if rows == 0 or classes == 0:
        raise ValueError("Offline ensemble cannot be empty.")
    if artifact.sample_ids.ndim != 1 or artifact.sample_ids.dtype.kind not in {"U", "S"}:
        raise TypeError("Offline ensemble sample_ids must be a string vector.")
    if np.unique(artifact.sample_ids).shape[0] != rows:
        raise ValueError("Offline ensemble contains duplicate sample IDs.")
    if artifact.labels.dtype != np.dtype(np.int64):
        raise TypeError("Offline ensemble labels must use int64.")
    if artifact.predictions.dtype != np.dtype(np.int64):
        raise TypeError("Offline ensemble predictions must use int64.")
    if artifact.raw_class_ids.dtype != np.dtype(np.int64):
        raise TypeError("Offline ensemble raw_class_ids must use int64.")
    if artifact.raw_class_ids.ndim != 1 or np.unique(artifact.raw_class_ids).shape[0] != classes:
        raise ValueError("Offline ensemble raw_class_ids must be a unique vector.")
    if artifact.labels.shape != (rows,) or artifact.predictions.shape != (rows,):
        raise ValueError("Offline ensemble labels/predictions row count mismatch.")
    if artifact.probabilities.dtype != np.dtype(np.float32):
        raise TypeError("Offline ensemble probabilities must use float32.")
    if artifact.probabilities.shape != (rows, classes):
        raise ValueError("Offline ensemble probability width does not match raw_class_ids.")
    if not np.isfinite(artifact.probabilities).all():
        raise ValueError("Offline ensemble probabilities contain non-finite values.")
    if np.any(artifact.probabilities < -1e-7) or np.any(artifact.probabilities > 1.0 + 1e-7):
        raise ValueError("Offline ensemble probabilities must lie in [0, 1].")
    if not np.allclose(artifact.probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("Offline ensemble probability rows must sum to one.")
    if not np.isin(artifact.labels, artifact.raw_class_ids).all():
        raise ValueError("Offline ensemble labels are absent from raw_class_ids.")
    expected_predictions = artifact.raw_class_ids[artifact.probabilities.argmax(axis=1)]
    if not np.array_equal(artifact.predictions, expected_predictions):
        raise ValueError("Offline ensemble predictions do not match mean probabilities.")
    metric_names = (
        "predictive_entropy",
        "expected_member_entropy",
        "mutual_information",
        "normalized_entropy",
        "normalized_mi",
        "mean_probability_variance",
        "total_probability_variance",
        "member_normalized_mi",
        "entropy_confidence",
        "max_probability",
        "confidence",
        "pairwise_disagreement",
    )
    for name in metric_names:
        values = np.asarray(getattr(artifact, name))
        if values.shape != (rows,) or not np.isfinite(values).all():
            raise ValueError(f"Offline ensemble {name} must be a finite row vector.")
    tolerance = 1e-10
    if np.any(artifact.mutual_information < -tolerance):
        raise ValueError("Offline ensemble mutual information cannot be negative.")
    if np.any(artifact.expected_member_entropy > artifact.predictive_entropy + 1e-8):
        raise ValueError("Expected member entropy cannot exceed predictive entropy.")
    expected_predictive_entropy = _entropy(
        np.asarray(artifact.probabilities, dtype=np.float64)
    )
    if not np.allclose(
        artifact.predictive_entropy,
        expected_predictive_entropy,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("Predictive entropy does not match ensemble probabilities.")
    expected_mi = np.maximum(
        artifact.predictive_entropy - artifact.expected_member_entropy,
        0.0,
    )
    if not np.allclose(artifact.mutual_information, expected_mi, rtol=1e-10, atol=1e-12):
        raise ValueError("Mutual information does not match its entropy decomposition.")
    for name in (
        "normalized_entropy",
        "normalized_mi",
        "member_normalized_mi",
        "entropy_confidence",
        "max_probability",
        "confidence",
        "pairwise_disagreement",
    ):
        values = np.asarray(getattr(artifact, name))
        if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
            raise ValueError(f"Offline ensemble {name} must lie in [0, 1].")
    expected_entropy_confidence = 1.0 - artifact.normalized_entropy
    if not np.allclose(
        artifact.entropy_confidence,
        expected_entropy_confidence,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Offline ensemble entropy_confidence must equal 1-normalized_entropy.")
    if not np.array_equal(artifact.confidence, artifact.entropy_confidence):
        raise ValueError(
            "Offline ensemble confidence must be an exact alias of entropy_confidence."
        )
    expected_max_probability = artifact.probabilities.max(axis=1)
    if not np.allclose(
        artifact.max_probability,
        expected_max_probability,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("Offline ensemble max_probability does not match probabilities.")
    expected_total_variance = artifact.mean_probability_variance * classes
    if not np.allclose(
        artifact.total_probability_variance,
        expected_total_variance,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Offline ensemble total probability variance must equal class_count times mean variance."
        )
    expected_member_normalized_mi = _normalize_mi_by_member_count(
        artifact.mutual_information,
        context.member_count,
    )
    if not np.allclose(
        artifact.member_normalized_mi,
        expected_member_normalized_mi,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Offline ensemble member_normalized_mi does not match MI/log(M).")


def export_offline_ensemble_artifact(
    artifact: OfflineEnsembleArtifact,
    path: str | Path,
) -> Path:
    """Atomically export one validated offline ensemble/UE artifact."""

    _validate_ensemble_artifact(artifact)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Offline ensemble artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    context = artifact.context
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(OFFLINE_ENSEMBLE_SCHEMA_VERSION, dtype=np.int32),
                artifact_kind=np.asarray(OFFLINE_ENSEMBLE_KIND),
                method=np.asarray(context.method),
                method_protocol=np.asarray(context.method_protocol),
                task_id=np.asarray(context.task_id, dtype=np.int32),
                split=np.asarray(context.split),
                experiment_seed=np.asarray(context.experiment_seed, dtype=np.int64),
                member_ids=np.asarray(context.member_ids, dtype=np.int32),
                member_seeds=np.asarray(context.member_seeds, dtype=np.int64),
                member_count=np.asarray(context.member_count, dtype=np.int32),
                ensemble_mode=np.asarray(context.ensemble_mode),
                source_run_ids=np.asarray(context.source_run_ids),
                source_artifact_paths=np.asarray(artifact.source_artifact_paths),
                sample_ids=artifact.sample_ids,
                labels=artifact.labels,
                raw_class_ids=artifact.raw_class_ids,
                probabilities=artifact.probabilities,
                predictions=artifact.predictions,
                predictive_entropy=artifact.predictive_entropy,
                expected_member_entropy=artifact.expected_member_entropy,
                mutual_information=artifact.mutual_information,
                normalized_entropy=artifact.normalized_entropy,
                normalized_mi=artifact.normalized_mi,
                mean_probability_variance=artifact.mean_probability_variance,
                total_probability_variance=artifact.total_probability_variance,
                member_normalized_mi=artifact.member_normalized_mi,
                entropy_confidence=artifact.entropy_confidence,
                max_probability=artifact.max_probability,
                confidence=artifact.confidence,
                pairwise_disagreement=artifact.pairwise_disagreement,
            )
            handle.flush()
            os.fsync(handle.fileno())
        load_offline_ensemble_artifact(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _read_scalar(payload: Mapping[str, np.ndarray], key: str) -> object:
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError(f"Offline ensemble {key} must be a scalar.")
    return value.item()


def load_offline_ensemble_artifact(path: str | Path) -> OfflineEnsembleArtifact:
    """Load current or legacy ensemble artifacts with strict provenance checks."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing offline ensemble artifact: {path}")
    legacy_required = {
        "schema_version",
        "artifact_kind",
        "method",
        "method_protocol",
        "task_id",
        "split",
        "experiment_seed",
        "member_ids",
        "member_seeds",
        "source_run_ids",
        "source_artifact_paths",
        "sample_ids",
        "labels",
        "raw_class_ids",
        "probabilities",
        "predictions",
        "predictive_entropy",
        "expected_member_entropy",
        "mutual_information",
        "normalized_entropy",
        "normalized_mi",
        "mean_probability_variance",
        "confidence",
        "pairwise_disagreement",
    }
    schema_two_required = {
        "member_count",
        "total_probability_variance",
        "member_normalized_mi",
        "entropy_confidence",
        "max_probability",
    }
    schema_three_required = {"ensemble_mode"}
    with np.load(path, allow_pickle=False) as payload:
        if "schema_version" not in payload.files:
            raise ValueError("Offline ensemble artifact is missing keys: ['schema_version']")
        schema_version = int(_read_scalar(payload, "schema_version"))
        if schema_version not in {
            LEGACY_OFFLINE_ENSEMBLE_SCHEMA_VERSION,
            LEGACY_OFFLINE_ENSEMBLE_SCHEMA_VERSION_2,
            OFFLINE_ENSEMBLE_SCHEMA_VERSION,
        }:
            raise ValueError(f"Unsupported offline ensemble schema version: {schema_version}.")
        required = set(legacy_required)
        if schema_version >= LEGACY_OFFLINE_ENSEMBLE_SCHEMA_VERSION_2:
            required.update(schema_two_required)
        if schema_version >= OFFLINE_ENSEMBLE_SCHEMA_VERSION:
            required.update(schema_three_required)
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Offline ensemble artifact is missing keys: {missing}")
        kind = str(_read_scalar(payload, "artifact_kind"))
        if kind != OFFLINE_ENSEMBLE_KIND:
            raise ValueError(f"Unexpected offline ensemble artifact kind: {kind!r}.")
        member_ids = tuple(int(value) for value in payload["member_ids"].tolist())
        member_seeds = tuple(int(value) for value in payload["member_seeds"].tolist())
        source_run_ids = tuple(str(value) for value in payload["source_run_ids"].tolist())
        sources = tuple(str(value) for value in payload["source_artifact_paths"].tolist())
        if not all(
            len(values) == EXPECTED_MEMBER_COUNT
            for values in (member_ids, member_seeds, source_run_ids, sources)
        ):
            raise ValueError("Offline ensemble provenance must contain exactly three members.")
        member_count = (
            int(_read_scalar(payload, "member_count"))
            if schema_version >= OFFLINE_ENSEMBLE_SCHEMA_VERSION
            else len(member_ids)
        )
        probabilities = np.asarray(payload["probabilities"])
        raw_class_ids = np.asarray(payload["raw_class_ids"])
        mutual_information = np.asarray(payload["mutual_information"])
        normalized_entropy = np.asarray(payload["normalized_entropy"])
        mean_probability_variance = np.asarray(payload["mean_probability_variance"])
        if schema_version == LEGACY_OFFLINE_ENSEMBLE_SCHEMA_VERSION:
            # Schema 1 already stored every quantity required to derive the new
            # schema-2 views, so pilot artifacts do not need to be regenerated.
            entropy_confidence = 1.0 - normalized_entropy
            max_probability = probabilities.max(axis=1)
            total_probability_variance = (
                mean_probability_variance * raw_class_ids.shape[0]
            )
            member_normalized_mi = _normalize_mi_by_member_count(
                mutual_information,
                member_count,
            )
        else:
            entropy_confidence = np.asarray(payload["entropy_confidence"])
            max_probability = np.asarray(payload["max_probability"])
            total_probability_variance = np.asarray(
                payload["total_probability_variance"]
            )
            member_normalized_mi = np.asarray(payload["member_normalized_mi"])
        artifact = OfflineEnsembleArtifact(
            context=OfflineEnsembleContext(
                method=str(_read_scalar(payload, "method")),
                method_protocol=str(_read_scalar(payload, "method_protocol")),
                task_id=int(_read_scalar(payload, "task_id")),
                split=str(_read_scalar(payload, "split")),
                experiment_seed=int(_read_scalar(payload, "experiment_seed")),
                member_ids=member_ids,  # type: ignore[arg-type]
                member_seeds=member_seeds,  # type: ignore[arg-type]
                source_run_ids=source_run_ids,  # type: ignore[arg-type]
                member_count=member_count,
                ensemble_mode=(
                    str(_read_scalar(payload, "ensemble_mode"))
                    if schema_version >= OFFLINE_ENSEMBLE_SCHEMA_VERSION
                    else "seeded"
                ),
            ),
            sample_ids=np.asarray(payload["sample_ids"]),
            labels=np.asarray(payload["labels"]),
            raw_class_ids=raw_class_ids,
            probabilities=probabilities,
            predictions=np.asarray(payload["predictions"]),
            predictive_entropy=np.asarray(payload["predictive_entropy"]),
            expected_member_entropy=np.asarray(payload["expected_member_entropy"]),
            mutual_information=mutual_information,
            normalized_entropy=normalized_entropy,
            normalized_mi=np.asarray(payload["normalized_mi"]),
            mean_probability_variance=mean_probability_variance,
            total_probability_variance=total_probability_variance,
            member_normalized_mi=member_normalized_mi,
            entropy_confidence=entropy_confidence,
            max_probability=max_probability,
            confidence=np.asarray(payload["confidence"]),
            pairwise_disagreement=np.asarray(payload["pairwise_disagreement"]),
            source_artifact_paths=sources,  # type: ignore[arg-type]
        )
    _validate_ensemble_artifact(artifact)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate exactly three DDI-CIL member probability artifacts."
    )
    parser.add_argument(
        "--member-artifacts",
        nargs=EXPECTED_MEMBER_COUNT,
        required=True,
        type=Path,
        metavar=("MEMBER_0", "MEMBER_1", "MEMBER_2"),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["seeded", "stratified_3fold"],
        default=None,
        help="Defaults to the mode recorded in member artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members = [load_member_prediction_artifact(path) for path in args.member_artifacts]
    artifact = aggregate_member_predictions(
        members,
        source_artifact_paths=args.member_artifacts,
        ensemble_mode=args.mode,
    )
    output_path = export_offline_ensemble_artifact(artifact, args.out)
    print(
        f"exported={output_path} rows={artifact.row_count} "
        f"classes={artifact.class_count} task={artifact.context.task_id} "
        f"split={artifact.context.split} mode={artifact.context.ensemble_mode} "
        f"members={artifact.context.member_ids}",
        flush=True,
    )


if __name__ == "__main__":
    main()
