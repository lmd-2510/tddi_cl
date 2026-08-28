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

from src.eval.member_predictions import (  # noqa: E402
    MemberPredictionArtifact,
    load_member_prediction_artifact,
)


OFFLINE_ENSEMBLE_SCHEMA_VERSION = 1
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


def aggregate_member_predictions(
    artifacts: Sequence[MemberPredictionArtifact],
    *,
    source_artifact_paths: Sequence[str | Path] | None = None,
) -> OfflineEnsembleArtifact:
    """Validate alignment and aggregate exactly three members by mean probability."""

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

    sample_ids = _require_equal_array(artifacts, "sample_ids")
    labels = _require_equal_array(artifacts, "labels")
    raw_class_ids = _require_equal_array(artifacts, "raw_class_ids")
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
    if not np.isfinite(member_probabilities).all():
        raise ValueError("Member probabilities contain non-finite values.")
    if not np.allclose(member_probabilities.sum(axis=2), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("Member probability rows must sum to one before aggregation.")

    # The ensemble definition is the arithmetic mean of member posteriors. Raw
    # logits are deliberately not read or averaged here.
    mean_probabilities64 = member_probabilities.mean(axis=0)
    predictive_entropy = _entropy(mean_probabilities64)
    member_entropy = _entropy(member_probabilities)
    expected_member_entropy = member_entropy.mean(axis=0)
    mutual_information = np.maximum(
        predictive_entropy - expected_member_entropy,
        0.0,
    )
    class_count = raw_class_ids.shape[0]
    if class_count <= 1:
        normalized_entropy = np.zeros_like(predictive_entropy)
        normalized_mi = np.zeros_like(mutual_information)
    else:
        normalization = float(np.log(class_count))
        normalized_entropy = np.clip(predictive_entropy / normalization, 0.0, 1.0)
        normalized_mi = np.clip(mutual_information / normalization, 0.0, 1.0)
    confidence = 1.0 - normalized_entropy
    mean_probability_variance = member_probabilities.var(axis=0).mean(axis=1)
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
            split=split,
            experiment_seed=experiment_seed,
            member_ids=member_ids,  # type: ignore[arg-type]
            member_seeds=tuple(  # type: ignore[arg-type]
                int(member.context.member_seed) for member in artifacts
            ),
            source_run_ids=tuple(  # type: ignore[arg-type]
                member.context.run_id for member in artifacts
            ),
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
    for name in ("normalized_entropy", "normalized_mi", "confidence", "pairwise_disagreement"):
        values = np.asarray(getattr(artifact, name))
        if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
            raise ValueError(f"Offline ensemble {name} must lie in [0, 1].")
    if not np.allclose(
        artifact.confidence,
        1.0 - artifact.normalized_entropy,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Offline ensemble confidence must equal 1-normalized_entropy.")


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
    """Load and validate a versioned offline ensemble artifact."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing offline ensemble artifact: {path}")
    required = {
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
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Offline ensemble artifact is missing keys: {missing}")
        schema_version = int(_read_scalar(payload, "schema_version"))
        kind = str(_read_scalar(payload, "artifact_kind"))
        if schema_version != OFFLINE_ENSEMBLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported offline ensemble schema version: {schema_version}.")
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
            ),
            sample_ids=np.asarray(payload["sample_ids"]),
            labels=np.asarray(payload["labels"]),
            raw_class_ids=np.asarray(payload["raw_class_ids"]),
            probabilities=np.asarray(payload["probabilities"]),
            predictions=np.asarray(payload["predictions"]),
            predictive_entropy=np.asarray(payload["predictive_entropy"]),
            expected_member_entropy=np.asarray(payload["expected_member_entropy"]),
            mutual_information=np.asarray(payload["mutual_information"]),
            normalized_entropy=np.asarray(payload["normalized_entropy"]),
            normalized_mi=np.asarray(payload["normalized_mi"]),
            mean_probability_variance=np.asarray(payload["mean_probability_variance"]),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members = [load_member_prediction_artifact(path) for path in args.member_artifacts]
    artifact = aggregate_member_predictions(
        members,
        source_artifact_paths=args.member_artifacts,
    )
    output_path = export_offline_ensemble_artifact(artifact, args.out)
    print(
        f"exported={output_path} rows={artifact.row_count} "
        f"classes={artifact.class_count} task={artifact.context.task_id} "
        f"split={artifact.context.split} members={artifact.context.member_ids}",
        flush=True,
    )


if __name__ == "__main__":
    main()
