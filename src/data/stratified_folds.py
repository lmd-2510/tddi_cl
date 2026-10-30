"""Deterministic stratified development folds for paper-compatible T-DDI runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold

from src.data.ddi_dataset import DDIBatchArrays, DEFAULT_LABEL_COL
from src.data.sample_identity import (
    DRUG_ID_A_COLUMN,
    DRUG_ID_B_COLUMN,
    build_stable_sample_ids,
)


FOLD_ARTIFACT_SCHEMA_VERSION = 1
FOLD_ARTIFACT_KIND = "ddi_cil_development_folds"
FOLD_ASSIGNMENT_FILENAME = "fold_assignments.parquet"
FOLD_MANIFEST_FILENAME = "fold_manifest.json"
FOLD_ASSIGNMENT_SCHEMA = pa.schema([
    pa.field("source_split", pa.string(), nullable=False),
    pa.field("source_row_index", pa.int64(), nullable=False),
    pa.field("sample_id", pa.string(), nullable=False),
    pa.field("raw_class_id", pa.int64(), nullable=False),
    pa.field("fold_id", pa.int16(), nullable=False),
])
FOLD_SPLIT_STRATEGY = {
    "name": "sklearn.model_selection.StratifiedKFold",
    "shuffle": True,
    "stratify_by": "raw_class_id",
    "input_order": "train_then_validation_source_row_order",
    "sample_identity": "ordered_drug_a_pipe_drug_b_v1",
}


def fold_file_sha256(path: str | Path) -> str:
    """Hash file bytes in bounded chunks, including row-order changes in Parquet."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_fold_source(path: str | Path) -> dict[str, Any]:
    """Record source provenance without reading the feature matrix into memory."""
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": fold_file_sha256(path),
        "row_count": pq.ParquetFile(path).metadata.num_rows,
    }


def _require_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest.")


def _validate_fold_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("Fold manifest must be an object.")
    required = {
        "schema_version", "artifact_kind", "fold_seed", "n_splits", "sources",
        "split_strategy", "member_to_validation_fold", "assignment_file",
        "assignment_sha256", "assignment_row_count", "created_at_utc", "creation_command",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Fold manifest missing fields: {missing}")
    version = _require_integer(manifest["schema_version"], "schema_version")
    if version != FOLD_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported fold schema_version: {version}.")
    if manifest["artifact_kind"] != FOLD_ARTIFACT_KIND:
        raise ValueError("Unexpected fold artifact_kind.")
    if _require_integer(manifest["n_splits"], "n_splits") != 3:
        raise ValueError("Fold artifact requires n_splits=3.")
    if _require_integer(manifest["fold_seed"], "fold_seed") > 2**32 - 1:
        raise ValueError("fold_seed must fit in uint32.")
    if manifest["split_strategy"] != FOLD_SPLIT_STRATEGY:
        raise ValueError("Unsupported fold split_strategy.")
    mapping = manifest["member_to_validation_fold"]
    if not isinstance(mapping, dict) or set(mapping) != {"0", "1", "2"}:
        raise ValueError("member_to_validation_fold must map members 0, 1, 2.")
    for member, fold in mapping.items():
        if _require_integer(fold, f"member_to_validation_fold.{member}") != int(member):
            raise ValueError("member_to_validation_fold must use member_id == fold_id.")
    if manifest["assignment_file"] != FOLD_ASSIGNMENT_FILENAME:
        raise ValueError("Unexpected assignment_file in fold manifest.")
    _require_sha256(manifest["assignment_sha256"], "assignment_sha256")
    _require_integer(manifest["assignment_row_count"], "assignment_row_count", minimum=1)
    sources = manifest["sources"]
    if not isinstance(sources, dict) or set(sources) != {"train", "validation", "test"}:
        raise ValueError("Fold sources must contain exactly train, validation, test.")
    for split, source in sources.items():
        if not isinstance(source, dict) or not {"path", "sha256", "row_count"} <= set(source):
            raise ValueError(f"Fold source {split} missing path/sha256/row_count.")
        if not isinstance(source["path"], str) or not source["path"].strip():
            raise ValueError(f"Fold source {split}.path must be non-empty.")
        _require_sha256(source["sha256"], f"sources.{split}.sha256")
        _require_integer(source["row_count"], f"sources.{split}.row_count", minimum=1)
    try:
        timestamp = datetime.fromisoformat(manifest["created_at_utc"])
        if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("not UTC")
    except (TypeError, ValueError) as error:
        raise ValueError("created_at_utc must be an ISO-8601 UTC timestamp.") from error
    if not isinstance(manifest["creation_command"], str):
        raise ValueError("creation_command must be a string.")


def validate_fold_artifact(
    assignments: pa.Table,
    manifest: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Validate structure/coverage without coercion or reordering.

    File-byte hashes are verified by load_fold_artifact. This structural check
    does not establish source label/ID agreement, stratification quality or
    test overlap; the independent data audit must check those against sources.
    """
    _validate_fold_manifest(manifest)
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            if key not in manifest or manifest[key] != expected:
                raise ValueError(f"Fold metadata mismatch: {key}.")
    if not isinstance(assignments, pa.Table):
        raise TypeError("Fold assignments must be a pyarrow.Table.")
    if assignments.column_names != FOLD_ASSIGNMENT_SCHEMA.names:
        raise ValueError("Fold assignment columns must match FOLD_ASSIGNMENT_SCHEMA.")
    for field in FOLD_ASSIGNMENT_SCHEMA:
        column = assignments[field.name]
        if column.type != field.type:
            raise ValueError(f"Fold assignment {field.name} dtype must be {field.type}.")
        if column.null_count:
            raise ValueError(f"Fold assignment {field.name} contains null values.")
    if assignments.num_rows != manifest["assignment_row_count"]:
        raise ValueError("Fold assignment_row_count mismatch.")
    splits = assignments["source_split"].to_numpy()
    if not np.isin(splits, ["train", "validation"]).all():
        raise ValueError("source_split must be train or validation; test is excluded.")
    indices = assignments["source_row_index"].to_numpy()
    for split in ("train", "validation"):
        selected = indices[splits == split]
        count = manifest["sources"][split]["row_count"]
        if len(selected) != count or not np.array_equal(np.sort(selected), np.arange(count)):
            raise ValueError(f"source_row_index coverage mismatch for {split}.")
    sample_ids = assignments["sample_id"].to_numpy()
    if any(not value.strip() for value in sample_ids):
        raise ValueError("sample_id must be non-empty.")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Fold assignment contains duplicate sample_id.")
    fold_ids = assignments["fold_id"].to_numpy()
    if not np.isin(fold_ids, [0, 1, 2]).all():
        raise ValueError("fold_id must be in [0, 2].")
    if set(fold_ids.tolist()) != {0, 1, 2}:
        raise ValueError("Fold assignment must contain all three folds.")


@dataclass(frozen=True)
class FoldArtifact:
    """Persisted partition with source row order and canonical Arrow dtypes."""

    assignments: pa.Table
    manifest: dict[str, Any]

    def to_lookup(self) -> StratifiedFoldAssignments:
        """Adapt validated artifacts to the existing runtime lookup interface."""
        validate_fold_artifact(self.assignments, self.manifest)
        ids = np.asarray(self.assignments["sample_id"].to_numpy(), dtype=np.str_)
        folds = self.assignments["fold_id"].to_numpy()
        order = np.argsort(ids, kind="stable")
        return StratifiedFoldAssignments(
            sorted_sample_ids=ids[order], sorted_fold_ids=folds[order],
            fold_count=self.manifest["n_splits"], seed=self.manifest["fold_seed"],
        )


def save_fold_artifact(
    outdir: str | Path,
    assignments: pa.Table,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    fold_seed: int,
    creation_command: str = "",
) -> FoldArtifact:
    """Save into a new directory; publish the manifest last as a commit marker.

    Each file uses same-directory staging + atomic replace. An interrupted save
    may leave an incomplete directory; it is never overwritten on retry. Source
    metadata is supplied by the caller (describe_fold_source can build it).
    """
    manifest = {
        "schema_version": FOLD_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": FOLD_ARTIFACT_KIND,
        "n_splits": 3,
        "fold_seed": fold_seed,
        "sources": {key: dict(value) for key, value in sources.items()},
        "split_strategy": dict(FOLD_SPLIT_STRATEGY),
        "member_to_validation_fold": {str(i): i for i in range(3)},
        "assignment_file": FOLD_ASSIGNMENT_FILENAME,
        # Replaced with the actual file digest before publication.
        "assignment_sha256": "0" * 64,
        "assignment_row_count": assignments.num_rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "creation_command": creation_command,
    }
    validate_fold_artifact(assignments, manifest)
    labels = assignments["raw_class_id"].to_numpy()
    folds = assignments["fold_id"].to_numpy()
    manifest["fold_summary"] = {}
    for fold_id in range(3):
        classes, counts = np.unique(labels[folds == fold_id], return_counts=True)
        manifest["fold_summary"][str(fold_id)] = {
            "row_count": int(counts.sum()),
            "class_counts": {
                str(int(label)): int(count) for label, count in zip(classes, counts, strict=True)
            },
        }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    assignment_tmp = outdir / f".assignment.{token}.tmp"
    manifest_tmp = outdir / f".manifest.{token}.tmp"
    try:
        with assignment_tmp.open("xb") as handle:
            pq.write_table(assignments, handle, compression="zstd")
            handle.flush()
            os.fsync(handle.fileno())
        manifest["assignment_sha256"] = fold_file_sha256(assignment_tmp)
        with manifest_tmp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(assignment_tmp, outdir / FOLD_ASSIGNMENT_FILENAME)
        os.replace(manifest_tmp, outdir / FOLD_MANIFEST_FILENAME)
    finally:
        assignment_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return FoldArtifact(assignments=assignments, manifest=manifest)


def load_fold_artifact(
    assignment_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    source_paths: Mapping[str, str | Path] | None = None,
) -> FoldArtifact:
    """Check assignment hash and schema; optionally verify relocated source files.

    With source_paths omitted this is an offline structural load, not a source
    audit. If supplied, all three sources are mandatory and checked by bytes and
    row counts rather than original absolute paths (to support the GPU server).
    """
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_fold_manifest(manifest)
    if fold_file_sha256(assignment_path) != manifest["assignment_sha256"]:
        raise ValueError("Fold assignment SHA256 mismatch.")
    assignments = pq.read_table(assignment_path)
    validate_fold_artifact(assignments, manifest, expected_metadata=expected_metadata)
    if source_paths is not None:
        if set(source_paths) != {"train", "validation", "test"}:
            raise ValueError("source_paths must contain train, validation, test.")
        for split, path in source_paths.items():
            expected = manifest["sources"][split]
            if fold_file_sha256(path) != expected["sha256"]:
                raise ValueError(f"Fold source SHA256 mismatch: {split}.")
            if pq.ParquetFile(path).metadata.num_rows != expected["row_count"]:
                raise ValueError(f"Fold source row_count mismatch: {split}.")
    return FoldArtifact(assignments=assignments, manifest=manifest)


@dataclass(frozen=True)
class StratifiedFoldAssignments:
    """Immutable sample-to-fold lookup built once from the full development set."""

    sorted_sample_ids: np.ndarray
    sorted_fold_ids: np.ndarray
    fold_count: int
    seed: int

    def lookup(self, sample_ids: np.ndarray) -> np.ndarray:
        sample_ids = np.asarray(sample_ids, dtype=np.str_)
        positions = np.searchsorted(self.sorted_sample_ids, sample_ids)
        valid = positions < self.sorted_sample_ids.shape[0]
        if np.any(valid):
            valid_indices = np.flatnonzero(valid)
            valid[valid_indices] = (
                self.sorted_sample_ids[positions[valid_indices]] == sample_ids[valid_indices]
            )
        if not np.all(valid):
            missing = sample_ids[~valid][:3].tolist()
            raise ValueError(f"Samples are absent from the frozen fold assignment: {missing}")
        return self.sorted_fold_ids[positions]


def build_stratified_fold_assignments(
    parquet_paths: Sequence[str | Path],
    *,
    fold_count: int = 3,
    seed: int = 42,
    label_col: str = DEFAULT_LABEL_COL,
) -> StratifiedFoldAssignments:
    """Match the paper's shuffled StratifiedKFold over train+validation."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two.")
    labels_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    columns = [label_col, DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN]
    for raw_path in parquet_paths:
        path = Path(raw_path)
        table = pq.read_table(path, columns=columns)
        labels = table[label_col].to_numpy().astype(np.int64, copy=False)
        metadata = {
            DRUG_ID_A_COLUMN: table[DRUG_ID_A_COLUMN].to_numpy(),
            DRUG_ID_B_COLUMN: table[DRUG_ID_B_COLUMN].to_numpy(),
        }
        labels_parts.append(labels)
        sample_parts.append(build_stable_sample_ids(metadata, labels.shape[0]))

    labels = np.concatenate(labels_parts)
    sample_ids = np.concatenate(sample_parts)
    if np.unique(sample_ids).shape[0] != sample_ids.shape[0]:
        raise ValueError("Development train+validation contains duplicate drug-pair IDs.")
    _, class_counts = np.unique(labels, return_counts=True)
    if class_counts.min(initial=fold_count) < fold_count:
        raise ValueError("Every class needs at least fold_count development samples.")

    fold_ids = np.full(labels.shape[0], -1, dtype=np.int16)
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    for fold_id, (_, held_out_indices) in enumerate(splitter.split(sample_ids, labels)):
        fold_ids[held_out_indices] = fold_id
    if np.any(fold_ids < 0):
        raise RuntimeError("Stratified fold assignment left unassigned samples.")

    order = np.argsort(sample_ids, kind="stable")
    return StratifiedFoldAssignments(
        sorted_sample_ids=sample_ids[order],
        sorted_fold_ids=fold_ids[order],
        fold_count=fold_count,
        seed=seed,
    )


def select_development_fold(
    parts: Sequence[DDIBatchArrays],
    assignments: StratifiedFoldAssignments,
    *,
    fold_id: int,
    held_out: bool,
    max_rows: int | None = None,
) -> DDIBatchArrays:
    """Concatenate development parts and select held-out or complementary rows."""

    if not 0 <= fold_id < assignments.fold_count:
        raise ValueError("fold_id is outside the configured fold range.")
    if not parts:
        raise ValueError("At least one development split is required.")
    if any(part.metadata is None for part in parts):
        raise ValueError("Fold selection requires drug-pair metadata.")

    features = np.concatenate([part.features for part in parts], axis=0)
    labels = np.concatenate([part.labels for part in parts], axis=0)
    metadata_keys = set(parts[0].metadata or {})
    if any(set(part.metadata or {}) != metadata_keys for part in parts[1:]):
        raise ValueError("Development metadata columns do not match across splits.")
    metadata = {
        key: np.concatenate([(part.metadata or {})[key] for part in parts])
        for key in metadata_keys
    }
    sample_ids = build_stable_sample_ids(metadata, labels.shape[0])
    assigned = assignments.lookup(sample_ids)
    mask = assigned == fold_id if held_out else assigned != fold_id
    indices = np.flatnonzero(mask)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive when supplied.")
        indices = indices[:max_rows]
    if indices.size == 0:
        role = "held-out" if held_out else "training"
        raise ValueError(f"Fold {fold_id} has no {role} rows after filtering.")
    return DDIBatchArrays(
        features=features[indices],
        labels=labels[indices],
        metadata={key: values[indices] for key, values in metadata.items()},
    )
