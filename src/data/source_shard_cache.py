"""Immutable raw task/fold shards for one-pass Parquet preprocessing.

The common task cache is materialised on demand.  This module provides the
lower-level source cache used to avoid scanning the large descriptor Parquets
once per task: train and validation sources are read once and partitioned by
raw task id and frozen fold id.  No scaler is fitted and no model state is
stored here.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ddi_dataset import DEFAULT_LABEL_COL, DDIBatchArrays, prepare_development_fold_context
from src.data.stratified_folds import fold_file_sha256


SOURCE_SHARD_KIND = "ddi_cil_raw_source_shard_cache"
SOURCE_SHARD_SCHEMA_VERSION = 1
SOURCE_SHARD_MANIFEST = "manifest.json"
METADATA_COLUMNS = (
    "sample_id", "source_split", "source_row_index", "source_path", "fold_id",
    "drugid-drug_a", "drugid-drug_b",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_map(task_file: str | Path) -> tuple[list[dict[str, Any]], dict[int, int]]:
    payload = json.loads(Path(task_file).read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task file must contain a non-empty tasks list.")
    mapping: dict[int, int] = {}
    normalised: list[dict[str, Any]] = []
    for expected_id, task in enumerate(tasks):
        if int(task.get("task_id", expected_id)) != expected_id:
            raise ValueError("Task IDs must be contiguous starting at zero.")
        classes = [int(value) for value in task["classes"]]
        for raw in classes:
            if raw in mapping:
                raise ValueError(f"Raw class appears in multiple tasks: {raw}")
            mapping[raw] = expected_id
        normalised.append({"task_id": expected_id, "classes": classes})
    return normalised, mapping


def _table_for_rows(
    batch: pa.RecordBatch,
    *,
    selected: np.ndarray,
    feature_columns: list[str],
    metadata: dict[str, np.ndarray],
    labels: np.ndarray,
) -> pa.Table:
    table = pa.Table.from_batches([batch]).take(pa.array(selected, type=pa.int64()))
    columns: dict[str, pa.Array] = {name: table[name] for name in feature_columns}
    columns[DEFAULT_LABEL_COL] = pa.array(labels[selected].astype(np.int64, copy=False), type=pa.int64())
    for name in METADATA_COLUMNS:
        values = metadata[name][selected]
        if name == "fold_id":
            columns[name] = pa.array(values.astype(np.int16, copy=False), type=pa.int16())
        elif name == "source_row_index":
            columns[name] = pa.array(values.astype(np.int64, copy=False), type=pa.int64())
        else:
            columns[name] = pa.array([str(value) for value in values], type=pa.string())
    return pa.table(columns)


def build_source_shard_cache(
    *,
    root: str | Path,
    assignments: str | Path,
    fold_manifest: str | Path,
    train: str | Path,
    validation: str | Path,
    test: str | Path,
    task_file: str | Path,
    feature_columns: Iterable[str],
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Read train/validation descriptor files once and write raw shards."""

    root = Path(root).resolve()
    manifest_path = root / SOURCE_SHARD_MANIFEST
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Source-shard output is non-empty without a manifest: {root}")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    columns = [str(value) for value in feature_columns]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be non-empty and unique")
    tasks, class_to_task = _task_map(task_file)
    context = prepare_development_fold_context(
        assignments, fold_manifest,
        source_paths={"train": train, "validation": validation, "test": test},
    )
    context_manifest = context.manifest
    fold_count = int(context_manifest["n_splits"])
    source_paths = {"train": Path(train).resolve(), "validation": Path(validation).resolve()}
    root.mkdir(parents=True, exist_ok=True)
    writers: dict[Path, pq.ParquetWriter] = {}
    row_counts: dict[str, int] = {}
    try:
        for source_index, source_split in enumerate(("train", "validation")):
            path = source_paths[source_split]
            rows = context._rows[source_index]
            labels = rows["raw_class_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            folds = rows["fold_id"].to_numpy(zero_copy_only=False).astype(np.int16, copy=False)
            task_ids = np.asarray([class_to_task[int(value)] for value in labels], dtype=np.int16)
            metadata = {
                "sample_id": rows["sample_id"].to_numpy(zero_copy_only=False),
                "source_split": np.full(len(labels), source_split, dtype=object),
                "source_row_index": rows["source_row_index"].to_numpy(zero_copy_only=False),
                "source_path": np.full(len(labels), str(path), dtype=object),
                "fold_id": folds,
                "drugid-drug_a": rows["drugid-drug_a"].to_numpy(zero_copy_only=False),
                "drugid-drug_b": rows["drugid-drug_b"].to_numpy(zero_copy_only=False),
            }
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=batch_size):
                end = offset + batch.num_rows
                if end > len(labels):
                    raise ValueError(f"Source row coverage changed while reading {path}")
                for task_id in range(len(tasks)):
                    for fold_id in range(fold_count):
                        selected = np.flatnonzero((task_ids[offset:end] == task_id) & (folds[offset:end] == fold_id))
                        if not len(selected):
                            continue
                        table = _table_for_rows(
                            batch, selected=selected, feature_columns=columns,
                            metadata={key: value[offset:end] for key, value in metadata.items()},
                            labels=labels[offset:end],
                        )
                        target = root / source_split / f"task_{task_id}" / f"fold_{fold_id}.parquet"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        writer = writers.get(target)
                        if writer is None:
                            writer = pq.ParquetWriter(target, table.schema, compression="zstd")
                            writers[target] = writer
                        writer.write_table(table)
                        key = target.relative_to(root).as_posix()
                        row_counts[key] = row_counts.get(key, 0) + table.num_rows
                offset = end
            if offset != len(labels):
                raise ValueError(f"Source row coverage mismatch: {path}")
    finally:
        for writer in writers.values():
            writer.close()
    shards = {
        key: {"path": key, "rows": rows, "sha256": _sha256_file(root / key)}
        for key, rows in sorted(row_counts.items())
    }
    manifest = {
        "kind": SOURCE_SHARD_KIND,
        "schema_version": SOURCE_SHARD_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_file_sha256": fold_file_sha256(task_file),
        "feature_columns": columns,
        "fold_count": fold_count,
        "fold_manifest_sha256": context.manifest_sha256,
        "assignment_sha256": context_manifest.get("assignment_sha256"),
        "sources": {
            split: {"path": str(path), "sha256": _sha256_file(path)}
            for split, path in source_paths.items()
        },
        "tasks": tasks,
        "shards": shards,
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def load_source_shard(
    root: str | Path,
    *,
    member_id: int,
    task_id: int,
    validation_fold: int,
    role: str,
    feature_columns: Iterable[str],
) -> DDIBatchArrays:
    """Load one task for a member from raw fold shards."""

    if role not in {"train", "validation"}:
        raise ValueError("role must be train or validation")
    root = Path(root).resolve()
    manifest = json.loads((root / SOURCE_SHARD_MANIFEST).read_text(encoding="utf-8"))
    fold_count = int(manifest["fold_count"])
    if not 0 <= int(validation_fold) < fold_count:
        raise ValueError("validation_fold is outside the shard manifest")
    columns = [str(value) for value in feature_columns]
    parts: list[DDIBatchArrays] = []
    folds = [int(validation_fold)] if role == "validation" else [f for f in range(fold_count) if f != int(validation_fold)]
    for source_split in ("train", "validation"):
        for fold_id in folds:
            path = root / source_split / f"task_{int(task_id)}" / f"fold_{fold_id}.parquet"
            if not path.is_file():
                continue
            table = pq.read_table(path)
            missing = [name for name in [*columns, DEFAULT_LABEL_COL, *METADATA_COLUMNS] if name not in table.column_names]
            if missing:
                raise ValueError(f"Source shard is missing columns: {missing}")
            parts.append(DDIBatchArrays(
                features=np.column_stack([table[name].to_numpy(zero_copy_only=False) for name in columns]).astype(np.float64, copy=False),
                labels=table[DEFAULT_LABEL_COL].to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
                metadata={name: table[name].to_numpy(zero_copy_only=False) for name in METADATA_COLUMNS},
            ))
    if not parts:
        return DDIBatchArrays(
            features=np.empty((0, len(columns)), dtype=np.float64),
            labels=np.empty((0,), dtype=np.int64),
            metadata={name: np.empty((0,), dtype=np.str_) for name in METADATA_COLUMNS},
        )
    metadata_keys = parts[0].metadata or {}
    return DDIBatchArrays(
        features=np.concatenate([part.features for part in parts]),
        labels=np.concatenate([part.labels for part in parts]),
        metadata={key: np.concatenate([(part.metadata or {})[key] for part in parts]) for key in metadata_keys},
    )


def validate_source_shard_cache(
    root: str | Path,
    *,
    task_file: str | Path,
    feature_columns: Iterable[str],
    fold_manifest_sha256: str | None,
    assignment_sha256: str | None = None,
    source_paths: dict[str, str | Path],
) -> dict[str, Any]:
    """Validate provenance before a trainer consumes raw source shards."""

    root = Path(root).resolve()
    manifest_path = root / SOURCE_SHARD_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source-shard manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    differences: list[str] = []
    if manifest.get("kind") != SOURCE_SHARD_KIND:
        differences.append("kind")
    if int(manifest.get("schema_version", -1)) != SOURCE_SHARD_SCHEMA_VERSION:
        differences.append("schema_version")
    if manifest.get("task_file_sha256") != fold_file_sha256(task_file):
        differences.append("task_file_sha256")
    if manifest.get("feature_columns") != [str(value) for value in feature_columns]:
        differences.append("feature_columns")
    if fold_manifest_sha256 is not None and manifest.get("fold_manifest_sha256") != str(fold_manifest_sha256):
        differences.append("fold_manifest_sha256")
    if assignment_sha256 is not None and manifest.get("assignment_sha256") != str(assignment_sha256):
        differences.append("assignment_sha256")
    for split, path in sorted(source_paths.items()):
        expected = manifest.get("sources", {}).get(split, {}).get("sha256")
        if expected != _sha256_file(path):
            differences.append(f"source:{split}")
    if differences:
        raise ValueError(f"Source-shard cache provenance mismatch: {sorted(set(differences))}")
    return manifest
