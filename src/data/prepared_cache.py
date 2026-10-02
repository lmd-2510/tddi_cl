"""Persistent, frozen-preprocessing feature caches.

The cache is deliberately a thin data-format adapter: it stores the same raw
labels and sample identity columns as the source split, but descriptor columns
already transformed by the immutable task-0 scaler.  It never fits a scaler
and it does not change fold assignment or task membership.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ddi_dataset import DEFAULT_LABEL_COL, transform_features
from src.data.sample_identity import DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN


CACHE_SCHEMA_VERSION = 1
CACHE_MANIFEST = "prepared_cache_manifest.json"
CACHE_SPLITS = ("train", "validation")
CACHE_META_COLUMNS = (DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN, "source_row_index", "source_split")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_columns_sha256(columns: Iterable[str]) -> str:
    payload = json.dumps(list(columns), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_split_path(root: str | Path, split: str) -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported prepared-cache split: {split!r}")
    return Path(root) / f"{split}.parquet"


def validate_cache_manifest(
    root: str | Path,
    *,
    feature_columns: list[str],
    scaler_path: str | Path,
    required_splits: Iterable[str] = CACHE_SPLITS,
    source_paths: dict[str, str | Path] | None = None,
) -> dict:
    """Validate a cache before it is used by a trainer."""

    root = Path(root)
    manifest_path = root / CACHE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing prepared-cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported prepared-cache schema version.")
    if manifest.get("feature_columns_sha256") != feature_columns_sha256(feature_columns):
        raise ValueError("Prepared-cache feature-column order does not match the run.")
    if manifest.get("scaler_sha256") != sha256_file(scaler_path):
        raise ValueError("Prepared-cache scaler does not match the run.")
    if manifest.get("policy") != "task0_standard_frozen":
        raise ValueError("Prepared cache must use task0_standard_frozen preprocessing.")
    for split in required_splits:
        path = cache_split_path(root, split)
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared-cache split: {path}")
        expected = manifest.get("splits", {}).get(split, {}).get("sha256")
        if expected and expected != sha256_file(path):
            raise ValueError(f"Prepared-cache split hash mismatch: {split}")
        if source_paths is not None and split in source_paths:
            expected_source = manifest.get("splits", {}).get(split, {}).get("source_sha256")
            if expected_source and expected_source != sha256_file(source_paths[split]):
                raise ValueError(f"Prepared-cache source hash mismatch: {split}")
    return manifest


def _batch_to_table(batch: pa.RecordBatch, feature_columns: list[str], scaler: dict, *, split: str, offset: int) -> pa.Table:
    source = pa.Table.from_batches([batch]).to_pandas()
    raw = source[feature_columns].to_numpy(dtype=np.float64, copy=False)
    transformed = transform_features(raw, scaler).astype(np.float32, copy=False)
    payload = {
        column: pa.array(transformed[:, index], type=pa.float32())
        for index, column in enumerate(feature_columns)
    }
    payload[DEFAULT_LABEL_COL] = pa.array(source[DEFAULT_LABEL_COL].to_numpy(dtype=np.int64, copy=False), type=pa.int64())
    payload[DRUG_ID_A_COLUMN] = pa.array(source[DRUG_ID_A_COLUMN].astype(str).to_numpy(), type=pa.string())
    payload[DRUG_ID_B_COLUMN] = pa.array(source[DRUG_ID_B_COLUMN].astype(str).to_numpy(), type=pa.string())
    payload["source_row_index"] = pa.array(np.arange(offset, offset + len(source), dtype=np.int64), type=pa.int64())
    payload["source_split"] = pa.array([split] * len(source), type=pa.string())
    return pa.table(payload)


def materialize_split(
    source_path: str | Path,
    destination: str | Path,
    *,
    feature_columns: list[str],
    scaler: dict,
    split: str,
    batch_size: int = 2048,
) -> int:
    """Transform one source Parquet once and write a model-ready cache split."""

    source_path = Path(source_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [*feature_columns, DEFAULT_LABEL_COL, DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN]
    parquet = pq.ParquetFile(source_path)
    writer = None
    offset = 0
    try:
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            table = _batch_to_table(batch, feature_columns, scaler, split=split, offset=offset)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            offset += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if offset == 0:
        raise ValueError(f"Prepared-cache source is empty: {source_path}")
    return offset
