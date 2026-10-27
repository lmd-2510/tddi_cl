"""Validated access to the original DDI train/validation/test split.

This module is intentionally independent from the legacy stratified-fold
implementation.  It never concatenates train and validation and it only
materialises rows whose raw classes are explicitly requested by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ddi_dataset import DDIBatchArrays, DEFAULT_LABEL_COL
from src.data.sample_identity import DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN, build_stable_sample_ids


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class OriginalSplitContext:
    paths: Mapping[str, Path]
    source_info: Mapping[str, dict]
    sample_ids: Mapping[str, np.ndarray]
    labels: Mapping[str, np.ndarray]
    stamps: Mapping[str, tuple[int, int, int]]
    label_col: str = DEFAULT_LABEL_COL

    def assert_unchanged(self) -> None:
        for split, path in self.paths.items():
            if _stamp(path) != self.stamps[split]:
                raise ValueError(f"Original {split} source changed during the run: {path}")


def prepare_original_split_context(
    *, train: str | Path, validation: str | Path, test: str | Path,
    label_col: str = DEFAULT_LABEL_COL,
) -> OriginalSplitContext:
    """Hash and validate the three original, mutually exclusive source files."""
    paths = {name: Path(value).resolve() for name, value in {
        "train": train, "validation": validation, "test": test,
    }.items()}
    if len(set(paths.values())) != 3:
        raise ValueError("Original train, validation and test paths must be distinct.")
    ids: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    source_info: dict[str, dict] = {}
    stamps = {split: _stamp(path) for split, path in paths.items()}
    for split, path in paths.items():
        table = pq.read_table(path, columns=[DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN, label_col])
        raw_labels = table[label_col]
        if raw_labels.null_count or not pa.types.is_integer(raw_labels.type):
            raise ValueError(f"{split} labels must be non-null integers.")
        labels[split] = raw_labels.cast(pa.int64(), safe=True).to_numpy()
        ids[split] = build_stable_sample_ids(
            {DRUG_ID_A_COLUMN: table[DRUG_ID_A_COLUMN].to_numpy(),
             DRUG_ID_B_COLUMN: table[DRUG_ID_B_COLUMN].to_numpy()}, table.num_rows,
        )
        if len(np.unique(ids[split])) != len(ids[split]):
            raise ValueError(f"Duplicate sample identity inside original {split} split.")
        source_info[split] = {
            "path": str(path), "sha256": sha256_file(path),
            "row_count": int(table.num_rows),
            "class_count": int(len(np.unique(labels[split]))),
        }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if np.intersect1d(ids[left], ids[right]).size:
            raise ValueError(f"Original split leakage: {left} and {right} share sample IDs.")
    context = OriginalSplitContext(paths, source_info, ids, labels, stamps, label_col)
    context.assert_unchanged()
    return context


def load_original_split_arrays(
    context: OriginalSplitContext, feature_columns: list[str], *, split: str,
    class_ids: Iterable[int], include_metadata: bool = True,
) -> DDIBatchArrays:
    """Load only requested raw classes from exactly one original source."""
    if split not in context.paths:
        raise ValueError("split must be train, validation or test.")
    classes = sorted({int(value) for value in class_ids})
    if not classes:
        raise ValueError("class_ids must not be empty.")
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns must be non-empty and unique.")
    context.assert_unchanged()
    columns = [*feature_columns, context.label_col, DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN]
    table = pq.read_table(
        context.paths[split], columns=columns,
        filters=[(context.label_col, "in", classes)],
    )
    values = np.column_stack([
        table[name].to_numpy(zero_copy_only=False) for name in feature_columns
    ]).astype(np.float64, copy=False)
    labels = table[context.label_col].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    if not len(labels) or set(np.unique(labels)) - set(classes):
        raise ValueError(f"Original {split} task selection is empty or contains unexpected classes.")
    metadata = None
    if include_metadata:
        drug_a = table[DRUG_ID_A_COLUMN].to_numpy(zero_copy_only=False)
        drug_b = table[DRUG_ID_B_COLUMN].to_numpy(zero_copy_only=False)
        metadata = {
            "sample_id": build_stable_sample_ids(
                {DRUG_ID_A_COLUMN: drug_a, DRUG_ID_B_COLUMN: drug_b}, len(labels)),
            "source_split": np.full(len(labels), split, dtype=np.str_),
            DRUG_ID_A_COLUMN: drug_a,
            DRUG_ID_B_COLUMN: drug_b,
        }
    context.assert_unchanged()
    return DDIBatchArrays(values, labels, metadata)
