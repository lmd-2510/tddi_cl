"""Dataset helpers for the original-split DDI2025-CIL workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

DEFAULT_LABEL_COL = "class"
DEFAULT_META_COLS = [
    "drugid-drug_a", "drugid-drug_b", "drugname-drug_a", "drugname-drug_b",
    "drugsmiles-drug_a", "drugsmiles-drug_b",
]


@dataclass
class DDIBatchArrays:
    features: np.ndarray
    labels: np.ndarray
    metadata: dict[str, np.ndarray] | None = None


class DDIArrayDataset(Dataset):
    def __init__(self, features, labels, metadata=None):
        if len(features) != len(labels):
            raise ValueError("Features and labels must have equal row counts.")
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.metadata = metadata

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, index):
        item: dict[str, Any] = {
            "features": torch.from_numpy(self.features[index]),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }
        if self.metadata is not None:
            item["metadata"] = {key: values[index] for key, values in self.metadata.items()}
        return item


def load_feature_columns(path: str | Path) -> list[str]:
    columns = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(columns, list) or not columns
            or not all(isinstance(column, str) and column for column in columns)
            or len(columns) != len(set(columns))):
        raise ValueError(f"Invalid feature column JSON: {path}")
    return columns


def load_class_counts(parquet_path, *, label_col=DEFAULT_LABEL_COL) -> dict[int, int]:
    labels = pq.read_table(parquet_path, columns=[label_col])[label_col].to_numpy()
    classes, counts = np.unique(labels.astype(np.int64, copy=False), return_counts=True)
    return {int(raw): int(count) for raw, count in zip(classes, counts, strict=True)}


def _read_filtered_frame(parquet_path, columns, *, label_col, class_ids=None, max_rows=None,
                         batch_size=2048):
    selected = None if class_ids is None else {int(value) for value in class_ids}
    frames, collected = [], 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(columns=columns, batch_size=batch_size):
        frame = pa.Table.from_batches([batch]).to_pandas()
        if selected is not None:
            frame = frame[frame[label_col].isin(selected)]
        if frame.empty:
            continue
        if max_rows is not None:
            remaining = max_rows - collected
            if remaining <= 0:
                break
            frame = frame.iloc[:remaining]
        frames.append(frame)
        collected += len(frame)
        if max_rows is not None and collected >= max_rows:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def load_split_frame(parquet_path, feature_columns, *, label_col=DEFAULT_LABEL_COL,
                     class_ids=None, include_metadata=False, meta_cols=None, max_rows=None):
    metadata = meta_cols or DEFAULT_META_COLS
    columns = [*feature_columns, label_col, *(metadata if include_metadata else [])]
    return _read_filtered_frame(parquet_path, columns, label_col=label_col,
                                class_ids=class_ids, max_rows=max_rows)


def load_split_arrays(parquet_path, feature_columns, *, label_col=DEFAULT_LABEL_COL,
                      class_ids=None, include_metadata=False, meta_cols=None, max_rows=None):
    metadata_columns = meta_cols or DEFAULT_META_COLS
    frame = load_split_frame(
        parquet_path, feature_columns, label_col=label_col, class_ids=class_ids,
        include_metadata=include_metadata, meta_cols=metadata_columns, max_rows=max_rows,
    )
    metadata = ({column: frame[column].to_numpy(copy=False) for column in metadata_columns}
                if include_metadata else None)
    return DDIBatchArrays(
        frame[feature_columns].to_numpy(dtype=np.float32, copy=False),
        frame[label_col].to_numpy(dtype=np.int64, copy=False), metadata,
    )


def load_unique_labels(parquet_path, *, label_col=DEFAULT_LABEL_COL, class_ids=None):
    selected = None if class_ids is None else sorted({int(value) for value in class_ids})
    table = pq.read_table(parquet_path, columns=[label_col],
                          filters=None if selected is None else [(label_col, "in", selected)])
    return sorted({int(value) for value in table[label_col].to_numpy()})


def build_dataset(parquet_path, feature_columns, **kwargs):
    arrays = load_split_arrays(parquet_path, feature_columns, **kwargs)
    return DDIArrayDataset(arrays.features, arrays.labels, arrays.metadata)
