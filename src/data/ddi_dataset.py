"""Dataset helpers for descriptor-only DDI2025-CIL experiments."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None

    class Dataset:  # type: ignore[override]
        """Fallback Dataset base so the module can import without torch."""

        pass


DEFAULT_LABEL_COL = "class"
DEFAULT_META_COLS = [
    "drugid-drug_a",
    "drugid-drug_b",
    "drugname-drug_a",
    "drugname-drug_b",
    "drugsmiles-drug_a",
    "drugsmiles-drug_b",
]


@dataclass
class DDIBatchArrays:
    """Simple container for transformed arrays."""

    features: np.ndarray
    labels: np.ndarray
    metadata: dict[str, np.ndarray] | None = None


class DDIArrayDataset(Dataset):
    """In-memory array-backed dataset for static or continual runs."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        metadata: dict[str, np.ndarray] | None = None,
    ) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Features and labels must have the same number of rows.")
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.metadata = metadata

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "features": self.features[index],
            "label": self.labels[index],
        }
        if self.metadata is not None:
            item["metadata"] = {key: values[index] for key, values in self.metadata.items()}
        if torch is not None:
            item["features"] = torch.from_numpy(np.asarray(item["features"], dtype=np.float32))
            item["label"] = torch.tensor(int(item["label"]), dtype=torch.long)
        return item


def load_feature_columns(path: str | Path) -> list[str]:
    columns = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise ValueError(f"Invalid feature column JSON: {path}")
    return columns


def load_scaler_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "scaler_type" not in payload:
        raise ValueError(f"Invalid scaler payload: {path}")
    return payload


def apply_imputation(values: np.ndarray, scaler_payload: dict[str, Any]) -> np.ndarray:
    impute_values = np.asarray(scaler_payload["impute_values"], dtype=np.float64)
    nonfinite_mask = ~np.isfinite(values)
    if nonfinite_mask.any():
        values = np.where(nonfinite_mask, impute_values, values)
    return values


def transform_features(values: np.ndarray, scaler_payload: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = apply_imputation(values, scaler_payload)
    scaler_type = scaler_payload["scaler_type"]

    if scaler_type == "standard":
        mean = np.asarray(scaler_payload["mean"], dtype=np.float64)
        scale = np.asarray(scaler_payload["scale"], dtype=np.float64)
        transformed = (values - mean) / scale
    elif scaler_type == "robust":
        center = np.asarray(scaler_payload["center"], dtype=np.float64)
        scale = np.asarray(scaler_payload["scale"], dtype=np.float64)
        transformed = (values - center) / scale
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")

    return transformed.astype(np.float32, copy=False)


def _read_table(
    parquet_path: str | Path,
    columns: list[str],
    class_ids: Iterable[int] | None = None,
) -> pa.Table:
    filters = None
    if class_ids is not None:
        class_values = sorted({int(class_id) for class_id in class_ids})
        filters = [(DEFAULT_LABEL_COL, "in", class_values)]
    return pq.read_table(parquet_path, columns=columns, filters=filters)


def _read_filtered_frame(
    parquet_path: str | Path,
    columns: list[str],
    *,
    label_col: str,
    class_ids: Iterable[int] | None = None,
    max_rows: int | None = None,
    batch_size: int = 2048,
) -> pd.DataFrame:
    class_id_set = None if class_ids is None else {int(class_id) for class_id in class_ids}
    if class_id_set is None and max_rows is None:
        return _read_table(parquet_path, columns=columns, class_ids=None).to_pandas()

    parquet_file = pq.ParquetFile(parquet_path)
    frames: list[pd.DataFrame] = []
    collected = 0
    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        frame = pa.Table.from_batches([batch]).to_pandas()
        if class_id_set is not None:
            frame = frame[frame[label_col].isin(class_id_set)]
        if frame.empty:
            continue
        if max_rows is not None:
            remaining = max_rows - collected
            if remaining <= 0:
                break
            if frame.shape[0] > remaining:
                frame = frame.iloc[:remaining].copy()
        frames.append(frame)
        collected += frame.shape[0]
        if max_rows is not None and collected >= max_rows:
            break

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def load_split_arrays(
    parquet_path: str | Path,
    feature_columns: list[str],
    *,
    label_col: str = DEFAULT_LABEL_COL,
    class_ids: Iterable[int] | None = None,
    scaler_payload: dict[str, Any] | None = None,
    include_metadata: bool = False,
    meta_cols: list[str] | None = None,
    max_rows: int | None = None,
) -> DDIBatchArrays:
    """Load a split into memory using explicit feature columns.

    This is intended for controlled subsets such as static baseline or per-task slices.
    For very large training runs, add more streaming-oriented loaders later.
    """

    meta_cols = meta_cols or DEFAULT_META_COLS
    columns = feature_columns + [label_col]
    if include_metadata:
        columns += meta_cols

    frame = _read_filtered_frame(
        parquet_path,
        columns=columns,
        label_col=label_col,
        class_ids=class_ids,
        max_rows=max_rows,
    )
    features = frame[feature_columns].to_numpy(dtype=np.float64, copy=False)
    if scaler_payload is not None:
        features = transform_features(features, scaler_payload)
    else:
        features = features.astype(np.float32, copy=False)

    labels = frame[label_col].to_numpy(dtype=np.int64, copy=False)
    metadata: dict[str, np.ndarray] | None = None
    if include_metadata:
        metadata = {
            column: frame[column].to_numpy(copy=False)
            for column in meta_cols
            if column in frame.columns
        }
    return DDIBatchArrays(features=features, labels=labels, metadata=metadata)


def load_split_frame(
    parquet_path: str | Path,
    feature_columns: list[str],
    *,
    label_col: str = DEFAULT_LABEL_COL,
    class_ids: Iterable[int] | None = None,
    include_metadata: bool = False,
    meta_cols: list[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Return a pandas frame with explicit feature and label columns."""

    meta_cols = meta_cols or DEFAULT_META_COLS
    columns = feature_columns + [label_col]
    if include_metadata:
        columns += meta_cols
    return _read_filtered_frame(
        parquet_path,
        columns=columns,
        label_col=label_col,
        class_ids=class_ids,
        max_rows=max_rows,
    )


def load_unique_labels(
    parquet_path: str | Path,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    class_ids: Iterable[int] | None = None,
) -> list[int]:
    """Load sorted unique raw labels from a Parquet split.

    This is used when a run trains on a row-limited subset but still needs
    the full label space defined by the split.
    """

    frame = _read_filtered_frame(
        parquet_path,
        columns=[label_col],
        label_col=label_col,
        class_ids=class_ids,
        max_rows=None,
    )
    return sorted(frame[label_col].astype(int).unique().tolist())


def build_dataset(
    parquet_path: str | Path,
    feature_columns: list[str],
    *,
    label_col: str = DEFAULT_LABEL_COL,
    class_ids: Iterable[int] | None = None,
    scaler_payload: dict[str, Any] | None = None,
    include_metadata: bool = False,
    meta_cols: list[str] | None = None,
    max_rows: int | None = None,
) -> DDIArrayDataset:
    """Load arrays and wrap them in an in-memory dataset."""

    arrays = load_split_arrays(
        parquet_path=parquet_path,
        feature_columns=feature_columns,
        label_col=label_col,
        class_ids=class_ids,
        scaler_payload=scaler_payload,
        include_metadata=include_metadata,
        meta_cols=meta_cols,
        max_rows=max_rows,
    )
    return DDIArrayDataset(
        features=arrays.features,
        labels=arrays.labels,
        metadata=arrays.metadata,
    )
