"""Dataset helpers for descriptor-only DDI2025-CIL experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.sample_identity import (
    DRUG_ID_A_COLUMN,
    DRUG_ID_B_COLUMN,
    build_stable_sample_ids,
)

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



def load_class_counts(
    parquet_path: str | Path,
    *,
    label_col: str = DEFAULT_LABEL_COL,
) -> dict[int, int]:
    """Count each raw class in a full split while reading only its label column."""

    labels = pq.read_table(parquet_path, columns=[label_col])[label_col].to_numpy()
    class_ids, counts = np.unique(labels.astype(np.int64, copy=False), return_counts=True)
    return {
        int(class_id): int(count)
        for class_id, count in zip(class_ids, counts, strict=True)
    }




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
        include_metadata=include_metadata,
        meta_cols=meta_cols,
        max_rows=max_rows,
    )
    return DDIArrayDataset(
        features=arrays.features,
        labels=arrays.labels,
        metadata=arrays.metadata,
    )


def _fold_file_stamp(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class DevelopmentFoldContext:
    """Run-local validated partition, not a serializable/resumable validation cache.

    Create with ``prepare_development_fold_context`` at EVERY run/resume entry.
    Only projected identity/label/assignment columns are cached, never features.
    Files must remain immutable for the context lifetime. Cheap stat checks catch
    changes between reads; they do not replace the mandatory startup byte hashes.
    """

    _source_paths: tuple[Path, Path]
    _rows: tuple[pa.Table, pa.Table]
    _file_stamps: tuple[tuple[Path, tuple[int, int, int, int, int]], ...]
    _manifest_json: str
    manifest_sha256: str
    label_col: str

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a copy so callers cannot mutate the validated member mapping."""
        return json.loads(self._manifest_json)

    def assert_unchanged(self) -> None:
        for path, expected in self._file_stamps:
            try:
                unchanged = _fold_file_stamp(path) == expected
            except OSError as error:
                raise ValueError(f"Fold context file unavailable: {path}") from error
            if not unchanged:
                raise ValueError(
                    f"Fold context file changed: {path}; recreate and validate context."
                )


def prepare_development_fold_context(
    assignment_path: str | Path,
    manifest_path: str | Path,
    *,
    source_paths: Mapping[str, str | Path],
    label_col: str = DEFAULT_LABEL_COL,
    expected_metadata: Mapping[str, Any] | None = None,
) -> DevelopmentFoldContext:
    """Validate frozen assignments against source bytes, coverage and actual rows.

    ``source_paths`` requires train/validation/test, including relocated files.
    Test is hashed and read ONLY for drug IDs to rule out development/test overlap;
    its descriptors and labels never enter this loader's returned arrays.
    ``expected_metadata`` can pin assignment_sha256/fold_seed/sources from a run
    or checkpoint. Do not restore this context from a checkpoint: call again.
    """
    # Lazy import: stratified_folds also supports the legacy DDIBatchArrays API.
    from src.data.stratified_folds import fold_file_sha256, load_fold_artifact
    from src.data.sample_identity import (
        DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN, build_stable_sample_ids,
    )

    if set(source_paths) != {"train", "validation", "test"}:
        raise ValueError("source_paths must contain exactly train, validation, test.")
    paths = {split: Path(path).resolve() for split, path in source_paths.items()}
    if len(set(paths.values())) != 3:
        raise ValueError("Fold sources must use distinct source paths.")
    assignment_path, manifest_path = Path(assignment_path).resolve(), Path(manifest_path).resolve()
    watched = (assignment_path, manifest_path, *(paths[s] for s in ("train", "validation", "test")))
    stamps = tuple((path, _fold_file_stamp(path)) for path in watched)
    artifact = load_fold_artifact(
        assignment_path, manifest_path, source_paths=paths,
        expected_metadata=expected_metadata,
    )
    manifest_hash = fold_file_sha256(manifest_path)
    rows = []
    development_ids = []
    id_columns = [DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN]
    for split in ("train", "validation"):
        source = pq.read_table(paths[split], columns=[*id_columns, label_col])
        label_array = source[label_col]
        if label_array.null_count or not pa.types.is_integer(label_array.type):
            raise ValueError(f"Source {split} labels must be non-null integers.")
        try:
            labels = label_array.cast(pa.int64(), safe=True).to_numpy()
        except pa.ArrowInvalid as error:
            raise ValueError(f"Source {split} labels must fit int64.") from error
        ids = build_stable_sample_ids(
            {column: source[column].to_numpy() for column in id_columns}, source.num_rows,
        )
        assignment_splits = artifact.assignments["source_split"].to_numpy()
        part = artifact.assignments.filter(pa.array(assignment_splits == split))
        part = part.sort_by([("source_row_index", "ascending")])
        if not np.array_equal(ids, part["sample_id"].to_numpy()):
            raise ValueError(f"Source/assignment sample ID agreement mismatch: {split}.")
        if not np.array_equal(labels, part["raw_class_id"].to_numpy()):
            raise ValueError(f"Source/assignment label agreement mismatch: {split}.")
        for column in id_columns:
            part = part.append_column(column, source[column])
        rows.append(part)
        development_ids.append(ids)
    all_ids = np.concatenate(development_ids)
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("Duplicate development sample IDs across sources.")
    test = pq.read_table(paths["test"], columns=id_columns)
    test_ids = build_stable_sample_ids(
        {column: test[column].to_numpy() for column in id_columns}, test.num_rows,
    )
    if np.intersect1d(all_ids, test_ids).size:
        raise ValueError("Development/test sample ID overlap; test cannot enter train/validation.")
    context = DevelopmentFoldContext(
        _source_paths=(paths["train"], paths["validation"]),
        _rows=(rows[0], rows[1]), _file_stamps=stamps,
        _manifest_json=json.dumps(artifact.manifest, sort_keys=True),
        manifest_sha256=manifest_hash, label_col=label_col,
    )
    # Catch replacement/writes during hashing or identity reads as well.
    context.assert_unchanged()
    return context


def iter_development_fold_arrays(
    context: DevelopmentFoldContext,
    feature_columns: list[str],
    *,
    role: str,
    member_id: int,
    validation_fold: int,
    class_ids: Iterable[int] | None = None,
    batch_size: int = 2048,
) -> Iterator[DDIBatchArrays]:
    """Stream raw descriptors in train-source then validation-source row order.

    Train selects the complement of the member's held-out fold. Class IDs are
    RAW IDs (not head-column positions); None selects all classes, [] selects none.
    Metadata always contains sample_id, source_split, source_row_index (original
    zero-based index, never reset after filtering), source_path, fold_id and both
    drug IDs. Empty selections return aligned zero-row arrays.

    Raw numerical values are returned as float64; nulls become NaN. No scaler,
    imputation, clipping or row dropping occurs, including for nonfinite values.
    Future preprocessing owns the explicit nonfinite policy. This function does
    not construct folds, shuffle rows, or load descriptors from the test source.
    """
    from src.data.sample_identity import DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN

    context.assert_unchanged()
    if role not in {"train", "validation"}:
        raise ValueError("role must be train or validation, never test.")
    mapping = context.manifest["member_to_validation_fold"]
    if type(member_id) is not int or str(member_id) not in mapping:
        raise ValueError("member_id must be 0, 1 or 2.")
    if type(validation_fold) is not int or validation_fold != mapping[str(member_id)]:
        raise ValueError("validation_fold does not match manifest member mapping.")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    reserved = {*DEFAULT_META_COLS, context.label_col, "sample_id", "source_row_index",
                "source_split", "source_path", "fold_id", "raw_class_id"}
    if (not feature_columns or not all(isinstance(c, str) for c in feature_columns)
            or len(set(feature_columns)) != len(feature_columns)
            or set(feature_columns) & reserved):
        raise ValueError("feature_columns must be unique descriptor names, excluding labels/metadata.")
    class_values = None
    if class_ids is not None:
        class_values = list(class_ids)
        if any(isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer))
               or not -(2**63) <= int(c) < 2**63 for c in class_values):
            raise ValueError("class_ids must contain raw int64 class IDs.")
        class_values = np.asarray(class_values, dtype=np.int64)

    selected_parts = []
    masks = []
    # Validate both feature schemas even if a source has no selected rows.
    for path, rows in zip(context._source_paths, context._rows, strict=True):
        schema = pq.ParquetFile(path).schema_arrow
        for column in feature_columns:
            if schema.names.count(column) != 1:
                raise ValueError(f"Missing or ambiguous descriptor {column}: {path}")
            dtype = schema.field(column).type
            if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
                raise ValueError(f"Descriptor {column} must be numeric: {path}")
        folds = rows["fold_id"].to_numpy()
        mask = folds == validation_fold if role == "validation" else folds != validation_fold
        if class_values is not None:
            mask &= np.isin(rows["raw_class_id"].to_numpy(), class_values)
        masks.append(mask)
        part = rows.filter(pa.array(mask))
        selected_parts.append(part.append_column("source_path", pa.array([str(path)] * part.num_rows, type=pa.string())))
    metadata_columns = ["sample_id", "source_split", "source_row_index", "source_path",
                        "fold_id", DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN]
    written = 0
    for path, mask, selected in zip(context._source_paths, masks, selected_parts, strict=True):
        if not mask.any():
            continue
        offset = 0
        selected_offset = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=feature_columns, batch_size=batch_size):
            keep = mask[offset:offset + batch.num_rows]
            offset += batch.num_rows
            count = int(keep.sum())
            if count:
                filtered = batch.filter(pa.array(keep))
                features = np.empty((count, len(feature_columns)), dtype=np.float64)
                for index, column in enumerate(feature_columns):
                    features[:, index] = filtered.column(column).to_numpy(zero_copy_only=False)
                part = selected.slice(selected_offset, count)
                selected_offset += count
                written += count
                context.assert_unchanged()
                yield DDIBatchArrays(
                    features=features, labels=part["raw_class_id"].to_numpy(),
                    metadata={column: part[column].to_numpy() for column in metadata_columns},
                )
        if offset != len(mask):
            raise ValueError(f"Source row coverage changed while loading: {path}")
    context.assert_unchanged()
    if written != sum(part.num_rows for part in selected_parts):
        raise ValueError("Loaded feature row count does not match selected metadata.")
    if written == 0:
        selected = pa.concat_tables(selected_parts)
        yield DDIBatchArrays(
            features=np.empty((0, len(feature_columns)), dtype=np.float64),
            labels=selected["raw_class_id"].to_numpy(),
            metadata={column: selected[column].to_numpy() for column in metadata_columns},
        )


def load_development_fold_arrays(
    context: DevelopmentFoldContext,
    feature_columns: list[str],
    *,
    role: str,
    member_id: int,
    validation_fold: int,
    class_ids: Iterable[int] | None = None,
    batch_size: int = 2048,
) -> DDIBatchArrays:
    """Collect the raw streaming API into aligned arrays (same Prompt 5 contract)."""
    parts = list(iter_development_fold_arrays(
        context, feature_columns, role=role, member_id=member_id,
        validation_fold=validation_fold, class_ids=class_ids, batch_size=batch_size,
    ))
    return DDIBatchArrays(
        features=np.concatenate([part.features for part in parts]),
        labels=np.concatenate([part.labels for part in parts]),
        metadata={key: np.concatenate([part.metadata[key] for part in parts])
                  for key in parts[0].metadata},
    )


def load_development_fold_pair(
    parquet_paths: Sequence[str | Path],
    feature_columns: list[str],
    assignments: Any,
    *,
    fold_id: int,
    current_class_ids: Iterable[int],
    seen_class_ids: Iterable[int],
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    batch_size: int = 2048,
) -> tuple[DDIBatchArrays, DDIBatchArrays]:
    """Load a frozen-fold train/validation pair in one source pass.

    The legacy trainer used to call ``load_development_fold_arrays`` twice:
    once for current training rows and once for held-out validation rows. Each
    call scanned both development Parquets and materialized intermediate
    Pandas frames. This function routes both roles while each source is being
    streamed once with Arrow. It does not fit preprocessing or read test data.

    ``assignments`` is deliberately duck-typed to avoid importing the fold
    module here (which imports this dataset module). It must provide the
    validated ``lookup(sample_ids)`` method.
    """

    if len(parquet_paths) != 2:
        raise ValueError("Frozen-fold loading requires train and validation Parquets.")
    if not 0 <= int(fold_id) < int(assignments.fold_count):
        raise ValueError("fold_id is outside the configured fold range.")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must be non-empty and unique.")
    current_set = {int(value) for value in current_class_ids}
    seen_set = {int(value) for value in seen_class_ids}
    if not current_set or not seen_set:
        raise ValueError("Current and seen class selections must be non-empty.")

    train_features: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    train_meta_a: list[np.ndarray] = []
    train_meta_b: list[np.ndarray] = []
    validation_features: list[np.ndarray] = []
    validation_labels: list[np.ndarray] = []
    validation_meta_a: list[np.ndarray] = []
    validation_meta_b: list[np.ndarray] = []
    train_rows = validation_rows = 0
    columns = [*feature_columns, DEFAULT_LABEL_COL, DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN]

    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            labels = table[DEFAULT_LABEL_COL].to_numpy().astype(np.int64, copy=False)
            meta_a = table[DRUG_ID_A_COLUMN].to_numpy(zero_copy_only=False)
            meta_b = table[DRUG_ID_B_COLUMN].to_numpy(zero_copy_only=False)
            sample_ids = build_stable_sample_ids(
                {DRUG_ID_A_COLUMN: meta_a, DRUG_ID_B_COLUMN: meta_b},
                len(labels),
            )
            fold_values = np.asarray(assignments.lookup(sample_ids), dtype=np.int16)
            train_mask = (fold_values != fold_id) & np.isin(labels, list(current_set))
            validation_mask = (fold_values == fold_id) & np.isin(labels, list(seen_set))
            if max_train_rows is not None:
                remaining = max_train_rows - train_rows
                if remaining <= 0:
                    train_mask.fill(False)
                elif int(train_mask.sum()) > remaining:
                    selected = np.flatnonzero(train_mask)
                    train_mask.fill(False)
                    train_mask[selected[:remaining]] = True
            if max_validation_rows is not None:
                remaining = max_validation_rows - validation_rows
                if remaining <= 0:
                    validation_mask.fill(False)
                elif int(validation_mask.sum()) > remaining:
                    selected = np.flatnonzero(validation_mask)
                    validation_mask.fill(False)
                    validation_mask[selected[:remaining]] = True

            if train_mask.any():
                selected = table.filter(pa.array(train_mask))
                values = np.column_stack([
                    selected[column].to_numpy(zero_copy_only=False)
                    for column in feature_columns
                ]).astype(np.float64, copy=False)
                values = values.astype(np.float32, copy=False)
                train_features.append(values)
                train_labels.append(selected[DEFAULT_LABEL_COL].to_numpy().astype(np.int64, copy=False))
                train_meta_a.append(selected[DRUG_ID_A_COLUMN].to_numpy(zero_copy_only=False))
                train_meta_b.append(selected[DRUG_ID_B_COLUMN].to_numpy(zero_copy_only=False))
                train_rows += int(values.shape[0])

            if validation_mask.any():
                selected = table.filter(pa.array(validation_mask))
                values = np.column_stack([
                    selected[column].to_numpy(zero_copy_only=False)
                    for column in feature_columns
                ]).astype(np.float64, copy=False)
                values = values.astype(np.float32, copy=False)
                validation_features.append(values)
                validation_labels.append(selected[DEFAULT_LABEL_COL].to_numpy().astype(np.int64, copy=False))
                validation_meta_a.append(selected[DRUG_ID_A_COLUMN].to_numpy(zero_copy_only=False))
                validation_meta_b.append(selected[DRUG_ID_B_COLUMN].to_numpy(zero_copy_only=False))
                validation_rows += int(values.shape[0])
    if not train_features:
        raise ValueError(f"Fold {fold_id} has no current training rows after filtering.")
    if not validation_features:
        raise ValueError(f"Fold {fold_id} has no held-out validation rows after filtering.")
    train = DDIBatchArrays(
        features=np.concatenate(train_features, axis=0),
        labels=np.concatenate(train_labels, axis=0),
        metadata={
            DRUG_ID_A_COLUMN: np.concatenate(train_meta_a),
            DRUG_ID_B_COLUMN: np.concatenate(train_meta_b),
        },
    )
    validation = DDIBatchArrays(
        features=np.concatenate(validation_features, axis=0),
        labels=np.concatenate(validation_labels, axis=0),
        metadata={
            DRUG_ID_A_COLUMN: np.concatenate(validation_meta_a),
            DRUG_ID_B_COLUMN: np.concatenate(validation_meta_b),
        },
    )
    return train, validation


def load_development_fold_identity(
    context: DevelopmentFoldContext,
    *,
    role: str,
    member_id: int,
    validation_fold: int,
    class_ids: Iterable[int] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select frozen-fold labels/provenance without reading descriptor columns.

    This is used to prove prediction/OOF coverage.  It preserves train-source
    then validation-source order and uses the already startup-validated frozen
    assignment tables; it does not rebuild folds or touch test descriptors.
    """

    context.assert_unchanged()
    if role not in {"train", "validation", "all"}:
        raise ValueError("role must be train, validation or all.")
    mapping = context.manifest["member_to_validation_fold"]
    if type(member_id) is not int or str(member_id) not in mapping:
        raise ValueError("member_id must be 0, 1 or 2.")
    if type(validation_fold) is not int or validation_fold != mapping[str(member_id)]:
        raise ValueError("validation_fold does not match manifest member mapping.")
    class_values = None if class_ids is None else np.asarray(list(class_ids), dtype=np.int64)
    selected = []
    for rows in context._rows:
        folds = rows["fold_id"].to_numpy()
        if role == "validation":
            mask = folds == validation_fold
        elif role == "train":
            mask = folds != validation_fold
        else:
            mask = np.ones(rows.num_rows, dtype=bool)
        if class_values is not None:
            mask &= np.isin(rows["raw_class_id"].to_numpy(), class_values)
        selected.append(rows.filter(pa.array(mask)))
    table = pa.concat_tables(selected)
    metadata_columns = ["sample_id", "source_split", "source_row_index", "fold_id"]
    context.assert_unchanged()
    return (
        table["raw_class_id"].to_numpy(),
        {column: table[column].to_numpy() for column in metadata_columns},
    )
