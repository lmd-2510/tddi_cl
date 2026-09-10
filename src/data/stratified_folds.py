"""Deterministic stratified development folds for paper-compatible T-DDI runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold

from src.data.ddi_dataset import DDIBatchArrays, DEFAULT_LABEL_COL
from src.data.sample_identity import (
    DRUG_ID_A_COLUMN,
    DRUG_ID_B_COLUMN,
    build_stable_sample_ids,
)


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
