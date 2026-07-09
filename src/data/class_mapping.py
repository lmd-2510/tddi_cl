"""Class mapping helpers for DDI2025-CIL.

This module avoids the unsafe assumption that class IDs are contiguous.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ClassMapping:
    """Immutable mapping between raw class IDs and dense indices."""

    raw_to_index: dict[int, int]

    @property
    def index_to_raw(self) -> dict[int, int]:
        return {index: raw for raw, index in self.raw_to_index.items()}

    @property
    def raw_classes(self) -> list[int]:
        return list(self.raw_to_index.keys())

    @property
    def num_classes(self) -> int:
        return len(self.raw_to_index)

    def remap(self, labels: Iterable[int]) -> np.ndarray:
        labels_array = np.asarray(list(labels), dtype=np.int64)
        return remap_labels(labels_array, self.raw_to_index)


def build_global_class_map(class_ids: Iterable[int]) -> dict[int, int]:
    """Create a dense map from sorted unique class IDs.

    Never infer class count via ``max(class_id) + 1``.
    """

    unique_classes = sorted({int(class_id) for class_id in class_ids})
    return {class_id: index for index, class_id in enumerate(unique_classes)}


def build_seen_class_map(seen_class_ids: Iterable[int]) -> dict[int, int]:
    """Create a task-local dense mapping for currently seen classes."""

    return build_global_class_map(seen_class_ids)


def remap_labels(labels: np.ndarray, class_map: dict[int, int]) -> np.ndarray:
    """Map raw labels into dense indices with validation."""

    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)
    missing = [int(label) for label in unique_labels if int(label) not in class_map]
    if missing:
        raise KeyError(f"Labels missing from class map: {missing[:10]}")
    return np.asarray([class_map[int(label)] for label in labels], dtype=np.int64)


def invert_class_map(class_map: dict[int, int]) -> dict[int, int]:
    """Invert a raw->index class map."""

    return {index: raw for raw, index in class_map.items()}


def load_class_map(path: str | Path) -> dict[int, int]:
    """Load a class map saved as JSON.

    JSON object keys are strings on disk and are converted back to integers.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object class map: {path}")
    return {int(raw): int(index) for raw, index in payload.items()}


def save_class_map(path: str | Path, class_map: dict[int, int]) -> None:
    """Save a class map to JSON with deterministic ordering."""

    ordered = {str(raw): int(class_map[raw]) for raw in sorted(class_map)}
    Path(path).write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def validate_dense_class_map(class_map: dict[int, int]) -> None:
    """Validate that mapped indices are dense 0..K-1."""

    indices = sorted(class_map.values())
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(
            f"Class map indices are not dense. Expected {expected[:10]}..., got {indices[:10]}..."
        )
