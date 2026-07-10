"""Replay buffer for continual DDI2025-CIL experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ReplayBuffer:
    """Simple class-balanced replay buffer storing preprocessed arrays."""

    memory_per_class: int
    random_seed: int = 0
    strategy: str = "herding"
    features_by_class: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    @property
    def total_size(self) -> int:
        return int(sum(values.shape[0] for values in self.features_by_class.values()))

    @property
    def classes(self) -> list[int]:
        return sorted(self.features_by_class.keys())

    def is_empty(self) -> bool:
        return self.total_size == 0

    def update(self, features: np.ndarray, raw_labels: np.ndarray) -> None:
        """Add exemplars from current task, capped per raw class."""

        features = np.asarray(features, dtype=np.float32)
        raw_labels = np.asarray(raw_labels, dtype=np.int64)
        for raw_class in sorted(np.unique(raw_labels)):
            class_mask = raw_labels == raw_class
            class_features = features[class_mask]
            if class_features.shape[0] == 0:
                continue
            if class_features.shape[0] > self.memory_per_class:
                if self.strategy == "herding":
                    mean = class_features.mean(axis=0, keepdims=True)
                    distances = np.linalg.norm(class_features - mean, axis=1)
                    indices = np.argpartition(distances, self.memory_per_class - 1)[: self.memory_per_class]
                else:
                    indices = self.rng.choice(class_features.shape[0], size=self.memory_per_class, replace=False)
                class_features = class_features[indices]
            self.features_by_class[int(raw_class)] = class_features.astype(np.float32, copy=False)

    def get_all(self) -> tuple[np.ndarray, np.ndarray]:
        if self.is_empty():
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        feature_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        for raw_class in self.classes:
            block = self.features_by_class[raw_class]
            feature_blocks.append(block)
            label_blocks.append(np.full(block.shape[0], raw_class, dtype=np.int64))
        return (
            np.concatenate(feature_blocks, axis=0),
            np.concatenate(label_blocks, axis=0),
        )

    def summary_frame(self) -> pd.DataFrame:
        rows = [
            {"raw_class_id": raw_class, "memory_count": int(block.shape[0])}
            for raw_class, block in sorted(self.features_by_class.items())
        ]
        return pd.DataFrame(rows)

    def save_summary(self, path: str | Path) -> None:
        summary = self.summary_frame()
        summary.to_csv(path, index=False)

    def save_snapshot(self, path: str | Path) -> None:
        """Save a lightweight memory snapshot as parquet."""

        features, labels = self.get_all()
        if features.size == 0:
            pd.DataFrame(columns=["raw_class_id"]).to_parquet(path, index=False)
            return
        feature_df = pd.DataFrame(features)
        feature_df.insert(0, "raw_class_id", labels)
        feature_df.to_parquet(path, index=False)
