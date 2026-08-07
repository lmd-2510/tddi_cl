"""Fixed-storage, fixed-exposure replay primitives for S04."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

try:
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover - training requires torch
    Sampler = object  # type: ignore[assignment,misc]


def max_min_uniform_allocation(
    capacities: dict[int, int],
    budget: int,
) -> dict[int, int]:
    """Allocate an exact capacity-constrained budget as uniformly as possible."""

    if budget <= 0:
        raise ValueError("budget must be positive.")
    normalized = {int(class_id): int(capacity) for class_id, capacity in capacities.items()}
    if not normalized or any(capacity < 0 for capacity in normalized.values()):
        raise ValueError("capacities must be a non-empty mapping of non-negative counts.")

    target = min(budget, sum(normalized.values()))
    allocation = {class_id: 0 for class_id in sorted(normalized)}
    active = [class_id for class_id in sorted(normalized) if normalized[class_id] > 0]
    remaining = target
    while remaining and active:
        share, remainder = divmod(remaining, len(active))
        if share == 0:
            for class_id in active[:remainder]:
                allocation[class_id] += 1
            remaining = 0
            break

        saturated = [
            class_id
            for class_id in active
            if normalized[class_id] - allocation[class_id] <= share
        ]
        if saturated:
            for class_id in saturated:
                increment = normalized[class_id] - allocation[class_id]
                allocation[class_id] += increment
                remaining -= increment
            active = [class_id for class_id in active if class_id not in set(saturated)]
            continue

        for class_id in active:
            allocation[class_id] += share
            remaining -= share
        for class_id in active[:remainder]:
            allocation[class_id] += 1
            remaining -= 1

    if sum(allocation.values()) != target:
        raise RuntimeError("max-min allocation did not consume the requested feasible budget.")
    if any(allocation[class_id] > normalized[class_id] for class_id in allocation):
        raise RuntimeError("max-min allocation exceeded a class capacity.")
    return allocation


@dataclass
class FixedBudgetReplayBuffer:
    """Class-balanced exemplar buffer with a fixed total unique-sample budget."""

    total_memory_budget: int
    random_seed: int = 0
    features_by_class: dict[int, np.ndarray] = field(default_factory=dict)
    available_count_by_class: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_memory_budget <= 0:
            raise ValueError("total_memory_budget must be positive.")

    @property
    def total_size(self) -> int:
        return int(sum(values.shape[0] for values in self.features_by_class.values()))

    @property
    def classes(self) -> list[int]:
        return sorted(self.available_count_by_class)

    @property
    def memory_counts(self) -> dict[int, int]:
        return {
            class_id: int(self.features_by_class[class_id].shape[0])
            for class_id in self.classes
        }

    def is_empty(self) -> bool:
        return self.total_size == 0

    @staticmethod
    def _rank_by_class_mean(features: np.ndarray) -> np.ndarray:
        mean = features.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(features - mean, axis=1)
        return np.lexsort((np.arange(features.shape[0]), distances))

    def update(self, features: np.ndarray, raw_labels: np.ndarray) -> None:
        """Add previously unseen classes, then rebalance all retained exemplars."""

        features = np.asarray(features, dtype=np.float32)
        raw_labels = np.asarray(raw_labels, dtype=np.int64)
        if features.ndim != 2 or raw_labels.shape != (features.shape[0],):
            raise ValueError("features and raw_labels must be aligned two-dimensional/one-dimensional arrays.")
        if features.shape[0] == 0 or not np.isfinite(features).all():
            raise ValueError("fixed-budget replay update requires non-empty finite features.")

        new_classes = sorted(int(class_id) for class_id in np.unique(raw_labels))
        duplicate_classes = sorted(set(new_classes) & set(self.available_count_by_class))
        if duplicate_classes:
            raise ValueError(f"Fixed-budget buffer received classes twice: {duplicate_classes}")

        current_features: dict[int, np.ndarray] = {}
        for class_id in new_classes:
            class_features = features[raw_labels == class_id]
            self.available_count_by_class[class_id] = int(class_features.shape[0])
            current_features[class_id] = class_features

        allocation = max_min_uniform_allocation(
            self.available_count_by_class,
            self.total_memory_budget,
        )
        for class_id in self.classes:
            target_count = allocation[class_id]
            if class_id in current_features:
                class_features = current_features[class_id]
                ranking = self._rank_by_class_mean(class_features)
                retained = class_features[ranking[:target_count]]
            else:
                retained = self.features_by_class[class_id][:target_count]
                if retained.shape[0] != target_count:
                    raise RuntimeError(
                        "An old class would need more exemplars than were retained previously."
                    )
            self.features_by_class[class_id] = retained.astype(np.float32, copy=False)

        expected_total = min(
            self.total_memory_budget,
            sum(self.available_count_by_class.values()),
        )
        if self.total_size != expected_total:
            raise RuntimeError("Fixed-budget buffer size does not match its feasible budget.")

    def get_all(self) -> tuple[np.ndarray, np.ndarray]:
        if self.is_empty():
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        feature_blocks = [self.features_by_class[class_id] for class_id in self.classes]
        label_blocks = [
            np.full(self.features_by_class[class_id].shape[0], class_id, dtype=np.int64)
            for class_id in self.classes
        ]
        return np.concatenate(feature_blocks), np.concatenate(label_blocks)

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "raw_class_id": class_id,
                    "available_count": self.available_count_by_class[class_id],
                    "memory_count": int(self.features_by_class[class_id].shape[0]),
                    "total_memory_budget": self.total_memory_budget,
                }
                for class_id in self.classes
            ]
        )

    def save_summary(self, path: str | Path) -> None:
        self.summary_frame().to_csv(path, index=False)

    def save_snapshot(self, path: str | Path) -> None:
        features, labels = self.get_all()
        if features.size == 0:
            pd.DataFrame(columns=["raw_class_id"]).to_parquet(path, index=False)
            return
        frame = pd.DataFrame(features)
        frame.insert(0, "raw_class_id", labels)
        frame.to_parquet(path, index=False)


@dataclass(frozen=True)
class ReplayEpochAudit:
    epoch: int
    current_draws: int
    replay_draws: int
    unique_replay_examples: int
    per_class_draws: dict[int, int]
    per_class_unique_examples: dict[int, int]
    replay_indices: tuple[int, ...]


class FixedReplaySampler(Sampler):  # type: ignore[type-arg]
    """Sample current data once and an exact class-uniform replay budget per epoch."""

    def __init__(
        self,
        *,
        current_count: int,
        replay_raw_labels: np.ndarray,
        replay_draws_per_epoch: int,
        seed: int,
        task_id: int,
    ) -> None:
        if current_count <= 0 or replay_draws_per_epoch < 0 or task_id < 0:
            raise ValueError("Invalid fixed replay sampler counts or task_id.")
        self.current_count = int(current_count)
        self.replay_raw_labels = np.asarray(replay_raw_labels, dtype=np.int64)
        if self.replay_raw_labels.ndim != 1:
            raise ValueError("replay_raw_labels must be one-dimensional.")
        if replay_draws_per_epoch and self.replay_raw_labels.shape[0] == 0:
            raise ValueError("A positive replay budget requires replay exemplars.")
        self.replay_draws_per_epoch = int(replay_draws_per_epoch)
        self.seed = int(seed)
        self.task_id = int(task_id)
        self._epoch = 0
        self.history: list[ReplayEpochAudit] = []

    def __len__(self) -> int:
        return self.current_count + self.replay_draws_per_epoch

    @property
    def last_audit(self) -> ReplayEpochAudit | None:
        return self.history[-1] if self.history else None

    @staticmethod
    def _draw_without_replacement_cycles(
        indices: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count == 0:
            return np.empty((0,), dtype=np.int64)
        full_cycles, remainder = divmod(count, indices.shape[0])
        blocks = [rng.permutation(indices) for _ in range(full_cycles)]
        if remainder:
            blocks.append(rng.choice(indices, size=remainder, replace=False))
        return np.concatenate(blocks).astype(np.int64, copy=False)

    def __iter__(self) -> Iterator[int]:
        epoch = self._epoch
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.task_id, epoch]))
        current_indices = rng.permutation(self.current_count).astype(np.int64, copy=False)
        replay_blocks: list[np.ndarray] = []
        per_class_draws: dict[int, int] = {}
        per_class_unique: dict[int, int] = {}

        classes = sorted(int(class_id) for class_id in np.unique(self.replay_raw_labels))
        if self.replay_draws_per_epoch:
            base, remainder = divmod(self.replay_draws_per_epoch, len(classes))
            extra_positions = {(epoch + offset) % len(classes) for offset in range(remainder)}
            for position, class_id in enumerate(classes):
                draw_count = base + int(position in extra_positions)
                local_indices = np.flatnonzero(self.replay_raw_labels == class_id).astype(np.int64)
                dataset_indices = local_indices + self.current_count
                draws = self._draw_without_replacement_cycles(dataset_indices, draw_count, rng)
                replay_blocks.append(draws)
                per_class_draws[class_id] = draw_count
                per_class_unique[class_id] = int(np.unique(draws).shape[0])

        replay_indices = (
            np.concatenate(replay_blocks)
            if replay_blocks
            else np.empty((0,), dtype=np.int64)
        )
        all_indices = np.concatenate([current_indices, replay_indices])
        all_indices = rng.permutation(all_indices)
        audit = ReplayEpochAudit(
            epoch=epoch,
            current_draws=self.current_count,
            replay_draws=int(replay_indices.shape[0]),
            unique_replay_examples=int(np.unique(replay_indices).shape[0]),
            per_class_draws=per_class_draws,
            per_class_unique_examples=per_class_unique,
            replay_indices=tuple(int(index) for index in replay_indices),
        )
        self.history.append(audit)
        self._epoch += 1
        return iter(all_indices.tolist())
