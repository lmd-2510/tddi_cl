"""Online DER++ memory and three-term optimization step.

The stored logits are the model outputs *before* the update that first saw the
sample.  The reservoir processes each training row once, in stream order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DERPPLosses:
    current_ce: float
    replay_logit_mse: float
    replay_label_ce: float
    total: float
    replay_logit_rows: int
    replay_label_rows: int


class ReservoirLogitBuffer:
    """Uniform reservoir of frozen inputs, labels, IDs and insertion logits."""

    def __init__(self, capacity: int, feature_dim: int, class_count: int, *, seed: int):
        if min(capacity, feature_dim, class_count) <= 0:
            raise ValueError("DER++ buffer dimensions must be positive.")
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.class_count = int(class_count)
        self.features = np.empty((capacity, feature_dim), dtype=np.float32)
        self.logits = np.empty((capacity, class_count), dtype=np.float32)
        self.labels = np.empty(capacity, dtype=np.int64)
        self.sample_ids = np.empty(capacity, dtype="U96")
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def sample(self, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if count <= 0 or self.size == 0:
            raise ValueError("Cannot draw from an empty DER++ buffer.")
        indices = self.rng.choice(self.size, size=min(count, self.size), replace=False)
        return (self.features[indices].copy(), self.labels[indices].copy(),
                self.logits[indices].copy())

    def add_batch(self, features: np.ndarray, labels: np.ndarray, logits: np.ndarray,
                  sample_ids: Sequence[str]) -> None:
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        logits = np.asarray(logits, dtype=np.float32)
        if (features.ndim != 2 or features.shape[1] != self.feature_dim
                or logits.shape != (len(features), self.class_count)
                or labels.shape != (len(features),) or len(sample_ids) != len(features)
                or not np.isfinite(features).all() or not np.isfinite(logits).all()
                or np.any(labels < 0) or np.any(labels >= self.class_count)):
            raise ValueError("Invalid DER++ buffer batch shape, label or numerical value.")
        if any(len(str(value)) > 96 for value in sample_ids):
            raise ValueError("DER++ sample ID exceeds the buffer's 96-character storage limit.")
        for index in range(len(features)):
            self.seen += 1
            slot = self.size if self.size < self.capacity else int(self.rng.integers(self.seen))
            if slot >= self.capacity:
                continue
            self.features[slot] = features[index]
            self.labels[slot] = labels[index]
            self.logits[slot] = logits[index]
            self.sample_ids[slot] = str(sample_ids[index])
            self.size = min(self.size + 1, self.capacity)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity, "feature_dim": self.feature_dim,
            "class_count": self.class_count, "size": self.size, "seen": self.seen,
            "features": self.features[:self.size].copy(),
            "logits": self.logits[:self.size].copy(),
            "labels": self.labels[:self.size].copy(),
            "sample_ids": self.sample_ids[:self.size].copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if (state["capacity"] != self.capacity or state["feature_dim"] != self.feature_dim
                or state["class_count"] != self.class_count or not 0 <= state["size"] <= self.capacity
                or state["seen"] < state["size"]):
            raise ValueError("DER++ reservoir checkpoint contract mismatch.")
        size = int(state["size"])
        if (np.asarray(state["features"]).shape != (size, self.feature_dim)
                or np.asarray(state["logits"]).shape != (size, self.class_count)
                or np.asarray(state["labels"]).shape != (size,)
                or np.asarray(state["sample_ids"]).shape != (size,)):
            raise ValueError("Malformed DER++ reservoir checkpoint arrays.")
        self.features[:size] = state["features"]
        self.logits[:size] = state["logits"]
        self.labels[:size] = state["labels"]
        self.sample_ids[:size] = state["sample_ids"]
        if not np.isfinite(self.features[:size]).all() or not np.isfinite(self.logits[:size]).all():
            raise ValueError("DER++ checkpoint contains nonfinite buffer values.")
        self.size, self.seen = size, int(state["seen"])
        self.rng.bit_generator.state = state["rng_state"]


def equal_class_quotas(counts: dict[int, int], capacity: int) -> dict[int, int]:
    """Water-fill seen training classes; never allocate more than observed rows."""
    if capacity <= 0 or not counts or any(label < 0 or count <= 0 for label, count in counts.items()):
        raise ValueError("Balanced reservoir requires positive capacity and observed class counts.")
    labels = sorted(counts)
    if sum(counts.values()) <= capacity:
        return {label: int(counts[label]) for label in labels}
    low, high = 0, max(counts.values())
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(counts[label], middle) for label in labels) <= capacity:
            low = middle
        else:
            high = middle - 1
    quotas = {label: min(counts[label], low) for label in labels}
    remaining = capacity - sum(quotas.values())
    for label in labels:
        if remaining == 0:
            break
        if counts[label] > quotas[label]:
            quotas[label] += 1
            remaining -= 1
    if remaining != 0 or sum(quotas.values()) != capacity:
        raise RuntimeError("Balanced reservoir quota allocation did not fill the budget.")
    return quotas


class ClassBalancedReservoirLogitBuffer(ReservoirLogitBuffer):
    """Class-quota reservoir; uniform random replacement within each seen class.

    At a task boundary, only current-task training counts and earlier observed
    counts set the water-filled quotas. Old retained rows are uniformly
    downsampled when their quotas shrink. New rows are then seen exactly once.
    Replay draws remain uniform over all retained slots, as in original DER++.
    """

    policy = "online_equal_class_reservoir_logits_v1"

    def __init__(self, capacity: int, feature_dim: int, class_count: int, *, seed: int):
        super().__init__(capacity, feature_dim, class_count, seed=seed)
        self.class_seen: dict[int, int] = {}
        self.quotas: dict[int, int] = {}
        self._class_slots: dict[int, list[int]] = {}
        self._position = np.empty(capacity, dtype=np.int32)

    def _remove_slot(self, slot: int) -> None:
        label = int(self.labels[slot])
        slots = self._class_slots[label]
        position = int(self._position[slot])
        tail_in_class = slots[-1]
        slots[position] = tail_in_class
        self._position[tail_in_class] = position
        slots.pop()
        tail = self.size - 1
        if slot != tail:
            moved_label = int(self.labels[tail])
            moved_position = int(self._position[tail])
            self.features[slot] = self.features[tail]
            self.logits[slot] = self.logits[tail]
            self.labels[slot] = self.labels[tail]
            self.sample_ids[slot] = self.sample_ids[tail]
            self._class_slots[moved_label][moved_position] = slot
            self._position[slot] = moved_position
        self.size -= 1

    def begin_task(self, new_training_counts: dict[int, int]) -> dict[int, int]:
        """Freeze task quotas without reading future tasks, validation or test."""
        if (not new_training_counts or any(label < 0 or label >= self.class_count or count <= 0
                                            for label, count in new_training_counts.items())
                or set(new_training_counts) & set(self.class_seen)):
            raise ValueError("Task quotas need positive counts for previously unseen training classes.")
        planned = {**self.class_seen, **{int(k): int(v) for k, v in new_training_counts.items()}}
        quotas = equal_class_quotas(planned, self.capacity)
        for label, slots in self._class_slots.items():
            if quotas[label] > len(slots):
                raise ValueError("Cannot recover previously discarded examples for a growing quota.")
            while len(slots) > quotas[label]:
                index = int(self.rng.integers(len(slots)))
                self._remove_slot(slots[index])
        self.quotas = quotas
        for label in new_training_counts:
            self._class_slots[int(label)] = []
        return quotas.copy()

    def add_batch(self, features: np.ndarray, labels: np.ndarray, logits: np.ndarray,
                  sample_ids: Sequence[str]) -> None:
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        logits = np.asarray(logits, dtype=np.float32)
        if (features.ndim != 2 or features.shape[1] != self.feature_dim
                or logits.shape != (len(features), self.class_count)
                or labels.shape != (len(features),) or len(sample_ids) != len(features)
                or not np.isfinite(features).all() or not np.isfinite(logits).all()
                or np.any(labels < 0) or np.any(labels >= self.class_count)
                or any(len(str(value)) > 96 for value in sample_ids)):
            raise ValueError("Invalid balanced DER++ buffer batch.")
        for index, raw_label in enumerate(labels):
            label = int(raw_label)
            if label not in self.quotas:
                raise ValueError("Balanced DER++ received a class before its task quota was set.")
            self.seen += 1
            count = self.class_seen.get(label, 0) + 1
            self.class_seen[label] = count
            slots = self._class_slots[label]
            if len(slots) < self.quotas[label]:
                slot = self.size
                self.size += 1
                self._position[slot] = len(slots)
                slots.append(slot)
            else:
                candidate = int(self.rng.integers(count))
                if candidate >= self.quotas[label]:
                    continue
                slot = slots[candidate]
            self.features[slot] = features[index]
            self.labels[slot] = label
            self.logits[slot] = logits[index]
            self.sample_ids[slot] = str(sample_ids[index])

    def finish_task(self) -> None:
        if any(self.class_seen.get(label, 0) < quota or len(self._class_slots[label]) != quota
               for label, quota in self.quotas.items()):
            raise ValueError("Balanced reservoir task ended before its planned class quotas were filled.")

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({"policy": self.policy, "class_seen": self.class_seen.copy(),
                      "quotas": self.quotas.copy()})
        return state

    def load_state_dict(self, state: dict) -> None:
        if state.get("policy") != self.policy:
            raise ValueError("Balanced DER++ reservoir checkpoint policy mismatch.")
        super().load_state_dict(state)
        self.class_seen = {int(k): int(v) for k, v in state["class_seen"].items()}
        self.quotas = {int(k): int(v) for k, v in state["quotas"].items()}
        self._class_slots = {label: [] for label in self.quotas}
        for slot in range(self.size):
            label = int(self.labels[slot])
            if label not in self._class_slots:
                raise ValueError("Balanced DER++ checkpoint contains a class without a quota.")
            self._position[slot] = len(self._class_slots[label])
            self._class_slots[label].append(slot)
        if (sum(self.class_seen.values()) != self.seen
                or set(self.class_seen) != set(self.quotas)
                or any(len(self._class_slots[label]) != quota for label, quota in self.quotas.items())
                or sum(self.quotas.values()) != self.size):
            raise ValueError("Balanced DER++ checkpoint class occupancy mismatch.")


class ClassBalancedClassUniformReplayBuffer(ClassBalancedReservoirLogitBuffer):
    """Equal-class storage with balanced-cycle class-uniform replay.

    Storage remains the water-filled, no-duplication reservoir implemented by
    :class:`ClassBalancedReservoirLogitBuffer`.  Only replay selection changes:
    classes are shuffled into cycles and one retained exemplar is selected
    uniformly within every selected class.  Original DER++ and the earlier
    equal-buffer pilot therefore keep their slot-uniform sampling semantics.
    """

    policy = "online_equal_class_reservoir_class_uniform_replay_v1"
    replay_policy = "balanced_cycle_class_uniform_then_exemplar_uniform_v1"

    def _sample_indices(self, count: int) -> np.ndarray:
        if count <= 0 or self.size == 0:
            raise ValueError("Cannot draw from an empty DER++ buffer.")
        active = np.asarray(sorted(label for label, slots in self._class_slots.items() if slots),
                            dtype=np.int64)
        if active.size == 0:
            raise ValueError("Balanced DER++ buffer has no active retained class.")
        selected_classes: list[int] = []
        remaining = int(count)
        while remaining:
            cycle = self.rng.permutation(active)
            take = min(remaining, len(cycle))
            selected_classes.extend(int(value) for value in cycle[:take])
            remaining -= take
        return np.asarray([
            self._class_slots[label][int(self.rng.integers(len(self._class_slots[label])))]
            for label in selected_classes
        ], dtype=np.int64)

    def sample(self, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = self._sample_indices(count)
        return (self.features[indices].copy(), self.labels[indices].copy(),
                self.logits[indices].copy())


def observe(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
            current_features: torch.Tensor, current_labels: torch.Tensor,
            current_sample_ids: Sequence[str], buffer: ReservoirLogitBuffer,
            *, alpha: float, beta: float, replay_batch_size: int,
            exposure: MutableMapping[int, int] | None = None,
            exposure_logit: MutableMapping[int, int] | None = None,
            exposure_label: MutableMapping[int, int] | None = None) -> DERPPLosses:
    """Perform one DER++ update, then insert the current stream batch."""
    if alpha < 0 or beta < 0 or replay_batch_size <= 0:
        raise ValueError("DER++ alpha/beta and replay batch size are invalid.")
    if len(current_sample_ids) != len(current_labels):
        raise ValueError("Current IDs and labels are not aligned.")
    optimizer.zero_grad(set_to_none=True)
    current_logits = model(current_features)
    current_ce = F.cross_entropy(current_logits, current_labels)
    loss = current_ce
    logit_mse = torch.zeros((), device=current_features.device)
    replay_ce = torch.zeros((), device=current_features.device)
    logit_rows = label_rows = 0
    if buffer.size:
        x_mse, labels_mse, stored_logits = buffer.sample(replay_batch_size)
        x_mse_tensor = torch.as_tensor(x_mse, device=current_features.device)
        stored_tensor = torch.as_tensor(stored_logits, device=current_features.device)
        logit_mse = F.mse_loss(model(x_mse_tensor), stored_tensor)
        logit_rows = len(x_mse)
        x_ce, labels_ce, _ = buffer.sample(replay_batch_size)
        if exposure_logit is not None:
            for raw_label in labels_mse:
                key = int(raw_label)
                exposure_logit[key] = exposure_logit.get(key, 0) + 1
        if exposure_label is not None:
            for raw_label in labels_ce:
                key = int(raw_label)
                exposure_label[key] = exposure_label.get(key, 0) + 1
        if exposure is not None:
            for raw_label in np.concatenate((labels_mse, labels_ce)):
                key = int(raw_label)
                exposure[key] = exposure.get(key, 0) + 1
        x_ce_tensor = torch.as_tensor(x_ce, device=current_features.device)
        labels_tensor = torch.as_tensor(labels_ce, device=current_features.device)
        replay_ce = F.cross_entropy(model(x_ce_tensor), labels_tensor)
        label_rows = len(x_ce)
        loss = loss + alpha * logit_mse + beta * replay_ce
    loss.backward()
    optimizer.step()
    buffer.add_batch(current_features.detach().cpu().numpy(),
                     current_labels.detach().cpu().numpy(),
                     current_logits.detach().cpu().numpy(), current_sample_ids)
    return DERPPLosses(float(current_ce.detach()), float(logit_mse.detach()),
                       float(replay_ce.detach()), float(loss.detach()), logit_rows, label_rows)
