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


def observe(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
            current_features: torch.Tensor, current_labels: torch.Tensor,
            current_sample_ids: Sequence[str], buffer: ReservoirLogitBuffer,
            *, alpha: float, beta: float, replay_batch_size: int,
            exposure: MutableMapping[int, int] | None = None) -> DERPPLosses:
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
