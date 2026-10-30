"""Paper-aligned Experience Replay primitives.

The ER baseline from Chaudhry et al. (2019) trains on a current minibatch
stacked with an equally-sized minibatch sampled from episodic memory.  The
canonical memory writer is reservoir sampling.  The existing CB-Replay loss is
retained as a compatibility function until a training runner explicitly opts
into the new ER implementation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.training.class_incremental_policy import fixed_head_cross_entropy

METHOD_NAME = "class_balanced_fixed_budget_replay"
LOSS_VARIANT = "focal_current_ce_replay"
PAPER_METHOD_NAME = "experience_replay"
PAPER_MEMORY_POLICY = "reservoir_sampling"


def classification_loss(logits, labels, old_class_indices=(), *, gamma=1.0):
    """Mean Focal loss on current examples and CE on retained old examples.

    Both groups use the full seen-class head. Each draw contributes once to the
    batch mean, preserving the replay sampler's requested exposure ratio.
    """
    log_probs = F.log_softmax(logits, dim=1)
    target_log_probs = log_probs.gather(1, labels[:, None]).squeeze(1)
    focal = -((1.0 - target_log_probs.exp()) ** gamma) * target_log_probs
    old_indices = torch.as_tensor(list(old_class_indices), device=labels.device, dtype=torch.long)
    replay_mask = torch.isin(labels, old_indices)
    ce = -target_log_probs
    return torch.where(replay_mask, ce, focal).mean()


class ReservoirMemory:
    """Fixed-capacity Algorithm-R reservoir over a sample stream.

    Samples are held on CPU. ``update`` must be called exactly once for each
    unique stream example; repeating epochs must not reinsert the same task
    examples.  After observing ``n`` examples, every example has equal
    probability ``capacity / n`` of being retained.
    """

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        self.capacity = capacity
        self.seed = int(seed)
        self.seen_examples = 0
        self.size = 0
        self._features: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

    def __len__(self) -> int:
        return self.size

    def _allocate(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self._features = torch.empty(
            (self.capacity, *features.shape[1:]), dtype=features.dtype, device="cpu"
        )
        self._labels = torch.empty(self.capacity, dtype=labels.dtype, device="cpu")

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Process one current-data minibatch in stream order."""

        if features.ndim < 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must be aligned batched tensors.")
        if labels.dtype != torch.long:
            raise ValueError("Reservoir labels must use torch.long.")
        features = features.detach().to(device="cpu").contiguous()
        labels = labels.detach().to(device="cpu").contiguous()
        if self._features is None:
            self._allocate(features, labels)
        assert self._features is not None and self._labels is not None
        if features.shape[1:] != self._features.shape[1:] or features.dtype != self._features.dtype:
            raise ValueError("Reservoir feature shape or dtype changed.")

        for row in range(features.shape[0]):
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                slot = int(
                    torch.randint(
                        0,
                        self.seen_examples + 1,
                        (1,),
                        generator=self._generator,
                    ).item()
                )
            if slot < self.capacity:
                self._features[slot].copy_(features[row])
                self._labels[slot].copy_(labels[row])
            self.seen_examples += 1

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample memory without replacement."""

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.size == 0 or self._features is None or self._labels is None:
            raise ValueError("Cannot sample an empty reservoir.")
        count = min(batch_size, self.size)
        indices = torch.randperm(self.size, generator=self._generator)[:count]
        return (
            self._features[indices].to(device=device),
            self._labels[indices].to(device=device),
        )

    def state_dict(self) -> dict[str, object]:
        """Return an exact checkpoint, including sampler RNG state."""

        features = None if self._features is None else self._features[: self.size].clone()
        labels = None if self._labels is None else self._labels[: self.size].clone()
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "seen_examples": self.seen_examples,
            "size": self.size,
            "features": features,
            "labels": labels,
            "generator_state": self._generator.get_state().clone(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ReservoirMemory":
        memory = cls(int(state["capacity"]), seed=int(state["seed"]))
        size = int(state["size"])
        seen_examples = int(state["seen_examples"])
        features = state.get("features")
        labels = state.get("labels")
        if size < 0 or size > memory.capacity or seen_examples < size:
            raise ValueError("Invalid reservoir checkpoint counters.")
        if size:
            if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
                raise ValueError("Reservoir checkpoint tensors are missing.")
            memory._allocate(features, labels)
            assert memory._features is not None and memory._labels is not None
            memory._features[:size].copy_(features)
            memory._labels[:size].copy_(labels)
        memory.size = size
        memory.seen_examples = seen_examples
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("Reservoir checkpoint RNG state is missing.")
        memory._generator.set_state(generator_state)
        return memory


def build_experience_replay_batch(
    current_features: torch.Tensor,
    current_labels: torch.Tensor,
    memory: ReservoirMemory | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack one current minibatch with an equally-sized memory minibatch."""

    if current_features.shape[0] != current_labels.shape[0]:
        raise ValueError("Current features and labels must align.")
    if memory is None or len(memory) == 0:
        return current_features, current_labels
    replay_features, replay_labels = memory.sample(
        int(current_labels.shape[0]), device=current_features.device
    )
    replay_labels = replay_labels.to(device=current_labels.device)
    return (
        torch.cat((current_features, replay_features), dim=0),
        torch.cat((current_labels, replay_labels), dim=0),
    )


def experience_replay_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_class_indices,
    *,
    total_classes: int = 178,
) -> torch.Tensor:
    """Paper ER objective: one CE mean over stacked current and memory rows."""

    return fixed_head_cross_entropy(
        logits,
        labels,
        active_class_indices,
        total_classes=total_classes,
    )
