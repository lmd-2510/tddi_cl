"""Paper-aligned IDER primitives shared by every fixed-head backbone.

IDER (Liu et al., ICLR 2026) splits a classifier into a feature function
``f1`` and a conditioned classifier ``f2``.  ``f2`` receives both the feature
representation and a class-probability vector.  Reapplying ``f2`` with its own
prediction is trained to be idempotent, while a frozen task-boundary model
provides the old-knowledge idempotence target.

The repository's three backbones already expose ``encode`` / ``classify`` and
one fixed 178-way linear head.  ``IDERAdapter`` therefore lives with the method
rather than inside any backbone: it projects the class signal into the native
latent width and adds it immediately before the existing classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.training.class_incremental_policy import (
    TOTAL_DDI_CLASSES,
    active_class_mask,
    fixed_head_cross_entropy,
    mask_unseen_logits,
)


METHOD_NAME = "ider"
METHOD_POLICY = "frozen_fold_fixed_head_ider_v1"
PAPER_REFERENCE = "Liu et al., IDER, ICLR 2026"


class IDERAdapter(nn.Module):
    """Project a class signal into a backbone's final latent representation."""

    def __init__(self, latent_dim: int, *, total_classes: int = TOTAL_DDI_CLASSES) -> None:
        super().__init__()
        if latent_dim <= 0 or total_classes <= 0:
            raise ValueError("IDER latent_dim and total_classes must be positive.")
        self.latent_dim = int(latent_dim)
        self.total_classes = int(total_classes)
        # This is the tabular analogue of the paper's Linear -> LeakyReLU
        # probability injection in the second half of ResNet18.
        self.label_projection = nn.Sequential(
            nn.Linear(self.total_classes, self.latent_dim),
            nn.LeakyReLU(),
        )

    @property
    def projection(self) -> nn.Linear:
        return self.label_projection[0]

    def forward(self, latent: torch.Tensor, class_signal: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"IDER expected latent shape [batch, {self.latent_dim}], got {tuple(latent.shape)}."
            )
        if class_signal.ndim != 2 or class_signal.shape != (
            latent.shape[0],
            self.total_classes,
        ):
            raise ValueError(
                "IDER class_signal must have shape "
                f"[batch, {self.total_classes}]."
            )
        if class_signal.device != latent.device:
            raise ValueError("IDER latent and class_signal must use the same device.")
        return latent + self.label_projection(class_signal.to(dtype=latent.dtype))


def attach_ider_adapter(
    model: nn.Module,
    *,
    active_class_indices: Iterable[int],
    total_classes: int = TOTAL_DDI_CLASSES,
) -> nn.Module:
    """Attach the method-specific adapter without modifying the backbone class."""

    if not all(hasattr(model, name) for name in ("encode", "classify", "classifier_head")):
        raise TypeError("IDER requires encode(), classify(), and classifier_head.")
    head = model.classifier_head
    if not isinstance(head, nn.Linear):
        raise TypeError("IDER currently requires a linear fixed classifier head.")
    if head.out_features != total_classes:
        raise ValueError(
            f"IDER requires a {total_classes}-way fixed head, got {head.out_features}."
        )
    active = tuple(int(index) for index in active_class_indices)
    # Validation also rejects duplicates and out-of-range indices.
    active_class_mask(active, total_classes=total_classes)
    existing = getattr(model, "ider_adapter", None)
    if existing is None:
        model.add_module(
            "ider_adapter",
            IDERAdapter(head.in_features, total_classes=total_classes),
        )
    elif not isinstance(existing, IDERAdapter):
        raise TypeError("model.ider_adapter is not an IDERAdapter.")
    elif existing.latent_dim != head.in_features or existing.total_classes != total_classes:
        raise ValueError("Existing IDER adapter is incompatible with the classifier head.")
    # The active set is policy state, not learned/checkpoint state.  It is
    # reconstructed from the immutable task protocol at every boundary.
    model._ider_active_class_indices = active
    return model


def active_indices_for(model: nn.Module) -> tuple[int, ...]:
    active = getattr(model, "_ider_active_class_indices", None)
    if active is None:
        raise ValueError("IDER model has no active-class policy state.")
    return tuple(int(index) for index in active)


def uniform_class_signal(
    batch_size: int,
    active_class_indices: Iterable[int],
    *,
    total_classes: int = TOTAL_DDI_CLASSES,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Uniform probability over seen classes and exactly zero on future rows."""

    mask = active_class_mask(
        active_class_indices,
        total_classes=total_classes,
        device=device,
    )
    signal = torch.zeros((batch_size, total_classes), device=device, dtype=dtype)
    signal[:, mask] = 1.0 / int(mask.sum().item())
    return signal


def seen_class_probabilities(
    logits: torch.Tensor,
    active_class_indices: Iterable[int],
) -> torch.Tensor:
    """Softmax over seen classes, retaining a zero-filled global vocabulary."""

    masked = mask_unseen_logits(
        logits,
        active_class_indices,
        total_classes=logits.shape[1],
    )
    return F.softmax(masked, dim=1)


def ider_forward_with_latent(
    model: nn.Module,
    features: torch.Tensor,
    class_signal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``f2(f1(x), y)``; inference defaults to a seen-class uniform y."""

    adapter = getattr(model, "ider_adapter", None)
    if not isinstance(adapter, IDERAdapter):
        raise TypeError("IDER forward requires an attached IDERAdapter.")
    latent = model.encode(features)
    if class_signal is None:
        class_signal = uniform_class_signal(
            latent.shape[0],
            active_indices_for(model),
            total_classes=adapter.total_classes,
            device=latent.device,
            dtype=latent.dtype,
        )
    conditioned = adapter(latent, class_signal)
    return model.classify(conditioned), conditioned


def ider_forward(
    model: nn.Module,
    features: torch.Tensor,
    class_signal: torch.Tensor | None = None,
) -> torch.Tensor:
    return ider_forward_with_latent(model, features, class_signal)[0]


def _random_initial_signal(
    labels: torch.Tensor,
    active_class_indices: tuple[int, ...],
    *,
    total_classes: int,
    uniform_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0.0 <= uniform_probability <= 1.0:
        raise ValueError("IDER uniform_probability must be in [0, 1].")
    use_uniform = bool(
        torch.rand((), generator=generator, device="cpu").item()
        < uniform_probability
    )
    if use_uniform:
        return uniform_class_signal(
            labels.shape[0],
            active_class_indices,
            total_classes=total_classes,
            device=labels.device,
            dtype=torch.float32,
        )
    return F.one_hot(labels, num_classes=total_classes).to(dtype=torch.float32)


def _active_mse(
    first: torch.Tensor,
    second: torch.Tensor,
    active_class_indices: tuple[int, ...],
) -> torch.Tensor:
    if not active_class_indices:
        return first.new_zeros(())
    index = torch.as_tensor(active_class_indices, device=first.device, dtype=torch.long)
    return F.mse_loss(first.index_select(1, index), second.index_select(1, index))


@dataclass(frozen=True)
class IDERLossBreakdown:
    current_supervised: torch.Tensor
    current_idempotence_distillation: torch.Tensor
    replay_supervised: torch.Tensor
    replay_idempotence_distillation: torch.Tensor
    total: torch.Tensor


@dataclass
class IDERTrainingContext:
    """Task/epoch-local IDER state; no future data and no hidden global state."""

    old_model: nn.Module | None
    replay_features: torch.Tensor | None
    replay_labels: torch.Tensor | None
    active_class_indices: tuple[int, ...]
    old_class_indices: tuple[int, ...]
    replay_batch_size: int
    weight_old_replay_id: float
    weight_current_id: float
    weight_replay_supervised: float
    uniform_probability: float
    generator: torch.Generator

    def __post_init__(self) -> None:
        if self.replay_batch_size <= 0:
            raise ValueError("IDER replay_batch_size must be positive.")
        for name in (
            "weight_old_replay_id",
            "weight_current_id",
            "weight_replay_supervised",
        ):
            value = float(getattr(self, name))
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f"IDER {name} must be finite and nonnegative.")
        if not 0.0 <= self.uniform_probability <= 1.0:
            raise ValueError("IDER uniform_probability must be in [0, 1].")
        if (self.replay_features is None) != (self.replay_labels is None):
            raise ValueError("IDER replay features and labels must be present together.")
        if self.replay_features is not None:
            if (
                self.replay_features.ndim != 2
                or self.replay_labels.ndim != 1
                or self.replay_features.shape[0] != self.replay_labels.shape[0]
            ):
                raise ValueError("IDER replay features and labels are not aligned.")

    @property
    def has_replay(self) -> bool:
        return self.replay_features is not None and self.replay_features.shape[0] > 0

    def sample_replay(
        self,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.has_replay:
            return None
        assert self.replay_features is not None and self.replay_labels is not None
        count = min(self.replay_batch_size, int(self.replay_labels.shape[0]))
        indices = torch.randperm(
            int(self.replay_labels.shape[0]),
            generator=self.generator,
        )[:count]
        return (
            self.replay_features.index_select(0, indices).to(device=device),
            self.replay_labels.index_select(0, indices).to(device=device),
        )


def compute_ider_loss(
    model: nn.Module,
    current_features: torch.Tensor,
    current_labels: torch.Tensor,
    context: IDERTrainingContext,
) -> IDERLossBreakdown:
    """Compute the four objectives used by the official ER+ID implementation."""

    adapter = getattr(model, "ider_adapter", None)
    if not isinstance(adapter, IDERAdapter):
        raise TypeError("IDER loss requires a model with IDERAdapter.")
    total_classes = adapter.total_classes
    active = context.active_class_indices

    initial = _random_initial_signal(
        current_labels,
        active,
        total_classes=total_classes,
        uniform_probability=context.uniform_probability,
        generator=context.generator,
    )
    first_logits, _ = ider_forward_with_latent(model, current_features, initial)
    second_logits, _ = ider_forward_with_latent(
        model,
        current_features,
        seen_class_probabilities(first_logits, active),
    )
    current_supervised = 0.5 * (
        fixed_head_cross_entropy(first_logits, current_labels, active, total_classes=total_classes)
        + fixed_head_cross_entropy(second_logits, current_labels, active, total_classes=total_classes)
    )
    zero = current_supervised.new_zeros(())
    current_id = zero
    replay_supervised = zero
    replay_id = zero

    if context.old_model is not None and context.old_class_indices:
        uniform = uniform_class_signal(
            current_labels.shape[0],
            active,
            total_classes=total_classes,
            device=current_features.device,
            dtype=current_features.dtype,
        )
        current_first = ider_forward(model, current_features, uniform)
        old_second = ider_forward(
            context.old_model,
            current_features,
            seen_class_probabilities(current_first, active),
        )
        current_id = _active_mse(current_first, old_second, context.old_class_indices)

    replay_batch = context.sample_replay(device=current_features.device)
    if replay_batch is not None:
        replay_features, replay_labels = replay_batch
        replay_initial = _random_initial_signal(
            replay_labels,
            active,
            total_classes=total_classes,
            uniform_probability=context.uniform_probability,
            generator=context.generator,
        )
        replay_first = ider_forward(model, replay_features, replay_initial)
        replay_second = ider_forward(
            model,
            replay_features,
            seen_class_probabilities(replay_first, active),
        )
        # The official code sums (rather than averages) the two replay CE terms.
        replay_supervised = (
            fixed_head_cross_entropy(replay_first, replay_labels, active, total_classes=total_classes)
            + fixed_head_cross_entropy(replay_second, replay_labels, active, total_classes=total_classes)
        )

    replay_id_batch = context.sample_replay(device=current_features.device)
    if (
        replay_id_batch is not None
        and context.old_model is not None
        and context.old_class_indices
    ):
        replay_features, _ = replay_id_batch
        replay_uniform = uniform_class_signal(
            replay_features.shape[0],
            active,
            total_classes=total_classes,
            device=replay_features.device,
            dtype=replay_features.dtype,
        )
        replay_first = ider_forward(model, replay_features, replay_uniform)
        replay_old_second = ider_forward(
            context.old_model,
            replay_features,
            seen_class_probabilities(replay_first, active),
        )
        replay_id = _active_mse(
            replay_first,
            replay_old_second,
            context.old_class_indices,
        )

    total = (
        current_supervised
        + context.weight_current_id * current_id
        + context.weight_replay_supervised * replay_supervised
        + context.weight_old_replay_id * replay_id
    )
    return IDERLossBreakdown(
        current_supervised=current_supervised,
        current_idempotence_distillation=current_id,
        replay_supervised=replay_supervised,
        replay_idempotence_distillation=replay_id,
        total=total,
    )


def optimizer_step_preserving_future_classes(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    class_policy,
) -> None:
    """Protect unseen head rows and unseen IDER projection columns."""

    adapter = getattr(model, "ider_adapter", None)
    if not isinstance(adapter, IDERAdapter):
        raise TypeError("IDER optimizer step requires an attached adapter.")
    head_snapshot = class_policy.capture_inactive_rows(model.classifier_head)
    active = active_class_mask(
        active_indices_for(model),
        total_classes=adapter.total_classes,
        device=adapter.projection.weight.device,
    )
    inactive = (~active).nonzero(as_tuple=False).flatten()
    projection_snapshot = adapter.projection.weight.detach()[:, inactive].clone()
    if adapter.projection.weight.grad is not None and inactive.numel():
        adapter.projection.weight.grad[:, inactive] = 0
    optimizer.step()
    class_policy.restore_inactive_rows(model.classifier_head, head_snapshot)
    if inactive.numel():
        with torch.no_grad():
            adapter.projection.weight[:, inactive] = projection_snapshot

