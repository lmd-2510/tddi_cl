"""Shared fixed-head policy for class-incremental learning.

Backbones keep their native classifier implementation and emit a fixed global
class vocabulary.  This policy controls which output rows are visible at a
task boundary and protects unseen linear-head rows from both gradients and
optimizer-side weight decay.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


TOTAL_DDI_CLASSES = 178


def active_class_mask(
    active_class_indices: Iterable[int],
    *,
    total_classes: int = TOTAL_DDI_CLASSES,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a validated Boolean mask for classes visible so far."""

    if total_classes <= 0:
        raise ValueError("total_classes must be positive.")
    indices = [int(index) for index in active_class_indices]
    if not indices:
        raise ValueError("At least one active class is required.")
    if len(indices) != len(set(indices)):
        raise ValueError("active_class_indices must not contain duplicates.")
    if min(indices) < 0 or max(indices) >= total_classes:
        raise ValueError("Active class index is outside the fixed head.")
    mask = torch.zeros(total_classes, dtype=torch.bool, device=device)
    mask[torch.as_tensor(indices, dtype=torch.long, device=device)] = True
    return mask


def mask_unseen_logits(
    logits: torch.Tensor,
    active_class_indices: Iterable[int],
    *,
    total_classes: int = TOTAL_DDI_CLASSES,
) -> torch.Tensor:
    """Mask unseen output columns while preserving the fixed head shape."""

    if logits.ndim != 2:
        raise ValueError("Fixed-head logits must have shape [batch, classes].")
    if logits.shape[1] != total_classes:
        raise ValueError(
            f"Expected {total_classes} classifier outputs, got {logits.shape[1]}."
        )
    if not logits.is_floating_point():
        raise ValueError("Classifier logits must use a floating-point dtype.")
    mask = active_class_mask(
        active_class_indices,
        total_classes=total_classes,
        device=logits.device,
    )
    return logits.masked_fill(~mask.unsqueeze(0), -torch.inf)


def fixed_head_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_class_indices: Iterable[int],
    *,
    total_classes: int = TOTAL_DDI_CLASSES,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy over seen classes only, using global dense labels."""

    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch] and align with logits.")
    mask = active_class_mask(
        active_class_indices,
        total_classes=total_classes,
        device=labels.device,
    )
    if labels.numel():
        if labels.dtype != torch.long:
            raise ValueError("labels must use torch.long global class indices.")
        if bool((labels < 0).any()) or bool((labels >= total_classes).any()):
            raise ValueError("Label index is outside the fixed head.")
        if not bool(mask[labels].all()):
            raise ValueError("A target label belongs to an unseen class.")
    masked = logits.masked_fill(~mask.to(logits.device).unsqueeze(0), -torch.inf)
    return F.cross_entropy(masked, labels, reduction=reduction)


@dataclass(frozen=True)
class InactiveLinearRows:
    """Snapshot used to undo optimizer updates to unseen classifier rows."""

    indices: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None


class ClassIncrementalPolicy:
    """One global class map and a monotonically growing active-class set."""

    def __init__(
        self,
        raw_to_global: Mapping[int, int],
        *,
        active_raw_classes: Iterable[int],
    ) -> None:
        mapping = {int(raw): int(index) for raw, index in raw_to_global.items()}
        if not mapping:
            raise ValueError("raw_to_global must not be empty.")
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Global class indices must be contiguous from zero.")
        self._raw_to_global = mapping
        self._active_raw_classes: set[int] = set()
        self.activate(active_raw_classes)

    @classmethod
    def from_raw_class_order(
        cls,
        raw_class_order: Sequence[int],
        *,
        active_raw_classes: Iterable[int],
    ) -> "ClassIncrementalPolicy":
        order = [int(raw) for raw in raw_class_order]
        if len(order) != len(set(order)):
            raise ValueError("raw_class_order must contain unique class IDs.")
        return cls(
            {raw: index for index, raw in enumerate(order)},
            active_raw_classes=active_raw_classes,
        )

    @property
    def total_classes(self) -> int:
        return len(self._raw_to_global)

    @property
    def raw_to_global(self) -> dict[int, int]:
        return dict(self._raw_to_global)

    @property
    def active_raw_classes(self) -> tuple[int, ...]:
        return tuple(
            sorted(self._active_raw_classes, key=self._raw_to_global.__getitem__)
        )

    @property
    def active_class_indices(self) -> tuple[int, ...]:
        return tuple(self._raw_to_global[raw] for raw in self.active_raw_classes)

    def activate(self, raw_classes: Iterable[int]) -> None:
        """Open new class rows; active classes can never be closed again."""

        requested = {int(raw) for raw in raw_classes}
        missing = requested - set(self._raw_to_global)
        if missing:
            raise ValueError(f"Unknown raw class IDs: {sorted(missing)}")
        self._active_raw_classes.update(requested)
        if not self._active_raw_classes:
            raise ValueError("At least one class must be active.")

    def global_labels(self, raw_labels: Iterable[int], *, device=None) -> torch.Tensor:
        """Map raw dataset labels into stable fixed-head indices."""

        raw = [int(label) for label in raw_labels]
        missing = set(raw) - set(self._raw_to_global)
        if missing:
            raise ValueError(f"Unknown raw labels: {sorted(missing)}")
        return torch.as_tensor(
            [self._raw_to_global[label] for label in raw],
            dtype=torch.long,
            device=device,
        )

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return mask_unseen_logits(
            logits,
            self.active_class_indices,
            total_classes=self.total_classes,
        )

    def classification_loss(
        self,
        logits: torch.Tensor,
        global_labels: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        return fixed_head_cross_entropy(
            logits,
            global_labels,
            self.active_class_indices,
            total_classes=self.total_classes,
            reduction=reduction,
        )

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Predict global class indices without allowing future classes."""

        return self.mask_logits(logits).argmax(dim=1)

    def validate_linear_head(self, head: torch.nn.Linear) -> None:
        if not isinstance(head, torch.nn.Linear):
            raise TypeError("The current adapter supports a torch.nn.Linear head.")
        if head.out_features != self.total_classes:
            raise ValueError(
                f"Policy has {self.total_classes} classes but head has "
                f"{head.out_features} outputs."
            )

    def capture_inactive_rows(self, head: torch.nn.Linear) -> InactiveLinearRows:
        """Snapshot unseen rows immediately before ``optimizer.step``."""

        self.validate_linear_head(head)
        active = active_class_mask(
            self.active_class_indices,
            total_classes=self.total_classes,
            device=head.weight.device,
        )
        indices = (~active).nonzero(as_tuple=False).flatten()
        return InactiveLinearRows(
            indices=indices,
            weight=head.weight.detach()[indices].clone(),
            bias=None if head.bias is None else head.bias.detach()[indices].clone(),
        )

    @staticmethod
    def restore_inactive_rows(
        head: torch.nn.Linear,
        snapshot: InactiveLinearRows,
    ) -> None:
        """Undo gradient, momentum and decoupled-weight-decay row changes."""

        if snapshot.indices.numel() == 0:
            return
        with torch.no_grad():
            head.weight.index_copy_(0, snapshot.indices, snapshot.weight)
            if head.bias is not None and snapshot.bias is not None:
                head.bias.index_copy_(0, snapshot.indices, snapshot.bias)

    def optimizer_step(self, optimizer, head: torch.nn.Linear) -> None:
        """Step the optimizer while keeping unseen TDDI rows bitwise frozen."""

        snapshot = self.capture_inactive_rows(head)
        optimizer.step()
        self.restore_inactive_rows(head, snapshot)
