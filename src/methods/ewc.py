"""Paper-aligned Elastic Weight Consolidation primitives.

The fixed 178-way head removes shape growth from EWC state.  Fisher entries for
unseen output rows remain zero because those logits are masked by the shared
class-incremental head policy.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    nn = None

from src.training.class_incremental_policy import fixed_head_cross_entropy


METHOD_NAME = "ewc"


def compute_fisher(
    model,
    loader,
    device: str,
    *,
    active_class_indices=None,
    total_classes: int = 178,
) -> dict[str, "torch.Tensor"]:
    """Empirical diagonal Fisher information (mean squared gradient of the
    cross-entropy log-likelihood) over the given loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    fisher = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
    }
    total = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        if active_class_indices is not None:
            logits = model(features)
            loss = fixed_head_cross_entropy(
                logits,
                labels,
                active_class_indices,
                total_classes=total_classes,
            )
        elif hasattr(model, "classification_loss"):
            _, loss = model.classification_loss(features, labels, criterion)
        else:
            logits = model(features)
            loss = criterion(logits, labels)
        loss.backward()
        batch_size = labels.shape[0]
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2 * batch_size
        total += batch_size
    for name in fisher:
        fisher[name] /= max(total, 1)
    model.zero_grad(set_to_none=True)
    return fisher


def ewc_penalty(model, fisher: dict[str, "torch.Tensor"], theta_star: dict[str, "torch.Tensor"]) -> "torch.Tensor":
    """Quadratic penalty sum(fisher * (theta - theta_star)^2) over matching params."""
    penalty = None
    for name, param in model.named_parameters():
        if name not in fisher or name not in theta_star:
            continue
        term = (fisher[name] * (param - theta_star[name]) ** 2).sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return penalty


def original_ewc_penalty(
    model,
    fisher: dict[str, "torch.Tensor"],
    theta_star: dict[str, "torch.Tensor"],
) -> "torch.Tensor":
    """Return the original EWC quadratic term ``1/2 Σ F_i(θ_i-θ*_i)^2``."""

    return 0.5 * ewc_penalty(model, fisher, theta_star)


def parameter_snapshot(model) -> dict[str, "torch.Tensor"]:
    """Freeze the task-boundary optimum used as the EWC anchor."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def consolidate_quadratic_penalties(
    previous_fisher: dict[str, "torch.Tensor"] | None,
    previous_theta_star: dict[str, "torch.Tensor"] | None,
    task_fisher: dict[str, "torch.Tensor"],
    task_theta_star: dict[str, "torch.Tensor"],
) -> tuple[dict[str, "torch.Tensor"], dict[str, "torch.Tensor"]]:
    """Combine per-task EWC penalties without storing every task copy.

    With a common ``lambda``, a sum of diagonal quadratic penalties is exactly
    represented by summed precision and its Fisher-weighted mean.  This is not
    the decayed online-EWC approximation: no previous Fisher mass is discarded.
    """

    if set(task_fisher) != set(task_theta_star):
        raise ValueError("Task Fisher and anchor parameters do not match.")
    if previous_fisher is None or previous_theta_star is None:
        return (
            {name: value.detach().cpu().clone() for name, value in task_fisher.items()},
            {name: value.detach().cpu().clone() for name, value in task_theta_star.items()},
        )
    if set(previous_fisher) != set(previous_theta_star) or set(previous_fisher) != set(task_fisher):
        raise ValueError("Previous and current EWC parameter sets do not match.")

    combined_fisher: dict[str, torch.Tensor] = {}
    combined_anchor: dict[str, torch.Tensor] = {}
    for name in task_fisher:
        old_fisher = previous_fisher[name].detach().cpu()
        old_anchor = previous_theta_star[name].detach().cpu()
        new_fisher = task_fisher[name].detach().cpu()
        new_anchor = task_theta_star[name].detach().cpu()
        if not (
            old_fisher.shape == old_anchor.shape == new_fisher.shape == new_anchor.shape
        ):
            raise ValueError(f"EWC tensor shape changed for parameter {name!r}.")
        precision = old_fisher + new_fisher
        numerator = old_fisher * old_anchor + new_fisher * new_anchor
        anchor = torch.where(precision > 0, numerator / precision.clamp_min(torch.finfo(precision.dtype).tiny), new_anchor)
        combined_fisher[name] = precision.clone()
        combined_anchor[name] = anchor.clone()
    return combined_fisher, combined_anchor


def grow_head_state(
    old_state: dict[str, "torch.Tensor"],
    previous_seen_map: dict[int, int],
    current_seen_map: dict[int, int],
    reference_state: dict[str, "torch.Tensor"],
    *,
    zero_new_rows: bool,
) -> dict[str, "torch.Tensor"]:
    """Grow head.weight/head.bias from the old (smaller) seen-class index
    space into the new (larger) one, keyed by raw class id, before training
    on a new task. Rows for classes not seen before are zero-filled (Fisher:
    no importance yet) or left at the freshly-initialized reference value
    (theta_star: harmless since Fisher is zero there). Backbone params are
    shape-invariant across tasks and pass through unchanged."""
    grown: dict[str, "torch.Tensor"] = {}
    for name, reference in reference_state.items():
        if name not in ("head.weight", "head.bias"):
            grown[name] = old_state[name]
            continue
        new_tensor = torch.zeros_like(reference) if zero_new_rows else reference.clone()
        for raw_class_id, previous_index in previous_seen_map.items():
            current_index = current_seen_map[raw_class_id]
            if reference_state["head.weight"].ndim == 3:
                new_tensor[..., current_index] = old_state[name][..., previous_index]
            else:
                new_tensor[current_index] = old_state[name][previous_index]
        grown[name] = new_tensor
    return grown
