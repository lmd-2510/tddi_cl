"""Online Elastic Weight Consolidation for continual DDI2025-CIL experiments."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    nn = None


METHOD_NAME = "ewc"


def compute_fisher(model, loader, device: str) -> dict[str, "torch.Tensor"]:
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
            new_tensor[current_index] = old_state[name][previous_index]
        grown[name] = new_tensor
    return grown
