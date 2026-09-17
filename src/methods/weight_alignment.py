"""Post-task classifier Weight Aligning (WA) for class-incremental learning."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn


WEIGHT_ALIGNMENT_NONE = "none"
WEIGHT_ALIGNMENT_NEW_CLASS_MEAN_NORM = "new_class_mean_norm_v1"
WEIGHT_ALIGNMENT_POLICIES = (
    WEIGHT_ALIGNMENT_NONE,
    WEIGHT_ALIGNMENT_NEW_CLASS_MEAN_NORM,
)


def align_new_class_weights_(
    model: nn.Module,
    previous_class_map: Mapping[int, int],
    seen_class_map: Mapping[int, int],
    *,
    policy: str,
) -> dict[str, object]:
    """Align new classifier-row norms to old rows, modifying ``model`` in place.

    This is the standard post-task WA operation: only classifier weights for
    newly introduced classes are scaled. Classifier bias is deliberately left
    unchanged. The returned dictionary is JSON-safe and suitable for task audit.
    """

    if policy not in WEIGHT_ALIGNMENT_POLICIES:
        raise ValueError(f"Unsupported weight-alignment policy: {policy}")
    if not hasattr(model, "head") or not isinstance(model.head, nn.Linear):
        raise TypeError("Weight alignment requires model.head to be torch.nn.Linear.")
    if not set(previous_class_map).issubset(seen_class_map):
        raise ValueError("Previous class map must be a subset of the seen class map.")
    if sorted(seen_class_map.values()) != list(range(len(seen_class_map))):
        raise ValueError("Seen class-map rows must be contiguous from zero.")
    if model.head.out_features != len(seen_class_map):
        raise ValueError("Classifier width does not match seen_class_map.")

    old_raw = sorted(previous_class_map, key=seen_class_map.__getitem__)
    new_raw = sorted(set(seen_class_map) - set(previous_class_map), key=seen_class_map.__getitem__)
    audit: dict[str, object] = {
        "schema_version": 1,
        "policy": policy,
        "semantics": "scale_new_classifier_weight_rows_to_old_mean_l2_norm",
        "bias_scaled": False,
        "old_class_count": len(old_raw),
        "new_class_count": len(new_raw),
        "old_raw_class_ids": old_raw,
        "new_raw_class_ids": new_raw,
        "applied": False,
        "skip_reason": None,
        "old_mean_weight_norm": None,
        "new_mean_weight_norm_before": None,
        "gamma": None,
        "new_mean_weight_norm_after": None,
    }
    if policy == WEIGHT_ALIGNMENT_NONE:
        audit["skip_reason"] = "disabled"
        return audit
    if not old_raw:
        audit["skip_reason"] = "no_old_classes_at_task_0"
        return audit
    if not new_raw:
        raise ValueError("Weight alignment requires at least one newly introduced class.")

    weight = model.head.weight
    old_rows = torch.as_tensor(
        [seen_class_map[raw] for raw in old_raw], device=weight.device, dtype=torch.long
    )
    new_rows = torch.as_tensor(
        [seen_class_map[raw] for raw in new_raw], device=weight.device, dtype=torch.long
    )
    with torch.no_grad():
        old_mean = weight.index_select(0, old_rows).norm(p=2, dim=1).mean()
        new_mean = weight.index_select(0, new_rows).norm(p=2, dim=1).mean()
        old_value, new_value = float(old_mean.item()), float(new_mean.item())
        if not math.isfinite(old_value) or not math.isfinite(new_value):
            raise FloatingPointError("Classifier row norms are non-finite before weight alignment.")
        if old_value <= 0 or new_value <= 0:
            raise ValueError("Classifier row norms must be positive for weight alignment.")
        gamma = old_mean / new_mean
        gamma_value = float(gamma.item())
        if not math.isfinite(gamma_value) or gamma_value <= 0:
            raise FloatingPointError("Weight-alignment scale is not finite and positive.")
        weight[new_rows] *= gamma
        new_after = float(weight.index_select(0, new_rows).norm(p=2, dim=1).mean().item())

    audit.update(
        {
            "applied": True,
            "old_mean_weight_norm": old_value,
            "new_mean_weight_norm_before": new_value,
            "gamma": gamma_value,
            "new_mean_weight_norm_after": new_after,
        }
    )
    return audit
