"""Averaged Gradient Episodic Memory projection.

Reference: https://arxiv.org/abs/1812.00420.
"""

from __future__ import annotations

import torch


METHOD_NAME = "agem"


def project_agem_gradient(
    current_gradient: torch.Tensor,
    reference_gradient: torch.Tensor,
    *,
    denominator_epsilon: float = 0.0,
) -> torch.Tensor:
    """Project ``current_gradient`` onto the A-GEM reference half-space.

    When ``g · g_ref < 0``, A-GEM uses the closed-form projection

    ``g - (g · g_ref / (g_ref · g_ref)) * g_ref``.
    """

    if current_gradient.ndim != 1 or reference_gradient.ndim != 1:
        raise ValueError("A-GEM gradients must be one-dimensional.")
    if current_gradient.shape != reference_gradient.shape:
        raise ValueError("Current and reference gradients must have the same shape.")
    if denominator_epsilon < 0:
        raise ValueError("denominator_epsilon must be non-negative.")
    if not current_gradient.is_floating_point() or not reference_gradient.is_floating_point():
        raise ValueError("A-GEM gradients must use a floating-point dtype.")
    if not bool(torch.isfinite(current_gradient).all()) or not bool(
        torch.isfinite(reference_gradient).all()
    ):
        raise ValueError("A-GEM gradients must be finite.")

    dot_product = torch.dot(current_gradient, reference_gradient)
    if bool(dot_product >= 0):
        return current_gradient.clone()

    reference_squared_norm = torch.dot(reference_gradient, reference_gradient)
    if bool(reference_squared_norm <= denominator_epsilon):
        # A zero reference gradient cannot produce a genuinely negative dot
        # product; this guard only protects against extreme numerical noise.
        return current_gradient.clone()
    return current_gradient - (dot_product / reference_squared_norm) * reference_gradient
