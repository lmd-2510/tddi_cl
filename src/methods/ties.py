"""Paper TIES recipe for the DDI-CIL neural merge scope."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from src.methods.model_merging import (
    StateDict,
    TIES_PAPER_DENSITY,
    TIES_PAPER_SCALE,
    ties_state,
)


def merge_ties(
    base: StateDict,
    experts: Sequence[Mapping[str, torch.Tensor]],
    *,
    parameter_keys: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Run top-20%, mass-sign, disjoint-mean TIES with lambda one."""

    return ties_state(
        base,
        experts,
        keys=parameter_keys,
        density=TIES_PAPER_DENSITY,
        scale=TIES_PAPER_SCALE,
    )


__all__ = ["merge_ties"]
