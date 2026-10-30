"""Paper Task Arithmetic recipe for the DDI-CIL neural merge scope."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from src.methods.model_merging import (
    TASK_ARITHMETIC_PAPER_SCALE,
    StateDict,
    task_arithmetic_state,
)


def merge_task_arithmetic(
    base: StateDict,
    experts: Sequence[Mapping[str, torch.Tensor]],
    *,
    parameter_keys: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Merge expert deltas with the locked paper scaling coefficient 0.4."""

    return task_arithmetic_state(
        base,
        experts,
        keys=parameter_keys,
        scale=TASK_ARITHMETIC_PAPER_SCALE,
    )


__all__ = ["merge_task_arithmetic"]
