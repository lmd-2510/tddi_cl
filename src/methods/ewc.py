"""EWC placeholder for future continual regularization experiments."""

from __future__ import annotations


METHOD_NAME = "ewc"


def not_implemented_message() -> str:
    return (
        "EWC is planned but not implemented yet in this workspace. "
        "Replay-based methods are the current supported continual baselines."
    )
