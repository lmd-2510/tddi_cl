"""Random seed helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None


SEED_DERIVATION = "single_model_identity_v1"
_MAX_NUMPY_SEED = 2**32 - 1


@dataclass(frozen=True)
class SeedConfiguration:
    """One fixed experiment/training seed for a single-model trajectory."""

    experiment_seed: int
    training_seed: int
    derivation: str


def _validate_seed(name: str, value: int) -> int:
    value = int(value)
    if value < 0 or value > _MAX_NUMPY_SEED:
        raise ValueError(f"{name} must be in [0, {_MAX_NUMPY_SEED}], got {value}.")
    return value


def resolve_seed_configuration(seed: int) -> SeedConfiguration:
    """Resolve the one seed shared by a single-model experiment."""
    experiment_seed = _validate_seed("seed", seed)
    return SeedConfiguration(
        experiment_seed=experiment_seed,
        training_seed=experiment_seed,
        derivation=SEED_DERIVATION,
    )


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and Torch when available."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_configured_seeds(configuration: SeedConfiguration) -> None:
    """Seed Python, NumPy and Torch for one single-model trajectory."""

    random.seed(configuration.experiment_seed)
    np.random.seed(configuration.experiment_seed)
    os.environ["PYTHONHASHSEED"] = str(configuration.experiment_seed)

    if torch is not None:
        torch.manual_seed(configuration.training_seed)
        torch.cuda.manual_seed_all(configuration.training_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
