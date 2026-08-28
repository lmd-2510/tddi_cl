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


MEMBER_SEED_DERIVATION = "numpy_seedsequence_v1"
LEGACY_SEED_DERIVATION = "legacy_identity_v1"
_MAX_NUMPY_SEED = 2**32 - 1


@dataclass(frozen=True)
class SeedConfiguration:
    """Resolved experiment/member seed roles for one training trajectory."""

    experiment_seed: int
    member_id: int | None
    member_seed: int
    mode: str
    derivation: str


def _validate_seed(name: str, value: int) -> int:
    value = int(value)
    if value < 0 or value > _MAX_NUMPY_SEED:
        raise ValueError(f"{name} must be in [0, {_MAX_NUMPY_SEED}], got {value}.")
    return value


def derive_member_seed(experiment_seed: int, member_id: int) -> int:
    """Derive a stable member RNG seed without changing experiment artifacts."""

    experiment_seed = _validate_seed("experiment_seed", experiment_seed)
    member_id = _validate_seed("member_id", member_id)
    sequence = np.random.SeedSequence(
        [experiment_seed, member_id, 0x444449, 0x43494C]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def resolve_seed_configuration(seed: int, member_id: int | None = None) -> SeedConfiguration:
    """Resolve legacy ``--seed`` or an independent ensemble-member stream.

    ``--seed`` remains the experiment seed. Omitting ``member_id`` preserves the
    legacy identity mapping so existing commands retain their exact RNG behavior.
    """

    experiment_seed = _validate_seed("seed", seed)
    if member_id is None:
        return SeedConfiguration(
            experiment_seed=experiment_seed,
            member_id=None,
            member_seed=experiment_seed,
            mode="legacy",
            derivation=LEGACY_SEED_DERIVATION,
        )
    member_id = _validate_seed("member_id", member_id)
    return SeedConfiguration(
        experiment_seed=experiment_seed,
        member_id=member_id,
        member_seed=derive_member_seed(experiment_seed, member_id),
        mode="member",
        derivation=MEMBER_SEED_DERIVATION,
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
    """Seed shared experiment RNGs and member-specific Torch RNGs.

    Python/NumPy retain the experiment stream for shared artifacts. Torch uses the
    member stream for initialization, dropout, and DataLoader sampling. In legacy
    mode both values are identical, matching ``set_global_seed`` behavior.
    """

    random.seed(configuration.experiment_seed)
    np.random.seed(configuration.experiment_seed)
    os.environ["PYTHONHASHSEED"] = str(configuration.experiment_seed)

    if torch is not None:
        torch.manual_seed(configuration.member_seed)
        torch.cuda.manual_seed_all(configuration.member_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
