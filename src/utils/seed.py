"""Random seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None


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
