"""Sequential fine-tuning helper."""

from __future__ import annotations

import numpy as np


METHOD_NAME = "sequential"


def build_training_arrays(
    current_features: np.ndarray,
    current_raw_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return current_features, current_raw_labels
