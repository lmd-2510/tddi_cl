"""Replay helper."""

from __future__ import annotations

import numpy as np


METHOD_NAME = "replay"


def build_training_arrays(
    current_features: np.ndarray,
    current_raw_labels: np.ndarray,
    replay_features: np.ndarray,
    replay_raw_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if replay_features.size == 0:
        return current_features, current_raw_labels
    features = np.concatenate([current_features, replay_features], axis=0)
    labels = np.concatenate([current_raw_labels, replay_raw_labels], axis=0)
    return features, labels
