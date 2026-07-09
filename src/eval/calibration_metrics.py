"""Calibration metrics for DDI2025-CIL."""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        if right == 1.0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not np.any(mask):
            continue
        bin_confidence = float(confidences[mask].mean())
        bin_accuracy = float(correctness[mask].mean())
        ece += (np.sum(mask) / labels.shape[0]) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    clipped = np.clip(probabilities[np.arange(labels.shape[0]), labels], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def calibration_summary(probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ece": expected_calibration_error(probabilities, labels, num_bins=num_bins),
                "brier_score": multiclass_brier_score(probabilities, labels),
                "negative_log_likelihood": negative_log_likelihood(probabilities, labels),
                "num_bins": num_bins,
            }
        ]
    )
