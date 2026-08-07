"""Validated multiclass calibration metrics and temperature scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationMetrics:
    accuracy: float
    ece: float
    brier_score: float
    negative_log_likelihood: float
    mean_confidence: float
    high_confidence_error_rate: float | None
    high_confidence_count: int
    num_samples: int
    num_classes: int


@dataclass(frozen=True)
class TemperatureFitResult:
    temperature: float
    inverse_temperature: float
    negative_log_likelihood: float
    iterations: int
    converged: bool
    boundary: str


def _validate_probabilities_and_labels(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [non-empty rows, at least two classes].")
    if labels.shape != (probabilities.shape[0],):
        raise ValueError("labels must be one-dimensional and align with probabilities.")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinite values.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must lie in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("probability rows must sum to one.")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels must be dense probability-column indices.")
    return probabilities, labels


def _validate_logits_and_labels(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [non-empty rows, at least two classes].")
    if labels.shape != (logits.shape[0],):
        raise ValueError("labels must be one-dimensional and align with logits.")
    if not np.issubdtype(logits.dtype, np.floating) or not np.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values.")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels must be dense logit-column indices.")
    return logits, labels


def map_raw_labels_to_indices(labels: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """Map non-contiguous raw DDI class IDs to their logit-column indices."""

    labels = np.asarray(labels, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    if labels.ndim != 1 or class_ids.ndim != 1 or class_ids.shape[0] < 2:
        raise ValueError("labels and class_ids must be one-dimensional.")
    if np.unique(class_ids).shape[0] != class_ids.shape[0]:
        raise ValueError("class_ids must be unique.")
    index_by_class = {int(class_id): index for index, class_id in enumerate(class_ids)}
    try:
        return np.fromiter(
            (index_by_class[int(label)] for label in labels),
            dtype=np.int64,
            count=labels.shape[0],
        )
    except KeyError as error:
        raise ValueError(f"Label {int(error.args[0])} is absent from class_ids.") from error


def softmax_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2 or not np.issubdtype(logits.dtype, np.floating):
        raise ValueError("logits must be a two-dimensional floating-point array.")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values.")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    probabilities, labels = _validate_probabilities_and_labels(probabilities, labels)
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = predictions == labels
    bin_indices = np.minimum((confidences * num_bins).astype(np.int64), num_bins - 1)
    counts = np.bincount(bin_indices, minlength=num_bins).astype(np.float64)
    confidence_sums = np.bincount(bin_indices, weights=confidences, minlength=num_bins)
    accuracy_sums = np.bincount(bin_indices, weights=correctness, minlength=num_bins)
    occupied = counts > 0
    gaps = np.abs(accuracy_sums[occupied] / counts[occupied] - confidence_sums[occupied] / counts[occupied])
    return float(np.sum(counts[occupied] * gaps) / labels.shape[0])


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities, labels = _validate_probabilities_and_labels(probabilities, labels)
    true_probabilities = probabilities[np.arange(labels.shape[0]), labels]
    per_sample = np.sum(probabilities * probabilities, axis=1) - 2.0 * true_probabilities + 1.0
    return float(per_sample.mean())


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probabilities, labels = _validate_probabilities_and_labels(probabilities, labels)
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    clipped = np.clip(probabilities[np.arange(labels.shape[0]), labels], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def compute_calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    temperature: float = 1.0,
    num_bins: int = 15,
    high_confidence_threshold: float = 0.9,
    batch_size: int = 8192,
) -> CalibrationMetrics:
    """Compute metrics without materializing probabilities for the whole shard."""

    logits, labels = _validate_logits_and_labels(logits, labels)
    if num_bins <= 0 or batch_size <= 0:
        raise ValueError("num_bins and batch_size must be positive.")
    if not 0.0 <= high_confidence_threshold <= 1.0:
        raise ValueError("high_confidence_threshold must lie in [0, 1].")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")

    total_correct = 0
    total_nll = 0.0
    total_brier = 0.0
    total_confidence = 0.0
    high_count = 0
    high_errors = 0
    bin_counts = np.zeros(num_bins, dtype=np.int64)
    bin_confidence_sums = np.zeros(num_bins, dtype=np.float64)
    bin_correct_sums = np.zeros(num_bins, dtype=np.float64)

    for start in range(0, labels.shape[0], batch_size):
        stop = min(start + batch_size, labels.shape[0])
        batch_labels = labels[start:stop]
        probabilities = softmax_probabilities(logits[start:stop], temperature)
        rows = np.arange(stop - start)
        predictions = probabilities.argmax(axis=1)
        correct = predictions == batch_labels
        confidence = probabilities[rows, predictions]
        true_probability = probabilities[rows, batch_labels]
        total_correct += int(correct.sum())
        total_nll += float(-np.log(np.clip(true_probability, 1e-300, 1.0)).sum())
        total_brier += float(
            (np.sum(probabilities * probabilities, axis=1) - 2.0 * true_probability + 1.0).sum()
        )
        total_confidence += float(confidence.sum())
        high_mask = confidence >= high_confidence_threshold
        high_count += int(high_mask.sum())
        high_errors += int(np.logical_and(high_mask, ~correct).sum())
        bin_indices = np.minimum((confidence * num_bins).astype(np.int64), num_bins - 1)
        bin_counts += np.bincount(bin_indices, minlength=num_bins)
        bin_confidence_sums += np.bincount(
            bin_indices, weights=confidence, minlength=num_bins
        )
        bin_correct_sums += np.bincount(bin_indices, weights=correct, minlength=num_bins)

    occupied = bin_counts > 0
    gaps = np.abs(
        bin_correct_sums[occupied] / bin_counts[occupied]
        - bin_confidence_sums[occupied] / bin_counts[occupied]
    )
    num_samples = labels.shape[0]
    ece = float(np.sum(bin_counts[occupied] * gaps) / num_samples)
    high_error_rate = float(high_errors / high_count) if high_count else None
    return CalibrationMetrics(
        accuracy=float(total_correct / num_samples),
        ece=ece,
        brier_score=float(total_brier / num_samples),
        negative_log_likelihood=float(total_nll / num_samples),
        mean_confidence=float(total_confidence / num_samples),
        high_confidence_error_rate=high_error_rate,
        high_confidence_count=high_count,
        num_samples=num_samples,
        num_classes=logits.shape[1],
    )


def _temperature_objective_stats(
    logits: np.ndarray,
    labels: np.ndarray,
    inverse_temperature: float,
    batch_size: int,
) -> tuple[float, float, float]:
    total_nll = 0.0
    total_gradient = 0.0
    total_hessian = 0.0
    for start in range(0, labels.shape[0], batch_size):
        stop = min(start + batch_size, labels.shape[0])
        batch_logits = np.asarray(logits[start:stop], dtype=np.float64)
        batch_labels = labels[start:stop]
        scaled = batch_logits * inverse_temperature
        maxima = scaled.max(axis=1, keepdims=True)
        exponentials = np.exp(scaled - maxima)
        denominators = exponentials.sum(axis=1, keepdims=True)
        probabilities = exponentials / denominators
        expected_logits = np.sum(probabilities * batch_logits, axis=1)
        true_logits = batch_logits[np.arange(stop - start), batch_labels]
        centered = batch_logits - expected_logits[:, None]
        variances = np.sum(probabilities * centered * centered, axis=1)
        total_nll += float(
            (np.log(denominators[:, 0]) + maxima[:, 0] - inverse_temperature * true_logits).sum()
        )
        total_gradient += float((expected_logits - true_logits).sum())
        total_hessian += float(variances.sum())
    count = labels.shape[0]
    return total_nll / count, total_gradient / count, total_hessian / count


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 100.0,
    max_iterations: int = 20,
    tolerance: float = 1e-7,
    batch_size: int = 8192,
) -> TemperatureFitResult:
    """Fit scalar temperature by safeguarded Newton optimization of validation NLL."""

    logits, labels = _validate_logits_and_labels(logits, labels)
    if not 0.0 < min_temperature < max_temperature:
        raise ValueError("temperature bounds must satisfy 0 < min < max.")
    if max_iterations <= 0 or tolerance <= 0.0 or batch_size <= 0:
        raise ValueError("optimizer controls must be positive.")

    beta_low = 1.0 / max_temperature
    beta_high = 1.0 / min_temperature
    low_nll, low_gradient, _ = _temperature_objective_stats(
        logits, labels, beta_low, batch_size
    )
    if low_gradient >= 0.0:
        return TemperatureFitResult(
            temperature=max_temperature,
            inverse_temperature=beta_low,
            negative_log_likelihood=low_nll,
            iterations=0,
            converged=True,
            boundary="max_temperature",
        )
    high_nll, high_gradient, _ = _temperature_objective_stats(
        logits, labels, beta_high, batch_size
    )
    if high_gradient <= 0.0:
        return TemperatureFitResult(
            temperature=min_temperature,
            inverse_temperature=beta_high,
            negative_log_likelihood=high_nll,
            iterations=0,
            converged=True,
            boundary="min_temperature",
        )

    beta = float(np.clip(1.0, beta_low, beta_high))
    converged = False
    iterations = 0
    nll = float("nan")
    for iterations in range(1, max_iterations + 1):
        nll, gradient, hessian = _temperature_objective_stats(
            logits, labels, beta, batch_size
        )
        if abs(gradient) <= tolerance:
            converged = True
            break
        if gradient < 0.0:
            beta_low = beta
        else:
            beta_high = beta
        candidate = beta - gradient / hessian if hessian > 1e-14 else float("nan")
        if not np.isfinite(candidate) or not beta_low < candidate < beta_high:
            candidate = 0.5 * (beta_low + beta_high)
        if abs(candidate - beta) <= tolerance * max(1.0, abs(beta)):
            beta = candidate
            converged = True
            break
        beta = candidate

    nll, final_gradient, _ = _temperature_objective_stats(logits, labels, beta, batch_size)
    converged = converged or abs(final_gradient) <= tolerance * 10.0
    return TemperatureFitResult(
        temperature=float(1.0 / beta),
        inverse_temperature=beta,
        negative_log_likelihood=nll,
        iterations=iterations,
        converged=converged,
        boundary="none",
    )


def calibration_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> pd.DataFrame:
    probabilities, labels = _validate_probabilities_and_labels(probabilities, labels)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    return pd.DataFrame(
        [
            {
                "ece": expected_calibration_error(probabilities, labels, num_bins=num_bins),
                "brier_score": multiclass_brier_score(probabilities, labels),
                "negative_log_likelihood": negative_log_likelihood(probabilities, labels),
                "mean_confidence": float(confidence.mean()),
                "high_confidence_error_rate": float(
                    np.mean(predictions[confidence >= 0.9] != labels[confidence >= 0.9])
                )
                if np.any(confidence >= 0.9)
                else float("nan"),
                "num_bins": num_bins,
            }
        ]
    )
