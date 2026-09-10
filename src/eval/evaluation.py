"""Model evaluation and optional prediction collection for CIL experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    F = None
    nn = None
    DataLoader = None

from src.eval.metrics import compute_classification_metrics, compute_classwise_metrics


@dataclass(frozen=True)
class PredictionOutputs:
    """Per-sample model outputs using raw class IDs for labels and predictions."""

    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    latent_features: np.ndarray
    class_ids: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics plus optional class-wise and per-sample outputs."""

    metrics: dict[str, float]
    classwise_metrics: pd.DataFrame | None = None
    outputs: PredictionOutputs | None = None


def _ordered_raw_class_ids(inverse_seen_map: dict[int, int]) -> np.ndarray:
    indices = sorted(inverse_seen_map)
    if indices != list(range(len(indices))):
        raise ValueError("inverse_seen_map must contain dense indices 0..C-1.")
    return np.asarray([inverse_seen_map[index] for index in indices], dtype=np.int64)


def evaluate_model(
    model: "nn.Module",
    loader: "DataLoader",
    criterion: "nn.Module",
    device: str,
    inverse_seen_map: dict[int, int],
    *,
    evaluation_class_indices: list[int] | None = None,
    include_classwise: bool = False,
    collect_outputs: bool = False,
) -> EvaluationResult:
    """Evaluate a model and optionally retain prediction-level arrays.

    The loader must yield dense local labels. Collected labels and predictions are
    converted back to raw class IDs using ``inverse_seen_map``.
    """

    if torch is None or F is None:
        raise ImportError("torch is required to evaluate CIL models.")
    if collect_outputs and not hasattr(model, "forward_with_latent"):
        raise TypeError("collect_outputs requires model.forward_with_latent().")

    class_ids = _ordered_raw_class_ids(inverse_seen_map)
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_latent: list[np.ndarray] = []

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(device)
                labels = labels.to(device)
                if collect_outputs:
                    logits, latent = model.forward_with_latent(features)
                else:
                    logits = model(features)
                    latent = None
                loss = criterion(logits, labels)
                batch_size = labels.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size
                predictions = logits.argmax(dim=1)
                probabilities = F.softmax(logits, dim=1)
                all_true.append(labels.cpu().numpy())
                all_pred.append(predictions.cpu().numpy())
                if include_classwise or collect_outputs:
                    all_probabilities.append(probabilities.cpu().numpy())
                if collect_outputs:
                    all_logits.append(logits.cpu().numpy())
                    if latent is None:  # pragma: no cover - guarded by branch above
                        raise RuntimeError("Latent outputs were not returned by the model.")
                    all_latent.append(latent.cpu().numpy())
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty loader.")

    y_true = np.concatenate(all_true).astype(np.int64, copy=False)
    y_pred = np.concatenate(all_pred).astype(np.int64, copy=False)
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=evaluation_class_indices,
    )
    metrics["loss"] = total_loss / total_examples

    probabilities_array: np.ndarray | None = None
    if include_classwise or collect_outputs:
        probabilities_array = np.concatenate(all_probabilities).astype(np.float32, copy=False)

    classwise = None
    if include_classwise:
        if probabilities_array is None:  # pragma: no cover - guarded above
            raise RuntimeError("Class-wise evaluation requires probabilities.")
        classwise = compute_classwise_metrics(
            y_true,
            y_pred,
            probabilities_array,
            class_indices=sorted(inverse_seen_map),
            inverse_class_map=inverse_seen_map,
        )

    outputs = None
    if collect_outputs:
        if probabilities_array is None:  # pragma: no cover - guarded above
            raise RuntimeError("Prediction collection requires probabilities.")
        if np.any(y_true < 0) or np.any(y_true >= class_ids.shape[0]):
            raise ValueError("Evaluation labels fall outside seen-class output columns.")
        if np.any(y_pred < 0) or np.any(y_pred >= class_ids.shape[0]):
            raise ValueError("Predictions fall outside seen-class output columns.")
        outputs = PredictionOutputs(
            logits=np.concatenate(all_logits).astype(np.float32, copy=False),
            probabilities=probabilities_array,
            predictions=class_ids[y_pred],
            labels=class_ids[y_true],
            latent_features=np.concatenate(all_latent).astype(np.float32, copy=False),
            class_ids=class_ids,
        )

    return EvaluationResult(
        metrics=metrics,
        classwise_metrics=classwise,
        outputs=outputs,
    )
