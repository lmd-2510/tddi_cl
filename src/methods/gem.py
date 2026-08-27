"""Gradient Episodic Memory primitives.

This module keeps the task-indexed episodic memory required by GEM and
implements the dual quadratic program from Lopez-Paz and Ranzato (2017):
https://arxiv.org/abs/1706.08840.
The training engine remains responsible for computing model-specific losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize


METHOD_NAME = "gem"


@dataclass
class TaskEpisodicMemory:
    """Fixed-budget memory with a deterministic, separate quota per task.

    GEM assumes the total number of tasks is known and assigns ``M / T``
    locations to every task. Remainder locations are assigned to the earliest
    task IDs. This keeps the final memory within the fixed global budget while
    ensuring that adding a new task never changes samples retained for an old
    task.
    """

    total_budget: int
    total_tasks: int
    random_seed: int = 0
    features_by_task: dict[int, np.ndarray] = field(default_factory=dict)
    labels_by_task: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_budget <= 0:
            raise ValueError("total_budget must be positive.")
        if self.total_tasks <= 0:
            raise ValueError("total_tasks must be positive.")
        if self.total_budget < self.total_tasks:
            raise ValueError(
                "total_budget must provide at least one memory location per task."
            )

    @property
    def task_ids(self) -> list[int]:
        return sorted(self.features_by_task)

    @property
    def total_size(self) -> int:
        return int(sum(block.shape[0] for block in self.features_by_task.values()))

    def quota_for_task(self, task_id: int) -> int:
        """Return the fixed quota assigned to one task."""

        if task_id < 0 or task_id >= self.total_tasks:
            raise ValueError(
                f"task_id must be in [0, {self.total_tasks}), got {task_id}."
            )
        base, remainder = divmod(self.total_budget, self.total_tasks)
        return base + int(task_id < remainder)

    def update(
        self,
        task_id: int,
        features: np.ndarray,
        raw_labels: np.ndarray,
    ) -> None:
        """Store a deterministic uniform subset from a newly completed task."""

        task_id = int(task_id)
        if task_id in self.features_by_task:
            raise ValueError(f"Episodic memory received task {task_id} twice.")

        features = np.asarray(features, dtype=np.float32)
        raw_labels = np.asarray(raw_labels, dtype=np.int64)
        if features.ndim != 2 or raw_labels.shape != (features.shape[0],):
            raise ValueError(
                "features and raw_labels must be aligned two-dimensional/one-dimensional arrays."
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot populate episodic memory from an empty task.")
        if not np.isfinite(features).all():
            raise ValueError("Episodic memory features must be finite.")

        quota = min(self.quota_for_task(task_id), features.shape[0])
        if quota == features.shape[0]:
            selected = np.arange(features.shape[0], dtype=np.int64)
        elif quota == 0:
            selected = np.empty((0,), dtype=np.int64)
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.random_seed, task_id])
            )
            # Sorting retains source-row order after the seeded uniform draw.
            selected = np.sort(
                rng.choice(features.shape[0], size=quota, replace=False)
            ).astype(np.int64, copy=False)

        self.features_by_task[task_id] = features[selected].astype(
            np.float32, copy=True
        )
        self.labels_by_task[task_id] = raw_labels[selected].astype(
            np.int64, copy=True
        )
        if self.total_size > self.total_budget:
            raise RuntimeError("Episodic memory exceeded its fixed total budget.")

    def get_task(self, task_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the retained inputs and raw labels for one previous task."""

        task_id = int(task_id)
        if task_id not in self.features_by_task:
            raise KeyError(f"Task {task_id} is not present in episodic memory.")
        return self.features_by_task[task_id], self.labels_by_task[task_id]

    def get_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the union of all task memories in task order."""

        nonempty_tasks = [
            task_id
            for task_id in self.task_ids
            if self.features_by_task[task_id].shape[0] > 0
        ]
        if not nonempty_tasks:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return (
            np.concatenate(
                [self.features_by_task[task_id] for task_id in nonempty_tasks],
                axis=0,
            ),
            np.concatenate(
                [self.labels_by_task[task_id] for task_id in nonempty_tasks],
                axis=0,
            ),
        )

    def sample(
        self,
        batch_size: int,
        *,
        seed_components: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample an A-GEM reference batch without replacement."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        features, labels = self.get_all()
        if labels.shape[0] == 0:
            raise ValueError("Cannot sample an empty episodic memory.")
        sample_size = min(int(batch_size), labels.shape[0])
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.random_seed, *(int(value) for value in seed_components)]
            )
        )
        indices = rng.choice(labels.shape[0], size=sample_size, replace=False)
        return features[indices], labels[indices]

    def summary_frame(self) -> pd.DataFrame:
        """Build a task-level memory audit table."""

        columns = [
            "task_id",
            "task_quota",
            "memory_count",
            "class_count",
            "total_memory_budget",
        ]
        rows = []
        for task_id in self.task_ids:
            labels = self.labels_by_task[task_id]
            rows.append(
                {
                    "task_id": task_id,
                    "task_quota": self.quota_for_task(task_id),
                    "memory_count": int(labels.shape[0]),
                    "class_count": int(np.unique(labels).shape[0]),
                    "total_memory_budget": self.total_budget,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def save_summary(self, path: str | Path) -> None:
        self.summary_frame().to_csv(path, index=False)

    def save_snapshot(self, path: str | Path) -> None:
        """Save a task-labelled snapshot compatible with existing memory artifacts."""

        rows: list[pd.DataFrame] = []
        for task_id in self.task_ids:
            features, labels = self.get_task(task_id)
            if labels.shape[0] == 0:
                continue
            frame = pd.DataFrame(
                features,
                columns=[f"feature_{index}" for index in range(features.shape[1])],
            )
            frame.insert(0, "raw_class_id", labels)
            frame.insert(0, "task_id", task_id)
            rows.append(frame)
        if not rows:
            pd.DataFrame(columns=["task_id", "raw_class_id"]).to_parquet(
                path, index=False
            )
            return
        pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return trainable parameters in the stable order used by gradient vectors."""

    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def flatten_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Flatten parameter gradients, replacing unused gradients with zeros."""

    if gradients is not None and len(gradients) != len(parameters):
        raise ValueError("gradients and parameters must have the same length.")
    blocks: list[torch.Tensor] = []
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad if gradients is None else gradients[index]
        if gradient is None:
            blocks.append(torch.zeros_like(parameter).reshape(-1))
        else:
            blocks.append(gradient.detach().reshape(-1))
    if not blocks:
        raise ValueError("Cannot flatten gradients for an empty parameter list.")
    return torch.cat(blocks).clone()


def assign_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradient_vector: torch.Tensor,
) -> None:
    """Copy one flat vector back into ``parameter.grad`` tensors."""

    if gradient_vector.ndim != 1:
        raise ValueError("gradient_vector must be one-dimensional.")
    expected = sum(parameter.numel() for parameter in parameters)
    if gradient_vector.numel() != expected:
        raise ValueError(
            f"Gradient length mismatch: expected {expected}, got {gradient_vector.numel()}."
        )

    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        block = gradient_vector[offset : offset + count].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = block.clone()
        else:
            parameter.grad.copy_(block)
        offset += count


def project_gem_gradient(
    current_gradient: torch.Tensor,
    reference_gradients: torch.Tensor,
    *,
    feasibility_tolerance: float = 1e-6,
) -> torch.Tensor:
    """Project a gradient onto all GEM half-space constraints.

    The primal problem is ``min 0.5 ||z-g||²`` subject to ``G z >= 0``,
    where rows of ``G`` are gradients from individual previous tasks. Its dual
    has one non-negative variable per previous task and is solved on CPU with
    ``scipy.optimize``; the potentially large gradient vectors stay on their
    original PyTorch device.
    """

    if current_gradient.ndim != 1:
        raise ValueError("current_gradient must be one-dimensional.")
    if reference_gradients.ndim != 2:
        raise ValueError("reference_gradients must be two-dimensional.")
    if reference_gradients.shape[1] != current_gradient.numel():
        raise ValueError("Current and reference gradient widths do not match.")
    if reference_gradients.shape[0] == 0:
        return current_gradient.clone()
    if feasibility_tolerance < 0:
        raise ValueError("feasibility_tolerance must be non-negative.")
    if not current_gradient.is_floating_point() or not reference_gradients.is_floating_point():
        raise ValueError("GEM gradients must use a floating-point dtype.")
    if not bool(torch.isfinite(current_gradient).all()) or not bool(
        torch.isfinite(reference_gradients).all()
    ):
        raise ValueError("GEM gradients must be finite.")

    dots_before = reference_gradients @ current_gradient
    if bool(torch.all(dots_before >= 0)):
        return current_gradient.clone()

    # Zero reference rows impose no constraint and make the dual unnecessarily
    # singular, so exclude them from the small QP.
    nonzero = reference_gradients.square().sum(dim=1) > 0
    active_references = reference_gradients[nonzero]
    if active_references.shape[0] == 0:
        return current_gradient.clone()

    current_norm = torch.linalg.vector_norm(current_gradient)
    if bool(current_norm == 0):
        return current_gradient.clone()
    # Positive row scaling leaves every half-space unchanged. Normalizing the
    # current/reference vectors keeps the tiny dual QP well-conditioned without
    # materializing millions of float64 parameters.
    normalized_current = current_gradient / current_norm
    normalized_references = active_references / torch.linalg.vector_norm(
        active_references, dim=1, keepdim=True
    )

    gram = (
        (normalized_references @ normalized_references.T)
        .detach()
        .cpu()
        .double()
        .numpy()
    )
    linear = (
        (normalized_references @ normalized_current)
        .detach()
        .cpu()
        .double()
        .numpy()
    )
    # A tiny diagonal term resolves dual non-uniqueness for duplicate reference
    # gradients without materially changing the unique primal projection.
    diagonal_scale = max(1.0, float(np.max(np.diag(gram))))
    gram = gram + np.eye(gram.shape[0], dtype=np.float64) * (
        np.finfo(np.float64).eps * diagonal_scale
    )

    def objective(coefficients: np.ndarray) -> float:
        return float(0.5 * coefficients @ gram @ coefficients + linear @ coefficients)

    def jacobian(coefficients: np.ndarray) -> np.ndarray:
        return gram @ coefficients + linear

    result = minimize(
        objective,
        np.zeros(active_references.shape[0], dtype=np.float64),
        jac=jacobian,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * active_references.shape[0],
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"GEM dual QP failed: {result.message}")

    coefficients = torch.as_tensor(
        result.x,
        device=current_gradient.device,
        dtype=current_gradient.dtype,
    )
    projected = (
        normalized_current + normalized_references.T.mv(coefficients)
    ) * current_norm

    dots_after = reference_gradients @ projected
    gradient_norm = torch.linalg.vector_norm(projected)
    reference_norms = torch.linalg.vector_norm(reference_gradients, dim=1)
    allowed_error = feasibility_tolerance * torch.maximum(
        torch.ones_like(reference_norms),
        reference_norms * gradient_norm,
    )
    if bool(torch.any(dots_after < -allowed_error)):
        minimum = float(dots_after.min().detach().cpu())
        raise RuntimeError(
            "GEM projection returned an infeasible gradient; "
            f"minimum constraint dot product is {minimum:.6e}."
        )
    return projected
