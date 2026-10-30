"""Averaged Gradient Episodic Memory projection.

Reference: https://arxiv.org/abs/1812.00420.
"""

from __future__ import annotations
from typing import Sequence

import torch

from src.training.class_incremental_policy import fixed_head_cross_entropy


METHOD_NAME = "agem"


class TaskEpisodicMemory:
    """Uniform random episodic examples stored independently per old task.

    A-GEM does not optimize directly on these examples.  A random reference
    minibatch is used only to form the constraint gradient.
    """

    def __init__(self, examples_per_task: int, *, seed: int = 0) -> None:
        if type(examples_per_task) is not int or examples_per_task <= 0:
            raise ValueError("examples_per_task must be a positive integer.")
        self.examples_per_task = examples_per_task
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)
        self._tasks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return sum(labels.shape[0] for _, labels in self._tasks.values())

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tasks))

    def add_task(
        self,
        task_id: int,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Store a seeded uniform subset after completing one task."""

        task_id = int(task_id)
        if task_id in self._tasks:
            raise ValueError(f"Task {task_id} is already present in A-GEM memory.")
        if features.ndim < 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must be aligned batched tensors.")
        if labels.dtype != torch.long:
            raise ValueError("A-GEM labels must use torch.long.")
        count = min(self.examples_per_task, int(labels.shape[0]))
        indices = torch.randperm(labels.shape[0], generator=self._generator)[:count]
        self._tasks[task_id] = (
            features.detach().cpu()[indices].contiguous().clone(),
            labels.detach().cpu()[indices].contiguous().clone(),
        )

    def sample_reference(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample uniformly from the union of all previous-task memories."""

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not self._tasks:
            raise ValueError("Cannot sample empty A-GEM memory.")
        features = torch.cat([self._tasks[key][0] for key in self.task_ids], dim=0)
        labels = torch.cat([self._tasks[key][1] for key in self.task_ids], dim=0)
        count = min(batch_size, int(labels.shape[0]))
        indices = torch.randperm(labels.shape[0], generator=self._generator)[:count]
        return features[indices].to(device=device), labels[indices].to(device=device)

    def state_dict(self) -> dict[str, object]:
        return {
            "examples_per_task": self.examples_per_task,
            "seed": self.seed,
            "tasks": {
                task_id: (features.clone(), labels.clone())
                for task_id, (features, labels) in self._tasks.items()
            },
            "generator_state": self._generator.get_state().clone(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "TaskEpisodicMemory":
        memory = cls(int(state["examples_per_task"]), seed=int(state["seed"]))
        tasks = state.get("tasks")
        if not isinstance(tasks, dict):
            raise ValueError("A-GEM checkpoint task memory is missing.")
        for task_id, value in tasks.items():
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("Invalid A-GEM task-memory entry.")
            features, labels = value
            if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
                raise ValueError("Invalid A-GEM task-memory tensors.")
            memory._tasks[int(task_id)] = (features.cpu().clone(), labels.cpu().clone())
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("A-GEM checkpoint RNG state is missing.")
        memory._generator.set_state(generator_state)
        return memory


def reference_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_class_indices,
    *,
    total_classes: int = 178,
) -> torch.Tensor:
    """CE objective whose gradient defines the A-GEM reference constraint."""

    return fixed_head_cross_entropy(
        logits,
        labels,
        active_class_indices,
        total_classes=total_classes,
    )


def project_agem_gradient(
    current_gradient: torch.Tensor,
    reference_gradient: torch.Tensor,
    *,
    denominator_epsilon: float = 0.0,
) -> torch.Tensor:
    """Project ``current_gradient`` onto the A-GEM reference half-space.

    When ``g · g_ref < 0``, A-GEM uses the closed-form projection

    ``g - (g · g_ref / (g_ref · g_ref)) * g_ref``.
    """

    if current_gradient.ndim != 1 or reference_gradient.ndim != 1:
        raise ValueError("A-GEM gradients must be one-dimensional.")
    if current_gradient.shape != reference_gradient.shape:
        raise ValueError("Current and reference gradients must have the same shape.")
    if denominator_epsilon < 0:
        raise ValueError("denominator_epsilon must be non-negative.")
    if not current_gradient.is_floating_point() or not reference_gradient.is_floating_point():
        raise ValueError("A-GEM gradients must use a floating-point dtype.")
    if not bool(torch.isfinite(current_gradient).all()) or not bool(
        torch.isfinite(reference_gradient).all()
    ):
        raise ValueError("A-GEM gradients must be finite.")

    dot_product = torch.dot(current_gradient, reference_gradient)
    if bool(dot_product >= 0):
        return current_gradient.clone()

    reference_squared_norm = torch.dot(reference_gradient, reference_gradient)
    if bool(reference_squared_norm <= denominator_epsilon):
        # A zero reference gradient cannot produce a genuinely negative dot
        # product; this guard only protects against extreme numerical noise.
        return current_gradient.clone()
    return current_gradient - (dot_product / reference_squared_norm) * reference_gradient


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
