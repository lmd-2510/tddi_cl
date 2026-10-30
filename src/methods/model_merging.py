"""Causal Task Arithmetic and TIES merging for fixed-head DDI models.

Only learned neural-network parameters outside the classifier head participate
in arithmetic merging.  Fixed-head rows are composed from the task expert that
owns each class.  BatchNorm running statistics are pooled separately; they are
never treated as task-vector coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Literal

import torch
from torch import nn


TASK_ARITHMETIC_METHOD = "task_arithmetic"
TIES_METHOD = "ties"
MODEL_MERGING_METHODS = (TASK_ARITHMETIC_METHOD, TIES_METHOD)

TASK_ARITHMETIC_PAPER_SCALE = 0.4
TIES_PAPER_DENSITY = 0.20
TIES_PAPER_SCALE = 1.0
TIES_PAPER_SIGN = "mass"
TIES_PAPER_REDUCTION = "disjoint_mean"

HEAD_POLICY = "task_owned_fixed_head_rows_v1"
BATCHNORM_POLICY = "weighted_pooled_expert_moments_v1"
MERGE_SCOPE_POLICY = "learned_backbone_parameters_only_v1"

StateDict = Mapping[str, torch.Tensor]
MergeMethod = Literal["task_arithmetic", "ties"]


@dataclass(frozen=True)
class BatchNormStateKeys:
    module_name: str
    weight: str | None
    bias: str | None
    running_mean: str
    running_var: str
    num_batches_tracked: str


@dataclass(frozen=True)
class NeuralMergeScope:
    """Exact state-dict contract for one backbone architecture."""

    parameter_keys: tuple[str, ...]
    backbone_parameter_keys: tuple[str, ...]
    head_weight_key: str
    head_bias_key: str | None
    batchnorm: tuple[BatchNormStateKeys, ...]
    excluded_buffer_keys: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "policy": MERGE_SCOPE_POLICY,
            "backbone_parameter_keys": list(self.backbone_parameter_keys),
            "head": {
                "weight": self.head_weight_key,
                "bias": self.head_bias_key,
                "policy": HEAD_POLICY,
            },
            "batchnorm": [asdict(item) for item in self.batchnorm],
            "excluded_buffer_keys": list(self.excluded_buffer_keys),
        }


@dataclass(frozen=True)
class ModelMergingRecipe:
    """Frozen paper recipes used by the DDI-CIL merging study."""

    method: MergeMethod
    scale: float
    density: float | None = None
    sign_election: str | None = None
    reduction: str | None = None

    def __post_init__(self) -> None:
        if self.method not in MODEL_MERGING_METHODS:
            raise ValueError(f"Unsupported model-merging method: {self.method}")
        if not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("Model-merging scale must be finite and non-negative.")
        if self.method == TASK_ARITHMETIC_METHOD:
            if any(value is not None for value in (self.density, self.sign_election, self.reduction)):
                raise ValueError("Task Arithmetic does not use TIES hyperparameters.")
        else:
            if self.density is None or not 0 < self.density <= 1:
                raise ValueError("TIES density must be in (0, 1].")
            if self.sign_election != TIES_PAPER_SIGN:
                raise ValueError("The locked TIES recipe uses mass sign election.")
            if self.reduction != TIES_PAPER_REDUCTION:
                raise ValueError("The locked TIES recipe uses a disjoint mean.")

    @classmethod
    def paper_task_arithmetic(cls) -> "ModelMergingRecipe":
        return cls(method=TASK_ARITHMETIC_METHOD, scale=TASK_ARITHMETIC_PAPER_SCALE)

    @classmethod
    def paper_ties(cls) -> "ModelMergingRecipe":
        return cls(
            method=TIES_METHOD,
            scale=TIES_PAPER_SCALE,
            density=TIES_PAPER_DENSITY,
            sign_election=TIES_PAPER_SIGN,
            reduction=TIES_PAPER_REDUCTION,
        )

    def manifest(self) -> dict[str, object]:
        paper_locked = (
            self.method == TASK_ARITHMETIC_METHOD
            and self.scale == TASK_ARITHMETIC_PAPER_SCALE
            and self.density is None
            and self.sign_election is None
            and self.reduction is None
        ) or (
            self.method == TIES_METHOD
            and self.scale == TIES_PAPER_SCALE
            and self.density == TIES_PAPER_DENSITY
            and self.sign_election == TIES_PAPER_SIGN
            and self.reduction == TIES_PAPER_REDUCTION
        )
        return {
            **asdict(self),
            "source": (
                "paper_fixed_recipe_no_merger_validation_tuning"
                if paper_locked
                else "explicit_nonpaper_override"
            ),
        }


@dataclass(frozen=True)
class BoundaryMergeResult:
    state_dict: dict[str, torch.Tensor]
    audit: dict[str, object]


def inspect_neural_merge_scope(model: nn.Module) -> NeuralMergeScope:
    """Discover the learned backbone, native head and non-mergeable buffers."""

    if not hasattr(model, "classifier_head"):
        raise TypeError("A mergeable backbone must expose classifier_head.")
    head = model.classifier_head
    if not isinstance(head, nn.Linear):
        raise TypeError("The fixed-head merger currently requires nn.Linear classifier_head.")
    head_module_names = [name for name, module in model.named_modules() if module is head]
    if len(head_module_names) != 1 or not head_module_names[0]:
        raise ValueError("classifier_head must be one uniquely named child module.")
    head_prefix = head_module_names[0]
    head_weight_key = f"{head_prefix}.weight"
    head_bias_key = f"{head_prefix}.bias" if head.bias is not None else None

    parameter_keys = tuple(name for name, _ in model.named_parameters())
    head_keys = {head_weight_key, *(() if head_bias_key is None else (head_bias_key,))}
    if not head_keys <= set(parameter_keys):
        raise ValueError("classifier_head parameters are absent from model.named_parameters().")
    backbone_keys = tuple(name for name in parameter_keys if name not in head_keys)
    if not backbone_keys:
        raise ValueError("A mergeable model must contain learned backbone parameters.")

    batchnorm: list[BatchNormStateKeys] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not name:
                raise ValueError("Root-level BatchNorm is not supported.")
            prefix = f"{name}."
            batchnorm.append(
                BatchNormStateKeys(
                    module_name=name,
                    weight=f"{prefix}weight" if module.affine else None,
                    bias=f"{prefix}bias" if module.affine else None,
                    running_mean=f"{prefix}running_mean",
                    running_var=f"{prefix}running_var",
                    num_batches_tracked=f"{prefix}num_batches_tracked",
                )
            )

    buffer_keys = tuple(name for name, _ in model.named_buffers())
    state_keys = set(model.state_dict())
    for group in batchnorm:
        required = {group.running_mean, group.running_var, group.num_batches_tracked}
        if not required <= state_keys:
            raise ValueError(f"Incomplete BatchNorm state for {group.module_name}.")
    return NeuralMergeScope(
        parameter_keys=parameter_keys,
        backbone_parameter_keys=backbone_keys,
        head_weight_key=head_weight_key,
        head_bias_key=head_bias_key,
        batchnorm=tuple(batchnorm),
        excluded_buffer_keys=buffer_keys,
    )


def _clone_state(state: StateDict) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def _validate_state_compatibility(base: StateDict, experts: Sequence[StateDict]) -> None:
    base_keys = set(base)
    for expert_index, expert in enumerate(experts):
        if set(expert) != base_keys:
            missing = sorted(base_keys - set(expert))
            extra = sorted(set(expert) - base_keys)
            raise ValueError(
                f"Expert {expert_index} state keys differ from base; "
                f"missing={missing[:3]}, extra={extra[:3]}."
            )
        for key, base_value in base.items():
            value = expert[key]
            if value.shape != base_value.shape or value.dtype != base_value.dtype:
                raise ValueError(f"Expert {expert_index} tensor contract mismatch: {key}.")


def _validate_merge_keys(base: StateDict, keys: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple(keys)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("Merge keys must be non-empty and unique.")
    for key in ordered:
        if key not in base:
            raise KeyError(f"Merge key is absent from base state: {key}")
        if not base[key].is_floating_point():
            raise TypeError(f"Task vectors require floating-point tensors: {key}")
    return ordered


def _delta(base: StateDict, expert: StateDict, key: str) -> torch.Tensor:
    value = expert[key] - base[key]
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite task-vector value: {key}")
    return value


def task_arithmetic_state(
    base: StateDict,
    experts: Sequence[StateDict],
    *,
    keys: Sequence[str],
    scale: float = TASK_ARITHMETIC_PAPER_SCALE,
) -> dict[str, torch.Tensor]:
    """Apply the paper Task Arithmetic sum to selected learned parameters."""

    _validate_state_compatibility(base, experts)
    merge_keys = _validate_merge_keys(base, keys)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Task Arithmetic scale must be finite and non-negative.")
    merged = _clone_state(base)
    if not experts:
        return merged
    with torch.no_grad():
        for key in merge_keys:
            total = torch.zeros_like(base[key])
            for expert in experts:
                total.add_(_delta(base, expert, key))
            merged[key] = base[key] + scale * total
    return merged


def _global_topk_threshold(
    base: StateDict,
    expert: StateDict,
    keys: Sequence[str],
    density: float,
) -> torch.Tensor:
    """Return an exact global top-k magnitude threshold with bounded candidates."""

    total = sum(base[key].numel() for key in keys)
    keep = max(1, min(total, math.ceil(total * density)))
    candidates: torch.Tensor | None = None
    for key in keys:
        magnitudes = _delta(base, expert, key).abs().reshape(-1)
        if magnitudes.numel() > keep:
            magnitudes = torch.topk(magnitudes, keep, sorted=False).values
        candidates = magnitudes if candidates is None else torch.cat((candidates, magnitudes))
        if candidates.numel() > keep:
            candidates = torch.topk(candidates, keep, sorted=False).values
    if candidates is None or candidates.numel() != keep:
        raise RuntimeError("Could not derive the global TIES trimming threshold.")
    return candidates.min()


def ties_state(
    base: StateDict,
    experts: Sequence[StateDict],
    *,
    keys: Sequence[str],
    density: float = TIES_PAPER_DENSITY,
    scale: float = TIES_PAPER_SCALE,
) -> dict[str, torch.Tensor]:
    """Trim, elect signs by mass and disjoint-mean task vectors."""

    _validate_state_compatibility(base, experts)
    merge_keys = _validate_merge_keys(base, keys)
    if not 0 < density <= 1:
        raise ValueError("TIES density must be in (0, 1].")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("TIES scale must be finite and non-negative.")
    merged = _clone_state(base)
    if not experts:
        return merged
    thresholds = [
        _global_topk_threshold(base, expert, merge_keys, density)
        for expert in experts
    ]
    with torch.no_grad():
        for key in merge_keys:
            trimmed = []
            for expert, threshold in zip(experts, thresholds, strict=True):
                value = _delta(base, expert, key)
                trimmed.append(torch.where(value.abs() >= threshold, value, torch.zeros_like(value)))
            stacked = torch.stack(trimmed)
            elected_sign = torch.sign(stacked.sum(dim=0))
            aligned = (torch.sign(stacked) == elected_sign.unsqueeze(0)) & (stacked != 0)
            aligned_sum = torch.where(aligned, stacked, torch.zeros_like(stacked)).sum(dim=0)
            aligned_count = aligned.sum(dim=0)
            merged_delta = torch.where(
                aligned_count > 0,
                aligned_sum / aligned_count.clamp_min(1).to(aligned_sum.dtype),
                torch.zeros_like(aligned_sum),
            )
            merged[key] = base[key] + scale * merged_delta
    return merged


def stitch_task_head_rows(
    state: dict[str, torch.Tensor],
    *,
    base: StateDict,
    experts_by_task: Mapping[int, StateDict],
    rows_by_task: Mapping[int, Sequence[int]],
    scope: NeuralMergeScope,
    upto_task_id: int,
) -> None:
    """Compose fixed-head rows; task zero remains owned by the base model."""

    if upto_task_id < 0:
        raise ValueError("upto_task_id must be non-negative.")
    weight = base[scope.head_weight_key]
    if weight.ndim != 2:
        raise ValueError("Fixed classifier head weight must be a matrix.")
    seen_rows: set[int] = set()
    for task_id in range(upto_task_id + 1):
        if task_id not in rows_by_task:
            raise ValueError(f"Missing fixed-head ownership for task {task_id}.")
        rows = tuple(int(row) for row in rows_by_task[task_id])
        if not rows or len(rows) != len(set(rows)):
            raise ValueError(f"Task {task_id} head rows must be non-empty and unique.")
        if min(rows) < 0 or max(rows) >= weight.shape[0]:
            raise ValueError(f"Task {task_id} owns a row outside the fixed head.")
        overlap = seen_rows.intersection(rows)
        if overlap:
            raise ValueError(f"Fixed-head rows have multiple owners: {sorted(overlap)}")
        seen_rows.update(rows)
        source = base if task_id == 0 else experts_by_task.get(task_id)
        if source is None:
            raise ValueError(f"Missing expert state for task {task_id}.")
        index = torch.as_tensor(rows, dtype=torch.long, device=weight.device)
        state[scope.head_weight_key].index_copy_(
            0, index, source[scope.head_weight_key].index_select(0, index)
        )
        if scope.head_bias_key is not None:
            state[scope.head_bias_key].index_copy_(
                0, index, source[scope.head_bias_key].index_select(0, index)
            )


def validate_expert_head_isolation(
    *,
    base: StateDict,
    experts_by_task: Mapping[int, StateDict],
    rows_by_task: Mapping[int, Sequence[int]],
    scope: NeuralMergeScope,
) -> None:
    """Prove that each expert changed only the fixed-head rows it owns."""

    head_rows = int(base[scope.head_weight_key].shape[0])
    all_rows = set(range(head_rows))
    for task_id, expert in experts_by_task.items():
        if task_id not in rows_by_task:
            raise ValueError(f"Missing fixed-head ownership for expert task {task_id}.")
        owned = {int(row) for row in rows_by_task[task_id]}
        if not owned or min(owned) < 0 or max(owned) >= head_rows:
            raise ValueError(f"Invalid fixed-head ownership for expert task {task_id}.")
        protected = sorted(all_rows - owned)
        index = torch.as_tensor(
            protected,
            dtype=torch.long,
            device=base[scope.head_weight_key].device,
        )
        if not torch.equal(
            expert[scope.head_weight_key].index_select(0, index),
            base[scope.head_weight_key].index_select(0, index),
        ):
            raise ValueError(f"Expert task {task_id} modified a non-owned head weight row.")
        if scope.head_bias_key is not None and not torch.equal(
            expert[scope.head_bias_key].index_select(0, index),
            base[scope.head_bias_key].index_select(0, index),
        ):
            raise ValueError(f"Expert task {task_id} modified a non-owned head bias row.")


def pool_batchnorm_moments(
    state: dict[str, torch.Tensor],
    *,
    states_by_task: Mapping[int, StateDict],
    sample_counts_by_task: Mapping[int, int],
    scope: NeuralMergeScope,
    upto_task_id: int,
) -> None:
    """Pool expert BN moments without retaining calibration examples.

    This is deliberately an approximation: source moments were observed under
    expert weights rather than the final merged weights.  The policy is causal,
    train-only and sample-count weighted.
    """

    task_ids = tuple(range(upto_task_id + 1))
    if set(task_ids) - set(states_by_task):
        raise ValueError("BatchNorm pooling is missing a seen task state.")
    counts: list[int] = []
    for task_id in task_ids:
        count = sample_counts_by_task.get(task_id)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Task {task_id} requires a positive integer train sample count.")
        counts.append(count)
    total = sum(counts)
    for group in scope.batchnorm:
        means = [states_by_task[task_id][group.running_mean] for task_id in task_ids]
        variances = [states_by_task[task_id][group.running_var] for task_id in task_ids]
        if any(not bool(torch.isfinite(value).all()) for value in (*means, *variances)):
            raise FloatingPointError(f"Non-finite BatchNorm moment: {group.module_name}")
        pooled_mean = torch.zeros_like(means[0])
        pooled_second = torch.zeros_like(means[0])
        for count, mean, variance in zip(counts, means, variances, strict=True):
            weight = count / total
            pooled_mean.add_(mean, alpha=weight)
            pooled_second.add_(variance + mean.square(), alpha=weight)
        pooled_variance = (pooled_second - pooled_mean.square()).clamp_min_(0)
        state[group.running_mean] = pooled_mean
        state[group.running_var] = pooled_variance
        state[group.num_batches_tracked] = torch.as_tensor(
            sum(int(states_by_task[task_id][group.num_batches_tracked]) for task_id in task_ids),
            dtype=state[group.num_batches_tracked].dtype,
            device=state[group.num_batches_tracked].device,
        )


def merge_task_boundary(
    model: nn.Module,
    *,
    base_state: StateDict,
    experts_by_task: Mapping[int, StateDict],
    rows_by_task: Mapping[int, Sequence[int]],
    train_sample_counts_by_task: Mapping[int, int],
    upto_task_id: int,
    recipe: ModelMergingRecipe,
) -> BoundaryMergeResult:
    """Build one causal merged checkpoint for tasks ``0..upto_task_id``."""

    if 0 in experts_by_task:
        raise ValueError("Task zero is the immutable base, not an expert delta.")
    expected_experts = set(range(1, upto_task_id + 1))
    if set(experts_by_task) != expected_experts:
        raise ValueError(
            "Boundary merge must receive exactly the already-seen experts; "
            f"expected={sorted(expected_experts)}, got={sorted(experts_by_task)}."
        )
    expected_tasks = set(range(upto_task_id + 1))
    if set(train_sample_counts_by_task) != expected_tasks:
        raise ValueError(
            "Boundary merge requires exactly the seen-task train sample counts; "
            f"expected={sorted(expected_tasks)}, "
            f"got={sorted(train_sample_counts_by_task)}."
        )
    for task_id, count in train_sample_counts_by_task.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Task {task_id} requires a positive integer train sample count.")
    scope = inspect_neural_merge_scope(model)
    model_state = model.state_dict()
    _validate_state_compatibility(base_state, [model_state, *experts_by_task.values()])
    validate_expert_head_isolation(
        base=base_state,
        experts_by_task=experts_by_task,
        rows_by_task=rows_by_task,
        scope=scope,
    )
    ordered_experts = [experts_by_task[task_id] for task_id in sorted(experts_by_task)]
    if recipe.method == TASK_ARITHMETIC_METHOD:
        merged = task_arithmetic_state(
            base_state,
            ordered_experts,
            keys=scope.backbone_parameter_keys,
            scale=recipe.scale,
        )
    else:
        merged = ties_state(
            base_state,
            ordered_experts,
            keys=scope.backbone_parameter_keys,
            density=float(recipe.density),
            scale=recipe.scale,
        )

    stitch_task_head_rows(
        merged,
        base=base_state,
        experts_by_task=experts_by_task,
        rows_by_task=rows_by_task,
        scope=scope,
        upto_task_id=upto_task_id,
    )
    states_by_task: dict[int, StateDict] = {0: base_state}
    states_by_task.update(experts_by_task)
    if scope.batchnorm:
        pool_batchnorm_moments(
            merged,
            states_by_task=states_by_task,
            sample_counts_by_task=train_sample_counts_by_task,
            scope=scope,
            upto_task_id=upto_task_id,
        )
    model.load_state_dict(merged, strict=True)

    seen_rows = [int(row) for task_id in range(upto_task_id + 1) for row in rows_by_task[task_id]]
    audit = {
        "artifact_kind": "ddi_cil_causal_model_merge_boundary",
        "base_task_id": 0,
        "upto_task_id": int(upto_task_id),
        "expert_task_ids": sorted(experts_by_task),
        "future_experts_used": False,
        "recipe": recipe.manifest(),
        "scope": scope.manifest(),
        "head_policy": HEAD_POLICY,
        "seen_head_rows": seen_rows,
        "batchnorm_policy": BATCHNORM_POLICY if scope.batchnorm else "not_applicable",
        "batchnorm_uses_stored_samples": False,
        "train_sample_counts_by_task": {
            str(task_id): int(train_sample_counts_by_task[task_id])
            for task_id in range(upto_task_id + 1)
        },
    }
    return BoundaryMergeResult(state_dict=merged, audit=audit)


__all__ = [
    "BATCHNORM_POLICY",
    "BoundaryMergeResult",
    "HEAD_POLICY",
    "MODEL_MERGING_METHODS",
    "MERGE_SCOPE_POLICY",
    "ModelMergingRecipe",
    "NeuralMergeScope",
    "TASK_ARITHMETIC_METHOD",
    "TASK_ARITHMETIC_PAPER_SCALE",
    "TIES_METHOD",
    "TIES_PAPER_DENSITY",
    "TIES_PAPER_SCALE",
    "inspect_neural_merge_scope",
    "merge_task_boundary",
    "pool_batchnorm_moments",
    "stitch_task_head_rows",
    "task_arithmetic_state",
    "ties_state",
    "validate_expert_head_isolation",
]
