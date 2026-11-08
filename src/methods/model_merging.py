"""Causal Task Arithmetic, TIES, and OPCM merging for fixed-head DDI models.

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
OPCM_METHOD = "opcm"
MAGMAX_METHOD = "magmax"
MODEL_MERGING_METHODS = (TASK_ARITHMETIC_METHOD, TIES_METHOD, OPCM_METHOD, MAGMAX_METHOD)

TASK_ARITHMETIC_PAPER_SCALE = 0.4
TIES_PAPER_DENSITY = 0.20
TIES_PAPER_SCALE = 1.0
TIES_PAPER_SIGN = "mass"
TIES_PAPER_REDUCTION = "disjoint_mean"
OPCM_PAPER_ALPHA = 0.5
MAGMAX_PAPER_SCALE = 0.5

HEAD_POLICY = "task_owned_fixed_head_rows_v1"
BATCHNORM_POLICY = "weighted_pooled_expert_moments_v1"
MERGE_SCOPE_POLICY = "learned_backbone_parameters_only_v1"

StateDict = Mapping[str, torch.Tensor]
MergeMethod = Literal["task_arithmetic", "ties", "opcm", "magmax"]


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
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.method not in MODEL_MERGING_METHODS:
            raise ValueError(f"Unsupported model-merging method: {self.method}")
        if not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("Model-merging scale must be finite and non-negative.")
        if self.method == TASK_ARITHMETIC_METHOD:
            if any(value is not None for value in (self.density, self.sign_election, self.reduction, self.alpha)):
                raise ValueError("Task Arithmetic does not use TIES hyperparameters.")
        elif self.method == TIES_METHOD:
            if self.alpha is not None:
                raise ValueError("TIES does not use OPCM alpha.")
            if self.density is None or not 0 < self.density <= 1:
                raise ValueError("TIES density must be in (0, 1].")
            if self.sign_election != TIES_PAPER_SIGN:
                raise ValueError("The locked TIES recipe uses mass sign election.")
            if self.reduction != TIES_PAPER_REDUCTION:
                raise ValueError("The locked TIES recipe uses a disjoint mean.")
        elif self.method == OPCM_METHOD:
            if any(value is not None for value in (self.density, self.sign_election, self.reduction)):
                raise ValueError("OPCM does not use TIES hyperparameters.")
            if self.alpha is None or not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
                raise ValueError("OPCM alpha must be finite and in (0, 1].")
        else:
            if any(value is not None for value in (self.density, self.sign_election, self.reduction, self.alpha)):
                raise ValueError("MagMax does not use TIES or OPCM hyperparameters.")

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

    @classmethod
    def paper_opcm(cls) -> "ModelMergingRecipe":
        return cls(method=OPCM_METHOD, scale=1.0, alpha=OPCM_PAPER_ALPHA)

    @classmethod
    def paper_magmax(cls) -> "ModelMergingRecipe":
        return cls(method=MAGMAX_METHOD, scale=MAGMAX_PAPER_SCALE)

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
        ) or (
            self.method == OPCM_METHOD
            and self.scale == 1.0
            and self.alpha == OPCM_PAPER_ALPHA
            and self.density is None
            and self.sign_election is None
            and self.reduction is None
        ) or (
            self.method == MAGMAX_METHOD
            and self.scale == MAGMAX_PAPER_SCALE
            and self.density is None
            and self.sign_election is None
            and self.reduction is None
            and self.alpha is None
        )
        manifest = {
            **asdict(self),
            "source": (
                "paper_fixed_recipe_no_merger_validation_tuning"
                if paper_locked
                else "explicit_nonpaper_override"
            ),
        }
        if self.alpha is None:
            manifest.pop("alpha", None)
        return manifest


@dataclass(frozen=True)
class BoundaryMergeResult:
    state_dict: dict[str, torch.Tensor]
    audit: dict[str, object]


@dataclass(frozen=True)
class OPCMBoundaryResult:
    state_dict: dict[str, torch.Tensor]
    audit: dict[str, object]
    lambda_t: float


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
    task0_source: StateDict | None = None,
) -> None:
    """Compose fixed-head rows; optionally use a task-0 expert as owner."""

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
        source = (task0_source if task0_source is not None else base) if task_id == 0 else experts_by_task.get(task_id)
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


def validate_sequential_head_isolation(
    *,
    task_states_by_task: Mapping[int, StateDict],
    rows_by_task: Mapping[int, Sequence[int]],
    scope: NeuralMergeScope,
) -> None:
    """Check that sequential fine-tuning changes only the current task rows."""

    head_rows = int(next(iter(task_states_by_task.values()))[scope.head_weight_key].shape[0])
    all_rows = set(range(head_rows))
    for task_id in sorted(task_states_by_task):
        if task_id == 0:
            continue
        if task_id - 1 not in task_states_by_task or task_id not in rows_by_task:
            raise ValueError(f"Sequential head isolation is missing task {task_id - 1} or {task_id}.")
        owned = {int(row) for row in rows_by_task[task_id]}
        protected = sorted(all_rows - owned)
        index = torch.as_tensor(protected, dtype=torch.long)
        previous = task_states_by_task[task_id - 1]
        current = task_states_by_task[task_id]
        if not torch.equal(
            current[scope.head_weight_key].index_select(0, index),
            previous[scope.head_weight_key].index_select(0, index),
        ):
            raise ValueError(f"Sequential task {task_id} modified a non-owned head weight row.")
        if scope.head_bias_key is not None and not torch.equal(
            current[scope.head_bias_key].index_select(0, index),
            previous[scope.head_bias_key].index_select(0, index),
        ):
            raise ValueError(f"Sequential task {task_id} modified a non-owned head bias row.")


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


def _state_vector_norm(reference: StateDict, state: StateDict, keys: Sequence[str]) -> float:
    """Compute a stable L2 norm without materializing one giant parameter vector."""

    squared = torch.zeros((), dtype=torch.float64)
    for key in keys:
        delta = state[key].detach() - reference[key].detach()
        if not delta.is_floating_point():
            continue
        squared = squared + torch.sum(delta.float() * delta.float(), dtype=torch.float64)
    return float(torch.sqrt(squared).item())


def _opcm_linear_cleaned_delta(
    previous_delta: torch.Tensor,
    task_delta: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, int]:
    """Project a linear-layer task vector using the paper OPCM SVD rule."""

    if previous_delta.ndim != 2 or task_delta.ndim != 2:
        raise ValueError("OPCM linear projection requires two-dimensional weight tensors.")
    if previous_delta.shape != task_delta.shape:
        raise ValueError("OPCM linear task-vector shapes must match.")
    if not bool(torch.isfinite(previous_delta).all()) or not bool(torch.isfinite(task_delta).all()):
        raise FloatingPointError("OPCM received a non-finite task vector.")
    previous_norm = torch.linalg.vector_norm(previous_delta)
    if float(previous_norm) == 0.0:
        return task_delta.clone(), 0

    # The official implementation uses a full SVD, then removes the diagonal
    # and the dominant singular subspace whose cumulative energy reaches alpha.
    u, singular_values, vh = torch.linalg.svd(previous_delta, full_matrices=True)
    v = vh.transpose(-2, -1)
    singular_sum = singular_values.sum()
    if float(singular_sum) == 0.0:
        return task_delta.clone(), 0
    cumulative = singular_values.cumsum(dim=0) / singular_sum
    reached = torch.nonzero(cumulative > alpha, as_tuple=False)
    split_rank = int(reached[0].item()) if reached.numel() else int(singular_values.numel())

    projected = u.transpose(-2, -1) @ task_delta @ v
    diagonal = projected.diagonal(offset=0, dim1=-2, dim2=-1)
    diagonal.zero_()
    if split_rank > 0:
        projected[:split_rank, :split_rank] = 0
    cleaned = u @ projected @ v.transpose(-2, -1)
    return cleaned, split_rank


def opcm_merge_task_vector(
    pretrained_state: StateDict,
    previous_state: StateDict,
    task_state: StateDict,
    *,
    keys: Sequence[str],
    previous_lambda: float,
    average_task_vector_norm: float,
    alpha: float = OPCM_PAPER_ALPHA,
) -> tuple[dict[str, torch.Tensor], float, dict[str, object]]:
    """Perform one paper-faithful sequential OPCM update.

    ``pretrained_state`` is the shared initialization, ``previous_state`` is
    the already merged model, and ``task_state`` is the next independently
    fine-tuned expert.  The update is training-free and only touches learned
    backbone parameters supplied in ``keys``.
    """

    _validate_state_compatibility(pretrained_state, [previous_state, task_state])
    merge_keys = _validate_merge_keys(pretrained_state, keys)
    if not math.isfinite(previous_lambda) or previous_lambda <= 0:
        raise ValueError("OPCM previous_lambda must be finite and positive.")
    if not math.isfinite(average_task_vector_norm) or average_task_vector_norm < 0:
        raise ValueError("OPCM average task-vector norm must be finite and non-negative.")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("OPCM alpha must be finite and in (0, 1].")

    raw_state = _clone_state(previous_state)
    projected_layers = 0
    projected_rank_sum = 0
    with torch.no_grad():
        for key in merge_keys:
            previous_delta = previous_state[key] - pretrained_state[key]
            task_delta = task_state[key] - pretrained_state[key]
            if previous_delta.ndim == 2:
                cleaned, split_rank = _opcm_linear_cleaned_delta(
                    previous_delta, task_delta, alpha=alpha
                )
                projected_layers += 1
                projected_rank_sum += split_rank
            else:
                cleaned = task_delta
            raw_state[key] = pretrained_state[key] + previous_lambda * previous_delta + cleaned

        raw_norm = _state_vector_norm(pretrained_state, raw_state, merge_keys)
        if raw_norm == 0.0 or average_task_vector_norm == 0.0:
            lambda_t = 1.0
            merged = raw_state
        else:
            lambda_t = raw_norm / average_task_vector_norm
            merged = _clone_state(raw_state)
            for key in merge_keys:
                merged[key] = pretrained_state[key] + (raw_state[key] - pretrained_state[key]) / lambda_t

    audit = {
        "method": OPCM_METHOD,
        "alpha": float(alpha),
        "previous_lambda_t": float(previous_lambda),
        "lambda_t": float(lambda_t),
        "average_task_vector_norm": float(average_task_vector_norm),
        "raw_merged_task_vector_norm": float(raw_norm),
        "projected_linear_layer_count": int(projected_layers),
        "projected_split_rank_sum": int(projected_rank_sum),
        "projection": "full_svd_remove_diagonal_and_dominant_subspace",
        "adaptive_scaling": "average_seen_task_vector_norm",
    }
    return merged, float(lambda_t), audit


def merge_opcm_task_boundary(
    model: nn.Module,
    *,
    pretrained_state: StateDict,
    previous_state: StateDict,
    task_state: StateDict,
    task_states_by_task: Mapping[int, StateDict],
    rows_by_task: Mapping[int, Sequence[int]],
    train_sample_counts_by_task: Mapping[int, int],
    task_vector_norms_by_task: Mapping[int, float],
    upto_task_id: int,
    previous_lambda: float,
    recipe: ModelMergingRecipe,
) -> OPCMBoundaryResult:
    """Merge the next expert into a causal OPCM state and compose fixed-head rows."""

    if recipe.method != OPCM_METHOD:
        raise ValueError("merge_opcm_task_boundary requires an OPCM recipe.")
    if upto_task_id <= 0:
        raise ValueError("OPCM boundary updates start at task 1; task 0 is the first expert.")
    expected_tasks = set(range(upto_task_id + 1))
    if set(task_states_by_task) != expected_tasks:
        raise ValueError(
            "OPCM boundary requires exactly the seen task experts; "
            f"expected={sorted(expected_tasks)}, got={sorted(task_states_by_task)}."
        )
    if set(task_vector_norms_by_task) != expected_tasks:
        raise ValueError("OPCM task-vector norms must cover exactly the seen tasks.")
    if set(train_sample_counts_by_task) != expected_tasks:
        raise ValueError("OPCM train sample counts must cover exactly the seen tasks.")
    for task_id, count in train_sample_counts_by_task.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Task {task_id} requires a positive integer train sample count.")

    scope = inspect_neural_merge_scope(model)
    _validate_state_compatibility(pretrained_state, [previous_state, task_state, *task_states_by_task.values()])
    # OPCM trains independent experts from one shared initialization, so its
    # protected-head invariant is relative to the pretrained/base state.
    validate_expert_head_isolation(
        base=pretrained_state,
        experts_by_task=task_states_by_task,
        rows_by_task=rows_by_task,
        scope=scope,
    )
    average_norm = float(sum(task_vector_norms_by_task.values()) / len(task_vector_norms_by_task))
    merged, lambda_t, projection_audit = opcm_merge_task_vector(
        pretrained_state,
        previous_state,
        task_state,
        keys=scope.backbone_parameter_keys,
        previous_lambda=previous_lambda,
        average_task_vector_norm=average_norm,
        alpha=float(recipe.alpha),
    )
    stitch_task_head_rows(
        merged,
        base=pretrained_state,
        experts_by_task=task_states_by_task,
        rows_by_task=rows_by_task,
        scope=scope,
        upto_task_id=upto_task_id,
        task0_source=task_states_by_task[0],
    )
    if scope.batchnorm:
        pool_batchnorm_moments(
            merged,
            states_by_task=task_states_by_task,
            sample_counts_by_task=train_sample_counts_by_task,
            scope=scope,
            upto_task_id=upto_task_id,
        )
    model.load_state_dict(merged, strict=True)
    seen_rows = [int(row) for task_id in range(upto_task_id + 1) for row in rows_by_task[task_id]]
    audit = {
        "artifact_kind": "ddi_cil_sequential_opcm_boundary",
        "base_task_id": 0,
        "upto_task_id": int(upto_task_id),
        "expert_task_ids": sorted(task_states_by_task),
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
        "task_vector_norms_by_task": {
            str(task_id): float(task_vector_norms_by_task[task_id])
            for task_id in range(upto_task_id + 1)
        },
        "projection": projection_audit,
    }
    return OPCMBoundaryResult(state_dict=merged, audit=audit, lambda_t=lambda_t)


def magmax_state(
    pretrained_state: StateDict,
    task_states: Sequence[StateDict],
    *,
    keys: Sequence[str],
    scale: float = MAGMAX_PAPER_SCALE,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Select the maximum-absolute task-vector entry across seen tasks."""

    if not task_states:
        raise ValueError("MagMax requires at least one task state.")
    _validate_state_compatibility(pretrained_state, task_states)
    merge_keys = _validate_merge_keys(pretrained_state, keys)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("MagMax scale must be finite and non-negative.")

    merged = _clone_state(pretrained_state)
    selected_counts = [0 for _ in task_states]
    with torch.no_grad():
        for key in merge_keys:
            if not pretrained_state[key].is_floating_point():
                raise TypeError(f"MagMax task vectors require floating-point tensors: {key}")
            selected = torch.zeros_like(pretrained_state[key])
            max_magnitude = torch.zeros_like(pretrained_state[key])
            selected_task = torch.zeros(
                pretrained_state[key].shape,
                dtype=torch.long,
                device=pretrained_state[key].device,
            )
            for task_index, state in enumerate(task_states):
                delta = state[key] - pretrained_state[key]
                if not bool(torch.isfinite(delta).all()):
                    raise FloatingPointError(f"Non-finite MagMax task-vector value: {key}")
                magnitude = delta.abs()
                mask = magnitude > max_magnitude
                selected = torch.where(mask, delta, selected)
                max_magnitude = torch.where(mask, magnitude, max_magnitude)
                selected_task = torch.where(
                    mask,
                    torch.full_like(selected_task, task_index),
                    selected_task,
                )
            for task_index in range(len(task_states)):
                selected_counts[task_index] += int((selected_task == task_index).sum().item())
            merged[key] = pretrained_state[key] + scale * selected

    return merged, {
        "method": MAGMAX_METHOD,
        "scale": float(scale),
        "task_count": len(task_states),
        "selection": "elementwise_maximum_absolute_task_vector",
        "selected_parameter_counts_by_task": {
            str(task_id): int(count) for task_id, count in enumerate(selected_counts)
        },
    }


def merge_magmax_task_boundary(
    model: nn.Module,
    *,
    pretrained_state: StateDict,
    task_states_by_task: Mapping[int, StateDict],
    rows_by_task: Mapping[int, Sequence[int]],
    train_sample_counts_by_task: Mapping[int, int],
    upto_task_id: int,
    recipe: ModelMergingRecipe,
) -> BoundaryMergeResult:
    """Build one causal MagMax boundary from sequential task checkpoints."""

    if recipe.method != MAGMAX_METHOD:
        raise ValueError("merge_magmax_task_boundary requires a MagMax recipe.")
    if upto_task_id < 0:
        raise ValueError("upto_task_id must be non-negative.")
    expected_tasks = set(range(upto_task_id + 1))
    if set(task_states_by_task) != expected_tasks:
        raise ValueError(
            "MagMax boundary requires exactly the seen task states; "
            f"expected={sorted(expected_tasks)}, got={sorted(task_states_by_task)}."
        )
    if set(train_sample_counts_by_task) != expected_tasks:
        raise ValueError("MagMax train sample counts must cover exactly the seen tasks.")
    for task_id, count in train_sample_counts_by_task.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Task {task_id} requires a positive integer train sample count.")

    scope = inspect_neural_merge_scope(model)
    # MagMax experts are sequential checkpoints.  Non-owned rows must remain
    # equal to the immediately preceding checkpoint, not to task-0 weights.
    validate_sequential_head_isolation(
        task_states_by_task=task_states_by_task,
        rows_by_task=rows_by_task,
        scope=scope,
    )
    ordered_states = [task_states_by_task[task_id] for task_id in range(upto_task_id + 1)]
    merged, selection_audit = magmax_state(
        pretrained_state,
        ordered_states,
        keys=scope.backbone_parameter_keys,
        scale=recipe.scale,
    )
    stitch_task_head_rows(
        merged,
        base=pretrained_state,
        experts_by_task=task_states_by_task,
        rows_by_task=rows_by_task,
        scope=scope,
        upto_task_id=upto_task_id,
        task0_source=task_states_by_task[0],
    )
    if scope.batchnorm:
        pool_batchnorm_moments(
            merged,
            states_by_task=task_states_by_task,
            sample_counts_by_task=train_sample_counts_by_task,
            scope=scope,
            upto_task_id=upto_task_id,
        )
    model.load_state_dict(merged, strict=True)
    seen_rows = [int(row) for task_id in range(upto_task_id + 1) for row in rows_by_task[task_id]]
    audit = {
        "artifact_kind": "ddi_cil_sequential_magmax_boundary",
        "base_task_id": 0,
        "upto_task_id": int(upto_task_id),
        "expert_task_ids": sorted(task_states_by_task),
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
        "selection": selection_audit,
    }
    return BoundaryMergeResult(state_dict=merged, audit=audit)


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
    "OPCM_METHOD",
    "OPCM_PAPER_ALPHA",
    "OPCMBoundaryResult",
    "MAGMAX_METHOD",
    "MAGMAX_PAPER_SCALE",
    "TASK_ARITHMETIC_METHOD",
    "TASK_ARITHMETIC_PAPER_SCALE",
    "TIES_METHOD",
    "TIES_PAPER_DENSITY",
    "TIES_PAPER_SCALE",
    "inspect_neural_merge_scope",
    "magmax_state",
    "merge_magmax_task_boundary",
    "merge_opcm_task_boundary",
    "merge_task_boundary",
    "opcm_merge_task_vector",
    "pool_batchnorm_moments",
    "stitch_task_head_rows",
    "task_arithmetic_state",
    "ties_state",
    "validate_expert_head_isolation",
    "validate_sequential_head_isolation",
]
