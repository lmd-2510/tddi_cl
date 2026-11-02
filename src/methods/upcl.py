"""Paper-aligned Uniform Prototype Contrastive Learning (UPCL).

UPCL replaces the learnt fixed classifier rows with non-learnable, unit-norm
prototypes.  New prototypes are generated causally at each task boundary by
Gram--Schmidt orthogonalisation against all prototypes already assigned to old
classes.  Assignment uses only the arriving task's feature centres; no future
examples or labels are read.

The implementation deliberately works through the common ``encode`` /
``classify`` / ``classifier_head`` interface shared by the repository's three
fixed-head backbones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.training.class_incremental_policy import (
    TOTAL_DDI_CLASSES,
    active_class_mask,
)


METHOD_NAME = "upcl"
METHOD_POLICY = "frozen_fold_fixed_head_upcl_iCaRL_replay_v1"
PAPER_REFERENCE = "Wang et al., UPCL, arXiv:2405.15157 (2024)"
PROTOTYPE_POLICY = "causal_online_gram_schmidt_hungarian_uniform_prototypes_v1"
TEMPERATURE_DEFAULT = 0.07


def _as_active(values: Iterable[int], *, total_classes: int) -> tuple[int, ...]:
    active = tuple(int(value) for value in values)
    active_class_mask(active, total_classes=total_classes)
    return active


def enable_upcl_prototype_head(model: nn.Module, *, total_classes: int = TOTAL_DDI_CLASSES) -> nn.Module:
    """Turn a shared linear head into the paper's frozen cosine prototype head."""

    if not all(hasattr(model, name) for name in ("encode", "classify", "classifier_head")):
        raise TypeError("UPCL requires encode(), classify(), and classifier_head.")
    head = model.classifier_head
    if not isinstance(head, nn.Linear) or head.out_features != total_classes:
        raise ValueError(f"UPCL requires a {total_classes}-way nn.Linear fixed head.")
    # The native model's classify() methods honour this flag by normalising the
    # latent representation before the linear operation.
    model.upcl_prototype_head = True
    with torch.no_grad():
        if head.bias is not None:
            head.bias.zero_()
    head.weight.requires_grad_(False)
    if head.bias is not None:
        head.bias.requires_grad_(False)
    return model


def _generator(*, seed: int, member_id: int, task_id: int) -> torch.Generator:
    # A dedicated CPU stream leaves global/model RNG streams untouched.
    value = int(np.random.SeedSequence([int(seed), int(member_id), int(task_id), 0x5550434C]).generate_state(1, dtype=np.uint64)[0])
    result = torch.Generator(device="cpu")
    result.manual_seed(value)
    return result


def _state(prototypes: torch.Tensor, assigned: torch.Tensor, *, total_classes: int, latent_dim: int) -> dict:
    if prototypes.shape != (total_classes, latent_dim) or assigned.shape != (total_classes,):
        raise ValueError("UPCL prototype state shape mismatch.")
    if prototypes.device.type != "cpu" or assigned.device.type != "cpu":
        raise ValueError("UPCL checkpoint state must be stored on CPU.")
    if prototypes.dtype != torch.float32 or assigned.dtype != torch.bool:
        raise ValueError("UPCL state uses float32 prototypes and boolean assignment mask.")
    if not torch.isfinite(prototypes).all():
        raise ValueError("UPCL prototype state contains nonfinite values.")
    return {
        "policy": PROTOTYPE_POLICY,
        "total_classes": int(total_classes),
        "latent_dim": int(latent_dim),
        "prototypes": prototypes.detach().cpu().clone(),
        "assigned": assigned.detach().cpu().clone(),
    }


def initialise_prototype_state(*, total_classes: int, latent_dim: int) -> dict:
    """Create an empty, checkpoint-portable causal prototype state."""

    if total_classes <= 0 or latent_dim < total_classes:
        raise ValueError("UPCL requires positive latent_dim >= total_classes.")
    return _state(torch.zeros((total_classes, latent_dim), dtype=torch.float32),
                  torch.zeros(total_classes, dtype=torch.bool),
                  total_classes=total_classes, latent_dim=latent_dim)


def validate_prototype_state(state: Mapping, *, total_classes: int, latent_dim: int) -> dict:
    """Validate and defensively copy checkpoint state before a resume."""

    expected = {"policy", "total_classes", "latent_dim", "prototypes", "assigned"}
    if set(state) != expected or state.get("policy") != PROTOTYPE_POLICY:
        raise ValueError("Malformed or incompatible UPCL prototype state.")
    if state["total_classes"] != total_classes or state["latent_dim"] != latent_dim:
        raise ValueError("UPCL prototype state class/latent dimensions changed.")
    return _state(state["prototypes"], state["assigned"], total_classes=total_classes, latent_dim=latent_dim)


def _orthogonal_candidates(existing: torch.Tensor, *, count: int, latent_dim: int,
                           generator: torch.Generator) -> torch.Tensor:
    """Generate ``count`` vectors orthogonal to all prior assigned rows."""

    basis = existing.detach().cpu().float()
    result: list[torch.Tensor] = []
    # Rejection is almost never exercised at 178 << 7,560, but makes the
    # routine valid for the smaller test backbones too.
    for _ in range(count):
        for _attempt in range(64):
            vector = torch.randn(latent_dim, generator=generator, dtype=torch.float32)
            if basis.numel():
                vector = vector - basis.T.matmul(basis.matmul(vector))
            if result:
                new_basis = torch.stack(result)
                vector = vector - new_basis.T.matmul(new_basis.matmul(vector))
            norm = torch.linalg.vector_norm(vector)
            if float(norm) > 1e-6:
                result.append(vector / norm)
                break
        else:  # pragma: no cover - protected by latent_dim >= class count
            raise RuntimeError("Could not generate an independent UPCL prototype.")
    return torch.stack(result) if result else torch.empty((0, latent_dim), dtype=torch.float32)


@torch.no_grad()
def _class_centres(model: nn.Module, features: torch.Tensor, labels: torch.Tensor,
                   classes: tuple[int, ...], *, device: torch.device | str,
                   batch_size: int = 1024) -> torch.Tensor:
    """Causal current-task centres used by the paper's Hungarian assignment."""

    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("UPCL assignment features/labels must be aligned matrices.")
    was_training = model.training
    model.eval()
    sums = counts = None
    class_to_offset = {raw: offset for offset, raw in enumerate(classes)}
    try:
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            batch = features[start:end].to(device=device, dtype=torch.float32)
            _, latent = model.forward_with_latent(batch)
            latent = F.normalize(latent, dim=1).detach().cpu()
            if sums is None:
                sums = torch.zeros((len(classes), latent.shape[1]), dtype=torch.float32)
                counts = torch.zeros(len(classes), dtype=torch.long)
            raw = labels[start:end].detach().cpu().tolist()
            for offset, value in enumerate(raw):
                index = class_to_offset.get(int(value))
                if index is not None:
                    sums[index].add_(latent[offset])
                    counts[index] += 1
    finally:
        model.train(was_training)
    if sums is None or counts is None or bool((counts == 0).any()):
        raise ValueError("UPCL assignment requires at least one current row per new class.")
    return F.normalize(sums / counts.unsqueeze(1), dim=1)


def _hungarian_assign(centres: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Minimum-distance class/prototype assignment, as specified by UPCL."""

    if centres.shape != candidates.shape or centres.ndim != 2:
        raise ValueError("UPCL centres and candidates must have matching [class, latent] shapes.")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - requirements include scipy
        raise ImportError("UPCL class-prototype assignment requires scipy.") from error
    # Both are normalised, so minimising Euclidean distance equals maximising
    # cosine similarity.  SciPy runs only on a small CxC task matrix.
    cost = torch.cdist(centres, candidates).cpu().numpy()
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(len(rows))):
        raise RuntimeError("UPCL Hungarian assignment did not cover all classes.")
    return torch.as_tensor(columns, dtype=torch.long)


@torch.no_grad()
def prepare_task_prototypes(
    model: nn.Module,
    state: Mapping | None,
    *,
    current_features: torch.Tensor,
    current_labels: torch.Tensor,
    new_class_indices: Iterable[int],
    active_class_indices: Iterable[int],
    seed: int,
    member_id: int,
    task_id: int,
    device: torch.device | str,
) -> dict:
    """Causally extend, assign and install frozen UPCL prototypes for one task."""

    # Read the shared head first so the focused unit tests can use a compact
    # vocabulary.  Production runs still use the fixed 178-way head.
    head = model.classifier_head
    enable_upcl_prototype_head(model, total_classes=head.out_features)
    head = model.classifier_head
    total, latent_dim = head.out_features, head.in_features
    active = _as_active(active_class_indices, total_classes=total)
    new = _as_active(new_class_indices, total_classes=total)
    if not set(new) <= set(active):
        raise ValueError("UPCL new classes must be part of the active fixed-head set.")
    state = (initialise_prototype_state(total_classes=total, latent_dim=latent_dim)
             if state is None else validate_prototype_state(state, total_classes=total, latent_dim=latent_dim))
    prototypes, assigned = state["prototypes"], state["assigned"]
    if any(bool(assigned[index]) for index in new):
        raise ValueError("UPCL cannot assign an already assigned new prototype row.")
    old = tuple(index for index in active if index not in new)
    if any(not bool(assigned[index]) for index in old):
        raise ValueError("UPCL old active classes are missing frozen prototypes.")
    candidates = _orthogonal_candidates(
        prototypes[assigned], count=len(new), latent_dim=latent_dim,
        generator=_generator(seed=seed, member_id=member_id, task_id=task_id),
    )
    centres = _class_centres(model, current_features, current_labels, new, device=device)
    assignment = _hungarian_assign(centres, candidates)
    updated = prototypes.clone()
    for row, prototype_offset in zip(new, assignment.tolist(), strict=True):
        updated[row] = candidates[prototype_offset]
    updated_assigned = assigned.clone()
    updated_assigned[list(new)] = True
    result = _state(updated, updated_assigned, total_classes=total, latent_dim=latent_dim)
    install_prototypes(model, result, active_class_indices=active)
    return result


@torch.no_grad()
def install_prototypes(model: nn.Module, state: Mapping, *, active_class_indices: Iterable[int]) -> None:
    """Copy assigned rows into the frozen shared head without touching future rows."""

    head = model.classifier_head
    enable_upcl_prototype_head(model, total_classes=head.out_features)
    head = model.classifier_head
    checked = validate_prototype_state(state, total_classes=head.out_features, latent_dim=head.in_features)
    active = _as_active(active_class_indices, total_classes=head.out_features)
    if any(not bool(checked["assigned"][index]) for index in active):
        raise ValueError("UPCL cannot install an unassigned active prototype row.")
    indices = torch.as_tensor(active, dtype=torch.long, device=head.weight.device)
    head.weight.index_copy_(0, indices, checked["prototypes"][indices.cpu()].to(head.weight.device))
    if head.bias is not None:
        head.bias.zero_()
    head.weight.requires_grad_(False)
    if head.bias is not None:
        head.bias.requires_grad_(False)


def prototype_classification_logits(model: nn.Module, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine-prototype logits and normalised latent representations."""

    if not getattr(model, "upcl_prototype_head", False):
        raise TypeError("UPCL loss requires a configured prototype head.")
    _, latent = model.forward_with_latent(features)
    latent = F.normalize(latent, dim=1)
    return model.classifier_head(latent), latent


def _supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor,
                                 *, temperature: float) -> torch.Tensor:
    """Standard batch supervised-contrastive term, safe for singleton labels."""

    if features.shape[0] <= 1:
        return features.new_zeros(())
    similarity = features.matmul(features.T) / temperature
    diagonal = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    similarity = similarity.masked_fill(diagonal, -torch.inf)
    same = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = same.any(dim=1)
    if not bool(valid.any()):
        return features.new_zeros(())
    log_denom = torch.logsumexp(similarity, dim=1)
    log_prob = similarity - log_denom[:, None]
    positive_mean = (log_prob.masked_fill(~same, 0.).sum(dim=1) /
                     same.sum(dim=1).clamp_min(1))
    return -positive_mean[valid].mean()


@dataclass(frozen=True)
class UPCLTrainingContext:
    old_model: nn.Module | None
    active_class_indices: tuple[int, ...]
    old_class_indices: tuple[int, ...]
    class_log_priors: torch.Tensor
    task_id: int
    temperature: float = TEMPERATURE_DEFAULT

    def __post_init__(self) -> None:
        if self.task_id < 0:
            raise ValueError("UPCL task_id must be nonnegative.")
        if not self.temperature > 0:
            raise ValueError("UPCL temperature must be positive.")
        total = int(self.class_log_priors.numel())
        active_class_mask(self.active_class_indices, total_classes=total)
        if not set(self.old_class_indices) <= set(self.active_class_indices):
            raise ValueError("UPCL old classes must be a subset of active classes.")
        if self.class_log_priors.ndim != 1 or not torch.isfinite(self.class_log_priors).all():
            raise ValueError("UPCL requires finite one-dimensional log-priors.")

    @property
    def feature_weight(self) -> float:
        return 1.0 / (2 ** self.task_id)

    @property
    def feature_kd_weight(self) -> float:
        return len(self.old_class_indices) / len(self.active_class_indices)


@dataclass(frozen=True)
class UPCLLossBreakdown:
    prototype: torch.Tensor
    feature_contrastive: torch.Tensor
    feature_distillation: torch.Tensor
    total: torch.Tensor


def compute_upcl_loss(model: nn.Module, features: torch.Tensor, labels: torch.Tensor,
                      context: UPCLTrainingContext) -> UPCLLossBreakdown:
    """UPCL Eq. 5/6 plus Appendix feature-KD mixing, all over seen classes."""

    logits, latent = prototype_classification_logits(model, features)
    total = logits.shape[1]
    active = tuple(context.active_class_indices)
    active_mask = active_class_mask(active, total_classes=total, device=logits.device)
    if not bool(active_mask[labels].all()):
        raise ValueError("UPCL batch contains a target from an unseen class.")
    # Equation 5: log p(y) supplies the dynamic imbalance margin. It is a
    # column-wise correction, never estimated from future task data.
    adjusted = logits + context.class_log_priors.to(device=logits.device, dtype=logits.dtype).unsqueeze(0)
    adjusted = adjusted.masked_fill(~active_mask.unsqueeze(0), -torch.inf)
    prototype = F.cross_entropy(adjusted / context.temperature, labels)
    feature = _supervised_contrastive_loss(latent, labels, temperature=context.temperature)
    upcl = prototype + context.feature_weight * feature
    distillation = upcl.new_zeros(())
    if context.old_model is not None and context.old_class_indices:
        with torch.no_grad():
            _, old_latent = context.old_model.forward_with_latent(features)
            old_latent = F.normalize(old_latent, dim=1)
        # Cosine similarity can exceed one by a few ULPs in float32; the
        # paper's 1-cosine distance is non-negative by definition.
        distillation = (1. - (latent * old_latent).sum(dim=1)).clamp_min(0.).mean()
    total_loss = (1. - context.feature_kd_weight) * upcl + context.feature_kd_weight * distillation
    return UPCLLossBreakdown(prototype, feature, distillation, total_loss)


def class_log_priors(labels: np.ndarray, *, total_classes: int,
                     active_class_indices: Iterable[int]) -> torch.Tensor:
    """Compute causal task-batch log priors for UPCL's dynamic margin."""

    values = np.asarray(labels, dtype=np.int64)
    active = _as_active(active_class_indices, total_classes=total_classes)
    counts = np.bincount(values, minlength=total_classes).astype(np.float64)
    if values.ndim != 1 or not np.all(np.isin(values, active)) or np.any(counts[list(active)] <= 0):
        raise ValueError("UPCL priors require at least one training item for every active class.")
    probability = counts / counts[list(active)].sum()
    # Inactive values are masked before use; use zero there to keep the
    # checkpoint/audit vector finite and easy to inspect.
    log_prior = np.zeros(total_classes, dtype=np.float32)
    log_prior[list(active)] = np.log(probability[list(active)]).astype(np.float32)
    return torch.from_numpy(log_prior)
