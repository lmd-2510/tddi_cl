"""Training entrypoint and numerical kernels for CB-Replay, EWC and A-GEM."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_mapping import remap_labels
from src.data.class_balanced_replay_buffer import BUFFER_POLICY, RANKING_POLICY
from src.eval.evaluation import evaluate_model
from src.methods.agem import project_agem_gradient, trainable_parameters, flatten_gradients, assign_gradients
from src.methods.ewc import ewc_penalty
from src.methods.replay import classification_loss as replay_classification_loss
from src.models.tddi_paper_member import TDDI_PAPER_INPUT_DIM, TDDIPaperMember, TDDIPaperMemberConfig

FIXED_BUDGET_METHOD = "class_balanced_fixed_budget_replay"
METHODS = (FIXED_BUDGET_METHOD, "ewc", "agem")
METHOD_POLICIES = {
    FIXED_BUDGET_METHOD: "frozen_fold_class_balanced_replay_v1",
    "ewc": "frozen_fold_online_ewc_v1",
    "agem": "frozen_fold_agem_v1",
}

class FocalLoss(nn.Module if nn is not None else object):
    """L_focal(p_t) = -(1-p_t)^gamma * log(p_t), unweighted (Lin et al. 2017),
    as used by the T-DDI paper on this same DDI2025 dataset to handle severe
    178-class imbalance."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        log_probs = F.log_softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = target_log_probs.exp()
        loss = -((1 - target_probs) ** self.gamma) * target_log_probs
        return loss.mean()

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("train", "validation", "test", "feature-cols", "task-file", "outdir",
                "fold-assignments", "fold-manifest", "fold-preprocessing"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default=FIXED_BUDGET_METHOD)
    parser.add_argument("--variant", choices=["tddi_paper_member", "tabm", "realmlp_td"], default="tddi_paper_member")
    parser.add_argument("--norm", choices=["layernorm", "none"], default=None)
    parser.add_argument("--activation", choices=["gelu", "relu", "selu"], default=None)
    parser.add_argument("--member-id", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preprocessing-policy", choices=["raw_identity", "task0_standard_frozen"], required=True)
    parser.add_argument("--fold-replay-policy", choices=["stratified_fraction_v1"], default="stratified_fraction_v1")
    parser.add_argument("--buffer-policy", choices=[BUFFER_POLICY], default=BUFFER_POLICY)
    parser.add_argument("--exemplar-ranking-policy", choices=[RANKING_POLICY], default=RANKING_POLICY)
    parser.add_argument("--loss-variant", choices=["focal_current_ce_replay", "cross_entropy"],
                        help="Derived from method when omitted.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--effective-batch-size", type=int, default=None)
    parser.add_argument("--evaluation-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="TabM: optional hard cap; default stops on validation patience.")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--replay-fraction", type=float, default=0.125)
    parser.add_argument("--replay-repeat-cap", type=int, default=3)
    parser.add_argument("--fold-member-budget", type=int)
    parser.add_argument("--fold-global-budget", type=int)
    parser.add_argument("--ewc-lambda", type=float, default=1000.0)
    parser.add_argument("--agem-reference-batch-size", type=int, default=256)
    parser.add_argument("--stop-after-task", type=int)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--resume-fold-checkpoint", type=Path)
    parser.add_argument("--export-member-predictions", action="store_true")
    parser.add_argument("--member-prediction-splits", choices=["validation", "test"],
                        nargs="+", default=["validation", "test"])
    parser.add_argument("--task-cache-root", type=Path)
    parser.add_argument("--source-shard-cache-root", type=Path)
    args = parser.parse_args(argv)
    tabm = args.variant == "tabm"
    realmlp = args.variant == "realmlp_td"
    defaults = dict(norm="none" if tabm else "layernorm", activation="relu" if tabm else "gelu",
        batch_size=256 if tabm else 64, evaluation_batch_size=256 if tabm else 512,
        patience=16 if tabm else 5, lr=0.002 if tabm else 0.001,
        weight_decay=0.0003 if tabm else 0.0001, dropout=0.1 if tabm else 0.2,
        gradient_clip_norm=1.0 if tabm else None)
    if realmlp:
        defaults.update(norm="none", activation="selu", batch_size=256,
            evaluation_batch_size=1024, patience=20, lr=.04, weight_decay=.02,
            dropout=.15)
    defaults["label_smoothing"] = .1 if realmlp else 0.
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.effective_batch_size is None:
        args.effective_batch_size = args.batch_size if tabm or realmlp else 1024
    if args.epochs is None and not tabm:
        args.epochs = 256 if realmlp else 30
    expected_loss = "focal_current_ce_replay" if args.method == FIXED_BUDGET_METHOD and args.variant == "tddi_paper_member" else "cross_entropy"
    if args.loss_variant is None:
        args.loss_variant = expected_loss
    if args.loss_variant != expected_loss:
        parser.error(f"{args.variant}/{args.method} requires --loss-variant {expected_loss}")
    return args


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_task_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "tasks" not in payload or not isinstance(payload["tasks"], list):
        raise ValueError(f"Invalid task file: {path}")
    return payload

def build_first_task_lookup(tasks: list[dict[str, Any]]) -> dict[int, int]:
    first_task_by_class: dict[int, int] = {}
    for task in tasks:
        task_id = int(task["task_id"])
        for raw_class_id in task["classes"]:
            class_id = int(raw_class_id)
            if class_id in first_task_by_class:
                raise ValueError(f"Class {class_id} appears in more than one task.")
            first_task_by_class[class_id] = task_id
    return first_task_by_class

def build_tensor_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return TensorDataset(x, y)

def ordered_raw_classes(class_map: dict[int, int]) -> list[int]:
    """Return raw class IDs in the exact order used by model output columns."""

    expected_indices = list(range(len(class_map)))
    actual_indices = sorted(class_map.values())
    if actual_indices != expected_indices:
        raise ValueError(
            "Class map must use dense output indices before aligning logits. "
            f"Expected {expected_indices}, got {actual_indices}."
        )
    return [raw_class for raw_class, _ in sorted(class_map.items(), key=lambda item: item[1])]

def _git_state(project_root: Path) -> dict[str, str | bool | None]:
    snapshot_commit = os.environ.get("CIL_SOURCE_COMMIT")
    if snapshot_commit:
        if len(snapshot_commit) != 40 or any(char not in "0123456789abcdef" for char in snapshot_commit):
            raise ValueError("CIL_SOURCE_COMMIT must be a lowercase 40-character Git commit.")
        return {
            "commit": snapshot_commit,
            "dirty": False,
            "snapshot": os.environ.get("CIL_SOURCE_SNAPSHOT"),
        }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def model_manifest(args):
    if args.variant == "tabm":
        from src.models.tabm_member import TabMMemberConfig
        return TabMMemberConfig(num_classes=1, dropout=args.dropout).manifest()
    if args.variant == "realmlp_td":
        from src.models.realmlp_td_member import RealMLPTDConfig
        return RealMLPTDConfig(num_classes=1, dropout=args.dropout).manifest()
    from src.models.tddi_paper_member import paper_member_manifest
    return paper_member_manifest(dropout=args.dropout, activation=args.activation)

def expand_model_for_seen_classes(
    previous_model: nn.Module | None,
    previous_seen_map: dict[int, int] | None,
    current_seen_map: dict[int, int],
    *,
    variant: str,
    input_dim: int,
    dropout: float,
    activation: str,
    norm: str,
) -> nn.Module:
    if variant == "realmlp_td":
        from src.models.realmlp_td_member import RealMLPTDConfig, RealMLPTDMember
        if norm != "none" or activation != "selu":
            raise ValueError("RealMLP-TD requires external preprocessing, norm=none and activation=selu.")
        model = RealMLPTDMember(RealMLPTDConfig(len(current_seen_map), input_dim=input_dim, dropout=dropout))
        if previous_model is not None and previous_seen_map is not None:
            return copy_previous_state_to_expanded_model(previous_model, model, previous_seen_map, current_seen_map, variant=variant)
        return model
    if variant == "tabm":
        from src.models.tabm_member import TabMMember, TabMMemberConfig
        if norm != "none" or activation != "relu":
            raise ValueError("Basic TabM requires --norm none --activation relu; preprocessing remains external.")
        model = TabMMember(TabMMemberConfig(num_classes=len(current_seen_map), input_dim=input_dim, dropout=dropout))
        if previous_model is not None and previous_seen_map is not None:
            return copy_previous_state_to_expanded_model(previous_model, model, previous_seen_map, current_seen_map, variant=variant)
        return model
    if variant != "tddi_paper_member":
        raise ValueError(f"Unsupported model variant: {variant}")
    if input_dim != TDDI_PAPER_INPUT_DIM:
        raise ValueError(
            "tddi_paper_member requires input_dim="
            f"{TDDI_PAPER_INPUT_DIM}, got {input_dim}."
        )
    if norm != "layernorm":
        raise ValueError(
            "tddi_paper_member has fixed input LayerNorm; use --norm layernorm."
        )
    model = TDDIPaperMember(
        TDDIPaperMemberConfig(
            input_dim=input_dim,
            num_classes=len(current_seen_map),
            dropout=dropout,
            activation=activation,  # type: ignore[arg-type]
        )
    )
    if previous_model is None or previous_seen_map is None:
        return model
    return copy_previous_state_to_expanded_model(
        previous_model,
        model,
        previous_seen_map,
        current_seen_map,
        variant=variant,
    )

def copy_previous_state_to_expanded_model(
    previous_model: nn.Module,
    expanded_model: nn.Module,
    previous_seen_map: dict[int, int],
    current_seen_map: dict[int, int],
    *,
    variant: str,
) -> nn.Module:
    """Copy a backbone and raw-class-aligned old head rows into a wider model."""

    current_state = expanded_model.state_dict()
    previous_state = previous_model.state_dict()
    for key, value in previous_state.items():
        if not key.startswith("head.") and key in current_state and current_state[key].shape == value.shape:
            current_state[key] = value.clone()

    previous_head_weight = previous_state["head.weight"]
    previous_head_bias = previous_state["head.bias"]
    if variant == "tabm":
        from src.models.tabm_member import copy_head_classes
        copy_head_classes(current_state, previous_state, previous_seen_map, current_seen_map)
        expanded_model.load_state_dict(current_state)
        return expanded_model
    for raw_class_id, previous_index in previous_seen_map.items():
        current_index = current_seen_map[raw_class_id]
        current_state["head.weight"][current_index] = previous_head_weight[
            previous_index
        ].clone()
        current_state["head.bias"][current_index] = previous_head_bias[
            previous_index
        ].clone()

    expanded_model.load_state_dict(current_state)
    return expanded_model

@dataclass(frozen=True)
class GradientProjectionAudit:
    constraint_count: int
    violated: bool
    minimum_dot_before: float
    minimum_dot_after: float


@dataclass(frozen=True)
class EpochLossComponents:
    classification_loss: float
    raw_ewc_penalty: float
    scaled_ewc_penalty: float
    total_loss: float
    optimizer_steps: int = 0
    examples_seen: int = 0
    tail_effective_batch: int = 0


def make_optimizer(model, args):
    if args.variant == "realmlp_td":
        from src.models.realmlp_td_member import make_optimizer as realmlp_optimizer
        return realmlp_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def training_criterion(args):
    if args.loss_variant == "focal_current_ce_replay":
        return FocalLoss(args.focal_gamma)
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def classification_forward(model, features, labels, criterion):
    if hasattr(model, "classification_loss"):
        logits, loss = model.classification_loss(features, labels, criterion)
        return logits, None, loss
    logits = model(features)
    return logits, None, criterion(logits, labels)


def compute_episodic_reference_gradient(
    model: nn.Module,
    criterion: nn.Module,
    memory_features: np.ndarray,
    memory_raw_labels: np.ndarray,
    current_seen_map: dict[int, int],
    device: str,
    parameters: list["torch.nn.Parameter"],
    *,
    reference_batch_size: int,
) -> "torch.Tensor":
    """Compute an exact mean memory gradient in bounded-size chunks.

    Chunk gradients are weighted by their sample counts, so the result equals
    the gradient of the mean loss over the complete task memory. ``autograd.grad``
    leaves the accumulated current-task ``parameter.grad`` tensors untouched.
    """

    if memory_raw_labels.shape[0] == 0:
        raise ValueError("Cannot compute a reference gradient from empty memory.")
    if reference_batch_size <= 0:
        raise ValueError("reference_batch_size must be positive.")
    local_labels = remap_labels(memory_raw_labels, current_seen_map)

    was_training = model.training
    model.eval()
    try:
        accumulated: torch.Tensor | None = None
        total_examples = int(memory_raw_labels.shape[0])
        for start in range(0, total_examples, reference_batch_size):
            end = min(start + reference_batch_size, total_examples)
            features_tensor = torch.from_numpy(
                np.asarray(memory_features[start:end], dtype=np.float32)
            ).to(device)
            labels_tensor = torch.from_numpy(
                np.asarray(local_labels[start:end], dtype=np.int64)
            ).to(device)
            _, _, reference_loss = classification_forward(
                model,
                features_tensor,
                labels_tensor,
                criterion,
            )
            gradients = torch.autograd.grad(
                reference_loss,
                parameters,
                allow_unused=True,
            )
            flat_gradient = flatten_gradients(parameters, gradients)
            weight = (end - start) / total_examples
            if accumulated is None:
                accumulated = flat_gradient * weight
            else:
                accumulated.add_(flat_gradient, alpha=weight)
    finally:
        model.train(was_training)
    if accumulated is None:  # pragma: no cover - guarded by non-empty validation
        raise RuntimeError("Reference gradient accumulation produced no chunks.")
    return accumulated

def apply_agem_projection(model, criterion, memory_features, memory_labels, seen_map, device):
    parameters = trainable_parameters(model)
    current_gradient = flatten_gradients(parameters)
    reference = compute_episodic_reference_gradient(
        model, criterion, memory_features, memory_labels, seen_map, device, parameters,
        reference_batch_size=len(memory_labels))
    projected = project_agem_gradient(current_gradient, reference)
    assign_gradients(parameters, projected)
    return GradientProjectionAudit(1, bool(torch.dot(current_gradient, reference) < 0),
        float(torch.dot(current_gradient, reference)), float(torch.dot(projected, reference)))


def train_one_epoch(
    model, loader, optimizer, criterion, device, *,
    old_class_indices=(), loss_variant="cross_entropy",
    fisher=None, theta_star=None, ewc_lambda=0.0,
    gradient_accumulation_steps=1, gradient_projector=None,
    return_loss_components=False,
    gradient_clip_norm=None,
    before_batch=None,
):
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if loss_variant not in {"cross_entropy", "focal_current_ce_replay"}:
        raise ValueError(f"Unsupported loss_variant: {loss_variant}")
    model.train()
    totals = np.zeros(4, dtype=np.float64)
    examples = steps = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (features, labels) in enumerate(loader):
        if before_batch is not None:
            before_batch(batch_index)
        features, labels = features.to(device), labels.to(device)
        if hasattr(model, "classification_loss"):
            if loss_variant != "cross_entropy":
                raise ValueError("TabM uses branch-wise cross entropy, not the inherited Focal/replay objective.")
            logits, classification = model.classification_loss(features, labels, criterion)
        elif loss_variant == "focal_current_ce_replay":
            logits = model(features)
            classification = replay_classification_loss(
                logits, labels, old_class_indices, gamma=float(criterion.gamma))
        else:
            logits = model(features)
            classification = criterion(logits, labels)
        raw = ewc_penalty(model, fisher, theta_star) if fisher is not None else logits.new_zeros(())
        scaled = ewc_lambda * raw
        loss = classification + scaled
        group_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
        group_end = min(group_start + gradient_accumulation_steps, len(loader))
        group_samples = min(group_end * loader.batch_size, len(loader.sampler)) - group_start * loader.batch_size
        (loss * (len(labels) / group_samples)).backward()
        if (batch_index + 1) % gradient_accumulation_steps == 0 or batch_index + 1 == len(loader):
            if gradient_projector is not None:
                gradient_projector(steps)
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        count = len(labels)
        totals += count * np.array([float(v.detach()) for v in (classification, raw, scaled, loss)])
        examples += count
    means = totals / max(examples, 1)
    target = loader.batch_size * gradient_accumulation_steps
    result = EpochLossComponents(*means.tolist(), optimizer_steps=steps,
        examples_seen=examples, tail_effective_batch=(examples - 1) % target + 1 if examples else 0)
    return result if return_loss_components else result.total_loss


def main():
    args = parse_args()
    for attr, key in (("task_cache_root", "CIL_COMMON_TASK_CACHE_ROOT"),
                      ("source_shard_cache_root", "CIL_SOURCE_SHARD_CACHE_ROOT")):
        if getattr(args, attr) is None and os.environ.get(key):
            setattr(args, attr, Path(os.environ[key]).expanduser())
    from src.training.fold_training import run_fold_training
    run_fold_training(args, engine=sys.modules[__name__])


if __name__ == "__main__":
    main()
