"""Class-incremental training engine for the DDI2025 T-DDI protocol study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None

from src.data.class_mapping import build_seen_class_map, invert_class_map, remap_labels, save_class_map
from src.data.ddi_dataset import (
    load_class_counts,
    load_feature_columns,
    load_scaler_payload,
    load_split_arrays,
)
from src.data.fixed_budget_replay import (
    FixedBudgetReplayBuffer,
    FixedReplaySampler,
    ReplayEpochAudit,
)
from src.data.backbone_inputs import load_ddi_gcn_split_arrays
from src.data.molecular_graphs import MolecularGraphBank, load_graph_bank
from src.data.replay_buffer import ReplayBuffer
from src.eval.cil_evaluation import EvaluationResult, evaluate_model
from src.eval.classwise_metrics import ClasswiseTracker
from src.eval.continual_metrics import compute_forgetting, init_result_matrix, result_matrix_to_frame
from src.eval.s02_artifacts import (
    DRUG_ID_A_COLUMN,
    DRUG_ID_B_COLUMN,
    S02ExportContext,
    export_s02_artifacts,
)
from src.methods.ewc import compute_fisher, ewc_penalty, grow_head_state
from src.methods.replay import build_training_arrays as build_replay_training_arrays
from src.methods.sequential import build_training_arrays as build_sequential_training_arrays
from src.models.mlp import MLP, preset_config
from src.models.ddi_gcn import DDIGCNClassifier
from src.models.tabm_classifier import TabMClassifier
from src.utils.logging import RunLogger, ensure_run_paths
from src.utils.seed import set_global_seed


FIXED_BUDGET_METHOD = "replay_distill_fixed_budget_uniform"
LEGACY_REPLAY_METHODS = {"replay", "replay_distill"}
DISTILL_METHODS = {"replay_distill", FIXED_BUDGET_METHOD}
ALL_REPLAY_METHODS = LEGACY_REPLAY_METHODS | {FIXED_BUDGET_METHOD}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the locked T-DDI protocol study or explicit legacy baselines."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--feature-cols", required=True, type=Path)
    parser.add_argument("--scaler", required=True, type=Path)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--method",
        choices=[
            "sequential",
            "joint_seen",
            "replay",
            "replay_distill",
            FIXED_BUDGET_METHOD,
            "ewc",
        ],
        default=FIXED_BUDGET_METHOD,
    )
    parser.add_argument(
        "--variant",
        choices=["small", "base", "large", "tddi", "tabm", "ddi_gcn"],
        default="tddi",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
        help="Gradient-accumulated batch size; defaults to --batch-size.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="gelu")
    parser.add_argument("--norm", choices=["none", "layernorm", "batchnorm"], default="layernorm")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--memory-per-class", type=int, default=50)
    parser.add_argument("--total-memory-budget", type=int, default=6800)
    parser.add_argument("--replay-draws-per-epoch", type=int, default=6800)
    parser.add_argument("--distill-alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--feature-distill-weight", type=float, default=0.5)
    parser.add_argument("--ewc-lambda", type=float, default=1000.0)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--graph-mapping", type=Path, default=None)
    parser.add_argument("--tabm-k", type=int, default=32)
    parser.add_argument("--tabm-blocks", type=int, default=3)
    parser.add_argument("--tabm-d-block", type=int, default=512)
    parser.add_argument("--tabm-dropout", type=float, default=0.1)
    parser.add_argument("--ddi-gcn-depth", type=int, default=8)
    parser.add_argument("--ddi-gcn-width", type=int, default=128)
    parser.add_argument("--ddi-gcn-attention-dim", type=int, default=65)
    parser.add_argument("--max-train-rows-per-task", type=int, default=None)
    parser.add_argument("--max-validation-rows-per-task", type=int, default=None)
    parser.add_argument("--max-test-rows-per-task", type=int, default=None)
    parser.add_argument("--export-s02", action="store_true")
    parser.add_argument(
        "--s02-splits",
        nargs="+",
        choices=["validation", "test"],
        default=["validation", "test"],
    )
    return parser.parse_args()


def require_torch() -> None:
    if torch is None or F is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError("torch is required to run train_cil.py. Install torch before training.")


def resolve_device(requested: str) -> str:
    require_torch()
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
    require_torch()
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return TensorDataset(x, y)


def build_balanced_sampler(labels: np.ndarray) -> "WeightedRandomSampler":
    """Inverse class-frequency sampler so replay-buffer classes (few samples)
    get resampled roughly as often as current-task classes (many samples)."""
    require_torch()
    labels = np.asarray(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


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


def build_student_old_indices(
    teacher_raw_classes: list[int],
    current_seen_map: dict[int, int],
) -> list[int]:
    """Align student logit columns to the teacher's raw-class column order."""

    if len(set(teacher_raw_classes)) != len(teacher_raw_classes):
        raise ValueError("Teacher raw classes must not contain duplicates.")
    missing = sorted(set(teacher_raw_classes) - set(current_seen_map))
    if missing:
        raise ValueError(f"Teacher classes missing from current class map: {missing}")
    return [current_seen_map[raw_class] for raw_class in teacher_raw_classes]


def method_protocol_name(method: str, memory_per_class: int) -> str:
    if method == "replay":
        return f"replay_balanced_per_class_cap{memory_per_class}"
    if method == "replay_distill":
        return f"replay_distill_balanced_per_class_cap{memory_per_class}"
    if method == FIXED_BUDGET_METHOD:
        return FIXED_BUDGET_METHOD
    if method == "joint_seen":
        return "cumulative_joint_seen_natural_sampling"
    return f"{method}_natural_sampling"


def sampler_policy_name(method: str) -> str:
    if method in LEGACY_REPLAY_METHODS:
        return "inverse_class_frequency_with_replacement"
    if method == FIXED_BUDGET_METHOD:
        return "current_once_plus_fixed_class_uniform_replay"
    return "natural_shuffle_without_replacement"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(project_root: Path) -> dict[str, str | bool | None]:
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


def write_run_config(
    path: Path,
    *,
    args: argparse.Namespace,
    run_id: str,
    device: str,
    task_spec: dict[str, Any],
) -> None:
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    resolved: dict[str, Any] = {
        "device": device,
        "method_protocol": method_protocol_name(args.method, args.memory_per_class),
        "sampler_policy": sampler_policy_name(args.method),
        "validation_policy": "all_seen_classes_for_early_stopping",
        "order_seed": task_spec.get("seed"),
        "training_seed": args.seed,
        "task_protocol": task_spec.get("protocol"),
        "num_tasks": len(task_spec["tasks"]),
        "task_file_sha256": _sha256_file(args.task_file),
    }
    if args.method == FIXED_BUDGET_METHOD:
        resolved.update(
            {
                "total_memory_budget": args.total_memory_budget,
                "replay_draws_per_epoch": args.replay_draws_per_epoch,
                "memory_allocation_policy": "capacity_constrained_max_min_raw_class_id",
                "exemplar_ranking_policy": (
                    "standardized_descriptor_distance_to_class_mean_stable_index_tiebreak"
                    if args.variant in {"tabm", "ddi_gcn"}
                    else "distance_to_class_mean_stable_index_tiebreak"
                ),
                "replay_replacement_policy": "per_class_without_replacement_cycles",
                "buffer_source_policy": "current_task_train_split_only",
                "implementation_sha256": {
                    "fixed_budget_replay": _sha256_file(
                        PROJECT_ROOT / "src/data/fixed_budget_replay.py"
                    ),
                    "training_orchestration": _sha256_file(Path(__file__)),
                },
            }
        )
    model_implementation = PROJECT_ROOT / "src/models" / f"{args.variant}.py"
    if args.variant == "tabm":
        model_implementation = PROJECT_ROOT / "src/models/tabm_classifier.py"
    elif args.variant == "ddi_gcn":
        model_implementation = PROJECT_ROOT / "src/models/ddi_gcn.py"
    if model_implementation.exists():
        resolved["model_implementation_sha256"] = _sha256_file(model_implementation)
    if args.graph_cache is not None:
        resolved["graph_cache_sha256"] = _sha256_file(args.graph_cache)
    payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_state(PROJECT_ROOT),
        "arguments": arguments,
        "resolved": resolved,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_backbone_split(
    args: argparse.Namespace,
    parquet_path: Path,
    feature_columns: list[str],
    scaler_payload: dict[str, Any],
    graph_bank: MolecularGraphBank | None,
    *,
    class_ids: list[int],
    max_rows: int | None,
    include_metadata: bool = False,
    include_ranking_features: bool = False,
) -> tuple[Any, np.ndarray | None]:
    """Load identical rows in the input representation required by a backbone."""

    if args.variant == "ddi_gcn":
        if graph_bank is None:
            raise RuntimeError("DDI-GCN graph bank was not initialized.")
        arrays, ranking = load_ddi_gcn_split_arrays(
            parquet_path,
            graph_bank,
            class_ids=class_ids,
            max_rows=max_rows,
            ranking_feature_columns=feature_columns if include_ranking_features else None,
            scaler_payload=scaler_payload if include_ranking_features else None,
        )
        return arrays, ranking
    arrays = load_split_arrays(
        parquet_path,
        feature_columns,
        class_ids=class_ids,
        scaler_payload=scaler_payload,
        include_metadata=include_metadata,
        meta_cols=[DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN],
        max_rows=max_rows,
    )
    return arrays, arrays.features if include_ranking_features else None


def export_s02_evaluation(
    result: EvaluationResult,
    metadata: dict[str, np.ndarray] | None,
    *,
    run_paths: dict[str, Path],
    run_id: str,
    args: argparse.Namespace,
    task_id: int,
    split: str,
    checkpoint_path: Path,
    logger: RunLogger,
) -> None:
    """Publish one already-collected evaluation result and log its completion."""

    if result.outputs is None or metadata is None:
        raise RuntimeError(f"S02 {split} export requires outputs and drug-pair metadata.")
    paths = export_s02_artifacts(
        result.outputs,
        metadata,
        S02ExportContext(
            run_id=run_id,
            seed=args.seed,
            method=args.method,
            method_protocol=method_protocol_name(args.method, args.memory_per_class),
            train_task=task_id,
            split=split,
            checkpoint_path=checkpoint_path,
            run_config_path=run_paths["run_config_json"],
        ),
        run_paths["outdir"] / "s02",
    )
    logger.log_event(
        "s02_exported",
        f"task={task_id} split={split} rows={result.outputs.labels.shape[0]}",
        payload_json=json.dumps(
            {
                "task": task_id,
                "split": split,
                "rows": int(result.outputs.labels.shape[0]),
                "latent_dim": int(result.outputs.latent_features.shape[1]),
                "predictions": str(paths.predictions_path),
                "latent_features": str(paths.latent_features_path),
                "manifest": str(paths.manifest_path),
            },
            sort_keys=True,
        ),
    )


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
    graph_bank: MolecularGraphBank | None = None,
    tabm_k: int = 32,
    tabm_blocks: int = 3,
    tabm_d_block: int = 512,
    tabm_dropout: float = 0.1,
    ddi_gcn_depth: int = 8,
    ddi_gcn_width: int = 128,
    ddi_gcn_attention_dim: int = 65,
) -> nn.Module:
    require_torch()
    if variant == "tabm":
        model = TabMClassifier(
            input_dim=input_dim,
            num_classes=len(current_seen_map),
            k=tabm_k,
            n_blocks=tabm_blocks,
            d_block=tabm_d_block,
            dropout=tabm_dropout,
        )
    elif variant == "ddi_gcn":
        if graph_bank is None:
            raise ValueError("DDI-GCN requires a molecular graph bank.")
        model = DDIGCNClassifier(
            graph_bank=graph_bank,
            num_classes=len(current_seen_map),
            depth=ddi_gcn_depth,
            width=ddi_gcn_width,
            attention_dim=ddi_gcn_attention_dim,
        )
    else:
        config = preset_config(
            variant,  # type: ignore[arg-type]
            input_dim=input_dim,
            num_classes=len(current_seen_map),
            dropout=dropout,
            activation=activation,  # type: ignore[arg-type]
            norm=norm,  # type: ignore[arg-type]
        )
        model = MLP(config)
    if previous_model is None or previous_seen_map is None:
        return model

    current_state = model.state_dict()
    previous_state = previous_model.state_dict()
    for key, value in previous_state.items():
        if not key.startswith("head.") and key in current_state and current_state[key].shape == value.shape:
            current_state[key] = value.clone()

    previous_head_weight = previous_state["head.weight"]
    previous_head_bias = previous_state["head.bias"]
    for raw_class_id, previous_index in previous_seen_map.items():
        current_index = current_seen_map[raw_class_id]
        if variant == "tabm":
            current_state["head.weight"][:, :, current_index] = previous_head_weight[
                :, :, previous_index
            ].clone()
            current_state["head.bias"][:, current_index] = previous_head_bias[
                :, previous_index
            ].clone()
        else:
            current_state["head.weight"][current_index] = previous_head_weight[
                previous_index
            ].clone()
            current_state["head.bias"][current_index] = previous_head_bias[
                previous_index
            ].clone()

    model.load_state_dict(current_state)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    *,
    teacher_model: nn.Module | None = None,
    teacher_raw_classes: list[int] | None = None,
    current_seen_map: dict[int, int] | None = None,
    distill_alpha: float = 1.0,
    temperature: float = 2.0,
    feature_distill_weight: float = 0.5,
    fisher: dict[str, "torch.Tensor"] | None = None,
    theta_star: dict[str, "torch.Tensor"] | None = None,
    ewc_lambda: float = 0.0,
    gradient_accumulation_steps: int = 1,
) -> float:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    model.train()
    total_loss = 0.0
    total_examples = 0

    student_old_indices: list[int] = []
    if teacher_model is not None and teacher_raw_classes is not None and current_seen_map is not None:
        student_old_indices = build_student_old_indices(
            teacher_raw_classes,
            current_seen_map,
        )
        teacher_model.eval()

    optimizer.zero_grad(set_to_none=True)
    for batch_index, (features, labels) in enumerate(loader):
        features = features.to(device)
        labels = labels.to(device)

        if isinstance(model, TabMClassifier):
            member_logits, student_features = model.forward_members_with_latent(features)
            logits = model.aggregate_member_logits(member_logits)
            member_labels = labels.unsqueeze(1).expand(-1, member_logits.shape[1]).reshape(-1)
            loss = criterion(member_logits.reshape(-1, member_logits.shape[-1]), member_labels)
        elif teacher_model is not None and student_old_indices:
            logits, student_features = model.forward_with_latent(features)
            loss = criterion(logits, labels)
        else:
            logits = model(features)
            student_features = None
            loss = criterion(logits, labels)

        if teacher_model is not None and student_old_indices:
            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model.forward_with_latent(features)
            if teacher_logits.shape[1] != len(student_old_indices):
                raise ValueError(
                    "Teacher logit width does not match the recorded teacher class order: "
                    f"{teacher_logits.shape[1]} != {len(student_old_indices)}"
                )
            student_old_logits = logits[:, student_old_indices]
            distill_loss = F.kl_div(
                F.log_softmax(student_old_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
            if student_features is None:
                student_features = model.encode(features)
            feature_distill_loss = F.mse_loss(student_features, teacher_features)
            loss = loss + distill_alpha * distill_loss + feature_distill_weight * feature_distill_loss

        if fisher is not None and theta_star is not None and ewc_lambda:
            loss = loss + ewc_lambda * ewc_penalty(model, fisher, theta_star)

        unscaled_loss = loss
        microbatch_capacity = loader.batch_size
        if microbatch_capacity is None:
            raise RuntimeError("Gradient accumulation requires a fixed DataLoader batch size.")
        group_start_batch = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
        group_end_batch = min(
            group_start_batch + gradient_accumulation_steps,
            len(loader),
        )
        group_start_sample = group_start_batch * microbatch_capacity
        group_end_sample = min(
            group_end_batch * microbatch_capacity,
            len(loader.sampler),
        )
        group_sample_count = group_end_sample - group_start_sample
        if group_sample_count <= 0:
            raise RuntimeError("Could not resolve the gradient-accumulation group size.")
        (loss * (labels.shape[0] / group_sample_count)).backward()
        should_step = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.shape[0]
        total_loss += float(unscaled_loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def build_fixed_budget_audit_rows(
    *,
    run_id: str,
    seed: int,
    task_id: int,
    seen_raw_classes: list[int],
    current_raw_classes: list[int],
    available_count_by_class: dict[int, int],
    memory_before_by_class: dict[int, int],
    memory_after_by_class: dict[int, int],
    total_memory_budget: int,
    replay_draws_per_epoch: int,
    current_dataset_size: int,
    epoch_audits: list[ReplayEpochAudit],
    optimizer_steps: int,
) -> list[dict[str, Any]]:
    """Create O06 rows and verify them against the sampler's emitted indices."""

    current_set = set(current_raw_classes)
    epochs_trained = len(epoch_audits)
    expected_replay = 0 if task_id == 0 else replay_draws_per_epoch
    expected_old_classes = set(memory_before_by_class)
    replay_index_upper = current_dataset_size + sum(memory_before_by_class.values())
    for audit in epoch_audits:
        if audit.current_draws != current_dataset_size:
            raise RuntimeError("Fixed replay sampler did not draw every current sample exactly once.")
        if audit.replay_draws != expected_replay:
            raise RuntimeError("Fixed replay sampler did not emit the exact replay budget.")
        if len(audit.replay_indices) != audit.replay_draws:
            raise RuntimeError("Fixed replay sampler audit does not match emitted replay indices.")
        if any(
            index < current_dataset_size or index >= replay_index_upper
            for index in audit.replay_indices
        ):
            raise RuntimeError("Fixed replay sampler emitted an index outside replay memory.")
        if expected_replay:
            if set(audit.per_class_draws) != expected_old_classes:
                raise RuntimeError("Fixed replay sampler did not cover exactly the old classes.")
            class_draws = list(audit.per_class_draws.values())
            if max(class_draws) - min(class_draws) > 1:
                raise RuntimeError("Fixed replay exposure is not uniform across old classes.")

    rows: list[dict[str, Any]] = []
    for class_id in sorted(seen_raw_classes):
        per_epoch_draws = [audit.per_class_draws.get(class_id, 0) for audit in epoch_audits]
        per_epoch_unique = [
            audit.per_class_unique_examples.get(class_id, 0) for audit in epoch_audits
        ]
        memory_start = memory_before_by_class.get(class_id, 0)
        replay_dataset_start = current_dataset_size
        class_replay_indices: set[int] = set()
        if memory_start:
            offset = sum(
                memory_before_by_class[prior]
                for prior in sorted(memory_before_by_class)
                if prior < class_id
            )
            lower = replay_dataset_start + offset
            upper = lower + memory_start
            for audit in epoch_audits:
                class_replay_indices.update(
                    index for index in audit.replay_indices if lower <= index < upper
                )
        rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "method": FIXED_BUDGET_METHOD,
                "method_protocol": FIXED_BUDGET_METHOD,
                "task": task_id,
                "raw_class_id": class_id,
                "class_role": "current" if class_id in current_set else "old",
                "available_samples": available_count_by_class.get(class_id, 0),
                "memory_before": memory_start,
                "memory_after": memory_after_by_class.get(class_id, 0),
                "total_memory_before": sum(memory_before_by_class.values()),
                "total_memory_after": sum(memory_after_by_class.values()),
                "total_memory_budget": total_memory_budget,
                "replay_draws_per_epoch_budget": replay_draws_per_epoch,
                "actual_current_draws_per_epoch": current_dataset_size,
                "actual_replay_draws_per_epoch": expected_replay,
                "old_new_draw_ratio": (
                    expected_replay / current_dataset_size if current_dataset_size else np.nan
                ),
                "epochs_trained": epochs_trained,
                "optimizer_steps": optimizer_steps,
                "replay_draws_total": sum(per_epoch_draws),
                "replay_draws_per_epoch_min": min(per_epoch_draws, default=0),
                "replay_draws_per_epoch_max": max(per_epoch_draws, default=0),
                "unique_replay_support": len(class_replay_indices),
                "unique_replay_per_epoch_min": min(per_epoch_unique, default=0),
                "unique_replay_per_epoch_mean": (
                    float(np.mean(per_epoch_unique)) if per_epoch_unique else 0.0
                ),
                "unique_replay_per_epoch_max": max(per_epoch_unique, default=0),
                "sampler_policy": sampler_policy_name(FIXED_BUDGET_METHOD),
            }
        )
    return rows


def write_run_summary(
    path: Path,
    *,
    run_id: str,
    method: str,
    method_protocol: str,
    task_file: Path,
    best_task_metrics: list[dict[str, Any]],
    final_test_metrics: dict[str, float],
) -> None:
    lines = [
        "# CIL Run Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- method: `{method}`",
        f"- method_protocol: `{method_protocol}`",
        f"- task_file: `{task_file}`",
        "- validation_policy: `all_seen_classes_for_early_stopping`",
        "",
        "## Per-Task Best Validation",
        "",
    ]
    for row in best_task_metrics:
        lines.append(
            f"- task={row['task_id']} best_epoch={row['best_epoch']} "
            f"val_macro_f1={row['val_macro_f1']:.6f} "
            f"val_bal_acc={row['val_balanced_accuracy']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Final Seen-Class Test Metrics",
            "",
            f"- accuracy: `{final_test_metrics['accuracy']:.6f}`",
            f"- macro_f1: `{final_test_metrics['macro_f1']:.6f}`",
            f"- weighted_f1: `{final_test_metrics['weighted_f1']:.6f}`",
            f"- balanced_accuracy: `{final_test_metrics['balanced_accuracy']:.6f}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_torch()
    effective_batch_size = args.effective_batch_size or args.batch_size
    if args.batch_size <= 0 or effective_batch_size < args.batch_size:
        raise ValueError("Batch sizes must be positive and effective batch must be at least microbatch.")
    if effective_batch_size % args.batch_size:
        raise ValueError("--effective-batch-size must be divisible by --batch-size.")
    gradient_accumulation_steps = effective_batch_size // args.batch_size
    args.effective_batch_size = effective_batch_size
    if args.method == FIXED_BUDGET_METHOD:
        if args.total_memory_budget <= 0:
            raise ValueError("--total-memory-budget must be positive.")
        if args.replay_draws_per_epoch <= 0:
            raise ValueError("--replay-draws-per-epoch must be positive.")
    if args.variant == "ddi_gcn" and (args.graph_cache is None or args.graph_mapping is None):
        raise ValueError("DDI-GCN requires --graph-cache and --graph-mapping.")
    if args.variant == "ddi_gcn" and args.export_s02:
        raise ValueError("S02 descriptor export is not supported for DDI-GCN runs.")

    run_id = uuid.uuid4().hex
    run_paths = ensure_run_paths(args.outdir, require_empty=True)
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    feature_columns = load_feature_columns(args.feature_cols)
    scaler_payload = load_scaler_payload(args.scaler)
    graph_bank = (
        load_graph_bank(args.graph_cache, args.graph_mapping)
        if args.variant == "ddi_gcn"
        else None
    )
    task_spec = load_task_spec(args.task_file)
    write_run_config(
        run_paths["run_config_json"],
        args=args,
        run_id=run_id,
        device=device,
        task_spec=task_spec,
    )

    checkpoint_dir = run_paths["outdir"] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = run_paths["outdir"] / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(
        log_path=run_paths["train_log"],
        events_path=run_paths["events_csv"],
        mirror_log_path=run_paths["stdout_log"],
        run_id=run_id,
    )

    tasks = task_spec["tasks"]
    num_tasks = len(tasks)
    first_task_by_class = build_first_task_lookup(tasks)
    train_count_by_class = load_class_counts(args.train)
    missing_train_counts = sorted(set(first_task_by_class) - set(train_count_by_class))
    if missing_train_counts:
        raise ValueError(f"Task classes missing from the full train split: {missing_train_counts}")
    classwise_tracker = ClasswiseTracker(
        seed=args.seed,
        method=args.method,
        first_task_by_class=first_task_by_class,
        train_count_by_class=train_count_by_class,
    )
    result_matrix = init_result_matrix(num_tasks)
    result_rows: list[dict[str, Any]] = []
    best_task_rows: list[dict[str, Any]] = []
    training_audit_rows: list[dict[str, Any]] = []
    replay_budget_audit_rows: list[dict[str, Any]] = []

    logger.log_event(
        "run_started",
        f"CIL training started run_id={run_id} method={args.method} "
        f"protocol={method_protocol_name(args.method, args.memory_per_class)} "
        f"tasks={num_tasks} device={device}",
        payload_json=json.dumps(
            {
                "run_id": run_id,
                "method_protocol": method_protocol_name(args.method, args.memory_per_class),
                "sampler_policy": sampler_policy_name(args.method),
                "validation_policy": "all_seen_classes_for_early_stopping",
            },
            sort_keys=True,
        ),
    )

    replay_buffer: ReplayBuffer | FixedBudgetReplayBuffer
    if args.method == FIXED_BUDGET_METHOD:
        replay_buffer = FixedBudgetReplayBuffer(
            total_memory_budget=args.total_memory_budget,
            random_seed=args.seed,
        )
    else:
        replay_buffer = ReplayBuffer(
            memory_per_class=args.memory_per_class,
            random_seed=args.seed,
        )
    previous_model: nn.Module | None = None
    previous_seen_map: dict[int, int] | None = None
    previous_seen_raw_classes: list[int] | None = None
    fisher_total: dict[str, "torch.Tensor"] | None = None
    theta_star: dict[str, "torch.Tensor"] | None = None

    for task in tasks:
        task_id = int(task["task_id"])
        current_raw_classes = [int(class_id) for class_id in task["classes"]]
        seen_raw_classes = sorted(
            {
                int(class_id)
                for seen_task in tasks[: task_id + 1]
                for class_id in seen_task["classes"]
            }
        )
        current_seen_map = build_seen_class_map(seen_raw_classes)
        inverse_seen_map = invert_class_map(current_seen_map)
        memory_before = replay_buffer.total_size
        memory_before_by_class = (
            dict(replay_buffer.memory_counts)
            if isinstance(replay_buffer, FixedBudgetReplayBuffer)
            else {}
        )
        if task_id == 0 and memory_before != 0:
            raise RuntimeError("Replay memory must be empty before task 0.")

        logger.log_event(
            "task_started",
            f"task_id={task_id} current_classes={len(current_raw_classes)} seen_classes={len(seen_raw_classes)}",
        )

        current_train, current_ranking_features = load_backbone_split(
            args,
            args.train,
            feature_columns,
            scaler_payload,
            graph_bank,
            class_ids=current_raw_classes,
            max_rows=args.max_train_rows_per_task,
            include_ranking_features=args.method == FIXED_BUDGET_METHOD,
        )
        validation_seen, _ = load_backbone_split(
            args,
            args.validation,
            feature_columns,
            scaler_payload,
            graph_bank,
            class_ids=seen_raw_classes,
            include_metadata=args.export_s02 and "validation" in args.s02_splits,
            max_rows=args.max_validation_rows_per_task,
        )

        replay_examples_available = 0
        replay_raw_labels = np.empty((0,), dtype=np.int64)
        if args.method == "joint_seen":
            train_seen, _ = load_backbone_split(
                args,
                args.train,
                feature_columns,
                scaler_payload,
                graph_bank,
                class_ids=seen_raw_classes,
                max_rows=args.max_train_rows_per_task,
            )
            train_features = train_seen.features
            train_raw_labels = train_seen.labels
        elif args.method in {"sequential", "ewc"}:
            train_features, train_raw_labels = build_sequential_training_arrays(
                current_train.features,
                current_train.labels,
            )
        else:
            replay_features, replay_raw_labels = replay_buffer.get_all()
            replay_examples_available = int(replay_raw_labels.shape[0])
            if task_id == 0 and replay_examples_available != 0:
                raise RuntimeError("Replay examples must be empty at task 0.")
            train_features, train_raw_labels = build_replay_training_arrays(
                current_train.features,
                current_train.labels,
                replay_features,
                replay_raw_labels,
            )

        train_local_labels = remap_labels(train_raw_labels, current_seen_map)
        validation_local_labels = remap_labels(validation_seen.labels, current_seen_map)

        train_dataset = build_tensor_dataset(train_features, train_local_labels)
        validation_dataset = build_tensor_dataset(validation_seen.features, validation_local_labels)
        fixed_sampler: FixedReplaySampler | None = None
        if args.method in LEGACY_REPLAY_METHODS:
            sampler = build_balanced_sampler(train_local_labels)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        elif args.method == FIXED_BUDGET_METHOD:
            fixed_sampler = FixedReplaySampler(
                current_count=int(len(current_train.labels)),
                replay_raw_labels=replay_raw_labels,
                replay_draws_per_epoch=(0 if task_id == 0 else args.replay_draws_per_epoch),
                seed=args.seed,
                task_id=task_id,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=fixed_sampler,
            )
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
        samples_drawn_per_epoch = len(train_loader.sampler)
        old_class_count = len(previous_seen_map or {})
        expected_replay_draws = 0.0
        if args.method in LEGACY_REPLAY_METHODS and seen_raw_classes:
            expected_replay_draws = (
                samples_drawn_per_epoch * old_class_count / len(seen_raw_classes)
            )
        elif args.method == FIXED_BUDGET_METHOD and task_id > 0:
            expected_replay_draws = float(args.replay_draws_per_epoch)
        logger.log_event(
            "training_protocol",
            f"task={task_id} sampler={sampler_policy_name(args.method)} "
            f"current_examples={len(current_train.labels)} "
            f"replay_examples={replay_examples_available} "
            f"training_examples={len(train_local_labels)} "
            f"draws_per_epoch={samples_drawn_per_epoch}",
            payload_json=json.dumps(
                {
                    "task": task_id,
                    "sampler_policy": sampler_policy_name(args.method),
                    "current_examples": int(len(current_train.labels)),
                    "replay_examples_available": replay_examples_available,
                    "training_examples": int(len(train_local_labels)),
                    "samples_drawn_per_epoch": int(samples_drawn_per_epoch),
                    "expected_replay_draws_per_epoch": expected_replay_draws,
                },
                sort_keys=True,
            ),
        )

        model = expand_model_for_seen_classes(
            previous_model=previous_model,
            previous_seen_map=previous_seen_map,
            current_seen_map=current_seen_map,
            variant=args.variant,
            input_dim=train_features.shape[1],
            dropout=args.dropout,
            activation=args.activation,
            norm=args.norm,
            graph_bank=graph_bank,
            tabm_k=args.tabm_k,
            tabm_blocks=args.tabm_blocks,
            tabm_d_block=args.tabm_d_block,
            tabm_dropout=args.tabm_dropout,
            ddi_gcn_depth=args.ddi_gcn_depth,
            ddi_gcn_width=args.ddi_gcn_width,
            ddi_gcn_attention_dim=args.ddi_gcn_attention_dim,
        ).to(device)

        if args.method == "ewc" and fisher_total is not None and previous_seen_map is not None:
            reference_state = model.state_dict()
            fisher_total = grow_head_state(
                fisher_total, previous_seen_map, current_seen_map, reference_state, zero_new_rows=True
            )
            theta_star = grow_head_state(
                theta_star, previous_seen_map, current_seen_map, reference_state, zero_new_rows=False
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = FocalLoss(gamma=args.focal_gamma)

        teacher_model = None
        if (
            args.method in DISTILL_METHODS
            and previous_model is not None
            and previous_seen_raw_classes is not None
        ):
            if previous_seen_map is None:
                raise RuntimeError("Teacher class map is missing for replay distillation.")
            expected_teacher_order = ordered_raw_classes(previous_seen_map)
            if previous_seen_raw_classes != expected_teacher_order:
                raise RuntimeError(
                    "Teacher raw-class order does not match teacher logit columns."
                )
            teacher_model = previous_model.to(device)
            teacher_model.eval()

        best_epoch = -1
        best_val_metrics: dict[str, float] | None = None
        best_state: dict[str, Any] | None = None
        patience_counter = 0
        epochs_trained = 0

        for epoch in range(1, args.epochs + 1):
            epochs_trained = epoch
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                teacher_model=teacher_model,
                teacher_raw_classes=previous_seen_raw_classes,
                current_seen_map=current_seen_map,
                distill_alpha=args.distill_alpha,
                temperature=args.temperature,
                feature_distill_weight=args.feature_distill_weight,
                fisher=fisher_total if args.method == "ewc" else None,
                theta_star=theta_star if args.method == "ewc" else None,
                ewc_lambda=args.ewc_lambda if args.method == "ewc" else 0.0,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            if fixed_sampler is not None:
                if len(fixed_sampler.history) != epoch:
                    raise RuntimeError("Fixed replay sampler emitted an unexpected number of epochs.")
                sampler_audit = fixed_sampler.last_audit
                if sampler_audit is None:
                    raise RuntimeError("Fixed replay sampler did not publish an epoch audit.")
                expected_epoch_replay = 0 if task_id == 0 else args.replay_draws_per_epoch
                if (
                    sampler_audit.current_draws != len(current_train.labels)
                    or sampler_audit.replay_draws != expected_epoch_replay
                ):
                    raise RuntimeError("Fixed replay sampler violated its draw-count protocol.")
            val_result = evaluate_model(
                model,
                validation_loader,
                criterion,
                device,
                inverse_seen_map,
                evaluation_class_indices=sorted(inverse_seen_map),
            )
            val_metrics = val_result.metrics
            logger.log(
                f"task={task_id} epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_macro_f1={val_metrics['macro_f1']:.6f} "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.6f}"
            )

            if best_val_metrics is None or val_metrics["macro_f1"] > best_val_metrics["macro_f1"]:
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                best_state = {key: value.cpu() for key, value in model.state_dict().items()}
                checkpoint_path = checkpoint_dir / f"task_{task_id}_model.pt"
                torch.save(best_state, checkpoint_path)
                logger.log_event(
                    "checkpoint_saved",
                    f"task={task_id} epoch={epoch} checkpoint={checkpoint_path}",
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.log_event(
                        "early_stopped",
                        f"task={task_id} stopped at epoch={epoch} patience={args.patience}",
                    )
                    break

        if best_state is None or best_val_metrics is None:
            raise RuntimeError(f"Task {task_id} did not produce a valid checkpoint.")

        model.load_state_dict(best_state)
        checkpoint_path = checkpoint_dir / f"task_{task_id}_model.pt"
        if args.export_s02 and "validation" in args.s02_splits:
            validation_export_result = evaluate_model(
                model,
                validation_loader,
                criterion,
                device,
                inverse_seen_map,
                evaluation_class_indices=sorted(inverse_seen_map),
                collect_outputs=True,
            )
            export_s02_evaluation(
                validation_export_result,
                validation_seen.metadata,
                run_paths=run_paths,
                run_id=run_id,
                args=args,
                task_id=task_id,
                split="validation",
                checkpoint_path=checkpoint_path,
                logger=logger,
            )
            del validation_export_result
        best_task_rows.append(
            {
                "task_id": task_id,
                "best_epoch": best_epoch,
                "val_macro_f1": best_val_metrics["macro_f1"],
                "val_balanced_accuracy": best_val_metrics["balanced_accuracy"],
            }
        )

        if args.method in ALL_REPLAY_METHODS:
            if isinstance(replay_buffer, FixedBudgetReplayBuffer):
                replay_buffer.update(
                    current_train.features,
                    current_train.labels,
                    ranking_features=current_ranking_features,
                )
            else:
                replay_buffer.update(current_train.features, current_train.labels)
            replay_buffer.save_summary(memory_dir / "memory_summary.csv")
            replay_buffer.save_snapshot(memory_dir / f"memory_after_task_{task_id}.parquet")
        memory_after = replay_buffer.total_size

        if args.method == FIXED_BUDGET_METHOD:
            if not isinstance(replay_buffer, FixedBudgetReplayBuffer) or fixed_sampler is None:
                raise RuntimeError("Fixed-budget method is missing its buffer or sampler.")
            memory_after_by_class = dict(replay_buffer.memory_counts)
            feasible_memory = min(
                args.total_memory_budget,
                sum(replay_buffer.available_count_by_class.values()),
            )
            if memory_after != feasible_memory:
                raise RuntimeError("Fixed replay memory did not reach its feasible exact budget.")
            replay_budget_audit_rows.extend(
                build_fixed_budget_audit_rows(
                    run_id=run_id,
                    seed=args.seed,
                    task_id=task_id,
                    seen_raw_classes=seen_raw_classes,
                    current_raw_classes=current_raw_classes,
                    available_count_by_class=dict(replay_buffer.available_count_by_class),
                    memory_before_by_class=memory_before_by_class,
                    memory_after_by_class=memory_after_by_class,
                    total_memory_budget=args.total_memory_budget,
                    replay_draws_per_epoch=args.replay_draws_per_epoch,
                    current_dataset_size=int(len(current_train.labels)),
                    epoch_audits=list(fixed_sampler.history),
                    optimizer_steps=epochs_trained
                    * math.ceil(len(train_loader) / gradient_accumulation_steps),
                )
            )
            replay_budget_audit = pd.DataFrame(replay_budget_audit_rows)
            if replay_budget_audit.duplicated(["run_id", "task", "raw_class_id"]).any():
                raise RuntimeError("O06 contains duplicate (run, task, class) keys.")
            replay_budget_audit.to_csv(
                run_paths["outdir"] / "replay_budget_audit.csv",
                index=False,
            )

        training_audit_rows.append(
            {
                "run_id": run_id,
                "seed": args.seed,
                "method": args.method,
                "method_protocol": method_protocol_name(args.method, args.memory_per_class),
                "task": task_id,
                "current_class_count": len(current_raw_classes),
                "old_class_count": old_class_count,
                "seen_class_count": len(seen_raw_classes),
                "current_dataset_size": int(len(current_train.labels)),
                "replay_examples_available": replay_examples_available,
                "training_dataset_size": int(len(train_local_labels)),
                "validation_dataset_size": int(len(validation_local_labels)),
                "memory_before": memory_before,
                "memory_after": memory_after,
                "sampler_policy": sampler_policy_name(args.method),
                "samples_drawn_per_epoch": int(samples_drawn_per_epoch),
                "expected_replay_draws_per_epoch": expected_replay_draws,
                "expected_current_draws_per_epoch": (
                    float(samples_drawn_per_epoch) - expected_replay_draws
                ),
                "actual_replay_draws_per_epoch": (
                    int(fixed_sampler.history[0].replay_draws)
                    if fixed_sampler is not None and fixed_sampler.history
                    else np.nan
                ),
                "actual_current_draws_per_epoch": (
                    int(fixed_sampler.history[0].current_draws)
                    if fixed_sampler is not None and fixed_sampler.history
                    else np.nan
                ),
                "total_memory_budget": (
                    args.total_memory_budget if args.method == FIXED_BUDGET_METHOD else np.nan
                ),
                "replay_draws_per_epoch_budget": (
                    args.replay_draws_per_epoch if args.method == FIXED_BUDGET_METHOD else np.nan
                ),
                "distillation_active": teacher_model is not None,
                "batches_per_epoch": len(train_loader),
                "epochs_trained": epochs_trained,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "optimizer_steps": epochs_trained
                * math.ceil(len(train_loader) / gradient_accumulation_steps),
                "best_epoch": best_epoch,
            }
        )
        pd.DataFrame(training_audit_rows).to_csv(
            run_paths["training_audit_csv"],
            index=False,
        )

        if args.method in DISTILL_METHODS:
            teacher_checkpoint = checkpoint_dir / f"task_{task_id}_teacher.pt"
            torch.save(best_state, teacher_checkpoint)

        if args.method == "ewc":
            fisher_new = compute_fisher(model, train_loader, device)
            if fisher_total is not None:
                fisher_total = {name: fisher_total[name] + fisher_new[name] for name in fisher_new}
            else:
                fisher_total = fisher_new
            theta_star = {name: param.detach().clone() for name, param in model.named_parameters()}

        previous_model = expand_model_for_seen_classes(
            previous_model=None,
            previous_seen_map=None,
            current_seen_map=current_seen_map,
            variant=args.variant,
            input_dim=train_features.shape[1],
            dropout=args.dropout,
            activation=args.activation,
            norm=args.norm,
            graph_bank=graph_bank,
            tabm_k=args.tabm_k,
            tabm_blocks=args.tabm_blocks,
            tabm_d_block=args.tabm_d_block,
            tabm_dropout=args.tabm_dropout,
            ddi_gcn_depth=args.ddi_gcn_depth,
            ddi_gcn_width=args.ddi_gcn_width,
            ddi_gcn_attention_dim=args.ddi_gcn_attention_dim,
        )
        previous_model.load_state_dict(best_state)
        previous_seen_map = dict(current_seen_map)
        previous_seen_raw_classes = ordered_raw_classes(current_seen_map)
        save_class_map(run_paths["outdir"] / f"seen_class_map_task_{task_id}.json", current_seen_map)

        final_seen_metrics: dict[str, float] | None = None
        for eval_task in tasks[: task_id + 1]:
            eval_task_id = int(eval_task["task_id"])
            eval_classes = [int(class_id) for class_id in eval_task["classes"]]
            eval_arrays, _ = load_backbone_split(
                args,
                args.test,
                feature_columns,
                scaler_payload,
                graph_bank,
                class_ids=eval_classes,
                max_rows=args.max_test_rows_per_task,
            )
            eval_labels = remap_labels(eval_arrays.labels, current_seen_map)
            eval_dataset = build_tensor_dataset(eval_arrays.features, eval_labels)
            eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
            eval_result = evaluate_model(
                model.to(device),
                eval_loader,
                criterion,
                device,
                inverse_seen_map,
                evaluation_class_indices=[
                    current_seen_map[class_id] for class_id in eval_classes
                ],
            )
            metrics = eval_result.metrics
            result_matrix[task_id, eval_task_id] = metrics["macro_f1"]
            result_rows.append(
                {
                    "train_task_id": task_id,
                    "eval_task_id": eval_task_id,
                    "split": "test_task_group",
                    **metrics,
                }
            )
        seen_test_arrays, _ = load_backbone_split(
            args,
            args.test,
            feature_columns,
            scaler_payload,
            graph_bank,
            class_ids=seen_raw_classes,
            include_metadata=args.export_s02 and "test" in args.s02_splits,
            max_rows=args.max_test_rows_per_task,
        )
        seen_test_labels = remap_labels(seen_test_arrays.labels, current_seen_map)
        seen_test_dataset = build_tensor_dataset(seen_test_arrays.features, seen_test_labels)
        seen_test_loader = DataLoader(seen_test_dataset, batch_size=args.batch_size, shuffle=False)
        final_seen_result = evaluate_model(
            model.to(device),
            seen_test_loader,
            criterion,
            device,
            inverse_seen_map,
            evaluation_class_indices=sorted(inverse_seen_map),
            include_classwise=True,
            collect_outputs=args.export_s02 and "test" in args.s02_splits,
        )
        final_seen_metrics = final_seen_result.metrics
        classwise_metrics = final_seen_result.classwise_metrics
        if classwise_metrics is None:
            raise RuntimeError("Seen-class evaluation did not return class-wise metrics.")
        if args.export_s02 and "test" in args.s02_splits:
            export_s02_evaluation(
                final_seen_result,
                seen_test_arrays.metadata,
                run_paths=run_paths,
                run_id=run_id,
                args=args,
                task_id=task_id,
                split="test",
                checkpoint_path=checkpoint_path,
                logger=logger,
            )
        del final_seen_result
        classwise_tracker.add_task(task_id, classwise_metrics)
        trajectory_path, forgetting_path = classwise_tracker.save(run_paths["outdir"])
        result_rows.append(
            {
                "train_task_id": task_id,
                "eval_task_id": "seen_all",
                "split": "test_seen_all",
                **final_seen_metrics,
            }
        )
        logger.log_event(
            "task_completed",
            f"task={task_id} seen_macro_f1={final_seen_metrics['macro_f1']:.6f} "
            f"seen_bal_acc={final_seen_metrics['balanced_accuracy']:.6f} "
            f"class_trajectory={trajectory_path} class_forgetting={forgetting_path}",
        )

    result_matrix_frame = result_matrix_to_frame(result_matrix)
    forgetting_frame = compute_forgetting(result_matrix)
    metrics_frame = pd.DataFrame(result_rows)

    result_matrix_frame.to_csv(run_paths["outdir"] / "task_matrix.csv", index=False)
    forgetting_frame.to_csv(run_paths["outdir"] / "forgetting.csv", index=False)
    metrics_frame.to_csv(run_paths["metrics_csv"], index=False)
    final_seen_metrics = metrics_frame[metrics_frame["eval_task_id"] == "seen_all"].iloc[-1].to_dict()
    write_run_summary(
        run_paths["run_summary_md"],
        run_id=run_id,
        method=args.method,
        method_protocol=method_protocol_name(args.method, args.memory_per_class),
        task_file=args.task_file,
        best_task_metrics=best_task_rows,
        final_test_metrics={
            "accuracy": float(final_seen_metrics["accuracy"]),
            "macro_f1": float(final_seen_metrics["macro_f1"]),
            "weighted_f1": float(final_seen_metrics["weighted_f1"]),
            "balanced_accuracy": float(final_seen_metrics["balanced_accuracy"]),
        },
    )
    logger.event("run_completed", "CIL training completed successfully")


if __name__ == "__main__":
    main()
