"""Class-incremental training engine for the DDI2025 T-DDI protocol study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None

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
from src.data.sample_identity import DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN
from src.data.stratified_folds import (
    StratifiedFoldAssignments,
    build_stratified_fold_assignments,
    select_development_fold,
)
from src.eval.evaluation import EvaluationResult, evaluate_model
from src.eval.metrics import (
    ClasswiseTracker,
    compute_average_incremental_macro_f1,
    compute_forgetting,
    init_result_matrix,
    result_matrix_to_frame,
)
from src.eval.predictions import (
    MemberPredictionContext,
    export_member_prediction_artifact,
)
from src.methods.ewc import compute_fisher, ewc_penalty, grow_head_state
from src.methods.agem import project_agem_gradient
from src.methods.gem import (
    TaskEpisodicMemory,
    assign_gradients,
    flatten_gradients,
    project_gem_gradient,
    trainable_parameters,
)
from src.methods.replay import build_training_arrays as build_replay_training_arrays
from src.models.tddi_paper_member import (
    TDDI_PAPER_INPUT_DIM,
    TDDIPaperMember,
    TDDIPaperMemberConfig,
    paper_member_manifest,
)
from src.training.ewc_checkpoint import (
    LoadedEWCCheckpoint,
    load_ewc_checkpoint,
    restore_model_and_theta_star,
    restore_rng_state,
    save_ewc_checkpoint,
)
from src.training.replay_checkpoint import (
    LoadedReplayCheckpoint,
    load_replay_checkpoint,
    restore_fixed_replay_buffer,
    restore_replay_model,
    save_replay_checkpoint,
)
from src.utils.logging import RunLogger, ensure_run_paths
from src.utils.seed import resolve_seed_configuration, set_configured_seeds


FIXED_BUDGET_METHOD = "replay_distill_fixed_budget_uniform"
DISTILL_METHODS = {FIXED_BUDGET_METHOD}
GRADIENT_EPISODIC_METHODS = {"gem", "agem"}


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
        description="Train the numerical T-DDI paper-member continual-learning study."
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
            FIXED_BUDGET_METHOD,
            "ewc",
            "gem",
            "agem",
        ],
        default=FIXED_BUDGET_METHOD,
    )
    parser.add_argument(
        "--variant",
        choices=["tddi_paper_member"],
        default="tddi_paper_member",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Experiment seed. With no --member-id it also remains the legacy "
            "training seed."
        ),
    )
    parser.add_argument(
        "--member-id",
        type=int,
        default=None,
        help=(
            "Optional ensemble member ID. When set, model/dropout/DataLoader RNG "
            "uses a deterministic member seed derived from --seed."
        ),
    )
    parser.add_argument(
        "--ensemble-mode",
        choices=["seeded", "stratified_3fold"],
        default="seeded",
        help=(
            "seeded trains every member on the original train/validation split; "
            "stratified_3fold trains on two development folds and validates on the third."
        ),
    )
    parser.add_argument("--fold-id", type=int, default=None)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--total-memory-budget", type=int, default=6800)
    parser.add_argument("--replay-draws-per-epoch", type=int, default=6800)
    parser.add_argument(
        "--episodic-memory-budget",
        type=int,
        default=500,
        help="Fixed total number of task-indexed GEM/A-GEM memory examples.",
    )
    parser.add_argument(
        "--agem-reference-batch-size",
        type=int,
        default=256,
        help="Random union-memory batch used to compute each A-GEM reference gradient.",
    )
    parser.add_argument("--distill-alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--feature-distill-weight", type=float, default=0.5)
    parser.add_argument("--ewc-lambda", type=float, default=1000.0)
    parser.add_argument(
        "--resume-ewc-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume an EWC run from a versioned task-boundary checkpoint. "
            "The existing --outdir is reused without overwriting completed artifacts."
        ),
    )
    parser.add_argument(
        "--resume-replay-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume replay_distill_fixed_budget_uniform from a versioned "
            "task-boundary checkpoint in the existing --outdir."
        ),
    )
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--max-train-rows-per-task", type=int, default=None)
    parser.add_argument("--max-validation-rows-per-task", type=int, default=None)
    parser.add_argument("--max-test-rows-per-task", type=int, default=None)
    parser.add_argument(
        "--export-member-predictions",
        action="store_true",
        help="Export versioned per-member logits/probabilities for offline ensemble use.",
    )
    parser.add_argument(
        "--member-prediction-splits",
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


def method_protocol_name(method: str) -> str:
    if method == FIXED_BUDGET_METHOD:
        return FIXED_BUDGET_METHOD
    if method == "gem":
        return "gem_task_episodic_gradient_projection"
    if method == "agem":
        return "agem_averaged_episodic_gradient_projection"
    return f"{method}_natural_sampling"


def sampler_policy_name(method: str) -> str:
    if method == FIXED_BUDGET_METHOD:
        return "current_once_plus_fixed_class_uniform_replay"
    if method in GRADIENT_EPISODIC_METHODS:
        return "current_task_natural_shuffle_with_memory_gradient_constraints"
    return "natural_shuffle_without_replacement"


def seed_provenance(args: argparse.Namespace) -> dict[str, int | str | None]:
    """Return the canonical seed metadata shared by manifests and audits."""

    return {
        "experiment_seed": args.experiment_seed,
        "member_id": args.member_id,
        "member_seed": args.member_seed,
        "member_seed_derivation": args.member_seed_derivation,
        "seed_mode": args.seed_mode,
    }


def fixed_replay_seed_provenance(
    args: argparse.Namespace,
) -> dict[str, int | str]:
    """Describe shared-buffer and member-specific sampler seed responsibilities."""

    if args.method != FIXED_BUDGET_METHOD:
        return {}
    return {
        "replay_buffer_seed": args.experiment_seed,
        "replay_buffer_seed_role": "experiment_seed",
        "sampler_seed": args.member_seed,
        "sampler_seed_role": "member_seed",
        "sampler_seed_derivation": args.member_seed_derivation,
    }


def fold_provenance(args: argparse.Namespace) -> dict[str, int | str | None]:
    return {
        "ensemble_mode": args.ensemble_mode,
        "fold_id": args.fold_id,
        "fold_count": args.fold_count if args.ensemble_mode == "stratified_3fold" else None,
        "fold_seed": args.fold_seed if args.ensemble_mode == "stratified_3fold" else None,
    }


def build_ewc_checkpoint_config(
    args: argparse.Namespace,
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable training contract checked when an EWC run resumes."""

    config = {
        "method": args.method,
        "variant": args.variant,
        "train": str(args.train.resolve()),
        "validation": str(args.validation.resolve()),
        "test": str(args.test.resolve()),
        "feature_cols_sha256": _sha256_file(args.feature_cols),
        "scaler_sha256": _sha256_file(args.scaler),
        "task_file_sha256": _sha256_file(args.task_file),
        "task_protocol": task_spec.get("protocol"),
        "num_tasks": len(task_spec["tasks"]),
        "batch_size": args.batch_size,
        "effective_batch_size": args.effective_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "activation": args.activation,
        "norm": args.norm,
        "patience": args.patience,
        "ewc_lambda": args.ewc_lambda,
        "focal_gamma": args.focal_gamma,
        "max_train_rows_per_task": args.max_train_rows_per_task,
        "max_validation_rows_per_task": args.max_validation_rows_per_task,
        "max_test_rows_per_task": args.max_test_rows_per_task,
        # Retained in the schema-v1 checkpoint contract so runs created with the
        # former, disabled-by-default S02 exporter can still resume.
        "export_s02": False,
        "s02_splits": ["validation", "test"],
    }
    if args.export_member_predictions:
        config["export_member_predictions"] = True
        config["member_prediction_splits"] = list(args.member_prediction_splits)
    config["ensemble"] = fold_provenance(args)
    if args.variant == "tddi_paper_member":
        config["model_architecture"] = paper_member_manifest(
            dropout=args.dropout,
            activation=args.activation,
        )
    return config


def build_replay_checkpoint_config(
    args: argparse.Namespace,
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable contract for fixed-budget replay resume."""

    config = {
        "method": args.method,
        "variant": args.variant,
        "device": args.device,
        "train": str(args.train.resolve()),
        "validation": str(args.validation.resolve()),
        "test": str(args.test.resolve()),
        "feature_cols_sha256": _sha256_file(args.feature_cols),
        "scaler_sha256": _sha256_file(args.scaler),
        "task_file_sha256": _sha256_file(args.task_file),
        "task_protocol": task_spec.get("protocol"),
        "num_tasks": len(task_spec["tasks"]),
        "batch_size": args.batch_size,
        "effective_batch_size": args.effective_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "activation": args.activation,
        "norm": args.norm,
        "patience": args.patience,
        "focal_gamma": args.focal_gamma,
        "distill_alpha": args.distill_alpha,
        "temperature": args.temperature,
        "feature_distill_weight": args.feature_distill_weight,
        "total_memory_budget": args.total_memory_budget,
        "replay_draws_per_epoch": args.replay_draws_per_epoch,
        "sampler_metadata": fixed_replay_seed_provenance(args),
        "max_train_rows_per_task": args.max_train_rows_per_task,
        "max_validation_rows_per_task": args.max_validation_rows_per_task,
        "max_test_rows_per_task": args.max_test_rows_per_task,
        # Retained in the schema-v1 checkpoint contract so existing default
        # replay checkpoints remain resumable after removing the S02 pipeline.
        "export_s02": False,
        "s02_splits": ["validation", "test"],
    }
    if args.export_member_predictions:
        config["export_member_predictions"] = True
        config["member_prediction_splits"] = list(args.member_prediction_splits)
    config["ensemble"] = fold_provenance(args)
    if args.variant == "tddi_paper_member":
        config["model_architecture"] = paper_member_manifest(
            dropout=args.dropout,
            activation=args.activation,
        )
    return config


def expected_seen_map_after_task(
    tasks: list[dict[str, Any]],
    completed_task_id: int,
    *,
    resume_kind: str = "EWC",
) -> dict[int, int]:
    """Rebuild the canonical class map implied by a completed task boundary."""

    if completed_task_id < 0 or completed_task_id >= len(tasks):
        raise ValueError(
            f"{resume_kind} completed task {completed_task_id} is outside "
            f"0..{len(tasks) - 1}."
        )
    task_ids = [int(task["task_id"]) for task in tasks]
    if task_ids != list(range(len(tasks))):
        raise ValueError(
            f"Task IDs must be contiguous from zero for {resume_kind} resume."
        )
    seen_classes = {
        int(class_id)
        for task in tasks[: completed_task_id + 1]
        for class_id in task["classes"]
    }
    return build_seen_class_map(seen_classes)


def validate_resume_run_config(
    path: Path,
    checkpoint: LoadedEWCCheckpoint,
) -> None:
    """Bind a resume checkpoint to the pre-existing run directory."""

    if not path.is_file():
        raise FileNotFoundError(f"EWC resume requires the existing run config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") != checkpoint.run_id:
        raise ValueError("EWC checkpoint run_id does not match the existing run_config.json.")
    resolved = payload.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError("Existing run_config.json is missing resolved metadata.")
    mismatches = [
        key
        for key, expected in checkpoint.seed_metadata.items()
        if resolved.get(key) != expected
    ]
    if mismatches:
        raise ValueError(f"Existing run_config seed metadata mismatch: {mismatches}")


def validate_replay_resume_run_config(
    path: Path,
    checkpoint: LoadedReplayCheckpoint,
) -> None:
    """Bind a fixed-replay checkpoint to its pre-existing run directory."""

    if not path.is_file():
        raise FileNotFoundError(f"Replay resume requires the existing run config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") != checkpoint.run_id:
        raise ValueError("Replay checkpoint run_id does not match run_config.json.")
    resolved = payload.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError("Existing run_config.json is missing resolved metadata.")
    mismatches = [
        key
        for key, expected in checkpoint.seed_metadata.items()
        if resolved.get(key) != expected
    ]
    if mismatches:
        raise ValueError(f"Existing run_config seed metadata mismatch: {mismatches}")


def build_ewc_progress(
    *,
    result_matrix: np.ndarray,
    result_rows: list[dict[str, Any]],
    best_task_rows: list[dict[str, Any]],
    training_audit_rows: list[dict[str, Any]],
    classwise_tracker: ClasswiseTracker,
) -> dict[str, Any]:
    """Capture reporting state so a resumed run can finish the same artifacts."""

    return {
        "result_matrix": np.asarray(result_matrix, dtype=np.float64).copy(),
        "result_rows": list(result_rows),
        "best_task_rows": list(best_task_rows),
        "training_audit_rows": list(training_audit_rows),
        "class_trajectory_rows": classwise_tracker.trajectory_frame().to_dict("records"),
    }


def restore_ewc_progress(
    checkpoint: LoadedEWCCheckpoint,
    *,
    num_tasks: int,
    classwise_tracker: ClasswiseTracker,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and restore reporting state stored at the task boundary."""

    required = {
        "result_matrix",
        "result_rows",
        "best_task_rows",
        "training_audit_rows",
        "class_trajectory_rows",
    }
    missing = sorted(required - set(checkpoint.progress))
    if missing:
        raise ValueError(f"EWC checkpoint progress is missing keys: {missing}")
    result_matrix = np.asarray(checkpoint.progress["result_matrix"], dtype=np.float64)
    if result_matrix.shape != (num_tasks, num_tasks):
        raise ValueError(
            "EWC result matrix shape does not match the task count: "
            f"{result_matrix.shape} != {(num_tasks, num_tasks)}."
        )
    result_rows = checkpoint.progress["result_rows"]
    best_task_rows = checkpoint.progress["best_task_rows"]
    training_audit_rows = checkpoint.progress["training_audit_rows"]
    class_trajectory_rows = checkpoint.progress["class_trajectory_rows"]
    if not all(
        isinstance(rows, list)
        for rows in (result_rows, best_task_rows, training_audit_rows, class_trajectory_rows)
    ):
        raise TypeError("EWC checkpoint progress row collections must be lists.")
    expected_completed_count = checkpoint.completed_task_id + 1
    if len(best_task_rows) != expected_completed_count:
        raise ValueError("EWC best-task progress does not match completed_task_id.")
    if len(training_audit_rows) != expected_completed_count:
        raise ValueError("EWC training audit progress does not match completed_task_id.")
    classwise_tracker.restore_trajectory(pd.DataFrame(class_trajectory_rows))
    return (
        result_matrix.copy(),
        list(result_rows),
        list(best_task_rows),
        list(training_audit_rows),
    )


def build_replay_progress(
    *,
    result_matrix: np.ndarray,
    result_rows: list[dict[str, Any]],
    best_task_rows: list[dict[str, Any]],
    training_audit_rows: list[dict[str, Any]],
    replay_budget_audit_rows: list[dict[str, Any]],
    classwise_tracker: ClasswiseTracker,
) -> dict[str, Any]:
    """Capture all reporting state needed to append after replay resume."""

    return {
        "result_matrix": np.asarray(result_matrix, dtype=np.float64).copy(),
        "result_rows": list(result_rows),
        "best_task_rows": list(best_task_rows),
        "training_audit_rows": list(training_audit_rows),
        "replay_budget_audit_rows": list(replay_budget_audit_rows),
        "class_trajectory_rows": classwise_tracker.trajectory_frame().to_dict("records"),
    }


def restore_replay_progress(
    checkpoint: LoadedReplayCheckpoint,
    *,
    num_tasks: int,
    classwise_tracker: ClasswiseTracker,
) -> tuple[
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate and restore reporting state stored at a replay boundary."""

    required = {
        "result_matrix",
        "result_rows",
        "best_task_rows",
        "training_audit_rows",
        "replay_budget_audit_rows",
        "class_trajectory_rows",
    }
    missing = sorted(required - set(checkpoint.progress))
    if missing:
        raise ValueError(f"Replay checkpoint progress is missing keys: {missing}")
    result_matrix = np.asarray(checkpoint.progress["result_matrix"], dtype=np.float64)
    if result_matrix.shape != (num_tasks, num_tasks):
        raise ValueError(
            "Replay result matrix shape does not match the task count: "
            f"{result_matrix.shape} != {(num_tasks, num_tasks)}."
        )
    row_keys = (
        "result_rows",
        "best_task_rows",
        "training_audit_rows",
        "replay_budget_audit_rows",
        "class_trajectory_rows",
    )
    if not all(isinstance(checkpoint.progress[key], list) for key in row_keys):
        raise TypeError("Replay checkpoint progress row collections must be lists.")
    expected_completed_count = checkpoint.completed_task_id + 1
    if len(checkpoint.progress["best_task_rows"]) != expected_completed_count:
        raise ValueError("Replay best-task progress does not match completed_task_id.")
    if len(checkpoint.progress["training_audit_rows"]) != expected_completed_count:
        raise ValueError("Replay training audit progress does not match completed_task_id.")
    classwise_tracker.restore_trajectory(
        pd.DataFrame(checkpoint.progress["class_trajectory_rows"])
    )
    return (
        result_matrix.copy(),
        list(checkpoint.progress["result_rows"]),
        list(checkpoint.progress["best_task_rows"]),
        list(checkpoint.progress["training_audit_rows"]),
        list(checkpoint.progress["replay_budget_audit_rows"]),
    )


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
        "method_protocol": method_protocol_name(args.method),
        "sampler_policy": sampler_policy_name(args.method),
        "validation_policy": "all_seen_classes_for_early_stopping",
        "order_seed": task_spec.get("seed"),
        "training_seed": args.member_seed,
        **seed_provenance(args),
        "task_protocol": task_spec.get("protocol"),
        "num_tasks": len(task_spec["tasks"]),
        "task_file_sha256": _sha256_file(args.task_file),
    }
    if args.method == FIXED_BUDGET_METHOD:
        resolved.update(
            {
                **fixed_replay_seed_provenance(args),
                "total_memory_budget": args.total_memory_budget,
                "replay_draws_per_epoch": args.replay_draws_per_epoch,
                "memory_allocation_policy": "capacity_constrained_max_min_raw_class_id",
                "exemplar_ranking_policy": (
                    "distance_to_class_mean_stable_index_tiebreak"
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
    if args.method in GRADIENT_EPISODIC_METHODS:
        resolved.update(
            {
                "memory_budget": args.episodic_memory_budget,
                "episodic_memory_budget": args.episodic_memory_budget,
                "episodic_memory_allocation": "known_total_tasks_reserved_equal_quota",
                "episodic_memory_partition": "separate_by_task",
                "episodic_exemplar_selection": "seeded_uniform_without_replacement",
                "current_task_sampling": "natural_shuffle_without_replacement",
                "reference_model_mode": "eval_to_freeze_dropout_and_normalization_state",
                "projection_scope": "raw_gradient_before_adamw_preconditioning_and_weight_decay",
                "gradient_constraint": (
                    "one_constraint_per_previous_task"
                    if args.method == "gem"
                    else "one_average_union_memory_constraint"
                ),
                "agem_reference_batch_size": (
                    args.agem_reference_batch_size if args.method == "agem" else None
                ),
                "gem_reference_batch_size": (
                    args.batch_size if args.method == "gem" else None
                ),
                "implementation_sha256": {
                    "method": _sha256_file(
                        PROJECT_ROOT / "src/methods" / f"{args.method}.py"
                    ),
                    "shared_gem_memory_and_gradients": _sha256_file(
                        PROJECT_ROOT / "src/methods/gem.py"
                    ),
                    "training_orchestration": _sha256_file(Path(__file__)),
                },
            }
        )
    if args.variant == "tddi_paper_member":
        resolved["model"] = paper_member_manifest(
            dropout=args.dropout,
            activation=args.activation,
        )
    model_implementation = PROJECT_ROOT / "src/models/tddi_paper_member.py"
    if model_implementation.exists():
        resolved["model_implementation_sha256"] = _sha256_file(model_implementation)
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
    *,
    class_ids: list[int],
    max_rows: int | None,
    include_metadata: bool = False,
    include_ranking_features: bool = False,
) -> tuple[Any, np.ndarray | None]:
    """Load identical rows in the input representation required by a backbone."""
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


def load_development_fold_split(
    args: argparse.Namespace,
    feature_columns: list[str],
    scaler_payload: dict[str, Any],
    assignments: StratifiedFoldAssignments,
    *,
    class_ids: list[int],
    held_out: bool,
    max_rows: int | None,
    include_ranking_features: bool = False,
) -> tuple[Any, np.ndarray | None]:
    """Load train+validation and select one frozen fold or its complement."""

    parts = [
        load_backbone_split(
            args,
            path,
            feature_columns,
            scaler_payload,
            class_ids=class_ids,
            max_rows=None,
            include_metadata=True,
        )[0]
        for path in (args.train, args.validation)
    ]
    arrays = select_development_fold(
        parts,
        assignments,
        fold_id=args.fold_id,
        held_out=held_out,
        max_rows=max_rows,
    )
    return arrays, arrays.features if include_ranking_features else None


def export_member_evaluation(
    result: EvaluationResult,
    metadata: dict[str, np.ndarray] | None,
    *,
    run_paths: dict[str, Path],
    run_id: str,
    args: argparse.Namespace,
    task_id: int,
    split: str,
    logger: RunLogger,
) -> None:
    """Publish one member's aligned outputs for later offline aggregation."""

    if result.outputs is None or metadata is None:
        raise RuntimeError(
            f"Member {split} export requires outputs and drug-pair metadata."
        )
    if args.member_id is None:
        raise ValueError("Member prediction export requires --member-id.")
    context = MemberPredictionContext(
        run_id=run_id,
        method=args.method,
        method_protocol=method_protocol_name(args.method),
        task_id=task_id,
        split=split,
        member_id=args.member_id,
        experiment_seed=args.experiment_seed,
        member_seed=args.member_seed,
        ensemble_mode=getattr(args, "ensemble_mode", "seeded"),
        fold_id=getattr(args, "fold_id", None),
        fold_count=(
            getattr(args, "fold_count", 3)
            if getattr(args, "ensemble_mode", "seeded") == "stratified_3fold"
            else None
        ),
        fold_seed=(
            getattr(args, "fold_seed", 42)
            if getattr(args, "ensemble_mode", "seeded") == "stratified_3fold"
            else None
        ),
    )
    artifact_path = export_member_prediction_artifact(
        result.outputs,
        metadata,
        context,
        run_paths["outdir"]
        / "member_predictions"
        / f"task_{task_id}"
        / f"{split}.npz",
    )
    logger.log_event(
        "member_predictions_exported",
        f"task={task_id} split={split} member_id={args.member_id} "
        f"rows={result.outputs.labels.shape[0]}",
        payload_json=json.dumps(
            {
                "task": task_id,
                "split": split,
                "member_id": args.member_id,
                "experiment_seed": args.experiment_seed,
                "member_seed": args.member_seed,
                "rows": int(result.outputs.labels.shape[0]),
                "class_count": int(result.outputs.class_ids.shape[0]),
                "artifact": str(artifact_path),
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
) -> nn.Module:
    require_torch()
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
    """One optimizer-step audit for GEM or A-GEM."""

    constraint_count: int
    violated: bool
    minimum_dot_before: float
    minimum_dot_after: float


@dataclass(frozen=True)
class EpochLossComponents:
    """Sample-weighted training losses for one completed epoch."""

    classification_loss: float
    raw_ewc_penalty: float
    scaled_ewc_penalty: float
    total_loss: float


def classification_forward(
    model: nn.Module,
    features: "torch.Tensor",
    labels: "torch.Tensor",
    criterion: nn.Module,
    *,
    include_latent: bool = False,
) -> tuple["torch.Tensor", "torch.Tensor | None", "torch.Tensor"]:
    """Run the T-DDI classification path shared by current and memory gradients."""
    if include_latent:
        logits, latent = model.forward_with_latent(features)
        return logits, latent, criterion(logits, labels)
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


def apply_episodic_gradient_projection(
    model: nn.Module,
    criterion: nn.Module,
    memory: TaskEpisodicMemory,
    current_seen_map: dict[int, int],
    device: str,
    *,
    method: str,
    gem_reference_batch_size: int,
    agem_reference_batch_size: int,
    seed: int,
    task_id: int,
    epoch: int,
    optimizer_step: int,
) -> GradientProjectionAudit:
    """Apply GEM/A-GEM to the accumulated current gradient before an update."""

    if method not in GRADIENT_EPISODIC_METHODS:
        raise ValueError(f"Unsupported episodic gradient method: {method}")
    parameters = trainable_parameters(model)
    current_gradient = flatten_gradients(parameters)

    if method == "gem":
        reference_blocks = []
        for previous_task_id in memory.task_ids:
            memory_features, memory_labels = memory.get_task(previous_task_id)
            if memory_labels.shape[0] == 0:
                continue
            reference_blocks.append(
                compute_episodic_reference_gradient(
                    model,
                    criterion,
                    memory_features,
                    memory_labels,
                    current_seen_map,
                    device,
                    parameters,
                    reference_batch_size=gem_reference_batch_size,
                )
            )
        if not reference_blocks:
            return GradientProjectionAudit(0, False, math.nan, math.nan)
        reference_gradients = torch.stack(reference_blocks)
        dots_before = reference_gradients @ current_gradient
        projected = project_gem_gradient(current_gradient, reference_gradients)
        dots_after = reference_gradients @ projected
        violated = bool(torch.any(dots_before < 0))
        constraint_count = int(reference_gradients.shape[0])
    else:
        if memory.total_size == 0:
            return GradientProjectionAudit(0, False, math.nan, math.nan)
        memory_features, memory_labels = memory.sample(
            agem_reference_batch_size,
            seed_components=[seed, task_id, epoch, optimizer_step],
        )
        reference_gradient = compute_episodic_reference_gradient(
            model,
            criterion,
            memory_features,
            memory_labels,
            current_seen_map,
            device,
            parameters,
            reference_batch_size=agem_reference_batch_size,
        )
        dot_before = torch.dot(current_gradient, reference_gradient)
        projected = project_agem_gradient(current_gradient, reference_gradient)
        dot_after = torch.dot(projected, reference_gradient)
        dots_before = dot_before.reshape(1)
        dots_after = dot_after.reshape(1)
        violated = bool(dot_before < 0)
        constraint_count = 1

    assign_gradients(parameters, projected)
    return GradientProjectionAudit(
        constraint_count=constraint_count,
        violated=violated,
        minimum_dot_before=float(dots_before.min().detach().cpu()),
        minimum_dot_after=float(dots_after.min().detach().cpu()),
    )


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
    gradient_projector: Callable[[int], None] | None = None,
    return_loss_components: bool = False,
) -> float | EpochLossComponents:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_raw_ewc_penalty = 0.0
    total_scaled_ewc_penalty = 0.0
    total_examples = 0

    student_old_indices: list[int] = []
    if teacher_model is not None and teacher_raw_classes is not None and current_seen_map is not None:
        student_old_indices = build_student_old_indices(
            teacher_raw_classes,
            current_seen_map,
        )
        teacher_model.eval()

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    for batch_index, (features, labels) in enumerate(loader):
        features = features.to(device)
        labels = labels.to(device)

        logits, student_features, classification_loss = classification_forward(
            model,
            features,
            labels,
            criterion,
            include_latent=teacher_model is not None and bool(student_old_indices),
        )
        loss = classification_loss

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

        raw_ewc_penalty = torch.zeros((), device=loss.device)
        scaled_ewc_penalty = torch.zeros((), device=loss.device)
        if fisher is not None and theta_star is not None:
            raw_ewc_penalty = ewc_penalty(model, fisher, theta_star)
            scaled_ewc_penalty = ewc_lambda * raw_ewc_penalty
            loss = loss + scaled_ewc_penalty

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
            if gradient_projector is not None:
                gradient_projector(optimizer_step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

        batch_size = labels.shape[0]
        total_loss += float(unscaled_loss.item()) * batch_size
        total_classification_loss += float(classification_loss.detach().item()) * batch_size
        total_raw_ewc_penalty += float(raw_ewc_penalty.detach().item()) * batch_size
        total_scaled_ewc_penalty += float(scaled_ewc_penalty.detach().item()) * batch_size
        total_examples += batch_size

    denominator = max(total_examples, 1)
    components = EpochLossComponents(
        classification_loss=total_classification_loss / denominator,
        raw_ewc_penalty=total_raw_ewc_penalty / denominator,
        scaled_ewc_penalty=total_scaled_ewc_penalty / denominator,
        total_loss=total_loss / denominator,
    )
    return components if return_loss_components else components.total_loss


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
    average_incremental_macro_f1: float,
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
            f"- average_incremental_macro_f1: `{average_incremental_macro_f1:.6f}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_torch()
    seed_configuration = resolve_seed_configuration(args.seed, args.member_id)
    args.experiment_seed = seed_configuration.experiment_seed
    args.member_seed = seed_configuration.member_seed
    args.seed_mode = seed_configuration.mode
    args.member_seed_derivation = seed_configuration.derivation
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
    if args.method in GRADIENT_EPISODIC_METHODS:
        if args.episodic_memory_budget <= 0:
            raise ValueError("--episodic-memory-budget must be positive.")
        if args.agem_reference_batch_size <= 0:
            raise ValueError("--agem-reference-batch-size must be positive.")
    if args.resume_ewc_checkpoint is not None and args.method != "ewc":
        raise ValueError("--resume-ewc-checkpoint is only valid with --method ewc.")
    if (
        args.resume_replay_checkpoint is not None
        and args.method != FIXED_BUDGET_METHOD
    ):
        raise ValueError(
            "--resume-replay-checkpoint is only valid with "
            f"--method {FIXED_BUDGET_METHOD}."
        )
    if (
        args.resume_ewc_checkpoint is not None
        and args.resume_replay_checkpoint is not None
    ):
        raise ValueError("Only one resume checkpoint option may be used.")
    if args.export_member_predictions and args.member_id is None:
        raise ValueError("--export-member-predictions requires --member-id.")
    if args.ensemble_mode == "stratified_3fold":
        if args.member_id is None:
            raise ValueError("stratified_3fold requires --member-id.")
        if args.fold_count != 3:
            raise ValueError("stratified_3fold requires --fold-count 3.")
        if args.fold_id is None or not 0 <= args.fold_id < args.fold_count:
            raise ValueError("stratified_3fold requires --fold-id in [0, 2].")
    elif args.fold_id is not None:
        raise ValueError("--fold-id is only valid with stratified_3fold.")

    is_ewc_resume = args.resume_ewc_checkpoint is not None
    is_replay_resume = args.resume_replay_checkpoint is not None
    is_resume = is_ewc_resume or is_replay_resume
    if is_resume and not args.outdir.is_dir():
        resume_kind = "EWC" if is_ewc_resume else "Replay"
        raise FileNotFoundError(
            f"{resume_kind} resume requires the existing run output directory: {args.outdir}"
        )
    run_paths = ensure_run_paths(args.outdir, require_empty=not is_resume)
    set_configured_seeds(seed_configuration)
    device = resolve_device(args.device)
    feature_columns = load_feature_columns(args.feature_cols)
    if args.variant == "tddi_paper_member" and len(feature_columns) != TDDI_PAPER_INPUT_DIM:
        raise ValueError(
            "tddi_paper_member requires exactly "
            f"{TDDI_PAPER_INPUT_DIM} numerical features, got {len(feature_columns)}."
        )
    scaler_payload = load_scaler_payload(args.scaler)
    task_spec = load_task_spec(args.task_file)
    tasks = task_spec["tasks"]
    num_tasks = len(tasks)
    fold_assignments = (
        build_stratified_fold_assignments(
            (args.train, args.validation),
            fold_count=args.fold_count,
            seed=args.fold_seed,
        )
        if args.ensemble_mode == "stratified_3fold"
        else None
    )
    ewc_checkpoint_config = (
        build_ewc_checkpoint_config(args, task_spec)
        if args.method == "ewc"
        else None
    )
    replay_checkpoint_config = (
        build_replay_checkpoint_config(args, task_spec)
        if args.method == FIXED_BUDGET_METHOD
        else None
    )
    resume_checkpoint: LoadedEWCCheckpoint | None = None
    replay_resume_checkpoint: LoadedReplayCheckpoint | None = None
    if is_ewc_resume:
        if ewc_checkpoint_config is None:
            raise RuntimeError("EWC resume config was not initialized.")
        resume_checkpoint = load_ewc_checkpoint(
            args.resume_ewc_checkpoint,
            expected_seed_metadata=seed_provenance(args),
            expected_config=ewc_checkpoint_config,
        )
        expected_resume_map = expected_seen_map_after_task(
            tasks,
            resume_checkpoint.completed_task_id,
        )
        if resume_checkpoint.seen_class_map != expected_resume_map:
            raise ValueError(
                "EWC checkpoint class map does not match the configured task protocol."
            )
        if resume_checkpoint.next_task_id >= num_tasks:
            raise ValueError("EWC checkpoint already completed the final configured task.")
        run_id = resume_checkpoint.run_id
        validate_resume_run_config(run_paths["run_config_json"], resume_checkpoint)
    elif is_replay_resume:
        if replay_checkpoint_config is None:
            raise RuntimeError("Replay resume config was not initialized.")
        replay_resume_checkpoint = load_replay_checkpoint(
            args.resume_replay_checkpoint,
            expected_seed_metadata=seed_provenance(args),
            expected_config=replay_checkpoint_config,
        )
        expected_resume_map = expected_seen_map_after_task(
            tasks,
            replay_resume_checkpoint.completed_task_id,
            resume_kind="Replay",
        )
        if replay_resume_checkpoint.seen_class_map != expected_resume_map:
            raise ValueError(
                "Replay checkpoint class map does not match the configured task protocol."
            )
        if replay_resume_checkpoint.next_task_id >= num_tasks:
            raise ValueError("Replay checkpoint already completed the final configured task.")
        run_id = replay_resume_checkpoint.run_id
        validate_replay_resume_run_config(
            run_paths["run_config_json"], replay_resume_checkpoint
        )
    else:
        run_id = uuid.uuid4().hex
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

    first_task_by_class = build_first_task_lookup(tasks)
    train_count_by_class = load_class_counts(args.train)
    missing_train_counts = sorted(set(first_task_by_class) - set(train_count_by_class))
    if missing_train_counts:
        raise ValueError(f"Task classes missing from the full train split: {missing_train_counts}")
    classwise_tracker = ClasswiseTracker(
        seed=args.experiment_seed,
        method=args.method,
        first_task_by_class=first_task_by_class,
        train_count_by_class=train_count_by_class,
    )
    result_matrix = init_result_matrix(num_tasks)
    result_rows: list[dict[str, Any]] = []
    best_task_rows: list[dict[str, Any]] = []
    training_audit_rows: list[dict[str, Any]] = []
    replay_budget_audit_rows: list[dict[str, Any]] = []

    if not is_resume:
        logger.log_event(
            "run_started",
            f"CIL training started run_id={run_id} method={args.method} "
            f"protocol={method_protocol_name(args.method)} "
            f"tasks={num_tasks} device={device}",
            payload_json=json.dumps(
                {
                    "run_id": run_id,
                    "method_protocol": method_protocol_name(args.method),
                    "sampler_policy": sampler_policy_name(args.method),
                    "validation_policy": "all_seen_classes_for_early_stopping",
                    **seed_provenance(args),
                },
                sort_keys=True,
            ),
        )
    elif resume_checkpoint is not None:
        logger.log_event(
            "run_resumed",
            f"EWC run resumed after task={resume_checkpoint.completed_task_id} "
            f"next_task={resume_checkpoint.next_task_id} "
            f"checkpoint={args.resume_ewc_checkpoint}",
            payload_json=json.dumps(
                {
                    "completed_task_id": resume_checkpoint.completed_task_id,
                    "next_task_id": resume_checkpoint.next_task_id,
                    **seed_provenance(args),
                },
                sort_keys=True,
            ),
        )
    else:
        if replay_resume_checkpoint is None:
            raise RuntimeError("Replay resume checkpoint was not loaded.")
        logger.log_event(
            "run_resumed",
            f"Replay run resumed after task={replay_resume_checkpoint.completed_task_id} "
            f"next_task={replay_resume_checkpoint.next_task_id} "
            f"checkpoint={args.resume_replay_checkpoint}",
            payload_json=json.dumps(
                {
                    "completed_task_id": replay_resume_checkpoint.completed_task_id,
                    "next_task_id": replay_resume_checkpoint.next_task_id,
                    **seed_provenance(args),
                },
                sort_keys=True,
            ),
        )

    replay_buffer: FixedBudgetReplayBuffer | None = None
    if args.method == FIXED_BUDGET_METHOD:
        if replay_resume_checkpoint is not None:
            replay_buffer = restore_fixed_replay_buffer(
                replay_resume_checkpoint.replay_buffer_state,
                expected_seen_class_map=replay_resume_checkpoint.seen_class_map,
                expected_total_memory_budget=args.total_memory_budget,
                expected_random_seed=args.experiment_seed,
            )
        else:
            replay_buffer = FixedBudgetReplayBuffer(
                total_memory_budget=args.total_memory_budget,
                random_seed=args.experiment_seed,
            )
    episodic_memory = (
        TaskEpisodicMemory(
            total_budget=args.episodic_memory_budget,
            total_tasks=num_tasks,
            random_seed=args.experiment_seed,
        )
        if args.method in GRADIENT_EPISODIC_METHODS
        else None
    )
    previous_model: nn.Module | None = None
    previous_seen_map: dict[int, int] | None = None
    previous_seen_raw_classes: list[int] | None = None
    fisher_total: dict[str, "torch.Tensor"] | None = None
    theta_star: dict[str, "torch.Tensor"] | None = None
    start_task_id = 0
    if resume_checkpoint is not None:
        (
            result_matrix,
            result_rows,
            best_task_rows,
            training_audit_rows,
        ) = restore_ewc_progress(
            resume_checkpoint,
            num_tasks=num_tasks,
            classwise_tracker=classwise_tracker,
        )
        previous_seen_map = dict(resume_checkpoint.seen_class_map)
        previous_seen_raw_classes = ordered_raw_classes(previous_seen_map)
        previous_model = expand_model_for_seen_classes(
            previous_model=None,
            previous_seen_map=None,
            current_seen_map=previous_seen_map,
            variant=args.variant,
            input_dim=len(feature_columns),
            dropout=args.dropout,
            activation=args.activation,
            norm=args.norm,
        )
        theta_star = restore_model_and_theta_star(previous_model, resume_checkpoint)
        fisher_total = {
            name: value.detach().clone()
            for name, value in resume_checkpoint.fisher_total.items()
        }
        start_task_id = resume_checkpoint.next_task_id
        # Rebuilding the previous model consumes Torch RNG. Restore only after that
        # construction so the first operation for the next task matches a continuous run.
        restore_rng_state(resume_checkpoint.rng_state)
    elif replay_resume_checkpoint is not None:
        (
            result_matrix,
            result_rows,
            best_task_rows,
            training_audit_rows,
            replay_budget_audit_rows,
        ) = restore_replay_progress(
            replay_resume_checkpoint,
            num_tasks=num_tasks,
            classwise_tracker=classwise_tracker,
        )
        previous_seen_map = dict(replay_resume_checkpoint.seen_class_map)
        previous_seen_raw_classes = list(replay_resume_checkpoint.seen_raw_classes)
        previous_model = expand_model_for_seen_classes(
            previous_model=None,
            previous_seen_map=None,
            current_seen_map=previous_seen_map,
            variant=args.variant,
            input_dim=len(feature_columns),
            dropout=args.dropout,
            activation=args.activation,
            norm=args.norm,
        )
        restore_replay_model(previous_model, replay_resume_checkpoint)
        previous_model.eval()
        for parameter in previous_model.parameters():
            parameter.requires_grad_(False)
        start_task_id = replay_resume_checkpoint.next_task_id
        # Model reconstruction consumes Torch RNG; reset it so task N starts exactly
        # where the uninterrupted trajectory did at the preceding task boundary.
        restore_rng_state(replay_resume_checkpoint.rng_state)

    for task in tasks[start_task_id:]:
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
        memory_before = (
            episodic_memory.total_size
            if episodic_memory is not None
            else replay_buffer.total_size if replay_buffer is not None else 0
        )
        memory_before_by_class = (
            dict(replay_buffer.memory_counts)
            if replay_buffer is not None
            else {}
        )
        if task_id == 0 and memory_before != 0:
            raise RuntimeError("Replay memory must be empty before task 0.")

        logger.log_event(
            "task_started",
            f"task_id={task_id} current_classes={len(current_raw_classes)} seen_classes={len(seen_raw_classes)}",
        )

        if fold_assignments is None:
            current_train, current_ranking_features = load_backbone_split(
                args,
                args.train,
                feature_columns,
                scaler_payload,
                class_ids=current_raw_classes,
                max_rows=args.max_train_rows_per_task,
                include_ranking_features=args.method == FIXED_BUDGET_METHOD,
            )
        else:
            current_train, current_ranking_features = load_development_fold_split(
                args,
                feature_columns,
                scaler_payload,
                fold_assignments,
                class_ids=current_raw_classes,
                held_out=False,
                max_rows=args.max_train_rows_per_task,
                include_ranking_features=args.method == FIXED_BUDGET_METHOD,
            )
        export_validation_member = (
            args.export_member_predictions
            and "validation" in args.member_prediction_splits
        )
        if fold_assignments is None:
            validation_seen, _ = load_backbone_split(
                args,
                args.validation,
                feature_columns,
                scaler_payload,
                class_ids=seen_raw_classes,
                include_metadata=export_validation_member,
                max_rows=args.max_validation_rows_per_task,
            )
        else:
            validation_seen, _ = load_development_fold_split(
                args,
                feature_columns,
                scaler_payload,
                fold_assignments,
                class_ids=seen_raw_classes,
                held_out=True,
                max_rows=args.max_validation_rows_per_task,
            )

        replay_examples_available = 0
        replay_raw_labels = np.empty((0,), dtype=np.int64)
        if args.method != FIXED_BUDGET_METHOD:
            train_features = current_train.features
            train_raw_labels = current_train.labels
        else:
            if replay_buffer is None:
                raise RuntimeError("Fixed-budget replay buffer was not initialized.")
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
        if args.method == FIXED_BUDGET_METHOD:
            fixed_sampler = FixedReplaySampler(
                current_count=int(len(current_train.labels)),
                replay_raw_labels=replay_raw_labels,
                replay_draws_per_epoch=(0 if task_id == 0 else args.replay_draws_per_epoch),
                # Buffer membership is shared under experiment_seed, while the
                # presentation order is deliberately independent per member.
                # Legacy runs resolve member_seed == experiment_seed, preserving
                # their exact sampler behavior.
                seed=args.member_seed,
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
        episodic_memory_examples_available = (
            episodic_memory.total_size if episodic_memory is not None else 0
        )
        expected_replay_draws = 0.0
        if args.method == FIXED_BUDGET_METHOD and task_id > 0:
            expected_replay_draws = float(args.replay_draws_per_epoch)
        logger.log_event(
            "training_protocol",
            f"task={task_id} sampler={sampler_policy_name(args.method)} "
            f"current_examples={len(current_train.labels)} "
            f"replay_examples={replay_examples_available} "
            f"episodic_memory_examples={episodic_memory_examples_available} "
            f"training_examples={len(train_local_labels)} "
            f"draws_per_epoch={samples_drawn_per_epoch}",
            payload_json=json.dumps(
                {
                    "task": task_id,
                    "sampler_policy": sampler_policy_name(args.method),
                    "current_examples": int(len(current_train.labels)),
                    "replay_examples_available": replay_examples_available,
                    "episodic_memory_examples_available": episodic_memory_examples_available,
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
        ).to(device)

        if args.method == "ewc" and fisher_total is not None and previous_seen_map is not None:
            fisher_total = {
                name: value.to(device) for name, value in fisher_total.items()
            }
            if theta_star is None:
                raise RuntimeError("EWC theta_star is missing for a resumed task.")
            theta_star = {name: value.to(device) for name, value in theta_star.items()}
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
        projection_audits: list[GradientProjectionAudit] = []

        for epoch in range(1, args.epochs + 1):
            epochs_trained = epoch
            gradient_projector: Callable[[int], None] | None = None
            if episodic_memory is not None and episodic_memory.total_size > 0:

                def gradient_projector(
                    optimizer_step: int,
                    *,
                    current_epoch: int = epoch,
                ) -> None:
                    projection_audits.append(
                        apply_episodic_gradient_projection(
                            model,
                            criterion,
                            episodic_memory,
                            current_seen_map,
                            device,
                            method=args.method,
                            gem_reference_batch_size=args.batch_size,
                            agem_reference_batch_size=args.agem_reference_batch_size,
                            seed=args.experiment_seed,
                            task_id=task_id,
                            epoch=current_epoch,
                            optimizer_step=optimizer_step,
                        )
                    )

            loss_components = train_one_epoch(
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
                gradient_projector=gradient_projector,
                return_loss_components=True,
            )
            if not isinstance(loss_components, EpochLossComponents):
                raise RuntimeError("Training did not return epoch loss components.")
            train_loss = loss_components.total_loss
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
            epoch_message = (
                f"task={task_id} epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_macro_f1={val_metrics['macro_f1']:.6f} "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.6f}"
            )
            if args.method == "ewc":
                epoch_message += (
                    f" classification_loss={loss_components.classification_loss:.6f}"
                    f" raw_ewc_penalty={loss_components.raw_ewc_penalty:.6f}"
                    f" scaled_ewc_penalty={loss_components.scaled_ewc_penalty:.6f}"
                    f" total_loss={loss_components.total_loss:.6f}"
                )
                logger.log_event(
                    "ewc_loss_components",
                    epoch_message,
                    payload_json=json.dumps(
                        {
                            "task": task_id,
                            "epoch": epoch,
                            "classification_loss": loss_components.classification_loss,
                            "raw_ewc_penalty": loss_components.raw_ewc_penalty,
                            "scaled_ewc_penalty": loss_components.scaled_ewc_penalty,
                            "total_loss": loss_components.total_loss,
                            "ewc_lambda": args.ewc_lambda,
                        },
                        sort_keys=True,
                    ),
                )
            else:
                logger.log(epoch_message)

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

        projection_constraint_checks = sum(
            audit.constraint_count for audit in projection_audits
        )
        projection_violations = sum(audit.violated for audit in projection_audits)
        finite_before = [
            audit.minimum_dot_before
            for audit in projection_audits
            if math.isfinite(audit.minimum_dot_before)
        ]
        finite_after = [
            audit.minimum_dot_after
            for audit in projection_audits
            if math.isfinite(audit.minimum_dot_after)
        ]
        minimum_constraint_dot_before = min(finite_before, default=math.nan)
        minimum_constraint_dot_after = min(finite_after, default=math.nan)
        if args.method in GRADIENT_EPISODIC_METHODS:
            logger.log_event(
                "gradient_projection_audit",
                f"task={task_id} method={args.method} "
                f"optimizer_steps_checked={len(projection_audits)} "
                f"violations={projection_violations}",
                payload_json=json.dumps(
                    {
                        "task": task_id,
                        "method": args.method,
                        "optimizer_steps_checked": len(projection_audits),
                        "constraint_checks": projection_constraint_checks,
                        "violations": projection_violations,
                        "minimum_dot_before": (
                            minimum_constraint_dot_before
                            if math.isfinite(minimum_constraint_dot_before)
                            else None
                        ),
                        "minimum_dot_after": (
                            minimum_constraint_dot_after
                            if math.isfinite(minimum_constraint_dot_after)
                            else None
                        ),
                    },
                    sort_keys=True,
                ),
            )

        model.load_state_dict(best_state)
        checkpoint_path = checkpoint_dir / f"task_{task_id}_model.pt"
        if export_validation_member:
            validation_export_result = evaluate_model(
                model,
                validation_loader,
                criterion,
                device,
                inverse_seen_map,
                evaluation_class_indices=sorted(inverse_seen_map),
                collect_outputs=True,
            )
            export_member_evaluation(
                validation_export_result,
                validation_seen.metadata,
                run_paths=run_paths,
                run_id=run_id,
                args=args,
                task_id=task_id,
                split="validation",
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

        if episodic_memory is not None:
            episodic_memory.update(
                task_id,
                current_train.features,
                current_train.labels,
            )
            episodic_memory.save_summary(memory_dir / "memory_summary.csv")
            episodic_memory.save_snapshot(
                memory_dir / f"memory_after_task_{task_id}.parquet"
            )
        elif replay_buffer is not None:
            replay_buffer.update(
                current_train.features,
                current_train.labels,
                ranking_features=current_ranking_features,
            )
            replay_buffer.save_summary(memory_dir / "memory_summary.csv")
            replay_buffer.save_snapshot(memory_dir / f"memory_after_task_{task_id}.parquet")
        memory_after = (
            episodic_memory.total_size
            if episodic_memory is not None
            else replay_buffer.total_size if replay_buffer is not None else 0
        )

        if args.method == FIXED_BUDGET_METHOD:
            if replay_buffer is None or fixed_sampler is None:
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
                    seed=args.experiment_seed,
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
                "seed": args.experiment_seed,
                **seed_provenance(args),
                "method": args.method,
                "method_protocol": method_protocol_name(args.method),
                "task": task_id,
                "current_class_count": len(current_raw_classes),
                "old_class_count": old_class_count,
                "seen_class_count": len(seen_raw_classes),
                "current_dataset_size": int(len(current_train.labels)),
                "replay_examples_available": replay_examples_available,
                "episodic_memory_examples_available": episodic_memory_examples_available,
                "training_dataset_size": int(len(train_local_labels)),
                "validation_dataset_size": int(len(validation_local_labels)),
                "memory_before": memory_before,
                "memory_after": memory_after,
                "sampler_policy": sampler_policy_name(args.method),
                **fixed_replay_seed_provenance(args),
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
                    args.total_memory_budget
                    if args.method == FIXED_BUDGET_METHOD
                    else (
                        args.episodic_memory_budget
                        if args.method in GRADIENT_EPISODIC_METHODS
                        else np.nan
                    )
                ),
                "episodic_memory_budget": (
                    args.episodic_memory_budget
                    if args.method in GRADIENT_EPISODIC_METHODS
                    else np.nan
                ),
                "replay_draws_per_epoch_budget": (
                    args.replay_draws_per_epoch if args.method == FIXED_BUDGET_METHOD else np.nan
                ),
                "distillation_active": teacher_model is not None,
                "gradient_projection_active": args.method in GRADIENT_EPISODIC_METHODS,
                "gradient_projection_steps_checked": len(projection_audits),
                "gradient_constraint_checks": projection_constraint_checks,
                "gradient_constraint_violations": projection_violations,
                "gradient_constraint_violation_rate": (
                    projection_violations / len(projection_audits)
                    if projection_audits
                    else 0.0
                ),
                "minimum_constraint_dot_before": minimum_constraint_dot_before,
                "minimum_constraint_dot_after": minimum_constraint_dot_after,
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
        )
        previous_model.load_state_dict(best_state)
        if args.method == FIXED_BUDGET_METHOD:
            previous_model.eval()
            for parameter in previous_model.parameters():
                parameter.requires_grad_(False)
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
        export_test_member = (
            args.export_member_predictions
            and "test" in args.member_prediction_splits
        )
        seen_test_arrays, _ = load_backbone_split(
            args,
            args.test,
            feature_columns,
            scaler_payload,
            class_ids=seen_raw_classes,
            include_metadata=export_test_member,
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
            collect_outputs=export_test_member,
        )
        final_seen_metrics = final_seen_result.metrics
        classwise_metrics = final_seen_result.classwise_metrics
        if classwise_metrics is None:
            raise RuntimeError("Seen-class evaluation did not return class-wise metrics.")
        if export_test_member:
            export_member_evaluation(
                final_seen_result,
                seen_test_arrays.metadata,
                run_paths=run_paths,
                run_id=run_id,
                args=args,
                task_id=task_id,
                split="test",
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
        if args.method == "ewc":
            if fisher_total is None:
                raise RuntimeError("EWC Fisher is missing at the completed task boundary.")
            if ewc_checkpoint_config is None:
                raise RuntimeError("EWC checkpoint config was not initialized.")
            ewc_checkpoint_path = checkpoint_dir / "latest_ewc_state.pt"
            save_ewc_checkpoint(
                ewc_checkpoint_path,
                run_id=run_id,
                completed_task_id=task_id,
                model_state=best_state,
                fisher_total=fisher_total,
                seen_class_map=current_seen_map,
                seed_metadata=seed_provenance(args),
                config=ewc_checkpoint_config,
                progress=build_ewc_progress(
                    result_matrix=result_matrix,
                    result_rows=result_rows,
                    best_task_rows=best_task_rows,
                    training_audit_rows=training_audit_rows,
                    classwise_tracker=classwise_tracker,
                ),
            )
            logger.log_event(
                "ewc_checkpoint_saved",
                f"task={task_id} path={ewc_checkpoint_path}",
                payload_json=json.dumps(
                    {
                        "completed_task_id": task_id,
                        "next_task_id": task_id + 1,
                        "path": str(ewc_checkpoint_path),
                    },
                    sort_keys=True,
                ),
            )
        elif args.method == FIXED_BUDGET_METHOD:
            if replay_buffer is None:
                raise RuntimeError("Fixed-budget replay buffer is missing at task boundary.")
            if replay_checkpoint_config is None:
                raise RuntimeError("Replay checkpoint config was not initialized.")
            replay_checkpoint_path = checkpoint_dir / "latest_replay_distill_state.pt"
            save_replay_checkpoint(
                replay_checkpoint_path,
                run_id=run_id,
                completed_task_id=task_id,
                model_state=best_state,
                seen_class_map=current_seen_map,
                seen_raw_classes=ordered_raw_classes(current_seen_map),
                replay_buffer=replay_buffer,
                seed_metadata=seed_provenance(args),
                config=replay_checkpoint_config,
                progress=build_replay_progress(
                    result_matrix=result_matrix,
                    result_rows=result_rows,
                    best_task_rows=best_task_rows,
                    training_audit_rows=training_audit_rows,
                    replay_budget_audit_rows=replay_budget_audit_rows,
                    classwise_tracker=classwise_tracker,
                ),
            )
            logger.log_event(
                "replay_checkpoint_saved",
                f"task={task_id} path={replay_checkpoint_path}",
                payload_json=json.dumps(
                    {
                        "completed_task_id": task_id,
                        "next_task_id": task_id + 1,
                        "path": str(replay_checkpoint_path),
                    },
                    sort_keys=True,
                ),
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
        method_protocol=method_protocol_name(args.method),
        task_file=args.task_file,
        best_task_metrics=best_task_rows,
        final_test_metrics={
            "accuracy": float(final_seen_metrics["accuracy"]),
            "macro_f1": float(final_seen_metrics["macro_f1"]),
            "weighted_f1": float(final_seen_metrics["weighted_f1"]),
            "balanced_accuracy": float(final_seen_metrics["balanced_accuracy"]),
        },
        average_incremental_macro_f1=compute_average_incremental_macro_f1(result_matrix),
    )
    logger.event("run_completed", "CIL training completed successfully")


if __name__ == "__main__":
    main()
