"""Class-incremental training entrypoint for DDI2025-CIL."""

from __future__ import annotations

import argparse
import json
import sys
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
from src.data.ddi_dataset import load_feature_columns, load_scaler_payload, load_split_arrays
from src.data.replay_buffer import ReplayBuffer
from src.eval.classification_metrics import compute_classification_metrics, compute_per_class_f1
from src.eval.continual_metrics import compute_forgetting, init_result_matrix, result_matrix_to_frame
from src.methods.ewc import compute_fisher, ewc_penalty, grow_head_state
from src.methods.replay import build_training_arrays as build_replay_training_arrays
from src.methods.sequential import build_training_arrays as build_sequential_training_arrays
from src.models.mlp import MLP, preset_config
from src.utils.logging import RunLogger, ensure_run_paths
from src.utils.seed import set_global_seed


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
        description="Train class-incremental MLP baselines for DDI2025-CIL."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--feature-cols", required=True, type=Path)
    parser.add_argument("--scaler", required=True, type=Path)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--method", choices=["sequential", "joint_seen", "replay", "replay_distill", "ewc"], default="sequential")
    parser.add_argument("--variant", choices=["small", "base", "large"], default="base")
    parser.add_argument("--batch-size", type=int, default=512)
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
    parser.add_argument("--distill-alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--feature-distill-weight", type=float, default=0.5)
    parser.add_argument("--ewc-lambda", type=float, default=1000.0)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--max-train-rows-per-task", type=int, default=None)
    parser.add_argument("--max-validation-rows-per-task", type=int, default=None)
    parser.add_argument("--max-test-rows-per-task", type=int, default=None)
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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    inverse_seen_map: dict[int, int],
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            preds = logits.argmax(dim=1)
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_examples, 1)

    class_indices = sorted(np.unique(y_true).tolist())
    raw_class_ids = [inverse_seen_map[index] for index in class_indices]
    per_class = compute_per_class_f1(y_true, y_pred, class_indices, raw_class_ids)
    return metrics, per_class


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
    config = preset_config(
        variant,
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
        if key.startswith("backbone.") and key in current_state and current_state[key].shape == value.shape:
            current_state[key] = value.clone()

    previous_head_weight = previous_state["head.weight"]
    previous_head_bias = previous_state["head.bias"]
    for raw_class_id, previous_index in previous_seen_map.items():
        current_index = current_seen_map[raw_class_id]
        current_state["head.weight"][current_index] = previous_head_weight[previous_index].clone()
        current_state["head.bias"][current_index] = previous_head_bias[previous_index].clone()

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
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    student_old_indices: list[int] = []
    if teacher_model is not None and teacher_raw_classes is not None and current_seen_map is not None:
        student_old_indices = [current_seen_map[raw_class] for raw_class in teacher_raw_classes]
        teacher_model.eval()

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)

        if teacher_model is not None and student_old_indices:
            with torch.no_grad():
                teacher_logits = teacher_model(features)
                teacher_features = teacher_model.backbone(features)
            student_old_logits = logits[:, student_old_indices]
            distill_loss = F.kl_div(
                F.log_softmax(student_old_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
            student_features = model.backbone(features)
            feature_distill_loss = F.mse_loss(student_features, teacher_features)
            loss = loss + distill_alpha * distill_loss + feature_distill_weight * feature_distill_loss

        if fisher is not None and theta_star is not None and ewc_lambda:
            loss = loss + ewc_lambda * ewc_penalty(model, fisher, theta_star)

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def write_run_summary(
    path: Path,
    *,
    method: str,
    task_file: Path,
    best_task_metrics: list[dict[str, Any]],
    final_test_metrics: dict[str, float],
) -> None:
    lines = [
        "# CIL Run Summary",
        "",
        f"- method: `{method}`",
        f"- task_file: `{task_file}`",
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

    run_paths = ensure_run_paths(args.outdir)
    logger = RunLogger(
        log_path=run_paths["train_log"],
        events_path=run_paths["events_csv"],
        mirror_log_path=run_paths["stdout_log"],
    )
    checkpoint_dir = run_paths["outdir"] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = run_paths["outdir"] / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    device = resolve_device(args.device)
    feature_columns = load_feature_columns(args.feature_cols)
    scaler_payload = load_scaler_payload(args.scaler)
    task_spec = load_task_spec(args.task_file)
    tasks = task_spec["tasks"]
    num_tasks = len(tasks)
    result_matrix = init_result_matrix(num_tasks)
    result_rows: list[dict[str, Any]] = []
    best_task_rows: list[dict[str, Any]] = []
    final_per_class_frames: list[pd.DataFrame] = []

    logger.log_event(
        "run_started",
        f"CIL training started method={args.method} tasks={num_tasks} device={device}",
    )

    replay_buffer = ReplayBuffer(memory_per_class=args.memory_per_class, random_seed=args.seed)
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

        logger.log_event(
            "task_started",
            f"task_id={task_id} current_classes={len(current_raw_classes)} seen_classes={len(seen_raw_classes)}",
        )

        current_train = load_split_arrays(
            args.train,
            feature_columns,
            class_ids=current_raw_classes,
            scaler_payload=scaler_payload,
            max_rows=args.max_train_rows_per_task,
        )
        validation_seen = load_split_arrays(
            args.validation,
            feature_columns,
            class_ids=seen_raw_classes,
            scaler_payload=scaler_payload,
            max_rows=args.max_validation_rows_per_task,
        )

        if args.method == "joint_seen":
            train_seen = load_split_arrays(
                args.train,
                feature_columns,
                class_ids=seen_raw_classes,
                scaler_payload=scaler_payload,
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
        if args.method in {"replay", "replay_distill"}:
            sampler = build_balanced_sampler(train_local_labels)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

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
        if args.method == "replay_distill" and previous_model is not None and previous_seen_raw_classes is not None:
            teacher_model = previous_model.to(device)
            teacher_model.eval()

        best_epoch = -1
        best_val_metrics: dict[str, float] | None = None
        best_state: dict[str, Any] | None = None
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
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
            )
            val_metrics, _ = evaluate_model(model, validation_loader, criterion, device, inverse_seen_map)
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
        best_task_rows.append(
            {
                "task_id": task_id,
                "best_epoch": best_epoch,
                "val_macro_f1": best_val_metrics["macro_f1"],
                "val_balanced_accuracy": best_val_metrics["balanced_accuracy"],
            }
        )

        if args.method in {"replay", "replay_distill"}:
            replay_buffer.update(current_train.features, current_train.labels)
            replay_buffer.save_summary(memory_dir / "memory_summary.csv")
            replay_buffer.save_snapshot(memory_dir / f"memory_after_task_{task_id}.parquet")

        if args.method == "replay_distill":
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
        )
        previous_model.load_state_dict(best_state)
        previous_seen_map = dict(current_seen_map)
        previous_seen_raw_classes = list(seen_raw_classes)
        save_class_map(run_paths["outdir"] / f"seen_class_map_task_{task_id}.json", current_seen_map)

        final_seen_metrics: dict[str, float] | None = None
        for eval_task in tasks[: task_id + 1]:
            eval_task_id = int(eval_task["task_id"])
            eval_classes = [int(class_id) for class_id in eval_task["classes"]]
            eval_arrays = load_split_arrays(
                args.test,
                feature_columns,
                class_ids=eval_classes,
                scaler_payload=scaler_payload,
                max_rows=args.max_test_rows_per_task,
            )
            eval_labels = remap_labels(eval_arrays.labels, current_seen_map)
            eval_dataset = build_tensor_dataset(eval_arrays.features, eval_labels)
            eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
            metrics, per_class = evaluate_model(model.to(device), eval_loader, criterion, device, inverse_seen_map)
            result_matrix[task_id, eval_task_id] = metrics["macro_f1"]
            result_rows.append(
                {
                    "train_task_id": task_id,
                    "eval_task_id": eval_task_id,
                    "split": "test_task_group",
                    **metrics,
                }
            )
            if eval_task_id == task_id:
                per_class.insert(0, "train_task_id", task_id)
                per_class.insert(1, "eval_task_id", eval_task_id)
                final_per_class_frames.append(per_class)

        seen_test_arrays = load_split_arrays(
            args.test,
            feature_columns,
            class_ids=seen_raw_classes,
            scaler_payload=scaler_payload,
            max_rows=args.max_test_rows_per_task,
        )
        seen_test_labels = remap_labels(seen_test_arrays.labels, current_seen_map)
        seen_test_dataset = build_tensor_dataset(seen_test_arrays.features, seen_test_labels)
        seen_test_loader = DataLoader(seen_test_dataset, batch_size=args.batch_size, shuffle=False)
        final_seen_metrics, _ = evaluate_model(model.to(device), seen_test_loader, criterion, device, inverse_seen_map)
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
            f"seen_bal_acc={final_seen_metrics['balanced_accuracy']:.6f}",
        )

    result_matrix_frame = result_matrix_to_frame(result_matrix)
    forgetting_frame = compute_forgetting(result_matrix)
    metrics_frame = pd.DataFrame(result_rows)
    per_class_frame = (
        pd.concat(final_per_class_frames, ignore_index=True)
        if final_per_class_frames
        else pd.DataFrame(columns=["train_task_id", "eval_task_id", "raw_class_id", "class_index", "f1"])
    )

    result_matrix_frame.to_csv(run_paths["outdir"] / "task_matrix.csv", index=False)
    forgetting_frame.to_csv(run_paths["outdir"] / "forgetting.csv", index=False)
    metrics_frame.to_csv(run_paths["metrics_csv"], index=False)
    per_class_frame.to_csv(run_paths["per_class_metrics_csv"], index=False)
    final_seen_metrics = metrics_frame[metrics_frame["eval_task_id"] == "seen_all"].iloc[-1].to_dict()
    write_run_summary(
        run_paths["run_summary_md"],
        method=args.method,
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
