"""Static supervised training entrypoint for descriptor-only DDI2025-CIL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from src.data.class_mapping import build_global_class_map, invert_class_map, remap_labels, save_class_map
from src.data.ddi_dataset import (
    load_feature_columns,
    load_scaler_payload,
    load_split_arrays,
    load_unique_labels,
)
from src.models.mlp import MLP, preset_config
from src.utils.logging import RunLogger, ensure_run_paths
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a static descriptor-only MLP baseline for DDI2025-CIL."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--feature-cols", required=True, type=Path)
    parser.add_argument("--scaler", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--variant",
        choices=["small", "base", "large", "tddi"],
        default="tddi",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="gelu")
    parser.add_argument("--norm", choices=["none", "layernorm", "batchnorm"], default="layernorm")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    return parser.parse_args()


def require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError(
            "torch is required to run train_static.py. Install torch before training."
        )


def resolve_device(requested: str) -> str:
    require_torch()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_tensor_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    require_torch()
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return TensorDataset(x, y)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
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
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_examples, 1)

    per_class = pd.DataFrame(
        {
            "class_index": sorted(np.unique(y_true)),
            "f1": f1_score(y_true, y_pred, average=None, labels=sorted(np.unique(y_true)), zero_division=0),
        }
    )
    return metrics, per_class


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def write_run_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    best_epoch: int,
    best_val: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    lines = [
        "# Static Run Summary",
        "",
        "## Configuration",
        "",
        f"- variant: `{args.variant}`",
        f"- batch_size: `{args.batch_size}`",
        f"- epochs: `{args.epochs}`",
        f"- lr: `{args.lr}`",
        f"- weight_decay: `{args.weight_decay}`",
        f"- seed: `{args.seed}`",
        "",
        "## Best Validation",
        "",
        f"- best_epoch: `{best_epoch}`",
        f"- val_loss: `{best_val['loss']:.6f}`",
        f"- val_macro_f1: `{best_val['macro_f1']:.6f}`",
        f"- val_balanced_accuracy: `{best_val['balanced_accuracy']:.6f}`",
        "",
        "## Test Metrics",
        "",
        f"- test_loss: `{test_metrics['loss']:.6f}`",
        f"- test_accuracy: `{test_metrics['accuracy']:.6f}`",
        f"- test_macro_f1: `{test_metrics['macro_f1']:.6f}`",
        f"- test_weighted_f1: `{test_metrics['weighted_f1']:.6f}`",
        f"- test_balanced_accuracy: `{test_metrics['balanced_accuracy']:.6f}`",
        "",
    ]
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

    set_global_seed(args.seed)
    feature_columns = load_feature_columns(args.feature_cols)
    scaler_payload = load_scaler_payload(args.scaler)
    device = resolve_device(args.device)

    logger.log_event("run_started", f"Static training started on device={device}")
    logger.log(
        f"variant={args.variant} batch_size={args.batch_size} epochs={args.epochs} "
        f"lr={args.lr} weight_decay={args.weight_decay} seed={args.seed}"
    )

    train_arrays = load_split_arrays(
        args.train,
        feature_columns,
        scaler_payload=scaler_payload,
        max_rows=args.max_train_rows,
    )
    validation_arrays = load_split_arrays(
        args.validation,
        feature_columns,
        scaler_payload=scaler_payload,
        max_rows=args.max_validation_rows,
    )
    full_train_labels = load_unique_labels(args.train)
    class_map = build_global_class_map(full_train_labels)
    save_class_map(run_paths["outdir"] / "global_class_map.json", class_map)

    y_train = remap_labels(train_arrays.labels, class_map)
    y_validation = remap_labels(validation_arrays.labels, class_map)

    train_dataset = build_tensor_dataset(train_arrays.features, y_train)
    validation_dataset = build_tensor_dataset(validation_arrays.features, y_validation)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    config = preset_config(
        args.variant,
        input_dim=train_arrays.features.shape[1],
        num_classes=len(class_map),
        dropout=args.dropout,
        activation=args.activation,
        norm=args.norm,
    )
    model = MLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    metrics_rows: list[dict[str, Any]] = []
    best_epoch = -1
    best_val_metrics: dict[str, float] | None = None
    best_state: dict[str, Any] | None = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate(model, validation_loader, criterion, device)
        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "lr": current_lr,
        }
        metrics_rows.append(row)
        logger.log(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_macro_f1={val_metrics['macro_f1']:.6f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.6f} lr={current_lr:.6g}"
        )

        if best_val_metrics is None or val_metrics["macro_f1"] > best_val_metrics["macro_f1"]:
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            torch.save(best_state, run_paths["checkpoint_pt"])
            run_paths["checkpoint_paths_txt"].write_text(
                str(run_paths["checkpoint_pt"]) + "\n",
                encoding="utf-8",
            )
            logger.log_event(
                "checkpoint_saved",
                f"Saved best checkpoint at epoch={epoch} val_macro_f1={val_metrics['macro_f1']:.6f}",
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.log_event(
                    "early_stopped",
                    f"Early stopping at epoch={epoch} after patience={args.patience}",
                )
                break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_arrays = load_split_arrays(
        args.test,
        feature_columns,
        scaler_payload=scaler_payload,
        max_rows=args.max_test_rows,
    )
    y_test = remap_labels(test_arrays.labels, class_map)
    test_dataset = build_tensor_dataset(test_arrays.features, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_metrics, per_class = evaluate(model, test_loader, criterion, device)
    inverse_class_map = invert_class_map(class_map)
    per_class["raw_class_id"] = per_class["class_index"].map(inverse_class_map)
    per_class = per_class[["raw_class_id", "class_index", "f1"]]

    pd.DataFrame(metrics_rows).to_csv(run_paths["metrics_csv"], index=False)
    per_class.to_csv(run_paths["per_class_metrics_csv"], index=False)
    write_run_summary(
        run_paths["run_summary_md"],
        args=args,
        best_epoch=best_epoch,
        best_val=best_val_metrics,
        test_metrics=test_metrics,
    )

    logger.log(
        f"finished best_epoch={best_epoch} test_macro_f1={test_metrics['macro_f1']:.6f} "
        f"test_bal_acc={test_metrics['balanced_accuracy']:.6f}"
    )
    logger.event("run_completed", "Static training completed successfully")


if __name__ == "__main__":
    main()
