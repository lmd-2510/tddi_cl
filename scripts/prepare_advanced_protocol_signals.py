#!/usr/bin/env python3
"""Prepare train-only class statistics and validation-only T-DDI signals."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.class_mapping import load_class_map, remap_labels
from src.data.ddi_dataset import load_feature_columns, load_scaler_payload, load_split_arrays
from src.models.mlp import MLP, preset_config


DRUG_A = "drugid-drug_a"
DRUG_B = "drugid-drug_b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--feature-cols", required=True, type=Path)
    parser.add_argument("--scaler", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--class-map", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--variant", choices=["tddi", "base"], default="tddi")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_train_statistics(
    args: argparse.Namespace,
    feature_columns: list[str],
    scaler_payload: dict[str, object],
) -> pd.DataFrame:
    print("[signals] Loading train features and drug IDs...", flush=True)
    arrays = load_split_arrays(
        args.train,
        feature_columns,
        scaler_payload=scaler_payload,
        include_metadata=True,
        meta_cols=[DRUG_A, DRUG_B],
    )
    assert arrays.metadata is not None
    rows: list[dict[str, float | int]] = []
    for class_id in sorted(np.unique(arrays.labels).astype(int).tolist()):
        indices = np.flatnonzero(arrays.labels == class_id)
        class_features = arrays.features[indices]
        # Mean standardized within-class variance is scale-comparable across
        # descriptor families and does not use validation/test information.
        descriptor_diversity = float(
            np.mean(np.var(class_features, axis=0, dtype=np.float64))
        )
        drugs = np.concatenate(
            [
                np.asarray(arrays.metadata[DRUG_A])[indices],
                np.asarray(arrays.metadata[DRUG_B])[indices],
            ]
        )
        count = int(indices.size)
        rows.append(
            {
                "class_id": class_id,
                "count": count,
                "effective_support_sqrt": float(np.sqrt(count)),
                "effective_support_capped_1000": min(count, 1000),
                "unique_drugs": int(np.unique(drugs).size),
                "descriptor_diversity": descriptor_diversity,
            }
        )
        if len(rows) % 25 == 0:
            print(f"[signals] Train statistics: {len(rows)} classes", flush=True)
    del arrays
    gc.collect()
    return pd.DataFrame(rows)


def build_validation_signals(
    args: argparse.Namespace,
    feature_columns: list[str],
    scaler_payload: dict[str, object],
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_map = load_class_map(args.class_map)
    inverse_map = {index: raw for raw, index in class_map.items()}
    validation = load_split_arrays(
        args.validation,
        feature_columns,
        scaler_payload=scaler_payload,
    )
    y_true = remap_labels(validation.labels, class_map)
    config = preset_config(
        args.variant,
        input_dim=validation.features.shape[1],
        num_classes=len(class_map),
    )
    model = MLP(config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation.features.astype(np.float32, copy=False)),
            torch.from_numpy(y_true.astype(np.int64, copy=False)),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    predictions: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            losses.append(F.cross_entropy(logits, labels, reduction="none").cpu().numpy())
    y_pred = np.concatenate(predictions)
    per_example_nll = np.concatenate(losses)
    labels = list(range(len(class_map)))
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    difficulty_rows = []
    for class_index in labels:
        mask = y_true == class_index
        difficulty_rows.append(
            {
                "class_id": int(inverse_map[class_index]),
                "validation_support": int(mask.sum()),
                "validation_f1": float(f1[class_index]),
                "validation_recall": float(recall[class_index]),
                "validation_mean_nll": float(per_example_nll[mask].mean()),
                "difficulty": float(1.0 - f1[class_index]),
            }
        )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = np.maximum(matrix.sum(axis=1), 1)
    edge_rows = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            raw_errors = int(matrix[left, right] + matrix[right, left])
            if raw_errors == 0:
                continue
            symmetric_rate = 0.5 * (
                matrix[left, right] / support[left]
                + matrix[right, left] / support[right]
            )
            edge_rows.append(
                {
                    "class_id_a": int(inverse_map[left]),
                    "class_id_b": int(inverse_map[right]),
                    "raw_symmetric_errors": raw_errors,
                    "symmetric_confusion_rate": float(symmetric_rate),
                }
            )
    return pd.DataFrame(difficulty_rows), pd.DataFrame(edge_rows)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    train_stats_path = args.outdir / "class_protocol_stats.csv"
    difficulty_path = args.outdir / "validation_class_difficulty.csv"
    edges_path = args.outdir / "validation_confusion_edges.csv"
    manifest_path = args.outdir / "manifest.json"
    for path in [train_stats_path, difficulty_path, edges_path, manifest_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite advanced signal artifact: {path}")

    feature_columns = load_feature_columns(args.feature_cols)
    scaler_payload = load_scaler_payload(args.scaler)
    device = resolve_device(args.device)
    train_stats = build_train_statistics(args, feature_columns, scaler_payload)
    difficulty, edges = build_validation_signals(
        args,
        feature_columns,
        scaler_payload,
        device,
    )
    train_stats.to_csv(train_stats_path, index=False)
    difficulty.to_csv(difficulty_path, index=False)
    edges.to_csv(edges_path, index=False)
    manifest = {
        "train_split": str(args.train),
        "validation_split": str(args.validation),
        "test_split_used": False,
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "class_map": str(args.class_map),
        "num_classes": int(train_stats.shape[0]),
        "num_confusion_edges": int(edges.shape[0]),
        "device": device,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] Wrote advanced protocol signals to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
