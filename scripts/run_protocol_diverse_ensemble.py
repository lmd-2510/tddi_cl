#!/usr/bin/env python3
"""Train one frozen-fold member per P2/P3/P4 protocol and review the final ensemble.

The existing same-protocol offline ensemble contract is intentionally not reused:
the three members have different task histories. Only their common task-7 test
probabilities are averaged. Validation artifacts remain member/protocol-specific.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_cil_tasks as task_builder  # noqa: E402
from src.data.ddi_dataset import DEFAULT_LABEL_COL  # noqa: E402
from src.eval.metrics import compute_aurc, compute_classification_metrics  # noqa: E402
from src.eval.predictions import load_member_prediction_artifact  # noqa: E402
from src.training.fold_ab_study import INPUT_KEYS, member_command  # noqa: E402
from src.training.fold_ensemble3_full import load_full_config  # noqa: E402
from src.training.fold_ensemble3_pilot import _validate_member_predictions, inspect_member  # noqa: E402

BASE_CONFIG = ROOT / "configs/p3_hybrid_equal_buffer_full8_e30_mem4.json"
PROTOCOLS = (
    {"member_id": 0, "id": "P2", "name": "head_to_tail", "task_file": None, "seed": None},
    {"member_id": 1, "id": "P3", "name": "tail_to_head", "task_file": ROOT / "study_assets/task_protocols/tail_to_head_tasks.json", "seed": None},
    {"member_id": 2, "id": "P4", "name": "constrained_mass_balanced", "task_file": ROOT / "study_assets/task_protocols/constrained_mass_balanced_seed0_tasks.json", "seed": 0},
)
LAYOUT = [38, 20, 20, 20, 20, 20, 20, 20]
HISTORICAL_TASK7 = {"macro_f1": 0.750904, "balanced_accuracy": 0.712910}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(part) for part in command) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def _p2_task_file(train_path: Path, path: Path) -> Path:
    """Freeze deterministic P2 from training-label counts only."""
    labels = pq.read_table(train_path, columns=[DEFAULT_LABEL_COL])[DEFAULT_LABEL_COL].to_numpy()
    counts = pd.Series(labels).value_counts(sort=False)
    count_frame = pd.DataFrame({"class_id": counts.index.astype(int), "count": counts.values.astype(int)})
    if count_frame["class_id"].nunique() != 178:
        raise ValueError(f"P2 requires 178 training classes; got {count_frame['class_id'].nunique()}.")
    order = task_builder.build_frequency_order(count_frame, descending=True)
    tasks = task_builder.split_order_into_tasks(order, LAYOUT)
    task_builder.validate_tasks(tasks, sorted(count_frame["class_id"].astype(int)), LAYOUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol") != "head_to_tail" or existing.get("tasks") != tasks:
            raise FileExistsError(f"Refusing to replace a different frozen P2 asset: {path}")
    else:
        task_builder.write_task_json(path, "head_to_tail", None, 178, tasks)
    return path.resolve()


def _protocol_configs(out_root: Path, input_paths: dict[str, Path]) -> list[dict[str, Any]]:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    if (base.get("training", {}).get("loss_variant") != "hybrid"
            or base.get("replay", {}).get("buffer_policy") != "fold_equal_class_capacity_v1"
            or base.get("budget_policy") != "per_member_4_percent"
            or base.get("training", {}).get("epochs") != 30
            or base.get("replay", {}).get("fraction") != 0.125
            or base.get("replay", {}).get("repeat_cap") != 3):
        raise ValueError("The selected best-recipe config changed; expected Hybrid + equal buffer, 4%/member, e30, replay 12.5%, cap 3.")
    loaded = []
    p2_path = out_root / "protocol_assets/head_to_tail_tasks.json"
    for protocol in PROTOCOLS:
        task_path = _p2_task_file(input_paths["train"], p2_path) if protocol["id"] == "P2" else Path(protocol["task_file"]).resolve()
        spec = json.loads(task_path.read_text(encoding="utf-8"))
        if (spec.get("protocol") != protocol["name"] or spec.get("seed") != protocol["seed"]
                or len(spec.get("tasks", [])) != 8
                or [len(task.get("classes", [])) for task in spec["tasks"]] != LAYOUT):
            raise ValueError(f"Invalid frozen task protocol asset for {protocol['id']}: {task_path}")
        member_root = out_root / "protocol_runs" / protocol["id"]
        prep_template = member_root / "preprocessing/member_{member_id}/B/fold_preprocessing.json"
        config = json.loads(json.dumps(base))
        config["protocol"].update({
            "id": protocol["id"], "name": protocol["name"],
            "task_file": str(task_path), "sha256": sha256(task_path),
        })
        config["preprocessing"]["artifact_template"] = str(prep_template)
        config["output_root"] = str(member_root)
        config["evaluation"]["ensemble_namespace"] = "protocol_specific_not_aggregated"
        for key, value in input_paths.items():
            config["inputs"][key] = str(value.resolve())
        config_dir = out_root / "protocol_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{protocol['id']}_{protocol['name']}.json"
        config_text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        if config_path.exists():
            if config_path.read_text(encoding="utf-8") != config_text:
                raise FileExistsError(f"Refusing to overwrite a changed protocol config: {config_path}")
        else:
            config_path.write_text(config_text, encoding="utf-8")
        loaded_config = load_full_config(config_path, project_root=ROOT)
        loaded.append({**protocol, "task_path": task_path, "config_path": config_path,
                       "config": loaded_config, "member_root": member_root})
    return loaded


def _prepare_member(config_row: dict[str, Any], input_paths: dict[str, Path], out_root: Path,
                    python: str, log_root: Path) -> None:
    member_id = int(config_row["member_id"])
    task_path = Path(config_row["task_path"])
    prep_dir = config_row["member_root"] / f"preprocessing/member_{member_id}/B"
    prep_file = prep_dir / "fold_preprocessing.json"
    if not prep_file.is_file():
        prep_dir.parent.mkdir(parents=True, exist_ok=True)
        prep_cmd = [python, str(ROOT / "scripts/prepare_fold_preprocessing.py"),
                    "--assignments", str(input_paths["fold_assignments"]),
                    "--manifest", str(input_paths["fold_manifest"]),
                    "--train", str(input_paths["train"]),
                    "--validation", str(input_paths["validation"]),
                    "--test", str(input_paths["test"]),
                    "--task-file", str(task_path),
                    "--feature-cols", str(input_paths["feature_cols"]),
                    "--member-id", str(member_id), "--validation-fold", str(member_id),
                    "--experiment-seed", "0", "--fold-seed", "42",
                    "--policy", "task0_standard_frozen", "--outdir", str(prep_dir)]
        _run(prep_cmd, log_path=log_root / f"{config_row['id']}_preprocessing.log")

    config = config_row["config"]
    command = member_command(config, member_id, python=python)
    state = inspect_member(config, member_id, command, require_inputs=True)
    if state["status"] == "complete":
        _validate_member_predictions(config, member_id)
        return
    if state["status"] == "resume":
        command.extend(("--resume-fold-checkpoint", str(state["resume_checkpoint"])))
    elif state["status"] != "fresh":
        raise RuntimeError(f"Unsafe member state for {config_row['id']}: {state['status']}")
    _run(command, log_path=log_root / f"{config_row['id']}_member{member_id}_training.log")
    final = inspect_member(config, member_id, member_command(config, member_id, python=python), require_inputs=True)
    if final["status"] != "complete":
        raise RuntimeError(f"{config_row['id']} member {member_id} ended in state {final['status']}.")
    _validate_member_predictions(config, member_id)


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, float]:
    if np.asarray(y_true).size == 0:
        return {key: float("nan") for key in (
            "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "weighted_precision", "balanced_accuracy",
        )}
    return compute_classification_metrics(y_true, y_pred, labels=labels)


def _classwise(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], task_id: int,
               group: str, protocol: str, member_id: int | None) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [{"protocol": protocol, "member_id": member_id, "task_id": task_id, "group": group,
             "raw_class_id": int(label), "test_support": int(support[i]),
             "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate(labels)]


def _summarize(out_root: Path, configs: list[dict[str, Any]], log_root: Path,
               input_paths: dict[str, Path]) -> None:
    member_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    final_artifacts = []
    task_specs: dict[str, dict[str, Any]] = {}
    train_labels = pq.read_table(input_paths["train"], columns=[DEFAULT_LABEL_COL])[DEFAULT_LABEL_COL].to_numpy()
    train_counts = Counter(int(value) for value in train_labels.tolist())
    for row in configs:
        protocol, member_id = row["id"], int(row["member_id"])
        spec = json.loads(Path(row["task_path"]).read_text(encoding="utf-8"))
        task_specs[protocol] = spec
        run_root = Path(row["config"]["output_root"])
        member_root = run_root / f"member_{member_id}"
        seen: list[int] = []
        for task in spec["tasks"]:
            task_id = int(task["task_id"])
            current = [int(value) for value in task["classes"]]
            seen.extend(current)
            artifact_path = member_root / f"member_predictions/task_{task_id}/test.npz"
            artifact = load_member_prediction_artifact(artifact_path)
            y = np.asarray(artifact.labels, dtype=np.int64)
            pred = np.asarray(artifact.raw_class_ids, dtype=np.int64)[np.asarray(artifact.probabilities).argmax(axis=1)]
            all_metrics = _metric_row(y, pred, sorted(seen))
            current_mask = np.isin(y, current)
            current_metrics = _metric_row(y[current_mask], pred[current_mask], sorted(current))
            validation_artifact = load_member_prediction_artifact(
                member_root / f"member_predictions/task_{task_id}/validation.npz")
            validation_y = np.asarray(validation_artifact.labels, dtype=np.int64)
            validation_pred = np.asarray(validation_artifact.raw_class_ids, dtype=np.int64)[
                np.asarray(validation_artifact.probabilities).argmax(axis=1)]
            validation_metrics = _metric_row(validation_y, validation_pred, sorted(seen))
            validation_current_mask = np.isin(validation_y, current)
            validation_current_metrics = _metric_row(
                validation_y[validation_current_mask], validation_pred[validation_current_mask], sorted(current))
            member_rows.append({
                "protocol": protocol, "protocol_name": row["name"], "member_id": member_id,
                "task_id": task_id, "task_classes": len(current), "seen_classes": len(seen),
                "current_test_samples": int(current_mask.sum()), "seen_test_samples": len(y),
                "validation_rows_member_fold": len(validation_y),
                **{f"seen_{key}": value for key, value in all_metrics.items()},
                **{f"current_{key}": value for key, value in current_metrics.items()},
                **{f"validation_seen_{key}": value for key, value in validation_metrics.items()},
                **{f"validation_current_{key}": value for key, value in validation_current_metrics.items()},
            })
            if task_id in (6, 7):
                classwise_rows.extend(_classwise(y, pred, sorted(seen), task_id, "seen", protocol, member_id))
            if task_id == 7:
                classwise_rows.extend(_classwise(y, pred, sorted(seen), task_id, "all_178", protocol, member_id))
                classwise_rows.extend(_classwise(y[current_mask], pred[current_mask], sorted(current), task_id, "current_task", protocol, member_id))
                final_artifacts.append((row, artifact))
            counts = Counter(y.tolist())
            schedules.append({"protocol": protocol, "member_id": member_id, "task_id": task_id,
                              "new_class_count": len(current), "seen_class_count": len(seen),
                              "new_train_samples": int(sum(train_counts[c] for c in current)),
                              "seen_train_samples": int(sum(train_counts[c] for c in seen)),
                              "new_test_samples": int(sum(counts[c] for c in current)),
                              "seen_test_samples": len(y), "new_class_ids": ",".join(map(str, current))})

    task_metrics = pd.DataFrame(member_rows)
    classwise_frame = pd.DataFrame(classwise_rows)
    schedule_frame = pd.DataFrame(schedules)
    task_metrics.to_csv(out_root / "final_results/member_task_metrics.csv", index=False)
    classwise_frame.to_csv(out_root / "final_results/member_classwise_task6_task7.csv", index=False)
    schedule_frame.to_csv(out_root / "final_results/protocol_task_support.csv", index=False)

    # Only task 7 has the same 178-class output space for all three members.
    artifacts = [item[1] for item in final_artifacts]
    reference = artifacts[0]
    ids = np.asarray(reference.sample_ids)
    labels = np.asarray(reference.labels, dtype=np.int64)
    raw_classes = np.asarray(reference.raw_class_ids, dtype=np.int64)
    probabilities_by_member = []
    for artifact in artifacts:
        if (not np.array_equal(np.asarray(artifact.sample_ids), ids)
                or not np.array_equal(np.asarray(artifact.labels, dtype=np.int64), labels)
                or not np.array_equal(np.asarray(artifact.raw_class_ids, dtype=np.int64), raw_classes)):
            raise ValueError("Task-7 members do not have identical test IDs/labels/raw-class columns.")
        probabilities = np.asarray(artifact.probabilities, dtype=np.float64)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Task-7 member probabilities are invalid.")
        probabilities_by_member.append(probabilities)
    stack = np.stack(probabilities_by_member, axis=0)
    ensemble_probabilities = stack.mean(axis=0)
    ensemble_predictions = raw_classes[ensemble_probabilities.argmax(axis=1)]
    metric = _metric_row(labels, ensemble_predictions, raw_classes.astype(int).tolist())
    entropy = -np.sum(np.clip(ensemble_probabilities, 1e-15, 1.0) *
                       np.log(np.clip(ensemble_probabilities, 1e-15, 1.0)), axis=1)
    member_pred = raw_classes[np.argmax(stack, axis=2)]
    disagreement = np.mean(np.stack([member_pred[i] != member_pred[j]
                                    for i in range(3) for j in range(i + 1, 3)]), axis=0)
    ensemble_dir = out_root / "offline_evaluation/task_7"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ensemble_dir / "test.npz", sample_ids=ids, labels=labels,
                        raw_class_ids=raw_classes, probabilities=ensemble_probabilities.astype(np.float32),
                        predictions=ensemble_predictions.astype(np.int64),
                        member_probabilities=stack.astype(np.float32), entropy=entropy,
                        pairwise_disagreement=disagreement)
    confusion = confusion_matrix(labels, ensemble_predictions, labels=raw_classes.astype(int))
    pd.DataFrame(confusion, index=raw_classes, columns=raw_classes).rename_axis("true_raw_class_id").to_csv(
        out_root / "final_results/ensemble_confusion_task7.csv")
    top_rows = []
    for i, true_id in enumerate(raw_classes):
        for j, pred_id in enumerate(raw_classes):
            if i != j and confusion[i, j]:
                top_rows.append({"true_raw_class_id": int(true_id), "predicted_raw_class_id": int(pred_id),
                                 "count": int(confusion[i, j]),
                                 "true_class_row_rate": float(confusion[i, j] / max(1, confusion[i].sum()))})
    top_rows.sort(key=lambda value: value["count"], reverse=True)
    pd.DataFrame(top_rows[:50]).to_csv(out_root / "final_results/ensemble_top_confusions_task7.csv", index=False)
    ensemble_classes = _classwise(labels, ensemble_predictions, raw_classes.astype(int).tolist(), 7,
                                  "ensemble_all_178", "protocol_diverse", None)
    pd.DataFrame(ensemble_classes).to_csv(out_root / "final_results/ensemble_classwise_task7.csv", index=False)

    # Report ensemble quality on each protocol's own final-task class group.
    group_rows = []
    for row in configs:
        spec = task_specs[row["id"]]
        current = sorted(int(c) for c in spec["tasks"][7]["classes"])
        mask = np.isin(labels, current)
        group_rows.append({"protocol": row["id"], "group": "task7_classes_under_this_protocol",
                           "class_count": len(current), "test_support": int(mask.sum()),
                           **_metric_row(labels[mask], ensemble_predictions[mask], current)})
    pd.DataFrame(group_rows).to_csv(out_root / "final_results/ensemble_protocol_final_groups.csv", index=False)

    old6 = classwise_frame[(classwise_frame.task_id == 6) & (classwise_frame.group == "seen")]
    old7 = classwise_frame[(classwise_frame.task_id == 7) & (classwise_frame.group == "seen")]
    forgetting = old6[["protocol", "member_id", "raw_class_id", "test_support", "f1"]].merge(
        old7[["protocol", "member_id", "raw_class_id", "test_support", "f1"]],
        on=["protocol", "member_id", "raw_class_id"], suffixes=("_task6", "_task7"), validate="one_to_one")
    forgetting["f1_drop_task6_to_task7"] = forgetting["f1_task6"] - forgetting["f1_task7"]
    forgetting.to_csv(out_root / "final_results/member_task6_to_task7_forgetting.csv", index=False)
    worst_forgetting = (forgetting.sort_values(["protocol", "f1_drop_task6_to_task7"], ascending=[True, False])
                        .groupby("protocol", as_index=False).head(10))
    worst_forgetting.to_csv(out_root / "final_results/member_worst_task6_to_task7_classes.csv", index=False)

    confidence = ensemble_probabilities.max(axis=1)
    expected_calibration_error = 0.0
    bin_edges = np.linspace(0.0, 1.0, 16)
    for bin_id in range(len(bin_edges) - 1):
        mask = ((confidence >= bin_edges[bin_id]) &
                (confidence <= bin_edges[bin_id + 1] if bin_id == len(bin_edges) - 2
                 else confidence < bin_edges[bin_id + 1]))
        if mask.any():
            expected_calibration_error += float(mask.mean()) * abs(
                float((ensemble_predictions[mask] == labels[mask]).mean()) - float(confidence[mask].mean()))
    one_hot = np.zeros_like(ensemble_probabilities)
    row_indices = np.arange(len(labels))
    label_positions = {int(raw): i for i, raw in enumerate(raw_classes.tolist())}
    one_hot[row_indices, np.asarray([label_positions[int(value)] for value in labels])] = 1.0
    brier = float(np.mean(np.sum((ensemble_probabilities - one_hot) ** 2, axis=1)))

    result = {
        "kind": "protocol_diverse_ensemble_task7_common_test",
        "protocols_by_member": [{"member_id": row["member_id"], "protocol_id": row["id"],
                                  "protocol_name": row["name"],
                                  "task_file_sha256": sha256(Path(row["task_path"]))}
                                 for row in configs],
        "task7_test_rows": int(len(labels)), "task7_seen_class_count": int(len(raw_classes)),
        "ensemble_task7_test": metric,
        "mean_pairwise_disagreement": float(disagreement.mean()),
        "mean_predictive_entropy": float(entropy.mean()),
        "mean_max_probability": float(confidence.mean()),
        "ece_15_bins": expected_calibration_error,
        "brier_score_multiclass": brier,
        "negative_log_likelihood": float(log_loss(labels, ensemble_probabilities,
                                                   labels=raw_classes.astype(int).tolist())),
        "aurc": compute_aurc(labels, ensemble_predictions, confidence),
        "historical_p3_focal_all_reference": HISTORICAL_TASK7,
        "delta_vs_historical": {key: float(metric[key] - value)
                                 for key, value in HISTORICAL_TASK7.items()},
        "exceeds_historical_on_both_primary_metrics": bool(
            metric["macro_f1"] > HISTORICAL_TASK7["macro_f1"]
            and metric["balanced_accuracy"] >= HISTORICAL_TASK7["balanced_accuracy"]
        ),
        "interpretation_limit": (
            "Descriptive single-seed comparison on the fixed test split; not statistical significance. "
            "No protocol-diverse OOF ensemble threshold is claimed because validation folds are disjoint "
            "and each model has a different task history."
        ),
    }
    (out_root / "final_results/protocol_diverse_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_results(out_root, task_metrics, classwise_frame, confusion, raw_classes,
                  ensemble_classes, group_rows, metric)
    _write_report(out_root, configs, task_metrics, schedule_frame, result, group_rows, top_rows,
                  worst_forgetting)


def _plot_results(out_root: Path, task_metrics: pd.DataFrame, classwise: pd.DataFrame,
                  confusion: np.ndarray, raw_classes: np.ndarray,
                  ensemble_classes: list[dict[str, Any]], group_rows: list[dict[str, Any]],
                  ensemble_metric: dict[str, float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out_root / "final_results/visualizations"
    out.mkdir(parents=True, exist_ok=True)
    final = task_metrics[task_metrics.task_id == 7]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    ax.bar(x - .18, final.seen_macro_f1, width=.36, label="Individual protocol member")
    ax.bar(x + .18, [ensemble_metric["macro_f1"]] * 3, width=.36, label="Protocol-diverse ensemble")
    ax.axhline(HISTORICAL_TASK7["macro_f1"], color="black", linestyle="--", label="Historical P3 Focal-all ensemble")
    ax.set_xticks(x, final.protocol.tolist())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Task-7 test Macro-F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_task7_macro_f1_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9))
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    image = ax.imshow(normalized, cmap="magma", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(image, ax=ax, label="True-class normalized rate")
    ax.set_title("Protocol-diverse ensemble confusion — task 7 test")
    ax.set_xlabel("Predicted raw class ID")
    ax.set_ylabel("True raw class ID")
    fig.tight_layout()
    fig.savefig(out / "02_task7_confusion_matrix.png", dpi=160)
    plt.close(fig)

    class_frame = pd.DataFrame(ensemble_classes).sort_values("raw_class_id")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(class_frame.raw_class_id, class_frame.f1, s=np.maximum(10, class_frame.test_support / 80), alpha=.75)
    ax.axhline(ensemble_metric["macro_f1"], color="black", linestyle="--",
               label=f"Macro-F1={ensemble_metric['macro_f1']:.4f}")
    ax.set(xlabel="Raw class ID", ylabel="Per-class F1", ylim=(-.03, 1.03), title="Task-7 ensemble classwise F1 (marker size ∝ support)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_task7_classwise_f1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([row["protocol"] for row in group_rows], [row["macro_f1"] for row in group_rows], color="#3288bd")
    ax.set(ylim=(0, 1), ylabel="Macro-F1", title="Same ensemble, grouped by each protocol's task-7 classes")
    fig.tight_layout()
    fig.savefig(out / "04_task7_protocol_group_metrics.png", dpi=160)
    plt.close(fig)

    seen = task_metrics[task_metrics.task_id.isin([6, 7])]
    fig, ax = plt.subplots(figsize=(10, 5))
    for protocol, rows in seen.groupby("protocol"):
        rows = rows.sort_values("task_id")
        ax.plot(rows.task_id, rows.seen_macro_f1, marker="o", label=protocol)
    ax.set(xticks=[6, 7], xlabel="Protocol-specific task boundary", ylabel="Seen-class Macro-F1",
           ylim=(0, 1), title="Member task-6 → task-7 change (each protocol's own schedule)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_member_task6_to_task7.png", dpi=160)
    plt.close(fig)


def _write_report(out_root: Path, configs: list[dict[str, Any]], task_metrics: pd.DataFrame,
                  schedule: pd.DataFrame, result: dict[str, Any], group_rows: list[dict[str, Any]],
                  top_rows: list[dict[str, Any]], worst_forgetting: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame, cols: list[str]) -> str:
        selected = frame[cols].copy()
        headers = [str(column) for column in selected.columns]
        lines = ["| " + " | ".join(headers) + " |",
                 "| " + " | ".join(["---"] * len(headers)) + " |"]
        for values in selected.itertuples(index=False, name=None):
            rendered = []
            for value in values:
                if isinstance(value, (float, np.floating)):
                    rendered.append(f"{float(value):.4f}")
                else:
                    rendered.append(str(value))
            lines.append("| " + " | ".join(rendered) + " |")
        return "\n".join(lines)

    lines = [
        "# Protocol-diverse ensemble — full-run results", "",
        "## Study design", "",
        "One model per member: member 0=P2 head-to-tail, member 1=P3 tail-to-head, member 2=P4 constrained mass-balanced. "
        "All three use the selected Hybrid + equal-class buffer recipe, 4% development memory/member, replay 12.5%, cap 3, "
        "up to 30 epochs/task, patience 5, same frozen 3-fold assignment, seed, backbone and optimizer.", "",
        "P2 was generated deterministically from training-label class counts only and frozen before training. "
        "Each member's scaler is fit only on its own task-0 training rows. P2/P3/P4 are different task histories; "
        "therefore task 0–6 scores below are protocol-specific and are not an ensemble comparison.", "",
        "Protocol is assigned one-to-one with member/fold (P2→member0/fold0, P3→member1/fold1, P4→member2/fold2). "
        "Thus a gain over the historical ensemble is evidence for this combined recipe, not a clean causal estimate of protocol diversity alone; "
        "replicate with protocol-to-member assignments rotated before making a general claim.", "",
        "## Final task-7 common-test ensemble", "",
        f"- Test rows: {result['task7_test_rows']:,}; output vocabulary: {result['task7_seen_class_count']} raw classes.",
        f"- Macro-F1: **{result['ensemble_task7_test']['macro_f1']:.4f}**; Balanced Accuracy: **{result['ensemble_task7_test']['balanced_accuracy']:.4f}**.",
        f"- Accuracy: {result['ensemble_task7_test']['accuracy']:.4f}; Weighted-F1: {result['ensemble_task7_test']['weighted_f1']:.4f}.",
        f"- Mean pairwise member disagreement: {result['mean_pairwise_disagreement']:.4f}.",
        f"- ECE (15 bins): {result['ece_15_bins']:.4f}; multiclass Brier: {result['brier_score_multiclass']:.4f}; NLL: {result['negative_log_likelihood']:.4f}; AURC: {result['aurc']:.4f}.",
        f"- Historical P3 Focal-all reference: Macro-F1 {HISTORICAL_TASK7['macro_f1']:.4f}, Balanced Accuracy {HISTORICAL_TASK7['balanced_accuracy']:.4f}.",
        f"- Deltas: Macro-F1 {result['delta_vs_historical']['macro_f1']:+.4f}; Balanced Accuracy {result['delta_vs_historical']['balanced_accuracy']:+.4f}.",
        f"- Exceeds historical on both predeclared primary metrics: **{result['exceeds_historical_on_both_primary_metrics']}**.",
        "", "This is a descriptive, single-seed comparison; exceeding the historical point estimates is not a significance test. "
        "The historical archive contains summary metrics, not paired per-example probabilities, so paired bootstrap confidence intervals cannot be computed.", "",
        "## Full per-task results by member/protocol", "",
        "For each member, `seen_*` evaluates all classes seen in that protocol by the boundary; `current_*` evaluates only classes introduced at that task.", "",
        table(task_metrics.sort_values(["member_id", "task_id"]),
              ["protocol", "member_id", "task_id", "task_classes", "seen_classes", "current_test_samples", "seen_test_samples", "seen_accuracy", "seen_macro_f1", "seen_balanced_accuracy", "seen_weighted_f1", "current_macro_f1", "validation_rows_member_fold", "validation_seen_macro_f1", "validation_seen_balanced_accuracy", "validation_current_macro_f1"]),
        "", "### Per-task sample support", "",
        table(schedule.sort_values(["member_id", "task_id"]),
              ["protocol", "member_id", "task_id", "new_class_count", "seen_class_count", "new_train_samples", "seen_train_samples", "new_test_samples", "seen_test_samples"]), "",
        "", "## Task-7 ensemble grouped by each protocol's final-task classes", "",
        "These are three descriptive groupings of the same 178-way ensemble output; no single shared notion of 'new class' exists across the three task orders.", "",
        table(pd.DataFrame(group_rows), ["protocol", "group", "class_count", "test_support", "accuracy", "macro_f1", "balanced_accuracy", "weighted_f1"]), "",
        "## Task-6 → task-7 member change", "",
        table(task_metrics[task_metrics.task_id.isin([6, 7])].sort_values(["member_id", "task_id"]),
              ["protocol", "task_id", "seen_test_samples", "seen_macro_f1", "seen_balanced_accuracy", "current_macro_f1"]),
        "", "Largest old-class F1 drops by protocol (positive means decline):", "",
        table(worst_forgetting, ["protocol", "raw_class_id", "test_support_task6", "f1_task6", "test_support_task7", "f1_task7", "f1_drop_task6_to_task7"]),
        "", "Task-7 ensemble top confusions:", "",
        table(pd.DataFrame(top_rows, columns=["true_raw_class_id", "predicted_raw_class_id", "count", "true_class_row_rate"]).head(15), ["true_raw_class_id", "predicted_raw_class_id", "count", "true_class_row_rate"]),
        "", "Class-by-class F1 and support at task 6/7 are in `member_classwise_task6_task7.csv`; final ensemble classwise scores and top confusion matrices are in adjacent CSVs.", "",
        "## Threshold and OOF caveat", "",
        "The current same-protocol OOF aggregator cannot calibrate this heterogeneous ensemble: each validation fold is held out by one member, "
        "while the other protocol members trained on that fold. This run therefore reports raw task-7 common-test ensemble metrics and uncertainty/diversity, "
        "but makes no protocol-diverse OOF threshold claim. Individual validation results remain available in each member run.", "",
        "## Reproducibility and artifacts", "",
        "| Member | Protocol | Task-file SHA256 | Config |", "|---:|---|---|---|",
    ]
    for row in configs:
        lines.append(f"| {row['member_id']} | {row['id']} `{row['name']}` | `{sha256(Path(row['task_path']))}` | `{Path(row['config_path']).name}` |")
    lines.extend([
        "", "Settings are archived per member in `protocol_configs/`, alongside task assets, full run summaries, metrics, audits, logs and plots. "
        "Checkpoints, source Parquet files and prediction NPZ arrays are excluded from the compact review ZIP; the already-computed aggregate metrics/figures remain included.",
        "", "## Figures", "",
        "- `visualizations/01_task7_macro_f1_comparison.png`", "- `visualizations/02_task7_confusion_matrix.png`",
        "- `visualizations/03_task7_classwise_f1.png`", "- `visualizations/04_task7_protocol_group_metrics.png`",
        "- `visualizations/05_member_task6_to_task7.png`", "",
        "Each individual member also has the standard five test-split diagnostic figures for its own task 6 and task 7 under `visualizations/P2/`, `visualizations/P3/`, or `visualizations/P4/`.", "",
    ])
    (out_root / "final_results/PROTOCOL_DIVERSE_ENSEMBLE_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(out_root: Path, configs: list[dict[str, Any]]) -> None:
    manifest = {
        "artifact_kind": "ddi_cil_protocol_diverse_ensemble_v1", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "complete": True,
        "experiment_seed": 0, "fold_seed": 42, "task_layout": LAYOUT,
        "best_historical_recipe_changes": {"loss": "hybrid_focal_current_ce_replay", "buffer": "fold_equal_class_capacity_v1"},
        "members": [{"member_id": row["member_id"], "protocol_id": row["id"], "protocol_name": row["name"],
                     "task_file": str(row["task_path"]), "task_file_sha256": sha256(Path(row["task_path"])),
                     "config": str(row["config_path"]), "config_sha256": sha256(Path(row["config_path"])),
                     "run_root": str(row["member_root"])} for row in configs],
        "ensemble_scope": "task_7_common_test_only", "validation_scope": "protocol_specific_member_validation",
        "preprocessing_scope": "each_member_task0_training_partition_only",
    }
    (out_root / "full_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _visualize_members(configs: list[dict[str, Any]], input_paths: dict[str, Path],
                       out_root: Path, python: str, log_root: Path) -> None:
    failures: list[str] = []
    for row in configs:
        for task_id in (6, 7):
            command = [python, str(ROOT / "scripts/visualize_cil_run.py"),
                       "--run-root", str(row["member_root"]),
                       "--outdir", str(out_root / f"visualizations/{row['id']}/task_{task_id}_test"),
                       "--task-id", str(task_id), "--member-id", str(row["member_id"]),
                       "--member-ids", str(row["member_id"]), "--split", "test",
                       "--task-file", str(row["task_path"]),
                       "--train", str(input_paths["train"]),
                       "--validation", str(input_paths["validation"]),
                       "--test", str(input_paths["test"])]
            try:
                _run(command, log_path=log_root / f"{row['id']}_visualization_task{task_id}.log")
            except subprocess.CalledProcessError as error:
                failures.append(f"{row['id']} task {task_id}: exit {error.returncode}")
    (out_root / "visualization_status.json").write_text(
        json.dumps({"exit": int(bool(failures)), "failures": failures}, indent=2) + "\n",
        encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_root = args.output_root.resolve()
    logs = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    input_paths = {key: Path(getattr(args, key)).resolve() for key in INPUT_KEYS}
    required = [BASE_CONFIG, *input_paths.values(), ROOT / "study_assets/data_schema/feature_columns.json",
                ROOT / "study_assets/task_protocols/tail_to_head_tasks.json",
                ROOT / "study_assets/task_protocols/constrained_mass_balanced_seed0_tasks.json",
                ROOT / "configs/eval_tddi_p3_ensemble_entropy_threshold.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    configs = _protocol_configs(out_root, input_paths)
    print("[OK] Protocol assets/configs validated: P2 -> member 0, P3 -> member 1, P4 -> member 2.", flush=True)
    if args.action == "check":
        print("[OK] Check complete. No preprocessing/model training was started.", flush=True)
        return
    for row in configs:
        print(f"[RUN] {row['id']} {row['name']} -> member {row['member_id']}", flush=True)
        _prepare_member(row, input_paths, out_root, args.python, logs)
    print("[RUN] Computing protocol-diverse task-7 common-test ensemble and full reports.", flush=True)
    (out_root / "final_results").mkdir(parents=True, exist_ok=True)
    _summarize(out_root, configs, logs, input_paths)
    _visualize_members(configs, input_paths, out_root, args.python, logs)
    _write_manifest(out_root, configs)
    print(f"[OK] Report: {out_root / 'final_results/PROTOCOL_DIVERSE_ENSEMBLE_RESULTS.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    for name in sorted(INPUT_KEYS):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
