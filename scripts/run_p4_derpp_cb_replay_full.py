#!/usr/bin/env python3
"""Run/resume P4 DER++ CB class-uniform members, ensemble, review and package."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ensemble_ue import (  # noqa: E402
    aggregate_member_predictions, export_offline_ensemble_artifact,
    load_offline_ensemble_artifact,
)
from src.eval.metrics import compute_classification_metrics  # noqa: E402
from src.eval.predictions import load_member_prediction_artifact  # noqa: E402
from src.data.stratified_folds import fold_file_sha256  # noqa: E402


METHOD = "DER++ CB Class-Uniform Replay"
BUFFER_POLICY = "online_equal_class_reservoir_class_uniform_replay_v1"
REPLAY_POLICY = "balanced_cycle_class_uniform_then_exemplar_uniform_v1"
LAYOUT = [38, 20, 20, 20, 20, 20, 20, 20]
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1",
           "weighted_f1", "balanced_accuracy")


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n[COMMAND] " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def _counts(path: Path) -> dict[int, int]:
    frame = pd.read_parquet(path, columns=["class"])
    values = frame["class"].astype(int).value_counts(sort=False)
    return {int(label): int(count) for label, count in values.items()}


def _group(count: int) -> str:
    return "Tail" if count <= 100 else ("Mid" if count <= 1000 else "Head")


def _bin4(count: int) -> str:
    if count <= 20:
        return "ultra_tail"
    if count <= 100:
        return "tail"
    if count <= 1000:
        return "medium"
    return "head"


def audit_p4(task_file: Path, train: Path, validation: Path, test: Path) -> dict[str, Any]:
    spec = json.loads(task_file.read_text(encoding="utf-8"))
    tasks = spec.get("tasks", [])
    classes = [int(raw) for task in tasks for raw in task.get("classes", [])]
    if (spec.get("protocol") != "constrained_mass_balanced" or len(tasks) != 8
            or [len(task.get("classes", [])) for task in tasks] != LAYOUT
            or len(classes) != 178 or len(set(classes)) != 178):
        raise ValueError("Frozen P4 must be constrained_mass_balanced, layout 38+7x20, 178 unique classes.")
    train_counts, val_counts, test_counts = _counts(train), _counts(validation), _counts(test)
    if set(classes) != set(train_counts):
        raise ValueError("P4 class universe does not match TRAIN class statistics.")
    rows = []
    for task in tasks:
        raw = [int(value) for value in task["classes"]]
        groups = Counter(_group(train_counts[value]) for value in raw)
        bins = Counter(_bin4(train_counts[value]) for value in raw)
        if any(groups[name] == 0 for name in ("Tail", "Mid", "Head")):
            raise ValueError(f"P4 task {task['task_id']} does not contain Tail+Mid+Head.")
        rows.append({
            "task_id": int(task["task_id"]), "num_classes": len(raw),
            "tail_classes": groups["Tail"], "mid_classes": groups["Mid"],
            "head_classes": groups["Head"], **{f"bin_{name}_classes": bins[name] for name in
                                                  ("ultra_tail", "tail", "medium", "head")},
            "train_samples": sum(train_counts[value] for value in raw),
            "validation_samples": sum(val_counts.get(value, 0) for value in raw),
            "test_samples": sum(test_counts.get(value, 0) for value in raw),
        })
    masses = np.asarray([row["train_samples"] for row in rows], dtype=np.float64)
    ideal = float(masses.sum() / 8)
    global_stats = {
        "total_train": int(masses.sum()), "ideal_train_mass": ideal,
        "mean_train_mass": float(masses.mean()), "std_train_mass": float(masses.std(ddof=0)),
        "cv_train_mass": float(masses.std(ddof=0) / masses.mean()),
        "max_min_ratio": float(masses.max() / masses.min()),
        "max_single_class_train_count": max(train_counts.values()),
        "exact_balance_impossible_without_class_split": bool(max(train_counts.values()) > ideal),
    }
    for row in rows:
        row["deviation_from_ideal"] = row["train_samples"] - ideal
        row["relative_deviation_from_ideal"] = row["deviation_from_ideal"] / ideal
    return {"protocol": spec["protocol"], "task_file_sha256": fold_file_sha256(task_file),
            "assignment_statistics_source": "TRAIN only", "tasks": rows, "global": global_stats}


def _check(args: argparse.Namespace) -> tuple[dict, dict[str, Any]]:
    required = [args.config, args.task_file, args.train, args.validation, args.test,
                args.feature_cols, args.fold_root / "fold_assignments.parquet",
                args.fold_root / "fold_manifest.json"]
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config.get("kind") != "p4_derpp_cb_class_uniform_full8"
            or config.get("member_ids") != [0, 1, 2]
            or config.get("protocol", {}).get("name") != "constrained_mass_balanced"
            or config.get("replay", {}).get("policy") != BUFFER_POLICY
            or config.get("replay", {}).get("sampling") != REPLAY_POLICY
            or config.get("replay", {}).get("budget_scope") != "per_member"
            or config.get("replay", {}).get("member_budget") != 27778
            or config.get("replay", {}).get("global_slot_budget") != 83334
            or config.get("training", {}).get("epochs_per_task") != 1
            or config.get("training", {}).get("alpha") != 0.3
            or config.get("training", {}).get("beta") != 0.5):
        raise ValueError("P4 DER++ frozen scientific config contract changed.")
    if fold_file_sha256(args.task_file) != config["protocol"]["sha256"]:
        raise ValueError("P4 task-file hash mismatch.")
    feature_columns = json.loads(args.feature_cols.read_text(encoding="utf-8"))
    if not isinstance(feature_columns, list) or len(feature_columns) != 3780:
        raise ValueError("P4 DER++ requires exactly 3,780 frozen feature columns.")
    audit = audit_p4(args.task_file, args.train, args.validation, args.test)
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    parent = args.output_root.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.exists() or not os.access(parent, os.W_OK):
        raise PermissionError(f"Nearest output ancestor is not writable: {parent}")
    print("[OK] P4 frozen hash/layout/classes and TRAIN-only mixed Tail/Mid/Head audit.", flush=True)
    print(f"[OK] Train-mass CV={audit['global']['cv_train_mass']:.6f}; "
          f"max/min={audit['global']['max_min_ratio']:.6f}", flush=True)
    print("[OK] DER++ loss frozen; CB water-fill storage; class-uniform balanced-cycle replay.", flush=True)
    print("[OK] Memory scope=per_member; 27,778/member; total physical slots=83,334.", flush=True)
    return config, audit


def _prepare(args: argparse.Namespace) -> None:
    for member in range(3):
        target = args.preprocessing_root / f"member_{member}/B/fold_preprocessing.json"
        if target.is_file() and target.stat().st_size:
            continue
        if target.parent.exists() and any(target.parent.iterdir()):
            raise FileExistsError(f"Partial preprocessing namespace requires inspection: {target.parent}")
        _run([args.python, str(ROOT / "scripts/prepare_fold_preprocessing.py"),
              "--assignments", str(args.fold_root / "fold_assignments.parquet"),
              "--manifest", str(args.fold_root / "fold_manifest.json"),
              "--train", str(args.train), "--validation", str(args.validation),
              "--test", str(args.test), "--task-file", str(args.task_file),
              "--feature-cols", str(args.feature_cols), "--member-id", str(member),
              "--validation-fold", str(member), "--experiment-seed", "0", "--fold-seed", "42",
              "--policy", "task0_standard_frozen", "--batch-size", "2048",
              "--outdir", str(target.parent)], args.output_root / "logs/preprocessing.log")


def _member_complete(root: Path, member: int) -> bool:
    member_root = root / f"member_{member}"
    try:
        summary = json.loads((member_root / "run_summary.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (summary.get("completed_task_id") == 7
            and (member_root / "member_manifest.json").is_file()
            and all((member_root / f"checkpoints/task_{task}.pt").is_file()
                    and (member_root / f"member_predictions/task_{task}/test.npz").is_file()
                    and (member_root / f"member_predictions/task_{task}/validation.npz").is_file()
                    for task in range(8)))


def _train_members(args: argparse.Namespace, config_path: Path) -> None:
    for member in range(3):
        if _member_complete(args.output_root, member):
            print(f"[SKIP] Member {member} already completed and validated.", flush=True)
            continue
        print(f"[RUN] Member {member}; tasks resume from latest valid checkpoint.", flush=True)
        _run([args.python, "-m", "src.training.derpp_fold", "--config", str(config_path),
              "--task-file", str(args.task_file), "--train", str(args.train),
              "--validation", str(args.validation), "--test", str(args.test),
              "--feature-cols", str(args.feature_cols),
              "--fold-assignments", str(args.fold_root / "fold_assignments.parquet"),
              "--fold-manifest", str(args.fold_root / "fold_manifest.json"),
              "--preprocessing-root", str(args.preprocessing_root),
              "--output-root", str(args.output_root), "--member-id", str(member),
              "--memory-budget", "27778", "--device", args.device],
             args.output_root / f"logs/member_{member}.log")
        if not _member_complete(args.output_root, member):
            raise RuntimeError(f"Member {member} returned without a valid task-7 completion.")


def _metrics(artifact) -> dict[str, float]:
    return compute_classification_metrics(artifact.labels, artifact.predictions,
                                          labels=artifact.raw_class_ids.astype(int).tolist())


def _classwise(artifact, train_counts: dict[int, int], origin: dict[int, int]) -> pd.DataFrame:
    labels = artifact.raw_class_ids.astype(int).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        artifact.labels, artifact.predictions, labels=labels, zero_division=0)
    return pd.DataFrame([{"raw_class_id": raw, "task_origin": origin[raw],
                          "frequency_group": _group(train_counts[raw]),
                          "train_samples": train_counts[raw], "test_support": int(support[i]),
                          "precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i])} for i, raw in enumerate(labels)])


def _subset_metrics(artifact, classes: list[int]) -> dict[str, float]:
    mask = np.isin(artifact.labels, classes)
    return compute_classification_metrics(artifact.labels[mask], artifact.predictions[mask], labels=classes)


def _ensemble(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for task in range(8):
        target_root = args.output_root / f"ensemble/task_{task}"
        target_root.mkdir(parents=True, exist_ok=True)
        result[task] = {}
        for split, filename in (("validation", "oof.npz"), ("test", "test.npz")):
            target = target_root / filename
            sources = [args.output_root / f"member_{member}/member_predictions/task_{task}/{split}.npz"
                       for member in range(3)]
            if not target.is_file():
                artifacts = [load_member_prediction_artifact(path) for path in sources]
                ensemble = aggregate_member_predictions(
                    artifacts, source_artifact_paths=sources, ensemble_mode="stratified_3fold")
                export_offline_ensemble_artifact(ensemble, target)
            artifact = load_offline_ensemble_artifact(target)
            result[task][split] = artifact
        metrics = _metrics(result[task]["test"])
        _json(target_root / "ensemble_metrics.json", metrics)
    _json(args.output_root / "ensemble/ensemble_manifest.json", {
        "kind": "offline_probability_ensemble", "members": [0, 1, 2],
        "sample_alignment": "verified_by_sample_id_labels_and_raw_class_ids",
        "probability_rule": "arithmetic_mean_of_member_softmax_probabilities",
        "completed_tasks": list(range(8)), "final_status": "completed",
    })
    return result


def _distribution(values: pd.Series) -> dict[str, float]:
    data = values.astype(float).to_numpy()
    if not len(data):
        return {key: 0.0 for key in ("min", "median", "mean", "max", "std", "cv")}
    mean, std = float(data.mean()), float(data.std(ddof=0))
    return {"min": float(data.min()), "median": float(np.median(data)), "mean": mean,
            "max": float(data.max()), "std": std, "cv": (std / mean if mean else 0.0)}


def _corr(frame: pd.DataFrame) -> float | None:
    if len(frame) < 2 or frame["retained_exemplars"].nunique() < 2 or frame["total_replay_draws"].nunique() < 2:
        return None
    value = frame[["retained_exemplars", "total_replay_draws"]].corr().iloc[0, 1]
    return None if not np.isfinite(value) else float(value)


def _review(args: argparse.Namespace, config: dict, audit: dict[str, Any], ensembles) -> dict[str, Any]:
    tasks = json.loads(args.task_file.read_text(encoding="utf-8"))["tasks"]
    origin = {int(raw): int(task["task_id"]) for task in tasks for raw in task["classes"]}
    train_counts = _counts(args.train)
    member_rows, ensemble_rows, classwise_frames = [], [], []
    member_artifacts: dict[tuple[int, int], Any] = {}
    for task in range(8):
        for member in range(3):
            artifact = load_member_prediction_artifact(
                args.output_root / f"member_{member}/member_predictions/task_{task}/test.npz")
            member_artifacts[(member, task)] = artifact
            member_rows.append({"task_id": task, "member_id": member, **_metrics(artifact)})
        ensemble_rows.append({"task_id": task, **_metrics(ensembles[task]["test"])})
    member_frame, ensemble_frame = pd.DataFrame(member_rows), pd.DataFrame(ensemble_rows)
    review_root = args.output_root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    member_frame.to_csv(review_root / "member_task_metrics.csv", index=False)
    ensemble_frame.to_csv(review_root / "ensemble_task_metrics.csv", index=False)
    final_ensemble = ensembles[7]["test"]
    for member in range(3):
        frame = _classwise(member_artifacts[(member, 7)], train_counts, origin)
        frame.insert(0, "member_id", member)
        classwise_frames.append(frame)
    ensemble_classwise = _classwise(final_ensemble, train_counts, origin)
    ensemble_classwise.to_csv(args.output_root / "ensemble/classwise_task7.csv", index=False)
    pd.concat(classwise_frames).to_csv(review_root / "member_classwise_task7.csv", index=False)

    final_metrics: dict[str, Any] = {}
    final_member_metrics = []
    for member in range(3):
        metrics = _metrics(member_artifacts[(member, 7)])
        final_member_metrics.append(metrics)
        final_metrics[f"member_{member}"] = metrics
    final_metrics["member_mean"] = {key: float(np.mean([row[key] for row in final_member_metrics])) for key in METRICS}
    final_metrics["member_std_descriptive"] = {key: float(np.std([row[key] for row in final_member_metrics])) for key in METRICS}
    final_metrics["ensemble"] = _metrics(final_ensemble)
    best_index = int(np.argmax([row["macro_f1"] for row in final_member_metrics]))
    final_metrics.update({"best_member": best_index,
                          "ensemble_minus_member_mean_macro_f1": final_metrics["ensemble"]["macro_f1"] - final_metrics["member_mean"]["macro_f1"],
                          "ensemble_minus_best_member_macro_f1": final_metrics["ensemble"]["macro_f1"] - final_member_metrics[best_index]["macro_f1"]})

    old_classes = [raw for raw, task in origin.items() if task < 7]
    current_classes = [raw for raw, task in origin.items() if task == 7]
    old_current = {}
    for name, artifact in [(f"member_{m}", member_artifacts[(m, 7)]) for m in range(3)] + [("ensemble", final_ensemble)]:
        old_current[name] = {"old": _subset_metrics(artifact, old_classes),
                             "current": _subset_metrics(artifact, current_classes),
                             "seen_all": _metrics(artifact)}

    retention_rows = []
    for name, getter in ([(f"member_{m}", lambda task, m=m: member_artifacts[(m, task)]) for m in range(3)]
                         + [("ensemble", lambda task: ensembles[task]["test"]) ]):
        per_source = []
        for task in range(8):
            classes = [int(value) for value in tasks[task]["classes"]]
            intro = _subset_metrics(getter(task), classes)["macro_f1"]
            final = _subset_metrics(getter(7), classes)["macro_f1"]
            forgetting = intro - final
            per_source.append({"source_task": task, "intro_macro_f1": intro,
                               "final_macro_f1": final, "forgetting": forgetting})
            retention_rows.append({"model": name, **per_source[-1]})
    pd.DataFrame(retention_rows).to_csv(review_root / "task_origin_retention.csv", index=False)
    pd.DataFrame([row for row in retention_rows if row["model"] == "ensemble"]).to_csv(
        args.output_root / "ensemble/task_origin_metrics_task7.csv", index=False)
    forgetting_summary = {}
    for name in [f"member_{member}" for member in range(3)] + ["ensemble"]:
        selected = [row for row in retention_rows if row["model"] == name]
        old = [row["forgetting"] for row in selected if row["source_task"] < 7]
        forgetting_summary[name] = {
            "per_task": [row for row in selected if row["source_task"] < 7],
            "average_forgetting": float(np.mean(old)),
            "final_retention_macro_f1_mean": float(
                np.mean([row["final_macro_f1"] for row in selected])),
        }

    buffer_rows, exposure_rows, exposure_stats = [], [], []
    for member in range(3):
        for task in range(8):
            audit_path = args.output_root / f"member_{member}/task_{task}/buffer_audit.json"
            data = json.loads(audit_path.read_text(encoding="utf-8"))
            retained = np.asarray(list(data["allocation"].values()), dtype=float)
            quotas = data.get("target_quotas") or {}
            buffer_rows.append({"member_id": member, "task_id": task,
                                "seen_classes": len(data["allocation"]), "memory_size": data["retained_count"],
                                "mean_retained_per_class": float(retained.mean()), "min_retained": int(retained.min()),
                                "max_retained": int(retained.max()),
                                "quota_utilization": (data["retained_count"] / sum(quotas.values()) if quotas else 1.0)})
            frame = pd.read_csv(args.output_root / f"member_{member}/task_{task}/replay_exposure.csv")
            frame.insert(0, "member_id", member)
            frame.insert(1, "task_id", task)
            exposure_rows.append(frame)
            stats = _distribution(frame["total_replay_draws"])
            stats.update({"member_id": member, "task_id": task,
                          "retained_draw_correlation": _corr(frame)})
            exposure_stats.append(stats)
    pd.DataFrame(buffer_rows).to_csv(review_root / "buffer_audit.csv", index=False)
    pd.concat(exposure_rows).to_csv(review_root / "replay_exposure_per_class.csv", index=False)
    pd.DataFrame(exposure_stats).to_csv(review_root / "replay_exposure_summary.csv", index=False)

    warnings = []
    zero = ensemble_classwise[ensemble_classwise.f1 == 0]
    low = ensemble_classwise[ensemble_classwise.test_support < 5]
    if len(zero): warnings.append(f"{len(zero)} final ensemble classes have F1=0.")
    if len(low): warnings.append(f"{len(low)} final ensemble classes have test support <5.")
    final_exposure = pd.DataFrame(exposure_stats).query("task_id == 7")
    if len(final_exposure) and float(final_exposure.cv.max()) > 0.20:
        warnings.append("Task-7 empirical class replay CV exceeds 0.20 for at least one member.")
    if audit["global"]["exact_balance_impossible_without_class_split"]:
        warnings.append("Exact P4 task-mass equality is mathematically impossible without splitting at least one class.")
    attempts_path = args.output_root / "runner_attempts.txt"
    if attempts_path.is_file():
        attempt_count = len([row for row in attempts_path.read_text(encoding="utf-8").splitlines()
                             if row.strip()])
        if attempt_count > 1:
            warnings.append(f"Run was resumed/retried; recorded runner attempts: {attempt_count}.")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    run_identity = {"run_id": args.output_root.name, "method": METHOD, "protocol": "P4 constrained_mass_balanced",
                    "task_file": str(args.task_file), "task_file_sha256": config["protocol"]["sha256"],
                    "buffer_policy": BUFFER_POLICY, "replay_policy": REPLAY_POLICY,
                    "memory_budget_semantics": "per_member: 27778 each; aggregate physical slots: 83334",
                    "members": [0, 1, 2], "config": str(args.config),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit}
    report = {"run_identity": run_identity, "protocol_stats": audit,
              "members": {f"member_{m}": {"completed": True} for m in range(3)},
              "per_task_metrics": {"members": member_rows, "ensemble": ensemble_rows},
              "final_metrics": final_metrics, "ensemble_metrics": final_metrics["ensemble"],
              "old_vs_current": old_current, "task_origin_retention": retention_rows,
              "buffer_stats": buffer_rows, "replay_exposure_stats": exposure_stats,
              "forgetting": forgetting_summary, "warnings": warnings}
    _json(review_root / "REVIEW_SUMMARY.json", report)

    def fmt(value: float) -> str: return f"{value:.6f}"
    lines = ["# P4 DER++ CB Class-Uniform Replay — Review Summary", "", "## 1. Run identity", "",
             f"- Run ID: `{run_identity['run_id']}`", f"- Method: `{METHOD}`",
             "- Protocol: `P4 constrained_mass_balanced`",
             f"- Buffer: `{BUFFER_POLICY}`", f"- Replay: `{REPLAY_POLICY}`",
             f"- Memory: {run_identity['memory_budget_semantics']}", f"- Git commit: `{commit}`", "",
             "## 2. P4 audit", "",
             "| Task | Classes | Tail | Mid | Head | Train | Validation | Test | Deviation from ideal |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in audit["tasks"]:
        lines.append(f"| {row['task_id']} | {row['num_classes']} | {row['tail_classes']} | {row['mid_classes']} | {row['head_classes']} | {row['train_samples']} | {row['validation_samples']} | {row['test_samples']} | {row['deviation_from_ideal']:.1f} |")
    g = audit["global"]
    lines += ["", f"Train mass mean={g['mean_train_mass']:.2f}, std={g['std_train_mass']:.2f}, CV={g['cv_train_mass']:.6f}, max/min={g['max_min_ratio']:.6f}.", "",
              "## 3. Member results", "", "| Task | M0 Macro-F1 | M1 Macro-F1 | M2 Macro-F1 | Member mean | Ensemble Macro-F1 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for task in range(8):
        vals = [float(member_frame[(member_frame.task_id == task) &
                                   (member_frame.member_id == member)].iloc[0].macro_f1)
                for member in range(3)]
        ens = float(ensemble_frame[ensemble_frame.task_id == task].iloc[0].macro_f1)
        lines.append(f"| {task} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {fmt(float(np.mean(vals)))} | {fmt(ens)} |")
    lines += ["", "## 4. Final task 7 results", "",
              "| Model | Macro-F1 | Accuracy | Weighted-F1 | Balanced accuracy |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("member_0", "member_1", "member_2", "member_mean", "member_std_descriptive", "ensemble"):
        row = final_metrics[name]
        lines.append(f"| {name} | {fmt(row['macro_f1'])} | {fmt(row['accuracy'])} | {fmt(row['weighted_f1'])} | {fmt(row['balanced_accuracy'])} |")
    lines += ["", "## 5. Old vs current", "", "| Model | Old Macro-F1 | Current Macro-F1 | Seen-all Macro-F1 |",
              "| --- | ---: | ---: | ---: |"]
    for name, row in old_current.items():
        lines.append(f"| {name} | {fmt(row['old']['macro_f1'])} | {fmt(row['current']['macro_f1'])} | {fmt(row['seen_all']['macro_f1'])} |")
    lines += ["", "## 6. Task-origin retention", "",
              "| Model | Origin task | Intro Macro-F1 | Final Macro-F1 |", "| --- | ---: | ---: | ---: |"]
    for row in retention_rows:
        lines.append(f"| {row['model']} | {row['source_task']} | {row['intro_macro_f1']:.6f} | {row['final_macro_f1']:.6f} |")
    lines += ["", "## 7. Forgetting", "", "Forgetting is intro-boundary Macro-F1 minus final task-7 Macro-F1.", "",
              "| Model | Average forgetting (T0–T6) | Final retention mean (T0–T7) |", "| --- | ---: | ---: |"]
    for name, row in forgetting_summary.items():
        lines.append(f"| {name} | {row['average_forgetting']:.6f} | {row['final_retention_macro_f1_mean']:.6f} |")
    lines += ["", "Full per-task values are in `task_origin_retention.csv`.", "",
              "## 8. Buffer audit", "",
              "| Member | Task | Seen classes | Memory | Mean/class | Min | Max | Quota utilization |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in buffer_rows:
        lines.append(f"| {row['member_id']} | {row['task_id']} | {row['seen_classes']} | {row['memory_size']} | {row['mean_retained_per_class']:.3f} | {row['min_retained']} | {row['max_retained']} | {row['quota_utilization']:.6f} |")
    lines += ["", "## 9. Replay exposure audit", "",
              "Both logit-MSE and replay-label CE draws use the class-uniform sampler. Per-class rows are in `replay_exposure_per_class.csv`.", "",
              "| Member | Task | Min | Median | Mean | Max | Std | CV | Retained/draw corr |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in exposure_stats:
        corr = "unavailable" if row["retained_draw_correlation"] is None else f"{row['retained_draw_correlation']:.6f}"
        lines.append(f"| {row['member_id']} | {row['task_id']} | {row['min']:.1f} | {row['median']:.1f} | {row['mean']:.3f} | {row['max']:.1f} | {row['std']:.3f} | {row['cv']:.6f} | {corr} |")
    lines += ["", "## 10. Worst/best classes", ""]
    for title, frame in (("Worst 10", ensemble_classwise.nsmallest(10, "f1")), ("Best 10", ensemble_classwise.nlargest(10, "f1"))):
        lines += [f"### {title}", "", "| Raw class | F1 | Support | Frequency group |", "| ---: | ---: | ---: | --- |"]
        for row in frame.itertuples(): lines.append(f"| {row.raw_class_id} | {row.f1:.6f} | {row.test_support} | {row.frequency_group} |")
        lines.append("")
    lines += ["## 11. Ensemble analysis", "",
              f"- Member mean Macro-F1: **{fmt(final_metrics['member_mean']['macro_f1'])}**",
              f"- Best member: M{best_index}, Macro-F1 **{fmt(final_member_metrics[best_index]['macro_f1'])}**",
              f"- Ensemble Macro-F1: **{fmt(final_metrics['ensemble']['macro_f1'])}**",
              f"- Ensemble − member mean: **{fmt(final_metrics['ensemble_minus_member_mean_macro_f1'])}**",
              f"- Ensemble − best member: **{fmt(final_metrics['ensemble_minus_best_member_macro_f1'])}**", "",
              "## 12. Warnings", ""]
    lines += [f"- {warning}" for warning in warnings] or ["- None."]
    (review_root / "REVIEW_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(review_root / "REVIEW_SUMMARY.md", args.output_root / "REVIEW_SUMMARY.md")
    return report


def _package(args: argparse.Namespace) -> tuple[Path, Path]:
    archive = args.output_root / f"p4_derpp_cb_class_uniform_{args.output_root.name}_review.zip"
    allowed = {".md", ".json", ".csv", ".log", ".txt", ".png", ".yaml", ".yml"}
    excluded_parts = {"checkpoints", "interrupted_attempts"}
    files = [path for path in args.output_root.rglob("*") if path.is_file()
             and path.suffix.lower() in allowed and not any(part in excluded_parts for part in path.parts)
             and path != archive and path.name != archive.name + ".sha256"]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(files):
            output.write(path, path.relative_to(args.output_root).as_posix())
    hasher = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, checksum


def _status(root: Path) -> None:
    print(f"Run ID: {root.name}")
    print("P4 protocol: constrained_mass_balanced")
    print(f"Method: {METHOD}")
    print(f"Memory/replay: {BUFFER_POLICY} / {REPLAY_POLICY}")
    failed_exit = ((root / "runner_exit_code.txt").read_text(encoding="utf-8").strip()
                   if (root / "runner_exit_code.txt").is_file() else None)
    earlier_complete = True
    for member in range(3):
        member_root = root / f"member_{member}"
        checkpoints = sorted(member_root.glob("checkpoints/task_*.pt"))
        complete = _member_complete(root, member)
        latest = max((int(path.stem.split("_")[1]) for path in checkpoints), default=None)
        macro = None
        if (member_root / "metrics.csv").is_file():
            frame = pd.read_csv(member_root / "metrics.csv")
            if len(frame): macro = float(frame.iloc[-1].test_macro_f1)
        if complete:
            state = "completed"
        elif failed_exit not in (None, "0") and earlier_complete and member_root.exists():
            state = "failed/incomplete"
        else:
            state = "running/incomplete" if member_root.exists() else "pending"
        print(f"M{member}: {state}; latest_task={latest}; latest_macro_f1={macro}")
        earlier_complete = earlier_complete and complete
    print(f"Ensemble: {'completed' if (root / 'ensemble/task_7/ensemble_metrics.json').is_file() else 'pending'}")
    print(f"Review: {'completed' if (root / 'review/REVIEW_SUMMARY.md').is_file() else 'pending'}")
    print(f"Output path: {root}")


def run(args: argparse.Namespace) -> None:
    config, audit = _check(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(exist_ok=True)
    config_dir = args.output_root / "config"
    config_dir.mkdir(exist_ok=True)
    frozen_config = config_dir / args.config.name
    if frozen_config.exists() and frozen_config.read_bytes() != args.config.read_bytes():
        raise FileExistsError("Frozen run config differs from repository config; refusing resume.")
    if not frozen_config.exists(): shutil.copy2(args.config, frozen_config)
    frozen_task = config_dir / args.task_file.name
    if not frozen_task.exists(): shutil.copy2(args.task_file, frozen_task)
    _json(config_dir / "P4_AUDIT.json", audit)
    _prepare(args)
    _train_members(args, frozen_config)
    ensembles = _ensemble(args)
    report = _review(args, config, audit, ensembles)
    manifest = {"completed_members": [0, 1, 2], "offline_ensemble": True,
                "review_generated": True, "protocol": "P4 constrained_mass_balanced",
                "method": METHOD, "buffer_policy": BUFFER_POLICY, "replay_policy": REPLAY_POLICY,
                "task_count": 8, "final_status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    _json(args.output_root / "final_manifest.json", manifest)
    archive, checksum = _package(args)
    final = report["final_metrics"]
    print("FULL RUN COMPLETED")
    print(f"Run ID: {args.output_root.name}")
    print(f"Output: {args.output_root}")
    for member in range(3): print(f"M{member} final Macro-F1: {final[f'member_{member}']['macro_f1']:.6f}")
    print(f"Member Mean: {final['member_mean']['macro_f1']:.6f}")
    print(f"Ensemble Macro-F1: {final['ensemble']['macro_f1']:.6f}")
    print(f"Ensemble gain vs mean: {final['ensemble_minus_member_mean_macro_f1']:.6f}")
    print(f"Review: {args.output_root / 'REVIEW_SUMMARY.md'}")
    print(f"Review package: {archive}")
    print(f"SHA256: {checksum}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "status"))
    for name in ("config", "task_file", "train", "validation", "test", "feature_cols",
                 "fold_root", "preprocessing_root", "output_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "status":
        _status(args.output_root.resolve())
    elif args.action == "check":
        _check(args)
        print("[OK] Check complete; no preprocessing/model training was started.")
    else:
        run(args)


if __name__ == "__main__":
    main()
