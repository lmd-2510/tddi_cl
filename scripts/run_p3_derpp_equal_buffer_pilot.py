#!/usr/bin/env python3
"""Preflight, train and visualize a member-0 DER++ equal-class-buffer pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stratified_folds import fold_file_sha256  # noqa: E402


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _check(args: argparse.Namespace) -> tuple[dict, Path]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    original = json.loads((ROOT / "configs/p3_derpp_full8_mem4.json").read_text(encoding="utf-8"))
    replay_unchanged = ("budget_policy", "member_budget", "draws_per_step", "logit_refresh")
    if (config.get("kind") != "p3_derpp_equal_class_pilot"
            or config.get("member_ids") != [0]
            or config.get("experiment_seed") != 0 or config.get("fold_seed") != 42
            or config.get("training") != original["training"]
            or config.get("model") != original["model"]
            or config.get("protocol") != original["protocol"]
            or config.get("preprocessing") != original["preprocessing"]
            or config.get("evaluation", {}).get("future_class_mask") is not True
            or config.get("replay", {}).get("policy") != "online_equal_class_reservoir_logits_v1"
            or any(config["replay"].get(key) != original["replay"].get(key)
                   for key in replay_unchanged)
            or config["replay"].get("member_budget") != 27778
            or config["replay"].get("global_slot_budget") != 27778
            or config["replay"].get("draws_per_step") != 2
            or config["replay"].get("logit_refresh") is not False
            or config["replay"].get("sampling") != "uniform_over_retained_slots"):
        raise ValueError("Pilot must differ from original DER++ only by member scope and buffer storage.")
    task_file = ROOT / config["protocol"]["task_file"]
    if fold_file_sha256(task_file) != config["protocol"]["sha256"]:
        raise ValueError("Frozen P3 task-file SHA256 mismatch.")
    required = [args.train, args.validation, args.test, args.feature_cols,
                args.fold_root / "fold_assignments.parquet", args.fold_root / "fold_manifest.json",
                args.preprocessing_root / "member_0/B/fold_preprocessing.json"]
    missing = [path for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError("Missing pilot input(s): " + ", ".join(map(str, missing)))
    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError("matplotlib is required for the five pilot visualizations.")
    print("[OK] P3 member 0, tasks 0–7, frozen folds/preprocessing, 27,778 buffer slots.", flush=True)
    print("[OK] Only storage changes: equal seen-class quota + random within class; replay/loss unchanged.", flush=True)
    return config, task_file


def _summarize(root: Path, task_file: Path, config: dict) -> None:
    member_root = root / "member_0"
    summary = json.loads((member_root / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("completed_task_id") != 7 or len(summary.get("tasks", [])) != 8:
        raise ValueError("Member-0 pilot is not complete through task 7.")
    tasks = json.loads(task_file.read_text(encoding="utf-8"))["tasks"]
    rows = []
    for task in tasks:
        task_id = int(task["task_id"])
        audit = json.loads((member_root / f"task_{task_id}" / "buffer_audit.json").read_text(encoding="utf-8"))
        allocation = {int(k): int(v) for k, v in audit["allocation"].items()}
        exposure = pd.read_csv(member_root / f"task_{task_id}" / "replay_exposure.csv")
        draws = {int(row.raw_class_id): int(row.replay_draws) for row in exposure.itertuples()}
        for source in tasks[:task_id + 1]:
            raw_classes = [int(raw) for raw in source["classes"]]
            rows.append({
                "boundary_task": task_id, "source_task": int(source["task_id"]),
                "class_count": len(raw_classes),
                "buffer_slots": sum(allocation.get(raw, 0) for raw in raw_classes),
                "classes_without_exemplar": sum(allocation.get(raw, 0) == 0 for raw in raw_classes),
                "replay_draws_in_boundary": sum(draws.get(raw, 0) for raw in raw_classes),
            })
    out = root / "pilot_results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "buffer_by_source_task.csv", index=False)
    per_task = pd.read_csv(member_root / "metrics.csv")
    per_task.to_csv(out / "member0_task_metrics.csv", index=False)
    old_new = pd.read_csv(member_root / "task_7" / "metrics.csv")
    old_new.to_csv(out / "task7_old_current_metrics.csv", index=False)
    task6 = per_task.loc[per_task.task_id == 6].iloc[0]
    task7 = per_task.loc[per_task.task_id == 7].iloc[0]
    final_buffer = pd.DataFrame(rows)
    final_buffer = final_buffer[final_buffer.boundary_task == 7]
    lines = [
        "# DER++ P3 member-0 equal-class-buffer pilot", "",
        "Only member 0 was trained. This is NOT an offline three-member ensemble.",
        "The original DER++ one-pass training, 178-way head, alpha=0.3, beta=0.5, "
        "optimizer, and uniform replay draws are unchanged. Only buffer storage changes.",
        "Class quotas use previously observed training counts plus the current task's "
        "training counts; never future-task, validation or test samples.", "",
        "| Task | Train rows | Validation Macro-F1 | Test Macro-F1 | Retained buffer |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_task.itertuples():
        lines.append(f"| {row.task_id} | {row.train_rows} | {row.validation_macro_f1:.6f} | "
                     f"{row.test_macro_f1:.6f} | {row.memory_after} |")
    lines.extend([
        "", f"Task 6→7 test Macro-F1 change: {task7.test_macro_f1 - task6.test_macro_f1:+.6f}.",
        "", "## Task-7 buffer by source task", "",
        "| Source task | Buffer slots | Classes without exemplar | Task-7 replay draws |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for row in final_buffer.itertuples():
        lines.append(f"| {row.source_task} | {row.buffer_slots} | "
                     f"{row.classes_without_exemplar}/{row.class_count} | "
                     f"{row.replay_draws_in_boundary} |")
    lines += ["", "Task-7 old/current metrics, per-class CSVs and five-figure sets "
              "for task 6/7 on validation and test are included beside this file.",
              "The pilot is not evidence of ensemble performance or of an isolated DER++ loss effect.", ""]
    (out / "PILOT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"kind": "p3_derpp_equal_class_member0_pilot",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": fold_file_sha256(ROOT / "configs/p3_derpp_equal_buffer_pilot.json"),
                "task_file_sha256": config["protocol"]["sha256"],
                "member_ids": [0], "completed_task_id": 7,
                "offline_ensemble": False,
                "test_macro_f1_task7": float(task7.test_macro_f1)}
    (root / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[RESULT] member0 task7 test Macro-F1={task7.test_macro_f1:.6f}; "
          f"task6→7={task7.test_macro_f1 - task6.test_macro_f1:+.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    for name in ("config", "train", "validation", "test", "feature_cols",
                 "fold_root", "preprocessing_root", "output_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    config, task_file = _check(args)
    if args.action == "check":
        print("[OK] Check complete; no training was started.", flush=True)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_command = [args.python, "-m", "src.training.derpp_fold",
                     "--config", str(args.config), "--task-file", str(task_file),
                     "--train", str(args.train), "--validation", str(args.validation),
                     "--test", str(args.test), "--feature-cols", str(args.feature_cols),
                     "--fold-assignments", str(args.fold_root / "fold_assignments.parquet"),
                     "--fold-manifest", str(args.fold_root / "fold_manifest.json"),
                     "--preprocessing-root", str(args.preprocessing_root),
                     "--output-root", str(args.output_root),
                     "--member-id", "0", "--memory-budget", "27778", "--device", args.device]
    _run(train_command, args.output_root / "logs" / "member0_training.log")
    for task_id in (6, 7):
        for split in ("validation", "test"):
            outdir = args.output_root / "visualizations_member0" / f"task_{task_id}_{split}"
            _run([args.python, str(ROOT / "scripts/visualize_cil_run.py"),
                  "--run-root", str(args.output_root), "--outdir", str(outdir),
                  "--task-id", str(task_id), "--member-id", "0", "--member-ids", "0",
                  "--split", split, "--task-file", str(task_file),
                  "--train", str(args.train), "--validation", str(args.validation),
                  "--test", str(args.test)], args.output_root / "logs" / "visualizations.log")
            for prefix in ("01_buffer_allocation", "02_replay_exposure", "03_classwise_performance",
                           "04_forgetting_heatmap", "05_confusion_matrix"):
                if not list(outdir.glob(f"{prefix}*.png")):
                    raise FileNotFoundError(f"Pilot visualization missing: {outdir}/{prefix}*.png")
            for prefix in ("buffer_allocation", "replay_exposure", "classwise_metrics",
                           "forgetting_f1_matrix", "top_confusions"):
                if not list(outdir.glob(f"{prefix}*.csv")):
                    raise FileNotFoundError(f"Pilot diagnostic table missing: {outdir}/{prefix}*.csv")
            print(f"[OK] Five member-0 visualizations: task={task_id}, split={split}", flush=True)
    _summarize(args.output_root, task_file, config)


if __name__ == "__main__":
    main()
