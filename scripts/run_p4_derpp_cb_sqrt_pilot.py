#!/usr/bin/env python3
"""Run and review a P4 DER++ equal-buffer, square-root-replay member-0 pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stratified_folds import fold_file_sha256  # noqa: E402
from src.methods.derpp import ClassBalancedTemperedReplayBuffer  # noqa: E402


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _check(args: argparse.Namespace) -> tuple[dict, Path]:
    required = [args.config, args.train, args.validation, args.test, args.feature_cols,
                args.fold_root / "fold_assignments.parquet",
                args.fold_root / "fold_manifest.json"]
    missing = [path for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError("Missing pilot input(s): " + ", ".join(map(str, missing)))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "configs/p4_derpp_cb_class_uniform_full8.json")
                          .read_text(encoding="utf-8"))
    if (config.get("kind") != "p4_derpp_cb_sqrt_replay_pilot"
            or config.get("member_ids") != [0]
            or config.get("protocol") != baseline["protocol"]
            or config.get("model") != baseline["model"]
            or config.get("training") != baseline["training"]
            or config.get("preprocessing") != baseline["preprocessing"]
            or config.get("evaluation") != baseline["evaluation"]
            or config.get("replay", {}).get("policy") != ClassBalancedTemperedReplayBuffer.policy
            or config["replay"].get("sampling") != ClassBalancedTemperedReplayBuffer.replay_policy
            or config["replay"].get("class_sampling_exponent") != 0.5
            or config["replay"].get("storage") != baseline["replay"].get("storage")
            or config["replay"].get("member_budget") != 27778
            or config["replay"].get("global_slot_budget") != 27778
            or config["replay"].get("draws_per_step") != 2
            or config["replay"].get("logit_refresh") is not False):
        raise ValueError("P4 sqrt pilot contract changed outside member scope and replay sampler.")
    task_file = ROOT / config["protocol"]["task_file"]
    if fold_file_sha256(task_file) != config["protocol"]["sha256"]:
        raise ValueError("Frozen P4 task-file SHA256 mismatch.")
    feature_columns = json.loads(args.feature_cols.read_text(encoding="utf-8"))
    if not isinstance(feature_columns, list) or len(feature_columns) != 3780:
        raise ValueError("P4 pilot requires exactly 3,780 frozen features.")
    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError("matplotlib is required for pilot visualizations.")
    print("[OK] P4 member 0 pilot; tasks 0-7; equal 27,778-slot class buffer.", flush=True)
    print("[OK] Only replay sampling is softened: P(class) proportional to sqrt(retained slots).",
          flush=True)
    print("[OK] DER++ one-pass loss remains CE + 0.3*MSE(logits) + 0.5*CE(replay).", flush=True)
    return config, task_file


def _prepare(args: argparse.Namespace, task_file: Path) -> None:
    target = args.preprocessing_root / "member_0/B/fold_preprocessing.json"
    if target.is_file() and target.stat().st_size:
        return
    if target.parent.exists() and any(target.parent.iterdir()):
        raise FileExistsError(f"Partial preprocessing namespace requires inspection: {target.parent}")
    _run([args.python, str(ROOT / "scripts/prepare_fold_preprocessing.py"),
          "--assignments", str(args.fold_root / "fold_assignments.parquet"),
          "--manifest", str(args.fold_root / "fold_manifest.json"),
          "--train", str(args.train), "--validation", str(args.validation),
          "--test", str(args.test), "--task-file", str(task_file),
          "--feature-cols", str(args.feature_cols), "--member-id", "0",
          "--validation-fold", "0", "--experiment-seed", "0", "--fold-seed", "42",
          "--policy", "task0_standard_frozen", "--batch-size", "2048",
          "--outdir", str(target.parent)], args.output_root / "logs/preprocessing.log")


def _summarize(root: Path, task_file: Path, config: dict) -> None:
    member_root = root / "member_0"
    summary = json.loads((member_root / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("completed_task_id") != 7 or len(summary.get("tasks", [])) != 8:
        raise ValueError("P4 sqrt pilot did not complete member 0 through task 7.")
    tasks = json.loads(task_file.read_text(encoding="utf-8"))["tasks"]
    origin = {int(raw): int(task["task_id"]) for task in tasks for raw in task["classes"]}
    output = root / "pilot_results"
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(member_root / "metrics.csv")
    metrics.to_csv(output / "member0_task_metrics.csv", index=False)

    source_rows: list[dict] = []
    exposure_rows: list[pd.DataFrame] = []
    for task in tasks:
        boundary = int(task["task_id"])
        task_root = member_root / f"task_{boundary}"
        audit = json.loads((task_root / "buffer_audit.json").read_text(encoding="utf-8"))
        allocation = {int(k): int(v) for k, v in audit["allocation"].items()}
        exposure = pd.read_csv(task_root / "replay_exposure.csv")
        exposure["boundary_task"] = boundary
        exposure["source_task"] = exposure.raw_class_id.astype(int).map(origin)
        retained = exposure.retained_exemplars.to_numpy(dtype=float)
        weights = np.sqrt(retained)
        exposure["expected_sqrt_probability"] = weights / weights.sum() if weights.sum() else 0.0
        exposure_rows.append(exposure)
        draws = dict(zip(exposure.raw_class_id.astype(int), exposure.total_replay_draws.astype(int)))
        for source in tasks[:boundary + 1]:
            raw_classes = [int(raw) for raw in source["classes"]]
            source_rows.append({
                "boundary_task": boundary, "source_task": int(source["task_id"]),
                "class_count": len(raw_classes),
                "buffer_slots": sum(allocation.get(raw, 0) for raw in raw_classes),
                "classes_without_exemplar": sum(allocation.get(raw, 0) == 0 for raw in raw_classes),
                "replay_draws_in_boundary": sum(draws.get(raw, 0) for raw in raw_classes),
            })
    pd.DataFrame(source_rows).to_csv(output / "buffer_replay_by_source_task.csv", index=False)
    pd.concat(exposure_rows, ignore_index=True).to_csv(
        output / "classwise_replay_exposure_all_boundaries.csv", index=False)
    pd.read_csv(member_root / "task_7/metrics.csv").to_csv(
        output / "task7_old_current_metrics.csv", index=False)

    task6 = metrics.loc[metrics.task_id == 6].iloc[0]
    task7 = metrics.loc[metrics.task_id == 7].iloc[0]
    final_source = pd.DataFrame(source_rows)
    final_source = final_source[final_source.boundary_task == 7]
    lines = [
        "# P4 DER++ square-root replay pilot (member 0)", "",
        "This is a one-member diagnostic pilot, not a three-member offline ensemble.",
        "P4, equal-class water-fill storage, 27,778 slots, model, optimizer and DER++ loss are frozen.",
        "The only scientific change from strict class-uniform replay is `P(class) ∝ sqrt(retained_count)`.",
        "", "| Task | Train rows | Validation Macro-F1 | Test Macro-F1 | Buffer |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples():
        lines.append(f"| {row.task_id} | {row.train_rows} | {row.validation_macro_f1:.6f} | "
                     f"{row.test_macro_f1:.6f} | {row.memory_after} |")
    lines.extend(["", f"Task 6→7 test Macro-F1: {task6.test_macro_f1:.6f} → "
                  f"{task7.test_macro_f1:.6f} "
                  f"({task7.test_macro_f1 - task6.test_macro_f1:+.6f}).",
                  "", "## Task-7 buffer and replay by source task", "",
                  "| Source task | Buffer slots | Missing classes | Replay draws |",
                  "| ---: | ---: | ---: | ---: |"])
    for row in final_source.itertuples():
        lines.append(f"| {row.source_task} | {row.buffer_slots} | "
                     f"{row.classes_without_exemplar}/{row.class_count} | "
                     f"{row.replay_draws_in_boundary} |")
    lines.extend(["", "Task-6/task-7 validation and test visualization sets, classwise metrics, "
                  "confusions, buffer allocation and replay exposure tables are included.", ""])
    (output / "PILOT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "kind": "p4_derpp_cb_sqrt_replay_member0_pilot",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": fold_file_sha256(ROOT / "configs/p4_derpp_cb_sqrt_replay_pilot.json"),
        "task_file_sha256": config["protocol"]["sha256"],
        "member_ids": [0], "completed_task_id": 7, "offline_ensemble": False,
        "class_sampling_exponent": 0.5,
        "test_macro_f1_task6": float(task6.test_macro_f1),
        "test_macro_f1_task7": float(task7.test_macro_f1),
        "task6_to_task7_delta": float(task7.test_macro_f1 - task6.test_macro_f1),
    }
    (root / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
    print(f"[RESULT] member0 task7 test Macro-F1={task7.test_macro_f1:.6f}; "
          f"task6→7={task7.test_macro_f1 - task6.test_macro_f1:+.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "summarize"))
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
        print("[OK] Check complete; no preprocessing or training started.", flush=True)
        return
    if args.action == "summarize":
        _summarize(args.output_root, task_file, config)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    _prepare(args, task_file)
    _run([args.python, "-m", "src.training.derpp_fold",
          "--config", str(args.config), "--task-file", str(task_file),
          "--train", str(args.train), "--validation", str(args.validation),
          "--test", str(args.test), "--feature-cols", str(args.feature_cols),
          "--fold-assignments", str(args.fold_root / "fold_assignments.parquet"),
          "--fold-manifest", str(args.fold_root / "fold_manifest.json"),
          "--preprocessing-root", str(args.preprocessing_root),
          "--output-root", str(args.output_root), "--member-id", "0",
          "--memory-budget", "27778", "--device", args.device],
         args.output_root / "logs/member0_training.log")
    for task_id in (6, 7):
        for split in ("validation", "test"):
            outdir = args.output_root / "visualizations_member0" / f"task_{task_id}_{split}"
            _run([args.python, str(ROOT / "scripts/visualize_cil_run.py"),
                  "--run-root", str(args.output_root), "--outdir", str(outdir),
                  "--task-id", str(task_id), "--member-id", "0", "--member-ids", "0",
                  "--split", split, "--task-file", str(task_file),
                  "--train", str(args.train), "--validation", str(args.validation),
                  "--test", str(args.test)], args.output_root / "logs/visualizations.log")
            for prefix in ("01_buffer_allocation", "02_replay_exposure",
                           "03_classwise_performance", "04_forgetting_heatmap",
                           "05_confusion_matrix"):
                if not list(outdir.glob(f"{prefix}*.png")):
                    raise FileNotFoundError(f"Missing visualization: {outdir}/{prefix}*.png")
            print(f"[OK] Five visualizations: task={task_id}, split={split}", flush=True)
    _summarize(args.output_root, task_file, config)


if __name__ == "__main__":
    main()
