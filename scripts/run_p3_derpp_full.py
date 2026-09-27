#!/usr/bin/env python3
"""Run three P3 DER++ members, offline ensemble, thresholds and diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stratified_folds import fold_file_sha256  # noqa: E402
from src.eval.ensemble_ue import (  # noqa: E402
    aggregate_member_predictions, export_offline_ensemble_artifact,
    load_offline_ensemble_artifact,
)
from src.eval.metrics import compute_classification_metrics  # noqa: E402
from src.eval.predictions import load_member_prediction_artifact  # noqa: E402


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def _check(args) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config.get("kind") != "p3_derpp_online_full8"
            or config.get("member_ids") != [0, 1, 2]
            or config.get("experiment_seed") != 0 or config.get("fold_seed") != 42
            or config.get("preprocessing") != "task0_standard_frozen"):
        raise ValueError("DER++ config does not match the approved P3 three-member study.")
    if (config["replay"]["member_budget"] != 27778
            or config["replay"]["global_slot_budget"] != 83334
            or config["training"]["epochs_per_task"] != 1
            or config["training"]["classification_loss"] != "cross_entropy"
            or config["model"]["head"] != "fixed_178_from_task0"):
        raise ValueError("DER++ budget, one-pass CE loss or fixed-head contract changed.")
    task_file = ROOT / config["protocol"]["task_file"]
    if fold_file_sha256(task_file) != config["protocol"]["sha256"]:
        raise ValueError("P3 task-file hash mismatch.")
    required = [args.train, args.validation, args.test, args.feature_cols,
                args.fold_root / "fold_assignments.parquet", args.fold_root / "fold_manifest.json",
                ROOT / config["evaluation"]["threshold_config"]]
    required += [args.preprocessing_root / f"member_{member}" / "B" / "fold_preprocessing.json"
                 for member in range(3)]
    missing = [path for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError("Missing DER++ input(s): " + ", ".join(map(str, missing)))
    print("[OK] P3 frozen schedule, 3 members, 4% per member, preprocessing and sources found.", flush=True)
    print("[OK] DER++: online one pass/task, fixed 178-way head, CE + alpha*MSE + beta*CE, reservoir.", flush=True)
    return config


def _member_command(args, member: int) -> list[str]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    return [args.python, "-m", "src.training.derpp_fold",
            "--config", str(args.config),
            "--task-file", str(ROOT / config["protocol"]["task_file"]),
            "--train", str(args.train), "--validation", str(args.validation),
            "--test", str(args.test), "--feature-cols", str(args.feature_cols),
            "--fold-assignments", str(args.fold_root / "fold_assignments.parquet"),
            "--fold-manifest", str(args.fold_root / "fold_manifest.json"),
            "--preprocessing-root", str(args.preprocessing_root),
            "--output-root", str(args.output_root), "--member-id", str(member),
            "--memory-budget", str(config["replay"]["member_budget"]),
            "--device", args.device]


def _evaluate(args, config: dict) -> None:
    root = args.output_root
    eval_root = root / "offline_evaluation"
    records = []
    for task_id in range(8):
        task_dir = eval_root / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        for split, name in (("validation", "oof.npz"), ("test", "test.npz")):
            target = task_dir / name
            if not target.is_file():
                sources = [root / f"member_{member}" / "member_predictions" /
                           f"task_{task_id}" / f"{split}.npz" for member in range(3)]
                artifacts = [load_member_prediction_artifact(path) for path in sources]
                ensemble = aggregate_member_predictions(
                    artifacts, source_artifact_paths=sources, ensemble_mode="stratified_3fold")
                export_offline_ensemble_artifact(ensemble, target)
            artifact = load_offline_ensemble_artifact(target)
            if split == "test":
                metrics = compute_classification_metrics(
                    artifact.labels, artifact.predictions,
                    labels=artifact.raw_class_ids.astype(int).tolist())
                records.append({"task_id": task_id, "test_samples": artifact.row_count,
                                "seen_classes": artifact.class_count, **metrics})
        threshold = ROOT / config["evaluation"]["threshold_config"]
        frozen = task_dir / "frozen_threshold.json"
        oof_report = task_dir / "oof_threshold_report.json"
        test_report = task_dir / "test_threshold_report.json"
        if not frozen.is_file():
            _run([args.python, str(ROOT / "src/eval/threshold.py"), "select",
                  "--ensemble", str(task_dir / "oof.npz"), "--config", str(threshold),
                  "--threshold-out", str(frozen), "--report-out", str(oof_report)],
                 root / "logs" / f"threshold_task{task_id}.log")
        if not oof_report.is_file():
            _run([args.python, str(ROOT / "src/eval/threshold.py"), "evaluate",
                  "--ensemble", str(task_dir / "oof.npz"), "--config", str(threshold),
                  "--threshold-artifact", str(frozen), "--report-out", str(oof_report)],
                 root / "logs" / f"threshold_task{task_id}.log")
        if not test_report.is_file():
            _run([args.python, str(ROOT / "src/eval/threshold.py"), "evaluate",
                  "--ensemble", str(task_dir / "test.npz"), "--config", str(threshold),
                  "--threshold-artifact", str(frozen), "--report-out", str(test_report)],
                 root / "logs" / f"threshold_task{task_id}.log")
        print(f"[OK] Offline OOF/test + frozen threshold task={task_id}", flush=True)
    final_results = root / "final_results"
    final_results.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(final_results / "derpp_task_metrics.csv", index=False)
    if not (final_results / "ensemble3_p3_task_summary.csv").exists():
        _run([args.python, str(ROOT / "src/eval/report.py"),
              "--full-root", str(root), "--task-file", str(ROOT / config["protocol"]["task_file"]),
              "--outdir", str(final_results)], root / "logs" / "final_report.log")
    if not (final_results / "boundary_task6_task7").is_dir():
        _run([args.python, str(ROOT / "scripts/analyze_p3_task6_task7.py"),
              "--full-root", str(root), "--task-file", str(ROOT / config["protocol"]["task_file"]),
              "--outdir", str(final_results / "boundary_task6_task7")],
             root / "logs" / "task6_task7_analysis.log")
    for member in range(3):
        for task_id in (6, 7):
            for split in ("validation", "test"):
                outdir = root / "diagnostics" / f"member_{member}" / f"task_{task_id}_{split}"
                if not (outdir / f"05_confusion_matrix_task{task_id}_member{member}.png").is_file():
                    _run([args.python, str(ROOT / "scripts/visualize_cil_run.py"),
                          "--run-root", str(root), "--outdir", str(outdir),
                          "--task-id", str(task_id), "--member-id", str(member),
                          "--member-ids", str(member), "--split", split,
                          "--task-file", str(ROOT / config["protocol"]["task_file"]),
                          "--train", str(args.train), "--validation", str(args.validation),
                          "--test", str(args.test)], root / "logs" / "visualizations.log")
    for task_id in (6, 7):
        for split in ("validation", "test"):
            outdir = root / "diagnostics" / "offline_ensemble" / f"task_{task_id}_{split}"
            if not (outdir / f"05_confusion_matrix_task{task_id}_ensemble.png").is_file():
                _run([args.python, str(ROOT / "scripts/visualize_cil_run.py"),
                      "--run-root", str(root), "--outdir", str(outdir),
                      "--task-id", str(task_id), "--member-id", "0",
                      "--member-ids", "0", "1", "2", "--ensemble", "--split", split,
                      "--task-file", str(ROOT / config["protocol"]["task_file"]),
                      "--train", str(args.train), "--validation", str(args.validation),
                      "--test", str(args.test)], root / "logs" / "visualizations.log")
    manifest = {
        "kind": "p3_derpp_online_full8", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config), "config_sha256": fold_file_sha256(args.config),
        "task_file_sha256": config["protocol"]["sha256"],
        "members": [{"member_id": member,
                     "run_config": str(root / f"member_{member}" / "run_config.json"),
                     "run_config_sha256": fold_file_sha256(root / f"member_{member}" / "run_config.json")}
                    for member in range(3)],
        "offline_evaluation": [str(eval_root / f"task_{task_id}" / "test.npz")
                               for task_id in range(8)],
    }
    (root / "full_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    task7 = records[-1]
    lines = [
        "# P3 DER++ full run",
        "",
        "Three P3 members, frozen 3-fold assignment, 4% reservoir per member (27,778 samples).",
        "Fixed 178-class head. One online pass per task; CE(current) + "
        f"{config['training']['alpha']} × MSE(stored logits) + "
        f"{config['training']['beta']} × CE(replay labels).",
        "Two independent uniform reservoir draws per update. Buffer stores insertion-time logits; no refresh.",
        "Evaluation masks future class columns at each task boundary. Test is report-only.",
        "",
        "| Task | Seen classes | Test samples | Accuracy | Macro-F1 | Balanced accuracy | Weighted-F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            f"| {row['task_id']} | {row['seen_classes']} | {row['test_samples']} | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['weighted_f1']:.4f} |"
        )
    lines += [
        "", "The standard P3 report, OOF threshold metrics, member trajectories, "
        "task 6→7 diagnostics and figures are included beside this file.",
        "", "Caveat: original DER++ online one-pass training differs from the historical "
        "30-epoch hybrid run in both method and training exposure; this is a method comparison, "
        "not an ablation of the DER++ loss alone.", "",
    ]
    (final_results / "DERPP_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[RESULT] task7 test macro_f1={task7['macro_f1']:.6f} "
          f"balanced_accuracy={task7['balanced_accuracy']:.6f} accuracy={task7['accuracy']:.6f}", flush=True)


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
    config = _check(args)
    if args.action == "check":
        print("[OK] Check complete; no model created.", flush=True)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    for member in range(3):
        print(f"[RUN] DER++ P3 member={member}", flush=True)
        _run(_member_command(args, member), args.output_root / "logs" / f"member_{member}.log")
    _evaluate(args, config)


if __name__ == "__main__":
    main()
