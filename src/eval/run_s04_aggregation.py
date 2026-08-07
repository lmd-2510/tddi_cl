#!/usr/bin/env python3
"""Validate complete S04 runs and publish the O07 fixed-budget baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_cil import FIXED_BUDGET_METHOD  # noqa: E402


O07_COLUMNS = [
    "run_id",
    "seed",
    "method",
    "method_protocol",
    "num_tasks",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "task_forgetting",
    "class_forgetting",
    "zero_f1_classes",
    "total_memory_budget",
    "replay_draws_per_epoch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and validate S04 fixed-budget runs.")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--expected-tasks", type=int, default=8)
    parser.add_argument("--total-memory-budget", type=int, default=6800)
    parser.add_argument("--replay-draws-per-epoch", type=int, default=6800)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _require_finite(row: pd.Series, columns: list[str], path: Path) -> None:
    for column in columns:
        if not math.isfinite(float(row[column])):
            raise ValueError(f"Non-finite {column} in {path}")


def discover_runs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        raise FileNotFoundError(f"S04 runs root does not exist: {runs_root}")
    runs = sorted(
        path.parent
        for path in runs_root.glob("*/run_config.json")
        if path.parent.is_dir()
    )
    if not runs:
        raise FileNotFoundError(f"No S04 run_config.json files found below: {runs_root}")
    return runs


def _validate_completed(run_dir: Path) -> None:
    events = pd.read_csv(run_dir / "events.csv")
    if "run_completed" not in set(events["event_type"].astype(str)):
        raise ValueError(f"Run is not complete: {run_dir}")


def _validate_s02(run_dir: Path, run_id: str, seed: int, expected_tasks: int) -> None:
    manifest = _read_json(run_dir / "s02/manifest.json")
    if manifest.get("run_id") != run_id or int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"S02 manifest provenance mismatch: {run_dir}")
    if manifest.get("method") != FIXED_BUDGET_METHOD:
        raise ValueError(f"S02 manifest method mismatch: {run_dir}")
    exports = manifest.get("exports")
    if not isinstance(exports, list):
        raise ValueError(f"Invalid S02 exports: {run_dir}")
    keys = {(int(row["train_task"]), str(row["split"])) for row in exports}
    expected = {
        (task, split)
        for task in range(expected_tasks)
        for split in ("validation", "test")
    }
    if keys != expected or len(exports) != len(expected):
        raise ValueError(f"S02 manifest is incomplete or duplicated: {run_dir}")


def _validate_o06(
    run_dir: Path,
    *,
    run_id: str,
    expected_tasks: int,
    total_memory_budget: int,
    replay_draws_per_epoch: int,
) -> None:
    path = run_dir / "replay_budget_audit.csv"
    audit = pd.read_csv(path)
    keys = ["run_id", "task", "raw_class_id"]
    if audit.duplicated(keys).any() or set(audit["run_id"].astype(str)) != {run_id}:
        raise ValueError(f"O06 duplicate keys or run provenance mismatch: {path}")
    if sorted(audit["task"].astype(int).unique().tolist()) != list(range(expected_tasks)):
        raise ValueError(f"O06 task coverage mismatch: {path}")

    for task, group in audit.groupby("task", sort=True):
        epochs = set(group["epochs_trained"].astype(int))
        total_after = set(group["total_memory_after"].astype(int))
        if len(epochs) != 1 or len(total_after) != 1:
            raise ValueError(f"O06 inconsistent task totals: {path}, task={task}")
        feasible = min(total_memory_budget, int(group["available_samples"].sum()))
        if total_after != {feasible}:
            raise ValueError(f"O06 memory budget mismatch: {path}, task={task}")
        actual_replay = set(group["actual_replay_draws_per_epoch"].astype(int))
        expected_replay = 0 if int(task) == 0 else replay_draws_per_epoch
        if actual_replay != {expected_replay}:
            raise ValueError(f"O06 replay budget mismatch: {path}, task={task}")
        total_replay = int(group["replay_draws_total"].sum())
        if total_replay != expected_replay * next(iter(epochs)):
            raise ValueError(f"O06 replay count does not match sampler epochs: {path}, task={task}")


def aggregate_run(
    run_dir: Path,
    *,
    expected_tasks: int,
    total_memory_budget: int,
    replay_draws_per_epoch: int,
) -> dict[str, Any]:
    config = _read_json(run_dir / "run_config.json")
    arguments = config.get("arguments", {})
    resolved = config.get("resolved", {})
    seed = int(arguments.get("seed"))
    if arguments.get("method") != FIXED_BUDGET_METHOD:
        raise ValueError(f"Unexpected S04 method: {run_dir}")
    if int(resolved.get("num_tasks", -1)) != expected_tasks:
        raise ValueError(f"Unexpected task count: {run_dir}")
    if int(resolved.get("total_memory_budget", -1)) != total_memory_budget:
        raise ValueError(f"Unexpected memory budget: {run_dir}")
    if int(resolved.get("replay_draws_per_epoch", -1)) != replay_draws_per_epoch:
        raise ValueError(f"Unexpected replay budget: {run_dir}")

    run_id = str(config["run_id"])
    _validate_completed(run_dir)
    _validate_s02(run_dir, run_id, seed, expected_tasks)
    _validate_o06(
        run_dir,
        run_id=run_id,
        expected_tasks=expected_tasks,
        total_memory_budget=total_memory_budget,
        replay_draws_per_epoch=replay_draws_per_epoch,
    )

    metrics_path = run_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    seen = metrics[metrics["eval_task_id"].astype(str) == "seen_all"].copy()
    seen["train_task_id"] = seen["train_task_id"].astype(int)
    if sorted(seen["train_task_id"].tolist()) != list(range(expected_tasks)):
        raise ValueError(f"Final seen-metric task coverage mismatch: {metrics_path}")
    final_metrics = seen.sort_values("train_task_id").iloc[-1]
    metric_columns = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]
    _require_finite(final_metrics, metric_columns, metrics_path)

    forgetting_path = run_dir / "forgetting.csv"
    forgetting = pd.read_csv(forgetting_path)
    task_forgetting_rows = forgetting[
        forgetting["task_id"].astype(str) == "mean_old_tasks"
    ]
    if len(task_forgetting_rows) != 1:
        raise ValueError(f"Missing task-level forgetting summary: {forgetting_path}")
    task_forgetting = float(task_forgetting_rows.iloc[0]["forgetting"])

    class_path = run_dir / "class_forgetting.csv"
    class_forgetting = pd.read_csv(class_path)
    final_classes = class_forgetting[
        class_forgetting["train_task"].astype(int) == expected_tasks - 1
    ]
    if final_classes["class_id"].duplicated().any() or final_classes.empty:
        raise ValueError(f"Invalid final class-forgetting rows: {class_path}")
    class_forgetting_mean = float(final_classes["forgetting"].mean())
    zero_f1_classes = int((final_classes["current_f1"] == 0.0).sum())
    if not math.isfinite(task_forgetting) or not math.isfinite(class_forgetting_mean):
        raise ValueError(f"Non-finite forgetting metric: {run_dir}")

    return {
        "run_id": run_id,
        "seed": seed,
        "method": FIXED_BUDGET_METHOD,
        "method_protocol": str(resolved["method_protocol"]),
        "num_tasks": expected_tasks,
        **{column: float(final_metrics[column]) for column in metric_columns},
        "task_forgetting": task_forgetting,
        "class_forgetting": class_forgetting_mean,
        "zero_f1_classes": zero_f1_classes,
        "total_memory_budget": total_memory_budget,
        "replay_draws_per_epoch": replay_draws_per_epoch,
    }


def run_aggregation(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.expected_tasks <= 0 or args.total_memory_budget <= 0 or args.replay_draws_per_epoch <= 0:
        raise ValueError("Expected tasks and S04 budgets must be positive.")
    expected_seeds = sorted(set(args.expected_seeds))
    if len(expected_seeds) != len(args.expected_seeds):
        raise ValueError("--expected-seeds must not contain duplicates.")

    rows = [
        aggregate_run(
            run_dir,
            expected_tasks=args.expected_tasks,
            total_memory_budget=args.total_memory_budget,
            replay_draws_per_epoch=args.replay_draws_per_epoch,
        )
        for run_dir in discover_runs(args.runs_root)
    ]
    frame = pd.DataFrame(rows, columns=O07_COLUMNS).sort_values("seed", ignore_index=True)
    if frame["seed"].duplicated().any():
        raise ValueError("More than one S04 run was found for a seed.")
    actual_seeds = frame["seed"].astype(int).tolist()
    if args.allow_partial:
        if not set(actual_seeds).issubset(expected_seeds):
            raise ValueError(f"Unexpected S04 seeds: {actual_seeds}")
    elif actual_seeds != expected_seeds:
        raise ValueError(f"Expected S04 seeds {expected_seeds}, found {actual_seeds}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    output_path = args.outdir / "fixed_budget_baseline.csv"
    manifest_path = args.outdir / "manifest.json"
    if not args.overwrite and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"S04 aggregate output already exists: {args.outdir}")
    frame.to_csv(output_path, index=False)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": FIXED_BUDGET_METHOD,
        "expected_seeds": expected_seeds,
        "completed_seeds": actual_seeds,
        "expected_tasks": args.expected_tasks,
        "total_memory_budget": args.total_memory_budget,
        "replay_draws_per_epoch": args.replay_draws_per_epoch,
        "allow_partial": bool(args.allow_partial),
        "runs_root": str(args.runs_root.resolve()),
        "output": output_path.name,
        "output_sha256": _sha256_file(output_path),
        "implementation_sha256": _sha256_file(Path(__file__)),
        "source_run_ids": frame["run_id"].tolist(),
    }
    try:
        manifest["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path, manifest_path


def main() -> None:
    output_path, _ = run_aggregation(parse_args())
    print(f"S04 aggregation completed: {output_path}")


if __name__ == "__main__":
    main()
