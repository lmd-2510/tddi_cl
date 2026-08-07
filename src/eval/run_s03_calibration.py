#!/usr/bin/env python3
"""Run post-hoc calibration over complete S02 prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.calibration_metrics import (  # noqa: E402
    CalibrationMetrics,
    TemperatureFitResult,
    compute_calibration_metrics,
    fit_temperature,
    map_raw_labels_to_indices,
)


S03_SCHEMA_VERSION = 1
OUTPUT_COLUMNS = [
    "run_id",
    "seed",
    "method",
    "method_protocol",
    "train_task",
    "evaluation_split",
    "calibration_stage",
    "temperature_fit_split",
    "temperature",
    "fit_iterations",
    "fit_converged",
    "fit_boundary",
    "num_samples",
    "num_classes",
    "num_bins",
    "high_confidence_threshold",
    "accuracy",
    "ece",
    "brier_score",
    "negative_log_likelihood",
    "mean_confidence",
    "high_confidence_error_rate",
    "high_confidence_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S03 temperature-scaling calibration.")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--min-temperature", type=float, default=0.05)
    parser.add_argument("--max-temperature", type=float, default=100.0)
    parser.add_argument("--max-fit-iterations", type=int, default=20)
    parser.add_argument("--fit-tolerance", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def discover_s02_manifests(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        raise FileNotFoundError(f"S02 runs root does not exist: {runs_root}")
    manifests = sorted(runs_root.glob("*/s02/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No S02 manifests found below: {runs_root}")
    return manifests


def validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    manifest = _load_json(manifest_path)
    required = {"schema_version", "run_id", "seed", "method", "method_protocol", "exports"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"S02 manifest is missing keys {missing}: {manifest_path}")
    if manifest["schema_version"] != 1 or not isinstance(manifest["exports"], list):
        raise ValueError(f"Unsupported S02 manifest: {manifest_path}")
    export_map: dict[tuple[int, str], dict[str, Any]] = {}
    for export in manifest["exports"]:
        key = (int(export["train_task"]), str(export["split"]))
        if key in export_map:
            raise ValueError(f"Duplicate S02 export {key}: {manifest_path}")
        if key[1] not in {"validation", "test"}:
            raise ValueError(f"Unsupported S02 split {key[1]}: {manifest_path}")
        export_map[key] = export
    tasks = sorted({task for task, _ in export_map})
    if tasks != list(range(len(tasks))):
        raise ValueError(f"S02 tasks must be contiguous from zero: {manifest_path}")
    expected_keys = {(task, split) for task in tasks for split in ("validation", "test")}
    if set(export_map) != expected_keys:
        raise ValueError(f"S02 manifest must contain validation and test for every task: {manifest_path}")
    return manifest, export_map


def _artifact_path(manifest_path: Path, relative_path: str, declared_bytes: int) -> Path:
    s02_root = manifest_path.parent.resolve()
    path = (s02_root / relative_path).resolve()
    if path != s02_root and s02_root not in path.parents:
        raise ValueError(f"Artifact escapes its S02 root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing S02 prediction artifact: {path}")
    if path.stat().st_size != int(declared_bytes):
        raise ValueError(f"S02 prediction size does not match manifest: {path}")
    return path


def load_prediction_shard(
    manifest_path: Path,
    manifest: dict[str, Any],
    export: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    path = _artifact_path(
        manifest_path,
        str(export["predictions"]),
        int(export["predictions_bytes"]),
    )
    parquet = pq.ParquetFile(path, memory_map=True)
    schema = parquet.schema_arrow
    required_fields = {"run_id", "seed", "method", "method_protocol", "train_task", "split", "label", "class_ids", "logits"}
    if not required_fields.issubset(schema.names):
        raise ValueError(f"S02 predictions schema is incomplete: {path}")
    logits_type = schema.field("logits").type
    if not pa.types.is_fixed_size_list(logits_type) or logits_type.value_type != pa.float32():
        raise ValueError(f"S02 logits must be fixed-size float32 lists: {path}")
    class_ids = np.asarray(export["class_ids"], dtype=np.int64)
    if logits_type.list_size != class_ids.shape[0] or int(export["class_count"]) != class_ids.shape[0]:
        raise ValueError(f"S02 class width does not match manifest: {path}")
    if parquet.metadata.num_rows != int(export["row_count"]):
        raise ValueError(f"S02 row count does not match manifest: {path}")

    first = next(
        parquet.iter_batches(
            batch_size=1,
            columns=["run_id", "seed", "method", "method_protocol", "train_task", "split", "class_ids"],
        )
    )
    actual_context = (
        first.column("run_id")[0].as_py(),
        first.column("seed")[0].as_py(),
        first.column("method")[0].as_py(),
        first.column("method_protocol")[0].as_py(),
        first.column("train_task")[0].as_py(),
        first.column("split")[0].as_py(),
    )
    expected_context = (
        manifest["run_id"],
        int(manifest["seed"]),
        manifest["method"],
        manifest["method_protocol"],
        int(export["train_task"]),
        export["split"],
    )
    if actual_context != expected_context:
        raise ValueError(f"S02 Parquet provenance does not match manifest: {path}")
    parquet_class_ids = np.asarray(first.column("class_ids")[0].as_py(), dtype=np.int64)
    if not np.array_equal(parquet_class_ids, class_ids):
        raise ValueError(f"S02 Parquet class IDs do not match manifest: {path}")

    table = pq.read_table(path, columns=["label", "logits"], memory_map=True)
    labels = table["label"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    logits_array = table["logits"].combine_chunks()
    logits = logits_array.values.to_numpy(zero_copy_only=False).reshape(
        labels.shape[0], class_ids.shape[0]
    )
    if logits.dtype != np.float32:
        logits = logits.astype(np.float32, copy=False)
    dense_labels = map_raw_labels_to_indices(labels, class_ids)
    return logits, dense_labels


def _metric_row(
    manifest: dict[str, Any],
    task: int,
    split: str,
    stage: str,
    temperature: float,
    fit: TemperatureFitResult,
    metrics: CalibrationMetrics,
    *,
    num_bins: int,
    high_confidence_threshold: float,
) -> dict[str, Any]:
    row = {
        "run_id": manifest["run_id"],
        "seed": int(manifest["seed"]),
        "method": manifest["method"],
        "method_protocol": manifest["method_protocol"],
        "train_task": task,
        "evaluation_split": split,
        "calibration_stage": stage,
        "temperature_fit_split": "validation",
        "temperature": temperature,
        "fit_iterations": fit.iterations,
        "fit_converged": fit.converged,
        "fit_boundary": fit.boundary,
        "num_bins": num_bins,
        "high_confidence_threshold": high_confidence_threshold,
        **asdict(metrics),
    }
    return {column: row[column] for column in OUTPUT_COLUMNS}


def calibrate_task(
    manifest_path: Path,
    manifest: dict[str, Any],
    export_map: dict[tuple[int, str], dict[str, Any]],
    task: int,
    *,
    num_bins: int,
    high_confidence_threshold: float,
    batch_size: int,
    min_temperature: float,
    max_temperature: float,
    max_fit_iterations: int,
    fit_tolerance: float,
) -> list[dict[str, Any]]:
    validation_logits, validation_labels = load_prediction_shard(
        manifest_path, manifest, export_map[(task, "validation")]
    )
    fit = fit_temperature(
        validation_logits,
        validation_labels,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
        max_iterations=max_fit_iterations,
        tolerance=fit_tolerance,
        batch_size=batch_size,
    )
    rows: list[dict[str, Any]] = []
    validation_raw = compute_calibration_metrics(
        validation_logits,
        validation_labels,
        temperature=1.0,
        num_bins=num_bins,
        high_confidence_threshold=high_confidence_threshold,
        batch_size=batch_size,
    )
    validation_scaled = compute_calibration_metrics(
        validation_logits,
        validation_labels,
        temperature=fit.temperature,
        num_bins=num_bins,
        high_confidence_threshold=high_confidence_threshold,
        batch_size=batch_size,
    )
    rows.extend(
        [
            _metric_row(
                manifest, task, "validation", "raw", 1.0, fit, validation_raw,
                num_bins=num_bins, high_confidence_threshold=high_confidence_threshold,
            ),
            _metric_row(
                manifest, task, "validation", "temperature_scaled", fit.temperature, fit,
                validation_scaled, num_bins=num_bins,
                high_confidence_threshold=high_confidence_threshold,
            ),
        ]
    )
    del validation_logits, validation_labels

    test_logits, test_labels = load_prediction_shard(
        manifest_path, manifest, export_map[(task, "test")]
    )
    test_raw = compute_calibration_metrics(
        test_logits,
        test_labels,
        temperature=1.0,
        num_bins=num_bins,
        high_confidence_threshold=high_confidence_threshold,
        batch_size=batch_size,
    )
    test_scaled = compute_calibration_metrics(
        test_logits,
        test_labels,
        temperature=fit.temperature,
        num_bins=num_bins,
        high_confidence_threshold=high_confidence_threshold,
        batch_size=batch_size,
    )
    rows.extend(
        [
            _metric_row(
                manifest, task, "test", "raw", 1.0, fit, test_raw,
                num_bins=num_bins, high_confidence_threshold=high_confidence_threshold,
            ),
            _metric_row(
                manifest, task, "test", "temperature_scaled", fit.temperature, fit,
                test_scaled, num_bins=num_bins,
                high_confidence_threshold=high_confidence_threshold,
            ),
        ]
    )
    for raw, scaled in ((validation_raw, validation_scaled), (test_raw, test_scaled)):
        if not math.isclose(raw.accuracy, scaled.accuracy, rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError("Positive temperature scaling changed argmax accuracy.")
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]], *, append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def run(args: argparse.Namespace) -> Path:
    if args.num_bins <= 0 or args.batch_size <= 0:
        raise ValueError("num-bins and batch-size must be positive.")
    output_csv = args.outdir / "calibration_by_task.csv"
    output_manifest = args.outdir / "manifest.json"
    output_log = args.outdir / "run.log"
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"S03 output directory is not empty: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in (output_csv, output_manifest, output_log):
            path.unlink(missing_ok=True)

    manifests = discover_s02_manifests(args.runs_root)
    source_inventory: list[dict[str, Any]] = []
    wrote_rows = False
    total_rows = 0
    with output_log.open("a", encoding="utf-8") as log_handle:
        for run_index, manifest_path in enumerate(manifests, start=1):
            manifest, export_map = validate_manifest(manifest_path)
            source_inventory.append(
                {
                    "path": str(manifest_path),
                    "sha256": _sha256_file(manifest_path),
                    "run_id": manifest["run_id"],
                    "seed": int(manifest["seed"]),
                    "method": manifest["method"],
                }
            )
            tasks = sorted({task for task, _ in export_map})
            for task in tasks:
                message = (
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"run={run_index}/{len(manifests)} method={manifest['method']} "
                    f"seed={manifest['seed']} task={task}/{tasks[-1]}"
                )
                print(message, flush=True)
                log_handle.write(message + "\n")
                log_handle.flush()
                rows = calibrate_task(
                    manifest_path,
                    manifest,
                    export_map,
                    task,
                    num_bins=args.num_bins,
                    high_confidence_threshold=args.high_confidence_threshold,
                    batch_size=args.batch_size,
                    min_temperature=args.min_temperature,
                    max_temperature=args.max_temperature,
                    max_fit_iterations=args.max_fit_iterations,
                    fit_tolerance=args.fit_tolerance,
                )
                _write_rows(output_csv, rows, append=wrote_rows)
                wrote_rows = True
                total_rows += len(rows)

    manifest_payload = {
        "schema_version": S03_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_runs_root": str(args.runs_root),
        "source_s02_schema_version": 1,
        "temperature_fit_split": "validation",
        "test_usage": "reporting_only",
        "undefined_metric_encoding": "empty_csv_field_with_zero_support_count",
        "num_bins": args.num_bins,
        "high_confidence_threshold": args.high_confidence_threshold,
        "temperature_bounds": [args.min_temperature, args.max_temperature],
        "max_fit_iterations": args.max_fit_iterations,
        "fit_tolerance": args.fit_tolerance,
        "run_count": len(manifests),
        "row_count": total_rows,
        "output": output_csv.name,
        "output_sha256": _sha256_file(output_csv),
        "implementation": {
            "git": _git_state(),
            "calibration_metrics_sha256": _sha256_file(PROJECT_ROOT / "src/eval/calibration_metrics.py"),
            "runner_sha256": _sha256_file(PROJECT_ROOT / "src/eval/run_s03_calibration.py"),
        },
        "source_manifests": source_inventory,
    }
    output_manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"S03 completed: rows={total_rows} output={output_csv}", flush=True)
    return output_csv


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
