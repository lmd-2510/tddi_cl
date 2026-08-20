#!/usr/bin/env python3
"""Build P5-P8 CIL schedules from train/validation reference signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_cil_tasks import (
    FREQUENCY_BINS,
    frequency_bin,
    load_counts,
    proportional_bin_quotas,
    split_order_into_tasks,
    summarize_tasks,
    task_sizes,
    validate_task_sizes,
    validate_tasks,
    write_task_json,
)


PROTOCOLS = {
    "P5": "multi_factor_balanced",
    "P6": "difficulty_balanced",
    "P7": "confusion_spread",
    "P8": "controlled_rarity_drift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-counts", required=True, type=Path)
    parser.add_argument("--class-stats", required=True, type=Path)
    parser.add_argument("--difficulty", required=True, type=Path)
    parser.add_argument("--confusion-edges", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--protocol", choices=[*PROTOCOLS, "all"], default="all")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-classes", type=int, default=178)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--base-task-classes", type=int, default=38)
    parser.add_argument("--increment-classes", type=int, default=20)
    return parser.parse_args()


def relative_mse(values: np.ndarray, targets: np.ndarray) -> float:
    scale = np.maximum(np.abs(targets), 1e-12)
    return float(np.mean(((values - targets) / scale) ** 2))


def controlled_drift_quotas(sizes: list[int], bin_counts: dict[str, int]) -> list[dict[str, int]]:
    if sizes != [38] + [20] * 7 or bin_counts != {
        "ultra_tail": 45,
        "tail": 39,
        "medium": 55,
        "head": 39,
    }:
        raise ValueError("P8 controlled drift quotas are locked to the audited 178-class layout.")
    ultra = [8, 4, 4, 5, 5, 6, 6, 7]
    tail = [8, 4, 5, 5, 5, 4, 4, 4]
    medium = [14, 6, 6, 5, 6, 6, 6, 6]
    head = [8, 6, 5, 5, 4, 4, 4, 3]
    return [
        {
            "ultra_tail": ultra[i],
            "tail": tail[i],
            "medium": medium[i],
            "head": head[i],
        }
        for i in range(8)
    ]


def build_edge_lookup(edges: pd.DataFrame) -> tuple[dict[tuple[int, int], float], float]:
    lookup: dict[tuple[int, int], float] = {}
    for row in edges.itertuples(index=False):
        pair = tuple(sorted((int(row.class_id_a), int(row.class_id_b))))
        lookup[pair] = float(row.symmetric_confusion_rate)
    return lookup, max(sum(lookup.values()), 1e-12)


def assign_initial(
    frame: pd.DataFrame,
    sizes: list[int],
    quotas: list[dict[str, int]],
    *,
    seed: int,
    edge_lookup: dict[tuple[int, int], float] | None = None,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    used = [{name: 0 for name in FREQUENCY_BINS} for _ in sizes]
    buckets: list[list[int]] = [[] for _ in sizes]
    mass = np.zeros(len(sizes), dtype=np.float64)
    target_mass = float(frame["count"].sum()) / len(sizes)
    task_rank = {task: rank for rank, task in enumerate(rng.permutation(len(sizes)))}
    ordered = frame.assign(_tie=rng.random(len(frame))).sort_values(
        ["count", "_tie", "class_id"], ascending=[False, True, True]
    )
    for row in ordered.itertuples(index=False):
        class_id = int(row.class_id)
        name = str(row.frequency_bin)
        candidates = [
            task
            for task in range(len(sizes))
            if used[task][name] < quotas[task][name]
        ]
        if not candidates:
            raise RuntimeError(f"No quota remains for {name}.")

        def candidate_score(task: int) -> tuple[float, int]:
            score = (mass[task] + float(row.count)) / target_mass
            if edge_lookup:
                score += 4.0 * sum(
                    edge_lookup.get(tuple(sorted((class_id, other))), 0.0)
                    for other in buckets[task]
                )
            return score, task_rank[task]

        task = min(candidates, key=candidate_score)
        buckets[task].append(class_id)
        mass[task] += float(row.count)
        used[task][name] += 1
    return buckets


def optimize_by_swaps(
    buckets: list[list[int]],
    frame: pd.DataFrame,
    objective: Callable[[list[list[int]]], float],
    *,
    seed: int,
    iterations: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed + 100_003)
    bin_by_class = frame.set_index("class_id")["frequency_bin"].to_dict()
    current = objective(buckets)
    for _ in range(iterations):
        name = str(rng.choice(FREQUENCY_BINS))
        eligible = [
            task
            for task, bucket in enumerate(buckets)
            if any(bin_by_class[class_id] == name for class_id in bucket)
        ]
        if len(eligible) < 2:
            continue
        left, right = rng.choice(eligible, size=2, replace=False).tolist()
        left_positions = [i for i, c in enumerate(buckets[left]) if bin_by_class[c] == name]
        right_positions = [i for i, c in enumerate(buckets[right]) if bin_by_class[c] == name]
        li = int(rng.choice(left_positions))
        ri = int(rng.choice(right_positions))
        if buckets[left][li] == buckets[right][ri]:
            continue
        buckets[left][li], buckets[right][ri] = buckets[right][ri], buckets[left][li]
        candidate = objective(buckets)
        if candidate + 1e-12 < current:
            current = candidate
        else:
            buckets[left][li], buckets[right][ri] = buckets[right][ri], buckets[left][li]
    return buckets


def build_objective(
    protocol: str,
    frame: pd.DataFrame,
    sizes: list[int],
    edge_lookup: dict[tuple[int, int], float],
    total_edge_weight: float,
) -> Callable[[list[list[int]]], float]:
    data = frame.set_index("class_id")
    equal_target_columns = ["count"]
    proportional_columns: list[str] = []
    if protocol == "P5":
        equal_target_columns += ["effective_support_sqrt", "unique_drugs"]
        proportional_columns += ["descriptor_diversity"]
    elif protocol == "P6":
        proportional_columns += ["difficulty", "validation_mean_nll"]

    def objective(buckets: list[list[int]]) -> float:
        score = 0.0
        for column in equal_target_columns:
            values = np.asarray(
                [float(data.loc[bucket, column].sum()) for bucket in buckets]
            )
            targets = np.full(len(buckets), float(data[column].sum()) / len(buckets))
            score += relative_mse(values, targets)
        for column in proportional_columns:
            values = np.asarray(
                [float(data.loc[bucket, column].sum()) for bucket in buckets]
            )
            targets = float(data[column].sum()) * np.asarray(sizes) / sum(sizes)
            score += relative_mse(values, targets)
        if protocol == "P7":
            within = 0.0
            for bucket in buckets:
                for i, left in enumerate(bucket):
                    for right in bucket[i + 1 :]:
                        within += edge_lookup.get(tuple(sorted((left, right))), 0.0)
            score += 8.0 * within / total_edge_weight
        return score

    return objective


def buckets_to_tasks(buckets: list[list[int]]) -> list[dict[str, object]]:
    return [
        {"task_id": task, "classes": classes, "num_classes": len(classes)}
        for task, classes in enumerate(buckets)
    ]


def audit_schedule(
    protocol: str,
    seed: int,
    buckets: list[list[int]],
    frame: pd.DataFrame,
    edge_lookup: dict[tuple[int, int], float],
) -> list[dict[str, object]]:
    data = frame.set_index("class_id")
    rows = []
    for task, bucket in enumerate(buckets):
        within = 0.0
        for i, left in enumerate(bucket):
            for right in bucket[i + 1 :]:
                within += edge_lookup.get(tuple(sorted((left, right))), 0.0)
        row: dict[str, object] = {
            "protocol": protocol,
            "seed": seed,
            "task_id": task,
            "num_classes": len(bucket),
            "train_samples": int(data.loc[bucket, "count"].sum()),
            "effective_support_sqrt": float(data.loc[bucket, "effective_support_sqrt"].sum()),
            "unique_drugs_sum": int(data.loc[bucket, "unique_drugs"].sum()),
            "mean_descriptor_diversity": float(data.loc[bucket, "descriptor_diversity"].mean()),
            "mean_difficulty": float(data.loc[bucket, "difficulty"].mean()),
            "mean_validation_nll": float(data.loc[bucket, "validation_mean_nll"].mean()),
            "within_task_confusion": within,
        }
        for name in FREQUENCY_BINS:
            row[f"classes_{name}"] = int(
                (data.loc[bucket, "frequency_bin"] == name).sum()
            )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    sizes = task_sizes(args.num_tasks, args.base_task_classes, args.increment_classes)
    validate_task_sizes(sizes, args.num_classes, args.base_task_classes, args.increment_classes)
    counts = load_counts(args.class_counts)
    stats = pd.read_csv(args.class_stats)
    difficulty = pd.read_csv(args.difficulty)
    edges = pd.read_csv(args.confusion_edges)
    frame = counts.merge(stats.drop(columns=["count"], errors="ignore"), on="class_id", validate="one_to_one")
    frame = frame.merge(difficulty, on="class_id", validate="one_to_one")
    frame["frequency_bin"] = frame["count"].astype(int).map(frequency_bin)
    if frame.shape[0] != args.num_classes or frame.isna().any().any():
        raise ValueError("Advanced protocol signals are incomplete or contain NaN.")
    class_ids = sorted(frame["class_id"].astype(int).tolist())
    bin_counts = {
        name: int((frame["frequency_bin"] == name).sum()) for name in FREQUENCY_BINS
    }
    edge_lookup, total_edge_weight = build_edge_lookup(edges)
    requested = list(PROTOCOLS) if args.protocol == "all" else [args.protocol]
    args.outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    audits: list[dict[str, object]] = []
    for protocol in requested:
        slug = PROTOCOLS[protocol]
        for seed in args.seeds:
            quotas = (
                controlled_drift_quotas(sizes, bin_counts)
                if protocol == "P8"
                else proportional_bin_quotas(bin_counts, sizes, seed=seed)
            )
            buckets = assign_initial(
                frame,
                sizes,
                quotas,
                seed=seed,
                edge_lookup=edge_lookup if protocol == "P7" else None,
            )
            objective = build_objective(protocol, frame, sizes, edge_lookup, total_edge_weight)
            buckets = optimize_by_swaps(
                buckets,
                frame,
                objective,
                seed=seed,
                # The constrained greedy initialization already gives a good
                # mass balance. A few thousand accepted/rejected same-bin
                # swaps are sufficient without turning schedule generation
                # into a long CPU-bound experiment of its own.
                iterations=1_500 if protocol == "P7" else 3_000,
            )
            tasks = buckets_to_tasks(buckets)
            validate_tasks(tasks, class_ids, sizes)
            path = args.outdir / f"{slug}_seed{seed}_tasks.json"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite advanced task file: {path}")
            write_task_json(path, slug, seed, args.num_classes, tasks)
            summaries.append(
                summarize_tasks(slug, seed, tasks, counts, None, None)
            )
            audits.extend(audit_schedule(slug, seed, buckets, frame, edge_lookup))
            print(f"[tasks] Built {protocol} seed={seed}: {path}", flush=True)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.outdir / "advanced_task_summary.csv", index=False
    )
    pd.DataFrame(audits).to_csv(args.outdir / "advanced_task_audit.csv", index=False)
    manifest = {
        "protocols": {key: PROTOCOLS[key] for key in requested},
        "seeds": args.seeds,
        "test_split_used": False,
        "p7_variant": "confusion_spread",
        "frequency_bins": {
            "ultra_tail": "count <= 20",
            "tail": "21 <= count <= 100",
            "medium": "101 <= count <= 1000",
            "head": "count > 1000",
        },
    }
    (args.outdir / "advanced_protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[done] Wrote P5-P8 schedules to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
