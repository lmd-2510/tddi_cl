#!/usr/bin/env python3
"""Build class-incremental task files for DDI2025-CIL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_NUM_CLASSES = 178
DEFAULT_NUM_TASKS = 8
DEFAULT_BASE_TASK_CLASSES = 38
DEFAULT_INCREMENT_CLASSES = 20
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
RARE_THRESHOLDS = [5, 10, 20]
FREQUENCY_BINS = ("ultra_tail", "tail", "medium", "head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build class-incremental task protocols from class counts."
    )
    parser.add_argument("--class-counts", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        choices=[
            "random",
            "frequency_balanced",
            "long_tail",
            "head_to_tail",
            "tail_to_head",
            "constrained_mass_balanced",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--num-tasks", type=int, default=DEFAULT_NUM_TASKS)
    parser.add_argument("--base-task-classes", type=int, default=DEFAULT_BASE_TASK_CLASSES)
    parser.add_argument("--increment-classes", type=int, default=DEFAULT_INCREMENT_CLASSES)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--validation-counts", type=Path, default=None)
    parser.add_argument("--test-counts", type=Path, default=None)
    return parser.parse_args()


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns.tolist()]
    rows = frame.astype(object).where(pd.notna(frame), "").values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def resolve_optional_counts(train_counts_path: Path, explicit: Path | None, split: str) -> Path | None:
    if explicit is not None:
        return explicit
    candidate = train_counts_path.with_name(f"class_counts_{split}.csv")
    return candidate if candidate.exists() else None


def load_counts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {"class_id", "count"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Counts file missing required columns {expected}: {path}")
    return frame[["class_id", "count"]].copy()


def task_sizes(num_tasks: int, base_task_classes: int, increment_classes: int) -> list[int]:
    sizes = [base_task_classes] + [increment_classes] * (num_tasks - 1)
    return sizes


def validate_task_sizes(
    sizes: list[int],
    num_classes: int,
    base_task_classes: int,
    increment_classes: int,
) -> None:
    if sum(sizes) != num_classes:
        raise ValueError(
            f"Task sizes sum to {sum(sizes)}, expected {num_classes}. "
            f"Check --num-classes/--num-tasks/--base-task-classes/--increment-classes."
        )
    if sizes[0] != base_task_classes:
        raise ValueError("Task 0 size mismatch.")
    if any(size != increment_classes for size in sizes[1:]):
        raise ValueError("Incremental task size mismatch.")


def build_random_order(class_ids: list[int], seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    order = class_ids.copy()
    rng.shuffle(order)
    return order


def build_long_tail_order(train_counts: pd.DataFrame) -> list[int]:
    """Backward-compatible alias for the P2 head-to-tail ordering."""

    return build_frequency_order(train_counts, descending=True)


def build_frequency_order(
    train_counts: pd.DataFrame,
    *,
    descending: bool,
) -> list[int]:
    """Order classes deterministically by train frequency, then class ID."""

    ordered = train_counts.sort_values(["count", "class_id"], ascending=[False, True])
    if not descending:
        ordered = train_counts.sort_values(["count", "class_id"], ascending=[True, True])
    return ordered["class_id"].astype(int).tolist()


def build_frequency_balanced_tasks(train_counts: pd.DataFrame, sizes: list[int]) -> list[dict[str, Any]]:
    ordered = train_counts.sort_values(["count", "class_id"], ascending=[False, True]).reset_index(drop=True)
    stratum_size = int(np.ceil(len(ordered) / 3))
    strata = [
        ordered.iloc[0:stratum_size]["class_id"].astype(int).tolist(),
        ordered.iloc[stratum_size : 2 * stratum_size]["class_id"].astype(int).tolist(),
        ordered.iloc[2 * stratum_size :]["class_id"].astype(int).tolist(),
    ]
    buckets: list[list[int]] = [[] for _ in sizes]
    remaining = sizes.copy()
    cursor = 0

    for stratum in strata:
        for class_id in stratum:
            attempts = 0
            while remaining[cursor % len(sizes)] == 0:
                cursor += 1
                attempts += 1
                if attempts > len(sizes):
                    raise ValueError("No remaining task capacity while building frequency-balanced tasks.")
            task_id = cursor % len(sizes)
            buckets[task_id].append(int(class_id))
            remaining[task_id] -= 1
            cursor += 1

    return [
        {
            "task_id": task_id,
            "classes": classes,
            "num_classes": len(classes),
        }
        for task_id, classes in enumerate(buckets)
    ]


def frequency_bin(count: int) -> str:
    """Return the P4 rarity stratum using train-only sample counts."""

    if count <= 20:
        return "ultra_tail"
    if count <= 100:
        return "tail"
    if count <= 1000:
        return "medium"
    return "head"


def proportional_bin_quotas(
    bin_counts: dict[str, int],
    sizes: list[int],
    *,
    seed: int,
) -> list[dict[str, int]]:
    """Apportion every frequency bin across tasks while preserving capacities."""

    total_classes = sum(bin_counts.values())
    if total_classes != sum(sizes):
        raise ValueError("Frequency-bin totals must equal total task capacity.")
    missing = set(FREQUENCY_BINS) - set(bin_counts)
    if missing:
        raise ValueError(f"Missing frequency bins: {sorted(missing)}")

    expected = [
        {name: size * bin_counts[name] / total_classes for name in FREQUENCY_BINS}
        for size in sizes
    ]
    quotas = [
        {name: int(np.floor(row[name])) for name in FREQUENCY_BINS}
        for row in expected
    ]
    row_remaining = [sizes[i] - sum(quotas[i].values()) for i in range(len(sizes))]
    col_remaining = {
        name: bin_counts[name] - sum(row[name] for row in quotas)
        for name in FREQUENCY_BINS
    }
    rng = np.random.default_rng(seed)
    tie_break = {
        (task_id, name): float(rng.random())
        for task_id in range(len(sizes))
        for name in FREQUENCY_BINS
    }

    while sum(row_remaining):
        candidates = [
            (task_id, name)
            for task_id in range(len(sizes))
            for name in FREQUENCY_BINS
            if row_remaining[task_id] > 0 and col_remaining[name] > 0
        ]
        if not candidates:
            raise RuntimeError("Unable to complete P4 frequency-bin quotas.")
        task_id, name = max(
            candidates,
            key=lambda item: (
                expected[item[0]][item[1]] - quotas[item[0]][item[1]],
                tie_break[item],
            ),
        )
        quotas[task_id][name] += 1
        row_remaining[task_id] -= 1
        col_remaining[name] -= 1

    if any(col_remaining.values()):
        raise RuntimeError(f"P4 quota columns are incomplete: {col_remaining}")
    return quotas


def build_constrained_mass_balanced_tasks(
    train_counts: pd.DataFrame,
    sizes: list[int],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Build P4: balanced rarity composition and approximately balanced mass."""

    frame = train_counts.copy()
    frame["frequency_bin"] = frame["count"].astype(int).map(frequency_bin)
    bin_counts = {
        name: int((frame["frequency_bin"] == name).sum())
        for name in FREQUENCY_BINS
    }
    quotas = proportional_bin_quotas(bin_counts, sizes, seed=seed)
    used = [{name: 0 for name in FREQUENCY_BINS} for _ in sizes]
    buckets: list[list[tuple[int, int, str]]] = [[] for _ in sizes]
    sample_mass = [0 for _ in sizes]
    rng = np.random.default_rng(seed)
    task_priority = rng.permutation(len(sizes)).tolist()
    task_rank = {task_id: rank for rank, task_id in enumerate(task_priority)}
    frame["tie_break"] = rng.random(len(frame))
    ordered = frame.sort_values(
        ["count", "tie_break", "class_id"],
        ascending=[False, True, True],
    )

    for row in ordered.itertuples(index=False):
        name = str(row.frequency_bin)
        candidates = [
            task_id
            for task_id in range(len(sizes))
            if used[task_id][name] < quotas[task_id][name]
        ]
        if not candidates:
            raise RuntimeError(f"No P4 quota remains for frequency bin {name}.")
        task_id = min(
            candidates,
            key=lambda candidate: (sample_mass[candidate], task_rank[candidate]),
        )
        count = int(row.count)
        buckets[task_id].append((int(row.class_id), count, name))
        sample_mass[task_id] += count
        used[task_id][name] += 1

    # Swapping within the same bin preserves all rarity quotas.  Greedily take
    # the best mass-balancing swap until reaching a local optimum.
    for _ in range(1000):
        best: tuple[int, int, int, int, int] | None = None
        for left in range(len(buckets)):
            for right in range(left + 1, len(buckets)):
                old_score = sample_mass[left] ** 2 + sample_mass[right] ** 2
                for left_index, (_, left_count, left_bin) in enumerate(buckets[left]):
                    for right_index, (_, right_count, right_bin) in enumerate(buckets[right]):
                        if left_bin != right_bin or left_count == right_count:
                            continue
                        new_left = sample_mass[left] - left_count + right_count
                        new_right = sample_mass[right] - right_count + left_count
                        improvement = old_score - (new_left ** 2 + new_right ** 2)
                        candidate = (improvement, left, right, left_index, right_index)
                        if improvement > 0 and (best is None or candidate > best):
                            best = candidate
        if best is None:
            break
        _, left, right, left_index, right_index = best
        left_item = buckets[left][left_index]
        right_item = buckets[right][right_index]
        buckets[left][left_index], buckets[right][right_index] = right_item, left_item
        sample_mass[left] += right_item[1] - left_item[1]
        sample_mass[right] += left_item[1] - right_item[1]

    return [
        {
            "task_id": task_id,
            "classes": [class_id for class_id, _, _ in bucket],
            "num_classes": len(bucket),
        }
        for task_id, bucket in enumerate(buckets)
    ]


def split_order_into_tasks(order: list[int], sizes: list[int]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    start = 0
    for task_id, size in enumerate(sizes):
        classes = order[start : start + size]
        tasks.append(
            {
                "task_id": task_id,
                "classes": classes,
                "num_classes": len(classes),
            }
        )
        start += size
    return tasks


def validate_tasks(tasks: list[dict[str, Any]], expected_class_ids: list[int], sizes: list[int]) -> None:
    flat = [class_id for task in tasks for class_id in task["classes"]]
    if len(flat) != len(expected_class_ids):
        raise ValueError("Task assignment lost classes.")
    if len(set(flat)) != len(flat):
        raise ValueError("Task assignment has duplicated classes.")
    if sorted(flat) != sorted(expected_class_ids):
        raise ValueError("Task assignment does not match expected class ID set.")
    for task, size in zip(tasks, sizes, strict=True):
        if task["num_classes"] != size:
            raise ValueError(f"Task {task['task_id']} has wrong size.")


def counts_lookup(frame: pd.DataFrame | None) -> dict[int, int]:
    if frame is None:
        return {}
    return {int(row.class_id): int(row.count) for row in frame.itertuples(index=False)}


def summarize_tasks(
    protocol: str,
    seed: int | None,
    tasks: list[dict[str, Any]],
    train_counts: pd.DataFrame,
    validation_counts: pd.DataFrame | None,
    test_counts: pd.DataFrame | None,
) -> pd.DataFrame:
    train_lookup = counts_lookup(train_counts)
    validation_lookup = counts_lookup(validation_counts)
    test_lookup = counts_lookup(test_counts)
    rows: list[dict[str, Any]] = []
    seen_classes: list[int] = []

    for task in tasks:
        classes = task["classes"]
        seen_classes.extend(classes)
        row = {
            "protocol": protocol,
            "seed": seed if seed is not None else "",
            "task_id": task["task_id"],
            "num_classes": task["num_classes"],
            "classes": ",".join(map(str, classes)),
            "train_samples_current": int(sum(train_lookup.get(class_id, 0) for class_id in classes)),
            "validation_samples_current": int(sum(validation_lookup.get(class_id, 0) for class_id in classes)),
            "test_samples_current": int(sum(test_lookup.get(class_id, 0) for class_id in classes)),
            "train_samples_seen": int(sum(train_lookup.get(class_id, 0) for class_id in seen_classes)),
            "validation_samples_seen": int(sum(validation_lookup.get(class_id, 0) for class_id in seen_classes)),
            "test_samples_seen": int(sum(test_lookup.get(class_id, 0) for class_id in seen_classes)),
        }
        for threshold in RARE_THRESHOLDS:
            row[f"rare_classes_le_{threshold}"] = int(
                sum(train_lookup.get(class_id, 0) <= threshold for class_id in classes)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_task_json(
    outpath: Path,
    protocol: str,
    seed: int | None,
    num_classes: int,
    tasks: list[dict[str, Any]],
) -> None:
    payload = {
        "protocol": protocol,
        "seed": seed,
        "num_classes": num_classes,
        "tasks": tasks,
    }
    outpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_protocol_summary(outpath: Path, summaries: list[pd.DataFrame]) -> None:
    summary = pd.concat(summaries, ignore_index=True)
    lines = [
        "# Task Protocol Summary",
        "",
        "## Notes",
        "",
        "- Task files are built from actual class IDs, not `range(num_classes)`.",
        "- Task 0 contains 38 classes; Tasks 1-7 contain 20 classes each.",
        "- Rare-class counts use train split thresholds `<=5`, `<=10`, and `<=20`.",
        "",
        "## Task Summary",
        "",
        dataframe_to_markdown_table(summary),
        "",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sizes = task_sizes(args.num_tasks, args.base_task_classes, args.increment_classes)
    validate_task_sizes(
        sizes=sizes,
        num_classes=args.num_classes,
        base_task_classes=args.base_task_classes,
        increment_classes=args.increment_classes,
    )

    train_counts = load_counts(args.class_counts)
    validation_path = resolve_optional_counts(args.class_counts, args.validation_counts, "validation")
    test_path = resolve_optional_counts(args.class_counts, args.test_counts, "test")
    validation_counts = load_counts(validation_path) if validation_path else None
    test_counts = load_counts(test_path) if test_path else None

    class_ids = sorted(train_counts["class_id"].astype(int).tolist())
    if len(class_ids) != args.num_classes:
        raise ValueError(
            f"Train class count file has {len(class_ids)} classes, expected {args.num_classes}."
        )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []

    if args.protocol in {"random", "all"}:
        for seed in args.seeds:
            print(f"[tasks] Building random protocol for seed={seed}...", flush=True)
            order = build_random_order(class_ids, seed)
            tasks = split_order_into_tasks(order, sizes)
            validate_tasks(tasks, class_ids, sizes)
            outpath = outdir / f"random_seed{seed}_tasks.json"
            write_task_json(outpath, "random", seed, args.num_classes, tasks)
            summaries.append(
                summarize_tasks(
                    protocol="random",
                    seed=seed,
                    tasks=tasks,
                    train_counts=train_counts,
                    validation_counts=validation_counts,
                    test_counts=test_counts,
                )
            )

    if args.protocol in {"frequency_balanced", "all"}:
        print("[tasks] Building frequency-balanced protocol...", flush=True)
        tasks = build_frequency_balanced_tasks(train_counts, sizes)
        validate_tasks(tasks, class_ids, sizes)
        outpath = outdir / "frequency_balanced_tasks.json"
        write_task_json(outpath, "frequency_balanced", None, args.num_classes, tasks)
        summaries.append(
            summarize_tasks(
                protocol="frequency_balanced",
                seed=None,
                tasks=tasks,
                train_counts=train_counts,
                validation_counts=validation_counts,
                test_counts=test_counts,
            )
        )

    if args.protocol in {"long_tail", "head_to_tail", "all"}:
        print("[tasks] Building P2 head-to-tail protocol...", flush=True)
        order = build_frequency_order(train_counts, descending=True)
        tasks = split_order_into_tasks(order, sizes)
        validate_tasks(tasks, class_ids, sizes)
        outpath = outdir / "head_to_tail_tasks.json"
        write_task_json(outpath, "head_to_tail", None, args.num_classes, tasks)
        # Preserve the historical filename/schema used by existing commands.
        write_task_json(
            outdir / "long_tail_tasks.json",
            "long_tail",
            None,
            args.num_classes,
            tasks,
        )
        summaries.append(
            summarize_tasks(
                protocol="head_to_tail",
                seed=None,
                tasks=tasks,
                train_counts=train_counts,
                validation_counts=validation_counts,
                test_counts=test_counts,
            )
        )

    if args.protocol in {"tail_to_head", "all"}:
        print("[tasks] Building P3 tail-to-head protocol...", flush=True)
        order = build_frequency_order(train_counts, descending=False)
        tasks = split_order_into_tasks(order, sizes)
        validate_tasks(tasks, class_ids, sizes)
        outpath = outdir / "tail_to_head_tasks.json"
        write_task_json(outpath, "tail_to_head", None, args.num_classes, tasks)
        summaries.append(
            summarize_tasks(
                protocol="tail_to_head",
                seed=None,
                tasks=tasks,
                train_counts=train_counts,
                validation_counts=validation_counts,
                test_counts=test_counts,
            )
        )

    if args.protocol in {"constrained_mass_balanced", "all"}:
        for seed in args.seeds:
            print(f"[tasks] Building P4 constrained mass-balanced seed={seed}...", flush=True)
            tasks = build_constrained_mass_balanced_tasks(train_counts, sizes, seed=seed)
            validate_tasks(tasks, class_ids, sizes)
            outpath = outdir / f"constrained_mass_balanced_seed{seed}_tasks.json"
            write_task_json(
                outpath,
                "constrained_mass_balanced",
                seed,
                args.num_classes,
                tasks,
            )
            summaries.append(
                summarize_tasks(
                    protocol="constrained_mass_balanced",
                    seed=seed,
                    tasks=tasks,
                    train_counts=train_counts,
                    validation_counts=validation_counts,
                    test_counts=test_counts,
                )
            )

    if not summaries:
        raise ValueError("No protocol requested.")

    task_summary = pd.concat(summaries, ignore_index=True)
    task_summary.to_csv(outdir / "task_summary.csv", index=False)
    write_protocol_summary(outdir / "task_protocol_summary.md", summaries)
    print(f"[done] Wrote task artifacts to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
