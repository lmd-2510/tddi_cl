from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_cil_tasks import (
    build_frequency_order,
    build_shuffled_order,
    split_order_into_tasks,
)
from src.data.task_protocols import validate_full_task_spec


def test_ordered_protocol_is_head_to_tail_with_deterministic_ties() -> None:
    counts = pd.DataFrame(
        {"class_id": [40, 10, 30, 20], "count": [2, 100, 2, 7]}
    )
    assert build_frequency_order(counts, descending=True) == [10, 20, 40, 30]


def test_shuffled_protocol_is_seeded_and_does_not_change_the_class_set() -> None:
    ordered = list(range(32))
    first = build_shuffled_order(ordered, seed=7)
    second = build_shuffled_order(ordered, seed=7)
    other = build_shuffled_order(ordered, seed=8)
    assert first == second
    assert first != ordered
    assert first != other
    assert sorted(first) == ordered


def test_shuffled_order_is_sliced_without_frequency_rebalancing() -> None:
    order = [8, 2, 6, 1, 7, 0, 5, 3, 4]
    tasks = split_order_into_tasks(order, [4, 3, 2])
    assert [task["classes"] for task in tasks] == [order[:4], order[4:7], order[7:]]


def test_frozen_assets_are_exactly_the_two_canonical_protocols() -> None:
    root = Path("study_assets/task_protocols")
    task_files = sorted(path.name for path in root.glob("*_tasks.json"))
    assert task_files == ["head_to_tail_tasks.json", "shuffled_seed0_tasks.json"]

    ordered = json.loads((root / task_files[0]).read_text(encoding="utf-8"))
    shuffled = json.loads((root / task_files[1]).read_text(encoding="utf-8"))
    assert ordered["protocol"] == "head_to_tail"
    assert ordered["seed"] is None
    assert shuffled["protocol"] == "shuffled"
    assert shuffled["seed"] == 0
    validate_full_task_spec(ordered)
    validate_full_task_spec(shuffled)

    ordered_classes = [raw for task in ordered["tasks"] for raw in task["classes"]]
    shuffled_classes = [raw for task in shuffled["tasks"] for raw in task["classes"]]
    assert set(ordered_classes) == set(shuffled_classes)
    assert ordered_classes != shuffled_classes
