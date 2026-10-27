"""Canonical task-protocol contract for the DDI LT-CIL study.

The protocol names intentionally mirror the two scenarios introduced in
"Long-Tailed Class Incremental Learning" (ECCV 2022): Ordered LT-CIL and
Shuffled LT-CIL.  This repository applies those class-order rules to the
natural DDI class frequencies while retaining its 38 + 7x20 task layout.
"""

from __future__ import annotations

from typing import Any, Mapping


HEAD_TO_TAIL = "head_to_tail"
SHUFFLED = "shuffled"
SUPPORTED_PROTOCOLS = frozenset({HEAD_TO_TAIL, SHUFFLED})
PROTOCOL_IDS = {
    HEAD_TO_TAIL: "ORDERED",
    SHUFFLED: "SHUFFLED",
}
TASK_LAYOUT = [38, 20, 20, 20, 20, 20, 20, 20]
CLASS_COUNT = 178


def expected_seed(protocol: str) -> int | None:
    """Return the canonical seed contract for a task protocol."""

    if protocol == HEAD_TO_TAIL:
        return None
    if protocol == SHUFFLED:
        return 0
    raise ValueError(
        f"Unsupported protocol {protocol!r}; expected one of "
        f"{sorted(SUPPORTED_PROTOCOLS)}."
    )


def validate_full_task_spec(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the canonical eight-task, 178-class protocol artifact."""

    protocol = str(payload.get("protocol", ""))
    expected = expected_seed(protocol)
    if payload.get("seed") != expected:
        raise ValueError(
            f"Protocol {protocol!r} requires seed={expected!r}, "
            f"got {payload.get('seed')!r}."
        )
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(TASK_LAYOUT):
        raise ValueError("Task protocol must define exactly eight tasks.")
    if [int(task.get("task_id", -1)) for task in tasks] != list(range(8)):
        raise ValueError("Task IDs must be contiguous from 0 through 7.")
    if [len(task.get("classes", [])) for task in tasks] != TASK_LAYOUT:
        raise ValueError(f"Task layout must be {TASK_LAYOUT}.")
    classes = [int(raw) for task in tasks for raw in task["classes"]]
    if len(classes) != CLASS_COUNT or len(set(classes)) != CLASS_COUNT:
        raise ValueError("Task protocol must contain 178 distinct raw classes.")
    if int(payload.get("num_classes", CLASS_COUNT)) != CLASS_COUNT:
        raise ValueError("Task protocol num_classes must be 178.")
    return tasks
