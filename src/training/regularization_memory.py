"""Task-uniform episodic memory for A-GEM and empty memory for EWC.

Both use the same frozen source/ID guards as CB-Replay. A-GEM retains a seeded
uniform subset of each completed task; EWC stores Fisher/parameters separately.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from src.data.ddi_dataset import DDIBatchArrays
from src.data.class_balanced_replay_buffer import (
    ClassBalancedReplayBuffer, META_KEYS, _copy_arrays, _state_digest,
)


class RegularizationMemory(ClassBalancedReplayBuffer):
    def __init__(self, *, method, **kwargs):
        if method not in {"ewc", "agem"}:
            raise ValueError("Expected EWC or A-GEM memory.")
        self.method = method
        super().__init__(**kwargs)
        self._metadata.update(
            policy="agem_task_uniform_v1" if method == "agem" else "ewc_no_examples_v1",
            ranking={"policy": "seeded_task_uniform" if method == "agem" else "none"},
            allocation_rounding="fixed_task_quota" if method == "agem" else "none")
        self._total_tasks = len(self._tasks)

    def _quota(self, task_id):
        base, remainder = divmod(self._budget, self._total_tasks)
        return base + int(task_id < remainder) if self.method == "agem" else 0

    def update(self, current, *, task_id, feature_columns):
        self._check_sources()
        if task_id != self.next_task_id or tuple(feature_columns) != self._columns:
            raise ValueError("Memory task/feature order mismatch.")
        classes = self._tasks[task_id]
        self._validate_rows(current, self._expected_rows(classes), full=True)
        canonical = np.argsort(current.metadata["sample_id"], kind="stable")
        quota = min(len(canonical), self._quota(task_id))
        rng = np.random.default_rng(np.random.SeedSequence([
            self._metadata["seeds"]["member_seed"], task_id, 0x4147454D]))
        selected = canonical[rng.permutation(len(canonical))[:quota]]
        observed, entries = dict(self._observed), dict(self._entries)
        for c in classes:
            observed[c] = int(np.sum(current.labels == c))
            idx = selected[current.labels[selected] == c]
            entries[c] = DDIBatchArrays(
                np.asarray(current.features[idx], dtype=np.float64).copy(),
                current.labels[idx].astype(np.int64, copy=True),
                {k: current.metadata[k][idx].copy() for k in META_KEYS})
        self._entries, self._observed, self._completed_task_id = entries, observed, task_id
        audit = {"task_id": task_id, "policy": self._metadata["policy"],
                 "stored_slots": self.total_size, "budget": self._budget,
                 "observed_counts": observed}
        self._history.append(deepcopy(audit))
        self._check_sources()
        return audit

    @classmethod
    def from_state_dict(cls, state, **kwargs):
        result = cls(**kwargs)
        if state.get("state_sha256") != _state_digest(state) or state["metadata"] != result.metadata:
            raise ValueError("Regularization memory state SHA256/contract mismatch.")
        completed = state["completed_task_id"]
        if type(completed) is not int or not -1 <= completed < len(result._tasks):
            raise ValueError("Invalid memory completed task.")
        classes = [c for task in result._tasks[:completed + 1] for c in task]
        if set(state["entries"]) != set(classes) or set(state["observed_counts"]) != set(classes):
            raise ValueError("Memory class coverage mismatch.")
        for c, entry in state["entries"].items():
            arrays = DDIBatchArrays(entry["features"], entry["labels"], entry["metadata"])
            expected = result._expected_rows([c])
            result._validate_rows(arrays, expected, full=False)
            if state["observed_counts"][c] != len(expected):
                raise ValueError("Memory observed count mismatch.")
            result._entries[c] = _copy_arrays(arrays, slice(None))
        for task_id, task in enumerate(result._tasks[:completed + 1]):
            expected = min(len(result._expected_rows(task)), result._quota(task_id))
            if sum(len(result._entries[c].labels) for c in task) != expected:
                raise ValueError("A-GEM task quota mismatch.")
        if len(state["audit_history"]) != completed + 1:
            raise ValueError("Memory history coverage mismatch.")
        result._observed = dict(state["observed_counts"])
        result._completed_task_id = completed
        result._history = deepcopy(state["audit_history"])
        return result


def create_memory(**kwargs):
    method = kwargs.pop("method")
    if method == "class_balanced_fixed_budget_replay":
        return ClassBalancedReplayBuffer(**kwargs)
    return RegularizationMemory(method=method, **kwargs)


def restore_memory(state, **kwargs):
    method = kwargs.pop("method")
    if method == "class_balanced_fixed_budget_replay":
        return ClassBalancedReplayBuffer.from_state_dict(state, **kwargs)
    return RegularizationMemory.from_state_dict(state, method=method, **kwargs)
