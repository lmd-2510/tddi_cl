"""Method-specific memory selection for regularization and replay methods.

All methods use the same frozen source/ID guards. A-GEM retains a seeded
uniform subset of each completed task; EWC stores Fisher/parameters separately;
ER uses a paper-style uniform reservoir; IDER and UPCL reuse the proven
equal-class reservoir required by their paper-aligned rehearsal procedures.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from src.data.ddi_dataset import DDIBatchArrays
from src.data.class_balanced_replay_buffer import (
    BUFFER_POLICY, RANKING_POLICY, ClassBalancedReplayBuffer, META_KEYS, _copy_arrays, _state_digest,
)


ER_BUFFER_POLICY = "reservoir_uniform_v1"
ER_RANKING_POLICY = "reservoir_uniform_v1"


def _python_scalar(value):
    """Convert NumPy scalar metadata without assuming ``.item()`` exists.

    Arrow-backed string columns can arrive as plain Python ``str`` objects,
    while numeric columns commonly arrive as ``np.generic`` scalars.  ER
    stores metadata in JSON/state dictionaries, so normalize both forms here.
    """
    return value.item() if isinstance(value, np.generic) else value


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


def _pythonize(value):
    """Make NumPy generator state portable and JSON/digest friendly."""
    if isinstance(value, dict):
        return {str(key): _pythonize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pythonize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class ReservoirReplayBuffer(ClassBalancedReplayBuffer):
    """Standard ER reservoir over the incoming sample stream.

    Unlike CB-Replay this buffer has no class quota and no exemplar ranking.
    Each arriving training row gets equal probability of occupying one of the
    fixed slots; the stream order is the validated current-task row order.
    """

    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs["buffer_policy"] = BUFFER_POLICY
        kwargs["ranking_policy"] = RANKING_POLICY
        super().__init__(**kwargs)
        self._metadata.update(
            policy=ER_BUFFER_POLICY,
            ranking={
                "policy": ER_RANKING_POLICY,
                "algorithm": "uniform_reservoir_sampling",
                "stream_order": "validated_current_task_row_order",
                "replacement": "uniform_integer_0_through_seen_count",
                "status": "paper_er",
            },
            allocation_rounding="reservoir_uniform_stream",
        )
        seed = self._metadata["seeds"]["experiment_seed"]
        member = self._metadata["seeds"]["member_seed"]
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, member, 0x45525253]))
        self._seen_count = 0

    def _record(self, current, index):
        return {
            "features": np.asarray(current.features[index], dtype=np.float64).copy(),
            "label": int(current.labels[index]),
            "metadata": {key: _python_scalar(np.asarray(current.metadata[key])[index])
                         for key in META_KEYS},
        }

    def _rebuild_entries(self, records):
        grouped = {}
        for record in records:
            grouped.setdefault(record["label"], []).append(record)
        updated = {}
        for label in sorted(grouped):
            rows = grouped[label]
            updated[label] = DDIBatchArrays(
                np.stack([row["features"] for row in rows]).astype(np.float64, copy=False),
                np.asarray([row["label"] for row in rows], dtype=np.int64),
                {key: np.asarray([row["metadata"][key] for row in rows]) for key in META_KEYS},
            )
            updated[label].metadata.update(
                rank_priority=np.arange(len(rows), dtype=np.int64),
                rank_distance=np.zeros(len(rows), dtype=np.float64),
            )
        self._entries = updated

    def update(self, current, *, task_id, feature_columns):
        self._check_sources()
        task_id = int(task_id)
        if task_id != self.next_task_id or task_id >= len(self._tasks):
            raise ValueError("Update must use the next task exactly once.")
        if tuple(feature_columns) != self._columns:
            raise ValueError("Buffer feature order mismatch.")
        classes = self._tasks[task_id]
        expected = self._expected_rows(classes)
        self._validate_rows(current, expected, full=True)
        if set(np.unique(current.labels).tolist()) != set(classes):
            raise ValueError("Every arriving class must have member training samples.")

        records = []
        for entry in self._entries.values():
            for index in range(len(entry.labels)):
                records.append({
                    "features": np.asarray(entry.features[index], dtype=np.float64).copy(),
                    "label": int(entry.labels[index]),
                    "metadata": {key: _python_scalar(entry.metadata[key][index])
                                 for key in META_KEYS},
                })
        for index in range(len(current.labels)):
            record = self._record(current, index)
            if self._seen_count < self._budget:
                records.append(record)
            else:
                slot = int(self._rng.integers(0, self._seen_count + 1))
                if slot < self._budget:
                    records[slot] = record
            self._seen_count += 1
        if len(records) > self._budget:
            raise RuntimeError("Reservoir exceeded its fixed memory budget.")
        self._rebuild_entries(records)
        observed = dict(self._observed)
        for label in classes:
            observed[int(label)] = int(np.sum(current.labels == label))
        audit = {
            "task_id": task_id, "policy": ER_BUFFER_POLICY,
            "observed_counts": observed, "stored_slots": len(records),
            "budget": self._budget, "reservoir_seen_count": self._seen_count,
            "replacement_policy": "uniform_reservoir_sampling",
        }
        self._entries, self._observed, self._completed_task_id = self._entries, observed, task_id
        self._history.append(deepcopy(audit))
        self._check_sources()
        return deepcopy(audit)

    def state_dict(self):
        state = {
            "metadata": self.metadata, "completed_task_id": self._completed_task_id,
            "observed_counts": self.observed_counts, "audit_history": self.audit_history,
            "reservoir_seen_count": self._seen_count,
            "rng_state": _pythonize(self._rng.bit_generator.state),
            "entries": {c: {"features": a.features.copy(), "labels": a.labels.copy(),
                             "metadata": {k: v.copy() for k, v in a.metadata.items()}}
                        for c, a in self._entries.items()},
        }
        state["state_sha256"] = _state_digest(state)
        return state

    @classmethod
    def from_state_dict(cls, state, **kwargs):
        result = cls(**kwargs)
        try:
            required = {"metadata", "completed_task_id", "observed_counts", "audit_history",
                        "reservoir_seen_count", "rng_state", "entries", "state_sha256"}
            if set(state) != required or state["state_sha256"] != _state_digest(state):
                raise ValueError("Reservoir state SHA256/field mismatch.")
            if state["metadata"] != result.metadata:
                raise ValueError("Reservoir state policy/config/source metadata mismatch.")
            completed = state["completed_task_id"]
            if type(completed) is not int or not -1 <= completed < len(result._tasks):
                raise ValueError("Invalid reservoir completed task.")
            if type(state["reservoir_seen_count"]) is not int or state["reservoir_seen_count"] < 0:
                raise ValueError("Invalid reservoir seen count.")
            classes = [c for task in result._tasks[:completed + 1] for c in task]
            if set(state["observed_counts"]) != set(classes):
                raise ValueError("Reservoir observed class coverage mismatch.")
            if len(state["audit_history"]) != completed + 1:
                raise ValueError("Reservoir audit history coverage mismatch.")
            if len(state["entries"]) and sum(len(entry["labels"]) for entry in state["entries"].values()) > result._budget:
                raise ValueError("Reservoir state exceeds memory budget.")
            for label, entry in state["entries"].items():
                arrays = DDIBatchArrays(entry["features"], entry["labels"], entry["metadata"])
                arrays_expected = result._expected_rows([int(label)])
                result._validate_rows(arrays, arrays_expected, full=False)
                if set(entry["metadata"]) != {*META_KEYS, "rank_priority", "rank_distance"}:
                    raise ValueError("Malformed reservoir exemplar metadata.")
                result._entries[int(label)] = _copy_arrays(arrays, slice(None))
            result._observed = dict(state["observed_counts"])
            result._completed_task_id = completed
            result._history = deepcopy(state["audit_history"])
            result._seen_count = int(state["reservoir_seen_count"])
            result._rng.bit_generator.state = deepcopy(state["rng_state"])
            result._check_sources()
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ValueError("Malformed reservoir replay buffer state.") from error
        return result


def create_memory(**kwargs):
    method = kwargs.pop("method")
    if method == "er":
        return ReservoirReplayBuffer(**kwargs)
    if method in {"class_balanced_fixed_budget_replay", "ider", "upcl"}:
        return ClassBalancedReplayBuffer(**kwargs)
    return RegularizationMemory(method=method, **kwargs)


def restore_memory(state, **kwargs):
    method = kwargs.pop("method")
    if method == "er":
        return ReservoirReplayBuffer.from_state_dict(state, **kwargs)
    if method in {"class_balanced_fixed_budget_replay", "ider", "upcl"}:
        return ClassBalancedReplayBuffer.from_state_dict(state, **kwargs)
    return RegularizationMemory.from_state_dict(state, method=method, **kwargs)
