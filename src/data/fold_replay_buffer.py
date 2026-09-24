"""Opt-in fold-aware sqrt-quota replay buffer for the controlled A/B pilots.

Not the legacy uniform buffer. Only retained RAW exemplars are stored. Temporary
per-sample-normalized selection vectors are never used as model inputs or saved.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.data.ddi_dataset import DEFAULT_META_COLS, DDIBatchArrays, DevelopmentFoldContext
from src.data.fixed_budget_replay import max_min_uniform_allocation
from src.data.fold_preprocessing import FoldPreprocessing, SUPPORTED_PROTOCOLS
from src.data.sample_identity import DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN, build_stable_sample_ids
from src.utils.seed import resolve_seed_configuration


BUFFER_POLICY = "fold_min_quota_sqrt_capacity_v1"
EQUAL_CLASS_BUFFER_POLICY = "fold_equal_class_capacity_v1"
BUFFER_POLICIES = (BUFFER_POLICY, EQUAL_CLASS_BUFFER_POLICY)
SAMPLE_NORMALIZED_RANKING_POLICY = "raw_sample_normalized_class_mean_control_v1"
PIPELINE_INPUT_RANKING_POLICY = "frozen_preprocessed_input_class_mean_v1"
RANKING_POLICIES = (SAMPLE_NORMALIZED_RANKING_POLICY, PIPELINE_INPUT_RANKING_POLICY)
# Backward-compatible name used by Prompt 7–13 configs and imports.
RANKING_POLICY = SAMPLE_NORMALIZED_RANKING_POLICY
RANKING_EPSILON = 1e-5
SAMPLE_NORMALIZED_RANKING_CONVENTIONS = {
    "policy": RANKING_POLICY, "epsilon": RANKING_EPSILON, "variance_ddof": 0,
    "dtype": "float64", "distance": "euclidean", "tie_break": "sample_id_lexicographic",
    "reduction_order": "sample_id_ascending", "learned_affine": False,
    "nonfinite_policy": "error", "status": "temporary_AB_control_not_final_optimum",
}
# Backward-compatible immutable template; per-buffer metadata is built below.
RANKING_CONVENTIONS = SAMPLE_NORMALIZED_RANKING_CONVENTIONS
META_KEYS = ("sample_id", "source_split", "source_row_index", "fold_id", DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN)


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= int(value) <= 2**63 - 1:
        raise ValueError(f"{name} must be an integer in [{minimum}, int64_max].")
    return int(value)


def _class_id(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not -(2**63) <= int(value) < 2**63:
        raise ValueError("Raw class IDs must be int64 integers.")
    return int(value)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def four_percent_member_budgets(development_count: int) -> dict[int, int]:
    """Count stored copies, not union IDs: floor(4*N/100), remainder to low IDs."""
    total = 4 * _integer(development_count, "development_count") // 100
    quotient, remainder = divmod(total, 3)
    return {member: quotient + int(member < remainder) for member in range(3)}


def min_quota_sqrt_allocation(
    observed_counts: Mapping[int, int], capacities: Mapping[int, int],
    budget: int, *, base_quota: int = 10,
) -> dict[int, int]:
    """Min quota, then capped sqrt-weight water filling + largest remainder.

    Observed counts are fixed arrival counts (weights); capacities are CURRENT
    retained counts for old classes, and full current counts for new classes.
    If the base cannot fit, max-min allocate the clipped base quotas. All integer
    ties use ascending raw class ID. Empty/zero-capacity/zero-budget are supported.
    """
    budget, base_quota = _integer(budget, "budget"), _integer(base_quota, "base_quota")
    observed = {_class_id(c): _integer(n, "observed count", 1) for c, n in observed_counts.items()}
    caps = {_class_id(c): _integer(n, "capacity") for c, n in capacities.items()}
    if set(observed) != set(caps) or any(caps[c] > observed[c] for c in caps):
        raise ValueError("Observed/capacity classes must match and capacity cannot exceed observed count.")
    classes = sorted(caps)
    target = min(budget, sum(caps.values()))
    base = {c: min(base_quota, caps[c]) for c in classes}
    if target == 0:
        return {c: 0 for c in classes}
    if target < sum(base.values()):
        return max_min_uniform_allocation(base, target)
    allocation = dict(base)
    remaining = target - sum(allocation.values())
    while remaining:
        active = [c for c in classes if allocation[c] < caps[c]]
        weights = np.sqrt(np.asarray([observed[c] for c in active], dtype=np.float64))
        shares = remaining * (weights / weights.sum())
        saturated = [c for c, share in zip(active, shares, strict=True) if share >= caps[c] - allocation[c]]
        if saturated:
            for c in saturated:
                remaining -= caps[c] - allocation[c]
                allocation[c] = caps[c]
            continue
        floors = np.floor(shares).astype(np.int64)
        for c, amount in zip(active, floors, strict=True):
            allocation[c] += int(amount)
            remaining -= int(amount)
        order = sorted(range(len(active)), key=lambda i: (-(shares[i] - floors[i]), active[i]))
        if not 0 <= remaining <= len(order):
            raise ValueError("Budget too large for reliable float64 allocation rounding.")
        for i in order[:remaining]:
            allocation[active[i]] += 1
        remaining = 0
    if sum(allocation.values()) != target or any(allocation[c] > caps[c] for c in classes):
        raise RuntimeError("Capacity-constrained sqrt allocation invariant failed.")
    return allocation


def equal_class_capacity_allocation(capacities: Mapping[int, int], budget: int) -> dict[int, int]:
    """Allocate slots equally across seen classes, capped by retained availability."""
    budget = _integer(budget, "budget")
    caps = {_class_id(c): _integer(n, "capacity") for c, n in capacities.items()}
    if not caps:
        return {}
    if budget == 0:
        return {c: 0 for c in sorted(caps)}
    return max_min_uniform_allocation(caps, budget)


def rank_raw_class_exemplars(raw_features: np.ndarray, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return original-row indices in rank order and their Euclidean distances.

    One class per call. Compute mean in canonical ID order so input shuffling does
    not change reductions. Constant rows normalize to zero. Fail, never impute,
    on nonfinite input or overflow. No scaler, model latent, UE or random seed.
    """
    raw = np.asarray(raw_features)
    ids = np.asarray(sample_ids)
    if raw.ndim != 2 or not raw.shape[0] or not raw.shape[1] or raw.dtype.kind not in "fiu":
        raise ValueError("Ranking requires a nonempty numeric raw descriptor matrix.")
    if (ids.shape != (len(raw),) or not all(isinstance(s, (str, np.str_)) and s for s in ids)
            or len(set(ids.tolist())) != len(ids)):
        raise ValueError("Ranking sample IDs must be unique nonempty strings aligned to rows.")
    raw = raw.astype(np.float64, copy=False)
    if not np.isfinite(raw).all():
        raise ValueError("Ranking raw descriptors must be finite; nonfinite policy=error.")
    canonical = np.argsort(ids, kind="stable")
    x = raw[canonical]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centered = x - x.mean(axis=1, keepdims=True)
        variance = np.square(centered).mean(axis=1, keepdims=True)
        z = centered / np.sqrt(variance + RANKING_EPSILON)
        distances = np.linalg.norm(z - z.mean(axis=0, keepdims=True), axis=1)
    if not np.isfinite(variance).all() or not np.isfinite(distances).all() or not np.isfinite(z).all():
        raise ValueError("Ranking normalization/distance overflowed float64.")
    order = np.lexsort((ids[canonical], distances))
    return canonical[order], distances[order]


def rank_input_class_exemplars(input_features: np.ndarray, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank one class in frozen preprocessing-output space, before model LayerNorm.

    Reductions use sample-ID order so source row order cannot affect the mean or
    ranking. This is descriptor-space selection only: no latent, UE or RNG.
    """
    values = np.asarray(input_features)
    ids = np.asarray(sample_ids)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or values.dtype.kind not in "fiu":
        raise ValueError("Ranking requires a nonempty numeric preprocessing-output matrix.")
    if (ids.shape != (len(values),) or not all(isinstance(s, (str, np.str_)) and s for s in ids)
            or len(set(ids.tolist())) != len(ids)):
        raise ValueError("Ranking sample IDs must be unique nonempty strings aligned to rows.")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Ranking preprocessing outputs must be finite; nonfinite policy=error.")
    canonical = np.argsort(ids, kind="stable")
    x = values[canonical]
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(x - x.mean(axis=0, keepdims=True), axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("Ranking class-mean distance overflowed float64.")
    order = np.lexsort((ids[canonical], distances))
    return canonical[order], distances[order]


def _copy_arrays(arrays: DDIBatchArrays, indices) -> DDIBatchArrays:
    # copy even for slices: trimmed entries must not retain discarded backing arrays.
    return DDIBatchArrays(arrays.features[indices].copy(), arrays.labels[indices].copy(),
                          {k: v[indices].copy() for k, v in arrays.metadata.items()})


class FoldSqrtReplayBuffer:
    """Raw-only, provenance-checked buffer; update once per new CIL task.

    Context shares validated ID/label metadata, not old feature matrices. Caller
    must supply RAW current descriptors from the loader; transformed input cannot
    be inferred from array values. No readback of discarded old features occurs.
    """

    def __init__(self, *, context: DevelopmentFoldContext, task_file: str | Path,
                 feature_columns: Sequence[str], member_id: int, total_memory_budget: int,
                 base_quota: int = 10, experiment_seed: int = 0,
                 buffer_policy: str = BUFFER_POLICY,
                 ranking_policy: str = RANKING_POLICY,
                 ranking_preprocessing: FoldPreprocessing | None = None,
                 ranking_preprocessing_sha256: str | None = None) -> None:
        context.assert_unchanged()
        self._member_id = _integer(member_id, "member_id")
        if self._member_id not in (0, 1, 2):
            raise ValueError("member_id must be 0, 1 or 2.")
        self._budget = _integer(total_memory_budget, "total_memory_budget")
        self._base_quota = _integer(base_quota, "base_quota")
        if buffer_policy not in BUFFER_POLICIES:
            raise ValueError(f"buffer_policy must be one of {BUFFER_POLICIES}.")
        self._buffer_policy = buffer_policy
        seed = _integer(experiment_seed, "experiment_seed")
        if seed >= 2**32:
            raise ValueError("experiment_seed must fit uint32.")
        self._columns = tuple(feature_columns)
        if not self._columns or not all(isinstance(c, str) and c for c in self._columns) or len(set(self._columns)) != len(self._columns):
            raise ValueError("feature_columns must contain unique descriptor names.")
        if set(self._columns) & {*DEFAULT_META_COLS, context.label_col, *META_KEYS, "rank_priority", "rank_distance"}:
            raise ValueError("Labels and source/ID metadata cannot be descriptor columns.")
        if ranking_policy not in RANKING_POLICIES:
            raise ValueError(f"ranking_policy must be one of {RANKING_POLICIES}.")
        self._ranking_policy = ranking_policy
        self._ranking_preprocessing = ranking_preprocessing
        if ranking_policy == SAMPLE_NORMALIZED_RANKING_POLICY:
            if ranking_preprocessing is not None or ranking_preprocessing_sha256 is not None:
                raise ValueError("Sample-normalized ranking must not depend on a fitted preprocessing artifact.")
            ranking = dict(SAMPLE_NORMALIZED_RANKING_CONVENTIONS)
        else:
            if ranking_preprocessing is None or ranking_preprocessing_sha256 is None:
                raise ValueError("Pipeline-input ranking requires the frozen preprocessing object and SHA256.")
            prep_metadata = ranking_preprocessing.metadata
            if prep_metadata["policy"] != "task0_standard_frozen":
                raise ValueError("Pipeline-input ranking requires the approved task0_standard_frozen preprocessing.")
            if (not isinstance(ranking_preprocessing_sha256, str)
                    or len(ranking_preprocessing_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in ranking_preprocessing_sha256)):
                raise ValueError("ranking_preprocessing_sha256 must be a lowercase SHA256 digest.")
            provenance = prep_metadata["provenance"]
            if (provenance["seeds"]["member_id"] != self._member_id
                    or tuple(provenance["feature_columns"]) != self._columns):
                raise ValueError("Ranking preprocessing member/feature-order provenance mismatch.")
            ranking = {
                "policy": PIPELINE_INPUT_RANKING_POLICY,
                "policy_version": 1,
                "feature_space": "frozen_preprocessing_output_before_model_layernorm",
                "preprocessing_policy": prep_metadata["policy"],
                "preprocessing_sha256": ranking_preprocessing_sha256,
                "dtype": "float64", "distance": "euclidean",
                "tie_break": "sample_id_lexicographic", "reduction_order": "sample_id_ascending",
                "learned_latent": False, "uses_ue": False, "nonfinite_policy": "error",
                "status": "exemplar_ranking_pilot_candidate",
            }
        self._task_path = Path(task_file)
        content = self._task_path.read_bytes()
        spec = json.loads(content)
        protocol = spec.get("protocol") if isinstance(spec, dict) else None
        if (protocol not in SUPPORTED_PROTOCOLS
                or spec.get("seed") != SUPPORTED_PROTOCOLS.get(protocol)
                or not isinstance(spec.get("tasks"), list) or not spec["tasks"]):
            raise ValueError("Expected a supported P3 tail_to_head or P4 constrained_mass_balanced task file.")
        self._tasks = []
        for i, task in enumerate(spec["tasks"]):
            if not isinstance(task, dict) or type(task.get("task_id")) is not int or task["task_id"] != i or not isinstance(task.get("classes"), list) or not task["classes"]:
                raise ValueError("Tasks must be contiguous from 0 with nonempty raw class lists.")
            self._tasks.append(tuple(_class_id(c) for c in task["classes"]))
        all_classes = [c for group in self._tasks for c in group]
        if len(set(all_classes)) != len(all_classes):
            raise ValueError("Task file contains duplicate classes.")
        self._context = context
        manifest = context.manifest
        self._metadata = {
            "policy": self._buffer_policy, "schema_version": 1, "base_quota": self._base_quota,
            "total_memory_budget": self._budget, "ranking": ranking,
            "storage": "retained_raw_float64_only", "class_order": "raw_class_id_ascending",
            "allocation_rounding": (
                "capacity_constrained_equal_class_max_min_raw_id_tie"
                if self._buffer_policy == EQUAL_CLASS_BUFFER_POLICY else
                "clipped_base_max_min_then_capped_sqrt_largest_remainder_raw_id_tie"
            ),
            "seeds": asdict(resolve_seed_configuration(seed, self._member_id)),
            "validation_fold": self._member_id, "fold_seed": manifest["fold_seed"],
            "assignment_sha256": manifest["assignment_sha256"], "fold_manifest_sha256": context.manifest_sha256,
            "sources": {s: {k: v[k] for k in ("sha256", "row_count")} for s, v in manifest["sources"].items()},
            "task_file_sha256": hashlib.sha256(content).hexdigest(),
            "feature_columns": list(self._columns), "feature_order_sha256": _digest(list(self._columns)),
        }
        if self._ranking_policy == PIPELINE_INPUT_RANKING_POLICY:
            provenance = self._ranking_preprocessing.metadata["provenance"]
            for key in ("seeds", "validation_fold", "fold_seed", "assignment_sha256",
                        "fold_manifest_sha256", "sources", "task_file_sha256",
                        "feature_columns", "feature_order_sha256"):
                if provenance[key] != self._metadata[key]:
                    raise ValueError(f"Ranking preprocessing provenance mismatch: {key}.")
        self._completed_task_id = -1
        self._observed: dict[int, int] = {}
        self._entries: dict[int, DDIBatchArrays] = {}
        self._history: list[dict] = []

    def _rank_new_class(self, raw_features: np.ndarray, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._ranking_policy == SAMPLE_NORMALIZED_RANKING_POLICY:
            return rank_raw_class_exemplars(raw_features, sample_ids)
        preprocessing = self._ranking_preprocessing
        if preprocessing is None:  # construction validates this; defensive guard for corrupted objects
            raise RuntimeError("Missing frozen preprocessing for pipeline-input ranking.")
        values = preprocessing.transform(
            raw_features, feature_columns=list(self._columns), member_id=self._member_id,
            policy="task0_standard_frozen",
        )
        return rank_input_class_exemplars(values, sample_ids)

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    @property
    def observed_counts(self) -> dict[int, int]:
        return dict(self._observed)

    @property
    def memory_counts(self) -> dict[int, int]:
        return {c: len(a.labels) for c, a in sorted(self._entries.items())}

    @property
    def total_size(self) -> int:
        return sum(self.memory_counts.values())

    @property
    def next_task_id(self) -> int:
        return self._completed_task_id + 1

    @property
    def audit_history(self) -> list[dict]:
        return deepcopy(self._history)

    def _check_sources(self) -> None:
        self._context.assert_unchanged()
        if hashlib.sha256(self._task_path.read_bytes()).hexdigest() != self._metadata["task_file_sha256"]:
            raise ValueError("Task-file hash changed after buffer construction.")

    def _expected_rows(self, classes) -> dict[str, tuple]:
        expected = {}
        for table in self._context._rows:
            keep = (table["fold_id"].to_numpy() != self._member_id) & np.isin(table["raw_class_id"].to_numpy(), classes)
            for i in np.flatnonzero(keep):
                row = tuple(table[k][int(i)].as_py() for k in ("sample_id", "source_split", "source_row_index", "raw_class_id", "fold_id"))
                expected[row[0]] = row[1:]
        return expected

    def _validate_rows(self, arrays: DDIBatchArrays, expected: dict, *, full: bool) -> None:
        raw, labels, meta = np.asarray(arrays.features), np.asarray(arrays.labels), arrays.metadata
        if (raw.ndim != 2 or raw.shape[1] != len(self._columns) or raw.dtype.kind not in "fiu"
                or not np.isfinite(raw).all() or labels.shape != (len(raw),) or labels.dtype.kind not in "iu"):
            raise ValueError("Raw descriptors/labels must be finite, numeric, aligned and have expected width.")
        if meta is None or not set(META_KEYS) <= set(meta) or any(np.asarray(meta[k]).shape != (len(raw),) for k in META_KEYS):
            raise ValueError("Missing or misaligned sample/source provenance.")
        for key in ("source_row_index", "fold_id"):
            if np.asarray(meta[key]).dtype.kind not in "iu":
                raise ValueError(f"{key} must contain integer provenance.")
        ids = build_stable_sample_ids(meta, len(raw))
        if not np.array_equal(ids, meta["sample_id"]):
            raise ValueError("Sample IDs disagree with drug IDs.")
        for i, sample_id in enumerate(ids):
            actual = (meta["source_split"][i], int(meta["source_row_index"][i]), int(labels[i]), int(meta["fold_id"][i]))
            if sample_id not in expected or actual != expected[sample_id]:
                raise ValueError("Exemplar provenance mismatch: held-out/test/future/wrong source row or label.")
        if full and set(ids) != set(expected):
            raise ValueError("Current task must cover ALL member training rows, exactly once.")

    def update(self, current: DDIBatchArrays, *, task_id: int, feature_columns: Sequence[str]) -> dict:
        """Validate complete current-task data; rank new classes; trim old prefixes.

        Transactional: no mutation until every class/row and numeric check passes.
        Observed weights are recorded at arrival, never queried for future classes.
        """
        self._check_sources()
        task_id = _integer(task_id, "task_id")
        if task_id != self.next_task_id or task_id >= len(self._tasks):
            raise ValueError("Update must use the next task exactly once; old classes cannot be reintroduced.")
        if tuple(feature_columns) != self._columns:
            raise ValueError("Buffer feature order mismatch.")
        classes = self._tasks[task_id]
        expected = self._expected_rows(classes)
        self._validate_rows(current, expected, full=True)
        if set(np.unique(current.labels).tolist()) != set(classes):
            raise ValueError("Every arriving class must have member training samples.")
        observed = dict(self._observed)
        capacities = self.memory_counts
        for c in classes:
            observed[c] = int(np.sum(current.labels == c))
            capacities[c] = observed[c]
        allocation = (
            equal_class_capacity_allocation(capacities, self._budget)
            if self._buffer_policy == EQUAL_CLASS_BUFFER_POLICY else
            min_quota_sqrt_allocation(observed, capacities, self._budget, base_quota=self._base_quota)
        )
        updated = {}
        for c in sorted(observed):
            quota = allocation[c]
            if c in self._entries:
                updated[c] = _copy_arrays(self._entries[c], slice(0, quota))
            else:
                indices = np.flatnonzero(current.labels == c)
                rank, distances = self._rank_new_class(
                    current.features[indices], current.metadata["sample_id"][indices]
                )
                selected = indices[rank[:quota]]
                updated[c] = DDIBatchArrays(
                    np.asarray(current.features[selected], dtype=np.float64).copy(), current.labels[selected].astype(np.int64),
                    {k: np.asarray(current.metadata[k])[selected].copy() for k in META_KEYS},
                )
                updated[c].metadata.update(rank_priority=np.arange(quota, dtype=np.int64), rank_distance=distances[:quota].copy())
        audit = {"task_id": task_id, "policy": self._buffer_policy, "observed_counts": observed,
                 "feasible_capacities": capacities, "allocation": allocation,
                 "stored_slots": sum(allocation.values()), "budget": self._budget}
        self._check_sources()
        self._entries, self._observed, self._completed_task_id = updated, observed, task_id
        self._history.append(deepcopy(audit))
        return deepcopy(audit)

    def get_all(self) -> DDIBatchArrays:
        """Raw arrays in sorted class / original rank order, defensive copies."""
        parts = [self._entries[c] for c in sorted(self._entries)]
        if not parts:
            meta = {k: np.empty(0, dtype=np.int64 if k in ("source_row_index", "fold_id", "rank_priority") else np.float64 if k == "rank_distance" else str)
                    for k in (*META_KEYS, "rank_priority", "rank_distance")}
            return DDIBatchArrays(np.empty((0, len(self._columns)), dtype=np.float64), np.empty(0, dtype=np.int64), meta)
        return DDIBatchArrays(np.concatenate([p.features for p in parts]), np.concatenate([p.labels for p in parts]),
                              {k: np.concatenate([p.metadata[k] for p in parts]) for k in parts[0].metadata})

    def model_arrays(self, preprocessing: FoldPreprocessing) -> DDIBatchArrays:
        """Materialize model inputs without changing raw storage or selection IDs."""
        self._check_sources()
        preprocessing_metadata = preprocessing.metadata
        provenance = preprocessing_metadata["provenance"]
        for key in ("seeds", "validation_fold", "fold_seed", "assignment_sha256", "fold_manifest_sha256",
                    "sources", "task_file_sha256", "feature_columns", "feature_order_sha256"):
            if provenance[key] != self._metadata[key]:
                raise ValueError(f"Buffer/preprocessing provenance mismatch: {key}.")
        arrays = self.get_all()
        arrays.features = preprocessing.transform(arrays.features, feature_columns=list(self._columns),
                                                  member_id=self._member_id, policy=preprocessing_metadata["policy"])
        return arrays

    def state_dict(self) -> dict:
        """Snapshot of retained raw features only, suitable for a future checkpoint.

        This does not write a file or change any existing checkpoint format.
        No discarded raw features, normalization vectors or teacher are included.
        """
        state = {"metadata": self.metadata, "completed_task_id": self._completed_task_id,
                 "observed_counts": self.observed_counts, "audit_history": self.audit_history,
                 "entries": {c: {"features": a.features.copy(), "labels": a.labels.copy(),
                                  "metadata": {k: v.copy() for k, v in a.metadata.items()}}
                             for c, a in self._entries.items()}}
        state["state_sha256"] = _state_digest(state)
        return state

    @classmethod
    def from_state_dict(cls, state: dict, **kwargs) -> FoldSqrtReplayBuffer:
        """Rebind to newly validated context, check compatibility; do NOT rerank."""
        result = cls(**kwargs)
        try:
            if set(state) != {"metadata", "completed_task_id", "observed_counts", "audit_history", "entries", "state_sha256"}:
                raise ValueError("Unexpected buffer state fields.")
            if state.get("state_sha256") != _state_digest(state):
                raise ValueError("Buffer state SHA256 mismatch.")
            if state["metadata"] != result.metadata:
                raise ValueError("Buffer state policy/config/member/source/task metadata mismatch.")
            completed = state["completed_task_id"]
            if type(completed) is not int or not -1 <= completed < len(result._tasks):
                raise ValueError("Invalid completed_task_id in buffer state.")
            history = state["audit_history"]
            if len(history) != completed + 1:
                raise ValueError("Buffer audit task coverage mismatch.")
            observed, capacities = {}, {}
            for task_id in range(completed + 1):
                expected = result._expected_rows(result._tasks[task_id])
                for c in result._tasks[task_id]:
                    observed[c] = sum(row[2] == c for row in expected.values())
                    capacities[c] = observed[c]
                allocation = (
                    equal_class_capacity_allocation(capacities, result._budget)
                    if result._buffer_policy == EQUAL_CLASS_BUFFER_POLICY else
                    min_quota_sqrt_allocation(observed, capacities, result._budget, base_quota=result._base_quota)
                )
                audit = {"task_id": task_id, "policy": result._buffer_policy, "observed_counts": dict(observed),
                         "feasible_capacities": dict(capacities), "allocation": allocation,
                         "stored_slots": sum(allocation.values()), "budget": result._budget}
                if history[task_id] != audit:
                    raise ValueError("Buffer observed/capacity/allocation history mismatch.")
                capacities = dict(allocation)
            if state["observed_counts"] != observed or set(state["entries"]) != set(observed):
                raise ValueError("Buffer observed classes/counts mismatch.")
            for c, entry in state["entries"].items():
                if set(entry) != {"features", "labels", "metadata"} or set(entry["metadata"]) != {*META_KEYS, "rank_priority", "rank_distance"}:
                    raise ValueError("Unexpected retained entry fields.")
                arrays = DDIBatchArrays(entry["features"], entry["labels"], entry["metadata"])
                result._validate_rows(arrays, result._expected_rows([c]), full=False)
                if arrays.features.dtype != np.float64 or arrays.labels.dtype != np.int64 or len(arrays.labels) != capacities[c]:
                    raise ValueError("Retained feature dtype/count mismatch.")
                n = len(arrays.labels)
                rank, distances = arrays.metadata["rank_priority"], arrays.metadata["rank_distance"]
                if rank.dtype != np.int64 or not np.array_equal(rank, np.arange(n)) or distances.shape != (n,) or not np.isfinite(distances).all() or np.any(distances < 0):
                    raise ValueError("Invalid retained rank priorities/distances.")
                if not np.array_equal(np.lexsort((arrays.metadata["sample_id"], distances)), np.arange(n)):
                    raise ValueError("Retained rank order mismatch.")
                result._entries[c] = _copy_arrays(arrays, slice(None))
            result._observed, result._completed_task_id, result._history = dict(observed), completed, deepcopy(history)
            result._check_sources()
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Malformed fold replay buffer state.") from error
        return result


def _state_digest(state: dict) -> str:
    """Hash structural metadata and portable little-endian numeric array bytes."""
    digest = hashlib.sha256(_json({k: v for k, v in state.items() if k not in ("entries", "state_sha256")}).encode("utf-8"))
    for c in sorted(state["entries"]):
        digest.update(_json(c).encode())
        entry = state["entries"][c]
        for name, values in sorted({"features": entry["features"], "labels": entry["labels"], **entry["metadata"]}.items()):
            array = np.asarray(values)
            digest.update(_json([name, list(array.shape), array.dtype.kind, array.dtype.itemsize if array.dtype.kind not in "OUS" else None]).encode())
            if array.dtype.kind in "OUS":
                digest.update(_json(array.tolist()).encode("utf-8"))
            else:
                digest.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C"))
    return digest.hexdigest()


def ensemble_buffer_accounting(buffers: Sequence[FoldSqrtReplayBuffer], *, global_budget: int) -> dict:
    """Exactly three members: duplicate stored copies count against global budget."""
    global_budget = _integer(global_budget, "global_budget")
    if len(buffers) != 3 or {b._member_id for b in buffers} != {0, 1, 2}:
        raise ValueError("Accounting requires exactly members 0, 1, 2.")
    reference = buffers[0].metadata
    ids = {}
    for b in buffers:
        for key in ("policy", "ranking", "assignment_sha256", "fold_manifest_sha256", "sources", "task_file_sha256", "feature_order_sha256", "base_quota"):
            if b.metadata[key] != reference[key]:
                raise ValueError(f"Ensemble buffer metadata mismatch: {key}.")
        if b.next_task_id != buffers[0].next_task_id or b.metadata["seeds"]["experiment_seed"] != reference["seeds"]["experiment_seed"]:
            raise ValueError("Ensemble buffers must share task stage and experiment seed.")
        ids[b._member_id] = {sample_id for entry in b._entries.values() for sample_id in entry.metadata["sample_id"].tolist()}
    configured = sum(b._budget for b in buffers)
    slots = sum(b.total_size for b in buffers)
    if configured > global_budget or slots > configured:
        raise ValueError("Ensemble configured/stored slots exceed global budget.")
    unique = len(set().union(*ids.values()))
    return {"policy": reference["policy"], "global_budget": global_budget, "configured_slots": configured,
            "stored_slots": slots, "unique_sample_ids": unique, "duplicate_stored_copies": slots - unique,
            "per_member_slots": {b._member_id: b.total_size for b in buffers},
            "pairwise_overlap": {f"{a}_{b}": len(ids[a] & ids[b]) for a, b in ((0, 1), (0, 2), (1, 2))}}
