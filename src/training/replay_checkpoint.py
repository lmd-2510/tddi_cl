"""Versioned task-boundary checkpointing for fixed-budget replay distillation."""

from __future__ import annotations

import os
import uuid
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.data.class_mapping import validate_dense_class_map
from src.data.fixed_budget_replay import FixedBudgetReplayBuffer
from src.training.ewc_checkpoint import capture_rng_state, restore_rng_state


REPLAY_CHECKPOINT_SCHEMA_VERSION = 1
REPLAY_CHECKPOINT_KIND = "ddi_cil_replay_distill_task_boundary"
REPLAY_METHOD = "replay_distill_fixed_budget_uniform"
REPLAY_SEED_METADATA_KEYS = {
    "experiment_seed",
    "member_id",
    "member_seed",
    "member_seed_derivation",
    "seed_mode",
}


@dataclass(frozen=True)
class LoadedReplayCheckpoint:
    run_id: str
    completed_task_id: int
    next_task_id: int
    model_state: dict[str, torch.Tensor]
    seen_class_map: dict[int, int]
    seen_raw_classes: tuple[int, ...]
    replay_buffer_state: dict[str, Any]
    seed_metadata: dict[str, int | str | None]
    config: dict[str, Any]
    rng_state: dict[str, Any]
    progress: dict[str, Any]


def _cpu_tensor_mapping(
    state: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Replay {name} must be a non-empty tensor mapping.")
    copied: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"Replay {name} must map string names to tensors.")
        copied[key] = value.detach().cpu().clone()
    return copied


def _mapping_mismatches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    keys = sorted(set(expected) | set(actual))
    missing = object()
    return [key for key in keys if expected.get(key, missing) != actual.get(key, missing)]


def capture_fixed_replay_buffer(buffer: FixedBudgetReplayBuffer) -> dict[str, Any]:
    """Capture retained exemplars without changing their per-class order."""

    features, raw_labels = buffer.get_all()
    return {
        "total_memory_budget": int(buffer.total_memory_budget),
        "random_seed": int(buffer.random_seed),
        "features": np.asarray(features, dtype=np.float32).copy(),
        "raw_labels": np.asarray(raw_labels, dtype=np.int64).copy(),
        "available_count_by_class": {
            int(class_id): int(count)
            for class_id, count in buffer.available_count_by_class.items()
        },
    }


def _validate_buffer_state(
    state: Any,
    *,
    expected_seen_class_map: Mapping[int, int] | None = None,
    expected_total_memory_budget: int | None = None,
    expected_random_seed: int | None = None,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError("Replay replay_buffer_state must be a mapping.")
    required = {
        "total_memory_budget",
        "random_seed",
        "features",
        "raw_labels",
        "available_count_by_class",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Replay replay_buffer_state is missing keys: {missing}")

    budget = int(state["total_memory_budget"])
    random_seed = int(state["random_seed"])
    if budget <= 0:
        raise ValueError("Replay total_memory_budget must be positive.")
    if expected_total_memory_budget is not None and budget != int(expected_total_memory_budget):
        raise ValueError(
            "Replay memory budget mismatch: "
            f"{budget} != {int(expected_total_memory_budget)}."
        )
    if expected_random_seed is not None and random_seed != int(expected_random_seed):
        raise ValueError(
            f"Replay buffer seed mismatch: {random_seed} != {int(expected_random_seed)}."
        )

    features = np.asarray(state["features"])
    raw_labels = np.asarray(state["raw_labels"])
    if features.dtype != np.float32 or features.ndim != 2:
        raise TypeError("Replay retained features must be a two-dimensional float32 array.")
    if raw_labels.dtype != np.int64 or raw_labels.shape != (features.shape[0],):
        raise TypeError("Replay retained raw labels must be an aligned int64 array.")
    if features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("Replay retained features must be non-empty and finite.")

    raw_available = state["available_count_by_class"]
    if not isinstance(raw_available, Mapping) or not raw_available:
        raise ValueError("Replay available_count_by_class must be a non-empty mapping.")
    available = {int(class_id): int(count) for class_id, count in raw_available.items()}
    if any(count <= 0 for count in available.values()):
        raise ValueError("Replay available class counts must be positive.")
    retained_classes = {int(class_id) for class_id in np.unique(raw_labels)}
    if retained_classes != set(available):
        raise ValueError("Replay retained classes do not match available_count_by_class.")
    if expected_seen_class_map is not None and retained_classes != {
        int(class_id) for class_id in expected_seen_class_map
    }:
        raise ValueError("Replay buffer classes do not match seen_class_map.")
    for class_id, available_count in available.items():
        retained_count = int(np.count_nonzero(raw_labels == class_id))
        if retained_count <= 0 or retained_count > available_count:
            raise ValueError(
                f"Replay retained count is invalid for class {class_id}: "
                f"{retained_count} of {available_count}."
            )
    expected_size = min(budget, sum(available.values()))
    if features.shape[0] != expected_size:
        raise ValueError(
            "Replay retained size does not match its feasible fixed budget: "
            f"{features.shape[0]} != {expected_size}."
        )
    return {
        "total_memory_budget": budget,
        "random_seed": random_seed,
        "features": features.copy(),
        "raw_labels": raw_labels.copy(),
        "available_count_by_class": available,
    }


def restore_fixed_replay_buffer(
    state: Mapping[str, Any],
    *,
    expected_seen_class_map: Mapping[int, int] | None = None,
    expected_total_memory_budget: int | None = None,
    expected_random_seed: int | None = None,
) -> FixedBudgetReplayBuffer:
    """Rebuild an exact fixed-budget buffer from checkpoint state."""

    validated = _validate_buffer_state(
        state,
        expected_seen_class_map=expected_seen_class_map,
        expected_total_memory_budget=expected_total_memory_budget,
        expected_random_seed=expected_random_seed,
    )
    buffer = FixedBudgetReplayBuffer(
        total_memory_budget=validated["total_memory_budget"],
        random_seed=validated["random_seed"],
    )
    buffer.available_count_by_class = dict(validated["available_count_by_class"])
    features = validated["features"]
    raw_labels = validated["raw_labels"]
    buffer.features_by_class = {
        class_id: features[raw_labels == class_id].copy()
        for class_id in sorted(buffer.available_count_by_class)
    }
    return buffer


def _validate_payload(
    payload: Any,
    *,
    expected_seed_metadata: Mapping[str, int | str | None] | None,
    expected_config: Mapping[str, Any] | None,
    expected_seen_class_map: Mapping[int, int] | None,
) -> LoadedReplayCheckpoint:
    if not isinstance(payload, dict):
        raise TypeError("Replay checkpoint payload must be a dictionary.")
    required = {
        "schema_version",
        "kind",
        "run_id",
        "completed_task_id",
        "next_task_id",
        "model_state",
        "seen_class_map",
        "seen_raw_classes",
        "replay_buffer_state",
        "seed_metadata",
        "config",
        "rng_state",
        "progress",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Replay checkpoint is missing keys: {missing}")
    if payload["schema_version"] != REPLAY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported replay checkpoint schema version: "
            f"{payload['schema_version']} != {REPLAY_CHECKPOINT_SCHEMA_VERSION}."
        )
    if payload["kind"] != REPLAY_CHECKPOINT_KIND:
        raise ValueError(f"Unexpected replay checkpoint kind: {payload['kind']!r}.")

    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Replay checkpoint run_id must be a non-empty string.")
    completed_task_id = int(payload["completed_task_id"])
    next_task_id = int(payload["next_task_id"])
    if completed_task_id < 0 or next_task_id != completed_task_id + 1:
        raise ValueError("Replay next_task_id must equal completed_task_id + 1.")
    model_state = _cpu_tensor_mapping(payload["model_state"], name="model_state")

    raw_class_map = payload["seen_class_map"]
    if not isinstance(raw_class_map, Mapping) or not raw_class_map:
        raise ValueError("Replay seen_class_map must be a non-empty mapping.")
    seen_class_map = {int(raw): int(index) for raw, index in raw_class_map.items()}
    validate_dense_class_map(seen_class_map)
    if expected_seen_class_map is not None:
        normalized_expected = {int(raw): int(index) for raw, index in expected_seen_class_map.items()}
        if seen_class_map != normalized_expected:
            raise ValueError("Replay seen_class_map does not match expected task classes.")

    raw_order = payload["seen_raw_classes"]
    if not isinstance(raw_order, (list, tuple)):
        raise TypeError("Replay seen_raw_classes must be a sequence.")
    seen_raw_classes = tuple(int(class_id) for class_id in raw_order)
    expected_order = tuple(raw for raw, _ in sorted(seen_class_map.items(), key=lambda item: item[1]))
    if seen_raw_classes != expected_order:
        raise ValueError("Replay raw class order does not match seen_class_map columns.")

    seed_metadata = payload["seed_metadata"]
    config = payload["config"]
    progress = payload["progress"]
    if not isinstance(seed_metadata, dict):
        raise TypeError("Replay seed_metadata must be a dictionary.")
    missing_seed_keys = sorted(REPLAY_SEED_METADATA_KEYS - set(seed_metadata))
    if missing_seed_keys:
        raise ValueError(f"Replay seed_metadata is missing keys: {missing_seed_keys}")
    if not isinstance(config, dict) or not config:
        raise ValueError("Replay config must be a non-empty dictionary.")
    if config.get("method") != REPLAY_METHOD:
        raise ValueError(f"Replay checkpoint method must be {REPLAY_METHOD!r}.")
    if not isinstance(progress, dict):
        raise TypeError("Replay progress must be a dictionary.")
    if expected_seed_metadata is not None:
        mismatches = _mapping_mismatches(expected_seed_metadata, seed_metadata)
        if mismatches:
            raise ValueError(f"Replay seed metadata mismatch for keys: {mismatches}")
    if expected_config is not None:
        mismatches = _mapping_mismatches(expected_config, config)
        if mismatches:
            raise ValueError(f"Replay resume config mismatch for keys: {mismatches}")

    replay_buffer_state = _validate_buffer_state(
        payload["replay_buffer_state"],
        expected_seen_class_map=seen_class_map,
        expected_total_memory_budget=config.get("total_memory_budget"),
        expected_random_seed=seed_metadata.get("experiment_seed"),
    )
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, dict):
        raise TypeError("Replay rng_state must be a dictionary.")
    live_rng = capture_rng_state()
    try:
        restore_rng_state(rng_state)
    finally:
        restore_rng_state(live_rng)

    return LoadedReplayCheckpoint(
        run_id=run_id,
        completed_task_id=completed_task_id,
        next_task_id=next_task_id,
        model_state=model_state,
        seen_class_map=seen_class_map,
        seen_raw_classes=seen_raw_classes,
        replay_buffer_state=replay_buffer_state,
        seed_metadata=dict(seed_metadata),
        config=dict(config),
        rng_state=dict(rng_state),
        progress=dict(progress),
    )


def save_replay_checkpoint(
    path: str | Path,
    *,
    run_id: str,
    completed_task_id: int,
    model_state: Mapping[str, torch.Tensor],
    seen_class_map: Mapping[int, int],
    seen_raw_classes: list[int] | tuple[int, ...],
    replay_buffer: FixedBudgetReplayBuffer,
    seed_metadata: Mapping[str, int | str | None],
    config: Mapping[str, Any],
    progress: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically publish the latest complete fixed-replay task state."""

    path = Path(path)
    if not run_id:
        raise ValueError("Replay checkpoint run_id must be non-empty.")
    completed_task_id = int(completed_task_id)
    if completed_task_id < 0:
        raise ValueError("Replay completed_task_id must be non-negative.")
    missing_seed_keys = sorted(REPLAY_SEED_METADATA_KEYS - set(seed_metadata))
    if missing_seed_keys:
        raise ValueError(f"Replay seed_metadata is missing keys: {missing_seed_keys}")
    if not config or config.get("method") != REPLAY_METHOD:
        raise ValueError(f"Replay config method must be {REPLAY_METHOD!r}.")
    normalized_class_map = {int(raw): int(index) for raw, index in seen_class_map.items()}
    validate_dense_class_map(normalized_class_map)
    expected_order = tuple(raw for raw, _ in sorted(normalized_class_map.items(), key=lambda item: item[1]))
    normalized_order = tuple(int(class_id) for class_id in seen_raw_classes)
    if normalized_order != expected_order:
        raise ValueError("Replay raw class order does not match seen_class_map columns.")

    replay_buffer_state = _validate_buffer_state(
        capture_fixed_replay_buffer(replay_buffer),
        expected_seen_class_map=normalized_class_map,
        expected_total_memory_budget=config.get("total_memory_budget"),
        expected_random_seed=seed_metadata.get("experiment_seed"),
    )
    payload = {
        "schema_version": REPLAY_CHECKPOINT_SCHEMA_VERSION,
        "kind": REPLAY_CHECKPOINT_KIND,
        "run_id": run_id,
        "completed_task_id": completed_task_id,
        "next_task_id": completed_task_id + 1,
        "model_state": _cpu_tensor_mapping(model_state, name="model_state"),
        "seen_class_map": normalized_class_map,
        "seen_raw_classes": list(normalized_order),
        "replay_buffer_state": replay_buffer_state,
        "seed_metadata": dict(seed_metadata),
        "config": dict(config),
        "rng_state": capture_rng_state(),
        "progress": dict(progress or {}),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_replay_checkpoint(
    path: str | Path,
    *,
    expected_seed_metadata: Mapping[str, int | str | None] | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_seen_class_map: Mapping[int, int] | None = None,
) -> LoadedReplayCheckpoint:
    """Load and validate one fixed-replay task-boundary checkpoint on CPU."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing replay resume checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Could not load replay checkpoint {path}: {exc}") from exc
    return _validate_payload(
        payload,
        expected_seed_metadata=expected_seed_metadata,
        expected_config=expected_config,
        expected_seen_class_map=expected_seen_class_map,
    )


def restore_replay_model(
    model: torch.nn.Module,
    checkpoint: LoadedReplayCheckpoint,
) -> torch.nn.Module:
    """Restore the boundary model; callers may clone/freeze it as the teacher."""

    model.load_state_dict(checkpoint.model_state, strict=True)
    return model


# Separate kind and entrypoints: do not weaken/change the legacy schema above.
FOLD_REPLAY_CHECKPOINT_KIND = "ddi_cil_frozen_fold_replay_task_boundary"
FOLD_REPLAY_CHECKPOINT_VERSION = 1


def fold_state_digest(value: Any) -> str:
    """Content integrity for nested CPU tensors/NumPy/RNG/JSON state, not pickle bytes."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            visit(("tensor", str(item.dtype), tuple(item.shape)))
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            visit(("array", item.dtype.str, tuple(item.shape)))
            if item.dtype.kind in "OUS":
                visit(item.tolist())
            else:
                digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping[")
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        elif isinstance(item, np.generic):
            visit(item.item())
        else:
            digest.update((type(item).__name__ + ":" + json.dumps(item, allow_nan=False) + ";").encode())
    visit(value)
    return digest.hexdigest()


def atomic_publish_fold_file(path: str | Path, writer) -> Path:
    """Fsync then publish by hard link (no replace), same convention as fold artifacts.

    Only our private temporary file is removed. Existing user files are never
    replaced; unsupported filesystems fail rather than falling back to overwrite.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def fold_artifact_inventory(root: Path, directory: Path) -> dict[str, str]:
    from src.data.stratified_folds import fold_file_sha256
    return {p.relative_to(root).as_posix(): fold_file_sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def validate_fold_artifacts(root: Path, inventory: Mapping[str, str]) -> None:
    from src.data.stratified_folds import fold_file_sha256
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("Frozen-fold checkpoint requires completed artifact inventory.")
    root = root.resolve()
    for relative, expected_hash in inventory.items():
        path = (root / relative).resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError("Frozen-fold artifact path escapes run directory.")
        if not path.is_file() or fold_file_sha256(path) != expected_hash:
            raise ValueError(f"Frozen-fold completed artifact missing/hash mismatch: {relative}")


def save_fold_replay_checkpoint(path: str | Path, *, run_id: str, completed_task_id: int,
        model_state: Mapping[str, torch.Tensor], seen_class_map: Mapping[int, int],
        contract: dict, buffer, sampler, progress: dict, artifact_hashes: dict,
        run_config_sha256: str) -> Path:
    """Commit a task AFTER best-model evaluation/buffer update and artifact publication."""
    payload = {
        "kind": FOLD_REPLAY_CHECKPOINT_KIND, "schema_version": FOLD_REPLAY_CHECKPOINT_VERSION,
        "run_id": run_id, "completed_task_id": completed_task_id, "next_task_id": completed_task_id + 1,
        "full_trajectory_complete": completed_task_id == 7,
        "model_state": _cpu_tensor_mapping(model_state, name="fold model_state"),
        "seen_class_map": dict(seen_class_map),
        "raw_class_order": [raw for raw, _ in sorted(seen_class_map.items(), key=lambda p: p[1])],
        "contract": contract, "buffer_state": buffer.state_dict(), "sampler_state": sampler.state_dict(),
        "rng_state": capture_rng_state(), "progress": progress, "artifact_hashes": artifact_hashes,
        "run_config_sha256": run_config_sha256,
        "next_task_scheduling": {"task_id": completed_task_id + 1, "epoch": 0,
            "policy": "reset_from_new_current_and_retained_IDs_with_member_task_key",
            "loader_seed": contract["seeds"]["member_seed"] + completed_task_id + 1},
    }
    payload["state_sha256"] = fold_state_digest(payload)
    return atomic_publish_fold_file(path, lambda handle: torch.save(payload, handle))


def load_fold_replay_checkpoint(path: str | Path, *, root: Path, expected_contract: dict,
        tasks: list[dict], context, buffer_kwargs: dict) -> tuple[dict, Any]:
    """Validate against fresh sources/preprocessing; restore retained rows, never refit.

    Local torch checkpoints must be trusted: like the legacy API they contain
    NumPy/Python RNG state and are loaded with weights_only=False.
    """
    from src.data.fold_replay_buffer import FoldSqrtReplayBuffer
    from src.data.fold_replay_sampler import FoldReplayFractionSampler
    from src.data.stratified_folds import fold_file_sha256
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Cannot read frozen-fold checkpoint: {path}") from error
    if not isinstance(state, dict) or state.get("kind") != FOLD_REPLAY_CHECKPOINT_KIND:
        raise ValueError("Not a frozen-fold task-boundary checkpoint (legacy/model-only not accepted).")
    if state.get("schema_version") != FOLD_REPLAY_CHECKPOINT_VERSION:
        raise ValueError("Unsupported frozen-fold checkpoint schema version.")
    try:
        required = {"kind", "schema_version", "run_id", "completed_task_id", "next_task_id",
            "full_trajectory_complete", "model_state", "seen_class_map", "raw_class_order",
            "contract", "buffer_state", "sampler_state", "rng_state", "progress", "artifact_hashes",
            "run_config_sha256", "next_task_scheduling", "state_sha256"}
        if set(state) != required:
            raise ValueError("Unexpected/missing frozen-fold checkpoint fields.")
        if state["state_sha256"] != fold_state_digest({k: v for k, v in state.items() if k != "state_sha256"}):
            raise ValueError("Frozen-fold checkpoint state SHA256 mismatch.")
        differences = _mapping_mismatches(expected_contract, state["contract"])
        if differences:
            raise ValueError(f"Frozen-fold contract mismatch: {differences}")
        completed = state["completed_task_id"]
        if (type(completed) is not int or not 0 <= completed < len(tasks)
                or type(state["next_task_id"]) is not int or state["next_task_id"] != completed + 1
                or state["full_trajectory_complete"] != (completed == 7)):
            raise ValueError("Invalid frozen-fold completed/next task or full-trajectory metadata.")
        if path != root / "checkpoints" / f"task_{completed}.pt":
            raise ValueError("Checkpoint must belong to the requested run's checkpoints/task_N.pt.")
        if any((root / "checkpoints" / f"task_{i}.pt").exists() for i in range(completed + 1, len(tasks))):
            raise ValueError("A newer task checkpoint exists; resume the latest boundary, not an older task.")
        classes = sorted({raw for task in tasks[:completed + 1] for raw in task["classes"]})
        expected_map = {raw: i for i, raw in enumerate(classes)}
        if (state["seen_class_map"] != expected_map or state["raw_class_order"] != classes
                or any(type(k) is not int or type(v) is not int for k, v in state["seen_class_map"].items())):
            raise ValueError("Frozen-fold class map/raw class order mismatch.")
        model = state["model_state"]
        if (model["head.weight"].ndim != 2 or model["head.weight"].shape[0] != len(classes)
                or tuple(model["head.bias"].shape) != (len(classes),)
                or not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in model.values())):
            raise ValueError("Frozen-fold model/head shape or finite-state mismatch.")
        config_path = root / "run_config.json"
        if fold_file_sha256(config_path) != state["run_config_sha256"]:
            raise ValueError("Frozen-fold run_config hash mismatch.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["run_id"] != state["run_id"] or config["checkpoint_contract"] != expected_contract:
            raise ValueError("Frozen-fold checkpoint run_id/config mismatch.")
        validate_fold_artifacts(root, state["artifact_hashes"])
        progress = state["progress"]
        summaries = progress["task_summaries"]
        if (set(progress) != {"task_summaries", "epoch_rows", "metric_rows"}
                or any(r["task"] not in range(completed + 1) for r in progress["epoch_rows"] + progress["metric_rows"])):
            raise ValueError("Frozen-fold progress contains unknown/future rows.")
        if [s["task_id"] for s in summaries] != list(range(completed + 1)):
            raise ValueError("Frozen-fold progress task coverage mismatch.")
        for task_id, summary in enumerate(summaries):
            task_dir = (root / summary["artifact_directory"]).resolve()
            if not task_dir.is_relative_to(root):
                raise ValueError("Task artifact directory escapes run root.")
            epoch_count = summary["epochs_trained"]
            if type(epoch_count) is not int or not 1 <= epoch_count <= expected_contract["hyperparameters"]["epochs"]:
                raise ValueError("Invalid completed epoch count.")
            required_files = {"completed_task.json", "best_model.pt", "input_audit.json", "buffer_audit.json",
                              "metrics.json", "metrics.csv", "training_audit.csv"}
            required_files.update(f"epoch_{e}_audit.json" for e in range(1, epoch_count + 1))
            if any((task_dir / f).relative_to(root).as_posix() not in state["artifact_hashes"] for f in required_files):
                raise ValueError("Missing task completion evidence in checkpoint inventory.")
            disk_summary = json.loads((task_dir / "completed_task.json").read_text(encoding="utf-8"))
            if disk_summary != summary:
                raise ValueError("Task completion summary/progress mismatch.")
            epoch_rows = [r for r in progress["epoch_rows"] if r["task"] == task_id]
            if [r["epoch"] for r in epoch_rows] != list(range(1, summary["epochs_trained"] + 1)):
                raise ValueError("Frozen-fold epoch progress mismatch.")
            for row in epoch_rows:
                disk_row = json.loads((task_dir / f"epoch_{row['epoch']}_audit.json").read_text(encoding="utf-8"))
                if any(disk_row[k] != v for k, v in row.items()):
                    raise ValueError("Frozen-fold training audit/progress mismatch.")
            metric_rows = [r for r in progress["metric_rows"] if r["task"] == task_id]
            if json.loads((task_dir / "metrics.json").read_text(encoding="utf-8")) != metric_rows:
                raise ValueError("Frozen-fold metric progress mismatch.")
        restored = FoldSqrtReplayBuffer.from_state_dict(state["buffer_state"], **buffer_kwargs)
        if restored.next_task_id != completed + 1 or set(restored.observed_counts) != set(classes):
            raise ValueError("Frozen-fold buffer task/classes mismatch.")
        last_dir = root / summaries[-1]["artifact_directory"]
        best = torch.load(last_dir / "best_model.pt", map_location="cpu", weights_only=True)
        if (best["seen_class_map"] != expected_map or set(best["model_state"]) != set(model)
                or any(not torch.equal(model[k], best["model_state"][k]) for k in model)):
            raise ValueError("Boundary model must equal completed task's best model.")
        # Validate completed scheduling from identity-only input audit; no old features.
        audit = json.loads((last_dir / "input_audit.json").read_text(encoding="utf-8"))
        kwargs = dict(current_sample_ids=audit["current_ids"], replay_sample_ids=audit["replay_ids_before"],
            replay_raw_labels=audit["replay_raw_labels_before"], experiment_seed=expected_contract["seeds"]["experiment_seed"],
            member_id=expected_contract["seeds"]["member_id"], task_id=completed)
        sampler = FoldReplayFractionSampler.from_state_dict(state["sampler_state"], **kwargs)
        if sampler.state_dict()["next_epoch"] != summaries[-1]["epochs_trained"]:
            raise ValueError("Completed sampler epoch mismatch.")
        expected_schedule = {"task_id": completed + 1, "epoch": 0,
            "policy": "reset_from_new_current_and_retained_IDs_with_member_task_key",
            "loader_seed": expected_contract["seeds"]["member_seed"] + completed + 1}
        if state["next_task_scheduling"] != expected_schedule:
            raise ValueError("Next-task scheduling mismatch.")
        # Validate RNG without changing caller's streams or consuming randomness.
        before = capture_rng_state()
        try:
            restore_rng_state(state["rng_state"])
        finally:
            restore_rng_state(before)
        context.assert_unchanged()
    except (KeyError, TypeError, AttributeError, OSError) as error:
        raise ValueError(f"Malformed/incomplete frozen-fold checkpoint: {error}") from error
    return state, restored
