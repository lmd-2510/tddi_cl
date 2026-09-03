"""Versioned task-boundary checkpointing for fixed-budget replay distillation."""

from __future__ import annotations

import os
import uuid
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
