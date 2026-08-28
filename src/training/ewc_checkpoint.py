"""Versioned task-boundary checkpointing for resumable EWC runs."""

from __future__ import annotations

import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.data.class_mapping import validate_dense_class_map


EWC_CHECKPOINT_SCHEMA_VERSION = 1
EWC_CHECKPOINT_KIND = "ddi_cil_ewc_task_boundary"
EWC_SEED_METADATA_KEYS = {
    "experiment_seed",
    "member_id",
    "member_seed",
    "member_seed_derivation",
    "seed_mode",
}


@dataclass(frozen=True)
class LoadedEWCCheckpoint:
    run_id: str
    completed_task_id: int
    model_state: dict[str, torch.Tensor]
    fisher_total: dict[str, torch.Tensor]
    seen_class_map: dict[int, int]
    seed_metadata: dict[str, int | str | None]
    config: dict[str, Any]
    rng_state: dict[str, Any]
    progress: dict[str, Any]

    @property
    def next_task_id(self) -> int:
        return self.completed_task_id + 1


def capture_rng_state() -> dict[str, Any]:
    """Capture all RNG streams needed to continue at the next task boundary."""

    cuda_states = (
        [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().detach().cpu().clone(),
        "torch_cuda": cuda_states,
        "torch_cuda_device_count": len(cuda_states),
    }


def restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    required = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "torch_cuda_device_count",
    }
    missing = sorted(required - set(rng_state))
    if missing:
        raise ValueError(f"EWC RNG state is missing keys: {missing}")
    torch_cpu = rng_state["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8:
        raise TypeError("EWC torch_cpu RNG state must be a uint8 tensor.")
    cuda_states = rng_state["torch_cuda"]
    if not isinstance(cuda_states, list) or not all(
        isinstance(state, torch.Tensor) and state.dtype == torch.uint8
        for state in cuda_states
    ):
        raise TypeError("EWC torch_cuda RNG states must be a list of uint8 tensors.")
    declared_cuda_count = int(rng_state["torch_cuda_device_count"])
    if declared_cuda_count != len(cuda_states):
        raise ValueError("EWC CUDA RNG state count does not match its metadata.")
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore a CUDA EWC checkpoint without CUDA.")
        if torch.cuda.device_count() != declared_cuda_count:
            raise RuntimeError(
                "CUDA device count differs from the EWC checkpoint: "
                f"{torch.cuda.device_count()} != {declared_cuda_count}."
            )

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.random.set_rng_state(torch_cpu.cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _cpu_tensor_mapping(
    state: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"EWC {name} must be a non-empty tensor mapping.")
    copied: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"EWC {name} must map string names to tensors.")
        copied[key] = value.detach().cpu().clone()
    return copied


def _validate_model_and_fisher(
    model_state: Mapping[str, torch.Tensor],
    fisher_total: Mapping[str, torch.Tensor],
) -> None:
    missing_model_keys = sorted(set(fisher_total) - set(model_state))
    if missing_model_keys:
        raise ValueError(
            f"EWC Fisher contains names absent from model_state: {missing_model_keys[:5]}"
        )
    for name, fisher in fisher_total.items():
        model_value = model_state[name]
        if fisher.shape != model_value.shape:
            raise ValueError(
                f"EWC Fisher/model shape mismatch for {name}: "
                f"{tuple(fisher.shape)} != {tuple(model_value.shape)}."
            )
        if not fisher.is_floating_point():
            raise TypeError(f"EWC Fisher tensor must be floating point: {name}")
        if not torch.isfinite(fisher).all():
            raise ValueError(f"EWC Fisher contains non-finite values: {name}")
        if torch.any(fisher < 0):
            raise ValueError(f"EWC Fisher contains negative values: {name}")


def _mapping_mismatches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    keys = sorted(set(expected) | set(actual))
    missing = object()
    return [
        key
        for key in keys
        if expected.get(key, missing) != actual.get(key, missing)
    ]


def _validate_payload(
    payload: Any,
    *,
    expected_seed_metadata: Mapping[str, int | str | None] | None,
    expected_config: Mapping[str, Any] | None,
    expected_seen_class_map: Mapping[int, int] | None,
) -> LoadedEWCCheckpoint:
    if not isinstance(payload, dict):
        raise TypeError("EWC checkpoint payload must be a dictionary.")
    required = {
        "schema_version",
        "kind",
        "run_id",
        "completed_task_id",
        "model_state",
        "fisher_total",
        "seen_class_map",
        "seed_metadata",
        "config",
        "rng_state",
        "progress",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"EWC checkpoint is missing keys: {missing}")
    if payload["schema_version"] != EWC_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported EWC checkpoint schema version: "
            f"{payload['schema_version']} != {EWC_CHECKPOINT_SCHEMA_VERSION}."
        )
    if payload["kind"] != EWC_CHECKPOINT_KIND:
        raise ValueError(f"Unexpected EWC checkpoint kind: {payload['kind']!r}.")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("EWC checkpoint run_id must be a non-empty string.")
    completed_task_id = int(payload["completed_task_id"])
    if completed_task_id < 0:
        raise ValueError("EWC completed_task_id must be non-negative.")

    model_state = _cpu_tensor_mapping(payload["model_state"], name="model_state")
    fisher_total = _cpu_tensor_mapping(payload["fisher_total"], name="fisher_total")
    _validate_model_and_fisher(model_state, fisher_total)

    raw_class_map = payload["seen_class_map"]
    if not isinstance(raw_class_map, Mapping) or not raw_class_map:
        raise ValueError("EWC seen_class_map must be a non-empty mapping.")
    seen_class_map = {int(raw): int(index) for raw, index in raw_class_map.items()}
    validate_dense_class_map(seen_class_map)
    if expected_seen_class_map is not None:
        normalized_expected = {
            int(raw): int(index) for raw, index in expected_seen_class_map.items()
        }
        if seen_class_map != normalized_expected:
            raise ValueError(
                "EWC seen_class_map does not match the expected task classes."
            )

    seed_metadata = payload["seed_metadata"]
    config = payload["config"]
    progress = payload["progress"]
    if not isinstance(seed_metadata, dict):
        raise TypeError("EWC seed_metadata must be a dictionary.")
    missing_seed_keys = sorted(EWC_SEED_METADATA_KEYS - set(seed_metadata))
    if missing_seed_keys:
        raise ValueError(f"EWC seed_metadata is missing keys: {missing_seed_keys}")
    if not isinstance(config, dict) or not config:
        raise ValueError("EWC config must be a non-empty dictionary.")
    if not isinstance(progress, dict):
        raise TypeError("EWC progress must be a dictionary.")
    if expected_seed_metadata is not None:
        mismatches = _mapping_mismatches(expected_seed_metadata, seed_metadata)
        if mismatches:
            raise ValueError(f"EWC seed metadata mismatch for keys: {mismatches}")
    if expected_config is not None:
        mismatches = _mapping_mismatches(expected_config, config)
        if mismatches:
            raise ValueError(f"EWC resume config mismatch for keys: {mismatches}")

    rng_state = payload["rng_state"]
    if not isinstance(rng_state, dict):
        raise TypeError("EWC rng_state must be a dictionary.")
    # Validate without mutating the caller's live RNG streams.
    live_rng = capture_rng_state()
    try:
        restore_rng_state(rng_state)
    finally:
        restore_rng_state(live_rng)

    return LoadedEWCCheckpoint(
        run_id=run_id,
        completed_task_id=completed_task_id,
        model_state=model_state,
        fisher_total=fisher_total,
        seen_class_map=seen_class_map,
        seed_metadata=dict(seed_metadata),
        config=dict(config),
        rng_state=dict(rng_state),
        progress=dict(progress),
    )


def save_ewc_checkpoint(
    path: str | Path,
    *,
    run_id: str,
    completed_task_id: int,
    model_state: Mapping[str, torch.Tensor],
    fisher_total: Mapping[str, torch.Tensor],
    seen_class_map: Mapping[int, int],
    seed_metadata: Mapping[str, int | str | None],
    config: Mapping[str, Any],
    progress: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically publish the latest complete EWC task-boundary state."""

    path = Path(path)
    if not run_id:
        raise ValueError("EWC checkpoint run_id must be non-empty.")
    if int(completed_task_id) < 0:
        raise ValueError("EWC completed_task_id must be non-negative.")
    missing_seed_keys = sorted(EWC_SEED_METADATA_KEYS - set(seed_metadata))
    if missing_seed_keys:
        raise ValueError(f"EWC seed_metadata is missing keys: {missing_seed_keys}")
    if not config:
        raise ValueError("EWC config must be non-empty.")
    normalized_class_map = {
        int(raw): int(index) for raw, index in seen_class_map.items()
    }
    validate_dense_class_map(normalized_class_map)
    cpu_model_state = _cpu_tensor_mapping(model_state, name="model_state")
    cpu_fisher = _cpu_tensor_mapping(fisher_total, name="fisher_total")
    _validate_model_and_fisher(cpu_model_state, cpu_fisher)
    payload = {
        "schema_version": EWC_CHECKPOINT_SCHEMA_VERSION,
        "kind": EWC_CHECKPOINT_KIND,
        "run_id": run_id,
        "completed_task_id": int(completed_task_id),
        "model_state": cpu_model_state,
        "fisher_total": cpu_fisher,
        "seen_class_map": normalized_class_map,
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


def load_ewc_checkpoint(
    path: str | Path,
    *,
    expected_seed_metadata: Mapping[str, int | str | None] | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_seen_class_map: Mapping[int, int] | None = None,
) -> LoadedEWCCheckpoint:
    """Load and validate one EWC task-boundary checkpoint on CPU."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing EWC resume checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Could not load EWC checkpoint {path}: {exc}") from exc
    return _validate_payload(
        payload,
        expected_seed_metadata=expected_seed_metadata,
        expected_config=expected_config,
        expected_seen_class_map=expected_seen_class_map,
    )


def restore_model_and_theta_star(
    model: torch.nn.Module,
    checkpoint: LoadedEWCCheckpoint,
) -> dict[str, torch.Tensor]:
    """Restore model weights and reconstruct theta_star without storing a duplicate."""

    model.load_state_dict(checkpoint.model_state, strict=True)
    parameter_names = {name for name, _ in model.named_parameters()}
    fisher_names = set(checkpoint.fisher_total)
    if fisher_names != parameter_names:
        missing = sorted(parameter_names - fisher_names)
        unexpected = sorted(fisher_names - parameter_names)
        raise ValueError(
            "EWC Fisher parameter names do not match the restored model: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}."
        )
    theta_star: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name not in checkpoint.model_state:
            raise ValueError(f"EWC model_state is missing parameter {name}.")
        theta_star[name] = checkpoint.model_state[name].detach().clone().to(parameter.device)
    return theta_star
