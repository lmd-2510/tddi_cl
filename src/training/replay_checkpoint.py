"""Strict frozen-fold task-boundary checkpoints for CB-Replay, EWC and A-GEM."""
from __future__ import annotations
import os
import random
import uuid
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

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
        raise ValueError(f"RNG state is missing keys: {missing}")
    torch_cpu = rng_state["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8:
        raise TypeError("torch_cpu RNG state must be a uint8 tensor.")
    cuda_states = rng_state["torch_cuda"]
    if not isinstance(cuda_states, list) or not all(
        isinstance(state, torch.Tensor) and state.dtype == torch.uint8
        for state in cuda_states
    ):
        raise TypeError("torch_cuda RNG states must be a list of uint8 tensors.")
    declared_cuda_count = int(rng_state["torch_cuda_device_count"])
    if declared_cuda_count != len(cuda_states):
        raise ValueError("CUDA RNG state count does not match its metadata.")
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore a CUDA checkpoint without CUDA.")
        if torch.cuda.device_count() != declared_cuda_count:
            raise RuntimeError(
                "CUDA device count differs from the checkpoint: "
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

FOLD_REPLAY_CHECKPOINT_KIND = "ddi_cil_frozen_fold_method_task_boundary"
FOLD_REPLAY_CHECKPOINT_VERSION = 2


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
        method_state: dict, run_config_sha256: str) -> Path:
    """Commit a task AFTER best-model evaluation/buffer update and artifact publication."""
    payload = {
        "kind": FOLD_REPLAY_CHECKPOINT_KIND, "schema_version": FOLD_REPLAY_CHECKPOINT_VERSION,
        "run_id": run_id, "completed_task_id": completed_task_id, "next_task_id": completed_task_id + 1,
        "full_trajectory_complete": completed_task_id == 7,
        "model_state": _cpu_tensor_mapping(model_state, name="fold model_state"),
        "seen_class_map": dict(seen_class_map),
        "raw_class_order": [raw for raw, _ in sorted(seen_class_map.items(), key=lambda p: p[1])],
        "contract": contract, "buffer_state": buffer.state_dict(), "sampler_state": sampler.state_dict(),
        "method_state": method_state, "rng_state": capture_rng_state(), "progress": progress, "artifact_hashes": artifact_hashes,
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
    from src.training.regularization_memory import restore_memory
    from src.data.replay_sampler import FoldReplayFractionSampler
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
            "run_config_sha256", "next_task_scheduling", "method_state", "state_sha256"}
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
        manifest = expected_contract.get("model", {})
        if manifest.get("variant") != "tddi_paper_member":
            raise ValueError("Frozen-fold checkpoint requires the T-DDI backbone.")
        valid_head = (
            model["head.weight"].ndim == 2
            and model["head.weight"].shape[0] == len(classes)
            and tuple(model["head.bias"].shape) == (len(classes),)
        )
        if not valid_head or not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in model.values()):
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
            epoch_limit = expected_contract["hyperparameters"]["epochs"]
            if type(epoch_count) is not int or epoch_count < 1 or (epoch_limit is not None and epoch_count > epoch_limit):
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
        restored = restore_memory(state["buffer_state"], **buffer_kwargs)
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
            member_id=expected_contract["seeds"]["member_id"], task_id=completed,
            replay_fraction=expected_contract["sampler"]["replay_fraction"], repeat_cap=expected_contract["sampler"]["repeat_cap"])
        sampler = FoldReplayFractionSampler.from_state_dict(state["sampler_state"], **kwargs)
        if sampler.state_dict()["next_epoch"] != summaries[-1]["epochs_trained"]:
            raise ValueError("Completed sampler epoch mismatch.")
        expected_schedule = {"task_id": completed + 1, "epoch": 0,
            "policy": "reset_from_new_current_and_retained_IDs_with_member_task_key",
            "loader_seed": expected_contract["seeds"]["member_seed"] + completed + 1}
        if state["next_task_scheduling"] != expected_schedule:
            raise ValueError("Next-task scheduling mismatch.")
        method_state = state["method_state"]
        if expected_contract["method"] == "ewc":
            if set(method_state) != {"fisher", "theta_star"}:
                raise ValueError("EWC checkpoint requires Fisher and parameter anchors.")
            for name in ("fisher", "theta_star"):
                if set(method_state[name]) != set(model):
                    raise ValueError("EWC state parameter coverage mismatch.")
                for key, tensor in method_state[name].items():
                    if tensor.shape != model[key].shape or not torch.isfinite(tensor).all():
                        raise ValueError("EWC state shape/finite mismatch.")
                    if name == "fisher" and (tensor < 0).any():
                        raise ValueError("EWC Fisher must be nonnegative.")
        elif method_state:
            raise ValueError("Unexpected method state.")
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
