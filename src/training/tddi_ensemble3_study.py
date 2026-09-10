#!/usr/bin/env python3
"""Sequential orchestration for supported three-member T-DDI CIL studies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.predictions import (  # noqa: E402
    MemberPredictionArtifact,
    load_member_prediction_artifact,
)
from src.eval.ensemble_ue import (  # noqa: E402
    OfflineEnsembleArtifact,
    aggregate_member_predictions,
    load_offline_ensemble_artifact,
)
from src.data.class_mapping import build_seen_class_map  # noqa: E402
from src.training.replay_checkpoint import (  # noqa: E402
    LoadedReplayCheckpoint,
    load_replay_checkpoint,
)
from src.utils.seed import (  # noqa: E402
    MEMBER_SEED_DERIVATION,
    derive_member_seed,
)


STUDY_CONFIG_SCHEMA_VERSION = 1
STUDY_MANIFEST_SCHEMA_VERSION = 1
STUDY_MANIFEST_KIND = "ddi_cil_tddi_ensemble3_study"
MEMBER_IDS = (0, 1, 2)
COMPLETION_FILES = ("run_summary.md", "metrics.csv", "forgetting.csv")
EWC_METHOD = "ewc"
EWC_METHOD_PROTOCOL = "ewc_natural_sampling"
REPLAY_METHOD = "replay_distill_fixed_budget_uniform"
REPLAY_METHOD_PROTOCOL = REPLAY_METHOD
SUPPORTED_METHODS = (EWC_METHOD, REPLAY_METHOD)
PAPER_MEMBER_VARIANT = "tddi_paper_member"
SUPPORTED_STUDY_VARIANTS = (PAPER_MEMBER_VARIANT,)
SUPPORTED_PROTOCOLS = {
    "P3": ("tail_to_head", None),
    "P4": ("constrained_mass_balanced", 0),
}
EWC_RESUME_CHECKPOINT = Path("checkpoints/latest_ewc_state.pt")
REPLAY_RESUME_CHECKPOINT = Path("checkpoints/latest_replay_distill_state.pt")
# Backward-compatible public constant used by the original EWC orchestrator.
RESUME_CHECKPOINT = EWC_RESUME_CHECKPOINT
REPLAY_CHECKPOINT_POLICY = {
    "enabled": True,
    "boundary": "task",
    "checkpoint_path": REPLAY_RESUME_CHECKPOINT.as_posix(),
    "resume_cli": "--resume-replay-checkpoint",
    "skip_completed": True,
    "fail_incomplete_without_checkpoint": True,
}
CommandRunner = Callable[[Sequence[str], Path], None]


@dataclass(frozen=True)
class StudyConfig:
    source_path: Path
    sha256: str
    study_name: str
    experiment_seed: int
    member_ids: tuple[int, int, int]
    protocol_id: str
    protocol_name: str
    expected_task_count: int
    task_file: Path
    train: Path
    validation: Path
    test: Path
    feature_columns: Path
    scaler: Path
    variant: str
    dropout: float
    activation: str
    normalization: str
    method: str
    method_protocol: str
    optimizer: str
    microbatch_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    ewc_lambda: float | None
    total_memory_budget: int | None
    replay_draws_per_epoch: int | None
    distill_alpha: float | None
    temperature: float | None
    feature_distill_weight: float | None
    checkpoint_resume_policy: Mapping[str, Any]
    focal_gamma: float
    device: str
    ensemble_mode: str
    fold_count: int | None
    fold_seed: int | None
    prediction_splits: tuple[str, ...]
    output_root: Path
    member_namespace: str
    ensemble_namespace: str
    manifest_name: str
    tasks: tuple[Mapping[str, Any], ...]
    task_file_sha256: str


@dataclass(frozen=True)
class MemberPlan:
    member_id: int
    member_seed: int
    outdir: Path
    status: str
    completed_task_id: int | None
    resume_checkpoint: Path | None
    command: tuple[str, ...] | None


@dataclass(frozen=True)
class EnsemblePlan:
    task_id: int
    split: str
    member_artifacts: tuple[Path, Path, Path]
    output_path: Path
    status: str
    command: tuple[str, ...] | None


@dataclass(frozen=True)
class StudyPlan:
    members: tuple[MemberPlan, ...]
    ensembles: tuple[EnsemblePlan, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(payload: object, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return payload


def _resolve_path(value: object, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_study_config(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> StudyConfig:
    """Load a locked seed-0/three-member T-DDI study contract."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing T-DDI ensemble study config: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid study config JSON: {path}") from error
    payload = _object(payload, "Study config")
    if int(payload.get("schema_version", -1)) != STUDY_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported T-DDI ensemble study config schema_version.")
    if int(payload.get("experiment_seed", -1)) != 0:
        raise ValueError("tddi_ensemble3 study requires experiment_seed=0.")
    member_ids = tuple(int(value) for value in payload.get("member_ids", []))
    if member_ids != MEMBER_IDS:
        raise ValueError("tddi_ensemble3 study requires member_ids [0, 1, 2].")

    project_root = Path(project_root).resolve()
    protocol = _object(payload.get("protocol"), "protocol")
    inputs = _object(payload.get("inputs"), "inputs")
    model = _object(payload.get("model"), "model")
    training = _object(payload.get("training"), "training")
    outputs = _object(payload.get("outputs"), "outputs")
    ensemble = _object(payload.get("ensemble", {"mode": "seeded"}), "ensemble")
    protocol_id = str(protocol.get("id", ""))
    protocol_name = str(protocol.get("name", ""))
    if protocol_id not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"Unsupported tddi_ensemble3 protocol {protocol_id!r}; "
            f"expected one of {sorted(SUPPORTED_PROTOCOLS)}."
        )
    expected_protocol_name, expected_task_seed = SUPPORTED_PROTOCOLS[protocol_id]
    if protocol_name != expected_protocol_name:
        raise ValueError(
            f"Protocol {protocol_id} requires name={expected_protocol_name!r}."
        )
    variant = str(model.get("variant", ""))
    if variant != PAPER_MEMBER_VARIANT:
        raise ValueError(
            "tddi_ensemble3 requires the paper member variant "
            f"{PAPER_MEMBER_VARIANT!r}; got {variant!r}."
        )
    if model.get("activation") not in {"relu", "gelu"}:
        raise ValueError("model.activation must be relu or gelu.")
    if model.get("normalization") != "layernorm":
        raise ValueError("T-DDI numerical study requires normalization=layernorm.")
    method = str(training.get("method", ""))
    method_protocol = str(training.get("method_protocol", ""))
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            "training.method must be one of "
            f"{list(SUPPORTED_METHODS)}, got {method!r}."
        )
    if method == EWC_METHOD and method_protocol != EWC_METHOD_PROTOCOL:
        raise ValueError("tddi_ensemble3 requires the baseline EWC protocol.")
    if method == REPLAY_METHOD and method_protocol != REPLAY_METHOD_PROTOCOL:
        raise ValueError(
            "Replay ensemble requires method_protocol="
            f"{REPLAY_METHOD_PROTOCOL!r}."
        )
    prediction_splits = tuple(str(value) for value in payload.get("prediction_splits", []))
    if prediction_splits != ("validation", "test"):
        raise ValueError("prediction_splits must be exactly ['validation', 'test'].")
    ensemble_mode = str(ensemble.get("mode", "seeded"))
    if ensemble_mode not in {"seeded", "stratified_3fold"}:
        raise ValueError("ensemble.mode must be seeded or stratified_3fold.")
    fold_count = None
    fold_seed = None
    if ensemble_mode == "stratified_3fold":
        fold_count = int(ensemble.get("fold_count", 3))
        fold_seed = int(ensemble.get("fold_seed", 42))
        if fold_count != len(member_ids):
            raise ValueError("stratified_3fold fold_count must equal the three members.")
        if fold_seed < 0:
            raise ValueError("ensemble.fold_seed must be non-negative.")

    task_file = _resolve_path(protocol.get("task_file"), project_root, "protocol.task_file")
    if not task_file.is_file():
        raise FileNotFoundError(f"Missing {protocol_id} task file: {task_file}")
    task_spec = _object(
        json.loads(task_file.read_text(encoding="utf-8")),
        f"{protocol_id} task file",
    )
    task_seed = task_spec.get("seed")
    if task_spec.get("protocol") != protocol_name or task_seed != expected_task_seed:
        raise ValueError(
            f"Task file does not match protocol {protocol_id}/{protocol_name}; "
            f"expected task seed {expected_task_seed!r}."
        )
    raw_tasks = task_spec.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(f"{protocol_id} task file must contain a non-empty tasks list.")
    expected_task_count = int(protocol.get("expected_task_count", 0))
    if len(raw_tasks) != expected_task_count or expected_task_count <= 0:
        raise ValueError(
            f"{protocol_id} task count does not match protocol.expected_task_count."
        )
    task_ids = [int(_object(task, "task").get("task_id", -1)) for task in raw_tasks]
    if task_ids != list(range(expected_task_count)):
        raise ValueError("Task IDs must be contiguous from zero.")
    all_classes = [
        int(raw_class_id)
        for task in raw_tasks
        for raw_class_id in _object(task, "task").get("classes", [])
    ]
    if not all_classes or len(set(all_classes)) != len(all_classes):
        raise ValueError(
            f"{protocol_id} tasks must contain non-empty, disjoint raw class IDs."
        )
    task_layout = [
        len(_object(task, "task").get("classes", []))
        for task in raw_tasks
    ]
    if "expected_task_layout" in protocol:
        expected_layout = [int(value) for value in protocol["expected_task_layout"]]
        if task_layout != expected_layout:
            raise ValueError(
                f"{protocol_id} task layout mismatch: {task_layout} != {expected_layout}."
            )
    if "expected_class_count" in protocol:
        expected_class_count = int(protocol["expected_class_count"])
        if len(all_classes) != expected_class_count:
            raise ValueError(
                f"{protocol_id} class count does not match protocol.expected_class_count."
            )

    microbatch_size = int(training.get("microbatch_size", 0))
    effective_batch_size = int(training.get("effective_batch_size", 0))
    if (
        microbatch_size <= 0
        or effective_batch_size < microbatch_size
        or effective_batch_size % microbatch_size
    ):
        raise ValueError("Invalid microbatch/effective batch configuration.")
    optimizer = str(training.get("optimizer", "adamw")).casefold()
    if optimizer != "adamw":
        raise ValueError("tddi_ensemble3 training.optimizer must be adamw.")
    gradient_accumulation_steps = effective_batch_size // microbatch_size
    configured_accumulation = int(
        training.get("gradient_accumulation_steps", gradient_accumulation_steps)
    )
    if configured_accumulation != gradient_accumulation_steps:
        raise ValueError(
            "training.gradient_accumulation_steps must equal "
            "effective_batch_size / microbatch_size."
        )
    epochs = int(training.get("epochs", 0))
    patience = int(training.get("patience", -1))
    if epochs <= 0 or patience < 0:
        raise ValueError("training.epochs must be positive and patience non-negative.")
    for key in ("learning_rate", "weight_decay", "focal_gamma"):
        value = float(training.get(key, -1.0))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"training.{key} must be finite and non-negative.")
    ewc_lambda: float | None = None
    total_memory_budget: int | None = None
    replay_draws_per_epoch: int | None = None
    distill_alpha: float | None = None
    temperature: float | None = None
    feature_distill_weight: float | None = None
    if method == EWC_METHOD:
        ewc_lambda = float(training.get("ewc_lambda", -1.0))
        if not np.isfinite(ewc_lambda) or ewc_lambda < 0.0:
            raise ValueError("training.ewc_lambda must be finite and non-negative.")
        checkpoint_resume_policy: Mapping[str, Any] = {
            "enabled": True,
            "boundary": "task",
            "checkpoint_path": EWC_RESUME_CHECKPOINT.as_posix(),
            "resume_cli": "--resume-ewc-checkpoint",
            "skip_completed": True,
            "fail_incomplete_without_checkpoint": True,
        }
    else:
        total_memory_budget = int(training.get("total_memory_budget", 0))
        replay_draws_per_epoch = int(training.get("replay_draws_per_epoch", 0))
        if total_memory_budget != 6800:
            raise ValueError("Replay training.total_memory_budget must equal 6800.")
        if replay_draws_per_epoch != 6800:
            raise ValueError("Replay training.replay_draws_per_epoch must equal 6800.")
        if int(training.get("replay_starts_at_task", 1)) != 1:
            raise ValueError("Replay training.replay_starts_at_task must equal 1.")
        distill_alpha = float(training.get("distill_alpha", -1.0))
        temperature = float(training.get("temperature", 0.0))
        feature_distill_weight = float(training.get("feature_distill_weight", -1.0))
        for key, value in (
            ("distill_alpha", distill_alpha),
            ("feature_distill_weight", feature_distill_weight),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Replay training.{key} must be finite and non-negative.")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Replay training.temperature must be finite and positive.")
        checkpoint_resume_policy = _object(
            training.get("checkpoint_resume_policy"),
            "training.checkpoint_resume_policy",
        )
        if dict(checkpoint_resume_policy) != REPLAY_CHECKPOINT_POLICY:
            raise ValueError(
                "Replay training.checkpoint_resume_policy does not match the locked "
                "task-boundary resume policy."
            )
    dropout = float(model.get("dropout", -1.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must lie in [0, 1).")
    member_namespace = str(outputs.get("member_namespace", ""))
    if member_namespace.count("{member_id}") != 1:
        raise ValueError("outputs.member_namespace must contain one {member_id} placeholder.")
    manifest_name = str(outputs.get("manifest", ""))
    if not manifest_name or Path(manifest_name).name != manifest_name:
        raise ValueError("outputs.manifest must be a file name inside the output root.")
    ensemble_namespace = str(outputs.get("ensemble_namespace", ""))
    if not ensemble_namespace or Path(ensemble_namespace).name != ensemble_namespace:
        raise ValueError("outputs.ensemble_namespace must be one directory name.")
    study_name = str(payload.get("study_name", ""))
    if not study_name:
        raise ValueError("study_name must be non-empty.")

    return StudyConfig(
        source_path=path,
        sha256=_sha256_file(path),
        study_name=study_name,
        experiment_seed=0,
        member_ids=member_ids,  # type: ignore[arg-type]
        protocol_id=protocol_id,
        protocol_name=protocol_name,
        expected_task_count=expected_task_count,
        task_file=task_file,
        train=_resolve_path(inputs.get("train"), project_root, "inputs.train"),
        validation=_resolve_path(inputs.get("validation"), project_root, "inputs.validation"),
        test=_resolve_path(inputs.get("test"), project_root, "inputs.test"),
        feature_columns=_resolve_path(
            inputs.get("feature_columns"), project_root, "inputs.feature_columns"
        ),
        scaler=_resolve_path(inputs.get("scaler"), project_root, "inputs.scaler"),
        variant=variant,
        dropout=dropout,
        activation=str(model.get("activation", "")),
        normalization=str(model.get("normalization", "")),
        method=method,
        method_protocol=method_protocol,
        optimizer=optimizer,
        microbatch_size=microbatch_size,
        effective_batch_size=effective_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        patience=patience,
        ewc_lambda=ewc_lambda,
        total_memory_budget=total_memory_budget,
        replay_draws_per_epoch=replay_draws_per_epoch,
        distill_alpha=distill_alpha,
        temperature=temperature,
        feature_distill_weight=feature_distill_weight,
        checkpoint_resume_policy=dict(checkpoint_resume_policy),
        focal_gamma=float(training["focal_gamma"]),
        device=str(training.get("device", "auto")),
        ensemble_mode=ensemble_mode,
        fold_count=fold_count,
        fold_seed=fold_seed,
        prediction_splits=prediction_splits,
        output_root=_resolve_path(outputs.get("root"), project_root, "outputs.root"),
        member_namespace=member_namespace,
        ensemble_namespace=ensemble_namespace,
        manifest_name=manifest_name,
        tasks=tuple(_object(task, "task") for task in raw_tasks),
        task_file_sha256=_sha256_file(task_file),
    )


def member_outdir(config: StudyConfig, member_id: int) -> Path:
    return config.output_root / config.member_namespace.format(member_id=member_id)


def member_prediction_path(
    config: StudyConfig,
    member_id: int,
    task_id: int,
    split: str,
) -> Path:
    return (
        member_outdir(config, member_id)
        / "member_predictions"
        / f"task_{task_id}"
        / f"{split}.npz"
    )


def ensemble_output_path(config: StudyConfig, task_id: int, split: str) -> Path:
    output_split = (
        "oof"
        if config.ensemble_mode == "stratified_3fold" and split == "validation"
        else split
    )
    return (
        config.output_root
        / config.ensemble_namespace
        / f"task_{task_id}"
        / f"{output_split}.npz"
    )


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _core_run_complete(outdir: Path) -> bool:
    return all(_is_nonempty_file(outdir / name) for name in COMPLETION_FILES)


def _all_predictions_exist(config: StudyConfig, member_id: int) -> bool:
    return all(
        _is_nonempty_file(member_prediction_path(config, member_id, task_id, split))
        for task_id in range(config.expected_task_count)
        for split in config.prediction_splits
    )


def _resume_checkpoint_relative_path(config: StudyConfig) -> Path:
    return (
        EWC_RESUME_CHECKPOINT
        if config.method == EWC_METHOD
        else REPLAY_RESUME_CHECKPOINT
    )


def _resume_cli_flag(config: StudyConfig) -> str:
    return (
        "--resume-ewc-checkpoint"
        if config.method == EWC_METHOD
        else "--resume-replay-checkpoint"
    )


def _checkpoint_event_type(config: StudyConfig) -> str:
    return (
        "ewc_checkpoint_saved"
        if config.method == EWC_METHOD
        else "replay_checkpoint_saved"
    )


def _completed_task_from_events(
    outdir: Path,
    *,
    event_type: str = "ewc_checkpoint_saved",
) -> int | None:
    events_path = outdir / "events.csv"
    if not events_path.is_file():
        return None
    completed: list[int] = []
    with events_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type") != event_type:
                continue
            try:
                payload = json.loads(row.get("payload_json") or "{}")
                completed.append(int(payload["completed_task_id"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return max(completed) if completed else None


def _expected_seen_map(config: StudyConfig, completed_task_id: int) -> dict[int, int]:
    if completed_task_id < 0 or completed_task_id >= config.expected_task_count:
        raise ValueError(
            f"Checkpoint completed_task_id={completed_task_id} is outside the "
            f"configured task range 0..{config.expected_task_count - 1}."
        )
    seen = {
        int(raw_class_id)
        for task in config.tasks[: completed_task_id + 1]
        for raw_class_id in task["classes"]
    }
    return build_seen_class_map(seen)


def _validate_replay_resume_checkpoint(
    config: StudyConfig,
    member_id: int,
    checkpoint_path: Path,
) -> LoadedReplayCheckpoint:
    checkpoint = load_replay_checkpoint(checkpoint_path)
    expected_seed = derive_member_seed(config.experiment_seed, member_id)
    expected_seed_metadata = {
        "experiment_seed": config.experiment_seed,
        "member_id": member_id,
        "member_seed": expected_seed,
        "member_seed_derivation": MEMBER_SEED_DERIVATION,
        "seed_mode": "member",
    }
    mismatched_seed_keys = [
        key
        for key, expected in expected_seed_metadata.items()
        if checkpoint.seed_metadata.get(key) != expected
    ]
    if mismatched_seed_keys:
        raise ValueError(
            f"Replay checkpoint seed metadata mismatch: {mismatched_seed_keys}."
        )
    expected_contract = {
        "method": config.method,
        "variant": config.variant,
        "task_file_sha256": config.task_file_sha256,
        "batch_size": config.microbatch_size,
        "effective_batch_size": config.effective_batch_size,
        "epochs": config.epochs,
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "dropout": config.dropout,
        "activation": config.activation,
        "norm": config.normalization,
        "patience": config.patience,
        "focal_gamma": config.focal_gamma,
        "distill_alpha": config.distill_alpha,
        "temperature": config.temperature,
        "feature_distill_weight": config.feature_distill_weight,
        "total_memory_budget": config.total_memory_budget,
        "replay_draws_per_epoch": config.replay_draws_per_epoch,
    }
    mismatched_config_keys = [
        key
        for key, expected in expected_contract.items()
        if checkpoint.config.get(key) != expected
    ]
    if mismatched_config_keys:
        raise ValueError(
            f"Replay checkpoint training contract mismatch: {mismatched_config_keys}."
        )
    expected_map = _expected_seen_map(config, checkpoint.completed_task_id)
    if checkpoint.seen_class_map != expected_map:
        raise ValueError("Replay checkpoint class map does not match configured tasks.")
    required_progress = {
        "result_matrix",
        "result_rows",
        "best_task_rows",
        "training_audit_rows",
        "replay_budget_audit_rows",
        "class_trajectory_rows",
    }
    missing_progress = sorted(required_progress - set(checkpoint.progress))
    if missing_progress:
        raise ValueError(
            f"Replay checkpoint progress is missing keys: {missing_progress}."
        )
    result_matrix = np.asarray(checkpoint.progress["result_matrix"])
    if result_matrix.shape != (
        config.expected_task_count,
        config.expected_task_count,
    ):
        raise ValueError("Replay checkpoint result matrix has the wrong task shape.")
    completed_count = checkpoint.completed_task_id + 1
    for key in ("best_task_rows", "training_audit_rows"):
        rows = checkpoint.progress[key]
        if not isinstance(rows, list) or len(rows) != completed_count:
            raise ValueError(
                f"Replay checkpoint {key} does not match completed_task_id."
            )
    for key in ("result_rows", "replay_budget_audit_rows", "class_trajectory_rows"):
        if not isinstance(checkpoint.progress[key], list):
            raise TypeError(f"Replay checkpoint {key} must be a list.")
    run_config_path = member_outdir(config, member_id) / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(
            f"Replay resume requires the member run_config.json: {run_config_path}"
        )
    run_config = _object(
        json.loads(run_config_path.read_text(encoding="utf-8")),
        "member run_config.json",
    )
    if run_config.get("run_id") != checkpoint.run_id:
        raise ValueError("Replay checkpoint run_id does not match member run_config.json.")
    return checkpoint


def _member_status(config: StudyConfig, member_id: int) -> tuple[str, int | None, Path | None]:
    outdir = member_outdir(config, member_id)
    core_complete = _core_run_complete(outdir)
    predictions_complete = _all_predictions_exist(config, member_id)
    if core_complete:
        if not predictions_complete:
            raise RuntimeError(
                f"Completed member {member_id} is missing prediction artifacts; "
                "refusing to overwrite or retrain it."
            )
        return "complete", config.expected_task_count - 1, None
    checkpoint = outdir / _resume_checkpoint_relative_path(config)
    if _is_nonempty_file(checkpoint):
        if config.method == REPLAY_METHOD:
            try:
                loaded = _validate_replay_resume_checkpoint(config, member_id, checkpoint)
            except Exception as error:
                raise RuntimeError(
                    f"Incomplete member {member_id} has an invalid replay checkpoint: "
                    f"{checkpoint}: {error}"
                ) from error
            if loaded.next_task_id >= config.expected_task_count:
                raise RuntimeError(
                    f"Incomplete member {member_id} has a final-task replay checkpoint "
                    "but is missing completion artifacts; refusing to overwrite it."
                )
            return "resume", loaded.completed_task_id, checkpoint
        return (
            "resume",
            _completed_task_from_events(
                outdir,
                event_type=_checkpoint_event_type(config),
            ),
            checkpoint,
        )
    if outdir.exists() and any(outdir.iterdir()):
        checkpoint_name = "EWC" if config.method == EWC_METHOD else "replay"
        raise RuntimeError(
            f"Incomplete member {member_id} has no resumable {checkpoint_name} "
            f"checkpoint: {outdir}"
        )
    return "fresh", None, None


def _training_command(
    config: StudyConfig,
    member_id: int,
    python_executable: str,
    resume_checkpoint: Path | None,
) -> tuple[str, ...]:
    command = [
        python_executable,
        str(PROJECT_ROOT / "src/training/train_cil.py"),
        "--train",
        str(config.train),
        "--validation",
        str(config.validation),
        "--test",
        str(config.test),
        "--feature-cols",
        str(config.feature_columns),
        "--scaler",
        str(config.scaler),
        "--task-file",
        str(config.task_file),
        "--outdir",
        str(member_outdir(config, member_id)),
        "--method",
        config.method,
        "--variant",
        config.variant,
        "--batch-size",
        str(config.microbatch_size),
        "--effective-batch-size",
        str(config.effective_batch_size),
        "--epochs",
        str(config.epochs),
        "--lr",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--dropout",
        str(config.dropout),
        "--activation",
        config.activation,
        "--norm",
        config.normalization,
        "--patience",
        str(config.patience),
        "--focal-gamma",
        str(config.focal_gamma),
        "--seed",
        str(config.experiment_seed),
        "--member-id",
        str(member_id),
        "--device",
        config.device,
        "--ensemble-mode",
        config.ensemble_mode,
        "--export-member-predictions",
        "--member-prediction-splits",
        *config.prediction_splits,
    ]
    if config.ensemble_mode == "stratified_3fold":
        command.extend(
            (
                "--fold-id",
                str(member_id),
                "--fold-count",
                str(config.fold_count),
                "--fold-seed",
                str(config.fold_seed),
            )
        )
    if config.method == EWC_METHOD:
        if config.ewc_lambda is None:
            raise RuntimeError("EWC study config is missing ewc_lambda.")
        command.extend(("--ewc-lambda", str(config.ewc_lambda)))
    else:
        replay_values = (
            config.total_memory_budget,
            config.replay_draws_per_epoch,
            config.distill_alpha,
            config.temperature,
            config.feature_distill_weight,
        )
        if any(value is None for value in replay_values):
            raise RuntimeError("Replay study config is missing replay/distillation values.")
        command.extend(
            (
                "--total-memory-budget",
                str(config.total_memory_budget),
                "--replay-draws-per-epoch",
                str(config.replay_draws_per_epoch),
                "--distill-alpha",
                str(config.distill_alpha),
                "--temperature",
                str(config.temperature),
                "--feature-distill-weight",
                str(config.feature_distill_weight),
            )
        )
    if resume_checkpoint is not None:
        command.extend((_resume_cli_flag(config), str(resume_checkpoint)))
    return tuple(command)


def _ensemble_command(
    member_artifacts: tuple[Path, Path, Path],
    output_path: Path,
    python_executable: str,
    ensemble_mode: str = "seeded",
) -> tuple[str, ...]:
    return (
        python_executable,
        str(PROJECT_ROOT / "src/eval/ensemble_ue.py"),
        "--member-artifacts",
        *(str(path) for path in member_artifacts),
        "--mode",
        ensemble_mode,
        "--out",
        str(output_path),
    )


def build_study_plan(
    config: StudyConfig,
    *,
    python_executable: str = sys.executable,
    selected_member_ids: Sequence[int] | None = None,
) -> StudyPlan:
    selected = config.member_ids if selected_member_ids is None else tuple(selected_member_ids)
    if len(set(selected)) != len(selected) or any(value not in config.member_ids for value in selected):
        raise ValueError("Selected member IDs must be a unique subset of [0, 1, 2].")
    members: list[MemberPlan] = []
    for member_id in config.member_ids:
        status, completed_task_id, resume_checkpoint = _member_status(config, member_id)
        should_plan = member_id in selected and status != "complete"
        members.append(
            MemberPlan(
                member_id=member_id,
                member_seed=derive_member_seed(config.experiment_seed, member_id),
                outdir=member_outdir(config, member_id),
                status=status,
                completed_task_id=completed_task_id,
                resume_checkpoint=resume_checkpoint,
                command=(
                    _training_command(
                        config,
                        member_id,
                        python_executable,
                        resume_checkpoint,
                    )
                    if should_plan
                    else None
                ),
            )
        )
    ensembles: list[EnsemblePlan] = []
    for task_id in range(config.expected_task_count):
        for split in config.prediction_splits:
            paths = tuple(
                member_prediction_path(config, member_id, task_id, split)
                for member_id in config.member_ids
            )
            output_path = ensemble_output_path(config, task_id, split)
            status = "complete" if _is_nonempty_file(output_path) else "pending"
            ensembles.append(
                EnsemblePlan(
                    task_id=task_id,
                    split=split,
                    member_artifacts=paths,  # type: ignore[arg-type]
                    output_path=output_path,
                    status=status,
                    command=(
                        None
                        if status == "complete"
                        else _ensemble_command(
                            paths,
                            output_path,
                            python_executable,
                            config.ensemble_mode,
                        )  # type: ignore[arg-type]
                    ),
                )
            )
    return StudyPlan(members=tuple(members), ensembles=tuple(ensembles))


def _validate_member_group(
    config: StudyConfig,
    task_id: int,
    split: str,
) -> tuple[list[MemberPredictionArtifact], OfflineEnsembleArtifact]:
    artifacts: list[MemberPredictionArtifact] = []
    paths: list[Path] = []
    for member_id in config.member_ids:
        path = member_prediction_path(config, member_id, task_id, split)
        artifact = load_member_prediction_artifact(path)
        context = artifact.context
        expected_seed = derive_member_seed(config.experiment_seed, member_id)
        if (
            context.member_id != member_id
            or context.member_seed != expected_seed
            or context.experiment_seed != config.experiment_seed
            or context.method != config.method
            or context.method_protocol != config.method_protocol
            or context.task_id != task_id
            or context.split != split
        ):
            raise ValueError(
                f"Member artifact provenance mismatch for member={member_id}, "
                f"task={task_id}, split={split}."
            )
        artifacts.append(artifact)
        paths.append(path)
    aggregate = aggregate_member_predictions(
        artifacts,
        source_artifact_paths=paths,
        ensemble_mode=config.ensemble_mode,
    )
    return artifacts, aggregate


def validate_all_member_alignment(
    config: StudyConfig,
) -> dict[tuple[int, str], OfflineEnsembleArtifact]:
    """Validate shared sample/class order before any offline-ensemble process runs."""

    validated: dict[tuple[int, str], OfflineEnsembleArtifact] = {}
    for task_id in range(config.expected_task_count):
        for split in config.prediction_splits:
            _, aggregate = _validate_member_group(config, task_id, split)
            validated[(task_id, split)] = aggregate
    return validated


def _default_runner(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _validate_execution_inputs(config: StudyConfig) -> None:
    missing = [
        path
        for path in (
            config.train,
            config.validation,
            config.test,
            config.feature_columns,
            config.scaler,
            config.task_file,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Study execution inputs are missing: {missing}")


def _validate_existing_ensemble(
    path: Path,
    expected: OfflineEnsembleArtifact,
) -> None:
    loaded = load_offline_ensemble_artifact(path)
    if loaded.context != expected.context:
        raise ValueError(f"Existing ensemble context mismatch: {path}")
    for field_name in ("sample_ids", "labels", "raw_class_ids", "probabilities"):
        if not np.array_equal(getattr(loaded, field_name), getattr(expected, field_name)):
            raise ValueError(f"Existing ensemble {field_name} mismatch: {path}")


def _read_run_id(outdir: Path) -> str:
    path = outdir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Completed member is missing run_config.json: {path}")
    payload = _object(json.loads(path.read_text(encoding="utf-8")), "run_config.json")
    run_id = str(payload.get("run_id", ""))
    if not run_id:
        raise ValueError(f"Completed member run_config has no run_id: {path}")
    return run_id


def _manifest_payload(
    config: StudyConfig,
    validated: Mapping[tuple[int, str], OfflineEnsembleArtifact],
    python_executable: str,
) -> dict[str, Any]:
    members = []
    for member_id in config.member_ids:
        outdir = member_outdir(config, member_id)
        members.append(
            {
                "member_id": member_id,
                "member_seed": derive_member_seed(config.experiment_seed, member_id),
                "member_seed_derivation": MEMBER_SEED_DERIVATION,
                "run_id": _read_run_id(outdir),
                "outdir": str(outdir),
                "run_config": str(outdir / "run_config.json"),
                "run_config_sha256": _sha256_file(outdir / "run_config.json"),
                "resume_checkpoint": str(
                    outdir / _resume_checkpoint_relative_path(config)
                ),
                "completion_files": [str(outdir / name) for name in COMPLETION_FILES],
            }
        )
    groups = []
    for task_id in range(config.expected_task_count):
        for split in config.prediction_splits:
            aggregate = validated[(task_id, split)]
            paths = tuple(
                member_prediction_path(config, member_id, task_id, split)
                for member_id in config.member_ids
            )
            output_path = ensemble_output_path(config, task_id, split)
            groups.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "member_artifacts": [
                        {
                            "member_id": member_id,
                            "path": str(path),
                            "sha256": _sha256_file(path),
                        }
                        for member_id, path in zip(config.member_ids, paths)
                    ],
                    "raw_class_ids": aggregate.raw_class_ids.astype(int).tolist(),
                    "sample_count": aggregate.row_count,
                    "offline_ensemble": {
                        "path": str(output_path),
                        "sha256": _sha256_file(output_path),
                        "command": list(
                            _ensemble_command(
                                paths,
                                output_path,
                                python_executable,
                                config.ensemble_mode,
                            )
                        ),  # type: ignore[arg-type]
                    },
                }
            )
    return {
        "schema_version": STUDY_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": STUDY_MANIFEST_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_name": config.study_name,
        "study_config": str(config.source_path),
        "study_config_sha256": config.sha256,
        "method": config.method,
        "method_protocol": config.method_protocol,
        "ensemble": {
            "mode": config.ensemble_mode,
            "fold_count": config.fold_count,
            "fold_seed": config.fold_seed,
            "threshold_source": (
                "oof" if config.ensemble_mode == "stratified_3fold" else "validation"
            ),
        },
        "model": {
            "variant": config.variant,
            "hidden_dimensions": (
                [7560, 7560]
            ),
            "dropout": config.dropout,
            "activation": config.activation,
            "normalization": config.normalization,
        },
        "experiment_seed": config.experiment_seed,
        "protocol": {
            "id": config.protocol_id,
            "name": config.protocol_name,
            "task_file": str(config.task_file),
            "task_file_sha256": config.task_file_sha256,
        },
        "checkpoint_resume_policy": dict(config.checkpoint_resume_policy),
        "execution_policy": "sequential_members_never_concurrent",
        "execution_order": list(config.member_ids),
        "members": members,
        "ensemble_groups": groups,
    }


def _write_manifest_once(payload: Mapping[str, Any], path: Path) -> Path:
    if path.exists():
        current = _object(json.loads(path.read_text(encoding="utf-8")), "study manifest")
        if (
            current.get("artifact_kind") == STUDY_MANIFEST_KIND
            and current.get("study_config_sha256") == payload.get("study_config_sha256")
        ):
            return path
        raise FileExistsError(f"Refusing to overwrite existing study manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def execute_study(
    config: StudyConfig,
    *,
    python_executable: str = sys.executable,
    selected_member_ids: Sequence[int] | None = None,
    runner: CommandRunner = _default_runner,
) -> Path | None:
    """Run selected trajectories serially, then ensemble only when all three finish."""

    _validate_execution_inputs(config)
    selected = config.member_ids if selected_member_ids is None else tuple(selected_member_ids)
    plan = build_study_plan(
        config,
        python_executable=python_executable,
        selected_member_ids=selected,
    )
    for member in plan.members:
        if member.member_id not in selected or member.status == "complete":
            continue
        if member.command is None:
            raise RuntimeError(f"Missing command for member {member.member_id}.")
        runner(member.command, PROJECT_ROOT)
        status, _, _ = _member_status(config, member.member_id)
        if status != "complete":
            raise RuntimeError(f"Member {member.member_id} did not finish successfully.")

    if any(_member_status(config, member_id)[0] != "complete" for member_id in config.member_ids):
        return None

    validated = validate_all_member_alignment(config)
    for task_id in range(config.expected_task_count):
        for split in config.prediction_splits:
            expected = validated[(task_id, split)]
            paths = tuple(
                member_prediction_path(config, member_id, task_id, split)
                for member_id in config.member_ids
            )
            output_path = ensemble_output_path(config, task_id, split)
            if output_path.exists():
                _validate_existing_ensemble(output_path, expected)
                continue
            runner(
                _ensemble_command(
                    paths,
                    output_path,
                    python_executable,
                    config.ensemble_mode,
                ),  # type: ignore[arg-type]
                PROJECT_ROOT,
            )
            _validate_existing_ensemble(output_path, expected)

    manifest_path = config.output_root / config.manifest_name
    return _write_manifest_once(
        _manifest_payload(config, validated, python_executable),
        manifest_path,
    )


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def print_dry_run(plan: StudyPlan) -> None:
    print("Execution policy: sequential members; no concurrent model/GPU process.")
    for member in plan.members:
        print(
            f"member={member.member_id} member_seed={member.member_seed} "
            f"status={member.status} completed_task={member.completed_task_id}"
        )
        if member.command is not None:
            print(f"  {_format_command(member.command)}")
    print("Offline ensemble commands run only after all three members are complete:")
    for ensemble in plan.ensembles:
        if ensemble.command is not None:
            print(
                f"task={ensemble.task_id} split={ensemble.split} "
                f"{_format_command(ensemble.command)}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or explicitly execute supported sequential tddi_ensemble3 CIL members."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly execute; omission is a side-effect-free dry run.",
    )
    parser.add_argument(
        "--member-id",
        action="append",
        type=int,
        dest="member_ids",
        help="Execute/plan only this member; repeat to select more than one.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = load_study_config(args.config)
    if not args.execute:
        print_dry_run(
            build_study_plan(
                config,
                python_executable=args.python,
                selected_member_ids=args.member_ids,
            )
        )
        return
    manifest = execute_study(
        config,
        python_executable=args.python,
        selected_member_ids=args.member_ids,
    )
    if manifest is None:
        print("Selected member run(s) finished; ensemble is pending other members.")
    else:
        print(f"Study complete: {manifest}")


if __name__ == "__main__":
    main()
