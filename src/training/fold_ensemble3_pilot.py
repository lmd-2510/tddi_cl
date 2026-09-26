#!/usr/bin/env python3
"""Sequential approved three-fold pilot and offline evaluation orchestration.

This entrypoint never implements training itself.  It invokes ``train_cil.py``
in blocking member order and only starts OOF/test evaluation after all three
task-0--1 trajectories pass the frozen-fold checkpoint/prediction checks.
Omit ``--execute`` for a read-only dry run.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from string import Formatter
import sys
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fold_replay_buffer import BUFFER_POLICIES, four_percent_member_budgets  # noqa: E402
from src.data.stratified_folds import fold_file_sha256  # noqa: E402
from src.eval.ensemble_ue import (  # noqa: E402
    aggregate_member_predictions,
    load_offline_ensemble_artifact,
)
from src.eval.predictions import load_member_prediction_artifact  # noqa: E402
from src.methods.weight_alignment import WEIGHT_ALIGNMENT_POLICIES  # noqa: E402
from src.eval.threshold import (  # noqa: E402
    MACRO_F1_SELECTION_RULE,
    load_frozen_threshold_artifact,
    load_threshold_selection_config,
)
from src.training import train_cil as engine  # noqa: E402
from src.training.fold_ab_study import (  # noqa: E402
    INPUT_KEYS,
    MODEL,
    REPLAY,
    TRAINING,
    _default_runner,
    inspect_member,
    member_command,
)
from src.training.fold_pilot_training import P3_LAYOUT, _publish_json  # noqa: E402
from src.utils.seed import resolve_seed_configuration  # noqa: E402


CONFIG_KIND = "tddi_frozen_fold_ensemble3_pilot"
MEMBER_IDS = (0, 1, 2)
STOP_AFTER_TASK = 1
EXECUTION = {
    "stop_after_task": STOP_AFTER_TASK,
    "validation_only": False,
    "sequential": True,
    "checkpoint_resume_policy": "immutable_frozen_fold_task_boundary_v1",
    "export_member_predictions": ["validation", "test"],
}
EVALUATION_KEYS = {"threshold_config", "ensemble_namespace"}
SUPPORTED_PROTOCOLS = {
    "P2": "head_to_tail",
    "P3": "tail_to_head",
    "P4": "constrained_mass_balanced",
}
Runner = Callable[[Sequence[str], Path], None]
Inspector = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class MemberPilotPlan:
    member_id: int
    member_seed: int
    status: str
    completed_task_id: int | None
    outdir: Path
    command: tuple[str, ...] | None
    resume_checkpoint: Path | None


@dataclass(frozen=True)
class EvaluationPilotPlan:
    task_id: int
    status: str
    oof_path: Path
    test_path: Path
    frozen_threshold_path: Path
    oof_report_path: Path
    test_report_path: Path
    commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class FoldEnsemblePilotPlan:
    members: tuple[MemberPilotPlan, ...]
    evaluations: tuple[EvaluationPilotPlan, ...]


def _keys(value: object, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(
            f"{name}: required keys {sorted(expected)}; unknown/missing keys are not ignored."
        )
    return value


def _resolve(value: object, root: Path, name: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value)
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def _validate_template(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("preprocessing.artifact_template must be a non-empty string.")
    fields = list(Formatter().parse(value))
    if sum(field == "member_id" for _, field, _, _ in fields) != 1:
        raise ValueError("Preprocessing template must contain one plain {member_id}.")
    if any(
        field is not None and (field != "member_id" or spec or conversion)
        for _, field, spec, conversion in fields
    ):
        raise ValueError("Only plain {member_id} is supported in preprocessing paths.")
    return value


def _load_locked_config(
    path: str | Path,
    *,
    expected_kind: str,
    expected_phase: str,
    expected_stop_after_task: int,
    project_root: str | Path = PROJECT_ROOT,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Load one locked fold-ensemble config without requiring server-only inputs."""

    path, root = Path(path).resolve(), Path(project_root).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    # Configs created before the explicit budget-policy field remain valid and
    # retain the historical meaning: 4% is the ensemble-wide slot budget,
    # split across the three members.  New runs may explicitly request 4% per
    # member without changing the old contract.
    if isinstance(value, dict) and "budget_policy" not in value:
        value["budget_policy"] = "ensemble_total_4_percent"
    _keys(
        value,
        {
            "schema_version", "kind", "phase", "experiment_seed", "fold_seed",
            "member_ids", "development_count", "global_slot_budget", "member_budgets",
            "budget_policy",
            "protocol", "inputs", "preprocessing", "model", "training", "replay",
            "execution", "evaluation", "device", "output_root",
        },
        "fold ensemble pilot config",
    )
    if (
        value["schema_version"] != 1
        or value["kind"] != expected_kind
        or value["phase"] != expected_phase
    ):
        raise ValueError("Unsupported fold ensemble kind/schema/phase.")
    if value["experiment_seed"] != 0 or value["fold_seed"] != 42:
        raise ValueError("Pilot requires experiment seed 0 and fold seed 42.")
    if tuple(value["member_ids"]) != MEMBER_IDS:
        raise ValueError("Pilot member order must be exactly [0, 1, 2].")
    if type(value["development_count"]) is not int or value["development_count"] != 694455:
        raise ValueError("Pilot development_count must be the audited 694455 rows.")
    budget_policy = value["budget_policy"]
    if budget_policy == "ensemble_total_4_percent":
        budgets = {
            str(key): amount
            for key, amount in four_percent_member_budgets(value["development_count"]).items()
        }
        expected_global_budget = sum(budgets.values())
        budget_error = "Pilot budgets must be 9260/9259/9259 from the global 4% slot budget."
    elif budget_policy == "per_member_4_percent":
        # Each member owns an independent 4% development-set buffer.  The
        # physical total across the ensemble is therefore 12% (three copies
        # of the 4% member budget), by design.
        per_member = 4 * value["development_count"] // 100
        budgets = {str(member): per_member for member in MEMBER_IDS}
        expected_global_budget = sum(budgets.values())
        budget_error = "Per-member 4% policy requires 27778 slots for each member (83334 total)."
    else:
        raise ValueError(
            "budget_policy must be ensemble_total_4_percent or per_member_4_percent."
        )
    if value["member_budgets"] != budgets or value["global_slot_budget"] != expected_global_budget:
        raise ValueError(budget_error)

    protocol = _keys(
        value["protocol"],
        {"id", "name", "task_file", "sha256", "layout", "class_count"},
        "protocol",
    )
    protocol_id = protocol["id"]
    protocol_name = SUPPORTED_PROTOCOLS.get(protocol_id)
    if (
        protocol_name is None
        or protocol["name"] != protocol_name
        or protocol["layout"] != P3_LAYOUT
        or protocol["class_count"] != 178
    ):
        raise ValueError(
            "Study requires a supported full eight-task/178-class protocol "
            f"({SUPPORTED_PROTOCOLS})."
        )
    _keys(value["inputs"], INPUT_KEYS, "inputs")
    _keys(value["preprocessing"], {"policy", "artifact_template"}, "preprocessing")
    # Schema-v1 configs created before WA/loss pilots remain valid.
    if isinstance(value.get("training"), dict) and "weight_alignment" not in value["training"]:
        value["training"]["weight_alignment"] = "none"
    if isinstance(value.get("training"), dict) and "loss_variant" not in value["training"]:
        value["training"]["loss_variant"] = "baseline"
    if isinstance(value.get("training"), dict):
        value["training"].setdefault("class_balance_beta", 0.9999)
        value["training"].setdefault("class_balance_max_weight", 4.0)
    _keys(value["training"], {*TRAINING, "epochs", "patience"}, "training")
    _keys(value["evaluation"], EVALUATION_KEYS, "evaluation")
    if value["preprocessing"]["policy"] != "task0_standard_frozen":
        raise ValueError("Approved pilot preprocessing is task0_standard_frozen.")
    if value["model"] != MODEL:
        raise ValueError("Pilot model must be the locked tddi_paper_member configuration.")
    expected_replay = {**REPLAY, "ranking_policy": "raw_sample_normalized_class_mean_control_v1"}
    replay = value["replay"]
    for key in ("buffer_policy", "ranking_policy", "base_quota", "fraction", "repeat_cap", "replay_starts_at_task"):
        if key not in replay:
            raise ValueError(f"replay.{key} is required.")
    for key in ("ranking_policy", "base_quota", "replay_starts_at_task"):
        if replay[key] != expected_replay[key]:
            raise ValueError(f"replay.{key} does not match the approved buffer policy.")
    if replay["buffer_policy"] not in BUFFER_POLICIES:
        raise ValueError(f"replay.buffer_policy must be one of {BUFFER_POLICIES}.")
    if not isinstance(replay["fraction"], (int, float)) or not 0.0 < float(replay["fraction"]) < 1.0:
        raise ValueError("replay.fraction must be in (0,1).")
    if type(replay["repeat_cap"]) is not int or replay["repeat_cap"] <= 0:
        raise ValueError("replay.repeat_cap must be a positive integer.")
    fold_policy = value["training"].get("fold_replay_policy")
    if fold_policy not in {
        "stratified_fraction_v1", "stratified_fraction_rotating_current_v2"
    }:
        raise ValueError("Unsupported fold replay sampler policy.")
    weight_alignment = value["training"].get("weight_alignment")
    if weight_alignment not in WEIGHT_ALIGNMENT_POLICIES:
        raise ValueError("Unsupported classifier weight-alignment policy.")
    if value["training"].get("loss_variant") not in {
        "baseline", "er", "hybrid", "hybrid_distill",
        "hybrid_distill_replay_only", "hybrid_logit_distill", "cb_hybrid", "focal_all", "er_ace",
    }:
        raise ValueError("Unsupported replay loss variant.")
    configured_epochs = value["training"].get("epochs")
    if fold_policy == "stratified_fraction_rotating_current_v2":
        expected_epochs = 30
    elif expected_phase == "full" and configured_epochs in (20, 25, 30):
        # 20 is the immutable historical baseline; 25/30 are convergence
        # extensions in separate output namespaces.
        expected_epochs = configured_epochs
    else:
        expected_epochs = 20
    expected_training = {
        **TRAINING,
        "fold_replay_policy": fold_policy,
        "weight_alignment": weight_alignment,
        "loss_variant": value["training"]["loss_variant"],
        "epochs": expected_epochs,
        "patience": 5,
    }
    if value["training"]["loss_variant"] in {"hybrid_logit_distill", "cb_hybrid"}:
        expected_training["feature_distill_weight"] = 0.0
    if value["training"] != expected_training:
        raise ValueError("Pilot training hyperparameters must match the approved baseline.")
    expected_execution = {
        **EXECUTION,
        "stop_after_task": expected_stop_after_task,
    }
    if value["execution"] != expected_execution:
        raise ValueError(
            f"Study must stop at task {expected_stop_after_task}, export validation/test, "
            "and run sequentially."
        )
    namespace = value["evaluation"]["ensemble_namespace"]
    if not isinstance(namespace, str) or not namespace or Path(namespace).name != namespace:
        raise ValueError("evaluation.ensemble_namespace must be one directory name.")

    overrides = dict(overrides or {})
    for name in INPUT_KEYS:
        source = overrides.get(name) or value["inputs"][name]
        value["inputs"][name] = _resolve(source, root, f"inputs.{name}")
    task_source = overrides.get("task_file") or protocol["task_file"]
    protocol["task_file"] = _resolve(task_source, root, "protocol.task_file")
    if fold_file_sha256(Path(protocol["task_file"])) != protocol["sha256"]:
        raise ValueError(
            f"{protocol_id} task-file SHA256 mismatch; path may move but contents may not."
        )
    tasks = engine.load_task_spec(Path(protocol["task_file"]))
    if (
        tasks.get("protocol") != protocol_name
        or [task["task_id"] for task in tasks["tasks"]] != list(range(8))
        or [len(task["classes"]) for task in tasks["tasks"]] != P3_LAYOUT
        or len({raw for task in tasks["tasks"] for raw in task["classes"]}) != 178
    ):
        raise ValueError(
            f"{protocol_id} task file does not contain the locked 8-task/178-class map."
        )
    expected_protocol_seed = 0 if protocol_id == "P4" else None
    if tasks.get("seed") != expected_protocol_seed:
        raise ValueError(
            f"{protocol_id} task file seed must be {expected_protocol_seed!r}."
        )

    template = _validate_template(value["preprocessing"]["artifact_template"])
    if overrides.get("preprocessing_root"):
        template = str(
            Path(overrides["preprocessing_root"]) / "member_{member_id}" / "B"
            / "fold_preprocessing.json"
        )
    value["preprocessing"]["artifact_template"] = _resolve(
        template, root, "preprocessing.artifact_template"
    )
    for member_id in MEMBER_IDS:
        value["preprocessing"]["artifact_template"].format(member_id=member_id)

    threshold_source = overrides.get("threshold_config") or value["evaluation"]["threshold_config"]
    value["evaluation"]["threshold_config"] = _resolve(
        threshold_source, root, "evaluation.threshold_config"
    )
    threshold = load_threshold_selection_config(value["evaluation"]["threshold_config"])
    threshold_common_ok = (
        threshold.selection_source != "oof"
        or threshold.confidence_score != "entropy_confidence"
        or threshold.probability_source != "raw"
        or threshold.candidate_grid != tuple(index / 100 for index in range(50, 100))
    )
    macro_f1_rule_ok = (
        threshold.selection_rule == MACRO_F1_SELECTION_RULE
        and threshold.minimum_coverage == 0.5
        and threshold.tie_breakers == ("accuracy", "coverage", "lower_threshold")
        and threshold.target_accuracy is None
    )
    if threshold_common_ok or not macro_f1_rule_ok:
        raise ValueError(
            "Fold ensemble threshold config must use the official OOF Macro-F1 policy."
        )

    output_source = overrides.get("output_root") or value["output_root"]
    value["output_root"] = _resolve(output_source, root, "output_root")
    value["device"] = str(overrides.get("device") or value["device"])
    value["config_path"] = str(path)
    value["config_sha256"] = fold_file_sha256(path)
    value["tasks"] = tasks["tasks"]
    return value


def load_pilot_config(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Load the locked approved task-0--1 pilot."""

    return _load_locked_config(
        path,
        expected_kind=CONFIG_KIND,
        expected_phase="pilot",
        expected_stop_after_task=STOP_AFTER_TASK,
        project_root=project_root,
        overrides=overrides,
    )


def _task_ids(config: Mapping[str, Any]) -> range:
    return range(int(config["execution"]["stop_after_task"]) + 1)


def _member_outdir(config: Mapping[str, Any], member_id: int) -> Path:
    return Path(config["output_root"]) / f"member_{member_id}"


def _prediction_path(config: Mapping[str, Any], member_id: int, task_id: int, split: str) -> Path:
    return _member_outdir(config, member_id) / "member_predictions" / f"task_{task_id}" / f"{split}.npz"


def _evaluation_root(config: Mapping[str, Any], task_id: int) -> Path:
    return Path(config["output_root"]) / config["evaluation"]["ensemble_namespace"] / f"task_{task_id}"


def _ensemble_command(config: Mapping[str, Any], task_id: int, split: str, python: str) -> tuple[str, ...]:
    output_name = "oof.npz" if split == "validation" else "test.npz"
    return (
        python,
        str(PROJECT_ROOT / "src/eval/ensemble_ue.py"),
        "--member-artifacts",
        *(str(_prediction_path(config, member, task_id, split)) for member in MEMBER_IDS),
        "--mode", "stratified_3fold",
        "--out", str(_evaluation_root(config, task_id) / output_name),
    )


def _threshold_commands(config: Mapping[str, Any], task_id: int, python: str) -> tuple[tuple[str, ...], ...]:
    root = _evaluation_root(config, task_id)
    threshold_config = str(config["evaluation"]["threshold_config"])
    frozen = root / "frozen_threshold.json"
    return (
        (
            python, str(PROJECT_ROOT / "src/eval/threshold.py"), "select",
            "--ensemble", str(root / "oof.npz"), "--config", threshold_config,
            "--threshold-out", str(frozen), "--report-out", str(root / "oof_threshold_report.json"),
        ),
        (
            python, str(PROJECT_ROOT / "src/eval/threshold.py"), "evaluate",
            "--ensemble", str(root / "test.npz"), "--config", threshold_config,
            "--threshold-artifact", str(frozen), "--report-out", str(root / "test_threshold_report.json"),
        ),
    )


def _eval_plan(config: Mapping[str, Any], task_id: int, python: str, all_complete: bool) -> EvaluationPilotPlan:
    root = _evaluation_root(config, task_id)
    paths = (
        root / "oof.npz", root / "test.npz", root / "frozen_threshold.json",
        root / "oof_threshold_report.json", root / "test_threshold_report.json",
    )
    if all(path.is_file() and path.stat().st_size > 0 for path in paths):
        status = "complete"
    elif any(path.exists() for path in paths):
        status = "resumable_partial"
    elif all_complete:
        status = "pending"
    else:
        status = "blocked_until_three_members_complete"
    commands = (
        _ensemble_command(config, task_id, "validation", python),
        _ensemble_command(config, task_id, "test", python),
        *_threshold_commands(config, task_id, python),
    )
    return EvaluationPilotPlan(task_id, status, *paths, commands)


def build_pilot_plan(
    config: Mapping[str, Any],
    *,
    member_ids: Sequence[int] | None = None,
    python: str = sys.executable,
    require_inputs: bool = False,
    inspector: Inspector = inspect_member,
) -> FoldEnsemblePilotPlan:
    selected = MEMBER_IDS if member_ids is None else tuple(member_ids)
    if not selected or len(set(selected)) != len(selected) or any(value not in MEMBER_IDS for value in selected):
        raise ValueError("Select a unique non-empty subset of members 0/1/2.")
    members: list[MemberPilotPlan] = []
    for member_id in MEMBER_IDS:
        command = member_command(config, member_id, python=python)
        info = inspector(config, member_id, command, require_inputs=require_inputs)
        status = str(info["status"])
        resume = Path(info["resume_checkpoint"]) if info.get("resume_checkpoint") else None
        if status == "resume":
            command.extend(("--resume-fold-checkpoint", str(resume)))
        planned = tuple(command) if member_id in selected and status != "complete" else None
        members.append(
            MemberPilotPlan(
                member_id=member_id,
                member_seed=resolve_seed_configuration(0, member_id).member_seed,
                status=status,
                completed_task_id=info.get("completed_task_id"),
                outdir=_member_outdir(config, member_id),
                command=planned,
                resume_checkpoint=resume,
            )
        )
    all_complete = all(member.status == "complete" for member in members)
    evaluations = tuple(
        _eval_plan(config, task_id, python, all_complete)
        for task_id in _task_ids(config)
    )
    return FoldEnsemblePilotPlan(tuple(members), evaluations)


def _validate_member_predictions(config: Mapping[str, Any], member_id: int) -> None:
    for task_id in _task_ids(config):
        expected_classes = sorted(
            raw
            for task in config["tasks"][: task_id + 1]
            for raw in task["classes"]
        )
        for split in ("validation", "test"):
            path = _prediction_path(config, member_id, task_id, split)
            artifact = load_member_prediction_artifact(path)
            if (
                artifact.context.member_id != member_id
                or artifact.context.task_id != task_id
                or artifact.context.split != split
                or artifact.context.ensemble_mode != "stratified_3fold"
                or artifact.context.fold_id != member_id
                or artifact.context.fold_count != 3
                or artifact.context.fold_seed != 42
                or artifact.context.method != "replay_distill_fixed_budget_uniform"
                or artifact.context.method_protocol != "frozen_fold_replay_distill_v1"
                or artifact.context.experiment_seed != 0
                or artifact.context.member_seed
                != resolve_seed_configuration(0, member_id).member_seed
                or artifact.raw_class_ids.astype(int).tolist() != expected_classes
                or artifact.provenance is None
                or artifact.provenance.preprocessing_member_id != member_id
                or artifact.provenance.preprocessing_policy != "task0_standard_frozen"
                or artifact.provenance.ranking_policy
                != "raw_sample_normalized_class_mean_control_v1"
                or artifact.provenance.buffer_policy != config["replay"]["buffer_policy"]
                or artifact.provenance.member_memory_budget != config["member_budgets"][str(member_id)]
                or artifact.provenance.global_memory_budget != config["global_slot_budget"]
            ):
                raise ValueError(f"Member prediction contract mismatch: {path}")


def _validate_ensemble(config: Mapping[str, Any], task_id: int, split: str, path: Path) -> None:
    artifact = load_offline_ensemble_artifact(path)
    expected_split = "oof" if split == "validation" else "test"
    expected_classes = sorted(
        raw for task in config["tasks"][: task_id + 1] for raw in task["classes"]
    )
    if (
        artifact.context.task_id != task_id
        or artifact.context.split != expected_split
        or artifact.context.ensemble_mode != "stratified_3fold"
        or artifact.context.member_ids != MEMBER_IDS
        or artifact.raw_class_ids.astype(int).tolist() != expected_classes
        or artifact.member_provenance is None
    ):
        raise ValueError(f"Offline ensemble contract mismatch: {path}")


def _run_ensemble(
    config: Mapping[str, Any], task_id: int, split: str, python: str, runner: Runner
) -> Path:
    output = _evaluation_root(config, task_id) / ("oof.npz" if split == "validation" else "test.npz")
    if output.exists():
        _validate_ensemble(config, task_id, split, output)
        return output
    paths = [_prediction_path(config, member, task_id, split) for member in MEMBER_IDS]
    artifacts = [load_member_prediction_artifact(path) for path in paths]
    aggregate_member_predictions(
        artifacts, source_artifact_paths=paths, ensemble_mode="stratified_3fold"
    )
    runner(_ensemble_command(config, task_id, split, python), PROJECT_ROOT)
    _validate_ensemble(config, task_id, split, output)
    return output


def _load_report(path: Path, *, task_id: int, split: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_id") != task_id or payload.get("evaluation_split") != split:
        raise ValueError(f"Threshold report context mismatch: {path}")
    return payload


def _run_thresholds(config: Mapping[str, Any], task_id: int, python: str, runner: Runner) -> None:
    root = _evaluation_root(config, task_id)
    frozen_path = root / "frozen_threshold.json"
    oof_report = root / "oof_threshold_report.json"
    test_report = root / "test_threshold_report.json"
    select_command, test_command = _threshold_commands(config, task_id, python)
    if not frozen_path.exists():
        if oof_report.exists() or test_report.exists():
            raise RuntimeError("Threshold reports exist without their frozen threshold; refusing overwrite.")
        runner(select_command, PROJECT_ROOT)
    frozen = load_frozen_threshold_artifact(
        frozen_path, config_path=config["evaluation"]["threshold_config"]
    )
    if frozen.task_id != task_id or frozen.source_split != "oof":
        raise ValueError(f"Frozen threshold context mismatch: {frozen_path}")
    if not oof_report.exists():
        # Resume a select command interrupted after atomically freezing the rule.
        command = (
            python, str(PROJECT_ROOT / "src/eval/threshold.py"), "evaluate",
            "--ensemble", str(root / "oof.npz"),
            "--config", str(config["evaluation"]["threshold_config"]),
            "--threshold-artifact", str(frozen_path), "--report-out", str(oof_report),
        )
        runner(command, PROJECT_ROOT)
    _load_report(oof_report, task_id=task_id, split="oof")
    if not test_report.exists():
        runner(test_command, PROJECT_ROOT)
    _load_report(test_report, task_id=task_id, split="test")


def _manifest(
    config: Mapping[str, Any],
    *,
    artifact_kind: str = "ddi_cil_frozen_fold_ensemble3_pilot_manifest",
    limitations: Sequence[str] | None = None,
) -> dict[str, Any]:
    members = []
    for member_id in MEMBER_IDS:
        run_config = _member_outdir(config, member_id) / "run_config.json"
        run_payload = json.loads(run_config.read_text(encoding="utf-8"))
        members.append(
            {
                "member_id": member_id,
                "member_seed": resolve_seed_configuration(0, member_id).member_seed,
                "validation_fold": member_id,
                "memory_budget": config["member_budgets"][str(member_id)],
                "run_id": run_payload["run_id"],
                "run_config": str(run_config),
                "run_config_sha256": fold_file_sha256(run_config),
                "predictions": {
                    str(task_id): {
                        split: str(_prediction_path(config, member_id, task_id, split))
                        for split in ("validation", "test")
                    }
                    for task_id in _task_ids(config)
                },
            }
        )
    evaluations = []
    for task_id in _task_ids(config):
        root = _evaluation_root(config, task_id)
        paths = {
            "oof_ensemble": root / "oof.npz",
            "test_ensemble": root / "test.npz",
            "frozen_threshold": root / "frozen_threshold.json",
            "oof_threshold_report": root / "oof_threshold_report.json",
            "test_threshold_report": root / "test_threshold_report.json",
        }
        evaluations.append(
            {
                "task_id": task_id,
                "artifacts": {
                    name: {"path": str(path), "sha256": fold_file_sha256(path)}
                    for name, path in paths.items()
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": artifact_kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_scope": {
            "completed_tasks": list(_task_ids(config)),
            "full_protocol_task_count": 8,
        },
        "config_path": config["config_path"],
        "config_sha256": config["config_sha256"],
        "task_file_sha256": config["protocol"]["sha256"],
        "experiment_seed": 0,
        "fold_seed": 42,
        "execution_order": [0, 1, 2],
        "global_slot_budget": config["global_slot_budget"],
        "preprocessing_policy": config["preprocessing"]["policy"],
        "ranking_policy": config["replay"]["ranking_policy"],
        "members": members,
        "evaluations": evaluations,
        "limitations": list(
            limitations
            if limitations is not None
            else (
                "Task-0--1 pilot does not establish task-7 continual-learning performance.",
                "OOF target accuracy does not guarantee the same selected test accuracy.",
            )
        ),
    }


def execute_ensemble_study(
    config: Mapping[str, Any],
    *,
    member_ids: Sequence[int] | None = None,
    python: str = sys.executable,
    runner: Runner = _default_runner,
    inspector: Inspector = inspect_member,
    manifest_name: str,
    manifest_kind: str,
    limitations: Sequence[str],
) -> Path | None:
    """Run selected members serially and evaluate only after all three complete."""

    selected = MEMBER_IDS if member_ids is None else tuple(member_ids)
    plan = build_pilot_plan(
        config, member_ids=selected, python=python, require_inputs=True, inspector=inspector
    )
    for member in plan.members:
        if member.status == "complete":
            _validate_member_predictions(config, member.member_id)
    for member in plan.members:
        if member.member_id not in selected or member.status == "complete":
            continue
        command = member_command(config, member.member_id, python=python)
        current = inspector(config, member.member_id, command, require_inputs=True)
        if current["status"] == "complete":
            continue
        if current["status"] not in {"fresh", "resume"}:
            raise RuntimeError(
                f"Member {member.member_id} is not safely runnable: {current['status']}."
            )
        if current["status"] == "resume":
            command.extend(("--resume-fold-checkpoint", str(current["resume_checkpoint"])))
        runner(tuple(command), PROJECT_ROOT)
        after = inspector(
            config,
            member.member_id,
            member_command(config, member.member_id, python=python),
            require_inputs=True,
        )
        expected_boundary = config["execution"]["stop_after_task"]
        if after["status"] != "complete":
            raise RuntimeError(
                f"Member {member.member_id} exited without a verified task-{expected_boundary} boundary."
            )
        _validate_member_predictions(config, member.member_id)

    final = build_pilot_plan(
        config, member_ids=selected, python=python, require_inputs=True, inspector=inspector
    )
    if any(member.status != "complete" for member in final.members):
        return None
    # A single selected member is a supported training operation (for example,
    # a protocol-diverse ensemble).  It must not trigger validation of missing
    # sibling predictions or run the same-protocol offline aggregation.
    if set(selected) != set(MEMBER_IDS):
        return None
    for member_id in MEMBER_IDS:
        _validate_member_predictions(config, member_id)
    for task_id in _task_ids(config):
        _run_ensemble(config, task_id, "validation", python, runner)
        _run_ensemble(config, task_id, "test", python, runner)
        _run_thresholds(config, task_id, python, runner)

    manifest_path = Path(config["output_root"]) / manifest_name
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("artifact_kind") != manifest_kind
            or existing.get("config_sha256") != config["config_sha256"]
        ):
            raise FileExistsError(
                f"Refusing to overwrite unrelated study manifest: {manifest_path}"
            )
        return manifest_path
    _publish_json(
        manifest_path,
        _manifest(config, artifact_kind=manifest_kind, limitations=limitations),
    )
    return manifest_path


def execute_pilot(
    config: Mapping[str, Any],
    *,
    member_ids: Sequence[int] | None = None,
    python: str = sys.executable,
    runner: Runner = _default_runner,
    inspector: Inspector = inspect_member,
) -> Path | None:
    """Run the selected task-0--1 pilot members serially."""

    return execute_ensemble_study(
        config,
        member_ids=member_ids,
        python=python,
        runner=runner,
        inspector=inspector,
        manifest_name="pilot_manifest.json",
        manifest_kind="ddi_cil_frozen_fold_ensemble3_pilot_manifest",
        limitations=(
            "Task-0--1 pilot does not establish task-7 continual-learning performance.",
            "OOF target accuracy does not guarantee the same selected test accuracy.",
        ),
    )


def print_dry_run(plan: FoldEnsemblePilotPlan) -> None:
    print("DRY RUN: no model/training/writes. Members run synchronously 0->1->2; scope task 0-1 only.")
    for member in plan.members:
        print(
            f"member={member.member_id} member_seed={member.member_seed} "
            f"status={member.status} completed_task={member.completed_task_id}"
        )
        if member.command:
            print(shlex.join(member.command))
    print("Offline OOF/test ensemble and frozen threshold run only after all three members are complete.")
    for evaluation in plan.evaluations:
        print(f"task={evaluation.task_id} evaluation_status={evaluation.status}")
        for command in evaluation.commands:
            print(shlex.join(command))
    print("No full eight-task trajectory is scheduled by this pilot entrypoint.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--member-id", type=int, choices=MEMBER_IDS, action="append", dest="member_ids")
    parser.add_argument("--python", default=sys.executable)
    for name in sorted(INPUT_KEYS | {"task_file", "preprocessing_root", "output_root", "threshold_config"}):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    override_names = INPUT_KEYS | {"task_file", "preprocessing_root", "output_root", "threshold_config", "device"}
    config = load_pilot_config(
        args.config,
        overrides={name: getattr(args, name) for name in override_names},
    )
    if not args.execute:
        print_dry_run(
            build_pilot_plan(config, member_ids=args.member_ids, python=args.python)
        )
        return
    result = execute_pilot(
        config, member_ids=args.member_ids, python=args.python
    )
    if result is None:
        print("Selected member(s) complete; offline evaluation waits for all three members.")
    else:
        print(f"Pilot complete: {result}")


if __name__ == "__main__":
    main()
