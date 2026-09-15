from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from src.eval.ensemble_ue import (
    aggregate_member_predictions,
    export_offline_ensemble_artifact,
    load_offline_ensemble_artifact,
)
from src.eval.evaluation import PredictionOutputs
from src.eval.predictions import (
    MemberPredictionContext,
    PredictionProvenance,
    export_member_prediction_artifact,
    load_member_prediction_artifact,
    partition_identity_sha256,
)
from src.eval.threshold import load_frozen_threshold_artifact, main as threshold_main
from src.training.fold_ensemble3_pilot import (
    MEMBER_IDS,
    build_pilot_plan,
    execute_pilot,
    load_pilot_config,
)
from src.utils.seed import derive_member_seed


CONFIG = Path("configs/pilot_tddi_p3_fold_ensemble3_seed0.json")


def _arg(command: Sequence[str], name: str) -> str:
    return command[command.index(name) + 1]


def _config(tmp_path: Path):
    return load_pilot_config(CONFIG, overrides={"output_root": tmp_path / "pilot"})


def _inspector(states: dict[int, str], resume: Path | None = None):
    def inspect(config, member, command, *, require_inputs=False):
        status = states[member]
        return {
            "status": status,
            "completed_task_id": 1 if status == "complete" else (0 if status == "resume" else None),
            "resume_checkpoint": str(resume) if status == "resume" else None,
        }

    return inspect


def _probabilities(class_count: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(class_count, dtype=np.int64)
    if mode == "no_selection":
        probabilities = np.full((class_count, class_count), 1 / class_count, dtype=np.float32)
        return probabilities, labels
    probabilities = np.full(
        (class_count, class_count), 0.01 / (class_count - 1), dtype=np.float32
    )
    predicted = labels.copy()
    if mode == "fallback":
        predicted[1::2] = (predicted[1::2] + 1) % class_count
    probabilities[np.arange(class_count), predicted] = 0.99
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, labels


def _write_member_predictions(config, member_id: int, modes: dict[int, str]) -> None:
    root = Path(config["output_root"]) / f"member_{member_id}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_config.json").write_text(
        json.dumps({"run_id": f"synthetic-member-{member_id}"}), encoding="utf-8"
    )
    for task_id in range(2):
        classes = np.asarray(
            sorted(
                raw
                for task in config["tasks"][: task_id + 1]
                for raw in task["classes"]
            ),
            dtype=np.int64,
        )
        probabilities, label_columns = _probabilities(len(classes), modes[task_id])
        labels = classes[label_columns]
        predictions = classes[probabilities.argmax(axis=1)]
        outputs = PredictionOutputs(
            logits=np.log(np.clip(probabilities, 1e-12, 1.0)).astype(np.float32),
            probabilities=probabilities,
            predictions=predictions,
            labels=labels,
            latent_features=np.zeros((len(classes), 2), dtype=np.float32),
            class_ids=classes,
        )
        validation_ids_by_member = [
            np.asarray([f"oof-m{fold}-t{task_id}-r{row}" for row in range(len(classes))])
            for fold in MEMBER_IDS
        ]
        all_oof_ids = np.concatenate(validation_ids_by_member)
        all_oof_labels = np.concatenate([labels for _ in MEMBER_IDS])
        all_oof_folds = np.concatenate(
            [np.full(len(classes), fold, dtype=np.int16) for fold in MEMBER_IDS]
        )
        oof_digest = partition_identity_sha256(
            all_oof_ids, all_oof_labels, all_oof_folds
        )
        for split in ("validation", "test"):
            if split == "validation":
                sample_ids = validation_ids_by_member[member_id]
                fold_ids = np.full(len(classes), member_id, dtype=np.int16)
                expected_oof_rows = len(all_oof_ids)
                expected_oof_sha256 = oof_digest
            else:
                sample_ids = np.asarray(
                    [f"test-t{task_id}-r{row}" for row in range(len(classes))]
                )
                fold_ids = np.full(len(classes), -1, dtype=np.int16)
                expected_oof_rows = None
                expected_oof_sha256 = None
            provenance = PredictionProvenance(
                assignment_sha256="a" * 64,
                fold_manifest_sha256="b" * 64,
                task_file_sha256=config["protocol"]["sha256"],
                study_contract_sha256="d" * 64,
                preprocessing_policy="task0_standard_frozen",
                preprocessing_sha256=str(member_id + 1) * 64,
                preprocessing_member_id=member_id,
                ranking_policy="raw_sample_normalized_class_mean_control_v1",
                buffer_policy="fold_min_quota_sqrt_capacity_v1",
                member_memory_budget=config["member_budgets"][str(member_id)],
                global_memory_budget=config["global_slot_budget"],
                expected_partition_rows=len(classes),
                expected_partition_sha256=partition_identity_sha256(
                    sample_ids, labels, fold_ids
                ),
                expected_oof_rows=expected_oof_rows,
                expected_oof_sha256=expected_oof_sha256,
            )
            export_member_prediction_artifact(
                outputs,
                {"sample_id": sample_ids, "fold_id": fold_ids},
                MemberPredictionContext(
                    run_id=f"synthetic-member-{member_id}",
                    method="replay_distill_fixed_budget_uniform",
                    method_protocol="frozen_fold_replay_distill_v1",
                    task_id=task_id,
                    split=split,
                    member_id=member_id,
                    experiment_seed=0,
                    member_seed=derive_member_seed(0, member_id),
                    ensemble_mode="stratified_3fold",
                    fold_id=member_id,
                    fold_count=3,
                    fold_seed=42,
                ),
                root / "member_predictions" / f"task_{task_id}" / f"{split}.npz",
                provenance=provenance,
            )


def test_config_and_dry_run_plan_lock_approved_pilot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    states = {member: "fresh" for member in MEMBER_IDS}
    plan = build_pilot_plan(
        config, python="pilot-python", inspector=_inspector(states)
    )

    assert not Path(config["output_root"]).exists()
    assert [member.member_id for member in plan.members] == [0, 1, 2]
    assert [member.member_seed for member in plan.members] == [
        409845317, 215626784, 3041879697
    ]
    for member in plan.members:
        assert member.command is not None
        assert _arg(member.command, "--stop-after-task") == "1"
        assert _arg(member.command, "--member-id") == str(member.member_id)
        assert _arg(member.command, "--preprocessing-policy") == "task0_standard_frozen"
        assert _arg(member.command, "--exemplar-ranking-policy") == (
            "raw_sample_normalized_class_mean_control_v1"
        )
        assert "--validation-only" not in member.command
        assert "--export-member-predictions" in member.command
    assert all(
        evaluation.status == "blocked_until_three_members_complete"
        for evaluation in plan.evaluations
    )


def test_selected_member_skip_and_resume_plan(tmp_path: Path) -> None:
    config = _config(tmp_path)
    resume = tmp_path / "task_0.pt"
    states = {0: "complete", 1: "resume", 2: "fresh"}
    plan = build_pilot_plan(
        config,
        member_ids=[0, 1],
        python="pilot-python",
        inspector=_inspector(states, resume),
    )

    assert plan.members[0].command is None
    assert plan.members[1].command is not None
    assert _arg(plan.members[1].command, "--resume-fold-checkpoint") == str(resume)
    assert plan.members[2].command is None


def test_incomplete_member_without_boundary_checkpoint_fails_without_overwrite(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    partial = Path(config["output_root"]) / "member_0"
    partial.mkdir(parents=True)
    marker = partial / "interrupted.log"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without a valid frozen-fold checkpoint"):
        build_pilot_plan(config, member_ids=[0], python="pilot-python")

    assert marker.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize(
    ("task1_mode", "expected_status"),
    [("fallback", "fallback"), ("no_selection", "no_selection")],
)
def test_sequential_members_then_oof_test_threshold_and_manifest(
    tmp_path: Path,
    task1_mode: str,
    expected_status: str,
) -> None:
    config = _config(tmp_path)
    states = {member: "fresh" for member in MEMBER_IDS}
    calls: list[tuple[str, int | None]] = []

    def runner(command: Sequence[str], cwd: Path) -> None:
        assert cwd.is_dir()
        script = Path(command[1]).name
        if script == "train_cil.py":
            member_id = int(_arg(command, "--member-id"))
            calls.append(("train", member_id))
            _write_member_predictions(
                config, member_id, {0: "target", 1: task1_mode}
            )
            states[member_id] = "complete"
            return
        if script == "ensemble_ue.py":
            start = command.index("--member-artifacts") + 1
            paths = [Path(value) for value in command[start : start + 3]]
            artifacts = [load_member_prediction_artifact(path) for path in paths]
            aggregate = aggregate_member_predictions(
                artifacts,
                source_artifact_paths=paths,
                ensemble_mode="stratified_3fold",
            )
            export_offline_ensemble_artifact(aggregate, Path(_arg(command, "--out")))
            calls.append((f"ensemble-{aggregate.context.split}", aggregate.context.task_id))
            return
        if script == "threshold.py":
            threshold_main(list(command[2:]))
            calls.append((f"threshold-{command[2]}", None))
            return
        raise AssertionError(f"Unexpected command: {command}")

    manifest_path = execute_pilot(
        config,
        python="pilot-python",
        runner=runner,
        inspector=_inspector(states),
    )

    assert [call for call in calls if call[0] == "train"] == [
        ("train", 0), ("train", 1), ("train", 2)
    ]
    first_eval = next(index for index, call in enumerate(calls) if call[0].startswith("ensemble-"))
    assert first_eval == 3
    assert manifest_path is not None and manifest_path.is_file()
    task0_frozen = load_frozen_threshold_artifact(
        Path(config["output_root"]) / "offline_evaluation/task_0/frozen_threshold.json"
    )
    task1_frozen = load_frozen_threshold_artifact(
        Path(config["output_root"]) / "offline_evaluation/task_1/frozen_threshold.json"
    )
    assert task0_frozen.selection_status == "target_met"
    assert task1_frozen.selection_status == expected_status
    oof = load_offline_ensemble_artifact(
        Path(config["output_root"]) / "offline_evaluation/task_1/oof.npz"
    )
    test = load_offline_ensemble_artifact(
        Path(config["output_root"]) / "offline_evaluation/task_1/test.npz"
    )
    np.testing.assert_array_equal(oof.prediction_count, 1)
    np.testing.assert_array_equal(test.prediction_count, 3)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["execution_order"] == [0, 1, 2]
    assert manifest["study_scope"] == {
        "completed_tasks": [0, 1], "full_protocol_task_count": 8
    }
    assert len(manifest["members"]) == 3
    assert len(manifest["evaluations"]) == 2


def test_partial_selected_member_never_invokes_offline_evaluation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    states = {member: "fresh" for member in MEMBER_IDS}
    calls: list[str] = []

    def runner(command: Sequence[str], cwd: Path) -> None:
        calls.append(Path(command[1]).name)
        member_id = int(_arg(command, "--member-id"))
        _write_member_predictions(config, member_id, {0: "target", 1: "fallback"})
        states[member_id] = "complete"

    result = execute_pilot(
        config,
        member_ids=[1],
        python="pilot-python",
        runner=runner,
        inspector=_inspector(states),
    )

    assert result is None
    assert calls == ["train_cil.py"]
    assert not (Path(config["output_root"]) / "offline_evaluation").exists()
