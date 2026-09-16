from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.training import fold_ensemble3_pilot as shared
from src.training.fold_ensemble3_full import (
    MANIFEST_KIND,
    MEMBER_IDS,
    execute_full,
    load_full_config,
)


CONFIG = Path("configs/full_tddi_p3_fold_ensemble3_seed0.json")
BASELINE25_CONFIG = Path(
    "configs/full_tddi_p3_fold_ensemble3_seed0_epochs25.json"
)
REPLAY12P5_CONFIG = Path(
    "configs/full_tddi_p3_fold_ensemble3_seed0_replay12p5.json"
)


def _arg(command: Sequence[str], name: str) -> str:
    return command[command.index(name) + 1]


def _config(tmp_path: Path):
    return load_full_config(CONFIG, overrides={"output_root": tmp_path / "full"})


def _inspector(states: dict[int, str], resume: Path | None = None):
    def inspect(config, member, command, *, require_inputs=False):
        status = states[member]
        return {
            "status": status,
            "completed_task_id": 7 if status == "complete" else (4 if status == "resume" else None),
            "resume_checkpoint": str(resume) if status == "resume" else None,
        }

    return inspect


def test_full_config_and_dry_run_are_locked_to_eight_tasks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    states = {member: "fresh" for member in MEMBER_IDS}
    plan = shared.build_pilot_plan(
        config, python="full-python", inspector=_inspector(states)
    )

    assert config["phase"] == "full"
    assert config["execution"]["stop_after_task"] == 7
    assert config["training"]["epochs"] == 20
    assert config["training"]["patience"] == 5
    assert config["member_budgets"] == {"0": 9260, "1": 9259, "2": 9259}
    assert not Path(config["output_root"]).exists()
    assert len(plan.evaluations) == 8
    assert [evaluation.task_id for evaluation in plan.evaluations] == list(range(8))
    for member in plan.members:
        assert member.command is not None
        assert _arg(member.command, "--stop-after-task") == "7"
        assert _arg(member.command, "--member-id") == str(member.member_id)
        assert _arg(member.command, "--fold-replay-policy") == "stratified_fraction_v1"
        assert _arg(member.command, "--epochs") == "20"
        assert _arg(member.command, "--exemplar-ranking-policy") == (
            "raw_sample_normalized_class_mean_control_v1"
        )
        assert "--export-member-predictions" in member.command


def test_replay12p5_full_config_selects_rotating_current_and_balanced_threshold(
    tmp_path: Path,
) -> None:
    config = load_full_config(
        REPLAY12P5_CONFIG,
        overrides={"output_root": tmp_path / "replay12p5"},
    )
    plan = shared.build_pilot_plan(
        config,
        python="full-python",
        inspector=_inspector({member: "fresh" for member in MEMBER_IDS}),
    )

    assert config["training"]["fold_replay_policy"] == (
        "stratified_fraction_rotating_current_v2"
    )
    assert config["training"]["epochs"] == 30
    assert config["training"]["patience"] == 5
    assert config["evaluation"]["threshold_config"].endswith(
        "eval_tddi_p3_ensemble_entropy_balanced_accuracy_threshold.json"
    )
    for member in plan.members:
        assert _arg(member.command, "--fold-replay-policy") == (
            "stratified_fraction_rotating_current_v2"
        )


def test_baseline25_uses_separate_namespace_and_keeps_original_sampler(
    tmp_path: Path,
) -> None:
    config = load_full_config(
        BASELINE25_CONFIG,
        overrides={"output_root": tmp_path / "baseline25"},
    )
    plan = shared.build_pilot_plan(
        config,
        python="full-python",
        inspector=_inspector({member: "fresh" for member in MEMBER_IDS}),
    )

    assert config["training"]["epochs"] == 25
    assert config["training"]["patience"] == 5
    assert config["training"]["fold_replay_policy"] == "stratified_fraction_v1"
    assert config["evaluation"]["threshold_config"].endswith(
        "eval_tddi_p3_ensemble_entropy_balanced_accuracy_threshold.json"
    )
    for member in plan.members:
        assert _arg(member.command, "--epochs") == "25"
        assert _arg(member.command, "--fold-replay-policy") == "stratified_fraction_v1"


def test_full_selected_member_skip_and_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    resume = tmp_path / "task_4.pt"
    states = {0: "complete", 1: "resume", 2: "fresh"}
    plan = shared.build_pilot_plan(
        config,
        member_ids=[0, 1],
        python="full-python",
        inspector=_inspector(states, resume),
    )

    assert plan.members[0].command is None
    assert plan.members[1].command is not None
    assert _arg(plan.members[1].command, "--resume-fold-checkpoint") == str(resume)
    assert plan.members[2].command is None


def test_full_members_are_sequential_then_all_eight_tasks_are_evaluated(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    states = {member: "fresh" for member in MEMBER_IDS}
    calls: list[tuple[str, int, str | None]] = []

    def runner(command: Sequence[str], cwd: Path) -> None:
        member = int(_arg(command, "--member-id"))
        calls.append(("train", member, None))
        root = Path(_arg(command, "--outdir"))
        root.mkdir(parents=True)
        (root / "run_config.json").write_text(
            json.dumps({"run_id": f"full-member-{member}"}), encoding="utf-8"
        )
        states[member] = "complete"

    def fake_validate(config, member):
        calls.append(("validate", member, None))

    def fake_ensemble(config, task, split, python, runner):
        calls.append(("ensemble", task, split))
        root = Path(config["output_root"]) / "offline_evaluation" / f"task_{task}"
        root.mkdir(parents=True, exist_ok=True)
        path = root / ("oof.npz" if split == "validation" else "test.npz")
        path.write_bytes(b"synthetic ensemble")
        return path

    def fake_thresholds(config, task, python, runner):
        calls.append(("threshold", task, None))
        root = Path(config["output_root"]) / "offline_evaluation" / f"task_{task}"
        for name in (
            "frozen_threshold.json",
            "oof_threshold_report.json",
            "test_threshold_report.json",
        ):
            (root / name).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(shared, "_validate_member_predictions", fake_validate)
    monkeypatch.setattr(shared, "_run_ensemble", fake_ensemble)
    monkeypatch.setattr(shared, "_run_thresholds", fake_thresholds)

    manifest_path = execute_full(
        config,
        python="full-python",
        runner=runner,
        inspector=_inspector(states),
    )

    assert [call for call in calls if call[0] == "train"] == [
        ("train", 0, None),
        ("train", 1, None),
        ("train", 2, None),
    ]
    first_eval = next(index for index, call in enumerate(calls) if call[0] == "ensemble")
    last_train = max(index for index, call in enumerate(calls) if call[0] == "train")
    assert first_eval > last_train
    assert [call[1] for call in calls if call[0] == "threshold"] == list(range(8))
    assert manifest_path is not None and manifest_path.name == "full_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == MANIFEST_KIND
    assert manifest["study_scope"] == {
        "completed_tasks": list(range(8)),
        "full_protocol_task_count": 8,
    }
    assert len(manifest["evaluations"]) == 8
