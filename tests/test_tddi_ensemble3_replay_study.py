from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import torch

from src.data.fixed_budget_replay import FixedBudgetReplayBuffer
from src.eval.cil_evaluation import PredictionOutputs
from src.eval.member_predictions import (
    MemberPredictionContext,
    export_member_prediction_artifact,
    load_member_prediction_artifact,
)
from src.eval.offline_ensemble import (
    aggregate_member_predictions,
    export_offline_ensemble_artifact,
)
from src.eval.s02_artifacts import DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN
from src.training.replay_checkpoint import save_replay_checkpoint
from src.training.tddi_ensemble3_study import (
    REPLAY_CHECKPOINT_POLICY,
    build_study_plan,
    execute_study,
    load_study_config,
    main as study_main,
    member_outdir,
    member_prediction_path,
)
from src.utils.seed import MEMBER_SEED_DERIVATION, derive_member_seed


REPLAY_METHOD = "replay_distill_fixed_budget_uniform"


def _write_replay_study_fixture(root: Path):
    for name in (
        "train.parquet",
        "validation.parquet",
        "test.parquet",
        "features.json",
        "scaler.pkl",
    ):
        (root / name).write_bytes(b"synthetic-input")
    task_file = root / "tasks.json"
    task_file.write_text(
        json.dumps(
            {
                "protocol": "tail_to_head",
                "seed": None,
                "tasks": [
                    {"task_id": 0, "classes": [30, 10]},
                    {"task_id": 1, "classes": [70]},
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = root / "replay_study.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_name": "synthetic_tddi_replay_ensemble3",
                "experiment_seed": 0,
                "member_ids": [0, 1, 2],
                "protocol": {
                    "id": "P3",
                    "name": "tail_to_head",
                    "expected_task_count": 2,
                    "task_file": str(task_file),
                },
                "inputs": {
                    "train": str(root / "train.parquet"),
                    "validation": str(root / "validation.parquet"),
                    "test": str(root / "test.parquet"),
                    "feature_columns": str(root / "features.json"),
                    "scaler": str(root / "scaler.pkl"),
                },
                "model": {
                    "variant": "tddi_paper_member",
                    "dropout": 0.2,
                    "activation": "gelu",
                    "normalization": "layernorm",
                },
                "training": {
                    "method": REPLAY_METHOD,
                    "method_protocol": REPLAY_METHOD,
                    "optimizer": "adamw",
                    "microbatch_size": 2,
                    "effective_batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "patience": 0,
                    "focal_gamma": 1.0,
                    "total_memory_budget": 6800,
                    "replay_draws_per_epoch": 6800,
                    "replay_starts_at_task": 1,
                    "distill_alpha": 1.0,
                    "temperature": 2.0,
                    "feature_distill_weight": 0.5,
                    "checkpoint_resume_policy": REPLAY_CHECKPOINT_POLICY,
                    "device": "cpu",
                },
                "prediction_splits": ["validation", "test"],
                "outputs": {
                    "root": str(root / "study_output"),
                    "member_namespace": "member_{member_id}",
                    "ensemble_namespace": "offline_ensemble",
                    "manifest": "study_manifest.json",
                },
            }
        ),
        encoding="utf-8",
    )
    return load_study_config(config_path, project_root=root)


def _argument(command: Sequence[str], name: str) -> str:
    return command[command.index(name) + 1]


def _prediction_outputs(class_ids: np.ndarray) -> PredictionOutputs:
    class_count = class_ids.shape[0]
    probabilities = np.full((class_count, class_count), 0.05, dtype=np.float32)
    np.fill_diagonal(probabilities, 0.9)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return PredictionOutputs(
        logits=np.log(probabilities).astype(np.float32),
        probabilities=probabilities,
        predictions=class_ids.copy(),
        labels=class_ids.copy(),
        latent_features=np.zeros((class_count, 2), dtype=np.float32),
        class_ids=class_ids.copy(),
    )


def _export_completed_member(config, member_id: int) -> None:
    outdir = member_outdir(config, member_id)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("run_summary.md", "metrics.csv", "forgetting.csv"):
        (outdir / name).write_text("complete\n", encoding="utf-8")
    (outdir / "run_config.json").write_text(
        json.dumps({"run_id": f"replay-member-{member_id}"}),
        encoding="utf-8",
    )
    seen: set[int] = set()
    for task in config.tasks:
        task_id = int(task["task_id"])
        seen.update(int(raw_class_id) for raw_class_id in task["classes"])
        class_ids = np.asarray(sorted(seen), dtype=np.int64)
        outputs = _prediction_outputs(class_ids)
        metadata = {
            DRUG_ID_A_COLUMN: np.asarray(
                [f"drug-a-{index}" for index in range(class_ids.shape[0])]
            ),
            DRUG_ID_B_COLUMN: np.asarray(
                [f"drug-b-{index}" for index in range(class_ids.shape[0])]
            ),
        }
        for split in config.prediction_splits:
            export_member_prediction_artifact(
                outputs,
                metadata,
                MemberPredictionContext(
                    run_id=f"replay-member-{member_id}",
                    method=REPLAY_METHOD,
                    method_protocol=REPLAY_METHOD,
                    task_id=task_id,
                    split=split,
                    member_id=member_id,
                    experiment_seed=0,
                    member_seed=derive_member_seed(0, member_id),
                ),
                member_prediction_path(config, member_id, task_id, split),
            )


def _write_valid_resume_checkpoint(config, member_id: int) -> Path:
    outdir = member_outdir(config, member_id)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = f"resume-replay-member-{member_id}"
    (outdir / "run_config.json").write_text(
        json.dumps({"run_id": run_id}),
        encoding="utf-8",
    )
    buffer = FixedBudgetReplayBuffer(total_memory_budget=6800, random_seed=0)
    buffer.update(
        np.asarray(
            [[-1.0, -1.0], [-0.8, -0.9], [0.8, 0.9], [1.0, 1.0]],
            dtype=np.float32,
        ),
        np.asarray([10, 10, 30, 30], dtype=np.int64),
    )
    model = torch.nn.Linear(2, 2)
    checkpoint_config = {
        "method": REPLAY_METHOD,
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
    checkpoint_path = outdir / "checkpoints" / "latest_replay_distill_state.pt"
    save_replay_checkpoint(
        checkpoint_path,
        run_id=run_id,
        completed_task_id=0,
        model_state=model.state_dict(),
        seen_class_map={10: 0, 30: 1},
        seen_raw_classes=[10, 30],
        replay_buffer=buffer,
        seed_metadata={
            "experiment_seed": 0,
            "member_id": member_id,
            "member_seed": derive_member_seed(0, member_id),
            "member_seed_derivation": MEMBER_SEED_DERIVATION,
            "seed_mode": "member",
        },
        config=checkpoint_config,
        progress={
            "result_matrix": np.full((2, 2), np.nan, dtype=np.float64),
            "result_rows": [],
            "best_task_rows": [{"task_id": 0}],
            "training_audit_rows": [{"task": 0}],
            "replay_budget_audit_rows": [],
            "class_trajectory_rows": [],
        },
    )
    return checkpoint_path


def test_replay_dry_run_validates_method_config_and_selected_member(tmp_path: Path) -> None:
    config = _write_replay_study_fixture(tmp_path)
    plan = build_study_plan(
        config,
        python_executable="study-python",
        selected_member_ids=[1],
    )

    assert not config.output_root.exists()
    assert config.method == REPLAY_METHOD
    assert config.optimizer == "adamw"
    assert config.gradient_accumulation_steps == 2
    assert [member.member_id for member in plan.members] == [0, 1, 2]
    assert plan.members[0].command is None
    assert plan.members[2].command is None
    command = plan.members[1].command
    assert command is not None
    assert _argument(command, "--method") == REPLAY_METHOD
    assert _argument(command, "--member-id") == "1"
    assert _argument(command, "--total-memory-budget") == "6800"
    assert _argument(command, "--replay-draws-per-epoch") == "6800"
    assert _argument(command, "--distill-alpha") == "1.0"
    assert _argument(command, "--temperature") == "2.0"
    assert _argument(command, "--feature-distill-weight") == "0.5"
    assert "--resume-replay-checkpoint" not in command
    assert "--resume-ewc-checkpoint" not in command


def test_replay_cli_dry_run_is_side_effect_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_replay_study_fixture(tmp_path)

    study_main(
        [
            "--config",
            str(config.source_path),
            "--member-id",
            "0",
        ]
    )

    output = capsys.readouterr().out
    assert "Execution policy: sequential members" in output
    assert "member=0" in output
    assert "--member-id 0" in output
    assert "--execute" not in output
    assert not config.output_root.exists()


def test_full8_p3_config_is_locked_to_requested_contract() -> None:
    config = load_study_config(
        Path("configs/tddi_ensemble3_replay_distill_p3_seed0.json")
    )

    assert config.task_file.as_posix().endswith(
        "study_assets/task_protocols/tail_to_head_tasks.json"
    )
    assert config.protocol_id == "P3"
    assert config.protocol_name == "tail_to_head"
    assert "configs/smoke" not in config.task_file.as_posix()
    assert config.expected_task_count == 8
    assert [len(task["classes"]) for task in config.tasks] == [
        38,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
    ]
    assert len({int(value) for task in config.tasks for value in task["classes"]}) == 178
    assert config.optimizer == "adamw"
    assert config.microbatch_size == 64
    assert config.effective_batch_size == 1024
    assert config.gradient_accumulation_steps == 16
    assert config.epochs == 20
    assert config.patience == 5
    assert config.learning_rate == 0.001
    assert config.weight_decay == 0.0001
    assert config.activation == "gelu"
    assert config.dropout == 0.2
    assert config.normalization == "layernorm"
    assert config.focal_gamma == 1.0
    assert config.distill_alpha == 1.0
    assert config.temperature == 2.0
    assert config.feature_distill_weight == 0.5
    assert config.total_memory_budget == 6800
    assert config.replay_draws_per_epoch == 6800
    assert config.prediction_splits == ("validation", "test")
    assert config.member_ids == (0, 1, 2)
    assert len({derive_member_seed(0, member_id) for member_id in config.member_ids}) == 3
    assert "outputs/full/" in config.output_root.as_posix()
    assert config.output_root.name.endswith("_v1")
    assert "smoke" not in config.output_root.as_posix().casefold()


def test_replay_config_rejects_wrong_budget_and_method_protocol(tmp_path: Path) -> None:
    config = _write_replay_study_fixture(tmp_path)
    payload = json.loads(config.source_path.read_text(encoding="utf-8"))
    payload["training"]["total_memory_budget"] = 100
    config.source_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="total_memory_budget must equal 6800"):
        load_study_config(config.source_path, project_root=tmp_path)

    payload["training"]["total_memory_budget"] = 6800
    payload["training"]["method_protocol"] = "ewc_natural_sampling"
    config.source_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Replay ensemble requires method_protocol"):
        load_study_config(config.source_path, project_root=tmp_path)


def test_replay_incomplete_member_resumes_only_from_valid_checkpoint(
    tmp_path: Path,
) -> None:
    config = _write_replay_study_fixture(tmp_path)
    checkpoint = _write_valid_resume_checkpoint(config, 1)
    plan = build_study_plan(
        config,
        python_executable="study-python",
        selected_member_ids=[1],
    )
    member = plan.members[1]
    assert member.status == "resume"
    assert member.completed_task_id == 0
    assert member.resume_checkpoint == checkpoint
    assert member.command is not None
    assert _argument(member.command, "--resume-replay-checkpoint") == str(checkpoint)

    broken = member_outdir(config, 2)
    broken.mkdir(parents=True)
    (broken / "partial.log").write_text("interrupted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no resumable replay checkpoint"):
        build_study_plan(config, python_executable="study-python")


def test_replay_invalid_checkpoint_fails_without_overwrite(tmp_path: Path) -> None:
    config = _write_replay_study_fixture(tmp_path)
    outdir = member_outdir(config, 0)
    checkpoint = outdir / "checkpoints" / "latest_replay_distill_state.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not-a-checkpoint")
    with pytest.raises(RuntimeError, match="invalid replay checkpoint"):
        build_study_plan(config, python_executable="study-python")
    assert checkpoint.read_bytes() == b"not-a-checkpoint"


def test_replay_selected_member_runs_alone_and_ensemble_waits(tmp_path: Path) -> None:
    config = _write_replay_study_fixture(tmp_path)
    calls: list[tuple[str, int]] = []

    def runner(command: Sequence[str], cwd: Path) -> None:
        assert cwd.is_dir()
        assert "train_cil.py" in command[1]
        member_id = int(_argument(command, "--member-id"))
        calls.append(("train", member_id))
        _export_completed_member(config, member_id)

    result = execute_study(
        config,
        python_executable="study-python",
        selected_member_ids=[1],
        runner=runner,
    )
    assert result is None
    assert calls == [("train", 1)]
    assert not (config.output_root / config.ensemble_namespace).exists()


def test_replay_execution_is_0_1_2_then_aligned_ensemble_and_manifest(
    tmp_path: Path,
) -> None:
    config = _write_replay_study_fixture(tmp_path)
    calls: list[tuple[str, int, str | None]] = []
    active_member: int | None = None

    def runner(command: Sequence[str], cwd: Path) -> None:
        nonlocal active_member
        assert cwd.is_dir()
        if "train_cil.py" in command[1]:
            member_id = int(_argument(command, "--member-id"))
            assert active_member is None
            active_member = member_id
            calls.append(("train", member_id, None))
            _export_completed_member(config, member_id)
            active_member = None
            return

        assert "offline_ensemble.py" in command[1]
        assert active_member is None
        assert [call[1] for call in calls if call[0] == "train"] == [0, 1, 2]
        source_start = command.index("--member-artifacts") + 1
        paths = [Path(value) for value in command[source_start : source_start + 3]]
        artifacts = [load_member_prediction_artifact(path) for path in paths]
        task_id = artifacts[0].context.task_id
        split = artifacts[0].context.split
        calls.append(("ensemble", task_id, split))
        aggregate = aggregate_member_predictions(
            artifacts,
            source_artifact_paths=paths,
        )
        export_offline_ensemble_artifact(
            aggregate,
            Path(_argument(command, "--out")),
        )

    manifest_path = execute_study(
        config,
        python_executable="study-python",
        runner=runner,
    )
    assert [call for call in calls if call[0] == "train"] == [
        ("train", 0, None),
        ("train", 1, None),
        ("train", 2, None),
    ]
    assert [call for call in calls if call[0] == "ensemble"] == [
        ("ensemble", 0, "validation"),
        ("ensemble", 0, "test"),
        ("ensemble", 1, "validation"),
        ("ensemble", 1, "test"),
    ]
    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["method"] == REPLAY_METHOD
    assert manifest["protocol"]["task_file_sha256"] == config.task_file_sha256
    assert manifest["execution_order"] == [0, 1, 2]
    assert len(manifest["members"]) == 3
    assert len(manifest["ensemble_groups"]) == 4
    for member in manifest["members"]:
        assert Path(member["run_config"]).is_file()
        assert member["run_config_sha256"]
    for group in manifest["ensemble_groups"]:
        assert [item["member_id"] for item in group["member_artifacts"]] == [0, 1, 2]
        assert Path(group["offline_ensemble"]["path"]).is_file()
        assert group["raw_class_ids"] == (
            [10, 30] if group["task_id"] == 0 else [10, 30, 70]
        )


def test_replay_complete_member_is_skipped(tmp_path: Path) -> None:
    config = _write_replay_study_fixture(tmp_path)
    _export_completed_member(config, 0)
    plan = build_study_plan(config, python_executable="study-python")
    assert plan.members[0].status == "complete"
    assert plan.members[0].command is None


def test_replay_execution_skips_complete_and_executes_resume_command(
    tmp_path: Path,
) -> None:
    config = _write_replay_study_fixture(tmp_path)
    _export_completed_member(config, 0)
    checkpoint = _write_valid_resume_checkpoint(config, 1)
    calls: list[int] = []

    def runner(command: Sequence[str], cwd: Path) -> None:
        assert cwd.is_dir()
        member_id = int(_argument(command, "--member-id"))
        calls.append(member_id)
        assert member_id == 1
        assert _argument(command, "--resume-replay-checkpoint") == str(checkpoint)
        _export_completed_member(config, member_id)

    result = execute_study(
        config,
        python_executable="study-python",
        selected_member_ids=[0, 1],
        runner=runner,
    )
    assert result is None
    assert calls == [1]
