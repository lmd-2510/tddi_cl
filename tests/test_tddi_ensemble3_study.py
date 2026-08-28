from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

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
from src.training.tddi_ensemble3_study import (
    build_study_plan,
    execute_study,
    load_study_config,
    member_outdir,
    member_prediction_path,
    validate_all_member_alignment,
)
from src.utils.seed import derive_member_seed


def _write_study_fixture(root: Path):
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
                "protocol": "constrained_mass_balanced",
                "seed": 0,
                "tasks": [
                    {"task_id": 0, "classes": [30, 10]},
                    {"task_id": 1, "classes": [70]},
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = root / "study.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_name": "synthetic_tddi_ensemble3",
                "experiment_seed": 0,
                "member_ids": [0, 1, 2],
                "protocol": {
                    "id": "P4",
                    "name": "constrained_mass_balanced",
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
                    "method": "ewc",
                    "method_protocol": "ewc_natural_sampling",
                    "microbatch_size": 2,
                    "effective_batch_size": 4,
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "patience": 0,
                    "ewc_lambda": 1.0,
                    "focal_gamma": 1.0,
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
    if class_count == 2:
        probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    else:
        probabilities = np.full((class_count, class_count), 0.05, dtype=np.float32)
        np.fill_diagonal(probabilities, 0.9)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    logits = np.log(probabilities).astype(np.float32)
    return PredictionOutputs(
        logits=logits,
        probabilities=probabilities,
        predictions=class_ids.copy(),
        labels=class_ids.copy(),
        latent_features=np.zeros((class_count, 2), dtype=np.float32),
        class_ids=class_ids.copy(),
    )


def _export_synthetic_member_artifacts(config, member_id: int) -> None:
    outdir = member_outdir(config, member_id)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("run_summary.md", "metrics.csv", "forgetting.csv"):
        (outdir / name).write_text("complete\n", encoding="utf-8")
    (outdir / "run_config.json").write_text(
        json.dumps({"run_id": f"synthetic-member-{member_id}"}),
        encoding="utf-8",
    )
    checkpoint = outdir / "checkpoints" / "latest_ewc_state.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")

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
                    run_id=f"synthetic-member-{member_id}",
                    method="ewc",
                    method_protocol="ewc_natural_sampling",
                    task_id=task_id,
                    split=split,
                    member_id=member_id,
                    experiment_seed=0,
                    member_seed=derive_member_seed(0, member_id),
                ),
                member_prediction_path(config, member_id, task_id, split),
            )


def test_dry_run_is_side_effect_free_and_plans_member_then_ensemble_order(
    tmp_path: Path,
) -> None:
    config = _write_study_fixture(tmp_path)
    plan = build_study_plan(config, python_executable="study-python")

    assert not config.output_root.exists()
    assert [member.member_id for member in plan.members] == [0, 1, 2]
    assert [member.status for member in plan.members] == ["fresh", "fresh", "fresh"]
    assert [member.member_seed for member in plan.members] == [
        derive_member_seed(0, member_id) for member_id in range(3)
    ]
    for member in plan.members:
        assert member.command is not None
        assert _argument(member.command, "--method") == "ewc"
        assert _argument(member.command, "--variant") == "tddi_paper_member"
        assert _argument(member.command, "--seed") == "0"
        assert _argument(member.command, "--member-id") == str(member.member_id)
        assert "--export-member-predictions" in member.command
        assert "--resume-ewc-checkpoint" not in member.command
        assert not any(option.startswith("--max-") for option in member.command)

    assert [(item.task_id, item.split) for item in plan.ensembles] == [
        (0, "validation"),
        (0, "test"),
        (1, "validation"),
        (1, "test"),
    ]
    for ensemble in plan.ensembles:
        assert ensemble.command is not None
        member_argument = ensemble.command.index("--member-artifacts") + 1
        assert list(ensemble.command[member_argument : member_argument + 3]) == [
            str(member_prediction_path(config, member_id, ensemble.task_id, ensemble.split))
            for member_id in range(3)
        ]


def test_dry_run_detects_member_task_resume_without_touching_other_members(
    tmp_path: Path,
) -> None:
    config = _write_study_fixture(tmp_path)
    outdir = member_outdir(config, 1)
    checkpoint = outdir / "checkpoints" / "latest_ewc_state.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (outdir / "events.csv").write_text(
        "timestamp,run_id,event_type,message,payload_json\n"
        'now,run-1,ewc_checkpoint_saved,saved,"{""completed_task_id"": 0}"\n',
        encoding="utf-8",
    )

    plan = build_study_plan(
        config,
        python_executable="study-python",
        selected_member_ids=[1],
    )

    resumed = plan.members[1]
    assert resumed.status == "resume"
    assert resumed.completed_task_id == 0
    assert resumed.command is not None
    assert _argument(resumed.command, "--resume-ewc-checkpoint") == str(checkpoint)
    assert plan.members[0].command is None
    assert plan.members[2].command is None


def test_completed_member_is_skipped_and_incomplete_unresumable_run_is_refused(
    tmp_path: Path,
) -> None:
    config = _write_study_fixture(tmp_path)
    _export_synthetic_member_artifacts(config, 0)

    plan = build_study_plan(config, python_executable="study-python")
    assert plan.members[0].status == "complete"
    assert plan.members[0].command is None

    broken = member_outdir(config, 1)
    broken.mkdir(parents=True)
    (broken / "partial.log").write_text("interrupted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no resumable EWC checkpoint"):
        build_study_plan(config, python_executable="study-python")


def test_synthetic_execution_is_strictly_sequential_and_invokes_aligned_ensembles(
    tmp_path: Path,
) -> None:
    config = _write_study_fixture(tmp_path)
    calls: list[tuple[str, int, str | None]] = []
    active_train_member: int | None = None

    def synthetic_runner(command: Sequence[str], cwd: Path) -> None:
        nonlocal active_train_member
        assert cwd.is_dir()
        if "train_cil.py" in command[1]:
            member_id = int(_argument(command, "--member-id"))
            assert active_train_member is None
            active_train_member = member_id
            calls.append(("train", member_id, None))
            _export_synthetic_member_artifacts(config, member_id)
            active_train_member = None
            return

        assert "offline_ensemble.py" in command[1]
        assert active_train_member is None
        assert [call[1] for call in calls if call[0] == "train"] == [0, 1, 2]
        source_start = command.index("--member-artifacts") + 1
        source_paths = [Path(value) for value in command[source_start : source_start + 3]]
        artifacts = [load_member_prediction_artifact(path) for path in source_paths]
        task_id = artifacts[0].context.task_id
        split = artifacts[0].context.split
        calls.append(("ensemble", task_id, split))
        aggregate = aggregate_member_predictions(
            artifacts,
            source_artifact_paths=source_paths,
        )
        export_offline_ensemble_artifact(
            aggregate,
            Path(_argument(command, "--out")),
        )

    manifest_path = execute_study(
        config,
        python_executable="study-python",
        runner=synthetic_runner,
    )

    assert manifest_path == config.output_root / "study_manifest.json"
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
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["execution_policy"] == "sequential_members_never_concurrent"
    assert payload["execution_order"] == [0, 1, 2]
    assert [member["member_id"] for member in payload["members"]] == [0, 1, 2]
    assert len(payload["ensemble_groups"]) == 4
    for group in payload["ensemble_groups"]:
        assert [item["member_id"] for item in group["member_artifacts"]] == [0, 1, 2]
        assert group["raw_class_ids"] == (
            [10, 30] if group["task_id"] == 0 else [10, 30, 70]
        )


def test_alignment_failure_prevents_any_ensemble_invocation(tmp_path: Path) -> None:
    config = _write_study_fixture(tmp_path)
    for member_id in range(3):
        _export_synthetic_member_artifacts(config, member_id)

    path = member_prediction_path(config, 2, 1, "validation")
    with np.load(path, allow_pickle=False) as original:
        payload = {key: np.asarray(original[key]) for key in original.files}
    payload["raw_class_ids"] = payload["raw_class_ids"][::-1].copy()
    payload["logits"] = payload["logits"][:, ::-1].copy()
    payload["probabilities"] = payload["probabilities"][:, ::-1].copy()
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    with pytest.raises(ValueError, match="raw_class_ids mismatch"):
        validate_all_member_alignment(config)
