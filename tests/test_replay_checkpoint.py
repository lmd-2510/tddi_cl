from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.fixed_budget_replay import FixedBudgetReplayBuffer, FixedReplaySampler
from src.training.ewc_checkpoint import restore_rng_state
from src.training.replay_checkpoint import (
    REPLAY_CHECKPOINT_SCHEMA_VERSION,
    load_replay_checkpoint,
    restore_fixed_replay_buffer,
    restore_replay_model,
    save_replay_checkpoint,
)
from src.training.train_cil import expand_model_for_seen_classes, main


SEED_METADATA = {
    "experiment_seed": 0,
    "member_id": 1,
    "member_seed": 12345,
    "member_seed_derivation": "test-derivation",
    "seed_mode": "ensemble_member",
}
CHECKPOINT_CONFIG = {
    "method": "replay_distill_fixed_budget_uniform",
    "variant": "small",
    "task_file_sha256": "task-hash",
    "total_memory_budget": 4,
    "replay_draws_per_epoch": 4,
}


def _model(class_map: dict[int, int]) -> torch.nn.Module:
    return expand_model_for_seen_classes(
        previous_model=None,
        previous_seen_map=None,
        current_seen_map=class_map,
        variant="small",
        input_dim=2,
        dropout=0.0,
        activation="relu",
        norm="none",
    )


def _buffer() -> FixedBudgetReplayBuffer:
    buffer = FixedBudgetReplayBuffer(total_memory_budget=4, random_seed=0)
    buffer.update(
        np.asarray(
            [[-1.0, -1.0], [-0.8, -0.9], [0.8, 0.9], [1.0, 1.0]],
            dtype=np.float32,
        ),
        np.asarray([10, 10, 20, 20], dtype=np.int64),
    )
    return buffer


def _save_boundary(path: Path) -> Path:
    class_map = {10: 0, 20: 1}
    return save_replay_checkpoint(
        path,
        run_id="run-001",
        completed_task_id=0,
        model_state=_model(class_map).state_dict(),
        seen_class_map=class_map,
        seen_raw_classes=[10, 20],
        replay_buffer=_buffer(),
        seed_metadata=SEED_METADATA,
        config=CHECKPOINT_CONFIG,
        progress={"marker": "task-zero-complete"},
    )


def test_replay_checkpoint_round_trip_is_atomic_and_omits_teacher_optimizer(
    tmp_path: Path,
) -> None:
    checkpoint_path = _save_boundary(tmp_path / "checkpoints" / "latest.pt")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert raw["schema_version"] == REPLAY_CHECKPOINT_SCHEMA_VERSION
    assert raw["completed_task_id"] == 0
    assert raw["next_task_id"] == 1
    assert "teacher_model" not in raw
    assert "optimizer_state" not in raw
    assert not list(checkpoint_path.parent.glob(".*.tmp"))

    loaded = load_replay_checkpoint(
        checkpoint_path,
        expected_seed_metadata=SEED_METADATA,
        expected_config=CHECKPOINT_CONFIG,
        expected_seen_class_map={10: 0, 20: 1},
    )
    restored_buffer = restore_fixed_replay_buffer(
        loaded.replay_buffer_state,
        expected_seen_class_map=loaded.seen_class_map,
        expected_total_memory_budget=4,
        expected_random_seed=0,
    )
    restored_model = restore_replay_model(_model(loaded.seen_class_map), loaded)

    expected_features, expected_labels = _buffer().get_all()
    actual_features, actual_labels = restored_buffer.get_all()
    np.testing.assert_array_equal(actual_features, expected_features)
    np.testing.assert_array_equal(actual_labels, expected_labels)
    assert restored_buffer.available_count_by_class == {10: 2, 20: 2}
    assert loaded.seen_raw_classes == (10, 20)
    assert loaded.progress == {"marker": "task-zero-complete"}
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, loaded.model_state[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("load_kwargs", "message"),
    [
        (
            {"expected_seed_metadata": {**SEED_METADATA, "member_id": 2}},
            "seed metadata mismatch",
        ),
        (
            {
                "expected_config": {
                    **CHECKPOINT_CONFIG,
                    "total_memory_budget": 8,
                }
            },
            "resume config mismatch",
        ),
        (
            {"expected_config": {**CHECKPOINT_CONFIG, "variant": "medium"}},
            "resume config mismatch",
        ),
        (
            {
                "expected_config": {
                    **CHECKPOINT_CONFIG,
                    "task_file_sha256": "different-task-hash",
                }
            },
            "resume config mismatch",
        ),
        (
            {
                "expected_config": {
                    **CHECKPOINT_CONFIG,
                    "replay_draws_per_epoch": 8,
                }
            },
            "resume config mismatch",
        ),
        (
            {"expected_seen_class_map": {10: 0, 30: 1}},
            "seen_class_map does not match",
        ),
    ],
)
def test_replay_checkpoint_rejects_resume_contract_mismatches(
    tmp_path: Path,
    load_kwargs: dict[str, object],
    message: str,
) -> None:
    checkpoint_path = _save_boundary(tmp_path / "latest.pt")
    with pytest.raises(ValueError, match=message):
        load_replay_checkpoint(checkpoint_path, **load_kwargs)


def test_replay_resume_reconstructs_teacher_and_next_sampler_deterministically(
    tmp_path: Path,
) -> None:
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    checkpoint_path = _save_boundary(tmp_path / "latest.pt")
    loaded = load_replay_checkpoint(checkpoint_path)

    continuous_teacher = _model(loaded.seen_class_map)
    continuous_teacher.load_state_dict(loaded.model_state)
    continuous_sampler = FixedReplaySampler(
        current_count=3,
        replay_raw_labels=loaded.replay_buffer_state["raw_labels"],
        replay_draws_per_epoch=4,
        seed=int(SEED_METADATA["member_seed"]),
        task_id=1,
    )
    continuous_order = list(iter(continuous_sampler))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed_teacher = restore_replay_model(_model(loaded.seen_class_map), loaded)
    resumed_teacher.eval()
    for parameter in resumed_teacher.parameters():
        parameter.requires_grad_(False)
    restore_rng_state(loaded.rng_state)
    resumed_sampler = FixedReplaySampler(
        current_count=3,
        replay_raw_labels=loaded.replay_buffer_state["raw_labels"],
        replay_draws_per_epoch=4,
        seed=int(loaded.seed_metadata["member_seed"]),
        task_id=loaded.next_task_id,
    )

    assert list(iter(resumed_sampler)) == continuous_order
    assert not resumed_teacher.training
    assert all(not parameter.requires_grad for parameter in resumed_teacher.parameters())
    for name, value in continuous_teacher.state_dict().items():
        torch.testing.assert_close(value, resumed_teacher.state_dict()[name], rtol=0, atol=0)


def test_replay_checkpoint_rejects_unsupported_schema(tmp_path: Path) -> None:
    checkpoint_path = _save_boundary(tmp_path / "latest.pt")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["schema_version"] = REPLAY_CHECKPOINT_SCHEMA_VERSION + 1
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="Unsupported replay checkpoint schema version"):
        load_replay_checkpoint(checkpoint_path)


def _write_synthetic_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    feature_columns = ["f0", "f1"]
    train = pd.DataFrame(
        {
            "f0": [-2.0, -1.8, -1.6, -1.4, 0.0, 0.2, 0.4, 0.6, 1.4, 1.6, 1.8, 2.0],
            "f1": [-1.0, -0.9, -0.8, -0.7, 0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0],
            "class": [10] * 4 + [20] * 4 + [30] * 4,
        }
    )
    validation = train.groupby("class", sort=True).head(2).reset_index(drop=True)
    test = train.groupby("class", sort=True).tail(2).reset_index(drop=True)
    paths: list[Path] = []
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        path = root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
    feature_path = root / "features.json"
    feature_path.write_text(json.dumps(feature_columns), encoding="utf-8")
    scaler_path = root / "scaler.pkl"
    with scaler_path.open("wb") as handle:
        pickle.dump(
            {
                "scaler_type": "standard",
                "mean": np.zeros(2),
                "scale": np.ones(2),
                "impute_values": np.zeros(2),
            },
            handle,
        )
    task_path = root / "tasks.json"
    task_path.write_text(
        json.dumps(
            {
                "protocol": "replay_resume_synthetic",
                "seed": 0,
                "tasks": [
                    {"task_id": 0, "classes": [10, 20]},
                    {"task_id": 1, "classes": [30]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return (*paths, feature_path, scaler_path, task_path)


def _argv(inputs: tuple[Path, ...], outdir: Path) -> list[str]:
    train, validation, test, features, scaler, tasks = inputs
    return [
        "train_cil.py",
        "--train", str(train),
        "--validation", str(validation),
        "--test", str(test),
        "--feature-cols", str(features),
        "--scaler", str(scaler),
        "--task-file", str(tasks),
        "--outdir", str(outdir),
        "--method", "replay_distill_fixed_budget_uniform",
        "--variant", "small",
        "--batch-size", "4",
        "--effective-batch-size", "4",
        "--epochs", "1",
        "--patience", "1",
        "--device", "cpu",
        "--dropout", "0",
        "--norm", "none",
        "--focal-gamma", "0",
        "--total-memory-budget", "6",
        "--replay-draws-per-epoch", "4",
        "--seed", "0",
        "--member-id", "1",
    ]


def _normalized_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "run_id" in frame:
        frame = frame.drop(columns=["run_id"])
    return frame


def test_continuous_and_task_boundary_resume_match_for_two_task_training(
    tmp_path: Path,
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    continuous_out = tmp_path / "continuous"
    resumed_out = tmp_path / "resumed"

    with patch.object(sys, "argv", _argv(inputs, continuous_out)):
        main()

    from src.training import train_cil

    real_save = train_cil.save_replay_checkpoint

    class BoundaryReached(RuntimeError):
        pass

    def stop_after_first_boundary(*args, **kwargs):
        result = real_save(*args, **kwargs)
        if int(kwargs["completed_task_id"]) == 0:
            raise BoundaryReached
        return result

    with patch.object(train_cil, "save_replay_checkpoint", stop_after_first_boundary):
        with patch.object(sys, "argv", _argv(inputs, resumed_out)):
            with pytest.raises(BoundaryReached):
                main()

    resume_path = resumed_out / "checkpoints" / "latest_replay_distill_state.pt"
    completed_task_model = resumed_out / "checkpoints" / "task_0_model.pt"
    completed_task_model_bytes = completed_task_model.read_bytes()
    resume_argv = [
        *_argv(inputs, resumed_out),
        "--resume-replay-checkpoint",
        str(resume_path),
    ]
    with patch.object(sys, "argv", resume_argv):
        main()

    assert completed_task_model.read_bytes() == completed_task_model_bytes

    continuous_checkpoint = load_replay_checkpoint(
        continuous_out / "checkpoints" / "latest_replay_distill_state.pt"
    )
    resumed_checkpoint = load_replay_checkpoint(resume_path)
    assert continuous_checkpoint.completed_task_id == resumed_checkpoint.completed_task_id == 1
    assert continuous_checkpoint.next_task_id == resumed_checkpoint.next_task_id == 2
    assert continuous_checkpoint.seen_class_map == resumed_checkpoint.seen_class_map
    assert continuous_checkpoint.seen_raw_classes == resumed_checkpoint.seen_raw_classes
    best_task_state = torch.load(
        resumed_out / "checkpoints" / "task_1_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_checkpoint.model_state["head.weight"].shape[0] == 3
    for name, value in continuous_checkpoint.model_state.items():
        torch.testing.assert_close(value, resumed_checkpoint.model_state[name], rtol=0, atol=0)
        torch.testing.assert_close(
            resumed_checkpoint.model_state[name], best_task_state[name], rtol=0, atol=0
        )
    np.testing.assert_array_equal(
        continuous_checkpoint.replay_buffer_state["features"],
        resumed_checkpoint.replay_buffer_state["features"],
    )
    np.testing.assert_array_equal(
        continuous_checkpoint.replay_buffer_state["raw_labels"],
        resumed_checkpoint.replay_buffer_state["raw_labels"],
    )
    assert continuous_checkpoint.replay_buffer_state["available_count_by_class"] == (
        resumed_checkpoint.replay_buffer_state["available_count_by_class"]
    )
    for relative in (
        "metrics.csv",
        "task_matrix.csv",
        "training_audit.csv",
        "replay_budget_audit.csv",
        "class_trajectory.csv",
    ):
        pd.testing.assert_frame_equal(
            _normalized_frame(continuous_out / relative),
            _normalized_frame(resumed_out / relative),
            check_exact=True,
        )
    assert not list(resumed_out.glob("checkpoints/task_*_teacher.pt"))
