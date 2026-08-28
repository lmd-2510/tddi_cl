from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.training.ewc_checkpoint import (
    EWC_CHECKPOINT_SCHEMA_VERSION,
    load_ewc_checkpoint,
    restore_model_and_theta_star,
    restore_rng_state,
    save_ewc_checkpoint,
)
from src.training.train_cil import expand_model_for_seen_classes


SEED_METADATA = {
    "experiment_seed": 0,
    "member_id": 1,
    "member_seed": 12345,
    "member_seed_derivation": "test-derivation",
    "seed_mode": "ensemble_member",
}
CHECKPOINT_CONFIG = {
    "method": "ewc",
    "variant": "small",
    "task_file_sha256": "task-hash",
}


def _model(class_map: dict[int, int]):
    return expand_model_for_seen_classes(
        previous_model=None,
        previous_seen_map=None,
        current_seen_map=class_map,
        variant="small",
        input_dim=4,
        dropout=0.0,
        activation="relu",
        norm="layernorm",
    )


def _fisher(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.full_like(parameter, float(position + 1) / 10.0)
        for position, (name, parameter) in enumerate(model.named_parameters())
    }


def _save_boundary(path: Path, model: torch.nn.Module, class_map: dict[int, int]) -> Path:
    return save_ewc_checkpoint(
        path,
        run_id="run-001",
        completed_task_id=0,
        model_state=model.state_dict(),
        fisher_total=_fisher(model),
        seen_class_map=class_map,
        seed_metadata=SEED_METADATA,
        config=CHECKPOINT_CONFIG,
        progress={"marker": "task-zero-complete"},
    )


def test_ewc_checkpoint_round_trip_is_versioned_atomic_and_omits_theta_star(
    tmp_path: Path,
) -> None:
    class_map = {10: 0, 30: 1}
    model = _model(class_map)
    checkpoint_path = _save_boundary(tmp_path / "checkpoints" / "latest.pt", model, class_map)

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert raw["schema_version"] == EWC_CHECKPOINT_SCHEMA_VERSION
    assert "theta_star" not in raw
    assert not list(checkpoint_path.parent.glob(".*.tmp"))

    loaded = load_ewc_checkpoint(
        checkpoint_path,
        expected_seed_metadata=SEED_METADATA,
        expected_config=CHECKPOINT_CONFIG,
        expected_seen_class_map=class_map,
    )
    restored = _model(class_map)
    theta_star = restore_model_and_theta_star(restored, loaded)

    assert loaded.completed_task_id == 0
    assert loaded.next_task_id == 1
    assert loaded.seen_class_map == class_map
    assert loaded.progress == {"marker": "task-zero-complete"}
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, loaded.model_state[name], rtol=0, atol=0)
        torch.testing.assert_close(theta_star[name], loaded.model_state[name], rtol=0, atol=0)
        torch.testing.assert_close(loaded.fisher_total[name], _fisher(model)[name])


@pytest.mark.parametrize(
    ("load_kwargs", "message"),
    [
        (
            {"expected_seed_metadata": {**SEED_METADATA, "member_id": 2}},
            "seed metadata mismatch",
        ),
        (
            {"expected_config": {**CHECKPOINT_CONFIG, "variant": "medium"}},
            "resume config mismatch",
        ),
        (
            {"expected_seen_class_map": {10: 0, 20: 1}},
            "seen_class_map does not match",
        ),
    ],
)
def test_ewc_checkpoint_rejects_resume_contract_mismatches(
    tmp_path: Path,
    load_kwargs: dict[str, object],
    message: str,
) -> None:
    class_map = {10: 0, 30: 1}
    checkpoint_path = _save_boundary(tmp_path / "latest.pt", _model(class_map), class_map)

    with pytest.raises(ValueError, match=message):
        load_ewc_checkpoint(checkpoint_path, **load_kwargs)


def test_save_resume_matches_continuous_next_task_initialization_and_ewc_state(
    tmp_path: Path,
) -> None:
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    previous_map = {10: 0, 30: 1}
    next_map = {10: 0, 20: 1, 30: 2}
    previous_model = _model(previous_map)
    checkpoint_path = _save_boundary(tmp_path / "latest.pt", previous_model, previous_map)

    continuous_model = expand_model_for_seen_classes(
        previous_model=previous_model,
        previous_seen_map=previous_map,
        current_seen_map=next_map,
        variant="small",
        input_dim=4,
        dropout=0.0,
        activation="relu",
        norm="layernorm",
    )
    continuous_next_random = torch.rand(5)

    # Simulate a fresh process whose startup and model reconstruction consumed RNG.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    loaded = load_ewc_checkpoint(checkpoint_path)
    resumed_previous_model = _model(previous_map)
    resumed_theta_star = restore_model_and_theta_star(resumed_previous_model, loaded)
    restore_rng_state(loaded.rng_state)
    resumed_model = expand_model_for_seen_classes(
        previous_model=resumed_previous_model,
        previous_seen_map=loaded.seen_class_map,
        current_seen_map=next_map,
        variant="small",
        input_dim=4,
        dropout=0.0,
        activation="relu",
        norm="layernorm",
    )
    resumed_next_random = torch.rand(5)

    assert resumed_model.head.out_features == continuous_model.head.out_features == 3
    assert loaded.next_task_id == 1
    for name, value in continuous_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
    for name, value in loaded.fisher_total.items():
        torch.testing.assert_close(value, _fisher(previous_model)[name], rtol=0, atol=0)
    for name, value in resumed_theta_star.items():
        torch.testing.assert_close(value, previous_model.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(continuous_next_random, resumed_next_random, rtol=0, atol=0)


def test_ewc_checkpoint_rejects_unsupported_schema(tmp_path: Path) -> None:
    class_map = {10: 0, 30: 1}
    checkpoint_path = _save_boundary(tmp_path / "latest.pt", _model(class_map), class_map)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["schema_version"] = EWC_CHECKPOINT_SCHEMA_VERSION + 1
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="Unsupported EWC checkpoint schema version"):
        load_ewc_checkpoint(checkpoint_path)
