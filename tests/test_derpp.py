"""Small deterministic checks for online DER++ and its resumable artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.methods.derpp import ReservoirLogitBuffer, observe
from src.training.derpp_fold import _preserve_incomplete_task


def test_reservoir_checkpoint_restores_future_sampling() -> None:
    first = ReservoirLogitBuffer(3, 2, 4, seed=19)
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    labels = np.arange(6) % 4
    logits = np.arange(24, dtype=np.float32).reshape(6, 4)
    first.add_batch(features[:4], labels[:4], logits[:4], [f"id-{i}" for i in range(4)])
    restored = ReservoirLogitBuffer(3, 2, 4, seed=99)
    restored.load_state_dict(first.state_dict())
    for buffer in (first, restored):
        buffer.add_batch(features[4:], labels[4:], logits[4:], ["id-4", "id-5"])
    assert first.seen == restored.seen == 6
    assert first.size == restored.size == 3
    np.testing.assert_array_equal(first.sample_ids[:3], restored.sample_ids[:3])
    for left, right in zip(first.sample(2), restored.sample(2), strict=True):
        np.testing.assert_array_equal(left, right)


def test_observe_stores_pre_update_logits_and_two_replay_terms() -> None:
    model = torch.nn.Linear(2, 3, bias=False)
    torch.nn.init.constant_(model.weight, 0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    buffer = ReservoirLogitBuffer(8, 2, 3, seed=5)
    x0 = torch.tensor([[1.0, 2.0]])
    before = model(x0).detach().numpy().copy()
    first = observe(model, optimizer, x0, torch.tensor([1]), ["first"], buffer,
                    alpha=0.3, beta=0.5, replay_batch_size=1)
    assert first.replay_logit_rows == first.replay_label_rows == 0
    np.testing.assert_allclose(buffer.logits[0], before[0])
    assert not np.allclose(model(x0).detach().numpy(), before)
    exposure: dict[int, int] = {}
    second = observe(model, optimizer, torch.tensor([[2.0, 0.0]]),
                     torch.tensor([2]), ["second"], buffer,
                     alpha=0.3, beta=0.5, replay_batch_size=1, exposure=exposure)
    assert second.replay_logit_rows == second.replay_label_rows == 1
    assert second.replay_label_ce > 0
    assert second.replay_logit_mse > 0
    assert second.total == pytest.approx(
        second.current_ce + 0.3 * second.replay_logit_mse
        + 0.5 * second.replay_label_ce, rel=1e-5)
    assert exposure == {1: 2}


def test_preserve_partial_task_before_resume(tmp_path) -> None:
    root = tmp_path / "member_0"
    task = root / "task_2"
    predictions = root / "member_predictions" / "task_2"
    task.mkdir(parents=True)
    predictions.mkdir(parents=True)
    (task / "partial.json").write_text("partial", encoding="utf-8")
    (predictions / "validation.npz").write_bytes(b"partial")
    _preserve_incomplete_task(root, 2)
    assert not task.exists() and not predictions.exists()
    attempts = list((root / "interrupted_attempts").glob("task_2_*"))
    assert len(attempts) == 1
    assert (attempts[0] / "task_2" / "partial.json").is_file()
    assert (attempts[0] / "member_predictions" / "validation.npz").is_file()


def test_visualizer_reads_only_the_requested_task_exposure(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from scripts.visualize_cil_run import _replay_exposure

    for task_id in (6, 7):
        folder = tmp_path / "member_0" / f"task_{task_id}"
        folder.mkdir(parents=True)
        pd.DataFrame([{"raw_class_id": 42, "replay_draws": task_id}]).to_csv(
            folder / "replay_exposure.csv", index=False)
    frame, mode = _replay_exposure(tmp_path, 0, 7, None, [])
    assert mode == "actual_replay_audit"
    assert frame is not None
    assert frame.loc[0, "replay_exposure"] == 7
