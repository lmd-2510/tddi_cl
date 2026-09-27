"""Small deterministic checks for online DER++ and its resumable artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.methods.derpp import (
    ClassBalancedReservoirLogitBuffer, ReservoirLogitBuffer, equal_class_quotas, observe,
)
from src.training.derpp_fold import _preserve_incomplete_task
from scripts.run_p3_derpp_equal_buffer_pilot import _summarize


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


def test_equal_class_quotas_redistribute_unfilled_tail_capacity() -> None:
    assert equal_class_quotas({0: 3, 1: 100, 2: 100}, 23) == {0: 3, 1: 10, 2: 10}
    assert equal_class_quotas({0: 2, 1: 4}, 10) == {0: 2, 1: 4}


def test_balanced_reservoir_retains_old_classes_and_restores_state() -> None:
    buffer = ClassBalancedReservoirLogitBuffer(6, 2, 4, seed=11)
    assert buffer.begin_task({0: 12, 1: 2}) == {0: 4, 1: 2}
    features = np.arange(28, dtype=np.float32).reshape(14, 2)
    labels = np.asarray([0] * 12 + [1] * 2)
    logits = np.arange(56, dtype=np.float32).reshape(14, 4)
    buffer.add_batch(features, labels, logits, [f"old-{i}" for i in range(14)])
    buffer.finish_task()
    assert sorted(np.bincount(buffer.labels[:buffer.size], minlength=4)) == [0, 0, 2, 4]
    assert buffer.begin_task({2: 12}) == {0: 2, 1: 2, 2: 2}
    assert buffer.size == 4
    new_features = np.arange(24, dtype=np.float32).reshape(12, 2) + 100
    new_logits = np.arange(48, dtype=np.float32).reshape(12, 4) + 100
    buffer.add_batch(new_features, np.full(12, 2), new_logits,
                     [f"new-{i}" for i in range(12)])
    buffer.finish_task()
    np.testing.assert_array_equal(np.bincount(buffer.labels[:buffer.size], minlength=4),
                                  [2, 2, 2, 0])
    assert set(buffer.sample_ids[:buffer.size]) & {"old-12", "old-13"} == {"old-12", "old-13"}
    restored = ClassBalancedReservoirLogitBuffer(6, 2, 4, seed=77)
    restored.load_state_dict(buffer.state_dict())
    for left, right in zip(buffer.sample(3), restored.sample(3), strict=True):
        np.testing.assert_array_equal(left, right)
    assert buffer.begin_task({3: 3}) == restored.begin_task({3: 3})
    for target in (buffer, restored):
        target.add_batch(np.ones((3, 2), dtype=np.float32), np.full(3, 3),
                         np.ones((3, 4), dtype=np.float32), ["last-0", "last-1", "last-2"])
        target.finish_task()
    np.testing.assert_array_equal(buffer.sample_ids[:buffer.size], restored.sample_ids[:restored.size])


def test_balanced_reservoir_slot_bookkeeping_across_many_boundaries() -> None:
    for seed in range(8):
        buffer = ClassBalancedReservoirLogitBuffer(11, 2, 8, seed=seed)
        serial = 0
        for task_classes in ((0, 1, 2), (3, 4), (5, 6, 7)):
            counts = {label: 7 + ((label + seed) % 5) for label in task_classes}
            buffer.begin_task(counts)
            labels = np.concatenate([np.full(count, label) for label, count in counts.items()])
            labels = np.random.default_rng(seed + serial).permutation(labels)
            features = np.arange(serial, serial + len(labels) * 2, dtype=np.float32).reshape(-1, 2)
            logits = np.zeros((len(labels), 8), dtype=np.float32)
            buffer.add_batch(features, labels, logits,
                             [f"sample-{serial + i}" for i in range(len(labels))])
            serial += len(labels)
            buffer.finish_task()
            assert buffer.size == 11
            assert len(set(buffer.sample_ids[:buffer.size])) == buffer.size
            for label, quota in buffer.quotas.items():
                assert int(np.count_nonzero(buffer.labels[:buffer.size] == label)) == quota


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


def test_equal_buffer_pilot_summary_marks_member_only(tmp_path) -> None:
    import json

    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({"tasks": [{"task_id": i, "classes": [i]}
                                               for i in range(8)]}), encoding="utf-8")
    member = tmp_path / "member_0"
    member.mkdir()
    rows = [{"task_id": i, "train_rows": 2, "memory_after": i + 1,
             "validation_macro_f1": 0.8, "test_macro_f1": 0.7 - i * 0.01}
            for i in range(8)]
    (member / "run_summary.json").write_text(
        json.dumps({"completed_task_id": 7, "tasks": rows}), encoding="utf-8")
    pd.DataFrame(rows).to_csv(member / "metrics.csv", index=False)
    for task_id in range(8):
        folder = member / f"task_{task_id}"
        folder.mkdir()
        (folder / "buffer_audit.json").write_text(
            json.dumps({"allocation": {str(i): 1 for i in range(task_id + 1)}}),
            encoding="utf-8")
        pd.DataFrame([{"raw_class_id": i, "replay_draws": 2}
                      for i in range(task_id + 1)]).to_csv(
                          folder / "replay_exposure.csv", index=False)
    pd.DataFrame([{"task": 7, "split": "test", "group": "old", "macro_f1": 0.2}]).to_csv(
        member / "task_7" / "metrics.csv", index=False)
    _summarize(tmp_path, task_file, {"protocol": {"sha256": "dummy"}})
    manifest = json.loads((tmp_path / "pilot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["member_ids"] == [0] and manifest["offline_ensemble"] is False
    assert (tmp_path / "pilot_results" / "PILOT_RESULTS.md").is_file()
    groups = pd.read_csv(tmp_path / "pilot_results" / "buffer_by_source_task.csv")
    assert len(groups[groups.boundary_task == 7]) == 8
