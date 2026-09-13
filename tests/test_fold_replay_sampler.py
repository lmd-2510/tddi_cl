"""Synthetic ID-only sampler tests; no dataset training or GPU work."""

from collections import Counter
from copy import deepcopy
import json
import random

import numpy as np
import pytest

from src.data import fold_replay_sampler as replay


def config(n=70, sizes=(4, 4, 4), **kwargs):
    labels = [10, 57, 901, -3][:len(sizes)]
    ids, raw_labels = [], []
    for c, size in zip(labels, sizes, strict=True):
        ids.extend(f"old_{c}_{i}" for i in range(size))
        raw_labels.extend([c] * size)
    result = dict(current_sample_ids=[f"current_{i}" for i in range(n)],
                  replay_sample_ids=ids, replay_raw_labels=raw_labels,
                  experiment_seed=0, member_id=0, task_id=1)
    result.update(kwargs)
    return result


def selected_ids(cfg, plan):
    ids = cfg["current_sample_ids"] + cfg["replay_sample_ids"]
    return [ids[i] for i in plan.indices]


@pytest.mark.parametrize("n,target", [(0, 0), (1, 0), (6, 0), (7, 1), (8, 1), (13, 1), (14, 2), (70, 10), (149, 21)])
def test_rounding_current_once_and_audit(n, target):
    cfg = config(n=n, sizes=(10, 10, 10))
    sampler = replay.FoldReplayFractionSampler(**cfg)
    state = sampler.state_dict()
    plan = sampler.plan_epoch(0)
    assert sampler.state_dict() == state  # preview must be pure
    assert Counter(i for i in plan.indices if i < n) == Counter(range(n))
    assert len(sampler) == len(plan.indices) == n + target
    assert plan.audit["target_replay_draws"] == target
    assert plan.audit["actual_replay_draws"] == target
    assert plan.audit["actual_fraction"] <= 0.125
    counts = Counter(i - n for i in plan.indices if i >= n)
    assert max(counts.values(), default=0) <= 3
    assert plan.audit["unique_replay_exemplars"] == len(counts)
    assert sum(plan.audit["repeat_histogram"].values()) == 30
    assert sum(int(k) * v for k, v in plan.audit["repeat_histogram"].items()) == target
    assert sum(v["draws"] for v in plan.audit["per_class"].values()) == target
    assert plan.audit["exemplar_coverage"] == len(counts) / 30
    assert not plan.audit["capacity_limited"]


def test_cap_saturation_redistribution_and_all_current_kept():
    cfg = config(n=140, sizes=(1, 10))  # target20: rare class max3, other gets17
    sampler = replay.FoldReplayFractionSampler(**cfg)
    plan = sampler.plan_epoch(0)
    assert plan.audit["per_class"]["10"]["draws"] == 3
    assert plan.audit["per_class"]["57"]["draws"] == 17
    assert plan.audit["max_repeat"] == 3
    assert plan.audit["per_class"]["57"]["max_repeat"] == 2
    cfg = config(n=140, sizes=(1, 2))  # total capacity9 < target20
    sampler = replay.FoldReplayFractionSampler(**cfg)
    plan = sampler.plan_epoch(0)
    assert plan.audit["capacity_limited"]
    assert plan.audit["actual_replay_draws"] == 9
    assert Counter(i for i in plan.indices if i < 140) == Counter(range(140))
    assert plan.audit["repeat_histogram"] == {"0": 0, "1": 0, "2": 0, "3": 3}
    assert plan.audit["actual_fraction"] == 9 / 149


@pytest.mark.parametrize("kwargs", [{"task_id": 0}, {"task_id": 1}, {"task_id": 3}])
def test_no_memory_and_task0(kwargs):
    sampler = replay.FoldReplayFractionSampler(**config(n=20, sizes=(), **kwargs))
    plan = sampler.plan_epoch(0)
    assert sorted(plan.indices) == list(range(20))
    assert plan.audit["actual_replay_draws"] == 0
    assert plan.audit["target_replay_draws"] == (0 if kwargs["task_id"] == 0 else 2)
    assert plan.audit["class_coverage"] == plan.audit["exemplar_coverage"] == 0


def test_small_replay_rotates_classes_then_exemplars():
    cfg = config(n=7, sizes=(4, 4, 4))  # one replay / epoch
    sampler = replay.FoldReplayFractionSampler(**cfg)
    state = sampler.state_dict()
    seen_ids = []
    class_draws = []
    for epoch in range(24):
        plan = sampler.plan_epoch(epoch)
        index = next(i - 7 for i in plan.indices if i >= 7)
        seen_ids.append(cfg["replay_sample_ids"][index])
        class_draws.append(cfg["replay_raw_labels"][index])
    for start in range(0, 24, 3):
        assert set(class_draws[start:start + 3]) == {10, 57, 901}
    assert len(set(seen_ids[:12])) == 12
    assert seen_ids[:12] == seen_ids[12:]
    assert sampler.state_dict() == state


def test_no_exemplar_repeat_before_class_cycle_even_after_mix():
    cfg = config(n=140, sizes=(3, 8))
    sampler = replay.FoldReplayFractionSampler(**cfg)
    for epoch in range(4):
        plan = sampler.plan_epoch(epoch)
        for c in (10, 57):
            queue = [cfg["replay_sample_ids"][i - 140] for i in plan.indices
                     if i >= 140 and cfg["replay_raw_labels"][i - 140] == c]
            width = sum(label == c for label in cfg["replay_raw_labels"])
            assert len(set(queue[:min(len(queue), width)])) == min(len(queue), width)
            assert max(Counter(queue).values(), default=0) <= 3
            for i in range(width, len(queue)):
                assert queue[i] == queue[i - width]


def test_global_rng_model_rng_and_other_member_independence():
    cfg = config()
    sampler = replay.FoldReplayFractionSampler(**cfg)
    expected = sampler.plan_epoch(3).indices
    np.random.seed(88)
    random.seed(77)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    assert sampler.plan_epoch(3).indices == expected
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and after[2:] == numpy_state[2:]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert random.getstate() == python_state
    np.random.random(100)
    random.random()
    torch = pytest.importorskip("torch")
    torch.rand(20)  # simulate unrelated model/dropout RNG consumption, CPU only
    before = torch.random.get_rng_state().clone()
    assert sampler.plan_epoch(3).indices == expected
    assert torch.equal(before, torch.random.get_rng_state())
    different = replay.FoldReplayFractionSampler(**{**cfg, "member_id": 1})
    assert different.plan_epoch(3).indices != expected


def test_early_stopping_previous_task_cannot_shift_ab_next_task():
    a0 = replay.FoldReplayFractionSampler(**config(n=20, sizes=(), task_id=0))
    b0 = replay.FoldReplayFractionSampler(**config(n=20, sizes=(), task_id=0))
    for _ in range(3):
        list(a0)
    for _ in range(13):
        list(b0)
    # Fresh task instance resets all class/sample cursors independent of 3 vs13.
    a, b = replay.FoldReplayFractionSampler(**config()), replay.FoldReplayFractionSampler(**config())
    for epoch in range(5):
        assert list(a) == list(b)
        assert a.last_audit == b.last_audit
        assert a.next_epoch == b.next_epoch == epoch + 1


def test_same_id_order_independent_of_input_row_order_but_resume_requires_alignment():
    cfg = config(n=70, sizes=(3, 7))
    reordered = {**cfg, "current_sample_ids": cfg["current_sample_ids"][::-1],
                 "replay_sample_ids": cfg["replay_sample_ids"][::-1], "replay_raw_labels": cfg["replay_raw_labels"][::-1]}
    first, second = replay.FoldReplayFractionSampler(**cfg), replay.FoldReplayFractionSampler(**reordered)
    for epoch in range(3):
        assert selected_ids(cfg, first.plan_epoch(epoch)) == selected_ids(reordered, second.plan_epoch(epoch))
    with pytest.raises(ValueError, match="metadata"):
        replay.FoldReplayFractionSampler.from_state_dict(first.state_dict(), **reordered)


@pytest.mark.parametrize("checkpoint_epoch", [0, 1, 5])
def test_json_state_roundtrip_matches_continuous_and_seek(checkpoint_epoch):
    cfg = config(n=70, sizes=(1, 7, 2))
    continuous = replay.FoldReplayFractionSampler(**cfg)
    for _ in range(checkpoint_epoch):
        list(continuous)
    snapshot = json.loads(json.dumps(continuous.state_dict()))
    restored = replay.FoldReplayFractionSampler.from_state_dict(snapshot, **cfg)
    direct = replay.FoldReplayFractionSampler(**cfg)
    direct.set_epoch(checkpoint_epoch)
    assert restored.state_dict() == direct.state_dict() == continuous.state_dict()
    for _ in range(4):
        assert list(restored) == list(direct) == list(continuous)
        assert restored.last_audit == continuous.last_audit
        assert restored.state_dict() == continuous.state_dict()


def test_active_iterator_checkpoint_guard_and_abandoned_retry():
    cfg = config()
    sampler = replay.FoldReplayFractionSampler(**cfg)
    original = sampler.state_dict()
    iterator = iter(sampler)
    first = next(iterator)
    assert sampler.next_epoch == 0 and sampler.last_audit is None
    with pytest.raises(RuntimeError, match="completed epoch"):
        sampler.state_dict()
    with pytest.raises(RuntimeError, match="active"):
        sampler.set_epoch(2)
    with pytest.raises(RuntimeError, match="active"):
        iter(sampler)
    iterator.close()
    assert sampler.state_dict() == original
    retried = list(sampler)
    assert retried[0] == first and sampler.next_epoch == 1
    assert sampler.last_audit["epoch"] == 0


def test_state_config_and_tamper_guards():
    cfg = config()
    sampler = replay.FoldReplayFractionSampler(**cfg)
    list(sampler)
    state = sampler.state_dict()
    changed = deepcopy(state)
    changed["class_cursor"] += 1
    with pytest.raises(ValueError, match="SHA256"):
        replay.FoldReplayFractionSampler.from_state_dict(changed, **cfg)
    changed["state_sha256"] = replay._digest({k: v for k, v in changed.items() if k != "state_sha256"})
    with pytest.raises(ValueError, match="cursor/queue/RNG"):
        replay.FoldReplayFractionSampler.from_state_dict(changed, **cfg)
    for change in ({"member_id": 1}, {"experiment_seed": 1}, {"task_id": 2}, {"repeat_cap": 2}, {"replay_fraction": 0.2}):
        with pytest.raises(ValueError, match="metadata"):
            replay.FoldReplayFractionSampler.from_state_dict(state, **{**cfg, **change})


def test_dataloader_tail_not_dropped_and_no_fixed_microbatch_ratio():
    torch = pytest.importorskip("torch")
    cfg = config(n=70, sizes=(5,))
    sampler = replay.FoldReplayFractionSampler(**cfg)
    loader = torch.utils.data.DataLoader(list(range(75)), batch_size=64, sampler=sampler, drop_last=False)
    batches = list(loader)
    assert [len(b) for b in batches] == [64, 16]
    counts = Counter(int(i) for b in batches for i in b if i < 70)
    assert counts == Counter(range(70))
    assert sampler.last_audit["total_draws"] == 80


@pytest.mark.parametrize("change", [
    {"current_sample_ids": ["a", "a"]}, {"replay_sample_ids": ["a", "a"], "replay_raw_labels": [10, 10]},
    {"current_sample_ids": [""]}, {"current_sample_ids": [123]},
    {"current_sample_ids": ["old_10_0"]}, {"replay_raw_labels": [10]},
    {"replay_raw_labels": [1.5] * 12}, {"replay_raw_labels": [True] * 12},
    {"task_id": 0}, {"member_id": 3}, {"experiment_seed": -1}, {"task_id": -1},
    {"repeat_cap": 0}, {"repeat_cap": 2.5}, {"replay_fraction": -0.1},
    {"replay_fraction": 1}, {"replay_fraction": float("nan")},
])
def test_invalid_inputs(change):
    with pytest.raises(ValueError):
        replay.FoldReplayFractionSampler(**{**config(), **change})
