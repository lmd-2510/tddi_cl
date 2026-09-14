"""Task-boundary continuation on synthetic full-layout P3, tiny CPU models only."""
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pytest
import torch

from tests.test_fold_pilot_training import data, tiny, cli
from src.training import fold_pilot_training as pilot, train_cil as engine
from src.training import replay_checkpoint as checkpoint


def run(data, monkeypatch, *, policy="raw_identity", suffix="", stop=2, resume=None):
    command = cli(data, policy, suffix, extra=["--stop-after-task", str(stop)])
    if resume:
        command += ["--resume-fold-checkpoint", str(resume)]
    monkeypatch.setattr(sys, "argv", command)
    engine.main()
    return data["root"] / f"run_{policy}{suffix}"


def load(root, task):
    return torch.load(root / "checkpoints" / f"task_{task}.pt", map_location="cpu", weights_only=False)


def fingerprint(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("policy", ["raw_identity", "task0_standard_frozen"])
@pytest.mark.parametrize("boundary", [0, 1])
def test_continuous_vs_resume_three_tasks(data, monkeypatch, tiny, policy, boundary):
    continuous = run(data, monkeypatch, policy=policy, suffix="_continuous")
    split = run(data, monkeypatch, policy=policy, suffix="_split", stop=boundary)
    saved = fingerprint(split / f"task_{boundary}")
    saved_summary = (split / "run_summary.json").read_bytes()
    from src.data import fold_preprocessing, stratified_folds
    def forbidden(*a, **kw):
        pytest.fail("Resume must not fit preprocessing or rebuild folds")
    monkeypatch.setattr(fold_preprocessing, "prepare_fold_preprocessing", forbidden)
    monkeypatch.setattr(stratified_folds, "build_stratified_fold_assignments", forbidden)
    # Unrelated global RNG draws before resume must not change the trajectory.
    random.random()
    np.random.rand(11)
    torch.rand(9)
    teacher_hashes = []
    original = engine.train_one_epoch
    def train(model, *a, **kw):
        teacher = kw["teacher_model"]
        assert teacher is not None and not any(p.requires_grad for p in teacher.parameters())
        teacher_hashes.append(checkpoint.fold_state_digest(teacher.state_dict()))
        return original(model, *a, **kw)
    monkeypatch.setattr(engine, "train_one_epoch", train)
    run(data, monkeypatch, policy=policy, suffix="_split", resume=split / f"checkpoints/task_{boundary}.pt")
    assert saved == fingerprint(split / f"task_{boundary}")
    assert saved_summary == (split / "run_summary.json").read_bytes()  # immutable old pilot report
    first_teacher = checkpoint.fold_state_digest(load(continuous, boundary)["model_state"])
    assert teacher_hashes[0] == first_teacher
    left, right = load(continuous, 2), load(split, 2)
    assert left["next_task_id"] == right["next_task_id"] == 3
    assert not right["full_trajectory_complete"]
    assert left["seen_class_map"] == right["seen_class_map"] and len(right["seen_class_map"]) == 78
    for key in ("model_state", "buffer_state", "sampler_state", "rng_state", "next_task_scheduling"):
        assert checkpoint.fold_state_digest(left[key]) == checkpoint.fold_state_digest(right[key]), key
    assert left["progress"]["metric_rows"] == right["progress"]["metric_rows"]
    for a, b in zip(left["progress"]["epoch_rows"], right["progress"]["epoch_rows"], strict=True):
        assert {k: v for k, v in a.items() if k != "seconds"} == {k: v for k, v in b.items() if k != "seconds"}
    for task in range(3):
        for epoch in (1, 2):
            a, b = [json.loads((r / f"task_{task}/epoch_{epoch}_audit.json").read_text()) for r in (continuous, split)]
            assert a["sampling"] == b["sampling"]
    assert "teacher" not in right and "optimizer" not in right
    assert list(split.glob("reports/through_task_2_*/run_summary.json"))


def test_partial_attempt_preserved_and_completed_scope_noop(data, monkeypatch, tiny):
    root = run(data, monkeypatch, stop=0)
    partial = root / "task_1"
    partial.mkdir()
    (partial / "stdout.log").write_text("incomplete training, keep me")
    before = fingerprint(partial)
    run(data, monkeypatch, stop=1, resume=root / "checkpoints/task_0.pt")
    assert fingerprint(partial) == before
    state = load(root, 1)
    assert state["progress"]["task_summaries"][-1]["artifact_directory"].startswith("attempts/task_1_")
    entire_run = fingerprint(root)
    def forbidden(*a, **kw):
        pytest.fail("Verified complete scope must not create/train/export a model")
    monkeypatch.setattr(engine, "expand_model_for_seen_classes", forbidden)
    run(data, monkeypatch, stop=1, resume=root / "checkpoints/task_1.pt")
    assert entire_run == fingerprint(root)
    with pytest.raises(ValueError, match="newer"):
        run(data, monkeypatch, stop=2, resume=root / "checkpoints/task_0.pt")


@pytest.mark.parametrize("mutation,match", [
    ("schema", "schema"), ("kind", "Not a frozen"), ("model_digest", "SHA256"),
    ("class_map", "class map"), ("budget", "contract mismatch"), ("ranking", "contract mismatch"),
    ("sampler", "contract mismatch"), ("member", "contract mismatch"), ("fold", "contract mismatch"),
    ("scaler", "contract mismatch"), ("source", "contract mismatch"), ("task_hash", "contract mismatch"),
    ("next_task", "completed/next"), ("full_complete", "full-trajectory"),
    ("model_not_best", "best model"), ("buffer", "state SHA256"),
    ("metrics", "metric progress"), ("schedule", "scheduling"), ("rng", "RNG state"),
])
def test_checkpoint_metadata_guards(data, monkeypatch, tiny, mutation, match):
    root = run(data, monkeypatch, stop=0)
    state = load(root, 0)
    if mutation == "schema": state["schema_version"] = 99
    elif mutation == "kind": state["kind"] = "legacy"
    elif mutation in ("model_digest", "model_not_best"): state["model_state"]["head.bias"][0] += 1
    elif mutation == "class_map": state["raw_class_order"].reverse()
    elif mutation == "budget": state["contract"]["global_slot_budget"] += 1
    elif mutation == "ranking": state["contract"]["buffer"]["ranking"]["epsilon"] = 1
    elif mutation == "sampler": state["contract"]["sampler"]["repeat_cap"] = 4
    elif mutation == "member": state["contract"]["seeds"]["member_id"] = 1
    elif mutation == "fold": state["contract"]["validation_fold"] = 1
    elif mutation == "scaler": state["contract"]["preprocessing_sha256"] = "bad"
    elif mutation == "source": state["contract"]["sources"]["train"]["sha256"] = "bad"
    elif mutation == "task_hash": state["contract"]["task_file_sha256"] = "bad"
    elif mutation == "next_task": state["next_task_id"] = 2
    elif mutation == "full_complete": state["full_trajectory_complete"] = True
    elif mutation == "buffer": next(iter(state["buffer_state"]["entries"].values()))["features"][0, 0] += 1
    elif mutation == "metrics": state["progress"]["metric_rows"][0]["accuracy"] += .1
    elif mutation == "schedule": state["next_task_scheduling"]["epoch"] = 1
    elif mutation == "rng": del state["rng_state"]["torch_cpu"]
    if mutation != "model_digest":
        state["state_sha256"] = checkpoint.fold_state_digest({k: v for k, v in state.items() if k != "state_sha256"})
    path = root / "checkpoints/task_0.pt"
    torch.save(state, path)  # intentionally corrupt this test-owned checkpoint
    before = fingerprint(root)
    with pytest.raises((ValueError, TypeError), match=match):
        run(data, monkeypatch, stop=1, resume=path)
    assert fingerprint(root) == before


@pytest.mark.parametrize("change", ["preprocessing", "hyper", "artifact", "config", "source"])
def test_external_mismatch_no_writes(data, monkeypatch, tiny, change):
    root = run(data, monkeypatch, stop=0)
    extra = []
    if change == "preprocessing":
        extra = ["--preprocessing-policy", "task0_standard_frozen", "--fold-preprocessing", str(data["preps"]["task0_standard_frozen"])]
    elif change == "hyper": extra = ["--lr", "0.02"]
    elif change == "artifact": (root / "task_0/metrics.csv").write_text("tampered")
    elif change == "config": (root / "run_config.json").write_text("{}")
    elif change == "source":
        path = data["paths"]["train"]
        path.write_bytes(path.read_bytes() + b"tampered")
    command = cli(data, extra=["--resume-fold-checkpoint", str(root / "checkpoints/task_0.pt"), *extra])
    monkeypatch.setattr(sys, "argv", command)
    before = fingerprint(root)
    with pytest.raises(ValueError): engine.main()
    assert fingerprint(root) == before


def test_atomic_publication_never_replaces_and_cleans_only_own_temp(tmp_path):
    path = tmp_path / "state.pt"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        checkpoint.atomic_publish_fold_file(path, lambda handle: handle.write(b"new"))
    assert path.read_bytes() == b"existing"
    def broken(handle):
        handle.write(b"partial")
        raise RuntimeError("disk write failed")
    with pytest.raises(RuntimeError): checkpoint.atomic_publish_fold_file(tmp_path / "other.pt", broken)
    assert list(tmp_path.iterdir()) == [path]


def test_portable_sources_and_run_directory(data, monkeypatch, tiny):
    root = run(data, monkeypatch, stop=0)
    relocated = data["root"] / "relocated"
    relocated.mkdir()
    moved_data = {**data, "paths": {}}
    for split, path in data["paths"].items():
        target = relocated / path.name
        shutil.copy2(path, target)
        moved_data["paths"][split] = target
    moved_root = data["root"] / "run_raw_identity_moved"
    shutil.copytree(root, moved_root)
    run(moved_data, monkeypatch, suffix="_moved", stop=1, resume=moved_root / "checkpoints/task_0.pt")
    assert load(moved_root, 1)["next_task_id"] == 2


@pytest.mark.parametrize("policy", ["raw_identity", "task0_standard_frozen"])
def test_interrupted_epoch_restarts_task_from_previous_boundary(data, monkeypatch, tiny, policy):
    continuous = run(data, monkeypatch, policy=policy, suffix="_control", stop=1)
    original = pilot._publish_json
    def interrupted(path, value):
        original(path, value)
        if Path(path).name == "epoch_1_audit.json" and Path(path).parent.name == "task_1":
            raise RuntimeError("synthetic interruption after epoch 1")
    monkeypatch.setattr(pilot, "_publish_json", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run(data, monkeypatch, policy=policy, suffix="_interrupted", stop=1)
    root = data["root"] / f"run_{policy}_interrupted"
    old_attempt = fingerprint(root / "task_1")
    assert (root / "checkpoints/task_0.pt").is_file()
    assert not (root / "checkpoints/task_1.pt").exists()
    monkeypatch.setattr(pilot, "_publish_json", original)
    run(data, monkeypatch, policy=policy, suffix="_interrupted", stop=1, resume=root / "checkpoints/task_0.pt")
    assert old_attempt == fingerprint(root / "task_1")
    for key in ("model_state", "buffer_state", "sampler_state", "rng_state"):
        assert checkpoint.fold_state_digest(load(root, 1)[key]) == checkpoint.fold_state_digest(load(continuous, 1)[key])


def test_missing_completion_evidence_and_model_only_not_resumable(data, monkeypatch, tiny):
    root = run(data, monkeypatch, stop=0)
    with pytest.raises(ValueError, match="Not a frozen"):
        run(data, monkeypatch, stop=1, resume=root / "task_0/best_model.pt")
    state = load(root, 0)
    del state["artifact_hashes"]["task_0/metrics.json"]
    state["state_sha256"] = checkpoint.fold_state_digest({k: v for k, v in state.items() if k != "state_sha256"})
    torch.save(state, root / "checkpoints/task_0.pt")
    with pytest.raises(ValueError, match="completion evidence"):
        run(data, monkeypatch, stop=1, resume=root / "checkpoints/task_0.pt")
