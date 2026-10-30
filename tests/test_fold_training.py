"""Frozen-fold training integration with tiny tensors and files."""
import json
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.data.ddi_dataset import prepare_development_fold_context
from src.data.fold_preprocessing import prepare_fold_preprocessing, save_fold_preprocessing
from src.data.stratified_folds import FOLD_ASSIGNMENT_SCHEMA, describe_fold_source, save_fold_artifact
from src.models.tddi_paper_member import TDDIPaperMember, TDDIPaperMemberConfig
from src.eval.predictions import load_member_prediction_artifact
from src.training import fold_training, train_cil as engine


@pytest.fixture
def data(tmp_path, request):
    paths, rows = {}, []
    # Noncontiguous IDs; task 1 inserts old rows at different dense indices.
    classes = list(reversed(range(1000, 1534, 3)))
    for split in ("train", "validation", "test"):
        paths[split] = tmp_path / f"{split}.parquet"
        n = 178 * 3 if split != "test" else 178 if getattr(request, "param", None) == "full_test" else 1
        values = {"drugid-drug_a": [f"{split}_{i}" for i in range(n)], "drugid-drug_b": ["B"] * n}
        if split != "test":
            values.update({"class": classes * 3,
                           "x": [float(i % 17 + (50 if split == "validation" else 0)) for i in range(n)],
                           "y": [float(i % 11 + 1) for i in range(n)]})
            rows.extend({"source_split": split, "source_row_index": i, "sample_id": f"{split}_{i}|B",
                         "raw_class_id": values["class"][i], "fold_id": i // 178} for i in range(n))
        if split == "test" and getattr(request, "param", False):
            values.update({"class": classes if n == 178 else [classes[0]], "x": [3.0] * n, "y": [4.0] * n})
        # Normally test has no descriptors/labels: validation-only must work.
        pq.write_table(pa.table(values), paths[split])
    folds = tmp_path / "folds"
    save_fold_artifact(folds, pa.Table.from_pylist(rows, schema=FOLD_ASSIGNMENT_SCHEMA),
                       sources={s: describe_fold_source(p) for s, p in paths.items()}, fold_seed=42)
    tasks, offset = [], 0
    for i, size in enumerate(fold_training.FULL_TASK_LAYOUT):
        tasks.append({"task_id": i, "classes": classes[offset:offset + size]})
        offset += size
    taskfile = tmp_path / "tasks.json"
    taskfile.write_text(json.dumps({"protocol": "head_to_tail", "tasks": tasks}))
    cols = tmp_path / "columns.json"
    cols.write_text('["x", "y"]')
    ctx = prepare_development_fold_context(folds / "fold_assignments.parquet", folds / "fold_manifest.json", source_paths=paths)
    preps = {}
    for policy in ("raw_identity", "task0_standard_frozen"):
        artifact = prepare_fold_preprocessing(context=ctx, task_file=taskfile, feature_columns=["x", "y"],
            member_id=0, validation_fold=0, policy=policy)
        preps[policy] = save_fold_preprocessing(artifact, tmp_path / policy)
    return dict(paths=paths, folds=folds, tasks=taskfile, cols=cols, preps=preps, context=ctx, root=tmp_path)


def cli(data, policy="raw_identity", suffix="", extra=()):
    return ["train_cil.py", "--train", str(data["paths"]["train"]), "--validation", str(data["paths"]["validation"]),
            "--test", str(data["paths"]["test"]), "--feature-cols", str(data["cols"]), "--task-file", str(data["tasks"]),
            "--outdir", str(data["root"] / f"run_{policy}{suffix}"), "--member-id", "0", "--device", "cpu",
            "--fold-replay-policy", "stratified_fraction_v1", "--fold-assignments", str(data["folds"] / "fold_assignments.parquet"),
            "--fold-manifest", str(data["folds"] / "fold_manifest.json"), "--fold-preprocessing", str(data["preps"][policy]),
            "--preprocessing-policy", policy, "--validation-only", "--stop-after-task", "1", "--epochs", "2", *extra]


@pytest.fixture
def tiny(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(fold_training, "TDDI_PAPER_INPUT_DIM", 2)
    calls = []
    def expand(previous, previous_map, seen_map, **kwargs):
        model = TDDIPaperMember(TDDIPaperMemberConfig(input_dim=2, hidden_dims=(8, 4), num_classes=len(seen_map),
                                                   dropout=kwargs["dropout"], activation=kwargs["activation"]))
        if previous is not None:
            engine.copy_previous_state_to_expanded_model(previous, model, previous_map, seen_map,
                                                         variant="tddi_paper_member")
            for raw, old_row in previous_map.items():
                torch.testing.assert_close(model.head.weight[seen_map[raw]], previous.head.weight[old_row])
                torch.testing.assert_close(model.head.bias[seen_map[raw]], previous.head.bias[old_row])
        calls.append(len(seen_map))
        return model
    monkeypatch.setattr(engine, "expand_model_for_seen_classes", expand)
    yield calls
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize("method", [engine.FIXED_BUDGET_METHOD, "ewc", "agem"])
def test_three_methods_train_on_frozen_folds(data, monkeypatch, tiny, method):
    monkeypatch.setattr(sys, "argv", cli(data, extra=["--method", method, "--export-member-predictions",
                                                   "--member-prediction-splits", "validation"]))
    engine.main()
    root = data["root"] / "run_raw_identity"
    state = torch.load(root / "checkpoints/task_1.pt", weights_only=False)
    assert len(state["seen_class_map"]) == 58
    assert state["contract"]["method"] == method
    artifact = load_member_prediction_artifact(root / "member_predictions/task_1/validation.npz")
    assert artifact.context.method == method
    assert artifact.context.method_protocol == engine.METHOD_POLICIES[method]
    losses = pd.read_csv(root / "training_audit.csv")
    assert np.isfinite(losses.select_dtypes(include="number")).all().all()
    assert not any("distill" in name for name in losses.columns)
    if method == "ewc":
        assert state["buffer_state"]["metadata"]["total_memory_budget"] == 0
        assert state["method_state"]["fisher"]
    else:
        assert state["method_state"] == {}
    for t in (0, 1):
        audit = json.loads((root / f"task_{t}/input_audit.json").read_text())
        assert not set(audit["current_ids"] + audit["replay_ids_before"]) & set(audit["validation_ids"])
    if method == engine.FIXED_BUDGET_METHOD:
        audit = json.loads((root / "task_1/epoch_1_audit.json").read_text())["sampling"]
        assert audit["current_draws"] == 80
        assert audit["actual_replay_draws"] == 11
        assert audit["max_repeat"] <= 3


def test_cb_loss_matches_focal_current_and_ce_replay():
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 3)
    x = torch.randn(4, 2)
    y = torch.tensor([0, 1, 2, 0])
    logits = model(x)
    ce = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    focal = (1 - torch.exp(-ce)) * ce
    expected = torch.where(y == 0, ce, focal).mean().item()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=4)
    actual = engine.train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0),
        engine.FocalLoss(), "cpu", old_class_indices=[0], loss_variant="focal_current_ce_replay")
    assert actual == pytest.approx(expected)


def test_task_cache_hit_matches_uncached_training(data, tiny, monkeypatch):
    cache = data["root"] / "common_cache"
    for suffix, extra in (("_plain", []), ("_cache_build", ["--task-cache-root", str(cache)]),
                          ("_cache_hit", ["--task-cache-root", str(cache)])):
        monkeypatch.setattr(sys, "argv", cli(data, suffix=suffix, extra=extra))
        engine.main()
    states = [torch.load(data["root"] / f"run_raw_identity{s}/checkpoints/task_1.pt", weights_only=False)
              for s in ("_plain", "_cache_build", "_cache_hit")]
    for key in states[0]["model_state"]:
        for state in states[1:]:
            torch.testing.assert_close(state["model_state"][key], states[0]["model_state"][key], atol=0, rtol=0)


def test_actual_accumulation_steps_and_tail_matches_large_batch(tiny):
    torch.manual_seed(1)
    x, y = torch.randn(2051, 2), torch.zeros(2051, dtype=torch.long)
    first = torch.nn.Linear(2, 2)
    second = deepcopy(first)
    dataset = torch.utils.data.TensorDataset(x, y)
    for model, micro, accumulation in ((first, 64, 16), (second, 1024, 1)):
        losses = engine.train_one_epoch(model, torch.utils.data.DataLoader(dataset, batch_size=micro),
            torch.optim.AdamW(model.parameters(), lr=.001), engine.FocalLoss(), "cpu",
            gradient_accumulation_steps=accumulation, return_loss_components=True)
        assert losses.optimizer_steps == 3 and losses.tail_effective_batch == 3 and losses.examples_seen == 2051
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

def test_single_reporting_pass_matches_separate_group_metrics():
    torch.manual_seed(7)
    values = np.random.default_rng(7).normal(size=(12, 2)).astype(np.float32)
    labels = np.asarray([10, 20, 30, 40] * 3, dtype=np.int64)
    seen_map = {10: 0, 20: 1, 30: 2, 40: 3}
    new_classes = [30, 40]
    model = torch.nn.Linear(2, 4).eval()
    criterion = engine.FocalLoss(1.0)

    optimized, outputs = fold_training._evaluate_fold_split(
        engine=engine,
        model=model,
        values=values,
        labels=labels,
        seen_map=seen_map,
        new_classes=new_classes,
        criterion=criterion,
        device="cpu",
        batch_size=5,
    )
    assert outputs.latent_features is None

    expected = {}
    groups = {
        "seen_all": sorted(seen_map),
        "old": [10, 20],
        "current": new_classes,
    }
    inverse = {index: raw for raw, index in seen_map.items()}
    for group, classes in groups.items():
        keep = np.isin(labels, classes)
        loader = torch.utils.data.DataLoader(
            engine.build_tensor_dataset(
                values[keep],
                np.asarray([seen_map[int(raw)] for raw in labels[keep]], dtype=np.int64),
            ),
            batch_size=5,
            shuffle=False,
        )
        result = engine.evaluate_model(
            model,
            loader,
            criterion,
            "cpu",
            inverse,
            evaluation_class_indices=[seen_map[raw] for raw in classes],
        )
        expected[group] = result.metrics

    for row in optimized:
        reference = expected[row["group"]]
        for metric in (
            "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "weighted_f1", "weighted_precision", "balanced_accuracy", "loss",
        ):
            assert row[metric] == pytest.approx(reference[metric], abs=1e-7)

def test_source_tamper_rejected_before_output(data, monkeypatch, tiny):
    path = data["paths"]["train"]
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(sys, "argv", cli(data))
    with pytest.raises(ValueError, match="hash|SHA256|sha256"):
        engine.main()
    assert not (data["root"] / "run_raw_identity").exists()


@pytest.mark.parametrize("data", [True], indirect=True)
def test_optional_test_report_only_after_best_selection(data, monkeypatch, tiny):
    command = cli(data, extra=["--stop-after-task", "0", "--epochs", "1"])
    command.remove("--validation-only")
    monkeypatch.setattr(sys, "argv", command)
    root = data["root"] / "run_raw_identity"
    original, calls = fold_training.load_split_frame, []
    def read(*a, **kw):
        assert (root / "task_0/best_model.pt").is_file()
        calls.append(str(a[0]))
        return original(*a, **kw)
    monkeypatch.setattr(fold_training, "load_split_frame", read)
    engine.main()
    assert calls == [str(data["paths"]["test"])]
    metrics = pd.read_csv(root / "metrics.csv")
    assert set(metrics.split) == {"validation", "test"}
    assert metrics[(metrics.split == "test") & (metrics.group == "seen_all")].sample_count.tolist() == [1]
