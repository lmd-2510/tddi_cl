"""Prompt 9 integration: tiny tensors/files only, no production model allocation."""
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
from src.training import fold_pilot_training as pilot, train_cil as engine


@pytest.fixture
def data(tmp_path, request):
    paths, rows = {}, []
    # Noncontiguous IDs; task 1 inserts old rows at different dense indices.
    classes = list(reversed(range(1000, 1534, 3)))
    for split in ("train", "validation", "test"):
        paths[split] = tmp_path / f"{split}.parquet"
        n = 178 * 3 if split != "test" else 1
        values = {"drugid-drug_a": [f"{split}_{i}" for i in range(n)], "drugid-drug_b": ["B"] * n}
        if split != "test":
            values.update({"class": classes * 3,
                           "x": [float(i % 17 + (50 if split == "validation" else 0)) for i in range(n)],
                           "y": [float(i % 11 + 1) for i in range(n)]})
            rows.extend({"source_split": split, "source_row_index": i, "sample_id": f"{split}_{i}|B",
                         "raw_class_id": values["class"][i], "fold_id": i // 178} for i in range(n))
        if split == "test" and getattr(request, "param", False):
            values.update({"class": [classes[0]], "x": [3.0], "y": [4.0]})
        # Normally test has no descriptors/labels: validation-only must work.
        pq.write_table(pa.table(values), paths[split])
    folds = tmp_path / "folds"
    save_fold_artifact(folds, pa.Table.from_pylist(rows, schema=FOLD_ASSIGNMENT_SCHEMA),
                       sources={s: describe_fold_source(p) for s, p in paths.items()}, fold_seed=42)
    tasks, offset = [], 0
    for i, size in enumerate(pilot.P3_LAYOUT):
        tasks.append({"task_id": i, "classes": classes[offset:offset + size]})
        offset += size
    taskfile = tmp_path / "tasks.json"
    taskfile.write_text(json.dumps({"protocol": "tail_to_head", "tasks": tasks}))
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
    monkeypatch.setattr(pilot, "TDDI_PAPER_INPUT_DIM", 2)
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


def test_ab_two_task_full_protocol_alignment_and_losses(data, monkeypatch, tiny):
    def forbidden(*a, **kw):
        pytest.fail("No runtime fold build, legacy scaler, test evaluation or legacy checkpoint allowed")
    for name in ("build_stratified_fold_assignments", "load_scaler_payload", "save_replay_checkpoint", "save_ewc_checkpoint"):
        monkeypatch.setattr(engine, name, forbidden)
    monkeypatch.setattr(pilot, "load_split_frame", forbidden)
    calls, original = [], pilot.load_development_fold_arrays
    def traced(*a, **kw):
        calls.append((kw["role"], kw["class_ids"]))
        return original(*a, **kw)
    monkeypatch.setattr(pilot, "load_development_fold_arrays", traced)
    roots = []
    for policy in ("raw_identity", "task0_standard_frozen"):
        command = cli(data, policy)
        monkeypatch.setattr(sys, "argv", command)
        engine.main()
        root = data["root"] / f"run_{policy}"
        roots.append(root)
        summary = json.loads((root / "run_summary.json").read_text())
        assert summary["completed_task_id"] == 1 and not summary["full_protocol_complete"]
        assert [t["head_size"] for t in summary["tasks"]] == [38, 58]
        resolved = json.loads((root / "run_config.json").read_text())["resolved"]
        assert resolved["protocol_task_count"] == 8 and resolved["global_slot_budget"] == 42
        assert resolved["gradient_accumulation"] == 16 and resolved["validation_only"]
        metrics = pd.read_csv(root / "metrics.csv")
        assert set(metrics.split) == {"validation"}
        assert set(metrics[metrics.task == 1].group) == {"seen_all", "old", "current"}
        losses = pd.read_csv(root / "training_audit.csv")
        assert np.isfinite(losses.select_dtypes(include="number")).all().all()
        np.testing.assert_allclose(losses.total_loss, losses.classification_loss + losses.scaled_logit_distillation_loss
                                   + losses.scaled_feature_distillation_loss, rtol=1e-6)
        assert (losses[losses.task == 0].logit_distillation_loss == 0).all()
        assert (losses[losses.task == 1].feature_distillation_loss > 0).any()
        assert (losses.optimizer_steps == 1).all()
        assert losses[losses.task == 0].examples_seen.tolist() == [152, 152]
        assert losses[losses.task == 1].examples_seen.tolist() == [91, 91]
        for task_id in (0, 1):
            ckpt = torch.load(root / f"task_{task_id}/best_model.pt", weights_only=True)
            assert not ckpt["resumable"] and ckpt["model_state"]["head.weight"].shape[0] == [38, 58][task_id]
        assert not (root / "task_2").exists()
    assert tiny == [38, 58, 38, 58]
    spec = json.loads(data["tasks"].read_text())["tasks"]
    for i, (role, classes) in enumerate(calls):
        task = (i % 4) // 2
        assert set(classes) == set(spec[task]["classes"] if role == "train" else
                                  [c for t in spec[:task + 1] for c in t["classes"]])
    for task_id in (0, 1):
        left, right = [r / f"task_{task_id}" for r in roots]
        a, b = [json.loads((r / "input_audit.json").read_text()) for r in (left, right)]
        for key in ("current_ids", "replay_ids_before", "validation_ids", "seen_class_map", "sampler"):
            assert a[key] == b[key]
        assert not set(a["current_ids"] + a["replay_ids_before"]) & set(a["validation_ids"])
        assert json.loads((left / "buffer_audit.json").read_text()) == json.loads((right / "buffer_audit.json").read_text())
        for epoch in (1, 2):
            assert json.loads((left / f"epoch_{epoch}_audit.json").read_text())["sampling"] == json.loads((right / f"epoch_{epoch}_audit.json").read_text())["sampling"]
    # Duplicate launch cannot erase successful artifacts.
    before = (roots[-1] / "run_summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        engine.main()
    assert (roots[-1] / "run_summary.json").read_bytes() == before


def test_fold_training_exports_assignment_bound_validation_predictions(data, monkeypatch, tiny):
    command = cli(
        data, suffix="_predictions",
        extra=["--export-member-predictions", "--member-prediction-splits", "validation"],
    )
    monkeypatch.setattr(sys, "argv", command)
    engine.main()
    root = data["root"] / "run_raw_identity_predictions"
    for task_id, class_count in ((0, 38), (1, 58)):
        path = root / "member_predictions" / f"task_{task_id}" / "validation.npz"
        artifact = load_member_prediction_artifact(path)
        assert artifact.context.ensemble_mode == "stratified_3fold"
        assert artifact.context.fold_id == artifact.context.member_id == 0
        assert artifact.class_count == class_count
        assert artifact.provenance is not None
        assert artifact.provenance.preprocessing_member_id == 0
        assert artifact.provenance.preprocessing_policy == "raw_identity"
        assert artifact.provenance.expected_oof_rows == artifact.row_count * 3
        np.testing.assert_array_equal(artifact.fold_ids, 0)


@pytest.mark.parametrize("extra,match", [
    (["--method", "ewc"], "requires replay"), (["--seed", "1"], "seed 0"),
    (["--fold-id", "1"], "member=held-out"), (["--scaler", "old.pkl"], "legacy --scaler"),
        (["--resume-replay-checkpoint", "old.pt"], "Prompt 10"),
        (["--max-train-rows-per-task", "4"], "complete"),
    (["--batch-size", "128"], "microbatch 64"), (["--stop-after-task", "8"], "between 0 and 7"),
])
def test_options_fail_before_io(data, monkeypatch, extra, match):
    monkeypatch.setattr(sys, "argv", cli(data, extra=extra))
    with pytest.raises(ValueError, match=match):
        engine.main()
    assert not (data["root"] / "run_raw_identity").exists()


def test_cli_legacy_and_explicit_budget_guard(data, monkeypatch):
    argv = cli(data)
    legacy = argv[:argv.index("--fold-replay-policy")] + ["--scaler", "legacy.pkl"]
    monkeypatch.setattr(sys, "argv", legacy)
    args = engine.parse_args()
    assert args.batch_size == 1024 and args.effective_batch_size is None
    assert args.total_memory_budget == args.replay_draws_per_epoch == 6800
    assert args.fold_replay_policy is None
    monkeypatch.setattr(sys, "argv", cli(data, extra=["--total-memory-budget=6800"]))
    with pytest.raises(SystemExit):
        engine.parse_args()


def test_protocol_and_preprocessing_mismatch_no_output(data, monkeypatch, tiny):
    spec = json.loads(data["tasks"].read_text())
    for bad in ({**spec, "tasks": spec["tasks"][:2]}, {**spec, "protocol": "random"}):
        with pytest.raises(ValueError, match="full P3"):
            pilot.validate_p3_spec(bad, data["context"])
    bad = deepcopy(spec)
    bad["tasks"][1]["classes"][0] = bad["tasks"][0]["classes"][0]
    with pytest.raises(ValueError, match="178"):
        pilot.validate_p3_spec(bad, data["context"])
    monkeypatch.setattr(sys, "argv", cli(data, extra=["--preprocessing-policy", "task0_standard_frozen"]))
    with pytest.raises(ValueError, match="policy mismatch"):
        engine.main()
    assert not (data["root"] / "run_raw_identity").exists()


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


def test_early_stopping_passes_best_not_last_to_teacher(data, monkeypatch, tiny):
    command = cli(data, suffix="_early", extra=["--epochs", "5", "--patience", "1"])
    monkeypatch.setattr(sys, "argv", command)
    original_train, original_eval = engine.train_one_epoch, engine.evaluate_model
    states, pending = {}, []
    def train(model, *a, **kw):
        result = original_train(model, *a, **kw)
        size = model.head.out_features
        if kw["teacher_model"] is not None:
            for k, v in kw["teacher_model"].state_dict().items():
                torch.testing.assert_close(v, states[38][0][k], rtol=0, atol=0)
            assert not any(p.requires_grad for p in kw["teacher_model"].parameters())
        states.setdefault(size, []).append({k: v.detach().clone() for k, v in model.state_dict().items()})
        pending.append(size)
        return result
    def evaluate(*a, **kw):
        result = original_eval(*a, **kw)
        if pending:
            result.metrics["macro_f1"] = 1.0 / len(states[pending.pop()])
        return result
    monkeypatch.setattr(engine, "train_one_epoch", train)
    monkeypatch.setattr(engine, "evaluate_model", evaluate)
    engine.main()
    root = data["root"] / "run_raw_identity_early"
    summary = json.loads((root / "run_summary.json").read_text())
    assert [s["epochs_trained"] for s in summary["tasks"]] == [2, 2]
    assert [s["best_epoch"] for s in summary["tasks"]] == [1, 1]
    assert any(not torch.equal(states[38][0][k], states[38][1][k]) for k in states[38][0])


def test_source_tamper_rejected_before_output(data, monkeypatch, tiny):
    path = data["paths"]["train"]
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(sys, "argv", cli(data))
    with pytest.raises(ValueError, match="hash|SHA256|sha256"):
        engine.main()
    assert not (data["root"] / "run_raw_identity").exists()


def test_legacy_ewc_loss_and_update_unchanged(tiny):
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    reference = deepcopy(model)
    x, y = torch.randn(5, 2), torch.tensor([0, 0, 1, 1, 0])
    fisher = {k: torch.ones_like(v) for k, v in model.named_parameters()}
    theta = {k: v.detach().clone() + .1 for k, v in model.named_parameters()}
    criterion = engine.FocalLoss()
    expected_classification = criterion(reference(x), y)
    expected_penalty = engine.ewc_penalty(reference, fisher, theta)
    expected = expected_classification + 2 * expected_penalty
    expected.backward()
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    actual = engine.train_one_epoch(model,
        torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=5),
        torch.optim.SGD(model.parameters(), lr=.01), criterion, "cpu", fisher=fisher,
        theta_star=theta, ewc_lambda=2, return_loss_components=True)
    assert actual.total_loss == pytest.approx(expected.item())
    assert actual.raw_ewc_penalty == pytest.approx(expected_penalty.item())
    assert actual.logit_distillation_loss == actual.feature_distillation_loss == 0
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("data", [True], indirect=True)
def test_optional_test_report_only_after_best_selection(data, monkeypatch, tiny):
    command = cli(data, extra=["--stop-after-task", "0", "--epochs", "1"])
    command.remove("--validation-only")
    monkeypatch.setattr(sys, "argv", command)
    root = data["root"] / "run_raw_identity"
    original, calls = pilot.load_split_frame, []
    def read(*a, **kw):
        assert (root / "task_0/best_model.pt").is_file()
        calls.append(str(a[0]))
        return original(*a, **kw)
    monkeypatch.setattr(pilot, "load_split_frame", read)
    engine.main()
    assert calls == [str(data["paths"]["test"])]
    metrics = pd.read_csv(root / "metrics.csv")
    assert set(metrics.split) == {"validation", "test"}
    assert metrics[(metrics.split == "test") & (metrics.group == "seen_all")].sample_count.tolist() == [1]
