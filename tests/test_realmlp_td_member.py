"""RealMLP-TD: upstream numerical parity and the frozen-fold CIL contracts."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.realmlp_td_member import (
    RealMLPTDConfig, RealMLPTDMember, PBLDEmbedding, ParametricSELU,
    make_optimizer, update_training_schedule, schedule_factors, should_stop,
    UPSTREAM_COMMIT,
)
from src.training import fold_training, train_cil as engine
from src.training.fold_ensemble3_full import load_full_config
from src.training.fold_run_setup import member_command
from src.eval.predictions import load_member_prediction_artifact
from tests.test_fold_training import data, cli

ORACLE = json.loads((Path(__file__).parent / "fixtures/realmlp_td_upstream.json").read_text())


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(classes=3):
    return RealMLPTDConfig(classes, input_dim=2, hidden_dims=(8, 4), dropout=0,
                           embedding_frequencies=4, embedding_dim=3)


@pytest.fixture
def tiny_realmlp(monkeypatch):
    monkeypatch.setattr(fold_training, "TDDI_PAPER_INPUT_DIM", 2)
    def expand(previous, previous_map, seen_map, **kwargs):
        model = RealMLPTDMember(tiny_config(len(seen_map)))
        if previous is not None:
            engine.copy_previous_state_to_expanded_model(previous, model, previous_map, seen_map, variant="realmlp_td")
        return model
    monkeypatch.setattr(engine, "expand_model_for_seen_classes", expand)
    monkeypatch.setattr(engine, "model_manifest", lambda args: tiny_config(1).manifest())


def realmlp_cli(data, **kwargs):
    return cli(data, **kwargs) + ["--variant", "realmlp_td", "--batch-size", "8", "--effective-batch-size", "8", "--dropout", "0"]


def test_pbld_forward_and_gradients_match_pinned_upstream_oracle():
    assert ORACLE["upstream_commit"] == UPSTREAM_COMMIT
    model = PBLDEmbedding(tiny_config()).double()
    case = ORACLE["pbld"]
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.tensor(case[name], dtype=torch.float64))
    outputs = model(torch.tensor(case["input"], dtype=torch.float64))
    torch.testing.assert_close(outputs, torch.tensor(case["output"], dtype=torch.float64), atol=1e-12, rtol=1e-12)
    outputs.square().mean().backward()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, torch.tensor(case["grads"][name], dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_adam_parameter_groups_match_upstream_updates():
    names = {(1., 1.): "head.weight", (.1, 1.): "embedding.weight_1",
             (6., 1.): "front_scale", (.1, 0.): "head.bias"}
    for case in ORACLE["optimizer"]:
        p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        name = names[(case["lr_factor"], case["wd_factor"])]
        class SingleParameter:
            def named_parameters(self):
                return [(name, p)]
        opt = make_optimizer(SingleParameter(), lr=.04, weight_decay=.02)
        for gradient in ([.3, -.6], [-.1, .2], [.5, .4]):
            p.grad = torch.tensor(gradient, dtype=torch.float64)
            opt.step()
        torch.testing.assert_close(p, torch.tensor(case["expected"], dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_schedules_match_upstream_and_update_dropout():
    for row in ORACLE["schedules"]:
        assert schedule_factors(row["t"]) == pytest.approx((row["lr"], row["regularization"]), abs=1e-14)
    model = RealMLPTDMember(RealMLPTDConfig(3, input_dim=2, hidden_dims=(4,)))
    opt = make_optimizer(model, lr=.04, weight_decay=.02)
    update_training_schedule(model, opt, .75)
    assert model.blocks[0]["dropout"].p == pytest.approx(.075)
    assert opt.defaults["betas"] == (.9, .95)
    assert next(g for g in opt.param_groups if g["realmlp_name"] == "head.bias")["weight_decay"] == 0


def test_native_architecture_and_parametric_activation():
    model = RealMLPTDMember(RealMLPTDConfig(178)).eval()
    assert len(model.blocks) == 3 and model.head.weight.shape == (178, 256)
    assert model.embedding.weight_1.shape == (3780, 16)
    with torch.no_grad():
        logits, latent = model.forward_with_latent(torch.randn(2, 3780))
    assert logits.shape == (2, 178) and latent.shape == (2, 256)
    activation = ParametricSELU(3)
    x = torch.tensor([[-2., 0., 3.]])
    torch.testing.assert_close(activation(x), torch.nn.functional.selu(x))
    with torch.no_grad():
        activation.weight.zero_()
    torch.testing.assert_close(activation(x), x)


def test_train_only_initialization_and_class_expansion():
    torch.manual_seed(8)
    original = RealMLPTDMember(tiny_config(2)).eval()
    x = torch.randn(24, 2)
    original.initialize_for_task(x, [0, 1], initialize_backbone=True)
    # First projection has unit empirical std after its train-only initialization.
    embedded = original.embedding(x) * original.front_scale
    first = original.blocks[0]["linear"](embedded)
    torch.testing.assert_close(first.std(0, correction=0), torch.ones(8), atol=2e-6, rtol=2e-6)
    previous_map, seen_map = {20: 0, 50: 1}, {10: 0, 20: 1, 50: 2}
    expanded = RealMLPTDMember(tiny_config(3)).eval()
    engine.copy_previous_state_to_expanded_model(original, expanded, previous_map, seen_map, variant="realmlp_td")
    expanded.initialize_for_task(x, [0], initialize_backbone=False)
    torch.testing.assert_close(expanded(x)[:, [1, 2]], original(x), rtol=0, atol=0)
    for name, value in original.state_dict().items():
        if not name.startswith("head."):
            torch.testing.assert_close(value, expanded.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("protocol", ["ordered", "shuffled_seed0"])
@pytest.mark.parametrize("method", [engine.FIXED_BUDGET_METHOD, "ewc", "agem"])
def test_full_configs_and_defaults(protocol, method, tmp_path):
    config = load_full_config(f"configs/{protocol}_realmlp_td.json", overrides={"method": method, "output_root": tmp_path / "run"})
    args = engine.parse_args(member_command(config, 0)[2:])
    assert args.variant == "realmlp_td" and args.activation == "selu" and args.norm == "none"
    assert (args.epochs, args.patience, args.batch_size, args.effective_batch_size) == (256, 20, 256, 256)
    assert (args.lr, args.weight_decay, args.dropout, args.label_smoothing) == (.04, .02, .15, .1)
    assert args.gradient_clip_norm is None and args.loss_variant == "cross_entropy"


@pytest.mark.parametrize("method", [engine.FIXED_BUDGET_METHOD, "ewc", "agem"])
def test_two_task_train_export_and_checkpoint(data, tiny_realmlp, monkeypatch, method):
    command = realmlp_cli(data, extra=["--method", method, "--export-member-predictions", "--member-prediction-splits", "validation"])
    monkeypatch.setattr(sys, "argv", command)
    engine.main()
    root = data["root"] / "run_raw_identity"
    prepared = fold_training.prepare_fold_run(engine.parse_args(command[1:]), engine=engine)
    from src.training.replay_checkpoint import load_fold_replay_checkpoint
    state, _ = load_fold_replay_checkpoint(root / "checkpoints/task_1.pt", root=root,
        expected_contract=prepared["contract"], tasks=prepared["tasks"],
        context=prepared["context"], buffer_kwargs=prepared["buffer_kwargs"])
    assert state["model_state"]["head.weight"].shape == (58, 4)
    artifact = load_member_prediction_artifact(root / "member_predictions/task_1/validation.npz")
    assert artifact.probabilities.shape[1] == 58 and np.isfinite(artifact.logits).all()
    np.testing.assert_allclose(artifact.probabilities.sum(1), 1, atol=1e-6)
    resolved = json.loads((root / "run_config.json").read_text())["resolved"]
    assert resolved["label_smoothing"] == .1
    assert resolved["evaluation_loss_scope"] == "unsmoothed_cross_entropy"
    assert resolved["scheduler"]["lr"] == "coslog4"


def test_shared_cache_and_boundary_resume_are_exact(data, tiny_realmlp, monkeypatch):
    cache = str(data["root"] / "common_cache")
    for suffix, extra in (("_full", []), ("_resume", ["--stop-after-task", "0"]),
            ("_resume", ["--resume-fold-checkpoint", str(data["root"] / "run_raw_identity_resume/checkpoints/task_0.pt")])):
        monkeypatch.setattr(sys, "argv", realmlp_cli(data, suffix=suffix, extra=["--task-cache-root", cache, *extra]))
        engine.main()
    states = [torch.load(data["root"] / f"run_raw_identity_{name}/checkpoints/task_1.pt", weights_only=False)
              for name in ("full", "resume")]
    for key in states[0]["model_state"]:
        torch.testing.assert_close(states[0]["model_state"][key], states[1]["model_state"][key], rtol=0, atol=0)
    assert states[0]["progress"]["metric_rows"] == states[1]["progress"]["metric_rows"]


def test_upstream_early_stopping_rule():
    assert not should_stop(1, None)
    assert not should_stop(40, 10)
    assert should_stop(41, 10)
    assert not should_stop(41, 11)


def test_early_stopping_really_ends_training_and_keeps_best(data, tiny_realmlp, monkeypatch):
    monkeypatch.setattr(sys, "argv", realmlp_cli(data, extra=[
        "--stop-after-task", "0", "--epochs", "6", "--patience", "1", "--lr", "1e-20"]))
    engine.main()
    state = torch.load(data["root"] / "run_raw_identity/checkpoints/task_0.pt", weights_only=False)
    task = state["progress"]["task_summaries"][0]
    assert task["best_epoch"] == 1
    assert task["epochs_trained"] == 4  # first epoch satisfying epoch > 2*best + 1


@pytest.mark.parametrize("data", ["full_test"], indirect=True)
def test_eight_tasks_three_members_ensemble_report(data, tiny_realmlp, monkeypatch):
    from tests import test_full_pipeline as pipeline
    monkeypatch.setattr(pipeline, "cli", realmlp_cli)
    pipeline.test_eight_task_ensemble_threshold_and_auto_protocol_report(data, tiny_realmlp, monkeypatch, "head_to_tail")
    reports = list((data["root"] / "ensemble").rglob("*.md"))
    assert any("- Backbone: `realmlp_td`" in p.read_text(encoding="utf-8") for p in reports)
