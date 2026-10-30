"""TabM math, raw-class head expansion and the existing frozen-fold pipeline."""
import json
import sys
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("tabm", reason="Install requirements-tabm.txt for TabM integration tests.")

from src.models.tabm_member import TabMMember, TabMMemberConfig
from src.methods.ewc import grow_head_state
from src.training import fold_training, train_cil as engine
from src.training.fold_ensemble3_full import load_full_config
from src.training.fold_run_setup import member_command
from src.eval.predictions import load_member_prediction_artifact
from tests.test_fold_training import data, cli


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model(classes=3):
    return TabMMember(TabMMemberConfig(classes, input_dim=2, d_block=8, k=3, n_blocks=2, dropout=0))


@pytest.fixture
def tiny_tabm(monkeypatch):
    monkeypatch.setattr(fold_training, "TDDI_PAPER_INPUT_DIM", 2)
    def expand(previous, previous_map, seen_map, **kwargs):
        model = small_model(len(seen_map))
        if previous is not None:
            engine.copy_previous_state_to_expanded_model(previous, model, previous_map, seen_map, variant="tabm")
        return model
    monkeypatch.setattr(engine, "expand_model_for_seen_classes", expand)
    monkeypatch.setattr(engine, "model_manifest", lambda args: TabMMemberConfig(
        1, input_dim=2, d_block=8, k=3, n_blocks=2, dropout=0).manifest())


def tabm_cli(data, **kwargs):
    return cli(data, **kwargs) + ["--variant", "tabm", "--batch-size", "8", "--effective-batch-size", "8", "--dropout", "0"]


def test_probability_ensemble_is_not_mean_logits():
    branches = torch.tensor([[[10., 0.], [0., 0.]], [[-1000., 1000.], [1000., -1000.]]])
    effective = TabMMember.aggregate_logits(branches)
    torch.testing.assert_close(effective.softmax(-1), branches.softmax(-1).mean(1))
    assert not torch.allclose(effective.softmax(-1), branches.mean(1).softmax(-1))
    assert torch.isfinite(effective).all()


def test_native_default_size_and_branch_shape():
    model = TabMMember(TabMMemberConfig(178)).eval()
    assert sum(p.numel() for p in model.parameters()) == 5_633_728
    with torch.no_grad():
        x = torch.randn(2, 3780)
        branches = model.forward_branches(x)
        logits, latent = model.forward_with_latent(x)
    assert branches.shape == (2, 32, 178)
    assert latent.shape == (2, 512)
    torch.testing.assert_close(logits.softmax(-1), branches.softmax(-1).mean(1))


def test_training_loss_and_gradients_match_mean_branch_ce():
    torch.manual_seed(13)
    model = small_model()
    reference = deepcopy(model)
    x, y = torch.randn(5, 2), torch.tensor([0, 1, 2, 0, 1])
    branches = reference.forward_branches(x)
    expected = torch.stack([F.cross_entropy(branches[:, i], y) for i in range(3)]).mean()
    expected.backward()
    actual = engine.train_one_epoch(model,
        torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=5),
        torch.optim.SGD(model.parameters(), lr=0), torch.nn.CrossEntropyLoss(), "cpu",
        return_loss_components=True)
    assert actual.classification_loss == pytest.approx(expected.item())
    assert actual.examples_seen == 5  # K branches are not K independent data rows.
    # Check directly as train_one_epoch clears gradients after its optimizer step.
    _, loss = model.classification_loss(x, y, torch.nn.CrossEntropyLoss())
    loss.backward()
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_expand_head_and_ewc_by_raw_class():
    old, new = small_model(2).eval(), small_model(3).eval()
    previous_map, seen_map = {20: 0, 50: 1}, {10: 0, 20: 1, 50: 2}
    engine.copy_previous_state_to_expanded_model(old, new, previous_map, seen_map, variant="tabm")
    x = torch.randn(4, 2)
    torch.testing.assert_close(new.forward_branches(x)[..., [1, 2]], old.forward_branches(x))
    fisher = {key: torch.ones_like(value) for key, value in old.state_dict().items()}
    grown = grow_head_state(fisher, previous_map, seen_map, new.state_dict(), zero_new_rows=True)
    assert torch.count_nonzero(grown["head.weight"][..., 0]) == 0
    assert torch.all(grown["head.weight"][..., 1:] == 1)
    assert torch.all(grown["head.bias"][..., 1:] == 1)


@pytest.mark.parametrize("protocol", ["ordered", "shuffled_seed0"])
@pytest.mark.parametrize("method", [engine.FIXED_BUDGET_METHOD, "ewc", "agem"])
def test_config_and_cli_defaults(protocol, method, tmp_path):
    config = load_full_config(f"configs/{protocol}_tabm.json", overrides={"method": method, "output_root": tmp_path / "out"})
    args = engine.parse_args(member_command(config, 0)[2:])
    assert args.variant == "tabm" and args.loss_variant == "cross_entropy"
    assert (args.batch_size, args.effective_batch_size, args.evaluation_batch_size) == (256, 256, 256)
    assert args.epochs is None and args.patience == 16 and args.gradient_clip_norm == 1
    assert (args.lr, args.weight_decay, args.dropout) == (.002, .0003, .1)
    assert args.norm == "none" and args.activation == "relu"


@pytest.mark.parametrize("method", [engine.FIXED_BUDGET_METHOD, "ewc", "agem"])
def test_two_task_training_prediction_export_and_checkpoint(data, tiny_tabm, monkeypatch, method):
    command = tabm_cli(data, extra=["--method", method, "--export-member-predictions", "--member-prediction-splits", "validation"])
    monkeypatch.setattr(sys, "argv", command)
    engine.main()
    root = data["root"] / "run_raw_identity"
    state = torch.load(root / "checkpoints/task_1.pt", weights_only=False)
    assert state["model_state"]["head.weight"].shape == (3, 8, 58)
    # Exercise strict validation on a completed scope without retraining.
    args = engine.parse_args(command[1:])
    prepared = fold_training.prepare_fold_run(args, engine=engine)
    from src.training.replay_checkpoint import load_fold_replay_checkpoint
    load_fold_replay_checkpoint(root / "checkpoints/task_1.pt", root=root,
        expected_contract=prepared["contract"], tasks=prepared["tasks"],
        context=prepared["context"], buffer_kwargs=prepared["buffer_kwargs"])
    artifact = load_member_prediction_artifact(root / "member_predictions/task_1/validation.npz")
    assert artifact.probabilities.shape[1] == 58
    assert np.isfinite(artifact.probabilities).all()
    np.testing.assert_allclose(artifact.probabilities.sum(1), 1, atol=1e-6)
    resolved = json.loads((root / "run_config.json").read_text())["resolved"]
    assert resolved["model"]["prediction_aggregation"] == "mean_branch_probabilities"


def test_cached_task_boundary_resume_is_exact(data, tiny_tabm, monkeypatch):
    cache = str(data["root"] / "common_cache")
    for suffix, extra in (("_full", []), ("_resume", ["--stop-after-task", "0"]),
                          ("_resume", ["--resume-fold-checkpoint", str(data["root"] / "run_raw_identity_resume/checkpoints/task_0.pt")])):
        monkeypatch.setattr(sys, "argv", tabm_cli(data, suffix=suffix,
            extra=["--task-cache-root", cache, *extra]))
        engine.main()
    full = torch.load(data["root"] / "run_raw_identity_full/checkpoints/task_1.pt", weights_only=False)
    resumed = torch.load(data["root"] / "run_raw_identity_resume/checkpoints/task_1.pt", weights_only=False)
    for key in full["model_state"]:
        torch.testing.assert_close(full["model_state"][key], resumed["model_state"][key], rtol=0, atol=0)
    assert full["progress"]["metric_rows"] == resumed["progress"]["metric_rows"]


def test_no_epoch_cap_stops_on_validation_patience(data, tiny_tabm, monkeypatch):
    command = tabm_cli(data, extra=["--stop-after-task", "0", "--lr", "1e-20", "--patience", "1"])
    index = command.index("--epochs")
    del command[index:index + 2]
    monkeypatch.setattr(sys, "argv", command)
    engine.main()
    root = data["root"] / "run_raw_identity"
    state = torch.load(root / "checkpoints/task_0.pt", weights_only=False)
    assert state["contract"]["hyperparameters"]["epochs"] is None
    assert state["progress"]["task_summaries"][0]["epochs_trained"] == 2


@pytest.mark.parametrize("data", ["full_test"], indirect=True)
def test_full_eight_task_three_member_ensemble_and_report(data, tiny_tabm, monkeypatch):
    from tests import test_full_pipeline as pipeline
    monkeypatch.setattr(pipeline, "cli", tabm_cli)
    pipeline.test_eight_task_ensemble_threshold_and_auto_protocol_report(data, tiny_tabm, monkeypatch, "head_to_tail")
    # The final report filenames are protocol-specific; read the generated Markdown.
    reports = list((data["root"] / "ensemble").rglob("*.md"))
    assert any("- Backbone: `tabm`" in p.read_text(encoding="utf-8") for p in reports)
