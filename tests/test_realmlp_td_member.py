"""RealMLP-TD numerical parity and architecture tests."""
import json
from pathlib import Path

import pytest
import torch

from src.models.realmlp_td_member import (
    PBLDEmbedding, ParametricSELU, RealMLPTDConfig, RealMLPTDMember,
    UPSTREAM_COMMIT, make_optimizer, schedule_factors, update_training_schedule,
)
from src.training import train_cil as engine

ORACLE = json.loads((Path(__file__).parent / "fixtures/realmlp_td_upstream.json").read_text())


def tiny_config(classes=3):
    return RealMLPTDConfig(classes, input_dim=2, hidden_dims=(8, 4), dropout=0,
                           embedding_frequencies=4, embedding_dim=3)


def test_pbld_forward_and_gradients_match_pinned_upstream_oracle():
    assert ORACLE["upstream_commit"] == UPSTREAM_COMMIT
    model = PBLDEmbedding(tiny_config()).double()
    case = ORACLE["pbld"]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.tensor(case[name], dtype=torch.float64))
    outputs = model(torch.tensor(case["input"], dtype=torch.float64))
    torch.testing.assert_close(outputs, torch.tensor(case["output"], dtype=torch.float64), atol=1e-12, rtol=1e-12)
    outputs.square().mean().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, torch.tensor(case["grads"][name], dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_schedules_and_optimizer_groups():
    for row in ORACLE["schedules"]:
        assert schedule_factors(row["t"]) == pytest.approx((row["lr"], row["regularization"]), abs=1e-14)
    model = RealMLPTDMember(RealMLPTDConfig(3, input_dim=2, hidden_dims=(4,)))
    optimizer = make_optimizer(model, lr=.04, weight_decay=.02)
    update_training_schedule(model, optimizer, .75)
    assert model.blocks[0]["dropout"].p == pytest.approx(.075)


def test_native_architecture_and_parametric_activation():
    model = RealMLPTDMember(RealMLPTDConfig(178)).eval()
    assert model.head.weight.shape == (178, 256)
    activation = ParametricSELU(3)
    x = torch.tensor([[-2., 0., 3.]])
    torch.testing.assert_close(activation(x), torch.nn.functional.selu(x))


def test_train_only_initialization_and_class_expansion():
    torch.manual_seed(8)
    original = RealMLPTDMember(tiny_config(2)).eval()
    x = torch.randn(24, 2)
    original.initialize_for_task(x, [0, 1], initialize_backbone=True)
    previous_map, seen_map = {20: 0, 50: 1}, {10: 0, 20: 1, 50: 2}
    expanded = RealMLPTDMember(tiny_config(3)).eval()
    engine.copy_previous_state_to_expanded_model(original, expanded, previous_map, seen_map, variant="realmlp_td")
    expanded.initialize_for_task(x, [0], initialize_backbone=False)
    torch.testing.assert_close(expanded(x)[:, [1, 2]], original(x), rtol=0, atol=0)
