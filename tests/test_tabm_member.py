"""TabM architecture, branch aggregation and raw-class expansion tests."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("tabm", reason="Install requirements-tabm.txt for TabM tests.")

from src.methods.ewc import grow_head_state
from src.models.tabm_member import TabMMember, TabMMemberConfig
from src.training import train_cil as engine


def small_model(classes=3):
    return TabMMember(TabMMemberConfig(classes, input_dim=2, d_block=8, k=3, n_blocks=2, dropout=0))


def test_probability_ensemble_is_not_mean_logits():
    branches = torch.tensor([[[10., 0.], [0., 0.]], [[-1000., 1000.], [1000., -1000.]]])
    effective = TabMMember.aggregate_logits(branches)
    torch.testing.assert_close(effective.softmax(-1), branches.softmax(-1).mean(1))
    assert not torch.allclose(effective.softmax(-1), branches.mean(1).softmax(-1))


def test_native_default_size_and_branch_shape():
    model = TabMMember(TabMMemberConfig(178)).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_633_728
    with torch.no_grad():
        branches = model.forward_branches(torch.randn(2, 3780))
    assert branches.shape == (2, 32, 178)


def test_training_loss_and_gradients_match_mean_branch_ce():
    torch.manual_seed(13)
    model, reference = small_model(), None
    reference = deepcopy(model)
    x, y = torch.randn(5, 2), torch.tensor([0, 1, 2, 0, 1])
    branches = reference.forward_branches(x)
    expected = torch.stack([F.cross_entropy(branches[:, index], y) for index in range(3)]).mean()
    expected.backward()
    _, loss = model.classification_loss(x, y, torch.nn.CrossEntropyLoss())
    loss.backward()
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad)


def test_expand_head_and_ewc_by_raw_class():
    old, new = small_model(2).eval(), small_model(3).eval()
    previous_map, seen_map = {20: 0, 50: 1}, {10: 0, 20: 1, 50: 2}
    engine.copy_previous_state_to_expanded_model(old, new, previous_map, seen_map, variant="tabm")
    x = torch.randn(4, 2)
    torch.testing.assert_close(new.forward_branches(x)[..., [1, 2]], old.forward_branches(x))
    fisher = {key: torch.ones_like(value) for key, value in old.state_dict().items()}
    grown = grow_head_state(fisher, previous_map, seen_map, new.state_dict(), zero_new_rows=True)
    assert torch.count_nonzero(grown["head.weight"][..., 0]) == 0
