from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.methods.ewc import compute_fisher, ewc_penalty, grow_head_state
from src.models.tddi_paper_member import TDDIPaperMember, TDDIPaperMemberConfig
from src.training.train_cil import (
    EpochLossComponents,
    FocalLoss,
    copy_previous_state_to_expanded_model,
    train_one_epoch,
)


def _tiny_member(num_classes: int) -> TDDIPaperMember:
    return TDDIPaperMember(
        TDDIPaperMemberConfig(
            input_dim=4,
            hidden_dims=(8, 6),
            num_classes=num_classes,
            dropout=0.0,
            activation="relu",
        )
    )


def _parameter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def test_ewc_penalty_is_zero_at_theta_star_and_positive_after_important_shift() -> None:
    model = _tiny_member(num_classes=2)
    theta_star = _parameter_state(model)
    fisher = {
        name: torch.ones_like(parameter)
        for name, parameter in model.named_parameters()
    }

    torch.testing.assert_close(ewc_penalty(model, fisher, theta_star), torch.tensor(0.0))
    with torch.no_grad():
        model.head.weight[0, 0].add_(2.0)

    penalty = ewc_penalty(model, fisher, theta_star)
    torch.testing.assert_close(penalty, torch.tensor(4.0), rtol=0, atol=0)
    assert penalty.item() > 0.0


def test_head_growth_maps_old_rows_and_new_rows_are_initially_unpenalized() -> None:
    previous_map = {30: 0, 10: 1}
    current_map = {10: 0, 20: 1, 30: 2}
    previous = _tiny_member(num_classes=2)
    with torch.no_grad():
        previous.head.weight[0].fill_(0.3)
        previous.head.weight[1].fill_(-0.7)
        previous.head.bias.copy_(torch.tensor([0.2, -0.4]))
    fisher = {
        name: torch.full_like(parameter, float(index + 1))
        for index, (name, parameter) in enumerate(previous.named_parameters())
    }
    theta_star = _parameter_state(previous)

    expanded = copy_previous_state_to_expanded_model(
        previous,
        _tiny_member(num_classes=3),
        previous_map,
        current_map,
        variant="tddi_paper_member",
    )
    reference_state = expanded.state_dict()
    grown_fisher = grow_head_state(
        fisher,
        previous_map,
        current_map,
        reference_state,
        zero_new_rows=True,
    )
    grown_theta = grow_head_state(
        theta_star,
        previous_map,
        current_map,
        reference_state,
        zero_new_rows=False,
    )

    torch.testing.assert_close(grown_fisher["head.weight"][2], fisher["head.weight"][0])
    torch.testing.assert_close(grown_fisher["head.weight"][0], fisher["head.weight"][1])
    torch.testing.assert_close(grown_fisher["head.weight"][1], torch.zeros(6))
    torch.testing.assert_close(grown_fisher["head.bias"][1], torch.tensor(0.0))
    torch.testing.assert_close(ewc_penalty(expanded, grown_fisher, grown_theta), torch.tensor(0.0))

    with torch.no_grad():
        expanded.head.weight[1].add_(100.0)
        expanded.head.bias[1].add_(100.0)
    torch.testing.assert_close(ewc_penalty(expanded, grown_fisher, grown_theta), torch.tensor(0.0))

    with torch.no_grad():
        expanded.head.weight[current_map[10], 0].add_(1.0)
    assert ewc_penalty(expanded, grown_fisher, grown_theta).item() > 0.0


def test_member_training_uses_focal_loss_and_fisher_uses_cross_entropy() -> None:
    torch.manual_seed(4)
    features = torch.tensor(
        [[0.1, -0.2, 0.3, 0.4], [-0.4, 0.5, 0.2, -0.1]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)
    model = _tiny_member(num_classes=2)
    focal = FocalLoss(gamma=1.0)
    with torch.no_grad():
        expected_focal = focal(model(features), labels).item()
        expected_cross_entropy = F.cross_entropy(model(features), labels).item()

    components = train_one_epoch(
        model,
        loader,
        torch.optim.SGD(model.parameters(), lr=0.0),
        focal,
        "cpu",
        return_loss_components=True,
    )
    assert isinstance(components, EpochLossComponents)
    assert abs(components.classification_loss - expected_focal) < 1e-7
    assert abs(components.total_loss - expected_focal) < 1e-7
    assert abs(expected_focal - expected_cross_entropy) > 1e-4

    manual_model = _tiny_member(num_classes=2)
    manual_model.load_state_dict(model.state_dict())
    manual_model.zero_grad(set_to_none=True)
    F.cross_entropy(manual_model(features), labels).backward()
    expected_fisher = {
        name: parameter.grad.detach().square().clone()
        for name, parameter in manual_model.named_parameters()
    }
    computed_fisher = compute_fisher(model, loader, "cpu")
    for name in expected_fisher:
        torch.testing.assert_close(computed_fisher[name], expected_fisher[name])


def test_two_task_tddi_member_ewc_synthetic_smoke() -> None:
    torch.manual_seed(12)
    previous_map = {10: 0, 30: 1}
    current_map = {10: 0, 20: 1, 30: 2}
    task0_features = torch.randn(8, 4)
    task0_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    task0_loader = DataLoader(
        TensorDataset(task0_features, task0_labels),
        batch_size=2,
        shuffle=False,
    )
    criterion = FocalLoss(gamma=1.0)
    task0_model = _tiny_member(num_classes=2)
    task0_components = train_one_epoch(
        task0_model,
        task0_loader,
        torch.optim.SGD(task0_model.parameters(), lr=0.03),
        criterion,
        "cpu",
        return_loss_components=True,
    )
    assert isinstance(task0_components, EpochLossComponents)
    assert task0_components.raw_ewc_penalty == 0.0

    fisher = compute_fisher(task0_model, task0_loader, "cpu")
    theta_star = _parameter_state(task0_model)
    task1_model = copy_previous_state_to_expanded_model(
        task0_model,
        _tiny_member(num_classes=3),
        previous_map,
        current_map,
        variant="tddi_paper_member",
    )
    reference_state = task1_model.state_dict()
    grown_fisher = grow_head_state(
        fisher,
        previous_map,
        current_map,
        reference_state,
        zero_new_rows=True,
    )
    grown_theta = grow_head_state(
        theta_star,
        previous_map,
        current_map,
        reference_state,
        zero_new_rows=False,
    )
    assert torch.count_nonzero(grown_fisher["head.weight"][current_map[20]]) == 0
    torch.testing.assert_close(ewc_penalty(task1_model, grown_fisher, grown_theta), torch.tensor(0.0))

    task1_features = torch.randn(8, 4)
    task1_labels = torch.tensor([1, 0, 1, 2, 1, 0, 1, 2])
    task1_loader = DataLoader(
        TensorDataset(task1_features, task1_labels),
        batch_size=2,
        shuffle=False,
    )
    ewc_lambda = 2.5
    task1_components = train_one_epoch(
        task1_model,
        task1_loader,
        torch.optim.SGD(task1_model.parameters(), lr=0.03),
        criterion,
        "cpu",
        fisher=grown_fisher,
        theta_star=grown_theta,
        ewc_lambda=ewc_lambda,
        return_loss_components=True,
    )

    assert isinstance(task1_components, EpochLossComponents)
    assert task1_model(task1_features[:2]).shape == (2, 3)
    assert task1_components.raw_ewc_penalty > 0.0
    assert abs(
        task1_components.scaled_ewc_penalty
        - ewc_lambda * task1_components.raw_ewc_penalty
    ) < 1e-7
    assert abs(
        task1_components.total_loss
        - task1_components.classification_loss
        - task1_components.scaled_ewc_penalty
    ) < 1e-6
