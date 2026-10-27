import torch
from src.methods.agem import project_agem_gradient
from src.methods.ewc import ewc_penalty, grow_head_state


def test_agem_projection_satisfies_reference_constraint():
    current = torch.tensor([-2.0, 1.0, 0.5])
    reference = torch.tensor([1.0, 0.0, 0.0])
    projected = project_agem_gradient(current, reference)
    assert torch.dot(projected, reference) >= -1e-7
    torch.testing.assert_close(projected, torch.tensor([0.0, 1.0, 0.5]))
    torch.testing.assert_close(project_agem_gradient(-current, reference), -current)


def test_ewc_penalty_and_class_remapping():
    model = torch.nn.Linear(2, 2)
    anchors = {k: v.detach().clone() for k, v in model.named_parameters()}
    fisher = {k: torch.ones_like(v) for k, v in anchors.items()}
    assert ewc_penalty(model, fisher, anchors).item() == 0
    with torch.no_grad():
        model.bias[0] += 2
    assert ewc_penalty(model, fisher, anchors).item() == 4
    state = {"head.weight": torch.tensor([[2., 3.]]), "head.bias": torch.tensor([4.])}
    grown = grow_head_state(state, {30: 0}, {10: 0, 30: 1},
        {"head.weight": torch.zeros(2, 2), "head.bias": torch.zeros(2)}, zero_new_rows=True)
    torch.testing.assert_close(grown["head.weight"], torch.tensor([[0., 0.], [2., 3.]]))
