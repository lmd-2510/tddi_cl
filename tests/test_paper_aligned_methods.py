import torch
from torch.utils.data import DataLoader, TensorDataset

from src.methods.agem import TaskEpisodicMemory, project_agem_gradient, reference_loss
from src.methods.ewc import compute_fisher, consolidate_quadratic_penalties
from src.methods.replay import (
    ReservoirMemory,
    build_experience_replay_batch,
    experience_replay_loss,
)
from src.training.class_incremental_policy import (
    fixed_head_cross_entropy,
    mask_unseen_logits,
)


def test_fixed_head_masks_unseen_rows_and_their_gradients():
    model = torch.nn.Linear(3, 178)
    features = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    labels = torch.tensor([2, 7])
    logits = model(features)
    masked = mask_unseen_logits(logits, [2, 7])
    assert torch.isneginf(masked[:, 0]).all()
    assert torch.isfinite(masked[:, [2, 7]]).all()

    fixed_head_cross_entropy(logits, labels, [2, 7]).backward()
    assert torch.count_nonzero(model.weight.grad[[2, 7]]) > 0
    inactive = torch.ones(178, dtype=torch.bool)
    inactive[[2, 7]] = False
    assert torch.count_nonzero(model.weight.grad[inactive]) == 0
    assert torch.count_nonzero(model.bias.grad[inactive]) == 0


def test_er_reservoir_checkpoint_and_equal_minibatch_stack():
    memory = ReservoirMemory(4, seed=13)
    features = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    labels = torch.arange(10, dtype=torch.long)
    memory.update(features[:5], labels[:5])
    memory.update(features[5:], labels[5:])
    assert len(memory) == 4
    assert memory.seen_examples == 10

    restored = ReservoirMemory.from_state_dict(memory.state_dict())
    expected_features, expected_labels = memory.sample(3)
    actual_features, actual_labels = restored.sample(3)
    torch.testing.assert_close(actual_features, expected_features)
    torch.testing.assert_close(actual_labels, expected_labels)

    current_features = torch.ones(2, 3)
    current_labels = torch.tensor([0, 1])
    batch_features, batch_labels = build_experience_replay_batch(
        current_features, current_labels, restored
    )
    assert batch_features.shape == (4, 3)
    assert batch_labels.shape == (4,)


def test_er_uses_one_cross_entropy_over_current_and_memory_rows():
    logits = torch.zeros(4, 178, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    loss = experience_replay_loss(logits, labels, [0, 1])
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 2:]) == 0


def test_original_ewc_consolidation_preserves_sum_gradient():
    old_fisher = {"weight": torch.tensor([2.0, 0.0])}
    old_anchor = {"weight": torch.tensor([1.0, 5.0])}
    new_fisher = {"weight": torch.tensor([3.0, 4.0])}
    new_anchor = {"weight": torch.tensor([3.0, 7.0])}
    fisher, anchor = consolidate_quadratic_penalties(
        old_fisher, old_anchor, new_fisher, new_anchor
    )
    torch.testing.assert_close(fisher["weight"], torch.tensor([5.0, 4.0]))
    torch.testing.assert_close(anchor["weight"], torch.tensor([2.2, 7.0]))
    theta = torch.tensor([4.0, 2.0])
    separate_gradient = (
        old_fisher["weight"] * (theta - old_anchor["weight"])
        + new_fisher["weight"] * (theta - new_anchor["weight"])
    )
    consolidated_gradient = fisher["weight"] * (theta - anchor["weight"])
    torch.testing.assert_close(consolidated_gradient, separate_gradient)


def test_ewc_fisher_is_zero_for_unseen_fixed_head_rows():
    model = torch.nn.Linear(2, 178)
    loader = DataLoader(
        TensorDataset(torch.randn(6, 2), torch.tensor([0, 1, 0, 1, 0, 1])),
        batch_size=3,
    )
    fisher = compute_fisher(model, loader, "cpu", active_class_indices=[0, 1])
    assert torch.count_nonzero(fisher["weight"][:2]) > 0
    assert torch.count_nonzero(fisher["weight"][2:]) == 0
    assert torch.count_nonzero(fisher["bias"][2:]) == 0


def test_agem_task_memory_and_reference_objective():
    memory = TaskEpisodicMemory(2, seed=9)
    memory.add_task(
        0,
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        torch.zeros(4, dtype=torch.long),
    )
    memory.add_task(
        1,
        torch.arange(12, 24, dtype=torch.float32).reshape(4, 3),
        torch.ones(4, dtype=torch.long),
    )
    assert memory.task_ids == (0, 1)
    assert len(memory) == 4

    restored = TaskEpisodicMemory.from_state_dict(memory.state_dict())
    expected = memory.sample_reference(3)
    actual = restored.sample_reference(3)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

    logits = torch.zeros(3, 178, requires_grad=True)
    loss = reference_loss(logits, actual[1], [0, 1])
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 2:]) == 0

    current = torch.tensor([-2.0, 1.0])
    reference = torch.tensor([1.0, 0.0])
    projected = project_agem_gradient(current, reference)
    assert torch.dot(projected, reference) >= -1e-7
