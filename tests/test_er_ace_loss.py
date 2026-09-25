import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.training import train_cil as engine
from src.training.train_cil import er_ace_classification_loss


def test_er_ace_masks_old_logits_for_current_examples_but_not_replay_examples():
    logits = torch.tensor(
        [[4.0, 3.0, 0.0, 2.0], [3.0, 1.0, 2.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([2, 0])  # current class 2, replay-old class 0

    loss = er_ace_classification_loss(logits, labels, old_class_indices=[0, 1])
    expected = F.cross_entropy(logits[0:1, 2:4], torch.tensor([0]))
    expected += F.cross_entropy(logits[1:2], torch.tensor([0]))
    torch.testing.assert_close(loss, expected)

    loss.backward()
    torch.testing.assert_close(logits.grad[0, :2], torch.zeros(2))
    assert torch.count_nonzero(logits.grad[1, 2:]) == 2


def test_er_ace_task_zero_uses_all_available_classes():
    logits = torch.tensor([[1.0, -1.0, 0.5]], requires_grad=True)
    labels = torch.tensor([2])
    loss = er_ace_classification_loss(logits, labels, old_class_indices=[])
    torch.testing.assert_close(loss, F.cross_entropy(logits, labels))


def test_er_ace_training_variant_dispatches_asymmetric_loss_without_distillation():
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 4)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([2, 0])
    expected = er_ace_classification_loss(model(features), labels, [0, 1]).item()

    result = engine.train_one_epoch(
        model,
        DataLoader(TensorDataset(features, labels), batch_size=2),
        torch.optim.SGD(model.parameters(), lr=0.0),
        engine.FocalLoss(gamma=1.0),
        "cpu",
        teacher_raw_classes=[10, 20],
        current_seen_map={10: 0, 20: 1, 30: 2, 40: 3},
        loss_variant="er_ace",
        return_loss_components=True,
    )

    assert result.classification_loss == expected
    assert result.logit_distillation_loss == result.feature_distillation_loss == 0.0
