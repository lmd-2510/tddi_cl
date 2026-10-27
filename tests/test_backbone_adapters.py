import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.train_cil import FocalLoss, train_one_epoch




def test_gradient_accumulation_matches_effective_batch_including_partial_group() -> None:
    torch.manual_seed(3)
    features = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 0, 1, 1])
    dataset = TensorDataset(features, labels)
    reference = nn.Sequential(nn.Linear(3, 2))
    accumulated = nn.Sequential(nn.Linear(3, 2))
    accumulated.load_state_dict(reference.state_dict())
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)

    train_one_epoch(
        reference,
        DataLoader(dataset, batch_size=4, shuffle=False),
        reference_optimizer,
        FocalLoss(gamma=0.0),
        "cpu",
    )
    train_one_epoch(
        accumulated,
        DataLoader(dataset, batch_size=2, shuffle=False),
        accumulated_optimizer,
        FocalLoss(gamma=0.0),
        "cpu",
        gradient_accumulation_steps=2,
    )

    for expected, actual in zip(reference.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
