from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from src.methods.weight_alignment import (
    WEIGHT_ALIGNMENT_NEW_CLASS_MEAN_NORM,
    WEIGHT_ALIGNMENT_NONE,
    align_new_class_weights_,
)


class TinyHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(2, 4)


def test_weight_alignment_scales_only_new_weight_rows() -> None:
    model = TinyHead()
    seen = {30: 0, 10: 1, 40: 2, 20: 3}
    previous = {10: 0, 20: 1}
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 4.0]]))
        model.head.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    before_weight = model.head.weight.detach().clone()
    before_bias = model.head.bias.detach().clone()

    audit = align_new_class_weights_(
        model, previous, seen, policy=WEIGHT_ALIGNMENT_NEW_CLASS_MEAN_NORM
    )

    assert audit["applied"] is True
    assert audit["gamma"] == pytest.approx(2.0)
    assert audit["old_mean_weight_norm"] == pytest.approx(3.0)
    assert audit["new_mean_weight_norm_before"] == pytest.approx(1.5)
    assert audit["new_mean_weight_norm_after"] == pytest.approx(3.0)
    torch.testing.assert_close(model.head.weight[[1, 3]], before_weight[[1, 3]])
    torch.testing.assert_close(model.head.weight[[0, 2]], before_weight[[0, 2]] * 2)
    torch.testing.assert_close(model.head.bias, before_bias)


def test_weight_alignment_task0_and_disabled_are_noops() -> None:
    model = TinyHead()
    original = deepcopy(model.state_dict())
    task0 = align_new_class_weights_(
        model, {}, {10: 0, 20: 1, 30: 2, 40: 3},
        policy=WEIGHT_ALIGNMENT_NEW_CLASS_MEAN_NORM,
    )
    assert task0["applied"] is False
    assert task0["skip_reason"] == "no_old_classes_at_task_0"
    disabled = align_new_class_weights_(
        model, {10: 0}, {10: 0, 20: 1, 30: 2, 40: 3}, policy=WEIGHT_ALIGNMENT_NONE
    )
    assert disabled["applied"] is False
    assert disabled["skip_reason"] == "disabled"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_weight_alignment_rejects_zero_new_norm() -> None:
    model = TinyHead()
    with torch.no_grad():
        model.head.weight.fill_(1)
        model.head.weight[1:].zero_()
    with pytest.raises(ValueError, match="positive"):
        align_new_class_weights_(
            model,
            {10: 0},
            {10: 0, 20: 1, 30: 2, 40: 3},
            policy=WEIGHT_ALIGNMENT_NEW_CLASS_MEAN_NORM,
        )
