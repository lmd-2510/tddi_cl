"""Focused IDER tests using the three real repository backbones."""
from copy import deepcopy

import pytest
import torch

from src.methods.ider import (
    IDERTrainingContext,
    attach_ider_adapter,
    compute_ider_loss,
    ider_forward,
    optimizer_step_preserving_future_classes,
    seen_class_probabilities,
    uniform_class_signal,
)
from src.models.ddinet_member import DDINetMember, DDINetMemberConfig
from src.models.mfp_siamese_member import MFPSiameseMember, MFPSiameseMemberConfig
from src.models.tddi_paper_member import TDDIPaperMember, TDDIPaperMemberConfig
from src.training.class_incremental_policy import ClassIncrementalPolicy


def _backbones():
    return [
        (
            TDDIPaperMember(TDDIPaperMemberConfig(
                input_dim=8, hidden_dims=(7, 5), num_classes=178, dropout=0.0,
            )),
            8,
        ),
        (
            DDINetMember(DDINetMemberConfig(
                input_dim=8, hidden_dims=(7, 6, 5, 4), num_classes=178, dropout=0.0,
            )),
            8,
        ),
        (
            MFPSiameseMember(MFPSiameseMemberConfig(
                drug_input_dim=4, encoder_hidden_dim=6, drug_latent_dim=3,
                classifier_hidden_dim=5, num_classes=178, dropout=0.0,
            )),
            8,
        ),
    ]


@pytest.mark.parametrize("model,input_dim", _backbones())
def test_ider_adapter_runs_on_all_real_backbones(model, input_dim):
    model.eval()
    attach_ider_adapter(model, active_class_indices=(0, 1, 2))
    features = torch.randn(4, input_dim)
    signal = uniform_class_signal(
        4, (0, 1, 2), device="cpu", dtype=features.dtype,
    )
    assert signal[:, :3].sum(dim=1).tolist() == pytest.approx([1.0] * 4)
    assert torch.count_nonzero(signal[:, 3:]).item() == 0
    logits = ider_forward(model, features)
    assert logits.shape == (4, 178)
    probabilities = seen_class_probabilities(logits, (0, 1, 2))
    assert torch.count_nonzero(probabilities[:, 3:]).item() == 0
    torch.testing.assert_close(probabilities[:, :3].sum(dim=1), torch.ones(4))


def test_ider_loss_and_optimizer_preserve_future_policy():
    torch.manual_seed(9)
    global_map = {raw: raw for raw in range(178)}
    old = TDDIPaperMember(TDDIPaperMemberConfig(
        input_dim=8, hidden_dims=(7, 5), num_classes=178, dropout=0.0,
    ))
    attach_ider_adapter(old, active_class_indices=(0, 1))
    old.eval().requires_grad_(False)
    model = deepcopy(old)
    model.train().requires_grad_(True)
    attach_ider_adapter(model, active_class_indices=(0, 1, 2))
    policy = ClassIncrementalPolicy(global_map, active_raw_classes=(0, 1, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    features = torch.randn(5, 8)
    labels = torch.tensor([0, 1, 2, 2, 1])
    context = IDERTrainingContext(
        old_model=old,
        replay_features=torch.randn(6, 8),
        replay_labels=torch.tensor([0, 1, 0, 1, 0, 1]),
        active_class_indices=(0, 1, 2),
        old_class_indices=(0, 1),
        replay_batch_size=4,
        weight_old_replay_id=0.5,
        weight_current_id=0.4,
        weight_replay_supervised=0.5,
        uniform_probability=1.0,
        generator=torch.Generator().manual_seed(13),
    )
    future_head = model.head.weight.detach()[3:].clone()
    future_projection = model.ider_adapter.projection.weight.detach()[:, 3:].clone()
    breakdown = compute_ider_loss(model, features, labels, context)
    assert all(torch.isfinite(value) for value in (
        breakdown.current_supervised,
        breakdown.current_idempotence_distillation,
        breakdown.replay_supervised,
        breakdown.replay_idempotence_distillation,
        breakdown.total,
    ))
    breakdown.total.backward()
    optimizer_step_preserving_future_classes(model, optimizer, policy)
    torch.testing.assert_close(model.head.weight.detach()[3:], future_head, atol=0, rtol=0)
    torch.testing.assert_close(
        model.ider_adapter.projection.weight.detach()[:, 3:],
        future_projection,
        atol=0,
        rtol=0,
    )
