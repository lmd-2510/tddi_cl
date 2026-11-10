"""Focused UPCL unit tests: no dataset or full training trajectory required."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from src.data.replay_sampler import FoldFullMemorySampler, FULL_MEMORY_SAMPLER_POLICY
from src.methods.upcl import (
    UPCLTrainingContext,
    class_log_priors,
    compute_upcl_loss,
    enable_upcl_prototype_head,
    prepare_task_prototypes,
)
from src.models.ddinet_member import DDINetMember, DDINetMemberConfig
from src.models.mfp_siamese_member import MFPSiameseMember, MFPSiameseMemberConfig
from src.models.tddi_paper_member import TDDIPaperMember, TDDIPaperMemberConfig


@pytest.mark.parametrize("model,input_dim", [
    (TDDIPaperMember(TDDIPaperMemberConfig(input_dim=8, hidden_dims=(9, 8), num_classes=4, dropout=0.0)), 8),
    (DDINetMember(DDINetMemberConfig(input_dim=8, hidden_dims=(10, 9, 8, 8), num_classes=4, dropout=0.0)), 8),
    (MFPSiameseMember(MFPSiameseMemberConfig(drug_input_dim=4, encoder_hidden_dim=9, drug_latent_dim=4,
                                               classifier_hidden_dim=8, num_classes=4, dropout=0.0)), 8),
])
def test_upcl_enables_frozen_cosine_head_for_each_backbone(model, input_dim):
    enable_upcl_prototype_head(model, total_classes=4)
    assert not model.classifier_head.weight.requires_grad
    assert model.classifier_head.bias is None or not model.classifier_head.bias.requires_grad
    logits, latent = model.forward_with_latent(torch.randn(3, input_dim))
    assert logits.shape == (3, 4)
    assert latent.shape == (3, model.classifier_head.in_features)


def test_upcl_causally_assigns_orthogonal_prototypes_and_preserves_old_rows():
    torch.manual_seed(7)
    model = TDDIPaperMember(TDDIPaperMemberConfig(input_dim=6, hidden_dims=(10, 8), num_classes=4, dropout=0.0))
    first_x, first_y = torch.randn(12, 6), torch.tensor([0, 1] * 6)
    first = prepare_task_prototypes(
        model, None, current_features=first_x, current_labels=first_y,
        new_class_indices=(0, 1), active_class_indices=(0, 1),
        seed=0, member_id=0, task_id=0, device="cpu",
    )
    old_rows = first["prototypes"][:2].clone()
    second_x, second_y = torch.randn(12, 6), torch.tensor([2, 3] * 6)
    second = prepare_task_prototypes(
        model, first, current_features=second_x, current_labels=second_y,
        new_class_indices=(2, 3), active_class_indices=(0, 1, 2, 3),
        seed=0, member_id=0, task_id=1, device="cpu",
    )
    torch.testing.assert_close(second["prototypes"][:2], old_rows, atol=0, rtol=0)
    matrix = second["prototypes"][:4].matmul(second["prototypes"][:4].T)
    torch.testing.assert_close(matrix, torch.eye(4), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.head.weight[:4], second["prototypes"][:4], atol=0, rtol=0)


def test_upcl_loss_is_finite_and_feature_kd_is_only_active_after_task_zero():
    torch.manual_seed(11)
    model = TDDIPaperMember(TDDIPaperMemberConfig(input_dim=6, hidden_dims=(10, 8), num_classes=4, dropout=0.0))
    x, y = torch.randn(10, 6), torch.tensor([0, 1] * 5)
    state = prepare_task_prototypes(
        model, None, current_features=x, current_labels=y,
        new_class_indices=(0, 1), active_class_indices=(0, 1),
        seed=0, member_id=0, task_id=0, device="cpu",
    )
    context0 = UPCLTrainingContext(None, (0, 1), (), class_log_priors(y.numpy(), total_classes=4, active_class_indices=(0, 1)), 0)
    loss0 = compute_upcl_loss(model, x, y, context0)
    assert loss0.feature_distillation.item() == 0
    loss0.total.backward()
    assert model.head.weight.grad is None

    teacher = deepcopy(model).eval().requires_grad_(False)
    x2, y2 = torch.randn(12, 6), torch.tensor([2, 3] * 6)
    state = prepare_task_prototypes(
        model, state, current_features=x2, current_labels=y2,
        new_class_indices=(2, 3), active_class_indices=(0, 1, 2, 3),
        seed=0, member_id=0, task_id=1, device="cpu",
    )
    mixed_x, mixed_y = torch.cat((x2, x)), torch.cat((y2, y))
    context1 = UPCLTrainingContext(teacher, (0, 1, 2, 3), (0, 1),
        class_log_priors(mixed_y.numpy(), total_classes=4, active_class_indices=(0, 1, 2, 3)), 1)
    loss1 = compute_upcl_loss(model, mixed_x, mixed_y, context1)
    assert all(torch.isfinite(value) for value in (loss1.prototype, loss1.feature_contrastive,
                                                   loss1.feature_distillation, loss1.total))
    assert loss1.feature_distillation.item() >= 0


def test_full_memory_sampler_uses_each_current_and_replay_row_once_and_resumes():
    kwargs = dict(current_sample_ids=("c2", "c1"), replay_sample_ids=("r3", "r1", "r2"),
                  replay_raw_labels=np.asarray([4, 3, 3], dtype=np.int64),
                  experiment_seed=0, member_id=0, task_id=1)
    sampler = FoldFullMemorySampler(**kwargs)
    assert sampler.metadata["policy"] == FULL_MEMORY_SAMPLER_POLICY
    assert sorted(iter(sampler)) == list(range(5))
    audit = sampler.last_audit
    assert audit["current_draws"] == 2 and audit["actual_replay_draws"] == 3
    assert audit["unique_replay_exemplars"] == 3 and audit["max_repeat"] == 1
    restored = FoldFullMemorySampler.from_state_dict(sampler.state_dict(), **kwargs)
    assert list(restored) == list(sampler)


def test_ddinet_upcl_trains_all_65_rows_including_singleton_tail():
    from src.training import train_cil
    from src.models.ddinet_member import DDINET_SINGLETON_BN_POLICY
    from torch.utils.data import DataLoader, TensorDataset

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = DDINetMember(DDINetMemberConfig(
            input_dim=8, hidden_dims=(10, 9, 8, 8), num_classes=4, dropout=0.0,
            singleton_batchnorm_policy=DDINET_SINGLETON_BN_POLICY,
        ))
        features = torch.randn(65, 8)
        labels = torch.arange(65) % 2
        prepare_task_prototypes(
            model, None, current_features=features, current_labels=labels,
            new_class_indices=(0, 1), active_class_indices=(0, 1),
            seed=0, member_id=0, task_id=0, device="cpu",
        )
        context = UPCLTrainingContext(None, (0, 1), (), class_log_priors(
            labels.numpy(), total_classes=4, active_class_indices=(0, 1),
        ), 0)
        loader = DataLoader(TensorDataset(features, labels), batch_size=64, drop_last=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = model.backbone[0].weight.detach().clone()
        result = train_cil.train_one_epoch(
            model, loader, optimizer, torch.nn.CrossEntropyLoss(), "cpu",
            gradient_accumulation_steps=16, loss_variant="upcl", upcl_context=context,
            return_loss_components=True,
        )
        assert result.examples_seen == 65
        assert result.optimizer_steps == 1
        assert np.isfinite(result.total_loss)
        assert not torch.equal(before, model.backbone[0].weight)
        assert all(int(layer.num_batches_tracked) == 1 for layer in model.modules()
                   if isinstance(layer, torch.nn.BatchNorm1d))
    finally:
        torch.set_num_threads(previous_threads)
