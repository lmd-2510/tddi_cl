import torch

from src.models.tddi_paper_member import (
    TDDI_FIXED_CLASS_COUNT,
    build_fixed_head_tddi_member,
)
from src.training.class_incremental_policy import ClassIncrementalPolicy


def test_tddi_native_fixed_head_uses_one_shared_cil_policy():
    torch.manual_seed(5)
    model = build_fixed_head_tddi_member(
        input_dim=3,
        hidden_dims=(6, 4),
        dropout=0.0,
        activation="relu",
    )
    policy = ClassIncrementalPolicy.from_raw_class_order(
        list(range(TDDI_FIXED_CLASS_COUNT)),
        active_raw_classes=[0, 1],
    )
    policy.validate_linear_head(model.classifier_head)
    assert model.head.out_features == TDDI_FIXED_CLASS_COUNT

    features = torch.tensor([[1.0, 0.0, -1.0], [0.5, 1.0, 0.0]])
    labels = policy.global_labels([0, 1])
    logits = model(features)
    masked = policy.mask_logits(logits)
    assert torch.isfinite(masked[:, :2]).all()
    assert torch.isneginf(masked[:, 2:]).all()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.2)
    inactive_weight = model.head.weight[2:].detach().clone()
    inactive_bias = model.head.bias[2:].detach().clone()
    active_weight = model.head.weight[:2].detach().clone()
    policy.classification_loss(logits, labels).backward()
    policy.optimizer_step(optimizer, model.classifier_head)
    optimizer.zero_grad(set_to_none=True)

    torch.testing.assert_close(model.head.weight[2:], inactive_weight, rtol=0, atol=0)
    torch.testing.assert_close(model.head.bias[2:], inactive_bias, rtol=0, atol=0)
    assert not torch.equal(model.head.weight[:2], active_weight)

    policy.activate([2])
    newly_opened = policy.mask_logits(model(features))
    assert torch.isfinite(newly_opened[:, 2]).all()
    assert torch.isneginf(newly_opened[:, 3:]).all()


def test_tddi_fixed_head_manifest_is_explicit():
    model = build_fixed_head_tddi_member(
        input_dim=2,
        hidden_dims=(4, 3),
        dropout=0.0,
    )
    manifest = model.config.manifest()
    assert manifest["classifier"] == "fixed_linear_head_178"
    assert manifest["num_classes"] == 178
