import torch
from torch import nn

from src.models.ddinet_member import (
    DDINET_FIXED_CLASS_COUNT,
    DDINetMemberConfig,
    build_fixed_head_ddinet_member,
)
from src.training.class_incremental_policy import ClassIncrementalPolicy


def test_ddinet_has_five_linear_layers_and_fixed_head_policy():
    torch.manual_seed(11)
    model = build_fixed_head_ddinet_member(
        input_dim=8,
        hidden_dims=(16, 12, 8, 4),
        dropout=0.0,
    )
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    assert len(linear_layers) == 5
    assert model.classifier_head is model.head
    assert model.head.out_features == DDINET_FIXED_CLASS_COUNT

    model.eval()
    features = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    )
    logits, latent = model.forward_with_latent(features)
    assert logits.shape == (2, 178)
    assert latent.shape == (2, 4)

    policy = ClassIncrementalPolicy.from_raw_class_order(
        list(range(178)),
        active_raw_classes=[0, 1],
    )
    policy.validate_linear_head(model.classifier_head)
    masked = policy.mask_logits(logits)
    assert torch.isfinite(masked[:, :2]).all()
    assert torch.isneginf(masked[:, 2:]).all()


def test_ddinet_manifest_separates_paper_facts_from_adaptations():
    manifest = DDINetMemberConfig().manifest()
    assert manifest["input"]["input_dim"] == 2048
    assert manifest["classifier"] == "fixed_linear_head_178"
    assert manifest["fully_connected_layer_count_including_head"] == 5
    assert manifest["task_adaptation"] == "multiclass_only_no_binary_auxiliary_head"
    assert "hidden_layer_widths" in manifest["fidelity"]["provisional"]
