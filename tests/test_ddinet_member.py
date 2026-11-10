import torch
import pytest
from torch import nn

from src.models.ddinet_member import (
    DDINET_FIXED_CLASS_COUNT,
    DDINET_SINGLETON_BN_POLICY,
    DDINetMember,
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


def test_singleton_policy_keeps_regular_batch_results_and_gradients_identical():
    config = dict(input_dim=8, hidden_dims=(16, 12, 8, 4), dropout=0.0)
    strict = DDINetMember(DDINetMemberConfig(**config))
    safe = DDINetMember(DDINetMemberConfig(
        **config, singleton_batchnorm_policy=DDINET_SINGLETON_BN_POLICY,
    ))
    safe.load_state_dict(strict.state_dict(), strict=True)
    features = torch.randn(4, 8)
    left, right = strict(features), safe(features)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.sum().backward()
    right.sum().backward()
    for (name, parameter), (_, other) in zip(strict.named_parameters(), safe.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    for key, value in strict.state_dict().items():
        torch.testing.assert_close(value, safe.state_dict()[key], rtol=0, atol=0)
    assert "singleton_batchnorm_policy" not in strict.config.manifest()
    assert safe.config.manifest()["singleton_batchnorm_policy"] == DDINET_SINGLETON_BN_POLICY


@pytest.mark.parametrize("warmup", [False, True])
def test_singleton_uses_running_stats_and_keeps_batchnorm_gradients(warmup):
    config = DDINetMemberConfig(input_dim=8, hidden_dims=(16, 12, 8, 4), dropout=0.0,
                               singleton_batchnorm_policy=DDINET_SINGLETON_BN_POLICY)
    model = DDINetMember(config)
    if warmup:
        model(torch.randn(4, 8))
    saved_buffers = {name: value.clone() for name, value in model.named_buffers()}
    features = torch.randn(1, 8, requires_grad=True)
    result = model(features)
    assert torch.isfinite(result).all()
    model.eval()
    torch.testing.assert_close(result, model(features), rtol=0, atol=0)
    result.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm1d):
            assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
            assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, saved_buffers[name], rtol=0, atol=0)


def test_default_ddinet_preserves_strict_batchnorm_behavior():
    model = DDINetMember(DDINetMemberConfig(input_dim=8, hidden_dims=(16, 12, 8, 4)))
    with pytest.raises(ValueError, match="more than 1 value"):
        model(torch.randn(1, 8))


@pytest.mark.parametrize("method", ["ewc", "agem", "er", "ider", "upcl", "task_arithmetic", "ties", "opcm", "magmax"])
def test_singleton_policy_is_scoped_to_ddinet_upcl_and_merging(method):
    from types import SimpleNamespace
    from src.training import train_cil
    for variant in ("tddi_paper_member", "ddinet_multiclass_cil"):
        manifest = train_cil.model_manifest(SimpleNamespace(
            variant=variant, method=method, dropout=.3, activation="relu",
        ))
        if variant == "ddinet_multiclass_cil" and method in {"upcl", "task_arithmetic", "ties", "opcm", "magmax"}:
            assert manifest["singleton_batchnorm_policy"] == DDINET_SINGLETON_BN_POLICY
        else:
            assert "singleton_batchnorm_policy" not in manifest


@pytest.mark.parametrize("method", ["opcm", "magmax"])
def test_sequential_merging_factory_handles_singleton_ddinet_batch(method):
    from src.training import train_cil

    model = train_cil.expand_model_for_seen_classes(
        None, None, {raw: raw for raw in range(178)},
        variant="ddinet_multiclass_cil", input_dim=2048,
        dropout=0.3, activation="relu", norm="none", method=method,
    )
    assert model.config.hidden_dims[-1] == 128
    assert model.config.singleton_batchnorm_policy == DDINET_SINGLETON_BN_POLICY
    logits = model(torch.randn(1, 2048))
    assert logits.shape == (1, 178)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert model.classifier_head.weight.grad is not None
