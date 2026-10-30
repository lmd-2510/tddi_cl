import numpy as np
import torch
from torch import nn

from src.data.morgan_fingerprints import (
    MorganFingerprintConfig,
    build_morgan_fingerprint_cache,
)
from src.models.mfp_siamese_member import (
    MFPSiameseMemberConfig,
    build_fixed_head_mfp_siamese_member,
)
from src.training.class_incremental_policy import ClassIncrementalPolicy


def test_mfp_siamese_uses_separate_2048_bit_cache_namespace(tmp_path):
    config = MorganFingerprintConfig(
        radius=2,
        n_bits=2048,
        include_chirality=False,
        consumer="mfp_siamese",
    )
    cache = build_morgan_fingerprint_cache(
        tmp_path / "mfp_siamese",
        {"a": "CCO", "b": "CCN"},
        config=config,
    )
    assert cache.matrix.shape == (2, 2048)
    assert cache.input_dim == 4096
    assert cache.manifest["config"]["consumer"] == "mfp_siamese"
    assert cache.ordered_pair("a", "b").dtype == np.float32


def test_shared_encoder_model_uses_fixed_head_policy_and_preserves_order():
    torch.manual_seed(19)
    model = build_fixed_head_mfp_siamese_member(
        drug_input_dim=4,
        encoder_hidden_dim=8,
        drug_latent_dim=3,
        classifier_hidden_dim=6,
        dropout=0.0,
    ).eval()
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    assert len(linear_layers) == 4
    assert model.head.out_features == 178

    drug_a = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    drug_b = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
    pair = torch.cat((drug_a, drug_b), dim=1)
    with torch.no_grad():
        separate_logits = model(drug_a, drug_b)
        concatenated_logits = model(pair)
        reversed_logits = model(drug_b, drug_a)
    torch.testing.assert_close(separate_logits, concatenated_logits)
    assert not torch.equal(separate_logits, reversed_logits)

    policy = ClassIncrementalPolicy.from_raw_class_order(
        list(range(178)),
        active_raw_classes=[0, 1, 2],
    )
    policy.validate_linear_head(model.classifier_head)
    masked = policy.mask_logits(separate_logits)
    assert torch.isfinite(masked[:, :3]).all()
    assert torch.isneginf(masked[:, 3:]).all()


def test_manifest_matches_paper_architecture_and_macro_f1_adaptation():
    manifest = MFPSiameseMemberConfig().manifest()
    assert manifest["representation"]["bits_per_drug"] == 2048
    assert manifest["shared_encoder"]["dimensions"] == [2048, 512, 256]
    assert manifest["classifier"]["dimensions"] == [512, 512, 178]
    assert manifest["paper_training_defaults"]["loss"] == "cross_entropy"
    assert manifest["cil_adaptation"]["checkpoint_selection"] == "validation_macro_f1"
