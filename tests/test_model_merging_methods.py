import pytest
import torch
from torch import nn

from src.methods.model_merging import (
    BATCHNORM_POLICY,
    HEAD_POLICY,
    ModelMergingRecipe,
    inspect_neural_merge_scope,
    merge_task_boundary,
    task_arithmetic_state,
    ties_state,
)
from src.models.ddinet_member import build_fixed_head_ddinet_member
from src.models.mfp_siamese_member import build_fixed_head_mfp_siamese_member
from src.models.tddi_paper_member import build_fixed_head_tddi_member
from src.methods.task_arithmetic import merge_task_arithmetic
from src.methods.ties import merge_ties


class TinyMergeModel(nn.Module):
    def __init__(self, *, batchnorm: bool = False):
        super().__init__()
        layers = [nn.Linear(2, 2, bias=False)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(2))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(2, 6)

    def forward(self, values):
        return self.head(self.backbone(values))

    @property
    def classifier_head(self):
        return self.head


def _state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _expert(base, *, backbone_delta, rows):
    expert = {key: value.clone() for key, value in base.items()}
    expert["backbone.0.weight"].add_(torch.as_tensor(backbone_delta))
    for row, value in rows.items():
        expert["head.weight"][row].fill_(value)
        expert["head.bias"][row] = value + 0.5
    return expert


def test_paper_recipes_are_locked_without_merger_validation_tuning():
    arithmetic = ModelMergingRecipe.paper_task_arithmetic()
    ties = ModelMergingRecipe.paper_ties()
    assert arithmetic.manifest() == {
        "method": "task_arithmetic",
        "scale": 0.4,
        "density": None,
        "sign_election": None,
        "reduction": None,
        "source": "paper_fixed_recipe_no_merger_validation_tuning",
    }
    assert ties.scale == 1.0
    assert ties.density == 0.2
    assert ties.sign_election == "mass"
    assert ties.reduction == "disjoint_mean"
    base = {"weight": torch.zeros(5)}
    expert = {"weight": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])}
    torch.testing.assert_close(
        merge_task_arithmetic(base, [expert], parameter_keys=["weight"])["weight"],
        0.4 * expert["weight"],
    )
    torch.testing.assert_close(
        merge_ties(base, [expert], parameter_keys=["weight"])["weight"],
        torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_task_arithmetic_merges_backbone_and_stitches_owned_head_rows():
    model = TinyMergeModel()
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
    base = _state(model)
    expert_1 = _expert(base, backbone_delta=[[1.0, 2.0], [3.0, 4.0]], rows={2: 12.0, 3: 13.0})
    expert_2 = _expert(base, backbone_delta=[[2.0, 1.0], [-1.0, 3.0]], rows={4: 24.0, 5: 25.0})
    result = merge_task_boundary(
        model,
        base_state=base,
        experts_by_task={1: expert_1, 2: expert_2},
        rows_by_task={0: [0, 1], 1: [2, 3], 2: [4, 5]},
        train_sample_counts_by_task={0: 20, 1: 10, 2: 5},
        upto_task_id=2,
        recipe=ModelMergingRecipe.paper_task_arithmetic(),
    )
    expected = 0.4 * torch.tensor([[3.0, 3.0], [2.0, 7.0]])
    torch.testing.assert_close(result.state_dict["backbone.0.weight"], expected)
    torch.testing.assert_close(result.state_dict["head.weight"][:2], base["head.weight"][:2])
    assert torch.equal(result.state_dict["head.weight"][2], expert_1["head.weight"][2])
    assert torch.equal(result.state_dict["head.weight"][5], expert_2["head.weight"][5])
    assert result.audit["future_experts_used"] is False
    assert result.audit["head_policy"] == HEAD_POLICY


def test_ties_trim_elect_mass_and_disjoint_mean():
    base = {
        "weight": torch.zeros(5),
        "untouched": torch.tensor([7.0]),
    }
    expert_1 = {
        "weight": torch.tensor([10.0, 4.0, -3.0, 2.0, 1.0]),
        "untouched": base["untouched"].clone(),
    }
    expert_2 = {
        "weight": torch.tensor([-6.0, 5.0, -2.0, -4.0, 0.5]),
        "untouched": base["untouched"].clone(),
    }
    merged = ties_state(base, [expert_1, expert_2], keys=["weight"], density=1.0, scale=1.0)
    # Mass elects + at indices 0/1, - at 2/3, and + at 4.
    torch.testing.assert_close(merged["weight"], torch.tensor([10.0, 4.5, -2.5, -4.0, 0.75]))
    torch.testing.assert_close(merged["untouched"], base["untouched"])

    trimmed = ties_state(base, [expert_1, expert_2], keys=["weight"])
    # Paper density 20% retains exactly one largest-magnitude coordinate per expert.
    torch.testing.assert_close(trimmed["weight"], torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]))


def test_batchnorm_moments_use_weighted_total_variance_without_samples():
    model = TinyMergeModel(batchnorm=True)
    base = _state(model)
    expert = _state(model)
    base["backbone.1.running_mean"] = torch.tensor([0.0, 2.0])
    base["backbone.1.running_var"] = torch.tensor([1.0, 4.0])
    base["backbone.1.num_batches_tracked"] = torch.tensor(3)
    expert["backbone.1.running_mean"] = torch.tensor([4.0, 6.0])
    expert["backbone.1.running_var"] = torch.tensor([9.0, 16.0])
    expert["backbone.1.num_batches_tracked"] = torch.tensor(2)
    result = merge_task_boundary(
        model,
        base_state=base,
        experts_by_task={1: expert},
        rows_by_task={0: [0, 1, 2], 1: [3, 4, 5]},
        train_sample_counts_by_task={0: 3, 1: 1},
        upto_task_id=1,
        recipe=ModelMergingRecipe.paper_task_arithmetic(),
    )
    mean = torch.tensor([1.0, 3.0])
    second = 0.75 * (base["backbone.1.running_var"] + base["backbone.1.running_mean"].square())
    second += 0.25 * (expert["backbone.1.running_var"] + expert["backbone.1.running_mean"].square())
    torch.testing.assert_close(result.state_dict["backbone.1.running_mean"], mean)
    torch.testing.assert_close(result.state_dict["backbone.1.running_var"], second - mean.square())
    assert int(result.state_dict["backbone.1.num_batches_tracked"]) == 5
    assert result.audit["batchnorm_policy"] == BATCHNORM_POLICY
    assert result.audit["batchnorm_uses_stored_samples"] is False


def test_three_backbones_have_expected_neural_only_merge_scopes():
    tddi = build_fixed_head_tddi_member(input_dim=4, hidden_dims=(8, 6), dropout=0.0)
    ddinet = build_fixed_head_ddinet_member(input_dim=8, hidden_dims=(7, 6, 5, 4), dropout=0.0)
    mfp = build_fixed_head_mfp_siamese_member(
        drug_input_dim=4,
        encoder_hidden_dim=6,
        drug_latent_dim=3,
        classifier_hidden_dim=5,
        dropout=0.0,
    )
    tddi_scope = inspect_neural_merge_scope(tddi)
    ddinet_scope = inspect_neural_merge_scope(ddinet)
    mfp_scope = inspect_neural_merge_scope(mfp)
    assert tddi_scope.head_weight_key == "head.weight"
    assert not tddi_scope.batchnorm
    assert len(ddinet_scope.batchnorm) == 4
    assert len(mfp_scope.batchnorm) == 3
    for scope in (tddi_scope, ddinet_scope, mfp_scope):
        assert "head.weight" not in scope.backbone_parameter_keys
        assert "head.bias" not in scope.backbone_parameter_keys
        assert all("running_" not in key for key in scope.backbone_parameter_keys)


def test_boundary_rejects_future_or_missing_experts_and_overlapping_head_rows():
    model = TinyMergeModel()
    base = _state(model)
    expert = _state(model)
    kwargs = dict(
        model=model,
        base_state=base,
        rows_by_task={0: [0, 1], 1: [1, 2]},
        train_sample_counts_by_task={0: 1, 1: 1},
        upto_task_id=1,
        recipe=ModelMergingRecipe.paper_task_arithmetic(),
    )
    with pytest.raises(ValueError, match="exactly the already-seen experts"):
        merge_task_boundary(experts_by_task={1: expert, 2: expert}, **kwargs)
    with pytest.raises(ValueError, match="multiple owners"):
        merge_task_boundary(experts_by_task={1: expert}, **kwargs)

    isolated_rows = {0: [0, 1, 2], 1: [3, 4, 5]}
    expert["head.weight"][0].add_(1)
    with pytest.raises(ValueError, match="non-owned head weight"):
        merge_task_boundary(
            model=model,
            base_state=base,
            experts_by_task={1: expert},
            rows_by_task=isolated_rows,
            train_sample_counts_by_task={0: 1, 1: 1},
            upto_task_id=1,
            recipe=ModelMergingRecipe.paper_task_arithmetic(),
        )


def test_arithmetic_and_ties_reject_nonfloating_or_incompatible_states():
    base = {"weight": torch.zeros(2), "counter": torch.tensor(0)}
    expert = {"weight": torch.ones(2), "counter": torch.tensor(1)}
    with pytest.raises(TypeError, match="floating-point"):
        task_arithmetic_state(base, [expert], keys=["counter"])
    with pytest.raises(ValueError, match="tensor contract mismatch"):
        ties_state(base, [{"weight": torch.ones(3), "counter": torch.tensor(1)}], keys=["weight"])
