from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.models.tddi_paper_member import (
    TDDI_PAPER_HIDDEN_DIMS,
    TDDI_PAPER_INPUT_DIM,
    TDDIPaperMember,
    TDDIPaperMemberConfig,
    paper_member_manifest,
)
from src.training.train_cil import (
    expand_model_for_seen_classes,
    parse_args,
    write_run_config,
)


def _expand(
    class_map: dict[int, int],
    *,
    previous_model: nn.Module | None = None,
    previous_map: dict[int, int] | None = None,
) -> nn.Module:
    return expand_model_for_seen_classes(
        previous_model=previous_model,
        previous_seen_map=previous_map,
        current_seen_map=class_map,
        variant="tddi_paper_member",
        input_dim=TDDI_PAPER_INPUT_DIM,
        dropout=0.2,
        activation="gelu",
        norm="layernorm",
    )


def test_paper_member_production_shapes_count_and_input_layernorm() -> None:
    class_count = 38
    with torch.device("meta"):
        model = _expand({raw_class: raw_class for raw_class in range(class_count)})

    assert isinstance(model, TDDIPaperMember)
    assert model.config.hidden_dims == TDDI_PAPER_HIDDEN_DIMS
    assert isinstance(model.backbone[0], nn.LayerNorm)
    assert model.backbone[0].normalized_shape == (TDDI_PAPER_INPUT_DIM,)
    assert isinstance(model.backbone[1], nn.Linear)
    assert model.backbone[1].weight.shape == (7560, 3780)
    assert isinstance(model.backbone[4], nn.Linear)
    assert model.backbone[4].weight.shape == (7560, 7560)
    assert isinstance(model.head, nn.Linear)
    assert model.head.weight.shape == (class_count, 7560)
    assert sum(isinstance(module, nn.LayerNorm) for module in model.modules()) == 1

    expected_parameter_count = (
        2 * 3780
        + (3780 * 7560 + 7560)
        + (7560 * 7560 + 7560)
        + (7560 * class_count + class_count)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameter_count


def test_paper_member_smoke_forward_latent_and_backward() -> None:
    model = TDDIPaperMember(
        TDDIPaperMemberConfig(
            num_classes=3,
            dropout=0.0,
            activation="relu",
        )
    )
    features = torch.randn(1, TDDI_PAPER_INPUT_DIM)

    logits, latent = model.forward_with_latent(features)
    direct_logits = model(features)
    logits.square().mean().backward()

    assert logits.shape == direct_logits.shape == (1, 3)
    assert latent.shape == (1, 7560)
    torch.testing.assert_close(logits, direct_logits)
    assert model.backbone[0].weight.grad is not None
    assert model.backbone[1].weight.grad is not None
    assert model.backbone[4].weight.grad is not None
    assert model.head.weight.grad is not None


def test_paper_member_head_expansion_copies_old_class_rows() -> None:
    old_map = {10: 0, 30: 1}
    expanded_map = {10: 0, 20: 1, 30: 2}
    original_dtype = torch.get_default_dtype()
    try:
        # Expansion only needs exact copy assertions; fp16 keeps this full-size
        # architecture test memory-bounded without changing production defaults.
        torch.set_default_dtype(torch.float16)
        previous = _expand(old_map)
        with torch.no_grad():
            previous.head.weight[0].fill_(0.125)
            previous.head.weight[1].fill_(-0.25)
            previous.head.bias.copy_(torch.tensor([0.5, -0.75]))
        expanded = _expand(
            expanded_map,
            previous_model=previous,
            previous_map=old_map,
        )
    finally:
        torch.set_default_dtype(original_dtype)

    assert expanded.head.out_features == 3
    torch.testing.assert_close(expanded.head.weight[0], previous.head.weight[0], rtol=0, atol=0)
    torch.testing.assert_close(expanded.head.weight[2], previous.head.weight[1], rtol=0, atol=0)
    torch.testing.assert_close(expanded.head.bias[0], previous.head.bias[0], rtol=0, atol=0)
    torch.testing.assert_close(expanded.head.bias[2], previous.head.bias[1], rtol=0, atol=0)


def test_paper_member_manifest_records_derived_architecture(tmp_path: Path) -> None:
    task_file = tmp_path / "tasks.json"
    task_spec = {
        "protocol": "tail_to_head",
        "seed": None,
        "tasks": [{"task_id": 0, "classes": [10, 30]}],
    }
    task_file.write_text(json.dumps(task_spec), encoding="utf-8")
    args = Namespace(
        method="ewc",
        memory_per_class=50,
        variant="tddi_paper_member",
        task_file=task_file,
        experiment_seed=0,
        member_id=0,
        member_seed=123,
        member_seed_derivation="test",
        seed_mode="member",
        dropout=0.15,
        activation="relu",
        graph_cache=None,
    )
    output_path = tmp_path / "run_config.json"

    write_run_config(
        output_path,
        args=args,
        run_id="paper-member-run",
        device="cpu",
        task_spec=task_spec,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["resolved"]["model"] == paper_member_manifest(
        dropout=0.15,
        activation="relu",
    )
    assert payload["resolved"]["model"]["hidden_dims"] == [7560, 7560]
    assert payload["resolved"]["model"]["input_normalization"] == "layernorm"
    assert payload["arguments"]["dropout"] == 0.15
    assert payload["arguments"]["activation"] == "relu"


def test_paper_member_cli_and_fixed_input_contract() -> None:
    required = [
        "--train",
        "train.parquet",
        "--validation",
        "validation.parquet",
        "--test",
        "test.parquet",
        "--feature-cols",
        "features.json",
        "--scaler",
        "scaler.pkl",
        "--task-file",
        "tasks.json",
        "--outdir",
        "run",
        "--variant",
        "tddi_paper_member",
    ]
    with patch("sys.argv", ["train_cil.py", *required]):
        args = parse_args()
    assert args.variant == "tddi_paper_member"

    with pytest.raises(ValueError, match="requires input_dim=3780"):
        expand_model_for_seen_classes(
            None,
            None,
            {10: 0},
            variant="tddi_paper_member",
            input_dim=4,
            dropout=0.2,
            activation="gelu",
            norm="layernorm",
        )
    with pytest.raises(ValueError, match="fixed input LayerNorm"):
        expand_model_for_seen_classes(
            None,
            None,
            {10: 0},
            variant="tddi_paper_member",
            input_dim=TDDI_PAPER_INPUT_DIM,
            dropout=0.2,
            activation="gelu",
            norm="none",
        )
