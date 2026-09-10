"""Small T-DDI instances used to test infrastructure without allocating the paper-size model."""

from __future__ import annotations

import torch

from src.models.tddi_paper_member import TDDIPaperMember, TDDIPaperMemberConfig
from src.training.train_cil import copy_previous_state_to_expanded_model


def tiny_tddi_model(
    class_map: dict[int, int],
    *,
    input_dim: int = 4,
    hidden_dim: int = 8,
) -> TDDIPaperMember:
    return TDDIPaperMember(
        TDDIPaperMemberConfig(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            num_classes=len(class_map),
            dropout=0.0,
            activation="relu",
        )
    )


def expand_tiny_tddi(
    previous_model: torch.nn.Module | None,
    previous_seen_map: dict[int, int] | None,
    current_seen_map: dict[int, int],
    *,
    input_dim: int = 4,
    hidden_dim: int = 8,
    **_: object,
) -> TDDIPaperMember:
    model = tiny_tddi_model(
        current_seen_map,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )
    if previous_model is None or previous_seen_map is None:
        return model
    return copy_previous_state_to_expanded_model(
        previous_model,
        model,
        previous_seen_map,
        current_seen_map,
        variant="tddi_paper_member",
    )
