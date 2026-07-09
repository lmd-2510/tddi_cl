"""Descriptor-only MLP baselines for DDI2025-CIL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    nn = None


ActivationName = Literal["relu", "gelu"]
NormName = Literal["none", "layernorm", "batchnorm"]


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dims: tuple[int, ...]
    num_classes: int
    dropout: float = 0.2
    activation: ActivationName = "gelu"
    norm: NormName = "layernorm"


def preset_config(
    variant: Literal["small", "base", "large"],
    *,
    input_dim: int,
    num_classes: int,
    dropout: float = 0.2,
    activation: ActivationName = "gelu",
    norm: NormName = "layernorm",
) -> MLPConfig:
    hidden_map = {
        "small": (512, 256),
        "base": (1024, 512),
        "large": (2048, 1024, 512),
    }
    return MLPConfig(
        input_dim=input_dim,
        hidden_dims=hidden_map[variant],
        num_classes=num_classes,
        dropout=dropout,
        activation=activation,
        norm=norm,
    )


if nn is not None:

    class MLP(nn.Module):
        """Feed-forward MLP for descriptor-only DDI classification."""

        def __init__(self, config: MLPConfig) -> None:
            super().__init__()
            self.config = config
            layers: list[nn.Module] = []
            in_dim = config.input_dim

            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(_build_norm(config.norm, hidden_dim))
                layers.append(_build_activation(config.activation))
                layers.append(nn.Dropout(config.dropout))
                in_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(in_dim, config.num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.backbone(x))


    def _build_activation(name: ActivationName) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation: {name}")


    def _build_norm(name: NormName, hidden_dim: int) -> nn.Module:
        if name == "none":
            return nn.Identity()
        if name == "layernorm":
            return nn.LayerNorm(hidden_dim)
        if name == "batchnorm":
            return nn.BatchNorm1d(hidden_dim)
        raise ValueError(f"Unsupported norm: {name}")


else:

    class MLP:  # pragma: no cover - runtime guard for environments without torch
        """Import-time placeholder when torch is unavailable."""

        def __init__(self, config: MLPConfig) -> None:
            raise ImportError(
                "torch is required to instantiate MLP. "
                "Install torch before using src/models/mlp.py."
            )
