"""Paper-size numerical-only T-DDI member with an expandable classifier head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    nn = None


TDDI_PAPER_INPUT_DIM = 3780
TDDI_PAPER_HIDDEN_DIMS = (7560, 7560)
ActivationName = Literal["relu", "gelu"]


@dataclass(frozen=True)
class TDDIPaperMemberConfig:
    """Immutable architecture metadata for one numerical-only paper-size member."""

    num_classes: int
    input_dim: int = TDDI_PAPER_INPUT_DIM
    hidden_dims: tuple[int, int] = TDDI_PAPER_HIDDEN_DIMS
    dropout: float = 0.2
    activation: ActivationName = "gelu"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("T-DDI paper member input_dim must be positive.")
        if len(self.hidden_dims) != 2 or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("T-DDI paper member requires exactly two positive hidden widths.")
        if self.num_classes <= 0:
            raise ValueError("T-DDI paper member num_classes must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("T-DDI paper member dropout must be in [0, 1).")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError(f"Unsupported T-DDI activation: {self.activation}")

    def manifest(self) -> dict[str, object]:
        """Return JSON-safe architecture metadata independent of current head width."""

        return {
            "variant": "tddi_paper_member",
            "architecture": "TDDIPaperMember",
            "input_dim": self.input_dim,
            "input_normalization": "layernorm",
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "dropout": self.dropout,
            "classifier": "expandable_linear_head",
        }


def paper_member_manifest(
    *,
    dropout: float,
    activation: ActivationName,
) -> dict[str, object]:
    """Build canonical production metadata without allocating the large model."""

    return TDDIPaperMemberConfig(
        num_classes=1,
        dropout=dropout,
        activation=activation,
    ).manifest()


if nn is not None:

    class TDDIPaperMember(nn.Module):
        """LayerNorm(input) -> 7560 -> 7560 -> expandable class head."""

        def __init__(self, config: TDDIPaperMemberConfig) -> None:
            super().__init__()
            self.config = config
            first_width, latent_width = config.hidden_dims
            self.backbone = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, first_width),
                _build_activation(config.activation),
                nn.Dropout(config.dropout),
                nn.Linear(first_width, latent_width),
                _build_activation(config.activation),
                nn.Dropout(config.dropout),
            )
            self.head = nn.Linear(latent_width, config.num_classes)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Map QSAR descriptors to the final 7,560-dimensional latent vector."""

            return self.backbone(x)

        def classify(self, latent: torch.Tensor) -> torch.Tensor:
            """Map latent vectors to logits for classes seen so far."""

            return self.head(latent)

        def forward_with_latent(
            self,
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            latent = self.encode(x)
            return self.classify(latent), latent

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classify(self.encode(x))


    def _build_activation(name: ActivationName) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported T-DDI activation: {name}")


else:

    class TDDIPaperMember:  # pragma: no cover - runtime guard without torch
        def __init__(self, config: TDDIPaperMemberConfig) -> None:
            raise ImportError("torch is required to instantiate TDDIPaperMember.")
