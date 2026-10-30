"""Multiclass DDINet adaptation with the shared 178-class CIL head policy."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    nn = None


DDINET_FINGERPRINT_DIM = 1024
DDINET_PAIR_INPUT_DIM = 2 * DDINET_FINGERPRINT_DIM
DDINET_FIXED_CLASS_COUNT = 178
DDINET_DEFAULT_HIDDEN_DIMS = (1024, 512, 256, 128)
DDINET_RECOMMENDED_WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class DDINetMemberConfig:
    """Architecture contract for the provisional multiclass DDINet adapter.

    The paper confirms five fully-connected layers, ReLU, BatchNorm and 0.3
    dropout.  Publicly accessible material does not expose layer widths, so the
    four hidden widths below are explicit adaptation defaults; the fifth linear
    layer is the native 178-way classifier.
    """

    num_classes: int = DDINET_FIXED_CLASS_COUNT
    input_dim: int = DDINET_PAIR_INPUT_DIM
    hidden_dims: tuple[int, int, int, int] = DDINET_DEFAULT_HIDDEN_DIMS
    dropout: float = 0.3
    recommended_weight_decay: float = DDINET_RECOMMENDED_WEIGHT_DECAY

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("DDINet input_dim must be positive.")
        if len(self.hidden_dims) != 4 or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("DDINet requires exactly four positive hidden widths.")
        if self.num_classes <= 0:
            raise ValueError("DDINet num_classes must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("DDINet dropout must be in [0, 1).")
        if self.recommended_weight_decay < 0:
            raise ValueError("DDINet recommended_weight_decay must be non-negative.")

    def manifest(self) -> dict[str, object]:
        return {
            "variant": "ddinet_multiclass_cil",
            "architecture": "DDINetMember",
            "paper_doi": "10.1016/j.knosys.2025.114981",
            "input": {
                "representation": "ordered_concatenated_morgan_fingerprints",
                "per_drug_dim": self.input_dim // 2,
                "input_dim": self.input_dim,
                "external_scaler": "none",
            },
            "hidden_dims": list(self.hidden_dims),
            "activation": "relu",
            "batch_normalization": True,
            "dropout": self.dropout,
            "fully_connected_layer_count_including_head": 5,
            "recommended_weight_decay": self.recommended_weight_decay,
            "classifier": (
                "fixed_linear_head_178"
                if self.num_classes == DDINET_FIXED_CLASS_COUNT
                else "linear_head"
            ),
            "num_classes": self.num_classes,
            "task_adaptation": "multiclass_only_no_binary_auxiliary_head",
            "fidelity": {
                "paper_confirmed": [
                    "morgan_1024_per_drug",
                    "five_fully_connected_layers",
                    "relu",
                    "batch_normalization",
                    "dropout_0.3",
                    "l2_1e-4",
                ],
                "provisional": [
                    "ordered_concatenation",
                    "hidden_layer_widths",
                    "morgan_radius_2",
                    "morgan_include_chirality_false",
                ],
            },
        }


if nn is not None:

    class DDINetMember(nn.Module):
        """2048-bit drug pair -> four regularized blocks -> fixed class head."""

        def __init__(self, config: DDINetMemberConfig = DDINetMemberConfig()) -> None:
            super().__init__()
            self.config = config
            blocks: list[nn.Module] = []
            previous_width = config.input_dim
            for width in config.hidden_dims:
                blocks.extend(
                    (
                        nn.Linear(previous_width, width),
                        nn.BatchNorm1d(width),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                    )
                )
                previous_width = width
            self.backbone = nn.Sequential(*blocks)
            self.head = nn.Linear(previous_width, config.num_classes)

        def encode(self, features: torch.Tensor) -> torch.Tensor:
            if features.ndim != 2 or features.shape[1] != self.config.input_dim:
                raise ValueError(
                    f"DDINet expects [batch, {self.config.input_dim}] fingerprints."
                )
            return self.backbone(features)

        def classify(self, latent: torch.Tensor) -> torch.Tensor:
            return self.head(latent)

        def forward_with_latent(
            self,
            features: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            latent = self.encode(features)
            return self.classify(latent), latent

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.classify(self.encode(features))

        @property
        def classifier_head(self) -> nn.Linear:
            """Expose the native linear layer to ClassIncrementalPolicy."""

            return self.head


    def build_fixed_head_ddinet_member(
        *,
        input_dim: int = DDINET_PAIR_INPUT_DIM,
        hidden_dims: tuple[int, int, int, int] = DDINET_DEFAULT_HIDDEN_DIMS,
        dropout: float = 0.3,
    ) -> DDINetMember:
        return DDINetMember(
            DDINetMemberConfig(
                num_classes=DDINET_FIXED_CLASS_COUNT,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )


else:

    class DDINetMember:  # pragma: no cover - runtime guard without torch
        def __init__(self, config: DDINetMemberConfig = DDINetMemberConfig()) -> None:
            raise ImportError("torch is required to instantiate DDINetMember.")


    def build_fixed_head_ddinet_member(**kwargs):  # pragma: no cover
        raise ImportError("torch is required to instantiate DDINetMember.")
