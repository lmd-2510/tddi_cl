"""Paper-aligned MFP-Siamese backbone with the shared 178-class CIL head."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment may not have torch yet
    torch = None
    nn = None


MFP_SIAMESE_DRUG_DIM = 2048
MFP_SIAMESE_PAIR_DIM = 2 * MFP_SIAMESE_DRUG_DIM
MFP_SIAMESE_ENCODER_HIDDEN_DIM = 512
MFP_SIAMESE_LATENT_DIM = 256
MFP_SIAMESE_CLASSIFIER_HIDDEN_DIM = 512
MFP_SIAMESE_FIXED_CLASS_COUNT = 178
MFP_SIAMESE_PAPER_LR = 1e-4
MFP_SIAMESE_PAPER_BATCH_SIZE = 512
MFP_SIAMESE_PAPER_EPOCHS = 200


@dataclass(frozen=True)
class MFPSiameseMemberConfig:
    """Exact MFP branch architecture adapted only at the CIL output policy."""

    num_classes: int = MFP_SIAMESE_FIXED_CLASS_COUNT
    drug_input_dim: int = MFP_SIAMESE_DRUG_DIM
    encoder_hidden_dim: int = MFP_SIAMESE_ENCODER_HIDDEN_DIM
    drug_latent_dim: int = MFP_SIAMESE_LATENT_DIM
    classifier_hidden_dim: int = MFP_SIAMESE_CLASSIFIER_HIDDEN_DIM
    dropout: float = 0.3

    def __post_init__(self) -> None:
        dimensions = (
            self.num_classes,
            self.drug_input_dim,
            self.encoder_hidden_dim,
            self.drug_latent_dim,
            self.classifier_hidden_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("MFP-Siamese dimensions must all be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("MFP-Siamese dropout must be in [0, 1).")

    @property
    def pair_input_dim(self) -> int:
        return 2 * self.drug_input_dim

    def manifest(self) -> dict[str, object]:
        return {
            "variant": "mfp_siamese_member",
            "display_name": "MFP-Siamese",
            "architecture": "Shared-Encoder Morgan Fingerprint Network",
            "paper_doi": "10.1186/s13321-025-01128-8",
            "representation": {
                "family": "morgan_ecfp4",
                "radius": 2,
                "bits_per_drug": self.drug_input_dim,
                "pair_order": "ordered_a_then_b",
                "external_scaler": "none",
            },
            "shared_encoder": {
                "dimensions": [
                    self.drug_input_dim,
                    self.encoder_hidden_dim,
                    self.drug_latent_dim,
                ],
                "batch_normalization": True,
                "activation": "relu",
                "dropout_after_first_layer": self.dropout,
            },
            "classifier": {
                "dimensions": [
                    2 * self.drug_latent_dim,
                    self.classifier_hidden_dim,
                    self.num_classes,
                ],
                "batch_normalization": True,
                "activation": "relu",
                "dropout": self.dropout,
                "head": (
                    "fixed_linear_head_178"
                    if self.num_classes == MFP_SIAMESE_FIXED_CLASS_COUNT
                    else "linear_head"
                ),
            },
            "paper_training_defaults": {
                "loss": "cross_entropy",
                "optimizer": "adam",
                "learning_rate": MFP_SIAMESE_PAPER_LR,
                "batch_size": MFP_SIAMESE_PAPER_BATCH_SIZE,
                "epochs": MFP_SIAMESE_PAPER_EPOCHS,
            },
            "cil_adaptation": {
                "global_class_count": self.num_classes,
                "head_policy": "shared_fixed_head_seen_class_mask",
                "checkpoint_selection": "validation_macro_f1",
            },
        }


if nn is not None:

    class MFPSiameseMember(nn.Module):
        """Encode A/B with shared weights, then classify the ordered pair."""

        def __init__(
            self,
            config: MFPSiameseMemberConfig = MFPSiameseMemberConfig(),
        ) -> None:
            super().__init__()
            self.config = config
            self.encoder = nn.Sequential(
                nn.Linear(config.drug_input_dim, config.encoder_hidden_dim),
                nn.BatchNorm1d(config.encoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.encoder_hidden_dim, config.drug_latent_dim),
                nn.BatchNorm1d(config.drug_latent_dim),
                nn.ReLU(),
            )
            self.pair_backbone = nn.Sequential(
                nn.Linear(2 * config.drug_latent_dim, config.classifier_hidden_dim),
                nn.BatchNorm1d(config.classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.head = nn.Linear(config.classifier_hidden_dim, config.num_classes)

        def _split_pair(
            self,
            fingerprint_a: torch.Tensor,
            fingerprint_b: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if fingerprint_b is None:
                if (
                    fingerprint_a.ndim != 2
                    or fingerprint_a.shape[1] != self.config.pair_input_dim
                ):
                    raise ValueError(
                        "MFP-Siamese expects a concatenated tensor shaped "
                        f"[batch, {self.config.pair_input_dim}]."
                    )
                return fingerprint_a.split(self.config.drug_input_dim, dim=1)
            if (
                fingerprint_a.ndim != 2
                or fingerprint_a.shape[1] != self.config.drug_input_dim
            ):
                raise ValueError(
                    f"Drug A fingerprint must have shape [batch, {self.config.drug_input_dim}]."
                )
            if (
                fingerprint_b.ndim != 2
                or fingerprint_b.shape[1] != self.config.drug_input_dim
                or fingerprint_b.shape[0] != fingerprint_a.shape[0]
            ):
                raise ValueError(
                    f"Drug B fingerprint must have shape [batch, {self.config.drug_input_dim}]."
                )
            return fingerprint_a, fingerprint_b

        def encode_drug(self, fingerprint: torch.Tensor) -> torch.Tensor:
            return self.encoder(fingerprint)

        def encode(
            self,
            fingerprint_a: torch.Tensor,
            fingerprint_b: torch.Tensor | None = None,
        ) -> torch.Tensor:
            fingerprint_a, fingerprint_b = self._split_pair(
                fingerprint_a,
                fingerprint_b,
            )
            latent_a = self.encode_drug(fingerprint_a)
            latent_b = self.encode_drug(fingerprint_b)
            return self.pair_backbone(torch.cat((latent_a, latent_b), dim=1))

        def classify(self, pair_latent: torch.Tensor) -> torch.Tensor:
            return self.head(pair_latent)

        def forward_with_latent(
            self,
            fingerprint_a: torch.Tensor,
            fingerprint_b: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            pair_latent = self.encode(fingerprint_a, fingerprint_b)
            return self.classify(pair_latent), pair_latent

        def forward(
            self,
            fingerprint_a: torch.Tensor,
            fingerprint_b: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self.classify(self.encode(fingerprint_a, fingerprint_b))

        @property
        def classifier_head(self) -> nn.Linear:
            """Expose the native head to the shared class-incremental policy."""

            return self.head


    def build_fixed_head_mfp_siamese_member(
        *,
        drug_input_dim: int = MFP_SIAMESE_DRUG_DIM,
        encoder_hidden_dim: int = MFP_SIAMESE_ENCODER_HIDDEN_DIM,
        drug_latent_dim: int = MFP_SIAMESE_LATENT_DIM,
        classifier_hidden_dim: int = MFP_SIAMESE_CLASSIFIER_HIDDEN_DIM,
        dropout: float = 0.3,
    ) -> MFPSiameseMember:
        return MFPSiameseMember(
            MFPSiameseMemberConfig(
                num_classes=MFP_SIAMESE_FIXED_CLASS_COUNT,
                drug_input_dim=drug_input_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                drug_latent_dim=drug_latent_dim,
                classifier_hidden_dim=classifier_hidden_dim,
                dropout=dropout,
            )
        )


else:

    class MFPSiameseMember:  # pragma: no cover - runtime guard without torch
        def __init__(
            self,
            config: MFPSiameseMemberConfig = MFPSiameseMemberConfig(),
        ) -> None:
            raise ImportError("torch is required to instantiate MFPSiameseMember.")


    def build_fixed_head_mfp_siamese_member(**kwargs):  # pragma: no cover
        raise ImportError("torch is required to instantiate MFPSiameseMember.")
