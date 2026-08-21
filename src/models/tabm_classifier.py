"""TabM classifier adapter with an expandable class-incremental head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tabm
except ImportError:  # pragma: no cover - guarded when the model is constructed
    tabm = None


class TabMClassifier(nn.Module):
    """Official TabM backbone plus a separately expandable ensemble head.

    Training uses every member prediction independently. Inference averages member
    probabilities, following the official TabM guidance.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tabm is None:
            raise ImportError("TabM requires the 'tabm' package.")
        self.k = int(k)
        self.latent_dim = int(d_block)
        self.backbone = tabm.TabM.make(
            n_num_features=input_dim,
            d_out=None,
            k=self.k,
            n_blocks=n_blocks,
            d_block=self.latent_dim,
            dropout=dropout,
            arch_type="tabm",
        )
        self.head = tabm.LinearEnsemble(
            self.latent_dim,
            num_classes,
            k=self.k,
        )

    def encode_members(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x_num=x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_members(x).mean(dim=1)

    def classify_members(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)

    def forward_members_with_latent(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_members = self.encode_members(x)
        return self.classify_members(latent_members), latent_members.mean(dim=1)

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_members(self.encode_members(x))

    @staticmethod
    def aggregate_member_logits(member_logits: torch.Tensor) -> torch.Tensor:
        if member_logits.ndim != 3:
            raise ValueError("TabM member logits must have shape (batch, k, classes).")
        return torch.logsumexp(F.log_softmax(member_logits, dim=-1), dim=1) - math.log(
            member_logits.shape[1]
        )

    def forward_with_latent(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        member_logits, latent = self.forward_members_with_latent(x)
        return self.aggregate_member_logits(member_logits), latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregate_member_logits(self.forward_members(x))
