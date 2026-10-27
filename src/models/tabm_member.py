"""Basic TabM adapter: branch-wise training and probability-ensemble inference.

The implementation of BatchEnsemble and its initialization comes from the
official ``tabm`` package. The native per-branch heads are an integration head,
not a decision about the future common head or model-merging algorithm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TabMMemberConfig:
    num_classes: int
    input_dim: int = 3780
    n_blocks: int = 3
    d_block: int = 512
    k: int = 32
    dropout: float = 0.1

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (
            self.num_classes, self.input_dim, self.n_blocks, self.d_block, self.k
        )):
            raise ValueError("TabM dimensions must be positive integers.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("TabM dropout must be finite and in [0,1).")

    def manifest(self):
        values = asdict(self)
        values.pop("num_classes")
        return {
            **values, "variant": "tabm", "architecture": "TabM_basic",
            "activation": "relu", "input_normalization": "external_frozen_scaler",
            "num_embeddings": None, "classifier": "native_expandable_branch_heads",
            "head_weight_class_axis": 2, "head_bias_class_axis": 1,
            "training_objective": "mean_branch_cross_entropy",
            "prediction_aggregation": "mean_branch_probabilities",
            "export_logits": "log_mean_branch_probabilities",
            "implementation": "tabm", "implementation_version": importlib.metadata.version("tabm"),
        }


class TabMMember(nn.Module):
    def __init__(self, config: TabMMemberConfig):
        super().__init__()
        try:
            from tabm import TabM, LinearEnsemble
        except ImportError as error:
            raise ImportError("TabM requires: python -m pip install -r requirements-tabm.txt") from error
        self.config = config
        self.backbone = TabM.make(
            n_num_features=config.input_dim, d_out=None, num_embeddings=None,
            n_blocks=config.n_blocks, d_block=config.d_block, k=config.k,
            dropout=config.dropout, activation="ReLU", arch_type="tabm",
        )
        self.head = LinearEnsemble(config.d_block, config.num_classes, k=config.k)

    def encode(self, features):
        """Return all branch features, shape [B,K,H]."""
        return self.backbone(features)

    def classify(self, latent):
        return self.head(latent)

    def forward_branches(self, features):
        return self.classify(self.encode(features))

    @staticmethod
    def aggregate_logits(branch_logits):
        """Effective log-probabilities, not the mean of branch logits.

        softmax(result) equals mean(softmax(branch_logits)), so the existing
        two-dimensional prediction artifact contract remains valid. logsumexp
        avoids clamping/underflow even for very confident branches.
        """
        if branch_logits.ndim != 3:
            raise ValueError("TabM logits must have shape [batch, branches, classes].")
        return torch.logsumexp(F.log_softmax(branch_logits, dim=-1), dim=1) - math.log(branch_logits.shape[1])

    def forward(self, features):
        return self.aggregate_logits(self.forward_branches(features))

    def forward_with_latent(self, features):
        latent = self.encode(features)
        # A summary for visualization only. The native heads see all K branches.
        return self.aggregate_logits(self.classify(latent)), latent.mean(dim=1)

    def classification_loss(self, features, labels, criterion):
        branches = self.forward_branches(features)
        targets = labels[:, None].expand(-1, branches.shape[1]).reshape(-1)
        loss = criterion(branches.reshape(-1, branches.shape[-1]), targets)
        return self.aggregate_logits(branches), loss


def copy_head_classes(target, source, previous_map, current_map):
    """Align native TabM [K,H,C]/[K,C] heads by raw class, never by row position."""
    with torch.no_grad():
        for raw, old_index in previous_map.items():
            new_index = current_map[raw]
            target["head.weight"][..., new_index].copy_(source["head.weight"][..., old_index])
            target["head.bias"][..., new_index].copy_(source["head.bias"][..., old_index])
