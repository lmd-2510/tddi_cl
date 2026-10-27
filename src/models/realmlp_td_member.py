"""Numerical RealMLP-TD, ported from the pinned PyTabKit implementation.

Sources/license: third_party/pytabkit/README.md. This keeps the full TD numerical
embedding and training parameterization, while using the repository's frozen
external preprocessing. It does not call sklearn.fit or recreate a CIL student
from scratch at every task.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_COMMIT = "c126ea51187c5080b91f28d352481dbd3b2194b0"


@dataclass(frozen=True)
class RealMLPTDConfig:
    num_classes: int
    input_dim: int = 3780
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    dropout: float = 0.15
    embedding_frequencies: int = 16
    embedding_dim: int = 4
    embedding_sigma: float = 0.1

    def __post_init__(self):
        if not self.hidden_dims or any(type(n) is not int or n <= 0 for n in (
            self.num_classes, self.input_dim, *self.hidden_dims,
            self.embedding_frequencies, self.embedding_dim,
        )) or self.embedding_dim < 2:
            raise ValueError("RealMLP-TD requires positive dimensions and embedding_dim >= 2.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be finite and in [0,1).")
        if not math.isfinite(self.embedding_sigma) or self.embedding_sigma <= 0:
            raise ValueError("embedding_sigma must be positive and finite.")

    def manifest(self):
        values = asdict(self)
        values.pop("num_classes")
        values["hidden_dims"] = list(self.hidden_dims)
        return {**values, "variant": "realmlp_td", "architecture": "RealMLP_TD_numerical",
            "num_embeddings": "pbld", "activation": "parametric_selu",
            "input_normalization": "external_frozen_scaler", "internal_scaler": None,
            "weight_parameterization": "ntk", "weight_initialization": "train_std",
            "bias_initialization": "he+5", "initialization_ram_limit_gib": 1.0,
            "initialization_policy": "train_only_bounded_uniform_subsample",
            "initialization_std_floor": 1e-6,
            "classifier": "native_expandable_ntk_linear", "head_weight_class_axis": 0,
            "head_bias_class_axis": 0, "training_objective": "cross_entropy_label_smoothing_0.1",
            "prediction_aggregation": "single_member_softmax",
            "implementation": "local_torch_port_of_pytabkit", "upstream_commit": UPSTREAM_COMMIT}


class PBLDEmbedding(nn.Module):
    """Per-feature cosine -> linear embedding, plus the original features."""
    def __init__(self, config: RealMLPTDConfig):
        super().__init__()
        d, f, channels = config.input_dim, config.embedding_frequencies, config.embedding_dim - 1
        self.weight_1 = nn.Parameter(config.embedding_sigma * torch.randn(d, f))
        self.bias_1 = nn.Parameter(torch.empty(d, f).uniform_(-math.pi, math.pi))
        self.weight_2 = nn.Parameter(torch.empty(d, f, channels).uniform_(-1 / math.sqrt(f), 1 / math.sqrt(f)))
        self.bias_2 = nn.Parameter(torch.empty(d, channels).uniform_(-1 / math.sqrt(f), 1 / math.sqrt(f)))

    def forward(self, x):
        periodic = torch.cos(2 * math.pi * x.unsqueeze(-1) * self.weight_1 + self.bias_1)
        projected = torch.einsum("bdf,dfh->bdh", periodic, self.weight_2) + self.bias_2
        # Upstream concatenates flattened learned features, then all raw features.
        return torch.cat((projected.flatten(1), x), dim=-1)


class NTKLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.factor = 1 / math.sqrt(in_features)

    def forward(self, x):
        return F.linear(x, self.factor * self.weight, self.bias)

    @torch.no_grad()
    def initialize_from_training(self, x, indices=None):
        """Fit projection std and He+5 bias using available training rows only."""
        if indices is None:
            indices = torch.arange(len(self.bias), device=x.device)
        else:
            indices = torch.as_tensor(indices, dtype=torch.long, device=x.device)
        w = self.weight[indices]
        out = F.linear(x, self.factor * w)
        std = out.std(0, correction=0).clamp_min(1e-6)
        w = w / std[:, None]
        out = F.linear(x, self.factor * w)
        choices = torch.randint(len(x), (len(indices), 5), device=x.device)
        simplex = torch.distributions.Exponential(1.0).sample((len(indices), 5)).to(x.device)
        simplex /= simplex.sum(-1, keepdim=True)
        selected = out[choices, torch.arange(len(indices), device=x.device)[:, None]]
        self.weight[indices] = w
        self.bias[indices] = -(selected * simplex).sum(-1)


class ParametricSELU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x + self.weight * (F.selu(x) - x)


class RealMLPTDMember(nn.Module):
    def __init__(self, config: RealMLPTDConfig):
        super().__init__()
        self.config = config
        self.embedding = PBLDEmbedding(config)
        width = config.input_dim * config.embedding_dim
        self.front_scale = nn.Parameter(torch.ones(width))
        self.blocks = nn.ModuleList()
        for out in config.hidden_dims:
            self.blocks.append(nn.ModuleDict({"linear": NTKLinear(width, out),
                "activation": ParametricSELU(out), "dropout": nn.Dropout(config.dropout)}))
            width = out
        self.head = NTKLinear(width, config.num_classes)

    def encode(self, x):
        x = self.embedding(x) * self.front_scale
        for block in self.blocks:
            x = block["dropout"](block["activation"](block["linear"](x)))
        return x

    def forward(self, x):
        return self.head(self.encode(x))

    def forward_with_latent(self, x):
        latent = self.encode(x)
        return self.head(latent), latent

    @torch.no_grad()
    def initialize_for_task(self, training_features, new_indices, *, initialize_backbone):
        # Match PyTabKit's bounded train-only initialization instead of fitting a
        # second scaler. Conservatively bound the largest PBLD intermediate.
        values_per_row = self.config.input_dim * (3 * self.config.embedding_frequencies + self.config.embedding_dim)
        max_rows = max(2, (1024 ** 3) // (4 * values_per_row))
        if len(training_features) > max_rows:
            selected = torch.randperm(len(training_features))[:max_rows]
            training_features = training_features[selected]
        x = training_features.to(device=self.front_scale.device, dtype=self.front_scale.dtype)
        if not len(x) or not torch.isfinite(x).all():
            raise ValueError("RealMLP initialization requires finite non-empty current training data.")
        was_training = self.training
        self.eval()
        try:
            x = self.embedding(x) * self.front_scale
            for block in self.blocks:
                if initialize_backbone:
                    block["linear"].initialize_from_training(x)
                x = block["activation"](block["linear"](x))
                if initialize_backbone:
                    # The upstream fit-transform pipeline has dropout enabled.
                    x = F.dropout(x, self.config.dropout, training=True)
            self.head.initialize_from_training(x, new_indices)
        finally:
            self.train(was_training)
        return int(len(x))


def schedule_factors(progress):
    """PyTabKit coslog4 LR and flat_cos dropout/weight-decay schedules."""
    t = min(1., max(0., float(progress)))
    lr = 0.5 * (1 - math.cos(2 * math.pi * math.log2(1 + 15 * t)))
    regularization = 1. if t <= .5 else .5 * (1 + math.cos(math.pi * (2 * t - 1)))
    return lr, regularization


def parameter_factors(name):
    if name.startswith("embedding."):
        return .1, 1.
    if name == "front_scale":
        return 6., 1.
    if ".activation." in name:
        return .1, 1.
    if name.endswith(".bias"):
        return .1, 0.
    return 1., 1.


def make_optimizer(model, *, lr, weight_decay):
    groups = []
    for name, parameter in model.named_parameters():
        lr_factor, wd_factor = parameter_factors(name)
        groups.append({"params": [parameter], "lr": lr * lr_factor,
            # Upstream OptimizerBase applies the hyper factors in its getters
            # and again in decoupled decay. Keep that pinned implementation's
            # semantics, rather than silently substituting uniform AdamW decay.
            "weight_decay": weight_decay * wd_factor ** 2 * lr_factor,
            "realmlp_name": name, "realmlp_lr_factor": lr_factor,
            "realmlp_wd_factor": wd_factor, "realmlp_base_lr": lr,
            "realmlp_base_wd": weight_decay})
    return torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8)


def update_training_schedule(model, optimizer, progress):
    lr_multiplier, regularization = schedule_factors(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["realmlp_base_lr"] * group["realmlp_lr_factor"] * lr_multiplier
        group["weight_decay"] = (group["realmlp_base_wd"] * regularization
            * group["realmlp_wd_factor"] ** 2 * group["realmlp_lr_factor"])
    for block in model.blocks:
        block["dropout"].p = model.config.dropout * regularization


def should_stop(epoch, best_epoch, *, additive_patience=20):
    """Optional upstream stopping rule, applied to the repo's Macro-F1 selection."""
    return best_epoch is not None and epoch > 2 * best_epoch + additive_patience
