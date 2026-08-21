"""PyTorch port of the categorical DDI-GCN molecular graph backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.molecular_graphs import MolecularGraphBank


class DegreeGraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bond_dim: int = 10) -> None:
        super().__init__()
        self.degree_layers = nn.ModuleList(
            nn.Linear(input_dim + bond_dim, output_dim) for _ in range(6)
        )
        self.self_layer = nn.Linear(input_dim, output_dim)
        self.norm = nn.BatchNorm1d(output_dim)

    @staticmethod
    def neighbor_sum(features: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
        batch_size, num_atoms, _ = features.shape
        valid = neighbor_indices >= 0
        safe_indices = neighbor_indices.clamp_min(0)
        batch_indices = torch.arange(batch_size, device=features.device).view(-1, 1, 1)
        batch_indices = batch_indices.expand_as(safe_indices)
        gathered = features[batch_indices, safe_indices]
        return (gathered * valid.unsqueeze(-1)).sum(dim=2)

    def forward(
        self,
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        bond_sums: torch.Tensor,
        atom_degrees: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neighbors = self.neighbor_sum(features, neighbor_indices)
        merged = torch.cat([neighbors, bond_sums], dim=-1)
        neighbor_activation = torch.zeros(
            (*features.shape[:2], self.self_layer.out_features),
            dtype=features.dtype,
            device=features.device,
        )
        for degree, layer in enumerate(self.degree_layers):
            degree_mask = atom_degrees == degree
            if degree_mask.any():
                neighbor_activation[degree_mask] = F.relu(layer(merged[degree_mask]))
        self_activation = F.relu(self.self_layer(features))
        output = neighbor_activation + self_activation
        flat = output.reshape(-1, output.shape[-1])
        flat = self.norm(flat)
        output = flat.reshape_as(output) * atom_mask.unsqueeze(-1)
        return output, self_activation * atom_mask.unsqueeze(-1)


class MolecularEncoder(nn.Module):
    def __init__(self, *, depth: int = 8, width: int = 128) -> None:
        super().__init__()
        self.depth = int(depth)
        self.width = int(width)
        layers = []
        input_dim = 51
        for _ in range(self.depth):
            layers.append(DegreeGraphConv(input_dim, self.width))
            input_dim = self.width
        self.layers = nn.ModuleList(layers)
        self.layer_gru = nn.GRU(
            input_size=self.width,
            hidden_size=self.width,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_attention = nn.Linear(self.width * 2, 1)

    def forward(
        self,
        atom_features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        bond_sums: torch.Tensor,
        atom_degrees: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        states: list[torch.Tensor] = []
        features = atom_features
        for layer_index, layer in enumerate(self.layers):
            features, self_activation = layer(
                features,
                neighbor_indices,
                bond_sums,
                atom_degrees,
                atom_mask,
            )
            if layer_index == 0:
                states.append(self_activation)
            states.append(features)
        sequence = torch.stack(states, dim=2)
        batch_size, num_atoms, num_states, width = sequence.shape
        sequence = sequence.reshape(batch_size * num_atoms, num_states, width)
        encoded, _ = self.layer_gru(sequence)
        weights = F.softmax(self.layer_attention(encoded), dim=1)
        pooled = (weights * encoded).sum(dim=1)
        pooled = pooled.reshape(batch_size, num_atoms, self.width * 2)
        return pooled * atom_mask.unsqueeze(-1)


class CoAttention(nn.Module):
    def __init__(self, feature_dim: int = 128, attention_dim: int = 65) -> None:
        super().__init__()
        self.w_m = nn.Parameter(torch.empty(attention_dim, feature_dim))
        self.w_v = nn.Parameter(torch.empty(attention_dim, feature_dim))
        self.w_q = nn.Parameter(torch.empty(attention_dim, feature_dim))
        self.w_h = nn.Parameter(torch.empty(1, attention_dim))
        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_mask: torch.Tensor,
        right_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_t = left.transpose(1, 2)
        right_t = right.transpose(1, 2)
        left_count = left_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(1)
        right_count = right_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(1)
        left_mean = torch.tanh(left_t.sum(dim=-1, keepdim=True) / left_count)
        right_mean = torch.tanh(right_t.sum(dim=-1, keepdim=True) / right_count)
        shared = left_mean * right_mean
        shared_projection = torch.tanh(torch.matmul(self.w_m, shared))
        left_hidden = torch.tanh(torch.matmul(self.w_v, left_t)) * shared_projection
        right_hidden = torch.tanh(torch.matmul(self.w_q, right_t)) * shared_projection
        left_scores = torch.matmul(self.w_h, left_hidden).squeeze(1)
        right_scores = torch.matmul(self.w_h, right_hidden).squeeze(1)
        left_scores = left_scores.masked_fill(~left_mask, torch.finfo(left_scores.dtype).min)
        right_scores = right_scores.masked_fill(~right_mask, torch.finfo(right_scores.dtype).min)
        left_weights = F.softmax(left_scores, dim=-1).unsqueeze(1)
        right_weights = F.softmax(right_scores, dim=-1).unsqueeze(1)
        return (
            torch.bmm(left_weights, left).squeeze(1),
            torch.bmm(right_weights, right).squeeze(1),
        )


class DDIGCNClassifier(nn.Module):
    """Shared molecular encoder, co-attention, FC merger and expandable head."""

    def __init__(
        self,
        *,
        graph_bank: MolecularGraphBank,
        num_classes: int,
        depth: int = 8,
        width: int = 128,
        attention_dim: int = 65,
    ) -> None:
        super().__init__()
        self.register_buffer("graph_atom_features", graph_bank.atom_features, persistent=False)
        self.register_buffer("graph_neighbor_indices", graph_bank.neighbor_indices, persistent=False)
        self.register_buffer("graph_bond_sums", graph_bank.bond_sums, persistent=False)
        self.register_buffer("graph_atom_degrees", graph_bank.atom_degrees, persistent=False)
        self.register_buffer("graph_atom_mask", graph_bank.atom_mask, persistent=False)
        self.backbone = MolecularEncoder(depth=depth, width=width)
        self.atom_projection = nn.Linear(width * 2, width)
        self.co_attention = CoAttention(width, attention_dim)
        self.merger = nn.Sequential(
            nn.Linear(width * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.head = nn.Linear(100, num_classes)

    def _encode_drug(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        atom_features = self.graph_atom_features[indices]
        neighbor_indices = self.graph_neighbor_indices[indices]
        bond_sums = self.graph_bond_sums[indices]
        atom_degrees = self.graph_atom_degrees[indices]
        atom_mask = self.graph_atom_mask[indices]
        encoded = self.backbone(
            atom_features,
            neighbor_indices,
            bond_sums,
            atom_degrees,
            atom_mask,
        )
        return self.atom_projection(encoded), atom_mask

    def encode(self, pair_indices: torch.Tensor) -> torch.Tensor:
        pair_indices = pair_indices.long()
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            raise ValueError("DDI-GCN inputs must have shape (batch, 2) drug indices.")
        left, left_mask = self._encode_drug(pair_indices[:, 0])
        right, right_mask = self._encode_drug(pair_indices[:, 1])
        attended_left, attended_right = self.co_attention(
            left,
            right,
            left_mask,
            right_mask,
        )
        return self.merger(torch.cat([attended_left, attended_right], dim=-1))

    def classify(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)

    def forward_with_latent(
        self,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(pair_indices)
        return self.classify(latent), latent

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(pair_indices))
