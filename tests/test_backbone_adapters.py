import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.fixed_budget_replay import FixedBudgetReplayBuffer
from src.data.molecular_graphs import MolecularGraphBank, molecule_to_arrays
from src.models.ddi_gcn import DDIGCNClassifier
from src.models.tabm_classifier import TabMClassifier
from src.training.train_cil import FocalLoss, expand_model_for_seen_classes, train_one_epoch


def _tiny_graph_bank() -> MolecularGraphBank:
    rows = [molecule_to_arrays(smiles, max_atoms=4) for smiles in ["C", "CC", "CCO"]]
    return MolecularGraphBank(
        drug_id_to_index={"a": 0, "b": 1, "c": 2},
        atom_features=torch.from_numpy(np.stack([row["atom_features"] for row in rows])),
        neighbor_indices=torch.from_numpy(
            np.stack([row["neighbor_indices"] for row in rows]).astype(np.int64)
        ),
        bond_sums=torch.from_numpy(np.stack([row["bond_sums"] for row in rows])),
        atom_degrees=torch.from_numpy(
            np.stack([row["atom_degrees"] for row in rows]).astype(np.int64)
        ),
        atom_mask=torch.from_numpy(np.stack([row["atom_mask"] for row in rows])),
        max_atoms=4,
    )


def test_fixed_buffer_can_rank_descriptors_while_storing_graph_inputs() -> None:
    buffer = FixedBudgetReplayBuffer(total_memory_budget=2)
    graph_pair_inputs = np.asarray([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    descriptor_ranking = np.asarray([[100], [0], [0], [100]], dtype=np.float32)

    buffer.update(graph_pair_inputs, labels, ranking_features=descriptor_ranking)

    retained, retained_labels = buffer.get_all()
    np.testing.assert_array_equal(retained_labels, [0, 1])
    # Equal distances tie; stable source-row index is retained for each class.
    np.testing.assert_array_equal(retained, [[10, 11], [30, 31]])


def test_tabm_member_training_and_probability_aggregation() -> None:
    model = TabMClassifier(input_dim=5, num_classes=3, k=2, n_blocks=1, d_block=8)
    model.eval()
    inputs = torch.randn(4, 5)
    members = model.forward_members(inputs)
    aggregated = model(inputs)

    assert members.shape == (4, 2, 3)
    assert aggregated.shape == (4, 3)
    torch.testing.assert_close(aggregated.exp(), members.softmax(-1).mean(1))


def test_tabm_expansion_preserves_old_member_heads() -> None:
    old = expand_model_for_seen_classes(
        None,
        None,
        {3: 0, 7: 1},
        variant="tabm",
        input_dim=5,
        dropout=0.2,
        activation="gelu",
        norm="layernorm",
        tabm_k=2,
        tabm_blocks=1,
        tabm_d_block=8,
    )
    expanded = expand_model_for_seen_classes(
        old,
        {3: 0, 7: 1},
        {3: 0, 7: 1, 9: 2},
        variant="tabm",
        input_dim=5,
        dropout=0.2,
        activation="gelu",
        norm="layernorm",
        tabm_k=2,
        tabm_blocks=1,
        tabm_d_block=8,
    )

    torch.testing.assert_close(expanded.head.weight[:, :, :2], old.head.weight)
    torch.testing.assert_close(expanded.head.bias[:, :2], old.head.bias)


def test_ddi_gcn_forward_and_expansion_preserve_old_heads() -> None:
    graph_bank = _tiny_graph_bank()
    model = DDIGCNClassifier(
        graph_bank=graph_bank,
        num_classes=2,
        depth=1,
        width=8,
        attention_dim=4,
    )
    pair_indices = torch.tensor([[0, 1], [2, 0]], dtype=torch.float32)
    logits, latent = model.forward_with_latent(pair_indices)
    assert logits.shape == (2, 2)
    assert latent.shape == (2, 100)

    expanded = expand_model_for_seen_classes(
        model,
        {3: 0, 7: 1},
        {3: 0, 7: 1, 9: 2},
        variant="ddi_gcn",
        input_dim=2,
        dropout=0.2,
        activation="gelu",
        norm="layernorm",
        graph_bank=graph_bank,
        ddi_gcn_depth=1,
        ddi_gcn_width=8,
        ddi_gcn_attention_dim=4,
    )
    torch.testing.assert_close(expanded.head.weight[:2], model.head.weight)
    torch.testing.assert_close(expanded.head.bias[:2], model.head.bias)


def test_gradient_accumulation_matches_effective_batch_including_partial_group() -> None:
    torch.manual_seed(3)
    features = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 0, 1, 1])
    dataset = TensorDataset(features, labels)
    reference = nn.Sequential(nn.Linear(3, 2))
    accumulated = nn.Sequential(nn.Linear(3, 2))
    accumulated.load_state_dict(reference.state_dict())
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)

    train_one_epoch(
        reference,
        DataLoader(dataset, batch_size=4, shuffle=False),
        reference_optimizer,
        FocalLoss(gamma=0.0),
        "cpu",
    )
    train_one_epoch(
        accumulated,
        DataLoader(dataset, batch_size=2, shuffle=False),
        accumulated_optimizer,
        FocalLoss(gamma=0.0),
        "cpu",
        gradient_accumulation_steps=2,
    )

    for expected, actual in zip(reference.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
