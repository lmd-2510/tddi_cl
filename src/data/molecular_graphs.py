"""Molecular graph cache primitives for the PyTorch DDI-GCN port."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - guarded by graph-cache construction
    Chem = None


ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
    "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Sn", "Ag", "Pd",
    "Co", "Se", "Ti", "Zn", "Li", "Ge", "Cu", "Au", "Ni", "Mn", "Other",
]
DEGREES = [0, 1, 2, 3, 4, 5]
STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]


def one_hot_unknown(value: object, choices: list[object]) -> list[bool]:
    normalized = value if value in choices else choices[-1]
    return [normalized == choice for choice in choices]


def atom_features(atom: object) -> np.ndarray:
    if Chem is None:
        raise ImportError("RDKit is required to construct molecular graphs.")
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        "Other",
    ]
    values: list[float | bool] = []
    values.extend(one_hot_unknown(atom.GetSymbol(), ATOM_SYMBOLS))
    degree = atom.GetDegree()
    if degree not in DEGREES:
        raise ValueError(f"DDI-GCN supports atom degrees 0..5, got {degree}.")
    values.extend(degree == choice for choice in DEGREES)
    values.extend([float(atom.GetFormalCharge()), float(atom.GetNumRadicalElectrons())])
    values.extend(one_hot_unknown(atom.GetHybridization(), hybridizations))
    values.append(atom.GetIsAromatic())
    try:
        values.extend(one_hot_unknown(atom.GetProp("_CIPCode"), ["R", "S"]))
        values.append(atom.HasProp("_ChiralityPossible"))
    except (KeyError, RuntimeError):
        values.extend([False, False, atom.HasProp("_ChiralityPossible")])
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (51,):
        raise RuntimeError(f"Expected 51 DDI-GCN atom features, got {result.shape}.")
    return result


def bond_features(bond: object) -> np.ndarray:
    if Chem is None:
        raise ImportError("RDKit is required to construct molecular graphs.")
    bond_type = bond.GetBondType()
    values: list[bool] = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    values.extend(one_hot_unknown(str(bond.GetStereo()), STEREO))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (10,):
        raise RuntimeError(f"Expected 10 DDI-GCN bond features, got {result.shape}.")
    return result


@dataclass(frozen=True)
class MolecularGraphBank:
    drug_id_to_index: dict[str, int]
    atom_features: torch.Tensor
    neighbor_indices: torch.Tensor
    bond_sums: torch.Tensor
    atom_degrees: torch.Tensor
    atom_mask: torch.Tensor
    max_atoms: int


def molecule_to_arrays(smiles: str, *, max_atoms: int) -> dict[str, np.ndarray]:
    if Chem is None:
        raise ImportError("RDKit is required to construct molecular graphs.")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    num_atoms = molecule.GetNumAtoms()
    if num_atoms > max_atoms:
        raise ValueError(f"Molecule has {num_atoms} atoms, above max_atoms={max_atoms}.")

    atoms = np.zeros((max_atoms, 51), dtype=np.float32)
    neighbors = np.full((max_atoms, 5), -1, dtype=np.int32)
    bonds = np.zeros((max_atoms, 10), dtype=np.float32)
    degrees = np.full((max_atoms,), -1, dtype=np.int8)
    mask = np.zeros((max_atoms,), dtype=np.bool_)
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        atoms[index] = atom_features(atom)
        atom_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        degree = len(atom_neighbors)
        if degree > 5:
            raise ValueError(f"Atom degree {degree} is unsupported by DDI-GCN.")
        neighbors[index, :degree] = atom_neighbors
        degrees[index] = degree
        mask[index] = True
        for bond in atom.GetBonds():
            bonds[index] += bond_features(bond)
    return {
        "atom_features": atoms,
        "neighbor_indices": neighbors,
        "bond_sums": bonds,
        "atom_degrees": degrees,
        "atom_mask": mask,
    }


def load_graph_bank(cache_path: str | Path, mapping_path: str | Path) -> MolecularGraphBank:
    arrays = np.load(cache_path, allow_pickle=False)
    mapping_payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    mapping = {str(key): int(value) for key, value in mapping_payload["drug_id_to_index"].items()}
    return MolecularGraphBank(
        drug_id_to_index=mapping,
        atom_features=torch.from_numpy(arrays["atom_features"].astype(np.float32, copy=False)),
        neighbor_indices=torch.from_numpy(arrays["neighbor_indices"].astype(np.int64, copy=False)),
        bond_sums=torch.from_numpy(arrays["bond_sums"].astype(np.float32, copy=False)),
        atom_degrees=torch.from_numpy(arrays["atom_degrees"].astype(np.int64, copy=False)),
        atom_mask=torch.from_numpy(arrays["atom_mask"].astype(np.bool_, copy=False)),
        max_atoms=int(arrays["atom_features"].shape[1]),
    )
