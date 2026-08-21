#!/usr/bin/env python3
"""Build a deterministic DDI-GCN molecular graph bank from all split metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.molecular_graphs import molecule_to_arrays


DRUG_COLUMNS = [
    ("drugid-drug_a", "drugsmiles-drug_a"),
    ("drugid-drug_b", "drugsmiles-drug_b"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--max-atoms", type=int, default=65)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_path = args.outdir / "graphs.npz"
    mapping_path = args.outdir / "drug_id_mapping.json"
    manifest_path = args.outdir / "manifest.json"
    for path in [cache_path, mapping_path, manifest_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite graph-cache artifact: {path}")

    records = []
    for split in args.splits:
        columns = [column for pair in DRUG_COLUMNS for column in pair]
        frame = pq.read_table(split, columns=columns).to_pandas()
        for drug_column, smiles_column in DRUG_COLUMNS:
            records.append(
                pd.DataFrame(
                    {
                        "drug_id": frame[drug_column].astype(str),
                        "smiles": frame[smiles_column].astype(str),
                    }
                )
            )
    drugs = pd.concat(records, ignore_index=True).drop_duplicates()
    conflicts = drugs.groupby("drug_id")["smiles"].nunique()
    if (conflicts > 1).any():
        raise ValueError("At least one drug ID maps to multiple SMILES strings.")
    drugs = drugs.sort_values("drug_id").drop_duplicates("drug_id").reset_index(drop=True)

    graph_rows = [
        molecule_to_arrays(row.smiles, max_atoms=args.max_atoms)
        for row in drugs.itertuples(index=False)
    ]
    np.savez_compressed(
        cache_path,
        atom_features=np.stack([row["atom_features"] for row in graph_rows]),
        neighbor_indices=np.stack([row["neighbor_indices"] for row in graph_rows]),
        bond_sums=np.stack([row["bond_sums"] for row in graph_rows]),
        atom_degrees=np.stack([row["atom_degrees"] for row in graph_rows]),
        atom_mask=np.stack([row["atom_mask"] for row in graph_rows]),
    )
    mapping = {drug_id: index for index, drug_id in enumerate(drugs["drug_id"].tolist())}
    mapping_path.write_text(
        json.dumps({"drug_id_to_index": mapping}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "source_splits": [str(path) for path in args.splits],
        "num_drugs": len(mapping),
        "max_atoms": args.max_atoms,
        "atom_feature_dim": 51,
        "bond_feature_dim": 10,
        "cache_sha256": sha256(cache_path),
        "mapping_sha256": sha256(mapping_path),
        "source": "PyTorch port of LabWeng/DDI-GCN categorical graph features",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] Built {len(mapping)} molecular graphs in {args.outdir}")


if __name__ == "__main__":
    main()
