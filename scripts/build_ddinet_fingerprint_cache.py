#!/usr/bin/env python3
"""Build the immutable, label-free Morgan cache consumed by DDINet.

Only drug IDs and SMILES are read from the three source Parquets. Labels,
task assignments and descriptor columns are never used to construct this
cache, so the artifact can be shared by all continual-learning methods.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ddinet_fingerprints import (  # noqa: E402
    DDINetFingerprintConfig,
    build_ddinet_fingerprint_cache,
)


PAIR_COLUMNS = (
    "drugid-drug_a", "drugsmiles-drug_a",
    "drugid-drug_b", "drugsmiles-drug_b",
)


def collect_records(paths: list[Path], *, batch_size: int) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = sorted(set(PAIR_COLUMNS) - set(parquet.schema.names))
        if missing:
            raise ValueError(f"{path} is missing DDINet columns: {missing}")
        for batch in parquet.iter_batches(columns=list(PAIR_COLUMNS), batch_size=batch_size):
            data = batch.to_pydict()
            for id_column, smiles_column in (
                ("drugid-drug_a", "drugsmiles-drug_a"),
                ("drugid-drug_b", "drugsmiles-drug_b"),
            ):
                for raw_id, raw_smiles in zip(data[id_column], data[smiles_column], strict=True):
                    if raw_id is None or raw_smiles is None:
                        raise ValueError(f"Null drug ID/SMILES found in {path}")
                    drug_id = str(raw_id).strip()
                    smiles = str(raw_smiles).strip()
                    previous = records.get(drug_id)
                    if previous is not None and previous != smiles:
                        raise ValueError(
                            f"Drug ID {drug_id!r} maps to conflicting raw SMILES."
                        )
                    records[drug_id] = smiles
    if not records:
        raise ValueError("No drug ID/SMILES records were found.")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200_000)
    args = parser.parse_args(argv)
    paths = [args.train, args.validation, args.test]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.outdir.exists() and not (args.outdir / "manifest.json").is_file():
        raise FileExistsError(f"Refusing to reuse incomplete cache directory: {args.outdir}")
    print("[DDINET] Collecting label-free drug ID/SMILES records...", flush=True)
    records = collect_records(paths, batch_size=args.batch_size)
    print(f"[DDINET] Unique drugs: {len(records)}; building Morgan-1024 cache...", flush=True)
    cache = build_ddinet_fingerprint_cache(args.outdir, records, config=DDINetFingerprintConfig())
    print(f"[DDINET] Cache ready: {cache.root}", flush=True)
    print(f"[DDINET] Input dimension: {cache.input_dim}; manifest: {cache.root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
