#!/usr/bin/env python3
"""Build the label-free Morgan-2048 cache for MFP-Siamese."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_ddinet_fingerprint_cache import collect_records
from src.data.morgan_fingerprints import MorganFingerprintConfig, build_morgan_fingerprint_cache


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200_000)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    paths = [args.train, args.validation, args.test]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    print("[MFP] Reading drug IDs and SMILES for the label-free cache.", flush=True)
    records = collect_records(paths, batch_size=args.batch_size)
    cache = build_morgan_fingerprint_cache(
        args.outdir, records,
        config=MorganFingerprintConfig(n_bits=2048, consumer="mfp_siamese"),
    )
    print(f"[MFP] Cache ready: {cache.root}; unique drugs={len(records)}; pair input={cache.input_dim}", flush=True)


if __name__ == "__main__":
    main()
