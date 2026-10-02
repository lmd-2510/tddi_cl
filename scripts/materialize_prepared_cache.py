#!/usr/bin/env python3
"""Materialize train/validation features with one frozen task-0 scaler.

This is an I/O optimization only.  The source rows, labels and sample
identities are preserved; no task split is rebuilt and no scaler is fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ddi_dataset import load_feature_columns, load_scaler_payload
from src.data.prepared_cache import (
    CACHE_MANIFEST,
    CACHE_SCHEMA_VERSION,
    cache_split_path,
    feature_columns_sha256,
    materialize_split,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, help="Optional test cache; omitted by default to save disk.")
    parser.add_argument("--feature-cols", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--scaler-source", type=Path, help="Frozen fold_preprocessing.json used to create the scaler.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    feature_columns = load_feature_columns(args.feature_cols)
    scaler = load_scaler_payload(args.scaler)
    args.output.mkdir(parents=True, exist_ok=True)

    sources = {"train": args.train, "validation": args.validation}
    if args.test is not None:
        sources["test"] = args.test
    split_metadata = {}
    for split, source in sources.items():
        destination = cache_split_path(args.output, split)
        print(f"[CACHE] {split}: {source} -> {destination}", flush=True)
        rows = materialize_split(
            source,
            destination,
            feature_columns=feature_columns,
            scaler=scaler,
            split=split,
            batch_size=args.batch_size,
        )
        split_metadata[split] = {
            "source": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "row_count": rows,
            "sha256": sha256_file(destination),
        }
        print(f"[CACHE] {split}: {rows} rows", flush=True)

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "ddi_cil_prepared_feature_cache",
        "policy": "task0_standard_frozen",
        "feature_columns_sha256": feature_columns_sha256(feature_columns),
        "feature_columns_path": str(args.feature_cols.resolve()),
        "scaler": str(args.scaler.resolve()),
        "scaler_sha256": sha256_file(args.scaler),
        "scaler_source": str(args.scaler_source.resolve()) if args.scaler_source else None,
        "splits": split_metadata,
    }
    (args.output / CACHE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[DONE] Prepared cache: {args.output}")


if __name__ == "__main__":
    main()
