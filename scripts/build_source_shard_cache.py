#!/usr/bin/env python
"""Build the immutable raw task/fold source cache in one Parquet pass."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ddi_dataset import load_feature_columns
from src.data.source_shard_cache import build_source_shard_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--fold-manifest", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--feature-cols", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_source_shard_cache(
        root=args.root,
        assignments=args.assignments,
        fold_manifest=args.fold_manifest,
        train=args.train,
        validation=args.validation,
        test=args.test,
        task_file=args.task_file,
        feature_columns=load_feature_columns(args.feature_cols),
        batch_size=args.batch_size,
    )
    print(f"[OK] Raw source-shard cache: {args.root}")
    print(f"[OK] Tasks: {len(manifest['tasks'])}; shards: {len(manifest['shards'])}")


if __name__ == "__main__":
    main()
