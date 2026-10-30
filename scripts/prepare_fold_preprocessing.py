#!/usr/bin/env python3
"""Prepare raw or task-0-frozen preprocessing; never launch model training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ddi_dataset import load_feature_columns, prepare_development_fold_context
from src.data.fold_preprocessing import POLICIES, prepare_fold_preprocessing, save_fold_preprocessing


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--feature-cols", type=Path, required=True)
    parser.add_argument("--member-id", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--validation-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--experiment-seed", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--outdir", type=Path, required=True, help="New namespace only; never overwrite.")
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(f"Output namespace already exists: {args.outdir}")
    if args.member_id != args.validation_fold or args.batch_size <= 0:
        parser.error("member-id must equal validation-fold and batch-size must be positive")
    columns = load_feature_columns(args.feature_cols)
    print("Validating frozen partition/source hashes (no training)...", flush=True)
    context = prepare_development_fold_context(
        args.assignments, args.manifest,
        source_paths={s: getattr(args, s) for s in ("train", "validation", "test")},
        expected_metadata={"fold_seed": args.fold_seed},
    )
    print(f"Scanning task-0 training rows: member={args.member_id} policy={args.policy} nonfinite=error", flush=True)
    artifact = prepare_fold_preprocessing(
        context, task_file=args.task_file, feature_columns=columns,
        member_id=args.member_id, validation_fold=args.validation_fold,
        policy=args.policy, experiment_seed=args.experiment_seed, batch_size=args.batch_size,
    )
    path = save_fold_preprocessing(artifact, args.outdir)
    metadata = artifact.metadata
    print(f"Saved: {path}\nreference_rows={metadata['provenance']['reference_selection']['row_count']} "
          f"fitted={metadata['fit'] is not None}\npayload_sha256={metadata['payload_sha256']}", flush=True)


if __name__ == "__main__":
    main()
