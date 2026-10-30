#!/usr/bin/env python3
"""Create a 3-fold stratified sample-assignment table and provenance manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Sequence

# Support `python scripts/build_stratified_3fold_assignments.py` from a server checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ddi_dataset import DEFAULT_LABEL_COL
from src.data.sample_identity import (
    DRUG_ID_A_COLUMN,
    DRUG_ID_B_COLUMN,
    build_stable_sample_ids,
)
from src.data.stratified_folds import (
    FOLD_ASSIGNMENT_SCHEMA,
    FoldArtifact,
    build_stratified_fold_assignments,
    describe_fold_source,
    fold_file_sha256,
    save_fold_artifact,
)


def _read_identity(path: Path, *, include_labels: bool) -> tuple[np.ndarray, np.ndarray | None]:
    columns = [DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN]
    if include_labels:
        columns.append(DEFAULT_LABEL_COL)
    table = pq.read_table(path, columns=columns)
    ids = build_stable_sample_ids(
        {column: table[column].to_numpy() for column in columns[:2]}, table.num_rows
    )
    labels = None
    if include_labels:
        column = table[DEFAULT_LABEL_COL]
        if column.null_count or not pa.types.is_integer(column.type):
            raise ValueError(f"{path}: class labels must be non-null integers.")
        labels = column.cast(pa.int64(), safe=True).to_numpy()
    return ids, labels


def build_development_folds(
    *,
    train: str | Path,
    validation: str | Path,
    test: str | Path,
    outdir: str | Path,
    fold_seed: int,
    n_splits: int = 3,
    creation_command: str = "",
) -> FoldArtifact:
    """Build a sidecar only; ordered-pair test overlap is a hard error.

    Source hashes are checked before and after reading so artifacts cannot be
    published from inputs changed during the build. Test labels/features never
    participate in assigning development rows. Group-aware policy is separate.
    """
    if type(n_splits) is not int or n_splits != 3:
        raise ValueError("n_splits must equal 3.")
    if type(fold_seed) is not int or not 0 <= fold_seed <= 2**32 - 1:
        raise ValueError("fold_seed must be an integer in [0, 2**32-1].")
    outdir = Path(outdir)
    if outdir.exists() or outdir.is_symlink():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {outdir}")
    paths = {"train": Path(train), "validation": Path(validation), "test": Path(test)}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} Parquet: {path}")
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected .parquet input for {split}: {path}")
    if len({path.resolve() for path in paths.values()}) != 3:
        raise ValueError("train, validation and test must be distinct source files.")

    sources = {}
    identity = {}
    for split, path in paths.items():
        print(f"[fold-build] Hashing {split}: {path}", flush=True)
        sources[split] = describe_fold_source(path)
        if sources[split]["row_count"] == 0:
            raise ValueError(f"Empty source split: {split}")
        print(f"[fold-build] Reading {split} identity columns", flush=True)
        identity[split] = _read_identity(path, include_labels=split != "test")

    ids = np.concatenate([identity[split][0] for split in ("train", "validation")])
    if np.unique(ids).size != ids.size:
        raise ValueError("Development train+validation contains duplicate sample IDs.")
    overlap = np.intersect1d(ids, identity["test"][0], assume_unique=True)
    if overlap.size:
        raise ValueError(
            f"Development/test ordered-pair overlap: {overlap.size} sample IDs; "
            f"examples={overlap[:3].tolist()}"
        )
    print(f"[fold-build] Assigning {ids.size} development rows with fold_seed={fold_seed}", flush=True)
    # The sole splitting algorithm lives in src/data/stratified_folds.py.
    lookup = build_stratified_fold_assignments(
        (paths["train"], paths["validation"]), fold_count=n_splits, seed=fold_seed,
    )
    if not np.array_equal(lookup.sorted_sample_ids, np.sort(ids)):
        raise ValueError("Development identity changed while building folds.")
    assignments = pa.Table.from_pydict({
        "source_split": np.concatenate([
            np.full(len(identity[split][0]), split) for split in ("train", "validation")
        ]),
        "source_row_index": np.concatenate([
            np.arange(len(identity[split][0]), dtype=np.int64) for split in ("train", "validation")
        ]),
        "sample_id": ids,
        "raw_class_id": np.concatenate([identity[split][1] for split in ("train", "validation")]),
        "fold_id": lookup.lookup(ids),
    }, schema=FOLD_ASSIGNMENT_SCHEMA)
    for split, path in paths.items():
        print(f"[fold-build] Verifying unchanged source: {split}", flush=True)
        if fold_file_sha256(path) != sources[split]["sha256"]:
            raise ValueError(f"Source changed during fold build: {split}")
    result = save_fold_artifact(
        outdir, assignments, sources=sources, fold_seed=fold_seed,
        creation_command=creation_command,
    )
    for fold, summary in result.manifest["fold_summary"].items():
        print(f"[fold-build] fold={fold} rows={summary['row_count']}", flush=True)
    print(f"[done] Saved fold_assignments.parquet and fold_manifest.json under {outdir}", flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--n-splits", type=int, choices=[3], default=3)
    parser.add_argument("--fold-seed", required=True, type=int)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        build_development_folds(
            **vars(args),
            creation_command=shlex.join([sys.executable, str(Path(__file__).resolve()), *arguments]),
        )
    except (OSError, ValueError, TypeError, pa.ArrowException) as error:
        parser.exit(2, f"[fold-build] ERROR: {error}\n")


if __name__ == "__main__":
    main()
