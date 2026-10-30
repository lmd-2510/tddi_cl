#!/usr/bin/env python3
"""Read-only audit of persisted development folds against their source Parquets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ddi_dataset import DEFAULT_LABEL_COL
from src.data.sample_identity import DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN, build_stable_sample_ids
from src.data.stratified_folds import describe_fold_source, fold_file_sha256, load_fold_artifact

CLASS_COLUMNS = ["raw_class_id", "development_count", "fold_0", "fold_1", "fold_2",
                 "max_minus_min", "member_0_train", "member_1_train", "member_2_train"]
MEMBER_COLUMNS = ["member_id", "validation_fold", "train_folds", "train_count",
                  "validation_count", "test_count", "train_fraction", "validation_fraction",
                  "train_class_count", "validation_class_count", "train_validation_overlap"]


def _read_source(path: Path, *, labels: bool):
    """Project identity/labels only; test labels are deliberately not read."""
    columns = [DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN]
    table = pq.read_table(path, columns=columns + ([DEFAULT_LABEL_COL] if labels else []))
    pairs = {key: table[key].to_numpy() for key in columns}
    ids = build_stable_sample_ids(pairs, table.num_rows)
    # Canonical pairs are diagnostic only; do not change the frozen row-level split.
    canonical = np.asarray([json.dumps(sorted(pair), ensure_ascii=False)
                            for pair in zip(pairs[columns[0]], pairs[columns[1]], strict=True)])
    raw_labels = None
    if labels:
        column = table[DEFAULT_LABEL_COL]
        if column.null_count or not pa.types.is_integer(column.type):
            raise ValueError(f"{path}: labels must be non-null integers.")
        raw_labels = column.cast(pa.int64(), safe=True).to_numpy()
    return ids, raw_labels, canonical


def _write_reports(outdir: Path, report: dict, classes: list, members: list) -> None:
    """New directory only; stage each file and publish JSON last as commit marker."""
    outdir.mkdir(parents=True, exist_ok=False)
    lines = ["# Development fold audit", "", f"Status: **{report['status']}**", "",
             "This is a data audit, not approval to start training or a buffer decision.", "",
             "## Checks", ""]
    lines += [f"- {item['name']}: {item['status']} — {item.get('detail', '')}"
              for item in report["checks"]]
    lines += ["", "## Member views", "",
              "| Member | Train folds | Validation fold | Train rows | Validation rows | Test rows |",
              "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {m['member_id']} | {m['train_folds']} | {m['validation_fold']} | "
              f"{m['train_count']} | {m['validation_count']} | {m['test_count']} |" for m in members]
    lines += ["", "## Warnings / scope", "",
              "- Required leakage check uses ordered drug pairs, matching the artifact schema.",
              "- Unordered-pair overlap is diagnostic; no drug-disjoint guarantee or automatic repair.",
              "- Missing downstream checks after a failure are not passes."]
    lines += [f"- {warning}" for warning in report["warnings"]]
    for filename, value in (("fold_class_counts.csv", (CLASS_COLUMNS, classes)),
                            ("fold_member_views.csv", (MEMBER_COLUMNS, members)),
                            ("fold_audit.md", "\n".join(lines) + "\n"),
                            ("fold_audit.json", report)):
        staging = outdir / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            with staging.open("x", encoding="utf-8", newline="") as handle:
                if filename.endswith(".csv"):
                    columns, rows = value
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
                elif filename.endswith(".json"):
                    json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
                    handle.write("\n")
                else:
                    handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staging, outdir / filename)
        finally:
            staging.unlink(missing_ok=True)


def audit_development_folds(*, assignments, manifest, train, validation, test, outdir) -> dict:
    """Never rebuild/repair folds. Invalid inputs produce FAILED reports when writable."""
    outdir = Path(outdir)
    if outdir.exists() or outdir.is_symlink():
        raise FileExistsError(f"Audit output already exists; refusing overwrite: {outdir}")
    paths = {"train": Path(train), "validation": Path(validation), "test": Path(test)}
    artifacts = {"assignments": Path(assignments), "manifest": Path(manifest)}
    report = {"schema_version": 1, "artifact_kind": "ddi_cil_development_fold_audit",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "FAILED",
              "checks": [], "warnings": [], "sources": {}, "artifact_sources": {},
              "leakage_policy": "ordered_pair_required_unordered_pair_diagnostic"}
    class_rows, member_rows = [], []
    stage = "artifact_integrity_structure_coverage_disjointness_mapping"

    def check(name, condition, detail=""):
        report["checks"].append({"name": name, "status": "PASS" if condition else "FAIL",
                                 "detail": detail})

    try:
        for name, path in artifacts.items():
            report["artifact_sources"][name] = {"path": str(path.resolve()), "sha256": fold_file_sha256(path)}
        artifact = load_fold_artifact(artifacts["assignments"], artifacts["manifest"])
        check(stage, True, "Valid hash/schema; each source row and sample ID occurs once; mapping 0/1/2.")
        table, metadata = artifact.assignments, artifact.manifest
        report["fold_seed"] = metadata["fold_seed"]
        report["member_to_validation_fold"] = metadata["member_to_validation_fold"]
        report["development_count"] = table.num_rows
        identity = {}
        for split, path in paths.items():
            stage = f"source_{split}"
            print(f"[fold-audit] Verifying source: {split}", flush=True)
            actual = describe_fold_source(path)
            report["sources"][split] = actual
            expected = metadata["sources"][split]
            if any(actual[key] != expected[key] for key in ("sha256", "row_count")):
                raise ValueError(f"Source hash/count mismatch: {split}")
            identity[split] = _read_source(path, labels=split != "test")
            check(stage, True, "Hash/count match; identity columns valid with unique IDs.")

        stage = "source_row_identity_and_labels"
        splits = table["source_split"].to_numpy()
        indices = table["source_row_index"].to_numpy()
        ids, labels, folds = (table[key].to_numpy() for key in ("sample_id", "raw_class_id", "fold_id"))
        canonical = np.empty(len(ids), dtype=object)
        for split in ("train", "validation"):
            mask = splits == split
            source_ids, source_labels, pairs = identity[split]
            positions = indices[mask]
            check(f"{split}_row_identity", np.array_equal(ids[mask], source_ids[positions]))
            check(f"{split}_row_labels", np.array_equal(labels[mask], source_labels[positions]))
            canonical[mask] = pairs[positions]
        dev_ids = np.concatenate([identity[s][0] for s in ("train", "validation")])
        check("development_unique_ids", len(np.unique(dev_ids)) == len(dev_ids))
        overlap = np.intersect1d(dev_ids, identity["test"][0])
        check("development_test_ordered_overlap", len(overlap) == 0, f"overlap_count={len(overlap)}")
        report["ordered_test_overlap_count"] = len(overlap)
        unordered_overlap = np.intersect1d(canonical, identity["test"][2])
        crossfold = sum(len(np.intersect1d(canonical[folds == a], canonical[folds == b]))
                        for a, b in ((0, 1), (0, 2), (1, 2)))
        report["unordered_test_overlap_count"] = len(unordered_overlap)
        report["unordered_crossfold_pair_intersections"] = crossfold
        if len(unordered_overlap) or crossfold:
            report["warnings"].append(
                f"Unordered pairs: development/test={len(unordered_overlap)}, "
                f"sum of pairwise fold intersections={crossfold}. Review group-aware policy at Prompt 4.")

        stage = "stratification_and_member_views"
        summary = {str(f): {"row_count": int(np.sum(folds == f)), "class_counts": {}} for f in range(3)}
        for label in np.unique(labels):
            counts = np.bincount(folds[labels == label], minlength=3)
            row = {"raw_class_id": int(label), "development_count": int(counts.sum()),
                   "max_minus_min": int(counts.max() - counts.min())}
            for f in range(3):
                row[f"fold_{f}"] = int(counts[f])
                row[f"member_{f}_train"] = int(counts.sum() - counts[f])
                if counts[f]:
                    summary[str(f)]["class_counts"][str(int(label))] = int(counts[f])
            class_rows.append(row)
        check("class_stratification", all(r["max_minus_min"] <= 1 and min(r[f"fold_{f}"] for f in range(3)) > 0
                                           for r in class_rows), "Each class present in all folds; per-class count gap <= 1.")
        sizes = [summary[str(f)]["row_count"] for f in range(3)]
        check("fold_size_balance", max(sizes) - min(sizes) <= 1)
        if "fold_summary" in metadata:
            check("manifest_fold_summary", summary == metadata["fold_summary"])
        for member in range(3):
            held_out = metadata["member_to_validation_fold"][str(member)]
            val_mask = folds == held_out
            member_rows.append({"member_id": member, "validation_fold": held_out,
                                "train_folds": "+".join(str(f) for f in range(3) if f != held_out),
                                "train_count": int(np.sum(~val_mask)), "validation_count": int(np.sum(val_mask)),
                                "test_count": len(identity["test"][0]), "train_fraction": float(np.mean(~val_mask)),
                                "validation_fraction": float(np.mean(val_mask)),
                                "train_class_count": len(np.unique(labels[~val_mask])),
                                "validation_class_count": len(np.unique(labels[val_mask])),
                                "train_validation_overlap": len(np.intersect1d(ids[~val_mask], ids[val_mask]))})
        check("member_train_validation_disjoint", all(m["train_validation_overlap"] == 0 for m in member_rows))
        stage = "inputs_unchanged_during_audit"
        for name, path in {**paths, **artifacts}.items():
            expected = report["sources"].get(name, report["artifact_sources"].get(name))
            check(f"{name}_unchanged", fold_file_sha256(path) == expected["sha256"])
        if all(item["status"] == "PASS" for item in report["checks"]):
            report["status"] = "PASSED"
    except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as error:
        check(stage, False, f"{type(error).__name__}: {error}")
        report["warnings"].append("Audit stopped at invalid input; later checks were not run.")
    report["class_counts"] = class_rows
    report["member_views"] = member_rows
    _write_reports(outdir, report, class_rows, member_rows)
    print(f"[fold-audit] {report['status']}: {outdir}", flush=True)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assignments", "manifest", "train", "validation", "test", "outdir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_development_folds(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(2, f"[fold-audit] ERROR: {error}\n")
    if report["status"] != "PASSED":
        parser.exit(1, "[fold-audit] FAILED: inspect fold_audit.json/md before proceeding.\n")


if __name__ == "__main__":
    main()
