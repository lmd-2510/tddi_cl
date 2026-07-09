#!/usr/bin/env python3
"""Leakage and overlap audit for the DDI2025-CIL benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_LABEL_COL = "class"
DRUG_A_COL = "drugid-drug_a"
DRUG_B_COL = "drugid-drug_b"
META_COLS = [
    "drugid-drug_a",
    "drugid-drug_b",
    "drugname-drug_a",
    "drugname-drug_b",
    "drugsmiles-drug_a",
    "drugsmiles-drug_b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check ordered/unordered pair leakage across DDI2025 splits."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--label-col",
        default=DEFAULT_LABEL_COL,
        help=f"Label column name. Default: {DEFAULT_LABEL_COL}",
    )
    return parser.parse_args()


def assert_parquet_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix != ".parquet":
        raise ValueError(f"Expected Parquet input, got: {path}")


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns.tolist()]
    rows = frame.astype(object).where(pd.notna(frame), "").values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_identity_frame(split_name: str, path: Path, label_col: str) -> pd.DataFrame:
    print(f"[leakage] Reading {split_name} identity columns from {path}...", flush=True)
    table = pq.read_table(path, columns=[DRUG_A_COL, DRUG_B_COL, label_col])
    frame = table.to_pandas()
    frame[DRUG_A_COL] = frame[DRUG_A_COL].fillna("<MISSING>").astype(str)
    frame[DRUG_B_COL] = frame[DRUG_B_COL].fillna("<MISSING>").astype(str)
    frame[label_col] = frame[label_col].astype("int64")
    frame["split"] = split_name
    frame["ordered_pair_key"] = frame[DRUG_A_COL] + "|||" + frame[DRUG_B_COL]
    mins = frame[[DRUG_A_COL, DRUG_B_COL]].min(axis=1)
    maxs = frame[[DRUG_A_COL, DRUG_B_COL]].max(axis=1)
    frame["unordered_a"] = mins
    frame["unordered_b"] = maxs
    frame["unordered_pair_key"] = mins + "|||" + maxs
    print(
        f"[leakage] {split_name}: rows={frame.shape[0]}, "
        f"unique_ordered_pairs={frame['ordered_pair_key'].nunique()}, "
        f"unique_unordered_pairs={frame['unordered_pair_key'].nunique()}",
        flush=True,
    )
    return frame


def schema_feature_leakage_summary(path: Path, label_col: str) -> dict[str, Any]:
    schema = pq.ParquetFile(path).schema_arrow
    columns = list(schema.names)
    feature_cols = [col for col in columns if col not in META_COLS + [label_col]]
    return {
        "label_present": label_col in columns,
        "meta_present": {col: col in columns for col in META_COLS},
        "class_in_feature_cols": label_col in feature_cols,
        "meta_in_feature_cols": [col for col in META_COLS if col in feature_cols],
        "feature_count": len(feature_cols),
    }


def build_ordered_group(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    grouped = (
        frame.groupby([DRUG_A_COL, DRUG_B_COL, "ordered_pair_key"], as_index=False)
        .agg(
            row_count=("ordered_pair_key", "size"),
            class_nunique=(label_col, "nunique"),
            classes=(label_col, lambda s: ",".join(map(str, sorted(s.unique())))),
        )
    )
    return grouped


def build_unordered_group(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["unordered_a", "unordered_b", "unordered_pair_key"], as_index=False)
        .agg(
            row_count=("unordered_pair_key", "size"),
            class_nunique=(label_col, "nunique"),
            classes=(label_col, lambda s: ",".join(map(str, sorted(s.unique())))),
            ordered_variants=("ordered_pair_key", "nunique"),
        )
    )
    return grouped


def build_duplicate_report(split_name: str, frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    exact_duplicate_rows = int(frame.duplicated(subset=[DRUG_A_COL, DRUG_B_COL, label_col]).sum())

    ordered_group = build_ordered_group(frame, label_col)
    unordered_group = build_unordered_group(frame, label_col)
    ordered_dupes = ordered_group[ordered_group["row_count"] > 1]
    unordered_dupes = unordered_group[unordered_group["row_count"] > 1]

    rows = [
        {
            "split": split_name,
            "duplicate_type": "exact_duplicate_identity_rows",
            "duplicate_groups": None,
            "duplicate_rows": exact_duplicate_rows,
            "notes": f"Duplicates on ({DRUG_A_COL}, {DRUG_B_COL}, {label_col}).",
        },
        {
            "split": split_name,
            "duplicate_type": "ordered_pair_duplicates",
            "duplicate_groups": int(ordered_dupes.shape[0]),
            "duplicate_rows": int((ordered_dupes["row_count"] - 1).sum()),
            "notes": "Repeated ordered pairs within the split.",
        },
        {
            "split": split_name,
            "duplicate_type": "unordered_pair_duplicates",
            "duplicate_groups": int(unordered_dupes.shape[0]),
            "duplicate_rows": int((unordered_dupes["row_count"] - 1).sum()),
            "notes": "Repeated unordered pairs within the split, including A-B/B-A.",
        },
    ]
    return pd.DataFrame(rows)


def build_multilabel_report(split_name: str, frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["unordered_a", "unordered_b", "unordered_pair_key"], as_index=False)
        .agg(
            row_count=("unordered_pair_key", "size"),
            class_nunique=(label_col, "nunique"),
            classes=(label_col, lambda s: ",".join(map(str, sorted(s.unique())))),
            ordered_variants=("ordered_pair_key", "nunique"),
        )
    )
    grouped = grouped[grouped["class_nunique"] > 1].copy()
    grouped.insert(0, "split", split_name)
    return grouped.sort_values(["class_nunique", "row_count"], ascending=[False, False]).reset_index(
        drop=True
    )


def build_ordered_overlap_detail(
    left_name: str,
    left_frame: pd.DataFrame,
    right_name: str,
    right_frame: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    left_group = build_ordered_group(left_frame, label_col).rename(
        columns={
            DRUG_A_COL: "drugid_drug_a_left",
            DRUG_B_COL: "drugid_drug_b_left",
            "row_count": "row_count_left",
            "class_nunique": "class_nunique_left",
            "classes": "classes_left",
        }
    )
    right_group = build_ordered_group(right_frame, label_col).rename(
        columns={
            DRUG_A_COL: "drugid_drug_a_right",
            DRUG_B_COL: "drugid_drug_b_right",
            "row_count": "row_count_right",
            "class_nunique": "class_nunique_right",
            "classes": "classes_right",
        }
    )
    merged = left_group.merge(
        right_group,
        on=["ordered_pair_key"],
        how="inner",
    )
    merged.insert(0, "split_left", left_name)
    merged.insert(1, "split_right", right_name)
    return merged


def build_unordered_overlap_detail(
    left_name: str,
    left_frame: pd.DataFrame,
    right_name: str,
    right_frame: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    left_group = build_unordered_group(left_frame, label_col).rename(
        columns={
            "row_count": "row_count_left",
            "class_nunique": "class_nunique_left",
            "classes": "classes_left",
            "ordered_variants": "ordered_variants_left",
        }
    )
    right_group = build_unordered_group(right_frame, label_col).rename(
        columns={
            "row_count": "row_count_right",
            "class_nunique": "class_nunique_right",
            "classes": "classes_right",
            "ordered_variants": "ordered_variants_right",
        }
    )
    merged = left_group.merge(
        right_group,
        on=["unordered_a", "unordered_b", "unordered_pair_key"],
        how="inner",
    )
    merged.insert(0, "split_left", left_name)
    merged.insert(1, "split_right", right_name)
    return merged


def build_reverse_overlap_detail(
    left_name: str,
    left_frame: pd.DataFrame,
    right_name: str,
    right_frame: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    left_group = build_ordered_group(left_frame, label_col).rename(
        columns={
            DRUG_A_COL: "drugid_drug_a_left",
            DRUG_B_COL: "drugid_drug_b_left",
            "ordered_pair_key": "ordered_pair_key_left",
            "row_count": "row_count_left",
            "class_nunique": "class_nunique_left",
            "classes": "classes_left",
        }
    )
    right_group = build_ordered_group(right_frame, label_col).rename(
        columns={
            DRUG_A_COL: "drugid_drug_a_right",
            DRUG_B_COL: "drugid_drug_b_right",
            "ordered_pair_key": "ordered_pair_key_right",
            "row_count": "row_count_right",
            "class_nunique": "class_nunique_right",
            "classes": "classes_right",
        }
    )
    left_group["unordered_a"] = left_group[["drugid_drug_a_left", "drugid_drug_b_left"]].min(axis=1)
    left_group["unordered_b"] = left_group[["drugid_drug_a_left", "drugid_drug_b_left"]].max(axis=1)
    right_group["unordered_a"] = right_group[["drugid_drug_a_right", "drugid_drug_b_right"]].min(axis=1)
    right_group["unordered_b"] = right_group[["drugid_drug_a_right", "drugid_drug_b_right"]].max(axis=1)

    merged = left_group.merge(
        right_group,
        on=["unordered_a", "unordered_b"],
        how="inner",
    )
    same_order = (
        (merged["drugid_drug_a_left"] == merged["drugid_drug_a_right"])
        & (merged["drugid_drug_b_left"] == merged["drugid_drug_b_right"])
    )
    reverse_only = merged[~same_order].copy()
    reverse_only.insert(0, "split_left", left_name)
    reverse_only.insert(1, "split_right", right_name)
    reverse_only["unordered_pair_key"] = (
        reverse_only["unordered_a"] + "|||" + reverse_only["unordered_b"]
    )
    reverse_only = reverse_only.drop_duplicates(
        subset=[
            "split_left",
            "split_right",
            "ordered_pair_key_left",
            "ordered_pair_key_right",
        ]
    ).reset_index(drop=True)
    return reverse_only


def build_drug_overlap_report(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(split_frames)
    rows: list[dict[str, Any]] = []
    drug_sets = {
        name: set(pd.concat([frame[DRUG_A_COL], frame[DRUG_B_COL]], ignore_index=True).unique().tolist())
        for name, frame in split_frames.items()
    }
    for idx, left_name in enumerate(names):
        for right_name in names[idx + 1 :]:
            left_set = drug_sets[left_name]
            right_set = drug_sets[right_name]
            overlap = left_set & right_set
            union = left_set | right_set
            rows.append(
                {
                    "split_left": left_name,
                    "split_right": right_name,
                    "unique_drugs_left": len(left_set),
                    "unique_drugs_right": len(right_set),
                    "overlap_drug_count": len(overlap),
                    "overlap_fraction_left": len(overlap) / len(left_set) if left_set else 0.0,
                    "overlap_fraction_right": len(overlap) / len(right_set) if right_set else 0.0,
                    "jaccard": len(overlap) / len(union) if union else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_pair_overlap_summary(
    split_names: list[str],
    ordered_details: pd.DataFrame,
    unordered_details: pd.DataFrame,
    reverse_details: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, left_name in enumerate(split_names):
        for right_name in split_names[idx + 1 :]:
            ordered_count = int(
                ordered_details[
                    (ordered_details["split_left"] == left_name)
                    & (ordered_details["split_right"] == right_name)
                ].shape[0]
            )
            unordered_count = int(
                unordered_details[
                    (unordered_details["split_left"] == left_name)
                    & (unordered_details["split_right"] == right_name)
                ].shape[0]
            )
            reverse_count = int(
                reverse_details[
                    (reverse_details["split_left"] == left_name)
                    & (reverse_details["split_right"] == right_name)
                ]["unordered_pair_key"].nunique()
            )
            rows.append(
                {
                    "split_left": left_name,
                    "split_right": right_name,
                    "ordered_overlap_count": ordered_count,
                    "unordered_overlap_count": unordered_count,
                    "reverse_overlap_unordered_count": reverse_count,
                }
            )
    return pd.DataFrame(rows).sort_values(["split_left", "split_right"]).reset_index(drop=True)


def write_summary(
    outpath: Path,
    feature_leakage: dict[str, Any],
    duplicate_report: pd.DataFrame,
    pair_overlap_report: pd.DataFrame,
    drug_overlap_report: pd.DataFrame,
    reverse_pair_overlap_report: pd.DataFrame,
    multilabel_pair_report: pd.DataFrame,
) -> None:
    duplicate_pivot = duplicate_report.copy()
    pair_preview = pair_overlap_report.copy()
    drug_preview = drug_overlap_report.copy()
    reverse_preview = reverse_pair_overlap_report.head(10)
    multilabel_preview = multilabel_pair_report.head(10)

    lines = [
        "# Leakage Summary",
        "",
        "## Scope",
        "",
        "- This audit checks ordered pair overlap, unordered/reverse overlap, drug-level overlap, duplicate pair patterns, and multilabel unordered pairs.",
        f"- Exact duplicate row check is performed on the identity subset `({DRUG_A_COL}, {DRUG_B_COL}, {DEFAULT_LABEL_COL})` for memory-aware auditing.",
        "",
        "## Feature Leakage Guard",
        "",
        f"- `class` present in schema: `{feature_leakage['label_present']}`",
        f"- `class` accidentally in `feature_cols`: `{feature_leakage['class_in_feature_cols']}`",
        f"- meta columns accidentally in `feature_cols`: `{feature_leakage['meta_in_feature_cols']}`",
        f"- explicit feature count from schema: `{feature_leakage['feature_count']}`",
        "",
        "## Pair Overlap Summary",
        "",
        dataframe_to_markdown_table(pair_preview),
        "",
        "## Drug Overlap Summary",
        "",
        dataframe_to_markdown_table(drug_preview),
        "",
        "## Duplicate Pair Summary",
        "",
        dataframe_to_markdown_table(duplicate_pivot),
        "",
        "## Reverse Overlap Preview",
        "",
        dataframe_to_markdown_table(reverse_preview) if not reverse_preview.empty else "_No reverse overlap rows found._",
        "",
        "## Multilabel Unordered Pair Preview",
        "",
        dataframe_to_markdown_table(multilabel_preview)
        if not multilabel_preview.empty
        else "_No unordered pair mapped to multiple classes within a split._",
        "",
        "## Interpretation Notes",
        "",
        "- Ordered overlap indicates the exact same directional pair appears across splits.",
        "- Unordered/reverse overlap indicates the same drug pair may reappear after swapping A/B.",
        "- If train/test overlap is non-trivial, static supervised metrics can be optimistic.",
        "- This audit does not silently redefine the original split; pair-disjoint variants remain future ablations.",
        "",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in [args.train, args.validation, args.test]:
        assert_parquet_path(path)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    split_frames = {
        "train": load_identity_frame("train", args.train, args.label_col),
        "validation": load_identity_frame("validation", args.validation, args.label_col),
        "test": load_identity_frame("test", args.test, args.label_col),
    }

    feature_leakage = schema_feature_leakage_summary(args.train, args.label_col)

    duplicate_report = pd.concat(
        [
            build_duplicate_report(name, frame, args.label_col)
            for name, frame in split_frames.items()
        ],
        ignore_index=True,
    )
    multilabel_pair_report = pd.concat(
        [
            build_multilabel_report(name, frame, args.label_col)
            for name, frame in split_frames.items()
        ],
        ignore_index=True,
    )

    ordered_details: list[pd.DataFrame] = []
    unordered_details: list[pd.DataFrame] = []
    reverse_details: list[pd.DataFrame] = []
    names = list(split_frames)
    for idx, left_name in enumerate(names):
        for right_name in names[idx + 1 :]:
            print(
                f"[leakage] Comparing {left_name} vs {right_name}...",
                flush=True,
            )
            ordered_details.append(
                build_ordered_overlap_detail(
                    left_name, split_frames[left_name], right_name, split_frames[right_name], args.label_col
                )
            )
            unordered_details.append(
                build_unordered_overlap_detail(
                    left_name, split_frames[left_name], right_name, split_frames[right_name], args.label_col
                )
            )
            reverse_details.append(
                build_reverse_overlap_detail(
                    left_name, split_frames[left_name], right_name, split_frames[right_name], args.label_col
                )
            )

    ordered_pair_overlap_report = pd.concat(ordered_details, ignore_index=True)
    unordered_pair_overlap_report = pd.concat(unordered_details, ignore_index=True)
    reverse_pair_overlap_report = pd.concat(reverse_details, ignore_index=True)
    pair_overlap_report = build_pair_overlap_summary(
        split_names=names,
        ordered_details=ordered_pair_overlap_report,
        unordered_details=unordered_pair_overlap_report,
        reverse_details=reverse_pair_overlap_report,
    )
    drug_overlap_report = build_drug_overlap_report(split_frames)

    pair_overlap_report.to_csv(outdir / "pair_overlap_report.csv", index=False)
    ordered_pair_overlap_report.to_csv(outdir / "ordered_pair_overlap_report.csv", index=False)
    reverse_pair_overlap_report.to_csv(outdir / "reverse_pair_overlap_report.csv", index=False)
    drug_overlap_report.to_csv(outdir / "drug_overlap_report.csv", index=False)
    duplicate_report.to_csv(outdir / "duplicate_pair_report.csv", index=False)
    multilabel_pair_report.to_csv(outdir / "multilabel_pair_report.csv", index=False)
    (outdir / "feature_leakage_schema_check.json").write_text(
        json.dumps(feature_leakage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(
        outpath=outdir / "leakage_summary.md",
        feature_leakage=feature_leakage,
        duplicate_report=duplicate_report,
        pair_overlap_report=pair_overlap_report,
        drug_overlap_report=drug_overlap_report,
        reverse_pair_overlap_report=reverse_pair_overlap_report,
        multilabel_pair_report=multilabel_pair_report,
    )
    print(f"[done] Wrote leakage artifacts to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
