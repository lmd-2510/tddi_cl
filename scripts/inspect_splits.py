#!/usr/bin/env python3
"""Audit Parquet splits for the DDI2025-CIL benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_META_COLS = [
    "drugid-drug_a",
    "drugid-drug_b",
    "drugname-drug_a",
    "drugname-drug_b",
    "drugsmiles-drug_a",
    "drugsmiles-drug_b",
]
DEFAULT_LABEL_COL = "class"
NEAR_CONSTANT_ZERO_RATIO = 0.999


@dataclass
class SplitAudit:
    name: str
    path: Path
    rows: int
    cols: int
    column_names: list[str]
    duplicate_columns: list[str]
    feature_columns: list[str]
    non_numeric_feature_columns: list[str]
    class_dtype: str
    class_ids: list[int]
    class_counts: dict[int, int]
    meta_positions: dict[str, int | None]
    label_position: int | None
    all_meta_present: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect DDI2025 Parquet splits and write audit artifacts."
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
    parser.add_argument(
        "--meta-cols",
        nargs="+",
        default=DEFAULT_META_COLS,
        help="Explicit metadata columns to exclude from feature space.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Row batch size for feature scanning. Default: 1024.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit for row batches per split.",
    )
    return parser.parse_args()


def assert_parquet_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix != ".parquet":
        raise ValueError(f"Expected a Parquet file, got: {path}")


def is_numeric_arrow_type(dtype: pa.DataType) -> bool:
    return any(
        (
            pa.types.is_integer(dtype),
            pa.types.is_floating(dtype),
            pa.types.is_decimal(dtype),
            pa.types.is_boolean(dtype),
        )
    )


def find_duplicates(values: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    duplicates: list[str] = []
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        if counts[value] == 2:
            duplicates.append(value)
    return duplicates


def inspect_split(
    name: str,
    path: Path,
    meta_cols: list[str],
    label_col: str,
) -> SplitAudit:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    column_names = list(schema.names)
    duplicate_columns = find_duplicates(column_names)
    feature_columns = [c for c in column_names if c not in meta_cols + [label_col]]
    non_numeric_feature_columns = [
        column
        for column in feature_columns
        if not is_numeric_arrow_type(schema.field(column).type)
    ]

    class_table = parquet_file.read(columns=[label_col])
    class_series = pd.Series(class_table.column(0).to_pandas(), copy=False)
    class_counts = class_series.value_counts().sort_index()
    class_ids = [int(value) for value in class_counts.index.tolist()]

    meta_positions = {
        column: (column_names.index(column) + 1) if column in column_names else None
        for column in meta_cols
    }
    label_position = (column_names.index(label_col) + 1) if label_col in column_names else None

    return SplitAudit(
        name=name,
        path=path,
        rows=parquet_file.metadata.num_rows,
        cols=parquet_file.metadata.num_columns,
        column_names=column_names,
        duplicate_columns=duplicate_columns,
        feature_columns=feature_columns,
        non_numeric_feature_columns=non_numeric_feature_columns,
        class_dtype=str(schema.field(label_col).type),
        class_ids=class_ids,
        class_counts={int(k): int(v) for k, v in class_counts.to_dict().items()},
        meta_positions=meta_positions,
        label_position=label_position,
        all_meta_present=all(value is not None for value in meta_positions.values()),
    )


def compute_feature_stats(
    split: SplitAudit,
    batch_size: int,
    max_batches: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if split.non_numeric_feature_columns:
        raise TypeError(
            "Non-numeric feature columns detected in explicit feature space: "
            + ", ".join(split.non_numeric_feature_columns[:10])
        )

    parquet_file = pq.ParquetFile(split.path)
    columns = split.feature_columns
    num_columns = len(columns)
    null_counts = np.zeros(num_columns, dtype=np.int64)
    nan_counts = np.zeros(num_columns, dtype=np.int64)
    inf_counts = np.zeros(num_columns, dtype=np.int64)
    non_null_counts = np.zeros(num_columns, dtype=np.int64)
    zero_counts = np.zeros(num_columns, dtype=np.int64)
    min_values = np.full(num_columns, np.inf, dtype=np.float64)
    max_values = np.full(num_columns, -np.inf, dtype=np.float64)
    scanned_rows = 0
    scanned_batches = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        if scanned_batches == 0:
            print(
                f"[{split.name}] Scanning feature batches "
                f"(rows={split.rows}, features={num_columns}, batch_size={batch_size})...",
                flush=True,
            )

        arrow_null_counts = np.array(
            [batch.column(idx).null_count for idx in range(num_columns)],
            dtype=np.int64,
        )
        batch_table = pa.Table.from_batches([batch])
        batch_df = batch_table.to_pandas(split_blocks=True, self_destruct=True)
        batch_values = batch_df.to_numpy(dtype=np.float64, copy=False)

        batch_null_mask = np.isnan(batch_values)
        batch_inf_mask = np.isinf(batch_values)
        batch_valid_mask = ~(batch_null_mask | batch_inf_mask)

        batch_nan_counts = batch_null_mask.sum(axis=0).astype(np.int64) - arrow_null_counts
        null_counts += arrow_null_counts
        nan_counts += batch_nan_counts
        inf_counts += batch_inf_mask.sum(axis=0).astype(np.int64)
        non_null_counts += batch_valid_mask.sum(axis=0).astype(np.int64)
        zero_counts += ((batch_values == 0.0) & batch_valid_mask).sum(axis=0).astype(np.int64)

        safe_min = np.where(batch_valid_mask, batch_values, np.inf)
        safe_max = np.where(batch_valid_mask, batch_values, -np.inf)
        min_values = np.minimum(min_values, safe_min.min(axis=0))
        max_values = np.maximum(max_values, safe_max.max(axis=0))

        scanned_rows += batch.num_rows
        scanned_batches += 1
        if scanned_batches == 1 or scanned_batches % 25 == 0:
            print(
                f"[{split.name}] batches={scanned_batches} scanned_rows={scanned_rows}/{split.rows}",
                flush=True,
            )
        if max_batches is not None and scanned_batches >= max_batches:
            break

    null_ratios = null_counts / scanned_rows if scanned_rows else np.zeros(num_columns)
    inf_ratios = inf_counts / scanned_rows if scanned_rows else np.zeros(num_columns)
    nan_ratios = nan_counts / scanned_rows if scanned_rows else np.zeros(num_columns)
    zero_ratios = np.divide(
        zero_counts,
        non_null_counts,
        out=np.zeros(num_columns, dtype=np.float64),
        where=non_null_counts > 0,
    )
    no_valid_mask = non_null_counts == 0
    min_values = np.where(no_valid_mask, np.nan, min_values)
    max_values = np.where(no_valid_mask, np.nan, max_values)

    missing_inf_report = pd.DataFrame(
        {
            "split": split.name,
            "column": columns,
            "scanned_rows": scanned_rows,
            "null_count": null_counts,
            "nan_count": nan_counts,
            "inf_count": inf_counts,
            "null_ratio": null_ratios,
            "nan_ratio": nan_ratios,
            "inf_ratio": inf_ratios,
            "non_finite_count": nan_counts + inf_counts,
        }
    )

    statuses: list[str] = []
    notes: list[str] = []
    for idx in range(num_columns):
        status = "ok"
        note = ""
        if non_null_counts[idx] == 0:
            status = "all_invalid_or_missing"
            note = "No finite values observed in scanned batches."
        elif min_values[idx] == max_values[idx]:
            status = "constant"
            note = "Finite min and max are identical."
        elif zero_ratios[idx] >= NEAR_CONSTANT_ZERO_RATIO:
            status = "zero_heavy_candidate"
            note = (
                f"Finite zero ratio >= {NEAR_CONSTANT_ZERO_RATIO:.3f}. "
                "Treat as near-constant candidate, not confirmed constant."
            )
        statuses.append(status)
        notes.append(note)

    constant_report = pd.DataFrame(
        {
            "split": split.name,
            "column": columns,
            "status": statuses,
            "scanned_rows": scanned_rows,
            "non_null_count": non_null_counts,
            "zero_count": zero_counts,
            "zero_ratio": zero_ratios,
            "min_value": min_values,
            "max_value": max_values,
            "notes": notes,
        }
    )
    constant_report = constant_report[constant_report["status"] != "ok"].reset_index(drop=True)

    scan_meta = {
        "scanned_rows": scanned_rows,
        "expected_rows": split.rows,
        "scanned_batches": scanned_batches,
        "partial_scan": scanned_rows != split.rows,
    }
    return missing_inf_report, constant_report, scan_meta


def build_row_col_counts(splits: list[SplitAudit]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": split.name,
                "path": str(split.path),
                "rows": split.rows,
                "cols": split.cols,
                "feature_count": len(split.feature_columns),
                "class_dtype": split.class_dtype,
                "class_nunique": len(split.class_ids),
            }
            for split in splits
        ]
    )


def build_column_positions(splits: list[SplitAudit], meta_cols: list[str], label_col: str) -> pd.DataFrame:
    tracked_cols = meta_cols + [label_col]
    rows: list[dict[str, Any]] = []
    for split in splits:
        for column in tracked_cols:
            position = split.column_names.index(column) + 1 if column in split.column_names else None
            rows.append(
                {
                    "split": split.name,
                    "column": column,
                    "position_1based": position,
                    "is_meta": column in meta_cols,
                    "is_label": column == label_col,
                }
            )
    return pd.DataFrame(rows)


def build_class_id_summary(splits: list[SplitAudit], train_split_name: str = "train") -> pd.DataFrame:
    all_class_ids = sorted({class_id for split in splits for class_id in split.class_ids})
    train_split = next(split for split in splits if split.name == train_split_name)
    global_class_map = {class_id: idx for idx, class_id in enumerate(train_split.class_ids)}

    rows: list[dict[str, Any]] = []
    for class_id in all_class_ids:
        row: dict[str, Any] = {
            "class_id": class_id,
            "global_index": global_class_map.get(class_id),
        }
        for split in splits:
            row[f"count_{split.name}"] = split.class_counts.get(class_id, 0)
            row[f"present_{split.name}"] = class_id in split.class_counts
        rows.append(row)
    return pd.DataFrame(rows).sort_values("global_index", kind="stable").reset_index(drop=True)


def build_meta_columns_check(splits: list[SplitAudit], meta_cols: list[str], label_col: str) -> dict[str, Any]:
    return {
        split.name: {
            "all_meta_present": split.all_meta_present,
            "label_present": split.label_position is not None,
            "meta_positions": split.meta_positions,
            "label_position": split.label_position,
            "feature_count": len(split.feature_columns),
            "class_in_feature_cols": label_col in split.feature_columns,
            "meta_in_feature_cols": [column for column in meta_cols if column in split.feature_columns],
        }
        for split in splits
    }


def build_schema_summary(
    splits: list[SplitAudit],
    meta_cols: list[str],
    label_col: str,
    scan_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    canonical_columns = splits[0].column_names
    schema_consistent = all(split.column_names == canonical_columns for split in splits[1:])
    train_split = next(split for split in splits if split.name == "train")
    checks = {
        "all_have_3787_columns": all(split.cols == 3787 for split in splits),
        "train_rows_match_expected": train_split.rows == 520841,
        "validation_rows_match_expected": next(split for split in splits if split.name == "validation").rows == 173614,
        "test_rows_match_expected": next(split for split in splits if split.name == "test").rows == 173614,
        "all_have_label": all(split.label_position is not None for split in splits),
        "all_have_meta_cols": all(split.all_meta_present for split in splits),
        "feature_count_is_3780": all(len(split.feature_columns) == 3780 for split in splits),
        "class_id_space_from_train_has_178_labels": len(train_split.class_ids) == 178,
        "class_dtype_integer_like": all("int" in split.class_dtype for split in splits),
        "no_duplicate_column_names": all(not split.duplicate_columns for split in splits),
        "schema_consistent_across_splits": schema_consistent,
    }
    return {
        "label_col": label_col,
        "meta_cols": meta_cols,
        "expected_feature_count": 3780,
        "splits": {
            split.name: {
                "path": str(split.path),
                "rows": split.rows,
                "cols": split.cols,
                "feature_count": len(split.feature_columns),
                "duplicate_columns": split.duplicate_columns,
                "non_numeric_feature_columns": split.non_numeric_feature_columns,
                "class_dtype": split.class_dtype,
                "class_nunique": len(split.class_ids),
                "class_ids": split.class_ids,
                "meta_positions": split.meta_positions,
                "label_position": split.label_position,
                "scan_meta": scan_meta[split.name],
            }
            for split in splits
        },
        "checks": checks,
    }


def format_check(passed: bool) -> str:
    return "[x]" if passed else "[ ]"


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns.tolist()]
    rows = frame.astype(object).where(pd.notna(frame), "").values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        rendered = [str(value) for value in row]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_audit_summary(
    outpath: Path,
    splits: list[SplitAudit],
    schema_summary: dict[str, Any],
    class_id_summary: pd.DataFrame,
    missing_inf_report: pd.DataFrame,
    constant_report: pd.DataFrame,
) -> None:
    checks = schema_summary["checks"]
    train_split = next(split for split in splits if split.name == "train")
    partial_scan = any(schema_summary["splits"][split.name]["scan_meta"]["partial_scan"] for split in splits)

    lines = [
        "# Audit Summary",
        "",
        "## Key Facts",
        "",
        f"- Canonical label column: `{DEFAULT_LABEL_COL}`",
        f"- Canonical meta columns: `{', '.join(DEFAULT_META_COLS)}`",
        f"- Canonical feature count from explicit exclusion: `{len(train_split.feature_columns)}`",
        f"- Train class cardinality from `nunique`: `{len(train_split.class_ids)}`",
        f"- Partial scan mode: `{partial_scan}`",
        "",
        "## Checklist",
        "",
        f"- {format_check(checks['all_have_3787_columns'])} Mỗi Parquet có 3787 cột",
        f"- {format_check(checks['train_rows_match_expected'])} train có 520841 dòng",
        f"- {format_check(checks['validation_rows_match_expected'])} validation có 173614 dòng",
        f"- {format_check(checks['test_rows_match_expected'])} test có 173614 dòng",
        f"- {format_check(checks['all_have_label'])} `class` tồn tại trong cả 3 split",
        f"- {format_check(checks['class_dtype_integer_like'])} `class` là integer-like",
        f"- {format_check(checks['class_id_space_from_train_has_178_labels'])} Có 178 class từ `nunique(train['class'])`",
        f"- {format_check(checks['all_have_meta_cols'])} 6 meta columns tồn tại đầy đủ",
        f"- {format_check(checks['feature_count_is_3780'])} `feature_cols` có đúng 3780 cột",
        f"- {format_check(checks['no_duplicate_column_names'])} Không có duplicate column names",
        f"- {format_check(checks['schema_consistent_across_splits'])} Column order nhất quán giữa các split",
        "",
        "## Class ID Policy",
        "",
        "- `num_classes` phải lấy từ `sorted(unique_classes)` hoặc `nunique`, không dùng `max(class)+1`.",
        f"- `global_class_map.json` được xây từ train split với `{len(train_split.class_ids)}` class IDs thực.",
        "",
        "## Missing / Inf Overview",
        "",
        f"- Số cột có `non_finite_count > 0`: `{int((missing_inf_report['non_finite_count'] > 0).sum())}`",
        f"- Số cột có `null_count > 0`: `{int((missing_inf_report['null_count'] > 0).sum())}`",
        "",
        "## Constant / Near-Constant Overview",
        "",
        f"- Số cột `constant`: `{int((constant_report['status'] == 'constant').sum())}`",
        (
            f"- Số cột `zero_heavy_candidate`: "
            f"`{int((constant_report['status'] == 'zero_heavy_candidate').sum())}`"
        ),
        "",
        "## Notes",
        "",
        "- Script này là Parquet-first audit; không dùng `awk -F,` để suy luận schema CSV.",
        "- `zero_heavy_candidate` là cờ near-constant bảo thủ, không đồng nghĩa cột vô dụng.",
        "",
        "## Output Artifacts",
        "",
        "- `schema_summary.json`",
        "- `row_col_counts.csv`",
        "- `column_positions.csv`",
        "- `meta_columns_check.json`",
        "- `feature_columns.json`",
        "- `global_class_map.json`",
        "- `class_id_summary.csv`",
        "- `missing_inf_report.csv`",
        "- `constant_columns.csv`",
        "",
        "## Sample Class IDs",
        "",
        dataframe_to_markdown_table(class_id_summary.head(10)),
        "",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in [args.train, args.validation, args.test]:
        assert_parquet_path(path)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    splits = [
        inspect_split("train", args.train, args.meta_cols, args.label_col),
        inspect_split("validation", args.validation, args.meta_cols, args.label_col),
        inspect_split("test", args.test, args.meta_cols, args.label_col),
    ]
    for split in splits:
        print(
            f"[inspect] {split.name}: rows={split.rows}, cols={split.cols}, "
            f"feature_count={len(split.feature_columns)}, class_nunique={len(split.class_ids)}",
            flush=True,
        )

    train_split = next(split for split in splits if split.name == "train")
    feature_columns = train_split.feature_columns
    global_class_map = {
        str(class_id): idx for idx, class_id in enumerate(train_split.class_ids)
    }

    missing_reports: list[pd.DataFrame] = []
    constant_reports: list[pd.DataFrame] = []
    scan_meta: dict[str, dict[str, Any]] = {}
    for split in splits:
        missing_report, constant_report, split_scan_meta = compute_feature_stats(
            split=split,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
        missing_reports.append(missing_report)
        constant_reports.append(constant_report)
        scan_meta[split.name] = split_scan_meta

    row_col_counts = build_row_col_counts(splits)
    column_positions = build_column_positions(splits, args.meta_cols, args.label_col)
    class_id_summary = build_class_id_summary(splits)
    meta_columns_check = build_meta_columns_check(splits, args.meta_cols, args.label_col)
    schema_summary = build_schema_summary(splits, args.meta_cols, args.label_col, scan_meta)
    missing_inf_report = pd.concat(missing_reports, ignore_index=True)
    constant_report = pd.concat(constant_reports, ignore_index=True)

    row_col_counts.to_csv(outdir / "row_col_counts.csv", index=False)
    column_positions.to_csv(outdir / "column_positions.csv", index=False)
    class_id_summary.to_csv(outdir / "class_id_summary.csv", index=False)
    missing_inf_report.to_csv(outdir / "missing_inf_report.csv", index=False)
    constant_report.to_csv(outdir / "constant_columns.csv", index=False)

    write_json(outdir / "schema_summary.json", schema_summary)
    write_json(outdir / "meta_columns_check.json", meta_columns_check)
    write_json(outdir / "feature_columns.json", feature_columns)
    write_json(outdir / "global_class_map.json", global_class_map)
    write_audit_summary(
        outpath=outdir / "audit_summary.md",
        splits=splits,
        schema_summary=schema_summary,
        class_id_summary=class_id_summary,
        missing_inf_report=missing_inf_report,
        constant_report=constant_report,
    )

    print(f"[done] Wrote audit artifacts to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
