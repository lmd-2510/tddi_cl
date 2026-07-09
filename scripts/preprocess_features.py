#!/usr/bin/env python3
"""Build preprocessing artifacts for DDI2025-CIL."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_LABEL_COL = "class"
DEFAULT_META_COLS = [
    "drugid-drug_a",
    "drugid-drug_b",
    "drugname-drug_a",
    "drugname-drug_b",
    "drugsmiles-drug_a",
    "drugsmiles-drug_b",
]
DEFAULT_BATCH_SIZE = 1024
DEFAULT_ROBUST_SAMPLE_SIZE = 10000


@dataclass
class SplitScanSummary:
    split: str
    rows_scanned: int
    nonfinite_rows: int
    columns_with_nonfinite: int
    null_count_total: int
    nan_count_total: int
    inf_count_total: int
    partial_scan: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit preprocessing artifacts from DDI2025 Parquet features."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--feature-cols", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--scaler",
        choices=["standard", "robust"],
        default="standard",
        help="Scaler type to fit on train split only. Default: standard.",
    )
    parser.add_argument(
        "--impute-strategy",
        choices=["zero", "mean"],
        default="zero",
        help="Non-finite imputation strategy. Default: zero.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Arrow row batch size. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit for batches per split.",
    )
    parser.add_argument(
        "--label-col",
        default=DEFAULT_LABEL_COL,
        help=f"Label column name. Default: {DEFAULT_LABEL_COL}",
    )
    parser.add_argument(
        "--meta-cols",
        nargs="+",
        default=DEFAULT_META_COLS,
        help="Explicit metadata columns excluded from feature space.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for robust scaler sampling.",
    )
    parser.add_argument(
        "--robust-sample-size",
        type=int,
        default=DEFAULT_ROBUST_SAMPLE_SIZE,
        help=f"Maximum sampled train rows for robust quantiles. Default: {DEFAULT_ROBUST_SAMPLE_SIZE}.",
    )
    return parser.parse_args()


def assert_parquet_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix != ".parquet":
        raise ValueError(f"Expected Parquet input, got: {path}")


def load_feature_columns(path: Path) -> list[str]:
    columns = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise ValueError(f"Feature column file must be a JSON string list: {path}")
    return columns


def validate_feature_columns(
    parquet_path: Path,
    feature_columns: list[str],
    meta_cols: list[str],
    label_col: str,
) -> dict[str, Any]:
    schema = pq.ParquetFile(parquet_path).schema_arrow
    columns = list(schema.names)
    missing = [col for col in feature_columns if col not in columns]
    if missing:
        raise ValueError(f"Feature columns missing from train schema: {missing[:10]}")
    if label_col in feature_columns:
        raise ValueError("Label column must not appear in feature columns.")
    leaked_meta = [col for col in meta_cols if col in feature_columns]
    if leaked_meta:
        raise ValueError(f"Metadata columns leaked into feature space: {leaked_meta}")
    return {
        "feature_count": len(feature_columns),
        "train_schema_column_count": len(columns),
        "label_in_schema": label_col in columns,
        "meta_present": {col: col in columns for col in meta_cols},
    }


def iterate_batches(path: Path, columns: list[str], batch_size: int, max_batches: int | None):
    parquet_file = pq.ParquetFile(path)
    for batch_idx, batch in enumerate(parquet_file.iter_batches(columns=columns, batch_size=batch_size), start=1):
        yield batch_idx, batch, parquet_file.metadata.num_rows
        if max_batches is not None and batch_idx >= max_batches:
            break


def batch_to_numpy(batch: pa.RecordBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    null_counts = np.array([batch.column(i).null_count for i in range(batch.num_columns)], dtype=np.int64)
    table = pa.Table.from_batches([batch])
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    values = df.to_numpy(dtype=np.float64, copy=False)
    nan_mask = np.isnan(values)
    inf_mask = np.isinf(values)
    arrow_null_mask = np.zeros_like(nan_mask, dtype=bool)
    if null_counts.sum() > 0:
        for idx, count in enumerate(null_counts):
            if count:
                arrow_null_mask[:, idx] = df.iloc[:, idx].isna().to_numpy()
    return values, arrow_null_mask, inf_mask


def summarize_split_nonfinite(
    split: str,
    path: Path,
    feature_columns: list[str],
    batch_size: int,
    max_batches: int | None,
) -> SplitScanSummary:
    rows_scanned = 0
    null_total = 0
    nan_total = 0
    inf_total = 0
    nonfinite_row_count = 0
    nonfinite_col_flags = np.zeros(len(feature_columns), dtype=bool)

    print(f"[preprocess] Scanning non-finite stats for {split}...", flush=True)
    total_rows = pq.ParquetFile(path).metadata.num_rows
    for batch_idx, batch, _ in iterate_batches(path, feature_columns, batch_size, max_batches):
        values, arrow_null_mask, inf_mask = batch_to_numpy(batch)
        nan_mask = np.isnan(values) & ~arrow_null_mask
        nonfinite_mask = arrow_null_mask | nan_mask | inf_mask

        rows_scanned += batch.num_rows
        null_total += int(arrow_null_mask.sum())
        nan_total += int(nan_mask.sum())
        inf_total += int(inf_mask.sum())
        nonfinite_row_count += int(nonfinite_mask.any(axis=1).sum())
        nonfinite_col_flags |= nonfinite_mask.any(axis=0)

        if batch_idx == 1 or batch_idx % 25 == 0:
            print(
                f"[preprocess] {split}: batches={batch_idx} rows_scanned={rows_scanned}/{total_rows}",
                flush=True,
            )

    return SplitScanSummary(
        split=split,
        rows_scanned=rows_scanned,
        nonfinite_rows=nonfinite_row_count,
        columns_with_nonfinite=int(nonfinite_col_flags.sum()),
        null_count_total=null_total,
        nan_count_total=nan_total,
        inf_count_total=inf_total,
        partial_scan=rows_scanned != total_rows,
    )


def fit_standard_scaler(
    train_path: Path,
    feature_columns: list[str],
    batch_size: int,
    max_batches: int | None,
    impute_strategy: str,
) -> dict[str, Any]:
    num_features = len(feature_columns)
    total_rows = 0
    total_sum = np.zeros(num_features, dtype=np.float64)
    total_sumsq = np.zeros(num_features, dtype=np.float64)
    finite_sum = np.zeros(num_features, dtype=np.float64)
    finite_count = np.zeros(num_features, dtype=np.int64)

    print("[preprocess] Fitting standard scaler on train split only...", flush=True)
    parquet_rows = pq.ParquetFile(train_path).metadata.num_rows

    if impute_strategy == "mean":
        # Pass 1: estimate finite means.
        for batch_idx, batch, _ in iterate_batches(train_path, feature_columns, batch_size, max_batches):
            values, arrow_null_mask, inf_mask = batch_to_numpy(batch)
            nan_mask = np.isnan(values) & ~arrow_null_mask
            finite_mask = ~(arrow_null_mask | nan_mask | inf_mask)
            safe_values = np.where(finite_mask, values, 0.0)
            finite_sum += safe_values.sum(axis=0)
            finite_count += finite_mask.sum(axis=0).astype(np.int64)
            total_rows += batch.num_rows
            if batch_idx == 1 or batch_idx % 25 == 0:
                print(
                    f"[preprocess] standard(mean) pass1: batches={batch_idx} rows_scanned={total_rows}/{parquet_rows}",
                    flush=True,
                )

        impute_values = np.divide(
            finite_sum,
            finite_count,
            out=np.zeros(num_features, dtype=np.float64),
            where=finite_count > 0,
        )
        total_rows_second = 0
        for batch_idx, batch, _ in iterate_batches(train_path, feature_columns, batch_size, max_batches):
            values, arrow_null_mask, inf_mask = batch_to_numpy(batch)
            nan_mask = np.isnan(values) & ~arrow_null_mask
            nonfinite_mask = arrow_null_mask | nan_mask | inf_mask
            filled = np.where(nonfinite_mask, impute_values, values)
            total_sum += filled.sum(axis=0)
            total_sumsq += np.square(filled).sum(axis=0)
            total_rows_second += batch.num_rows
            if batch_idx == 1 or batch_idx % 25 == 0:
                print(
                    f"[preprocess] standard(mean) pass2: batches={batch_idx} rows_scanned={total_rows_second}/{parquet_rows}",
                    flush=True,
                )
        total_rows = total_rows_second
    else:
        impute_values = np.zeros(num_features, dtype=np.float64)
        for batch_idx, batch, _ in iterate_batches(train_path, feature_columns, batch_size, max_batches):
            values, arrow_null_mask, inf_mask = batch_to_numpy(batch)
            nan_mask = np.isnan(values) & ~arrow_null_mask
            nonfinite_mask = arrow_null_mask | nan_mask | inf_mask
            filled = np.where(nonfinite_mask, 0.0, values)
            total_sum += filled.sum(axis=0)
            total_sumsq += np.square(filled).sum(axis=0)
            total_rows += batch.num_rows
            if batch_idx == 1 or batch_idx % 25 == 0:
                print(
                    f"[preprocess] standard(zero) fit: batches={batch_idx} rows_scanned={total_rows}/{parquet_rows}",
                    flush=True,
                )

    mean = total_sum / total_rows
    var = np.maximum(total_sumsq / total_rows - np.square(mean), 0.0)
    scale = np.sqrt(var)
    scale = np.where(scale == 0.0, 1.0, scale)
    return {
        "scaler_type": "standard",
        "impute_strategy": impute_strategy,
        "rows_fitted": int(total_rows),
        "mean": mean,
        "var": var,
        "scale": scale,
        "impute_values": impute_values,
    }


def fit_robust_scaler(
    train_path: Path,
    feature_columns: list[str],
    batch_size: int,
    max_batches: int | None,
    robust_sample_size: int,
    random_seed: int,
    impute_strategy: str,
) -> dict[str, Any]:
    if impute_strategy != "zero":
        raise ValueError("Robust scaler currently supports only --impute-strategy zero.")

    rng = np.random.default_rng(random_seed)
    reservoir: np.ndarray | None = None
    seen_rows = 0
    parquet_rows = pq.ParquetFile(train_path).metadata.num_rows

    print(
        f"[preprocess] Fitting robust scaler on sampled train rows "
        f"(sample_size={robust_sample_size}, random_seed={random_seed})...",
        flush=True,
    )
    for batch_idx, batch, _ in iterate_batches(train_path, feature_columns, batch_size, max_batches):
        values, arrow_null_mask, inf_mask = batch_to_numpy(batch)
        nan_mask = np.isnan(values) & ~arrow_null_mask
        nonfinite_mask = arrow_null_mask | nan_mask | inf_mask
        filled = np.where(nonfinite_mask, 0.0, values).astype(np.float32, copy=False)

        if reservoir is None:
            take = min(robust_sample_size, filled.shape[0])
            reservoir = filled[:take].copy()
            seen_rows = filled.shape[0]
        else:
            for row in filled:
                seen_rows += 1
                if reservoir.shape[0] < robust_sample_size:
                    reservoir = np.vstack([reservoir, row[None, :]])
                else:
                    j = rng.integers(0, seen_rows)
                    if j < robust_sample_size:
                        reservoir[j] = row

        if batch_idx == 1 or batch_idx % 25 == 0:
            current_size = 0 if reservoir is None else reservoir.shape[0]
            print(
                f"[preprocess] robust fit: batches={batch_idx} rows_seen={seen_rows}/{parquet_rows} "
                f"reservoir={current_size}",
                flush=True,
            )

    if reservoir is None or reservoir.shape[0] == 0:
        raise ValueError("Robust scaler could not collect any sample rows.")

    center = np.median(reservoir, axis=0).astype(np.float64)
    q1 = np.quantile(reservoir, 0.25, axis=0).astype(np.float64)
    q3 = np.quantile(reservoir, 0.75, axis=0).astype(np.float64)
    scale = q3 - q1
    scale = np.where(scale == 0.0, 1.0, scale)
    return {
        "scaler_type": "robust",
        "impute_strategy": impute_strategy,
        "rows_fitted": int(seen_rows),
        "center": center,
        "q1": q1,
        "q3": q3,
        "scale": scale,
        "impute_values": np.zeros(len(feature_columns), dtype=np.float64),
        "sample_size_used": int(reservoir.shape[0]),
        "sample_size_target": int(robust_sample_size),
        "approximate": True,
    }


def scaler_payload_to_config(payload: dict[str, Any], feature_count: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        "scaler_type": payload["scaler_type"],
        "impute_strategy": payload["impute_strategy"],
        "rows_fitted": payload["rows_fitted"],
        "feature_count": feature_count,
    }
    if payload["scaler_type"] == "robust":
        config["sample_size_used"] = payload["sample_size_used"]
        config["sample_size_target"] = payload["sample_size_target"]
        config["approximate"] = payload["approximate"]
    return config


def save_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def write_report(
    outpath: Path,
    feature_columns: list[str],
    meta_cols: list[str],
    scaler_config: dict[str, Any],
    schema_validation: dict[str, Any],
    split_summaries: list[SplitScanSummary],
    max_batches: int | None,
) -> None:
    summary_frame = pd.DataFrame(
        [
            {
                "split": summary.split,
                "rows_scanned": summary.rows_scanned,
                "nonfinite_rows": summary.nonfinite_rows,
                "columns_with_nonfinite": summary.columns_with_nonfinite,
                "null_count_total": summary.null_count_total,
                "nan_count_total": summary.nan_count_total,
                "inf_count_total": summary.inf_count_total,
                "partial_scan": summary.partial_scan,
            }
            for summary in split_summaries
        ]
    )

    notes = [
        "# Preprocessing Report",
        "",
        "## Configuration",
        "",
        f"- scaler: `{scaler_config['scaler_type']}`",
        f"- impute_strategy: `{scaler_config['impute_strategy']}`",
        f"- rows_fitted_on_train: `{scaler_config['rows_fitted']}`",
        f"- feature_count: `{len(feature_columns)}`",
        f"- partial_scan_mode: `{max_batches is not None}`",
        "",
        "## Explicit Feature Policy",
        "",
        f"- label column excluded: `{DEFAULT_LABEL_COL}`",
        f"- meta columns excluded: `{', '.join(meta_cols)}`",
        f"- feature columns source: `feature_columns.json`",
        "",
        "## Schema Validation",
        "",
        f"- train schema column count: `{schema_validation['train_schema_column_count']}`",
        f"- label present in train schema: `{schema_validation['label_in_schema']}`",
        f"- feature count validated: `{schema_validation['feature_count']}`",
        "",
        "## Split Scan Summary",
        "",
        dataframe_to_markdown_table(summary_frame),
        "",
        "## Notes",
        "",
        "- Scaler is fit on train split only.",
        "- This script does not materialize full transformed train/validation/test matrices by default.",
        "- `standard` scaler is exact under the chosen imputation strategy.",
        "- `robust` scaler uses a deterministic sampled approximation to avoid loading the full matrix into RAM.",
        "",
    ]
    outpath.write_text("\n".join(notes), encoding="utf-8")


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


def main() -> None:
    args = parse_args()
    for path in [args.train, args.validation, args.test]:
        assert_parquet_path(path)

    feature_columns = load_feature_columns(args.feature_cols)
    schema_validation = validate_feature_columns(
        parquet_path=args.train,
        feature_columns=feature_columns,
        meta_cols=args.meta_cols,
        label_col=args.label_col,
    )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"[preprocess] feature_count={len(feature_columns)} scaler={args.scaler} "
        f"impute_strategy={args.impute_strategy}",
        flush=True,
    )

    split_summaries = [
        summarize_split_nonfinite("train", args.train, feature_columns, args.batch_size, args.max_batches),
        summarize_split_nonfinite("validation", args.validation, feature_columns, args.batch_size, args.max_batches),
        summarize_split_nonfinite("test", args.test, feature_columns, args.batch_size, args.max_batches),
    ]

    if args.scaler == "standard":
        scaler_payload = fit_standard_scaler(
            train_path=args.train,
            feature_columns=feature_columns,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            impute_strategy=args.impute_strategy,
        )
    else:
        scaler_payload = fit_robust_scaler(
            train_path=args.train,
            feature_columns=feature_columns,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            robust_sample_size=args.robust_sample_size,
            random_seed=args.random_seed,
            impute_strategy=args.impute_strategy,
        )

    scaler_config = scaler_payload_to_config(scaler_payload, len(feature_columns))
    save_pickle(outdir / "scaler.pkl", scaler_payload)
    (outdir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / "meta_columns.json").write_text(
        json.dumps(args.meta_cols, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / "scaler_config.json").write_text(
        json.dumps(scaler_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(
        outpath=outdir / "preprocessing_report.md",
        feature_columns=feature_columns,
        meta_cols=args.meta_cols,
        scaler_config=scaler_config,
        schema_validation=schema_validation,
        split_summaries=split_summaries,
        max_batches=args.max_batches,
    )
    print(f"[done] Wrote preprocessing artifacts to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
