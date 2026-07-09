#!/usr/bin/env python3
"""Analyze class distribution for the DDI2025-CIL benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_LABEL_COL = "class"
DEFAULT_THRESHOLDS = [5, 10, 20, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze class distribution from DDI2025 Parquet splits."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--figdir",
        type=Path,
        default=None,
        help="Figure output directory. Defaults to sibling outputs/figures.",
    )
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


def load_class_counts(split_name: str, path: Path, label_col: str) -> pd.DataFrame:
    print(f"[class-dist] Reading {split_name} labels from {path}...", flush=True)
    table = pq.read_table(path, columns=[label_col])
    series = pd.Series(table.column(0).to_pandas(), copy=False)
    counts = series.value_counts().sort_index()
    frame = pd.DataFrame(
        {
            "class_id": counts.index.astype(int),
            "count": counts.values.astype(int),
            "split": split_name,
        }
    )
    print(
        f"[class-dist] {split_name}: num_rows={int(counts.sum())}, "
        f"num_classes={frame.shape[0]}, min_count={int(frame['count'].min())}, "
        f"max_count={int(frame['count'].max())}",
        flush=True,
    )
    return frame


def compute_summary_stats(counts: pd.Series) -> dict[str, Any]:
    min_count = int(counts.min())
    max_count = int(counts.max())
    return {
        "num_classes": int(counts.shape[0]),
        "min_count": min_count,
        "max_count": max_count,
        "median_count": float(counts.median()),
        "mean_count": float(counts.mean()),
        "num_classes_<=5": int((counts <= 5).sum()),
        "num_classes_<=10": int((counts <= 10).sum()),
        "num_classes_<=20": int((counts <= 20).sum()),
        "num_classes_<=50": int((counts <= 50).sum()),
        "imbalance_ratio": float(max_count / min_count) if min_count > 0 else float("inf"),
    }


def build_rare_class_table(
    train_counts: pd.DataFrame,
    validation_counts: pd.DataFrame,
    test_counts: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        train_counts[["class_id", "count"]]
        .rename(columns={"count": "train_count"})
        .merge(
            validation_counts[["class_id", "count"]].rename(columns={"count": "validation_count"}),
            on="class_id",
            how="outer",
        )
        .merge(
            test_counts[["class_id", "count"]].rename(columns={"count": "test_count"}),
            on="class_id",
            how="outer",
        )
        .fillna(0)
    )
    for column in ["train_count", "validation_count", "test_count"]:
        merged[column] = merged[column].astype(int)
    for threshold in DEFAULT_THRESHOLDS:
        merged[f"train_le_{threshold}"] = merged["train_count"] <= threshold
    merged["rare_bucket_train"] = np.select(
        [
            merged["train_count"] <= 5,
            merged["train_count"] <= 10,
            merged["train_count"] <= 20,
            merged["train_count"] <= 50,
        ],
        [
            "<=5",
            "<=10",
            "<=20",
            "<=50",
        ],
        default=">50",
    )
    return merged.sort_values(["train_count", "class_id"], ascending=[True, True]).reset_index(drop=True)


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


def save_distribution_plot(
    train_counts: pd.DataFrame,
    output_path: Path,
    *,
    log_scale: bool,
) -> None:
    ordered = train_counts.sort_values("count", ascending=False).reset_index(drop=True)
    x = np.arange(1, len(ordered) + 1)
    y = ordered["count"].to_numpy()

    plt.figure(figsize=(12, 5))
    plt.bar(x, y, width=1.0, color="#1f77b4", edgecolor="none")
    plt.xlabel("Class rank by train frequency")
    plt.ylabel("Train sample count")
    plt.title(
        "DDI2025 train class distribution"
        + (" (log scale)" if log_scale else "")
    )
    if log_scale:
        plt.yscale("log")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_summary(
    outpath: Path,
    train_counts: pd.DataFrame,
    validation_counts: pd.DataFrame,
    test_counts: pd.DataFrame,
    rare_classes: pd.DataFrame,
) -> None:
    train_stats = compute_summary_stats(train_counts["count"])
    validation_stats = compute_summary_stats(validation_counts["count"])
    test_stats = compute_summary_stats(test_counts["count"])

    top_train = train_counts.sort_values("count", ascending=False).head(10).reset_index(drop=True)
    top_validation = (
        validation_counts.sort_values("count", ascending=False).head(10).reset_index(drop=True)
    )
    top_test = test_counts.sort_values("count", ascending=False).head(10).reset_index(drop=True)
    rare_preview = rare_classes.head(15)

    lines = [
        "# Class Distribution Summary",
        "",
        "## Key Findings",
        "",
        "- Dataset exhibits strong long-tail behavior across all three splits.",
        "- Train split remains the reference split for task design and rare-class policy.",
        "- Main benchmark should keep all 178 classes; rare classes are reported separately, not removed.",
        "",
        "## Split-Level Summary",
        "",
        dataframe_to_markdown_table(
            pd.DataFrame(
                [
                    {"split": "train", **train_stats},
                    {"split": "validation", **validation_stats},
                    {"split": "test", **test_stats},
                ]
            )
        ),
        "",
        "## Top Classes",
        "",
        "### Train Top-10",
        "",
        dataframe_to_markdown_table(top_train),
        "",
        "### Validation Top-10",
        "",
        dataframe_to_markdown_table(top_validation),
        "",
        "### Test Top-10",
        "",
        dataframe_to_markdown_table(top_test),
        "",
        "## Rare-Class Policy",
        "",
        "- Keep the full 178-class benchmark as the main setting.",
        "- Report rare-class metrics separately for thresholds `<=5`, `<=10`, and `<=20`.",
        "- Do not overclaim per-class F1 for classes with only a few samples.",
        "- Optional `min-count filtered benchmark` remains an ablation, not the main protocol.",
        "",
        "## Rare-Class Preview",
        "",
        dataframe_to_markdown_table(rare_preview),
        "",
        "## Interpretation Notes",
        "",
        "- Large head classes can dominate optimization and hide tail failures if only overall accuracy is reported.",
        "- Macro-F1 and Balanced Accuracy are critical for this dataset.",
        "- Replay memory should be class-balanced rather than globally random.",
        "",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def infer_figdir(outdir: Path, figdir: Path | None) -> Path:
    if figdir is not None:
        return figdir
    if outdir.parent.name == "outputs":
        return outdir.parent / "figures"
    return outdir / "figures"


def main() -> None:
    args = parse_args()
    for path in [args.train, args.validation, args.test]:
        assert_parquet_path(path)

    outdir = args.outdir
    figdir = infer_figdir(outdir, args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    train_counts = load_class_counts("train", args.train, args.label_col)
    validation_counts = load_class_counts("validation", args.validation, args.label_col)
    test_counts = load_class_counts("test", args.test, args.label_col)

    train_counts.to_csv(outdir / "class_counts_train.csv", index=False)
    validation_counts.to_csv(outdir / "class_counts_validation.csv", index=False)
    test_counts.to_csv(outdir / "class_counts_test.csv", index=False)

    rare_classes = build_rare_class_table(train_counts, validation_counts, test_counts)
    rare_classes.to_csv(outdir / "rare_classes.csv", index=False)

    save_distribution_plot(
        train_counts=train_counts,
        output_path=figdir / "class_distribution_train.png",
        log_scale=False,
    )
    save_distribution_plot(
        train_counts=train_counts,
        output_path=figdir / "class_distribution_logscale.png",
        log_scale=True,
    )

    write_summary(
        outpath=outdir / "class_distribution_summary.md",
        train_counts=train_counts,
        validation_counts=validation_counts,
        test_counts=test_counts,
        rare_classes=rare_classes,
    )

    print(f"[done] Wrote class distribution artifacts to: {outdir}", flush=True)
    print(f"[done] Wrote figures to: {figdir}", flush=True)


if __name__ == "__main__":
    main()
