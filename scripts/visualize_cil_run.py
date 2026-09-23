"""Create five diagnostic figures for a completed CIL run.

The command reads only run artifacts.  It never changes checkpoints or model
state.  Prediction figures use member prediction NPZ files when available;
buffer figures use buffer_audit.json.  Missing optional artifacts produce a
visible placeholder figure and a warning in visualization_manifest.json.

Example:
    python scripts/visualize_cil_run.py \
        --run-root outputs/p3_hybrid_full8_e30_mem4_seed0 --ensemble
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.predictions import load_member_prediction_artifact


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.4, message, ha="center", va="center", wrap=True, fontsize=11)
    _save(fig, path)


def _task_ids(run_root: Path, member_id: int = 0) -> list[int]:
    root = run_root / f"member_{member_id}"
    ids = []
    for p in root.glob("task_*"):
        try:
            ids.append(int(p.name.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    if not ids:
        for p in root.glob("member_predictions/task_*"):
            try:
                ids.append(int(p.name.split("_", 1)[1]))
            except (ValueError, IndexError):
                continue
    return sorted(set(ids))


def _prediction_path(run_root: Path, member_id: int, task_id: int, split: str) -> Path | None:
    direct = run_root / f"member_{member_id}" / "member_predictions" / f"task_{task_id}" / f"{split}.npz"
    if direct.is_file():
        return direct
    matches = list((run_root / f"member_{member_id}").glob(f"**/task_{task_id}/{split}.npz"))
    return matches[0] if matches else None


def _load_predictions(
    run_root: Path,
    member_ids: list[int],
    task_id: int,
    split: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    artifacts = []
    sources = []
    for member_id in member_ids:
        path = _prediction_path(run_root, member_id, task_id, split)
        if path is None:
            warnings.append(f"missing prediction: member={member_id} task={task_id} split={split}")
            continue
        try:
            artifacts.append(load_member_prediction_artifact(path))
            sources.append(str(path))
        except Exception as exc:  # an invalid artifact should not hide other diagnostics
            warnings.append(f"cannot load {path}: {type(exc).__name__}: {exc}")
    if not artifacts:
        return None

    base = artifacts[0]
    sample_ids = np.asarray(base.sample_ids)
    labels = np.asarray(base.labels).astype(int)
    class_ids = sorted({int(x) for art in artifacts for x in np.asarray(art.raw_class_ids).tolist()})
    class_index = {raw: i for i, raw in enumerate(class_ids)}
    sample_index = {str(value): i for i, value in enumerate(sample_ids.tolist())}
    sums = np.zeros((len(sample_ids), len(class_ids)), dtype=float)
    used = 0
    for art in artifacts:
        ids = np.asarray(art.sample_ids)
        positions = [sample_index.get(str(value)) for value in ids.tolist()]
        if any(pos is None for pos in positions):
            warnings.append("prediction artifacts do not share identical sample IDs; skipped one member")
            continue
        art_labels = np.asarray(art.labels).astype(int)
        if not np.array_equal(labels[np.asarray(positions)], art_labels):
            warnings.append("prediction artifacts disagree on labels; skipped one member")
            continue
        probs = np.asarray(art.probabilities, dtype=float)
        mapped = np.zeros((len(ids), len(class_ids)), dtype=float)
        for col, raw in enumerate(np.asarray(art.raw_class_ids).astype(int).tolist()):
            mapped[:, class_index[int(raw)]] = probs[:, col]
        sums[np.asarray(positions)] += mapped
        used += 1
    if used == 0:
        return None
    probabilities = sums / float(used)
    predictions = np.asarray(class_ids)[np.argmax(probabilities, axis=1)]
    return {
        "labels": labels,
        "predictions": predictions.astype(int),
        "probabilities": probabilities,
        "class_ids": np.asarray(class_ids, dtype=int),
        "sources": sources,
        "members_used": used,
    }


def _class_metrics(labels: np.ndarray, predictions: np.ndarray, class_ids: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_ids.tolist(), zero_division=0
    )
    return pd.DataFrame(
        {
            "raw_class_id": class_ids.astype(int),
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def _load_buffer(run_root: Path, member_id: int, task_id: int, warnings: list[str]) -> pd.DataFrame | None:
    candidates = [
        run_root / f"member_{member_id}" / f"task_{task_id}" / "buffer_audit.json",
        run_root / f"member_{member_id}" / "checkpoints" / f"task_{task_id}_buffer_audit.json",
    ]
    candidates.extend(run_root.glob(f"member_{member_id}/**/task_{task_id}/buffer_audit.json"))
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        warnings.append(f"missing buffer audit: member={member_id} task={task_id}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        allocation = {int(k): int(v) for k, v in payload.get("allocation", {}).items()}
        observed = {int(k): int(v) for k, v in payload.get("observed_counts", {}).items()}
        feasible = {int(k): int(v) for k, v in payload.get("feasible_capacities", {}).items()}
        keys = sorted(set(allocation) | set(observed) | set(feasible))
        return pd.DataFrame(
            {
                "raw_class_id": keys,
                "observed_count": [observed.get(k, 0) for k in keys],
                "allocated_slots": [allocation.get(k, 0) for k in keys],
                "feasible_capacity": [feasible.get(k, 0) for k in keys],
                "budget": int(payload.get("budget", sum(allocation.values()))),
                "source": str(path),
            }
        )
    except Exception as exc:
        warnings.append(f"cannot parse buffer audit {path}: {type(exc).__name__}: {exc}")
        return None


def _replay_exposure(
    run_root: Path, member_id: int, task_id: int, buffer: pd.DataFrame | None, warnings: list[str]
) -> tuple[pd.DataFrame | None, str]:
    candidates = []
    for pattern in ("*replay*exposure*.csv", "*replay*draw*.csv", "*replay*audit*.csv"):
        candidates.extend(run_root.glob(f"member_{member_id}/**/{pattern}"))
    for path in candidates:
        try:
            df = pd.read_csv(path)
            class_col = next((c for c in ("raw_class_id", "class_id", "raw_label", "label") if c in df), None)
            count_col = next((c for c in ("replay_draws", "draws", "count", "samples", "exposure") if c in df), None)
            if class_col and count_col:
                out = df.rename(columns={class_col: "raw_class_id", count_col: "replay_exposure"})
                out["raw_class_id"] = pd.to_numeric(out["raw_class_id"], errors="raise").astype(int)
                out["replay_exposure"] = pd.to_numeric(out["replay_exposure"], errors="raise")
                out = out.groupby("raw_class_id", as_index=False)["replay_exposure"].sum()
                return out, "actual_replay_audit"
        except Exception as exc:
            warnings.append(f"cannot parse replay audit {path}: {type(exc).__name__}: {exc}")
    if buffer is not None:
        return buffer[["raw_class_id", "allocated_slots"]].rename(columns={"allocated_slots": "replay_exposure"}), "planned_buffer_allocation"
    warnings.append(f"missing replay exposure and buffer audit: member={member_id} task={task_id}")
    return None, "missing"


def _bands(df: pd.DataFrame, value_col: str) -> pd.Series:
    values = df[value_col].astype(float)
    low, high = values.quantile(1 / 3), values.quantile(2 / 3)
    return pd.Series(np.where(values <= low, "tail", np.where(values >= high, "head", "mid")), index=df.index)


def _plot_buffer(df: pd.DataFrame, path: Path, title: str) -> None:
    df = df.sort_values("raw_class_id").copy()
    df["band"] = _bands(df, "observed_count")
    colors = df["band"].map({"tail": "#d95f02", "mid": "#7570b3", "head": "#1b9e77"})
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axes[0].bar(df.raw_class_id, df.allocated_slots, color=colors, width=0.9)
    axes[0].set_ylabel("allocated buffer slots")
    axes[0].set_title(title)
    ax2 = axes[0].twinx()
    ax2.plot(df.raw_class_id, df.observed_count, color="black", linewidth=1, alpha=0.65, label="observed samples")
    ax2.set_ylabel("observed class samples")
    axes[1].plot(df.raw_class_id, df.allocated_slots / np.maximum(df.observed_count, 1), color="#2166ac")
    axes[1].set_ylabel("buffer / observed count")
    axes[1].set_xlabel("raw class id")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].legend(handles=[Patch(color="#d95f02", label="tail"), Patch(color="#7570b3", label="mid"), Patch(color="#1b9e77", label="head")], loc="upper left")
    _save(fig, path)


def _plot_exposure(df: pd.DataFrame, path: Path, title: str, mode: str) -> None:
    df = df.sort_values("raw_class_id")
    colors = _bands(df.rename(columns={"replay_exposure": "value"}), "value").map({"tail": "#d95f02", "mid": "#7570b3", "head": "#1b9e77"})
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(df.raw_class_id, df.replay_exposure, color=colors, width=0.9)
    ax.set_title(f"{title} ({mode.replace('_', ' ')})")
    ax.set_xlabel("raw class id")
    ax.set_ylabel("replay draws / planned slots")
    _save(fig, path)


def _plot_classwise(df: pd.DataFrame, path: Path, title: str) -> None:
    df = df.sort_values("raw_class_id")
    colors = _bands(df, "support").map({"tail": "#d95f02", "mid": "#7570b3", "head": "#1b9e77"})
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(df.raw_class_id, df.f1, color="#2166ac", marker=".", linewidth=1)
    axes[0].axhline(df.f1.mean(), color="black", linestyle="--", label=f"macro-F1={df.f1.mean():.3f}")
    axes[0].set_ylabel("class F1")
    axes[0].legend()
    axes[0].set_title(title)
    axes[1].scatter(df.raw_class_id, df.recall, c=colors, s=np.clip(df.support, 8, 160), alpha=0.8)
    axes[1].axhline(df.recall.mean(), color="black", linestyle="--", label=f"macro recall={df.recall.mean():.3f}")
    axes[1].set_ylabel("class recall / balanced recall")
    axes[1].set_xlabel("raw class id; marker size = support")
    axes[1].legend()
    _save(fig, path)


def _plot_forgetting(matrix: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    if matrix.empty or matrix.shape[1] < 2:
        ax.axis("off")
        ax.text(0.5, 0.5, "Need at least two task-boundary prediction artifacts", ha="center", va="center")
        _save(fig, path)
        return
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(image, ax=ax, label="class F1")
    ax.set_title(title)
    ax.set_xlabel("task boundary")
    ax.set_ylabel("raw class id")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    yticks = np.arange(0, len(matrix.index), max(1, len(matrix.index) // 20))
    ax.set_yticks(yticks, matrix.index.to_numpy()[yticks])
    _save(fig, path)


def _plot_confusion(labels: np.ndarray, predictions: np.ndarray, class_ids: np.ndarray, path: Path, title: str) -> pd.DataFrame:
    counts = confusion_matrix(labels, predictions, labels=class_ids.tolist())
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals != 0)
    off = counts.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    flat = np.argsort(off.ravel())[::-1]
    for idx in flat:
        true_i, pred_i = np.unravel_index(idx, off.shape)
        if off[true_i, pred_i] <= 0:
            break
        pairs.append({"true_class": int(class_ids[true_i]), "predicted_class": int(class_ids[pred_i]), "count": int(off[true_i, pred_i]), "row_rate": float(normalized[true_i, pred_i])})
        if len(pairs) >= 15:
            break
    top = pd.DataFrame(pairs)
    fig = plt.figure(figsize=(18, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.2])
    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(normalized, interpolation="nearest", aspect="auto", vmin=0, vmax=1, cmap="magma")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="row-normalized rate")
    ax.set_title(title)
    ax.set_xlabel("predicted raw class id")
    ax.set_ylabel("true raw class id")
    ticks = np.arange(0, len(class_ids), max(1, len(class_ids) // 18))
    ax.set_xticks(ticks, class_ids[ticks], rotation=90)
    ax.set_yticks(ticks, class_ids[ticks])
    side = fig.add_subplot(grid[0, 1])
    if top.empty:
        side.text(0.5, 0.5, "No off-diagonal errors", ha="center", va="center")
        side.axis("off")
    else:
        labels = [f"{r.true_class}->{r.predicted_class}" for r in top.itertuples()]
        side.barh(labels[::-1], top["count"].to_numpy()[::-1], color="#d73027")
        side.set_xlabel("count")
        side.set_title("Top confusions")
    _save(fig, path)
    return top


def _discover_task_ids(run_root: Path, member_ids: list[int]) -> list[int]:
    ids = []
    for member in member_ids:
        ids.extend(_task_ids(run_root, member))
    return sorted(set(ids))


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    outdir = (args.outdir or run_root / "visualizations").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    available = [m for m in args.member_ids if (run_root / f"member_{m}").exists()]
    if not available:
        available = [args.member_id]
    task_ids = _discover_task_ids(run_root, available)
    if not task_ids:
        task_ids = [args.task_id] if args.task_id is not None else [0]
    task_id = args.task_id if args.task_id is not None else max(task_ids)
    prediction_members = available if args.ensemble else [args.member_id]
    scope = "ensemble" if args.ensemble else f"member{args.member_id}"
    generated: list[str] = []

    buffer = _load_buffer(run_root, args.member_id, task_id, warnings)
    buffer_png = outdir / f"01_buffer_allocation_task{task_id}_member{args.member_id}.png"
    if buffer is None:
        _placeholder(buffer_png, "Buffer allocation", "buffer_audit.json was not found for this task/member")
    else:
        buffer.to_csv(outdir / f"buffer_allocation_task{task_id}_member{args.member_id}.csv", index=False)
        _plot_buffer(buffer, buffer_png, f"Buffer allocation: task {task_id}, member {args.member_id}")
    generated.append(str(buffer_png))

    exposure, exposure_mode = _replay_exposure(run_root, args.member_id, task_id, buffer, warnings)
    exposure_png = outdir / f"02_replay_exposure_task{task_id}_member{args.member_id}.png"
    if exposure is None:
        _placeholder(exposure_png, "Replay exposure", "No replay exposure audit was found")
    else:
        exposure.to_csv(outdir / f"replay_exposure_task{task_id}_member{args.member_id}.csv", index=False)
        _plot_exposure(exposure, exposure_png, f"Replay exposure: task {task_id}, member {args.member_id}", exposure_mode)
    generated.append(str(exposure_png))

    final_prediction = _load_predictions(run_root, prediction_members, task_id, args.split, warnings)
    classwise_png = outdir / f"03_classwise_performance_task{task_id}_{scope}.png"
    if final_prediction is None:
        _placeholder(classwise_png, "Class-wise performance", f"No {args.split}.npz prediction artifact was found")
    else:
        metrics = _class_metrics(final_prediction["labels"], final_prediction["predictions"], final_prediction["class_ids"])
        metrics.to_csv(outdir / f"classwise_metrics_task{task_id}_{scope}.csv", index=False)
        _plot_classwise(metrics, classwise_png, f"Class-wise performance: task {task_id} ({scope}, {args.split})")
    generated.append(str(classwise_png))

    matrix_rows = {}
    for boundary in task_ids:
        data = _load_predictions(run_root, prediction_members, boundary, args.split, warnings)
        if data is None:
            continue
        metrics = _class_metrics(data["labels"], data["predictions"], data["class_ids"])
        matrix_rows[boundary] = metrics.set_index("raw_class_id")["f1"]
    matrix = pd.DataFrame(matrix_rows).sort_index()
    matrix.columns = [f"task_{int(c)}" for c in matrix.columns]
    forgetting_png = outdir / f"04_forgetting_heatmap_{scope}.png"
    _plot_forgetting(matrix, forgetting_png, f"Class-wise F1 across task boundaries ({scope}, {args.split})")
    if not matrix.empty:
        matrix.to_csv(outdir / f"forgetting_f1_matrix_{scope}.csv")
        if matrix.shape[1] >= 2:
            drop = pd.DataFrame({"raw_class_id": matrix.index, "best_previous_f1": matrix.iloc[:, :-1].max(axis=1, skipna=True), "final_f1": matrix.iloc[:, -1]})
            drop["forgetting_drop"] = drop["best_previous_f1"] - drop["final_f1"]
            drop.to_csv(outdir / f"forgetting_by_class_{scope}.csv", index=False)
    generated.append(str(forgetting_png))

    confusion_png = outdir / f"05_confusion_matrix_task{task_id}_{scope}.png"
    if final_prediction is None:
        _placeholder(confusion_png, "Confusion matrix", f"No {args.split}.npz prediction artifact was found")
    else:
        top = _plot_confusion(final_prediction["labels"], final_prediction["predictions"], final_prediction["class_ids"], confusion_png, f"Confusion matrix: task {task_id} ({scope}, {args.split})")
        top.to_csv(outdir / f"top_confusions_task{task_id}_{scope}.csv", index=False)
    generated.append(str(confusion_png))

    manifest = {
        "run_root": str(run_root),
        "outdir": str(outdir),
        "task_id": task_id,
        "task_boundaries": task_ids,
        "split": args.split,
        "scope": scope,
        "members_requested": args.member_ids,
        "members_used_for_predictions": prediction_members,
        "generated": generated,
        "warnings": sorted(set(warnings)),
        "notes": [
            "Replay exposure uses actual replay audit when present; otherwise it is labelled as planned buffer allocation.",
            "Forgetting is computed from per-task prediction artifacts, not from aggregate task metrics.",
            "Macro-F1 is the unweighted mean of class F1 values in the class-wise CSV.",
        ],
    }
    (outdir / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--member-id", type=int, default=0)
    parser.add_argument("--member-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ensemble", action="store_true", help="average available member prediction artifacts")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"outdir": manifest["outdir"], "generated": len(manifest["generated"]), "warnings": len(manifest["warnings"])}, indent=2))


if __name__ == "__main__":
    main()
