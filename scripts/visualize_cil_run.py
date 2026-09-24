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
import pyarrow.parquet as pq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.predictions import load_member_prediction_artifact
from src.eval.ensemble_ue import load_offline_ensemble_artifact


DEFAULT_TASK_FILE = ROOT / "study_assets" / "task_protocols" / "tail_to_head_tasks.json"


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


def _load_task_info(
    run_root: Path,
    task_file: Path | None,
    warnings: list[str],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load raw-class -> human-readable task/stage labels.

    Raw IDs remain the stable key.  The added label is intentionally derived
    from the frozen task schedule, not from model predictions.
    """
    candidates: list[Path] = []
    if task_file is not None:
        candidates.append(task_file)
    # A completed member run records the resolved task-file path in arguments.
    for run_config in [run_root / "member_0" / "run_config.json", *run_root.glob("member_*/run_config.json")]:
        if not run_config.is_file():
            continue
        try:
            payload = json.loads(run_config.read_text(encoding="utf-8"))
            value = payload.get("arguments", {}).get("task_file")
            if value:
                candidate = Path(str(value))
                candidates.append(candidate if candidate.is_absolute() else ROOT / candidate)
        except Exception as exc:
            warnings.append(f"cannot read task source from {run_config}: {type(exc).__name__}: {exc}")
    candidates.append(DEFAULT_TASK_FILE)
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        warnings.append("task protocol file not found; using raw-class labels only")
        return {}, {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        protocol = str(payload.get("protocol", "unknown"))
        tasks = payload.get("tasks", [])
        info: dict[int, dict[str, Any]] = {}
        for task in tasks:
            task_id = int(task["task_id"])
            if protocol == "tail_to_head":
                stage = "tail" if task_id == 0 else "head" if task_id == len(tasks) - 1 else "mid"
            else:
                stage = "balanced"
            for raw in task.get("classes", []):
                raw_id = int(raw)
                info[raw_id] = {
                    "task_id": task_id,
                    "stage": stage,
                    "class_label": f"t{task_id}_{stage}:c{raw_id}",
                }
        return info, {"protocol": protocol, "source": str(source), "num_tasks": len(tasks)}
    except Exception as exc:
        warnings.append(f"cannot parse task protocol {source}: {type(exc).__name__}: {exc}")
        return {}, {}


def _decorate_class_table(frame: pd.DataFrame, task_info: dict[int, dict[str, Any]]) -> pd.DataFrame:
    out = frame.copy()
    out["task_id"] = [task_info.get(int(raw), {}).get("task_id", -1) for raw in out["raw_class_id"]]
    out["stage"] = [task_info.get(int(raw), {}).get("stage", "unknown") for raw in out["raw_class_id"]]
    out["class_label"] = [
        task_info.get(int(raw), {}).get("class_label", f"class_{int(raw)}")
        for raw in out["raw_class_id"]
    ]
    return out


def _class_label(task_info: dict[int, dict[str, Any]], raw: int) -> str:
    return task_info.get(int(raw), {}).get("class_label", f"class_{int(raw)}")


def _infer_source_paths(run_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for run_config in [run_root / "member_0" / "run_config.json", *run_root.glob("member_*/run_config.json")]:
        if not run_config.is_file():
            continue
        try:
            payload = json.loads(run_config.read_text(encoding="utf-8"))
            for split in ("train", "validation", "test"):
                value = payload.get("arguments", {}).get(split)
                if value:
                    candidate = Path(str(value))
                    candidate = candidate if candidate.is_absolute() else ROOT / candidate
                    if candidate.is_file():
                        paths[split] = candidate
            if len(paths) == 3:
                break
        except Exception:
            continue
    return paths


def _task_sample_counts(
    run_root: Path,
    task_info: dict[int, dict[str, Any]],
    source_paths: dict[str, Path],
    fallback_test_labels: np.ndarray | None,
    warnings: list[str],
    protocol: str = "tail_to_head",
) -> pd.DataFrame:
    task_ids = sorted({int(meta["task_id"]) for meta in task_info.values()})
    rows = []
    for task_id in task_ids:
        classes = sorted(raw for raw, meta in task_info.items() if int(meta["task_id"]) == task_id)
        stage = (
            "tail" if protocol == "tail_to_head" and task_id == 0 else
            "head" if protocol == "tail_to_head" and task_id == task_ids[-1] else
            "balanced" if protocol == "constrained_mass_balanced" else "mid"
        )
        row: dict[str, Any] = {
            "task_id": task_id,
            "task_label": f"task_{task_id}_{stage}",
            "class_count": len(classes),
            "raw_class_ids": ",".join(map(str, classes)),
        }
        for split in ("train", "validation", "test"):
            path = source_paths.get(split)
            if path is not None:
                try:
                    columns = pq.read_schema(path).names
                    label_col = "class" if "class" in columns else "raw_class_id" if "raw_class_id" in columns else None
                    if label_col is None:
                        raise ValueError("no class/raw_class_id column")
                    labels = pq.read_table(path, columns=[label_col])[label_col].to_numpy()
                    row[f"{split}_samples"] = int(np.isin(labels.astype(int), classes).sum())
                except Exception as exc:
                    warnings.append(f"cannot count {split} samples from {path}: {type(exc).__name__}: {exc}")
                    row[f"{split}_samples"] = None
            elif split == "test" and fallback_test_labels is not None:
                row["test_samples"] = int(np.isin(fallback_test_labels, classes).sum())
            else:
                row[f"{split}_samples"] = None
        row["all_split_samples"] = int(sum(row[f"{s}_samples"] or 0 for s in ("train", "validation", "test")))
        rows.append(row)
    return pd.DataFrame(rows)


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
    # When the normal offline ensemble stage has already run, visualize that
    # exact artifact instead of rebuilding ensemble probabilities from members.
    offline_path = run_root / "offline_evaluation" / f"task_{task_id}" / (
        "oof.npz" if split == "validation" else "test.npz"
    )
    if len(member_ids) == 3 and offline_path.is_file():
        try:
            artifact = load_offline_ensemble_artifact(offline_path)
            return {
                "labels": np.asarray(artifact.labels).astype(int),
                "predictions": np.asarray(artifact.predictions).astype(int),
                "probabilities": np.asarray(artifact.probabilities, dtype=float),
                "class_ids": np.asarray(artifact.raw_class_ids).astype(int),
                "sources": [str(offline_path)],
                "members_used": len(artifact.context.member_ids),
            }
        except Exception as exc:
            warnings.append(f"cannot load offline ensemble {offline_path}: {type(exc).__name__}: {exc}")
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


def _class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_ids: np.ndarray,
    task_info: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_ids.tolist(), zero_division=0
    )
    frame = pd.DataFrame(
        {
            "raw_class_id": class_ids.astype(int),
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return _decorate_class_table(frame, task_info or {})


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
    ticks = np.arange(0, len(df), max(1, len(df) // 18))
    axes[0].set_xticks(df.raw_class_id.iloc[ticks], df.class_label.iloc[ticks], rotation=90)
    axes[1].set_xticks(df.raw_class_id.iloc[ticks], df.class_label.iloc[ticks], rotation=90)
    _save(fig, path)


def _plot_exposure(df: pd.DataFrame, path: Path, title: str, mode: str) -> None:
    df = df.sort_values("raw_class_id")
    colors = _bands(df.rename(columns={"replay_exposure": "value"}), "value").map({"tail": "#d95f02", "mid": "#7570b3", "head": "#1b9e77"})
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(df.raw_class_id, df.replay_exposure, color=colors, width=0.9)
    ax.set_title(f"{title} ({mode.replace('_', ' ')})")
    ax.set_xlabel("raw class id")
    ax.set_ylabel("replay draws / planned slots")
    ticks = np.arange(0, len(df), max(1, len(df) // 18))
    ax.set_xticks(df.raw_class_id.iloc[ticks], df.class_label.iloc[ticks], rotation=90)
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
    ticks = np.arange(0, len(df), max(1, len(df) // 18))
    axes[0].set_xticks(df.raw_class_id.iloc[ticks], df.class_label.iloc[ticks], rotation=90)
    axes[1].set_xticks(df.raw_class_id.iloc[ticks], df.class_label.iloc[ticks], rotation=90)
    _save(fig, path)


def _plot_forgetting(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    task_info: dict[int, dict[str, Any]] | None = None,
) -> None:
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
    info = task_info or {}
    ax.set_yticks(yticks, [_class_label(info, int(raw)) for raw in matrix.index.to_numpy()[yticks]])
    _save(fig, path)


def _plot_confusion(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_ids: np.ndarray,
    path: Path,
    title: str,
    task_info: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
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
        true_raw = int(class_ids[true_i])
        pred_raw = int(class_ids[pred_i])
        pairs.append({
            "true_class": true_raw,
            "true_label": _class_label(task_info or {}, true_raw),
            "predicted_class": pred_raw,
            "predicted_label": _class_label(task_info or {}, pred_raw),
            "count": int(off[true_i, pred_i]),
            "row_rate": float(normalized[true_i, pred_i]),
        })
        if len(pairs) >= 15:
            break
    top = pd.DataFrame(pairs)
    fig = plt.figure(figsize=(18, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.2])
    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(normalized, interpolation="nearest", aspect="auto", vmin=0, vmax=1, cmap="magma")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="row-normalized rate")
    ax.set_title(title)
    ax.set_xlabel("predicted class (task/stage:raw id)")
    ax.set_ylabel("true class (task/stage:raw id)")
    ticks = np.arange(0, len(class_ids), max(1, len(class_ids) // 18))
    labels_for_ticks = [_class_label(task_info or {}, int(raw)) for raw in class_ids[ticks]]
    ax.set_xticks(ticks, labels_for_ticks, rotation=90)
    ax.set_yticks(ticks, labels_for_ticks)
    side = fig.add_subplot(grid[0, 1])
    if top.empty:
        side.text(0.5, 0.5, "No off-diagonal errors", ha="center", va="center")
        side.axis("off")
    else:
        labels = [f"{r.true_label}->{r.predicted_label}" for r in top.itertuples()]
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
    task_info, task_metadata = _load_task_info(run_root, args.task_file, warnings)
    source_paths = _infer_source_paths(run_root)
    for split in ("train", "validation", "test"):
        explicit = getattr(args, split, None)
        if explicit is not None:
            source_paths[split] = explicit.resolve()
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
        buffer = _decorate_class_table(buffer, task_info)
        buffer.to_csv(outdir / f"buffer_allocation_task{task_id}_member{args.member_id}.csv", index=False)
        _plot_buffer(buffer, buffer_png, f"Buffer allocation: task {task_id}, member {args.member_id}")
    generated.append(str(buffer_png))

    exposure, exposure_mode = _replay_exposure(run_root, args.member_id, task_id, buffer, warnings)
    exposure_png = outdir / f"02_replay_exposure_task{task_id}_member{args.member_id}.png"
    if exposure is None:
        _placeholder(exposure_png, "Replay exposure", "No replay exposure audit was found")
    else:
        exposure = _decorate_class_table(exposure, task_info)
        exposure.to_csv(outdir / f"replay_exposure_task{task_id}_member{args.member_id}.csv", index=False)
        _plot_exposure(exposure, exposure_png, f"Replay exposure: task {task_id}, member {args.member_id}", exposure_mode)
    generated.append(str(exposure_png))

    final_prediction = _load_predictions(run_root, prediction_members, task_id, args.split, warnings)
    classwise_png = outdir / f"03_classwise_performance_task{task_id}_{scope}.png"
    if final_prediction is None:
        _placeholder(classwise_png, "Class-wise performance", f"No {args.split}.npz prediction artifact was found")
    else:
        metrics = _class_metrics(
            final_prediction["labels"], final_prediction["predictions"],
            final_prediction["class_ids"], task_info,
        )
        metrics.to_csv(outdir / f"classwise_metrics_task{task_id}_{scope}.csv", index=False)
        _plot_classwise(metrics, classwise_png, f"Class-wise performance: task {task_id} ({scope}, {args.split})")
    generated.append(str(classwise_png))

    matrix_rows = {}
    for boundary in task_ids:
        data = _load_predictions(run_root, prediction_members, boundary, args.split, warnings)
        if data is None:
            continue
        metrics = _class_metrics(data["labels"], data["predictions"], data["class_ids"], task_info)
        matrix_rows[boundary] = metrics.set_index("raw_class_id")["f1"]
    matrix = pd.DataFrame(matrix_rows).sort_index()
    matrix.columns = [f"task_{int(c)}" for c in matrix.columns]
    forgetting_png = outdir / f"04_forgetting_heatmap_{scope}.png"
    _plot_forgetting(
        matrix, forgetting_png,
        f"Class-wise F1 across task boundaries ({scope}, {args.split})",
        task_info,
    )
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
        top = _plot_confusion(
            final_prediction["labels"], final_prediction["predictions"],
            final_prediction["class_ids"], confusion_png,
            f"Confusion matrix: task {task_id} ({scope}, {args.split})",
            task_info,
        )
        top.to_csv(outdir / f"top_confusions_task{task_id}_{scope}.csv", index=False)
    generated.append(str(confusion_png))

    fallback_labels = None if final_prediction is None else final_prediction["labels"]
    task_counts = _task_sample_counts(
        run_root, task_info, source_paths, fallback_labels, warnings,
        str(task_metadata.get("protocol", "tail_to_head")),
    )
    task_counts.to_csv(outdir / "task_sample_counts.csv", index=False)
    generated.append(str(outdir / "task_sample_counts.csv"))
    if task_info:
        (outdir / "class_labels.csv").write_text(
            pd.DataFrame(
                [
                    {"raw_class_id": raw, **meta}
                    for raw, meta in sorted(task_info.items())
                ]
            ).to_csv(index=False),
            encoding="utf-8",
        )
        generated.append(str(outdir / "class_labels.csv"))

    manifest = {
        "run_root": str(run_root),
        "outdir": str(outdir),
        "task_id": task_id,
        "task_boundaries": task_ids,
        "split": args.split,
        "scope": scope,
        "members_requested": args.member_ids,
        "members_used_for_predictions": prediction_members,
        "task_protocol": task_metadata,
        "source_splits": {split: str(path) for split, path in source_paths.items()},
        "generated": generated,
        "warnings": sorted(set(warnings)),
        "notes": [
            "Replay exposure uses actual replay audit when present; otherwise it is labelled as planned buffer allocation.",
            "Forgetting is computed from per-task prediction artifacts, not from aggregate task metrics.",
            "Macro-F1 is the unweighted mean of class F1 values in the class-wise CSV.",
            "Class labels use task/stage aliases (t0_tail, t7_head) while raw_class_id remains the stable key.",
            "task_sample_counts.csv reports train/validation/test rows per task when source Parquet paths are available; otherwise test counts fall back to the prediction artifact.",
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
    parser.add_argument("--task-file", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None, help="Optional train Parquet for per-task sample counts")
    parser.add_argument("--validation", type=Path, default=None, help="Optional validation Parquet for per-task sample counts")
    parser.add_argument("--test", type=Path, default=None, help="Optional test Parquet for per-task sample counts")
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"outdir": manifest["outdir"], "generated": len(manifest["generated"]), "warnings": len(manifest["warnings"])}, indent=2))


if __name__ == "__main__":
    main()
