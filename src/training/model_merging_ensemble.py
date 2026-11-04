"""Offline ensemble and lightweight review package for merging runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import zipfile

import numpy as np
import pandas as pd

# This file is launched by an absolute path from the nohup worker.  In that
# mode Python puts ``src/training`` on sys.path, not the repository root, so
# ``from src...`` imports fail on a clean server.  Bootstrap the package root
# explicitly instead of relying on the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import compute_classification_metrics


def _log(message: str) -> None:
    print(f"[MODEL-MERGING-ENSEMBLE] {message}", flush=True)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_ensemble(root: Path, *, method: str, exclude_npz: bool = True) -> Path:
    _log(f"start method={method} root={root}")
    member_roots = [root / f"member_{member}" for member in (0, 1, 2)]
    rows: list[dict[str, object]] = []
    ensemble_dir = root / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    for task_id in range(8):
        _log(f"boundary_start task={task_id} loading_member_predictions")
        artifacts = [
            np.load(member_root / "merged" / f"task_{task_id}" / "test_predictions.npz", allow_pickle=False)
            for member_root in member_roots
        ]
        sample_ids = [np.asarray(artifact["sample_id"]) for artifact in artifacts]
        labels = [np.asarray(artifact["labels"], dtype=np.int64) for artifact in artifacts]
        class_ids = [np.asarray(artifact["class_ids"], dtype=np.int64) for artifact in artifacts]
        if not all(np.array_equal(sample_ids[0], item) for item in sample_ids[1:]):
            raise ValueError(f"Task {task_id} member sample IDs are not aligned.")
        if not all(np.array_equal(labels[0], item) for item in labels[1:]):
            raise ValueError(f"Task {task_id} member labels are not aligned.")
        if not all(np.array_equal(class_ids[0], item) for item in class_ids[1:]):
            raise ValueError(f"Task {task_id} member class IDs are not aligned.")
        probabilities = np.stack([np.asarray(artifact["probabilities"], dtype=np.float64) for artifact in artifacts])
        mean_probabilities = probabilities.mean(axis=0)
        predictions = class_ids[0][np.argmax(mean_probabilities, axis=1)]
        metrics = compute_classification_metrics(labels[0], predictions, labels=class_ids[0].tolist())
        row = {"task_id": task_id, "sample_count": int(len(labels[0])), **metrics}
        rows.append(row)
        np.savez_compressed(
            ensemble_dir / f"task_{task_id}_test.npz",
            sample_id=sample_ids[0], labels=labels[0], predictions=predictions,
            probabilities=mean_probabilities.astype(np.float32), class_ids=class_ids[0],
        )
        _log(
            f"boundary_complete task={task_id} samples={len(labels[0])} "
            f"macro_f1={float(metrics['macro_f1']):.6f}"
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "ensemble_metrics.csv", index=False)
    final = frame.iloc[-1]
    summary = {
        "method": method,
        "members": [0, 1, 2],
        "tasks": 8,
        "final_task": 7,
        "final_accuracy": float(final["accuracy"]),
        "final_balanced_accuracy": float(final["balanced_accuracy"]),
        "final_macro_f1": float(final["macro_f1"]),
        "final_weighted_f1": float(final["weighted_f1"]),
        "mean_task_macro_f1": float(frame["macro_f1"].mean()),
        "independent_experts": True,
        "future_experts_used": False,
    }
    _json(root / "ensemble_summary.json", summary)
    _json(root / "ensemble_manifest.json", {
        "kind": "offline_model_merging_ensemble_v1",
        "method": method,
        "members": [0, 1, 2],
        "probability_rule": "arithmetic_mean_of_member_probabilities",
        "artifact_alignment": "sample_id_labels_class_ids_exact",
    })
    _json(root / "full_manifest.json", {
        "kind": "ddi_cil_model_merging_full_manifest",
        "method": method,
        "members": [0, 1, 2],
        "trajectory": "task_0_base_then_independent_task_experts",
        "merge_scope": "learned_backbone_parameters_only",
        "future_experts_used": False,
        "review_archive_excludes": ["*.pt", "*.npz", "train.log", "stdout.log", "events.csv"],
    })
    try:
        table = frame.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        table = "```csv\n" + frame.to_csv(index=False) + "```"
    report = [
        f"# {method} independent-expert ensemble",
        "",
        "- Three frozen-fold members; each task expert was reset from task-0 base.",
        "- Only backbone parameters were merged; fixed-head rows were stitched by task ownership.",
        "- No future expert was used at an earlier boundary.",
        "",
        "## Final task metrics",
        "",
        f"- Accuracy: **{summary['final_accuracy']:.6f}**",
        f"- Balanced Accuracy: **{summary['final_balanced_accuracy']:.6f}**",
        f"- Macro-F1: **{summary['final_macro_f1']:.6f}**",
        f"- Weighted-F1: **{summary['final_weighted_f1']:.6f}**",
        f"- Mean task Macro-F1: **{summary['mean_task_macro_f1']:.6f}**",
        "",
        "## Per-boundary metrics",
        "",
        table,
        "",
    ]
    (root / "REVIEW_SUMMARY.md").write_text("\n".join(report), encoding="utf-8")
    archive = root / f"{root.name}_metrics_review.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in root.rglob("*"):
            if not path.is_file() or path == archive:
                continue
            rel = path.relative_to(root)
            if exclude_npz and path.suffix == ".npz":
                continue
            if path.suffix == ".pt" or path.name in {"train.log", "stdout.log", "events.csv"}:
                continue
            handle.write(path, rel.as_posix())
    _log(f"complete archive={archive}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", required=True, choices=["task_arithmetic", "ties"])
    args = parser.parse_args()
    archive = build_ensemble(args.root, method=args.method)
    print(f"REVIEW: {archive}", flush=True)


if __name__ == "__main__":
    main()
