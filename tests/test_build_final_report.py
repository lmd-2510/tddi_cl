from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.report import build_final_report
from src.eval.metrics import compute_classification_metrics
from src.eval.predictions import MemberPredictionArtifact, MemberPredictionContext
from src.eval.ensemble_ue import (
    aggregate_member_predictions,
    export_offline_ensemble_artifact,
)


def _member(
    *,
    task_id: int,
    member_id: int,
    labels: np.ndarray,
    raw_class_ids: np.ndarray,
    probabilities: np.ndarray,
) -> MemberPredictionArtifact:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    return MemberPredictionArtifact(
        context=MemberPredictionContext(
            run_id=f"member-{member_id}-run",
            method="replay_distill_fixed_budget_uniform",
            method_protocol="replay_distill_fixed_budget_uniform",
            task_id=task_id,
            split="test",
            member_id=member_id,
            experiment_seed=0,
            member_seed=100 + member_id,
        ),
        sample_ids=np.asarray(
            [f"task-{task_id}-sample-{index}" for index in range(labels.size)]
        ),
        labels=labels.astype(np.int64),
        raw_class_ids=raw_class_ids.astype(np.int64),
        logits=np.log(probabilities).astype(np.float32),
        probabilities=probabilities,
    )


def _write_task_artifact(
    full_root: Path,
    *,
    task_id: int,
    labels: list[int],
    raw_class_ids: list[int],
    member_probabilities: list[list[list[float]]],
) -> None:
    labels_array = np.asarray(labels, dtype=np.int64)
    class_array = np.asarray(raw_class_ids, dtype=np.int64)
    members = [
        _member(
            task_id=task_id,
            member_id=member_id,
            labels=labels_array,
            raw_class_ids=class_array,
            probabilities=np.asarray(probabilities, dtype=np.float32),
        )
        for member_id, probabilities in enumerate(member_probabilities)
    ]
    ensemble = aggregate_member_predictions(members)
    path = full_root / "offline_ensemble" / f"task_{task_id}" / "test.npz"
    export_offline_ensemble_artifact(ensemble, path)

    metrics = compute_classification_metrics(
        ensemble.labels,
        ensemble.predictions,
        labels=raw_class_ids,
    )
    report = {
        "evaluation_split": "test",
        "task_id": task_id,
        "experiment_seed": 0,
        "raw_class_ids": raw_class_ids,
        "threshold": {
            "score_name": "entropy_confidence",
            "probability_source": "raw",
            "value": 0.0,
            "source_split": "validation",
        },
        "full_set": {
            "sample_count": len(labels),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "ece": 0.1,
            "brier_score": 0.2,
            "negative_log_likelihood": 0.3,
        },
        "threshold_score_selective_metrics": {
            "selected_count": len(labels),
            "total_count": len(labels),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "coverage": 1.0,
        },
    }
    report_path = full_root / "threshold" / f"task_{task_id}" / "test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")


def _write_member_forgetting(full_root: Path) -> None:
    for member_id in range(3):
        root = full_root / f"member_{member_id}"
        root.mkdir(parents=True, exist_ok=True)
        task_forgetting = 0.1 + member_id * 0.05
        pd.DataFrame(
            [
                {
                    "task_id": 0,
                    "best_macro_f1": 0.9,
                    "final_macro_f1": 0.9 - task_forgetting,
                    "forgetting": task_forgetting,
                },
                {
                    "task_id": 1,
                    "best_macro_f1": 0.8,
                    "final_macro_f1": 0.8,
                    "forgetting": 0.0,
                },
                {
                    "task_id": "mean_old_tasks",
                    "best_macro_f1": np.nan,
                    "final_macro_f1": np.nan,
                    "forgetting": task_forgetting,
                },
            ]
        ).to_csv(root / "forgetting.csv", index=False)
        pd.DataFrame(
            [
                {
                    "train_task": 1,
                    "class_id": class_id,
                    "current_f1": 0.0 if class_id == 30 else 0.7,
                    "forgetting": 0.05 * (class_id // 10) + 0.01 * member_id,
                }
                for class_id in (10, 20, 30)
            ]
        ).to_csv(root / "class_forgetting.csv", index=False)


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    full_root = tmp_path / "full"
    task_file = tmp_path / "tasks.json"
    task_file.write_text(
        json.dumps(
            {
                "protocol": "tail_to_head",
                "seed": None,
                "num_classes": 3,
                "tasks": [
                    {"task_id": 0, "classes": [10, 20]},
                    {"task_id": 1, "classes": [30]},
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_task_artifact(
        full_root,
        task_id=0,
        labels=[10, 10, 20, 20],
        raw_class_ids=[10, 20],
        member_probabilities=[
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.4, 0.6]],
            [[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4]],
            [[0.7, 0.3], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]],
        ],
    )
    _write_task_artifact(
        full_root,
        task_id=1,
        labels=[10, 20, 30, 10, 20, 30],
        raw_class_ids=[10, 20, 30],
        member_probabilities=[
            [
                [0.6, 0.2, 0.2],
                [0.5, 0.4, 0.1],
                [0.1, 0.2, 0.7],
                [0.4, 0.5, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.2, 0.6],
            ],
            [
                [0.5, 0.3, 0.2],
                [0.3, 0.6, 0.1],
                [0.2, 0.2, 0.6],
                [0.3, 0.6, 0.1],
                [0.3, 0.6, 0.1],
                [0.1, 0.3, 0.6],
            ],
            [
                [0.6, 0.2, 0.2],
                [0.6, 0.3, 0.1],
                [0.2, 0.1, 0.7],
                [0.2, 0.7, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.2, 0.6],
            ],
        ],
    )
    _write_member_forgetting(full_root)
    return full_root, task_file


def test_builds_extended_report_without_training(tmp_path: Path) -> None:
    full_root, task_file = _build_fixture(tmp_path)

    paths = build_final_report(full_root=full_root, task_file=task_file)

    assert all(path.is_file() for path in paths.values())
    summary = pd.read_csv(paths["task_summary"])
    assert summary["task_id"].tolist() == [0, 1]
    assert {
        "full_balanced_accuracy",
        "full_weighted_f1",
        "threshold_balanced_accuracy",
        "threshold_weighted_f1",
        "mean_pairwise_disagreement",
        "mean_member_normalized_mi",
    }.issubset(summary.columns)
    assert (summary["coverage"] == 1.0).all()

    forgetting = pd.read_csv(paths["ensemble_forgetting"])
    assert forgetting["task_id"].astype(str).tolist() == ["0", "1", "mean_old_tasks"]
    assert float(forgetting.iloc[-1]["forgetting"]) > 0.0

    member_forgetting = pd.read_csv(paths["member_forgetting"])
    assert member_forgetting["member_id"].tolist() == [0, 1, 2]
    assert member_forgetting["zero_f1_classes_final"].tolist() == [1, 1, 1]

    diversity = pd.read_csv(paths["diversity"])
    assert len(diversity) == 2
    assert (diversity["mean_pairwise_disagreement"] > 0.0).any()

    report = paths["report"].read_text(encoding="utf-8")
    assert "Full Balanced Accuracy" in report
    assert "Full Weighted F1" in report
    assert "Forgettting" not in report
    assert "Ensemble diversity" in report
    assert "no training or inference" not in report.casefold()


def test_refuses_existing_outputs_without_explicit_overwrite(tmp_path: Path) -> None:
    full_root, task_file = _build_fixture(tmp_path)
    build_final_report(full_root=full_root, task_file=task_file)

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_final_report(full_root=full_root, task_file=task_file)

    paths = build_final_report(
        full_root=full_root,
        task_file=task_file,
        overwrite=True,
    )
    assert paths["report"].is_file()
