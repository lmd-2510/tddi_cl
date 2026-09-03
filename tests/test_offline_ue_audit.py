from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.member_predictions import MemberPredictionArtifact, MemberPredictionContext
from src.eval.offline_ensemble import (
    aggregate_member_predictions,
    export_offline_ensemble_artifact,
)
from src.eval.offline_ue_audit import (
    SCORE_NAMES,
    count_train_classes_from_data,
    run_offline_ue_audit,
)


RAW_CLASS_IDS = np.asarray([10, 20], dtype=np.int64)
SAMPLE_IDS = np.asarray(["s0", "s1", "s2", "s3"])
LABELS = np.asarray([10, 20, 10, 20], dtype=np.int64)


def _member(
    member_id: int,
    *,
    include_errors: bool,
    all_wrong: bool = False,
) -> MemberPredictionArtifact:
    common = np.asarray(
        [
            [0.99, 0.01],
            [0.01, 0.99],
            [0.01, 0.99],
            [0.99, 0.01],
        ],
        dtype=np.float32,
    )
    if include_errors and member_id == 2:
        common[2] = [0.99, 0.01]
        common[3] = [0.01, 0.99]
    if not include_errors:
        common[2] = [0.99, 0.01]
        common[3] = [0.01, 0.99]
    if all_wrong:
        common = np.asarray(
            [
                [0.01, 0.99],
                [0.99, 0.01],
                [0.01, 0.99],
                [0.99, 0.01],
            ],
            dtype=np.float32,
        )
    return MemberPredictionArtifact(
        context=MemberPredictionContext(
            run_id=f"member-{member_id}",
            method="replay_distill_fixed_budget_uniform",
            method_protocol="replay_distill_fixed_budget_uniform",
            task_id=1,
            split="test",
            member_id=member_id,
            experiment_seed=0,
            member_seed=1000 + member_id,
        ),
        sample_ids=SAMPLE_IDS.copy(),
        labels=LABELS.copy(),
        raw_class_ids=RAW_CLASS_IDS.copy(),
        logits=np.log(common).astype(np.float32),
        probabilities=common,
    )


def _write_sources(
    tmp_path: Path,
    *,
    include_errors: bool,
    all_wrong: bool = False,
) -> tuple[Path, Path, Path]:
    ensemble = aggregate_member_predictions(
        [
            _member(
                member_id,
                include_errors=include_errors,
                all_wrong=all_wrong,
            )
            for member_id in range(3)
        ],
        source_artifact_paths=["m0.npz", "m1.npz", "m2.npz"],
    )
    ensemble_path = export_offline_ensemble_artifact(
        ensemble,
        tmp_path / "ensemble.npz",
    )
    task_file = tmp_path / "p3_tasks.json"
    task_file.write_text(
        json.dumps(
            {
                "protocol": "tail_to_head",
                "seed": None,
                "num_classes": 2,
                "tasks": [
                    {"task_id": 0, "classes": [10], "num_classes": 1},
                    {"task_id": 1, "classes": [20], "num_classes": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    count_file = tmp_path / "train_counts.csv"
    pd.DataFrame(
        {"class_id": [10, 20], "count": [10, 2001]}
    ).to_csv(count_file, index=False)
    return ensemble_path, task_file, count_file


def test_uncertainty_scores_detect_constructed_errors_and_export_report(
    tmp_path: Path,
) -> None:
    ensemble_path, task_file, count_file = _write_sources(
        tmp_path,
        include_errors=True,
    )
    output_json = tmp_path / "audit.json"
    output_csv = tmp_path / "audit.csv"

    report = run_offline_ue_audit(
        ensemble_path=ensemble_path,
        task_file=task_file,
        train_class_counts=count_file,
        output_json=output_json,
        output_csv=output_csv,
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "ddi_cil_offline_ue_audit"
    assert report["split"] == "test"
    assert report["task_id"] == 1
    assert report["experiment_seed"] == 0
    assert report["member_ids"] == [0, 1, 2]
    assert report["policy"] == "report_only_fixed_rules_no_threshold_or_score_selection"
    assert len(report["source_ensemble_sha256"]) == 64
    assert len(report["task_file_sha256"]) == 64
    assert len(report["config_sha256"]) == 64
    assert report["dataset"] == {
        "sample_count": 4,
        "correct_count": 2,
        "error_count": 2,
        "accuracy": 0.5,
    }

    assert set(report["uncertainty_scores"]) == set(SCORE_NAMES)
    for score_report in report["uncertainty_scores"].values():
        detection = score_report["error_detection"]
        assert detection["status"] == "ok"
        assert detection["positive_class"] == "prediction_error"
        np.testing.assert_allclose(detection["auroc"], 1.0)
        np.testing.assert_allclose(detection["auprc"], 1.0)
        assert score_report["mean_error"] > score_report["mean_correct"]
        selective = score_report["selective_prediction"]
        assert selective["coverage_metrics"][0]["actual_coverage"] == 0.25
        assert selective["coverage_metrics"][0]["accuracy"] == 1.0
        assert selective["coverage_metrics"][1]["actual_coverage"] == 0.5
        assert selective["coverage_metrics"][1]["accuracy"] == 1.0
        assert selective["coverage_metrics"][-1]["actual_coverage"] == 1.0
        assert selective["coverage_metrics"][-1]["accuracy"] == 0.5
        np.testing.assert_allclose(selective["aurc"], (0.0 + 0.0 + 1 / 3 + 0.5) / 4)

    old_group = report["breakdowns"]["old_vs_current"]["old"]
    current_group = report["breakdowns"]["old_vs_current"]["current_task"]
    assert old_group["class_ids"] == [10]
    assert current_group["class_ids"] == [20]
    assert old_group["sample_count"] == current_group["sample_count"] == 2
    assert old_group["coverage"] == current_group["coverage"] == 0.5
    assert old_group["accuracy"] == current_group["accuracy"] == 0.5

    rarity = report["breakdowns"]["rarity"]
    assert rarity["ultra_tail"]["class_ids"] == [10]
    assert rarity["head"]["class_ids"] == [20]
    assert rarity["tail"]["sample_count"] == 0
    assert rarity["medium"]["sample_count"] == 0
    assert rarity["tail"]["accuracy"] is None
    assert rarity["tail"]["mean_uncertainty"]["predictive_entropy"] is None

    loaded_json = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded_json == report
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {row["record_type"] for row in csv_rows} == {
        "score_summary",
        "risk_coverage",
        "selective_target",
        "breakdown",
    }


def test_single_error_class_is_reported_as_undefined_not_crashed(tmp_path: Path) -> None:
    for case_name, all_wrong, expected_errors in (
        ("all_correct", False, 0),
        ("all_wrong", True, 4),
    ):
        case_root = tmp_path / case_name
        case_root.mkdir()
        ensemble_path, task_file, count_file = _write_sources(
            case_root,
            include_errors=False,
            all_wrong=all_wrong,
        )
        report = run_offline_ue_audit(
            ensemble_path=ensemble_path,
            task_file=task_file,
            train_class_counts=count_file,
            output_json=case_root / "audit.json",
        )

        assert report["dataset"]["error_count"] == expected_errors
        for score_report in report["uncertainty_scores"].values():
            detection = score_report["error_detection"]
            assert detection["status"] == "undefined_single_error_class"
            assert detection["auroc"] is None
            assert detection["auprc"] is None
            assert (score_report["mean_error"] is None) is (not all_wrong)
            assert (score_report["mean_correct"] is None) is all_wrong


def test_train_data_parquet_counts_only_label_column(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "class": [10, 10, 20, 20, 20],
            "large_unused_descriptor": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).to_parquet(path, index=False)

    assert count_train_classes_from_data(path) == {10: 2, 20: 3}
