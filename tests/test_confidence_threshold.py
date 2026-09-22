from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.eval.threshold import (
    BALANCED_ACCURACY_SELECTION_RULE,
    export_frozen_threshold_artifact,
    export_threshold_report,
    evaluate_with_frozen_threshold,
    load_frozen_threshold_artifact,
    load_threshold_selection_config,
    select_confidence_threshold,
)
from src.eval.predictions import MemberPredictionArtifact, MemberPredictionContext
from src.eval.ensemble_ue import (
    aggregate_member_predictions,
    export_offline_ensemble_artifact,
)


RAW_CLASS_IDS = np.asarray([10, 30], dtype=np.int64)
SAMPLE_IDS = np.asarray(["s0", "s1", "s2", "s3"])
LABELS = np.asarray([10, 30, 30, 30], dtype=np.int64)
PROBABILITIES = np.asarray(
    [
        [0.99, 0.01],
        [0.70, 0.30],
        [0.05, 0.95],
        [0.60, 0.40],
    ],
    dtype=np.float32,
)


def _member(member_id: int, *, split: str) -> MemberPredictionArtifact:
    return MemberPredictionArtifact(
        context=MemberPredictionContext(
            run_id=f"run-{split}-{member_id}",
            method="ewc",
            method_protocol="P4",
            task_id=2,
            split=split,
            member_id=member_id,
            experiment_seed=0,
            member_seed=1000 + member_id,
        ),
        sample_ids=SAMPLE_IDS.copy(),
        labels=LABELS.copy(),
        raw_class_ids=RAW_CLASS_IDS.copy(),
        logits=np.log(PROBABILITIES).astype(np.float32),
        probabilities=PROBABILITIES.copy(),
    )


def _ensemble(split: str):
    return aggregate_member_predictions([_member(member_id, split=split) for member_id in range(3)])


def _write_config(
    path: Path,
    *,
    grid: list[float] | None = None,
    confidence_score: str | None = None,
    probability_source: str | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "candidate_grid": grid if grid is not None else [0.0, 0.5, 0.8],
        "selection_rule": {
            "name": "max_macro_f1_subject_to_min_coverage",
            "minimum_coverage": 0.25,
            "tie_breakers": ["accuracy", "coverage", "lower_threshold"],
        },
        "calibration_bins": 5,
    }
    if confidence_score is not None:
        payload["confidence_score"] = confidence_score
    if probability_source is not None:
        payload["probability_source"] = probability_source
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_paper_config(
    path: Path,
    *,
    grid: list[float],
    target_accuracy: float = 0.95,
    fallback_minimum_coverage: float = 0.5,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "confidence_score": "entropy_confidence",
                "probability_source": "raw",
                "low_threshold": 0.5,
                "candidate_grid": grid,
                "selection_rule": {
                    "name": "smallest_threshold_meeting_target_accuracy",
                    "target_accuracy": target_accuracy,
                    "fallback_minimum_coverage": fallback_minimum_coverage,
                    "tie_breakers": ["lower_threshold"],
                },
                "calibration_bins": 5,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_balanced_accuracy_config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "confidence_score": "entropy_confidence",
                "probability_source": "raw",
                "selection_source": "validation",
                "low_threshold": 0.4,
                "candidate_grid": [0.4, 0.7],
                "selection_rule": {
                    "name": BALANCED_ACCURACY_SELECTION_RULE,
                    "minimum_coverage": 0.25,
                    "tie_breakers": [
                        "macro_f1", "accuracy", "coverage", "lower_threshold"
                    ],
                },
                "calibration_bins": 5,
            }
        ),
        encoding="utf-8",
    )
    return path


def _with_entropy_confidence(values: list[float], *, split: str = "validation"):
    ensemble = _ensemble(split)
    confidence = np.asarray(values, dtype=np.float64)
    return replace(
        ensemble,
        entropy_confidence=confidence,
        normalized_entropy=1.0 - confidence,
        confidence=confidence.copy(),
    )


def _select_paper(
    tmp_path: Path,
    *,
    confidence: list[float],
    grid: list[float],
    target_accuracy: float = 0.95,
    fallback_minimum_coverage: float = 0.5,
):
    config = load_threshold_selection_config(
        _write_paper_config(
            tmp_path / "paper_threshold_config.json",
            grid=grid,
            target_accuracy=target_accuracy,
            fallback_minimum_coverage=fallback_minimum_coverage,
        )
    )
    source = tmp_path / "synthetic_oof_source.npz"
    source.write_bytes(b"synthetic selection source")
    selected = select_confidence_threshold(
        _with_entropy_confidence(confidence),
        config,
        source_ensemble_path=source,
    )
    return config, selected


def _select_and_freeze(tmp_path: Path):
    config_path = _write_config(tmp_path / "threshold_config.json")
    config = load_threshold_selection_config(config_path)
    validation = _ensemble("validation")
    ensemble_path = export_offline_ensemble_artifact(validation, tmp_path / "validation.npz")
    selected = select_confidence_threshold(
        validation,
        config,
        source_ensemble_path=ensemble_path,
    )
    frozen_path = export_frozen_threshold_artifact(selected, tmp_path / "threshold.json")
    frozen = load_frozen_threshold_artifact(frozen_path, config_path=config_path)
    return config_path, validation, selected, frozen_path, frozen


def test_selection_uses_predeclared_grid_and_records_required_metadata(tmp_path: Path) -> None:
    config_path, _, selected, frozen_path, frozen = _select_and_freeze(tmp_path)

    assert selected.selected_threshold == 0.5
    assert selected.candidate_grid == (0.0, 0.5, 0.8)
    assert selected.validation_accuracy == pytest.approx(1.0)
    assert selected.validation_macro_f1 == pytest.approx(1.0)
    assert selected.validation_coverage == pytest.approx(0.5)
    assert selected.source_split == "validation"
    assert selected.confidence_score == "entropy_confidence"
    assert selected.probability_source == "raw"
    assert selected.config_sha256 == load_threshold_selection_config(config_path).sha256
    assert selected.timestamp_utc.endswith("Z")
    assert frozen.loaded_from_path == frozen_path.resolve()

    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert payload["candidate_grid"] == [0.0, 0.5, 0.8]
    assert payload["schema_version"] == 4
    assert payload["confidence_score"] == "entropy_confidence"
    assert payload["probability_source"] == "raw"
    assert payload["selection_rule"]["name"] == "max_macro_f1_subject_to_min_coverage"
    assert payload["selection_status"] == "macro_f1_selected"
    assert payload["target_met"] is False
    assert payload["selected_threshold"] == 0.5
    assert payload["validation_metrics"] == {
        "accuracy": 1.0,
        "coverage": 0.5,
        "macro_f1": 1.0,
        "selected_count": 2,
        "total_count": 4,
    }
    assert payload["source"]["split"] == "validation"
    assert len(payload["candidate_results"]) == len(payload["candidate_grid"])
    assert [row["threshold"] for row in payload["candidate_results"]] == payload[
        "candidate_grid"
    ]


def test_legacy_config_defaults_to_entropy_confidence(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "legacy.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert "confidence_score" not in payload

    config = load_threshold_selection_config(config_path)

    assert config.confidence_score == "entropy_confidence"
    assert config.probability_source == "raw"


def test_full_p3_primary_config_matches_macro_f1_threshold_rule() -> None:
    primary_path = Path(
        "configs/eval_tddi_p3_ensemble_entropy_threshold.json"
    )
    primary = load_threshold_selection_config(primary_path)

    assert primary.confidence_score == "entropy_confidence"
    assert primary.probability_source == "raw"
    assert primary.selection_source == "oof"
    assert primary.selection_rule == "max_macro_f1_subject_to_min_coverage"
    assert primary.minimum_coverage == 0.5
    assert primary.target_accuracy is None
    assert primary.low_threshold == 0.5
    assert primary.candidate_grid == tuple(value / 100 for value in range(50, 100))


def test_balanced_accuracy_rule_prefers_class_recall_over_selected_accuracy(
    tmp_path: Path,
) -> None:
    config_path = _write_balanced_accuracy_config(tmp_path / "balanced.json")
    config = load_threshold_selection_config(config_path)
    validation = _with_entropy_confidence([0.4, 0.5, 0.9, 0.6])
    source = export_offline_ensemble_artifact(validation, tmp_path / "validation.npz")

    selected = select_confidence_threshold(
        validation, config, source_ensemble_path=source
    )

    # 0.7 selects one correct majority-class row (accuracy=1 but BA=1/6);
    # 0.4 keeps all classes and wins on balanced accuracy (2/3).
    assert selected.selection_rule == BALANCED_ACCURACY_SELECTION_RULE
    assert selected.selected_threshold == 0.4
    assert selected.selection_status == "balanced_selected"
    by_threshold = {row["threshold"]: row for row in selected.candidate_results}
    assert by_threshold[0.7]["accuracy"] == pytest.approx(1.0)
    assert by_threshold[0.4]["balanced_accuracy"] == pytest.approx(2 / 3)

    frozen_path = export_frozen_threshold_artifact(selected, tmp_path / "frozen.json")
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert payload["validation_metrics"]["balanced_accuracy"] == pytest.approx(2 / 3)
    loaded = load_frozen_threshold_artifact(frozen_path, config_path=config_path)
    report = evaluate_with_frozen_threshold(_ensemble("test"), loaded)
    assert report["threshold_score_selective_metrics"]["balanced_accuracy"] == pytest.approx(1.0)


def test_full_p3_config_rejects_seeded_validation_instead_of_oof(
    tmp_path: Path,
) -> None:
    config = load_threshold_selection_config(
        Path("configs/eval_tddi_p3_ensemble_entropy_threshold.json")
    )
    validation = _ensemble("validation")
    source = export_offline_ensemble_artifact(
        validation, tmp_path / "seeded_validation.npz"
    )

    with pytest.raises(ValueError, match="does not match config"):
        select_confidence_threshold(
            validation,
            config,
            source_ensemble_path=source,
        )


def test_explicit_max_probability_score_changes_threshold_selection(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "max_probability.json",
        confidence_score="max_probability",
    )
    config = load_threshold_selection_config(config_path)
    validation = _ensemble("validation")
    ensemble_path = export_offline_ensemble_artifact(
        validation,
        tmp_path / "validation_max_probability.npz",
    )

    selected = select_confidence_threshold(
        validation,
        config,
        source_ensemble_path=ensemble_path,
    )

    assert selected.confidence_score == "max_probability"
    assert selected.selected_threshold == 0.8
    assert len(selected.candidate_results) == len(config.candidate_grid)
    assert all(
        candidate["confidence_score"] == "max_probability"
        for candidate in selected.candidate_results
    )
    frozen_path = export_frozen_threshold_artifact(
        selected,
        tmp_path / "max_probability_frozen.json",
    )
    frozen = load_frozen_threshold_artifact(frozen_path, config_path=config_path)
    report = evaluate_with_frozen_threshold(_ensemble("test"), frozen)
    assert report["threshold_score_name"] == "max_probability"
    assert report["threshold_value"] == 0.8
    assert report["threshold_probability_source"] == "raw"
    assert report["high_confidence"]["coverage"] == 0.5
    assert report["entropy_confidence_selective_metrics"]["coverage"] == 0.25


def test_calibrated_probability_source_never_silently_uses_raw_ensemble(
    tmp_path: Path,
) -> None:
    config = load_threshold_selection_config(
        _write_config(
            tmp_path / "calibrated_source.json",
            probability_source="calibrated",
        )
    )
    validation = _ensemble("validation")
    ensemble_path = export_offline_ensemble_artifact(
        validation,
        tmp_path / "validation_raw.npz",
    )

    assert config.probability_source == "calibrated"
    with pytest.raises(ValueError, match="silently thresholding raw probabilities"):
        select_confidence_threshold(
            validation,
            config,
            source_ensemble_path=ensemble_path,
        )


def test_selection_rejects_test_split(tmp_path: Path) -> None:
    config = load_threshold_selection_config(_write_config(tmp_path / "config.json"))
    test_ensemble = _ensemble("test")
    test_path = export_offline_ensemble_artifact(test_ensemble, tmp_path / "test.npz")

    with pytest.raises(ValueError, match="validation-only"):
        select_confidence_threshold(test_ensemble, config, source_ensemble_path=test_path)


def test_test_evaluation_requires_frozen_artifact_loaded_from_disk(tmp_path: Path) -> None:
    _, _, selected, _, frozen = _select_and_freeze(tmp_path)
    test_ensemble = _ensemble("test")

    with pytest.raises(ValueError, match="loaded from its artifact file"):
        evaluate_with_frozen_threshold(test_ensemble, selected)

    report = evaluate_with_frozen_threshold(test_ensemble, frozen)
    assert report["evaluation_split"] == "test"
    assert report["schema_version"] == 3
    assert report["threshold_score_name"] == "entropy_confidence"
    assert report["threshold_value"] == 0.5
    assert report["threshold_probability_source"] == "raw"
    assert report["threshold"]["score_name"] == "entropy_confidence"
    assert report["threshold"]["value"] == 0.5
    assert report["high_confidence"] == {
        "selected_count": 2,
        "total_count": 4,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "coverage": 0.5,
    }

    full = report["full_set"]
    assert full["sample_count"] == 4
    assert full["accuracy"] == pytest.approx(0.5)
    assert full["macro_f1"] == pytest.approx(0.5)
    assert full["ece"] == pytest.approx(0.34)
    assert full["negative_log_likelihood"] == pytest.approx(
        -np.log(np.asarray([0.99, 0.30, 0.95, 0.40])).mean()
    )
    assert full["brier_score"] == pytest.approx(
        np.mean([2 * 0.01**2, 2 * 0.70**2, 2 * 0.05**2, 2 * 0.60**2])
    )
    assert report["entropy_confidence_selective_metrics"] == {
        "threshold_score_name": "entropy_confidence",
        "probability_source": "raw",
        "threshold_value": 0.5,
        "selected_count": 2,
        "total_count": 4,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "coverage": 0.5,
    }
    assert report["selection_candidates"]["source_split"] == "validation"
    assert report["selection_candidates"]["confidence_score"] == "entropy_confidence"
    assert report["selection_candidates"]["candidate_grid"] == [0.0, 0.5, 0.8]
    assert len(report["selection_candidates"]["results"]) == 3
    report_path = export_threshold_report(report, tmp_path / "test_report.json")
    assert json.loads(report_path.read_text(encoding="utf-8"))["evaluation_split"] == "test"


def test_frozen_threshold_load_rejects_config_change_and_test_provenance(tmp_path: Path) -> None:
    _, _, _, frozen_path, _ = _select_and_freeze(tmp_path)
    changed_config = _write_config(tmp_path / "changed_config.json", grid=[0.0, 0.4, 0.8])
    with pytest.raises(ValueError, match="config hash"):
        load_frozen_threshold_artifact(frozen_path, config_path=changed_config)

    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload["source"]["split"] = "test"
    tampered_path = tmp_path / "tampered_threshold.json"
    tampered_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be validation"):
        load_frozen_threshold_artifact(tampered_path)


def test_legacy_frozen_threshold_load_defaults_to_entropy_confidence(tmp_path: Path) -> None:
    config_path, _, _, frozen_path, _ = _select_and_freeze(tmp_path)
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload.pop("confidence_score")
    payload.pop("probability_source")
    payload.pop("selection_status")
    payload.pop("target_met")
    payload["selection_rule"].pop("fallback_minimum_coverage")
    payload["source"].pop("ensemble_mode")
    payload["source"].pop("member_ids")
    payload["source"].pop("study_provenance")
    for candidate in payload["candidate_results"]:
        candidate.pop("confidence_score")
    legacy_path = tmp_path / "legacy_frozen_threshold.json"
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_frozen_threshold_artifact(legacy_path, config_path=config_path)

    assert loaded.confidence_score == "entropy_confidence"
    assert loaded.probability_source == "raw"


def test_frozen_threshold_rejects_mismatched_test_context(tmp_path: Path) -> None:
    _, _, _, _, frozen = _select_and_freeze(tmp_path)
    mismatched = replace(
        _ensemble("test"),
        context=replace(_ensemble("test").context, experiment_seed=9),
    )
    with pytest.raises(ValueError, match="experiment_seed"):
        evaluate_with_frozen_threshold(mismatched, frozen)


def test_paper_rule_uses_smallest_target_threshold_without_primary_coverage_gate(
    tmp_path: Path,
) -> None:
    _, selected = _select_paper(
        tmp_path,
        confidence=[0.8, 0.9, 0.7, 0.6],
        grid=[0.6, 0.7, 0.8, 0.9],
        target_accuracy=2 / 3,
        # The selected primary candidate has coverage 0.75, proving this
        # fallback-only constraint is not imposed on the primary rule.
        fallback_minimum_coverage=0.99,
    )

    assert selected.selected_threshold == 0.7
    assert selected.selection_status == "target_met"
    assert selected.target_met is True
    assert selected.validation_coverage == pytest.approx(0.75)


def test_paper_rule_fallback_maximizes_accuracy_then_coverage(tmp_path: Path) -> None:
    _, selected = _select_paper(
        tmp_path,
        confidence=[0.8, 0.9, 0.6, 0.7],
        grid=[0.6, 0.7, 0.8, 0.9],
    )

    # Thresholds 0.6 and 0.8 both have accuracy 0.5; 0.6 wins on coverage.
    assert selected.selected_threshold == 0.6
    assert selected.validation_accuracy == pytest.approx(0.5)
    assert selected.validation_coverage == pytest.approx(1.0)
    assert selected.selection_status == "fallback"
    assert selected.target_met is False


def test_paper_rule_fallback_final_tie_prefers_lower_threshold(tmp_path: Path) -> None:
    _, selected = _select_paper(
        tmp_path,
        confidence=[0.9, 0.9, 0.7, 0.7],
        grid=[0.75, 0.8],
    )

    assert selected.selected_threshold == 0.75
    assert selected.selection_status == "fallback"


def test_zero_coverage_produces_frozen_no_selection_and_does_not_filter_test(
    tmp_path: Path,
) -> None:
    config, selected = _select_paper(
        tmp_path,
        confidence=[0.8, 0.8, 0.8, 0.8],
        grid=[0.99],
    )
    assert selected.selected_threshold is None
    assert selected.selection_status == "no_selection"
    assert selected.target_met is False
    assert selected.validation_selected_count == 0
    assert selected.validation_accuracy is None
    assert selected.candidate_results[0]["selected_count"] == 0
    assert selected.candidate_results[0]["accuracy"] is None

    frozen_path = export_frozen_threshold_artifact(
        selected, tmp_path / "no_selection.json"
    )
    loaded = load_frozen_threshold_artifact(
        frozen_path, config_path=config.source_path
    )
    report = evaluate_with_frozen_threshold(_ensemble("test"), loaded)

    assert report["selection_status"] == "no_selection"
    assert report["threshold_value"] is None
    assert report["threshold"]["selection_applied"] is False
    assert report["high_confidence"]["selected_count"] == 4
    assert report["high_confidence"]["coverage"] == pytest.approx(1.0)
    assert report["threshold_score_selective_metrics"]["selection_applied"] is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("task_id", 99, "task_id"),
        ("raw_class_ids", (10, 20), "raw_class_ids"),
        ("source_ensemble_mode", "stratified_3fold", "ensemble_mode"),
    ],
)
def test_frozen_threshold_rejects_task_class_and_partition_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    _, _, _, _, frozen = _select_and_freeze(tmp_path)
    mismatched = replace(frozen, **{field: value})

    with pytest.raises(ValueError, match=error):
        evaluate_with_frozen_threshold(_ensemble("test"), mismatched)


def test_frozen_oof_provenance_requires_matching_test_study_contract(
    tmp_path: Path,
) -> None:
    _, _, _, _, frozen = _select_and_freeze(tmp_path)
    linked = replace(
        frozen,
        assignment_sha256="a" * 64,
        fold_manifest_sha256="b" * 64,
        task_file_sha256="c" * 64,
        study_contract_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="study_provenance"):
        evaluate_with_frozen_threshold(_ensemble("test"), linked)
