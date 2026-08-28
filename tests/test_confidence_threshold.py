from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.eval.calibration_metrics import (
    expected_calibration_error,
    map_raw_labels_to_indices,
    multiclass_brier_score,
    negative_log_likelihood,
)
from src.eval.confidence_threshold import (
    export_frozen_threshold_artifact,
    export_threshold_report,
    evaluate_with_frozen_threshold,
    load_frozen_threshold_artifact,
    load_threshold_selection_config,
    select_confidence_threshold,
)
from src.eval.member_predictions import MemberPredictionArtifact, MemberPredictionContext
from src.eval.offline_ensemble import (
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


def _write_config(path: Path, *, grid: list[float] | None = None) -> Path:
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
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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
    assert selected.config_sha256 == load_threshold_selection_config(config_path).sha256
    assert selected.timestamp_utc.endswith("Z")
    assert frozen.loaded_from_path == frozen_path.resolve()

    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert payload["candidate_grid"] == [0.0, 0.5, 0.8]
    assert payload["selection_rule"]["name"] == "max_macro_f1_subject_to_min_coverage"
    assert payload["selected_threshold"] == 0.5
    assert payload["validation_metrics"] == {
        "accuracy": 1.0,
        "coverage": 0.5,
        "macro_f1": 1.0,
        "selected_count": 2,
        "total_count": 4,
    }
    assert payload["source"]["split"] == "validation"


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
    assert report["threshold"]["value"] == 0.5
    assert report["high_confidence"] == {
        "selected_count": 2,
        "total_count": 4,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "coverage": 0.5,
    }

    dense_labels = map_raw_labels_to_indices(LABELS, RAW_CLASS_IDS)
    full = report["full_set"]
    assert full["sample_count"] == 4
    assert full["accuracy"] == pytest.approx(0.5)
    assert full["macro_f1"] == pytest.approx(0.5)
    assert full["ece"] == pytest.approx(
        expected_calibration_error(PROBABILITIES, dense_labels, num_bins=5)
    )
    assert full["negative_log_likelihood"] == pytest.approx(
        negative_log_likelihood(PROBABILITIES, dense_labels)
    )
    assert full["brier_score"] == pytest.approx(
        multiclass_brier_score(PROBABILITIES, dense_labels)
    )
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


def test_frozen_threshold_rejects_mismatched_test_context(tmp_path: Path) -> None:
    _, _, _, _, frozen = _select_and_freeze(tmp_path)
    mismatched = replace(
        _ensemble("test"),
        context=replace(_ensemble("test").context, experiment_seed=9),
    )
    with pytest.raises(ValueError, match="experiment_seed"):
        evaluate_with_frozen_threshold(mismatched, frozen)
