"""Official OOF entropy/Macro-F1 selection, frozen provenance and rejection tests."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from src.eval.threshold import (
    export_frozen_threshold_artifact, load_frozen_threshold_artifact,
    load_threshold_selection_config, select_confidence_threshold, evaluate_with_frozen_threshold,
)
from src.eval.ensemble_ue import aggregate_member_predictions, export_offline_ensemble_artifact
from tests.test_stratified_ensemble_mode import _provenanced_fold_members


def setup_threshold(tmp_path):
    ensemble = aggregate_member_predictions(_provenanced_fold_members(), ensemble_mode="stratified_3fold")
    source = export_offline_ensemble_artifact(ensemble, tmp_path / "oof.npz")
    config = load_threshold_selection_config("configs/eval_macro_f1_entropy.json")
    selected = select_confidence_threshold(ensemble, config, source_ensemble_path=source)
    return ensemble, config, selected


def test_official_roundtrip_and_frozen_evaluation(tmp_path):
    ensemble, config, selected = setup_threshold(tmp_path)
    assert selected.selection_status == "macro_f1_selected"
    assert selected.selected_threshold == 0.5
    path = export_frozen_threshold_artifact(selected, tmp_path / "threshold.json")
    loaded = load_frozen_threshold_artifact(path, config_path=config.source_path)
    report = evaluate_with_frozen_threshold(ensemble, loaded)
    assert report["threshold"]["source_split"] == "oof"
    assert report["threshold_score_name"] == "entropy_confidence"
    assert report["full_set"]["sample_count"] == 6
    assert "target_met" not in report
    assert loaded == selected


def test_no_eligible_threshold_is_explicit(tmp_path):
    ensemble, config, _ = setup_threshold(tmp_path)
    cfg = replace(config, candidate_grid=(0.99,))
    selected = select_confidence_threshold(ensemble, cfg, source_ensemble_path=tmp_path / "oof.npz")
    assert selected.selected_threshold is None and selected.selection_status == "no_selection"
    path = export_frozen_threshold_artifact(selected, tmp_path / "none.json")
    report = evaluate_with_frozen_threshold(ensemble, load_frozen_threshold_artifact(path))
    assert report["threshold"]["selection_applied"] is False
    assert report["threshold_score_selective_metrics"]["coverage"] == 1


@pytest.mark.parametrize("change", [
    {"confidence_score": "max_probability"}, {"probability_source": "calibrated"},
    {"selection_source": "validation"},
    {"selection_rule": {"name": "smallest_threshold_meeting_target_accuracy", "target_accuracy": 0.95}},
    {"selection_rule": {"name": "max_balanced_accuracy_subject_to_min_coverage"}},
    {"candidate_grid": [0.8, 0.5]}, {"candidate_grid": [0.5, 0.5]},
])
def test_retired_policies_and_invalid_grids_rejected(tmp_path, change):
    payload = json.loads(Path("configs/eval_macro_f1_entropy.json").read_text())
    payload.update(change)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_threshold_selection_config(path)


def test_source_and_test_leakage_guards(tmp_path):
    ensemble, config, selected = setup_threshold(tmp_path)
    test = replace(ensemble, context=replace(ensemble.context, split="test"))
    with pytest.raises(ValueError, match="OOF"):
        select_confidence_threshold(test, config, source_ensemble_path=tmp_path / "oof.npz")
    with pytest.raises(ValueError, match="loaded"):
        evaluate_with_frozen_threshold(test, selected)
    path = export_frozen_threshold_artifact(selected, tmp_path / "frozen.json")
    frozen = load_frozen_threshold_artifact(path)
    bad = replace(ensemble, context=replace(ensemble.context, task_id=99))
    with pytest.raises(ValueError, match="provenance"):
        evaluate_with_frozen_threshold(bad, frozen)


def test_rejects_old_schema_and_tampered_selection(tmp_path):
    _, _, selected = setup_threshold(tmp_path)
    path = export_frozen_threshold_artifact(selected, tmp_path / "frozen.json")
    original = json.loads(path.read_text())
    for change in ({"schema_version": 4}, {"selected_threshold": 0.99}):
        path.write_text(json.dumps({**original, **change}))
        with pytest.raises(ValueError):
            load_frozen_threshold_artifact(path)


def test_macro_f1_objective_and_ties(tmp_path):
    ensemble, config, _ = setup_threshold(tmp_path)
    # Synthetic confidence separates an incorrect group from a correct group.
    labels = np.array([0, 0, 1, 1, 0, 1])
    predictions = np.array([0, 0, 1, 0, 1, 0])
    confidence = np.array([.9, .9, .9, .6, .6, .6])
    ensemble = replace(ensemble, labels=labels, predictions=predictions,
                       entropy_confidence=confidence, normalized_entropy=1-confidence, confidence=confidence)
    config = replace(config, candidate_grid=(.5, .6, .8, .9), minimum_coverage=.5)
    selected = select_confidence_threshold(ensemble, config, source_ensemble_path=tmp_path / "oof.npz")
    assert selected.selected_threshold == .8
    assert selected.validation_macro_f1 == 1.
