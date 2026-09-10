from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ddi_dataset import load_split_arrays
from src.data.stratified_folds import (
    build_stratified_fold_assignments,
    select_development_fold,
)
from src.eval.threshold import (
    load_threshold_selection_config,
    select_confidence_threshold,
)
from src.eval.predictions import MemberPredictionArtifact, MemberPredictionContext
from src.eval.ensemble_ue import aggregate_member_predictions, export_offline_ensemble_artifact


def _write_split(path: Path, prefix: str, labels: list[int]) -> None:
    frame = pd.DataFrame(
        {
            "feature": np.arange(len(labels), dtype=np.float32),
            "class": labels,
            "drugid-drug_a": [f"{prefix}A{i}" for i in range(len(labels))],
            "drugid-drug_b": [f"{prefix}B{i}" for i in range(len(labels))],
        }
    )
    frame.to_parquet(path, index=False)


def test_stratified_fold_assignment_is_fixed_disjoint_and_complete(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    _write_split(train, "T", [0, 1, 2] * 6)
    _write_split(validation, "V", [0, 1, 2] * 3)
    first = build_stratified_fold_assignments((train, validation), fold_count=3, seed=42)
    second = build_stratified_fold_assignments((train, validation), fold_count=3, seed=42)
    np.testing.assert_array_equal(first.sorted_sample_ids, second.sorted_sample_ids)
    np.testing.assert_array_equal(first.sorted_fold_ids, second.sorted_fold_ids)

    parts = [
        load_split_arrays(
            path,
            ["feature"],
            include_metadata=True,
            meta_cols=["drugid-drug_a", "drugid-drug_b"],
        )
        for path in (train, validation)
    ]
    held_out = select_development_fold(parts, first, fold_id=1, held_out=True)
    training = select_development_fold(parts, first, fold_id=1, held_out=False)
    assert held_out.labels.shape[0] + training.labels.shape[0] == 27
    assert np.bincount(held_out.labels, minlength=3).tolist() == [3, 3, 3]


def _fold_member(fold_id: int, sample_ids: list[str]) -> MemberPredictionArtifact:
    probabilities = np.asarray([[0.98, 0.02] for _ in sample_ids], dtype=np.float32)
    labels = np.zeros(len(sample_ids), dtype=np.int64)
    return MemberPredictionArtifact(
        context=MemberPredictionContext(
            run_id=f"fold-{fold_id}",
            method="replay_distill_fixed_budget_uniform",
            method_protocol="replay_distill_fixed_budget_uniform",
            task_id=0,
            split="validation",
            member_id=fold_id,
            experiment_seed=0,
            member_seed=100 + fold_id,
            ensemble_mode="stratified_3fold",
            fold_id=fold_id,
            fold_count=3,
            fold_seed=42,
        ),
        sample_ids=np.asarray(sample_ids),
        labels=labels,
        raw_class_ids=np.asarray([0, 1], dtype=np.int64),
        logits=np.log(probabilities).astype(np.float32),
        probabilities=probabilities,
    )


def test_three_disjoint_validation_folds_become_one_oof_artifact(tmp_path: Path) -> None:
    members = [
        _fold_member(0, ["c", "f"]),
        _fold_member(1, ["a", "d"]),
        _fold_member(2, ["b", "e"]),
    ]
    oof = aggregate_member_predictions(members, ensemble_mode="stratified_3fold")
    assert oof.context.split == "oof"
    assert oof.context.ensemble_mode == "stratified_3fold"
    assert oof.sample_ids.tolist() == ["a", "b", "c", "d", "e", "f"]
    np.testing.assert_allclose(oof.entropy_confidence, 1.0 - oof.normalized_entropy)
    np.testing.assert_array_equal(oof.mutual_information, 0.0)

    config_path = tmp_path / "threshold.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "confidence_score": "entropy_confidence",
                "probability_source": "raw",
                "low_threshold": 0.5,
                "candidate_grid": [0.5, 0.9, 0.99],
                "selection_rule": {
                    "name": "smallest_threshold_meeting_target_accuracy",
                    "target_accuracy": 0.95,
                    "tie_breakers": ["lower_threshold"],
                },
                "calibration_bins": 15,
            }
        ),
        encoding="utf-8",
    )
    oof_path = export_offline_ensemble_artifact(oof, tmp_path / "oof.npz")
    selected = select_confidence_threshold(
        oof,
        load_threshold_selection_config(config_path),
        source_ensemble_path=oof_path,
    )
    assert selected.source_split == "oof"
    assert selected.selected_threshold == 0.5


def test_oof_merge_rejects_overlapping_fold_samples() -> None:
    members = [
        _fold_member(0, ["same"]),
        _fold_member(1, ["same"]),
        _fold_member(2, ["unique"]),
    ]
    try:
        aggregate_member_predictions(members, ensemble_mode="stratified_3fold")
    except ValueError as error:
        assert "overlap" in str(error).lower()
    else:
        raise AssertionError("Overlapping OOF samples were accepted.")
