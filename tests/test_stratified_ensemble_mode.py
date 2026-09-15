from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ddi_dataset import load_split_arrays
from src.data.stratified_folds import (
    build_stratified_fold_assignments,
    select_development_fold,
)
from src.eval.threshold import (
    evaluate_with_frozen_threshold,
    export_frozen_threshold_artifact,
    load_frozen_threshold_artifact,
    load_threshold_selection_config,
    select_confidence_threshold,
)
from src.eval.predictions import (
    MemberPredictionArtifact,
    MemberPredictionContext,
    PredictionProvenance,
    partition_identity_sha256,
)
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


def _provenanced_fold_members() -> list[MemberPredictionArtifact]:
    rows = {0: ["c", "f"], 1: ["a", "d"], 2: ["b", "e"]}
    all_ids = np.asarray(["c", "f", "a", "d", "b", "e"])
    all_labels = np.zeros(6, dtype=np.int64)
    all_folds = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int16)
    oof_sha = partition_identity_sha256(all_ids, all_labels, all_folds)
    members = []
    for fold_id, sample_ids in rows.items():
        artifact = _fold_member(fold_id, sample_ids)
        fold_ids = np.full(2, fold_id, dtype=np.int16)
        provenance = PredictionProvenance(
            assignment_sha256="a" * 64,
            fold_manifest_sha256="b" * 64,
            task_file_sha256="c" * 64,
            study_contract_sha256="d" * 64,
            preprocessing_policy="task0_standard_frozen",
            # Per-member scaler hashes are expected to differ.
            preprocessing_sha256=str(fold_id + 1) * 64,
            preprocessing_member_id=fold_id,
            ranking_policy="raw_sample_normalized_class_mean_control_v1",
            buffer_policy="stratified_fraction_v1",
            member_memory_budget=9260 if fold_id == 0 else 9259,
            global_memory_budget=27778,
            expected_partition_rows=2,
            expected_partition_sha256=partition_identity_sha256(
                artifact.sample_ids, artifact.labels, fold_ids
            ),
            expected_oof_rows=6,
            expected_oof_sha256=oof_sha,
        )
        members.append(replace(artifact, fold_ids=fold_ids, provenance=provenance))
    return members


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
    np.testing.assert_array_equal(oof.prediction_count, np.ones(6, dtype=np.int16))
    assert "mutual_information" in oof.unavailable_metrics
    assert np.isnan(oof.mutual_information).all()
    assert np.isnan(oof.mean_probability_variance).all()
    assert np.isnan(oof.pairwise_disagreement).all()

    config_path = tmp_path / "threshold.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "confidence_score": "entropy_confidence",
                "probability_source": "raw",
                "selection_source": "oof",
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


def test_oof_provenance_proves_coverage_and_allows_different_member_scalers() -> None:
    oof = aggregate_member_predictions(
        _provenanced_fold_members(), ensemble_mode="stratified_3fold"
    )

    assert oof.row_count == 6
    assert oof.member_provenance is not None
    assert len({item.preprocessing_sha256 for item in oof.member_provenance}) == 3
    np.testing.assert_array_equal(oof.prediction_count, 1)


def test_oof_rejects_incomplete_wrong_fold_and_wrong_study_contract() -> None:
    incomplete = _provenanced_fold_members()
    first = incomplete[0]
    incomplete[0] = replace(
        first,
        sample_ids=first.sample_ids[:1], labels=first.labels[:1],
        logits=first.logits[:1], probabilities=first.probabilities[:1],
        fold_ids=first.fold_ids[:1],
        provenance=replace(
            first.provenance,
            expected_partition_rows=1,
            expected_partition_sha256=partition_identity_sha256(
                first.sample_ids[:1], first.labels[:1], first.fold_ids[:1]
            ),
        ),
    )
    with pytest.raises(ValueError, match="coverage"):
        aggregate_member_predictions(incomplete, ensemble_mode="stratified_3fold")

    wrong_fold = _provenanced_fold_members()
    wrong_fold[1] = replace(wrong_fold[1], fold_ids=np.zeros(2, dtype=np.int16))
    with pytest.raises(ValueError, match="wrong held-out fold"):
        aggregate_member_predictions(wrong_fold, ensemble_mode="stratified_3fold")

    wrong_contract = _provenanced_fold_members()
    wrong_contract[2] = replace(
        wrong_contract[2],
        provenance=replace(
            wrong_contract[2].provenance, study_contract_sha256="e" * 64
        ),
    )
    with pytest.raises(ValueError, match="same partition/study contract"):
        aggregate_member_predictions(wrong_contract, ensemble_mode="stratified_3fold")


def test_common_test_means_three_predictions_under_shared_study_contract() -> None:
    members = _provenanced_fold_members()
    sample_ids = np.asarray(["test-a", "test-b"])
    labels = np.zeros(2, dtype=np.int64)
    fold_ids = np.full(2, -1, dtype=np.int16)
    digest = partition_identity_sha256(sample_ids, labels, fold_ids)
    converted = []
    for artifact in members:
        converted.append(replace(
            artifact,
            context=replace(artifact.context, split="test"),
            sample_ids=sample_ids.copy(), labels=labels.copy(), fold_ids=fold_ids.copy(),
            provenance=replace(
                artifact.provenance, expected_partition_rows=2,
                expected_partition_sha256=digest,
                expected_oof_rows=None, expected_oof_sha256=None,
            ),
        ))

    ensemble = aggregate_member_predictions(converted, ensemble_mode="stratified_3fold")
    np.testing.assert_array_equal(ensemble.prediction_count, 3)
    assert not ensemble.unavailable_metrics
    assert np.isfinite(ensemble.mutual_information).all()


def test_frozen_oof_threshold_links_to_different_common_test_partition(
    tmp_path: Path,
) -> None:
    members = _provenanced_fold_members()
    oof = aggregate_member_predictions(members, ensemble_mode="stratified_3fold")
    oof_path = export_offline_ensemble_artifact(oof, tmp_path / "oof.npz")
    config_path = tmp_path / "threshold.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "confidence_score": "entropy_confidence",
                "probability_source": "raw",
                "selection_source": "oof",
                "low_threshold": 0.5,
                "candidate_grid": [0.5, 0.9, 0.99],
                "selection_rule": {
                    "name": "smallest_threshold_meeting_target_accuracy",
                    "target_accuracy": 0.95,
                    "fallback_minimum_coverage": 0.5,
                    "tie_breakers": ["lower_threshold"],
                },
                "calibration_bins": 5,
            }
        ),
        encoding="utf-8",
    )
    selected = select_confidence_threshold(
        oof,
        load_threshold_selection_config(config_path),
        source_ensemble_path=oof_path,
    )
    frozen_path = export_frozen_threshold_artifact(
        selected, tmp_path / "frozen_threshold.json"
    )
    frozen = load_frozen_threshold_artifact(frozen_path, config_path=config_path)

    test_ids = np.asarray(["test-a", "test-b"])
    test_labels = np.zeros(2, dtype=np.int64)
    test_folds = np.full(2, -1, dtype=np.int16)
    test_digest = partition_identity_sha256(test_ids, test_labels, test_folds)
    test_members = [
        replace(
            artifact,
            context=replace(artifact.context, split="test"),
            sample_ids=test_ids.copy(),
            labels=test_labels.copy(),
            fold_ids=test_folds.copy(),
            provenance=replace(
                artifact.provenance,
                expected_partition_rows=2,
                expected_partition_sha256=test_digest,
                expected_oof_rows=None,
                expected_oof_sha256=None,
            ),
        )
        for artifact in members
    ]
    test_ensemble = aggregate_member_predictions(
        test_members, ensemble_mode="stratified_3fold"
    )

    assert set(oof.sample_ids).isdisjoint(set(test_ensemble.sample_ids))
    report = evaluate_with_frozen_threshold(test_ensemble, frozen)
    assert report["evaluation_split"] == "test"
    assert report["threshold"]["source_split"] == "oof"
    assert report["selection_status"] == "target_met"
