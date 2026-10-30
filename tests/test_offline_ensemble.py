from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.eval.predictions import (
    MemberPredictionArtifact,
    MemberPredictionContext,
)
from src.eval.ensemble_ue import (
    _normalize_mi_by_member_count,
    aggregate_member_predictions,
    export_offline_ensemble_artifact,
    load_offline_ensemble_artifact,
)


RAW_CLASS_IDS = np.asarray([10, 30], dtype=np.int64)
SAMPLE_IDS = np.asarray(["A|B", "C|D"])
LABELS = np.asarray([10, 30], dtype=np.int64)


def _member(
    member_id: int,
    probabilities: np.ndarray,
    *,
    protocol: str = "ewc_natural_sampling",
    experiment_seed: int = 0,
    task_id: int = 1,
) -> MemberPredictionArtifact:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    logits = np.log(probabilities).astype(np.float32)
    return MemberPredictionArtifact(
        context=MemberPredictionContext(
            run_id=f"run-member-{member_id}",
            method="ewc",
            method_protocol=protocol,
            task_id=task_id,
            split="validation",
            member_id=member_id,
            experiment_seed=experiment_seed,
            member_seed=100 + member_id,
        ),
        sample_ids=SAMPLE_IDS.copy(),
        labels=LABELS.copy(),
        raw_class_ids=RAW_CLASS_IDS.copy(),
        logits=logits,
        probabilities=probabilities,
    )


def _numerical_members() -> list[MemberPredictionArtifact]:
    return [
        _member(0, [[0.9, 0.1], [0.8, 0.2]]),
        _member(1, [[0.1, 0.9], [0.8, 0.2]]),
        _member(2, [[0.5, 0.5], [0.8, 0.2]]),
    ]


def _three_different_class_members(
    class_count: int,
) -> list[MemberPredictionArtifact]:
    raw_class_ids = np.arange(100, 100 + class_count, dtype=np.int64)
    members = []
    for member_id in range(3):
        probabilities = np.zeros((1, class_count), dtype=np.float32)
        probabilities[0, member_id] = 1.0
        members.append(
            MemberPredictionArtifact(
                context=MemberPredictionContext(
                    run_id=f"run-member-{member_id}",
                    method="ewc",
                    method_protocol="ewc_natural_sampling",
                    task_id=1,
                    split="validation",
                    member_id=member_id,
                    experiment_seed=0,
                    member_seed=100 + member_id,
                ),
                sample_ids=np.asarray(["A|B"]),
                labels=np.asarray([raw_class_ids[0]], dtype=np.int64),
                raw_class_ids=raw_class_ids,
                logits=np.zeros_like(probabilities),
                probabilities=probabilities,
            )
        )
    return members


def _binary_entropy(probability: float) -> float:
    return -probability * np.log(probability) - (1.0 - probability) * np.log(
        1.0 - probability
    )


def test_offline_ensemble_matches_hand_computed_probability_and_ue_values() -> None:
    artifact = aggregate_member_predictions(
        _numerical_members(),
        source_artifact_paths=["m0.npz", "m1.npz", "m2.npz"],
    )
    log_two = np.log(2.0)
    entropy_point_nine = _binary_entropy(0.9)
    entropy_point_eight = _binary_entropy(0.8)
    expected_member_entropy_first = (2.0 * entropy_point_nine + log_two) / 3.0
    expected_mi_first = log_two - expected_member_entropy_first

    np.testing.assert_allclose(
        artifact.probabilities,
        [[0.5, 0.5], [0.8, 0.2]],
        rtol=1e-7,
        atol=1e-7,
    )
    np.testing.assert_array_equal(artifact.predictions, [10, 10])
    np.testing.assert_allclose(
        artifact.predictive_entropy,
        [log_two, entropy_point_eight],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        artifact.expected_member_entropy,
        [expected_member_entropy_first, entropy_point_eight],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        artifact.mutual_information,
        [expected_mi_first, 0.0],
        rtol=1e-7,
        atol=1e-12,
    )
    np.testing.assert_allclose(artifact.normalized_entropy[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(
        artifact.normalized_mi[0],
        expected_mi_first / log_two,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        artifact.mean_probability_variance,
        [0.10666666666666667, 0.0],
        rtol=1e-7,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        artifact.total_probability_variance,
        [0.21333333333333335, 0.0],
        rtol=1e-7,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        artifact.member_normalized_mi,
        [expected_mi_first / np.log(3.0), 0.0],
        rtol=1e-7,
        atol=1e-12,
    )
    np.testing.assert_allclose(artifact.entropy_confidence[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(artifact.confidence[0], 0.0, atol=1e-12)
    np.testing.assert_array_equal(artifact.confidence, artifact.entropy_confidence)
    np.testing.assert_allclose(artifact.max_probability, [0.5, 0.8], rtol=1e-7)
    assert artifact.entropy_confidence[0] != artifact.max_probability[0]
    assert artifact.context.member_count == 3
    np.testing.assert_allclose(artifact.pairwise_disagreement, [2.0 / 3.0, 0.0])


def test_aggregation_uses_mean_probabilities_not_mean_logits() -> None:
    members = [
        _member(0, [[0.99, 0.01], [0.5, 0.5]]),
        _member(1, [[0.6, 0.4], [0.5, 0.5]]),
        _member(2, [[0.6, 0.4], [0.5, 0.5]]),
    ]
    artifact = aggregate_member_predictions(members)
    arithmetic_mean = np.mean([member.probabilities for member in members], axis=0)
    mean_logits = np.mean([member.logits for member in members], axis=0)
    shifted = mean_logits - mean_logits.max(axis=1, keepdims=True)
    softmax_of_mean_logits = np.exp(shifted)
    softmax_of_mean_logits /= softmax_of_mean_logits.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(artifact.probabilities, arithmetic_mean, rtol=1e-7, atol=1e-7)
    assert not np.allclose(artifact.probabilities[0], softmax_of_mean_logits[0])


def test_offline_ensemble_rejects_every_required_alignment_mismatch() -> None:
    base = _numerical_members()
    cases: list[tuple[str, list[MemberPredictionArtifact], str]] = []

    protocol_members = _numerical_members()
    protocol_members[2] = replace(
        protocol_members[2],
        context=replace(protocol_members[2].context, method_protocol="wrong-protocol"),
    )
    cases.append(("protocol", protocol_members, "method_protocol mismatch"))

    seed_members = _numerical_members()
    seed_members[1] = replace(
        seed_members[1],
        context=replace(seed_members[1].context, experiment_seed=9),
    )
    cases.append(("experiment seed", seed_members, "experiment_seed mismatch"))

    task_members = _numerical_members()
    task_members[1] = replace(
        task_members[1],
        context=replace(task_members[1].context, task_id=2),
    )
    cases.append(("task", task_members, "task_id mismatch"))

    sample_members = _numerical_members()
    sample_members[2] = replace(sample_members[2], sample_ids=SAMPLE_IDS[::-1].copy())
    cases.append(("sample permutation", sample_members, "sample_ids mismatch or permutation"))

    label_members = _numerical_members()
    label_members[2] = replace(label_members[2], labels=LABELS[::-1].copy())
    cases.append(("labels", label_members, "labels mismatch"))

    class_members = _numerical_members()
    class_members[2] = replace(
        class_members[2],
        raw_class_ids=RAW_CLASS_IDS[::-1].copy(),
    )
    cases.append(("class order", class_members, "raw_class_ids mismatch or permutation"))

    member_id_members = _numerical_members()
    member_id_members[2] = replace(
        member_id_members[2],
        context=replace(member_id_members[2].context, member_id=1),
    )
    cases.append(("member ID", member_id_members, "Member IDs must be distinct"))

    for _, members, message in cases:
        with pytest.raises(ValueError, match=message):
            aggregate_member_predictions(members)

    with pytest.raises(ValueError, match="exactly 3"):
        aggregate_member_predictions(base[:2])


def test_identical_members_have_near_zero_epistemic_uncertainty() -> None:
    probabilities = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
    artifact = aggregate_member_predictions(
        [_member(member_id, probabilities) for member_id in range(3)]
    )

    np.testing.assert_allclose(artifact.mutual_information, 0.0, atol=1e-12)
    np.testing.assert_allclose(artifact.normalized_mi, 0.0, atol=1e-12)
    np.testing.assert_allclose(artifact.member_normalized_mi, 0.0, atol=1e-12)
    np.testing.assert_allclose(artifact.mean_probability_variance, 0.0, atol=1e-12)
    np.testing.assert_allclose(artifact.total_probability_variance, 0.0, atol=1e-12)
    np.testing.assert_allclose(artifact.pairwise_disagreement, 0.0, atol=0.0)


def test_three_members_predicting_three_classes_have_hand_computed_maximum_mi() -> None:
    artifact = aggregate_member_predictions(_three_different_class_members(3))

    np.testing.assert_allclose(artifact.probabilities, [[1.0 / 3.0] * 3], rtol=1e-7)
    np.testing.assert_allclose(artifact.predictive_entropy, np.log(3.0), rtol=1e-7)
    np.testing.assert_allclose(artifact.expected_member_entropy, 0.0, atol=0.0)
    np.testing.assert_allclose(artifact.mutual_information, np.log(3.0), rtol=1e-7)
    np.testing.assert_allclose(artifact.member_normalized_mi, 1.0, rtol=1e-7)
    np.testing.assert_allclose(artifact.pairwise_disagreement, 1.0, atol=0.0)


def test_member_normalized_mi_is_stable_when_class_count_changes() -> None:
    three_classes = aggregate_member_predictions(_three_different_class_members(3))
    six_classes = aggregate_member_predictions(_three_different_class_members(6))

    np.testing.assert_allclose(
        three_classes.member_normalized_mi,
        six_classes.member_normalized_mi,
        rtol=1e-7,
    )
    np.testing.assert_allclose(three_classes.member_normalized_mi, 1.0, rtol=1e-7)
    assert three_classes.normalized_mi[0] > six_classes.normalized_mi[0]


def test_member_normalized_mi_is_safe_for_one_or_zero_members() -> None:
    mutual_information = np.asarray([0.0, 0.5], dtype=np.float64)

    np.testing.assert_array_equal(
        _normalize_mi_by_member_count(mutual_information, 1),
        np.zeros(2),
    )
    np.testing.assert_array_equal(
        _normalize_mi_by_member_count(mutual_information, 0),
        np.zeros(2),
    )


def test_single_class_normalization_is_safe() -> None:
    members = []
    for member_id in range(3):
        member = _member(member_id, [[0.9, 0.1], [0.8, 0.2]])
        members.append(
            replace(
                member,
                raw_class_ids=np.asarray([10], dtype=np.int64),
                labels=np.asarray([10, 10], dtype=np.int64),
                probabilities=np.ones((2, 1), dtype=np.float32),
                logits=np.zeros((2, 1), dtype=np.float32),
            )
        )

    artifact = aggregate_member_predictions(members)
    np.testing.assert_allclose(artifact.predictive_entropy, 0.0)
    np.testing.assert_allclose(artifact.expected_member_entropy, 0.0)
    np.testing.assert_allclose(artifact.mutual_information, 0.0)
    np.testing.assert_allclose(artifact.normalized_entropy, 0.0)
    np.testing.assert_allclose(artifact.normalized_mi, 0.0)
    np.testing.assert_allclose(artifact.member_normalized_mi, 0.0)
    np.testing.assert_allclose(artifact.entropy_confidence, 1.0)
    np.testing.assert_allclose(artifact.confidence, 1.0)
    np.testing.assert_allclose(artifact.max_probability, 1.0)


def test_offline_ensemble_export_load_round_trip(tmp_path: Path) -> None:
    original = aggregate_member_predictions(
        _numerical_members(),
        source_artifact_paths=["m0.npz", "m1.npz", "m2.npz"],
    )
    path = export_offline_ensemble_artifact(original, tmp_path / "ensemble.npz")
    loaded = load_offline_ensemble_artifact(path)

    assert loaded.context == original.context
    assert loaded.source_artifact_paths == ("m0.npz", "m1.npz", "m2.npz")
    for field in (
        "sample_ids",
        "labels",
        "raw_class_ids",
        "probabilities",
        "predictions",
        "predictive_entropy",
        "expected_member_entropy",
        "mutual_information",
        "normalized_entropy",
        "normalized_mi",
        "mean_probability_variance",
        "total_probability_variance",
        "member_normalized_mi",
        "entropy_confidence",
        "max_probability",
        "confidence",
        "pairwise_disagreement",
    ):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(original, field))


def test_schema_one_artifact_loads_with_derived_schema_two_fields(tmp_path: Path) -> None:
    original = aggregate_member_predictions(
        _numerical_members(),
        source_artifact_paths=["m0.npz", "m1.npz", "m2.npz"],
    )
    schema_two_path = export_offline_ensemble_artifact(
        original,
        tmp_path / "schema_two.npz",
    )
    schema_two_only_fields = {
        "member_count",
        "total_probability_variance",
        "member_normalized_mi",
        "entropy_confidence",
        "max_probability",
    }
    with np.load(schema_two_path, allow_pickle=False) as payload:
        legacy_payload = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key not in schema_two_only_fields
        }
    legacy_payload["schema_version"] = np.asarray(1, dtype=np.int32)
    schema_one_path = tmp_path / "schema_one.npz"
    np.savez_compressed(schema_one_path, **legacy_payload)

    loaded = load_offline_ensemble_artifact(schema_one_path)

    assert loaded.context.member_count == 3
    np.testing.assert_allclose(
        loaded.entropy_confidence,
        1.0 - loaded.normalized_entropy,
    )
    np.testing.assert_allclose(
        loaded.max_probability,
        loaded.probabilities.max(axis=1),
    )
    np.testing.assert_allclose(
        loaded.total_probability_variance,
        loaded.mean_probability_variance * loaded.class_count,
    )
    np.testing.assert_allclose(
        loaded.member_normalized_mi,
        loaded.mutual_information / np.log(3.0),
    )
    np.testing.assert_array_equal(loaded.confidence, loaded.entropy_confidence)
