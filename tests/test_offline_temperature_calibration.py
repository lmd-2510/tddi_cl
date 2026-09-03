from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.eval.calibration_metrics import softmax_probabilities
from src.eval.member_predictions import MemberPredictionArtifact, MemberPredictionContext
from src.eval.offline_ensemble import (
    aggregate_member_predictions,
    export_offline_ensemble_artifact,
)
from src.eval.offline_temperature_calibration import (
    calibrate_mean_probabilities,
    evaluate_with_frozen_temperature,
    export_calibrated_probabilities,
    export_frozen_temperature,
    fit_offline_temperature,
    load_frozen_temperature,
    load_temperature_calibration_config,
)


CONFIG_PATH = Path("configs/tddi_ensemble_temperature_calibration_full_p3.json")
RAW_CLASS_IDS = np.asarray([10, 20, 30], dtype=np.int64)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensemble(split: str, *, seed: int) -> object:
    rng = np.random.default_rng(seed)
    base_logits = rng.normal(size=(2500, 3)).astype(np.float32) * 2.0
    raw_probabilities = softmax_probabilities(base_logits, temperature=1.0).astype(
        np.float32
    )
    generating_probabilities = softmax_probabilities(
        base_logits,
        temperature=2.5,
    )
    dense_labels = np.asarray(
        [rng.choice(3, p=row) for row in generating_probabilities],
        dtype=np.int64,
    )
    labels = RAW_CLASS_IDS[dense_labels]
    sample_ids = np.asarray([f"{split}-{index}" for index in range(labels.shape[0])])
    members = []
    for member_id in range(3):
        members.append(
            MemberPredictionArtifact(
                context=MemberPredictionContext(
                    run_id=f"run-{split}-{member_id}",
                    method="replay_distill_fixed_budget_uniform",
                    method_protocol="replay_distill_fixed_budget_uniform",
                    task_id=7,
                    split=split,
                    member_id=member_id,
                    experiment_seed=0,
                    member_seed=100 + member_id,
                ),
                sample_ids=sample_ids.copy(),
                labels=labels.copy(),
                raw_class_ids=RAW_CLASS_IDS.copy(),
                logits=base_logits.copy(),
                probabilities=raw_probabilities.copy(),
            )
        )
    return aggregate_member_predictions(members)


def test_calibration_formula_uses_log_mean_probability() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64)

    calibrated = calibrate_mean_probabilities(probabilities, 2.0)
    expected = np.sqrt(probabilities)
    expected /= expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(calibrated, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(
        calibrated.argmax(axis=1),
        probabilities.argmax(axis=1),
    )


def test_validation_fit_freeze_round_trip_and_test_application(tmp_path: Path) -> None:
    config = load_temperature_calibration_config(CONFIG_PATH)
    validation = _ensemble("validation", seed=120)
    validation_path = export_offline_ensemble_artifact(
        validation,
        tmp_path / "validation.npz",
    )
    raw_hash_before = _sha256(validation_path)

    fitted = fit_offline_temperature(
        validation,
        config,
        source_ensemble_path=validation_path,
    )
    assert fitted.temperature == pytest.approx(2.5, abs=0.35)
    assert fitted.optimization_diagnostics["member_logits_averaged"] is False
    assert fitted.optimization_diagnostics["input"] == "log_mean_ensemble_probability"
    assert fitted.optimization_diagnostics["optimized_nll"] <= fitted.optimization_diagnostics[
        "initial_nll"
    ]
    frozen_path = export_frozen_temperature(fitted, tmp_path / "temperature.json")
    frozen = load_frozen_temperature(frozen_path, config_path=CONFIG_PATH)
    assert frozen == fitted
    assert frozen.loaded_from_path == frozen_path.resolve()
    assert _sha256(validation_path) == raw_hash_before

    test = _ensemble("test", seed=121)
    test_path = export_offline_ensemble_artifact(test, tmp_path / "test.npz")
    report, calibrated = evaluate_with_frozen_temperature(
        test,
        frozen,
        config,
        source_ensemble_path=test_path,
    )

    assert report["temperature_fit_split"] == "validation"
    assert report["evaluation_split"] == "test"
    assert report["member_logits_averaged"] is False
    assert report["argmax_invariant"] is True
    assert report["accuracy_invariant"] is True
    assert report["raw"]["accuracy"] == report["calibrated"]["accuracy"]
    np.testing.assert_array_equal(
        test.probabilities.argmax(axis=1),
        calibrated.argmax(axis=1),
    )
    for stage in ("raw", "calibrated"):
        assert set(report[stage]) >= {
            "max_probability_ece",
            "negative_log_likelihood",
            "brier_score",
        }

    calibrated_path = export_calibrated_probabilities(
        test,
        calibrated,
        frozen,
        source_ensemble_path=test_path,
        path=tmp_path / "test_calibrated.npz",
    )
    with np.load(calibrated_path, allow_pickle=False) as payload:
        assert "calibrated_probabilities" in payload.files
        assert "calibrated_predictive_entropy" in payload.files
        assert "calibrated_normalized_entropy" in payload.files
        assert "calibrated_entropy_confidence" in payload.files
        assert "calibrated_max_probability" in payload.files
        assert "predictive_entropy" not in payload.files
        assert "probabilities" not in payload.files
        np.testing.assert_array_equal(
            payload["calibrated_predictions"],
            test.predictions,
        )


def test_fit_is_validation_only_and_test_requires_disk_artifact(tmp_path: Path) -> None:
    config = load_temperature_calibration_config(CONFIG_PATH)
    test = _ensemble("test", seed=222)
    test_path = export_offline_ensemble_artifact(test, tmp_path / "test.npz")

    with pytest.raises(ValueError, match="validation-only"):
        fit_offline_temperature(test, config, source_ensemble_path=test_path)

    validation = _ensemble("validation", seed=221)
    validation_path = export_offline_ensemble_artifact(
        validation,
        tmp_path / "validation.npz",
    )
    fitted = fit_offline_temperature(
        validation,
        config,
        source_ensemble_path=validation_path,
    )
    with pytest.raises(ValueError, match="loaded from disk"):
        evaluate_with_frozen_temperature(
            test,
            fitted,
            config,
            source_ensemble_path=test_path,
        )


def test_application_rejects_metadata_and_class_order_mismatch(tmp_path: Path) -> None:
    config = load_temperature_calibration_config(CONFIG_PATH)
    validation = _ensemble("validation", seed=330)
    validation_path = export_offline_ensemble_artifact(
        validation,
        tmp_path / "validation.npz",
    )
    fitted = fit_offline_temperature(
        validation,
        config,
        source_ensemble_path=validation_path,
    )
    frozen_path = export_frozen_temperature(fitted, tmp_path / "temperature.json")
    frozen = load_frozen_temperature(frozen_path, config_path=CONFIG_PATH)
    test = _ensemble("test", seed=331)
    test_path = export_offline_ensemble_artifact(test, tmp_path / "test.npz")

    wrong_seed = replace(
        test,
        context=replace(test.context, experiment_seed=9),
    )
    with pytest.raises(ValueError, match="experiment_seed"):
        evaluate_with_frozen_temperature(
            wrong_seed,
            frozen,
            config,
            source_ensemble_path=test_path,
        )

    wrong_order = replace(test, raw_class_ids=test.raw_class_ids[::-1].copy())
    with pytest.raises(ValueError, match="raw_class_ids"):
        evaluate_with_frozen_temperature(
            wrong_order,
            frozen,
            config,
            source_ensemble_path=test_path,
        )
