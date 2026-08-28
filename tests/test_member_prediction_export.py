from __future__ import annotations

import csv
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.eval.cil_evaluation import EvaluationResult, PredictionOutputs, evaluate_model
from src.eval.member_predictions import (
    MEMBER_PREDICTION_SCHEMA_VERSION,
    MemberPredictionContext,
    export_member_prediction_artifact,
    load_member_prediction_artifact,
)
from src.eval.s02_artifacts import DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN
from src.models.mlp import MLP, MLPConfig
from src.training.train_cil import export_member_evaluation
from src.utils.logging import RunLogger


def _outputs() -> PredictionOutputs:
    logits = np.asarray(
        [[2.0, 0.0], [0.0, 2.0], [1.5, -0.5]],
        dtype=np.float32,
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    raw_class_ids = np.asarray([10, 30], dtype=np.int64)
    return PredictionOutputs(
        logits=logits,
        probabilities=probabilities.astype(np.float32, copy=False),
        predictions=raw_class_ids[probabilities.argmax(axis=1)],
        labels=np.asarray([10, 30, 10], dtype=np.int64),
        latent_features=np.ones((3, 4), dtype=np.float32),
        class_ids=raw_class_ids,
    )


def _metadata() -> dict[str, np.ndarray]:
    return {
        DRUG_ID_A_COLUMN: np.asarray(["DB001", "DB002", "DB003"]),
        DRUG_ID_B_COLUMN: np.asarray(["DB101", "DB102", "DB103"]),
    }


def _context(*, member_id: int = 0, member_seed: int = 123) -> MemberPredictionContext:
    return MemberPredictionContext(
        run_id=f"member-{member_id}-run",
        method="ewc",
        method_protocol="ewc_natural_sampling",
        task_id=1,
        split="validation",
        member_id=member_id,
        experiment_seed=0,
        member_seed=member_seed,
    )


def test_member_prediction_export_load_round_trip(tmp_path: Path) -> None:
    context = _context()
    path = export_member_prediction_artifact(
        _outputs(),
        _metadata(),
        context,
        tmp_path / "task_1" / "validation.npz",
    )
    artifact = load_member_prediction_artifact(path, expected_context=context)

    assert artifact.context == context
    assert artifact.row_count == 3
    assert artifact.class_count == 2
    np.testing.assert_array_equal(
        artifact.sample_ids,
        ["DB001|DB101", "DB002|DB102", "DB003|DB103"],
    )
    np.testing.assert_array_equal(artifact.labels, _outputs().labels)
    np.testing.assert_array_equal(artifact.raw_class_ids, [10, 30])
    np.testing.assert_array_equal(artifact.logits, _outputs().logits)
    np.testing.assert_array_equal(artifact.probabilities, _outputs().probabilities)
    with np.load(path, allow_pickle=False) as payload:
        assert int(payload["schema_version"].item()) == MEMBER_PREDICTION_SCHEMA_VERSION
        assert int(payload["task_id"].item()) == 1
        assert int(payload["member_id"].item()) == 0
        assert int(payload["experiment_seed"].item()) == 0
        assert int(payload["member_seed"].item()) == 123
        assert "raw_class_ids" in payload.files


def test_training_export_helper_writes_namespaced_artifact_and_event(tmp_path: Path) -> None:
    logger = RunLogger(
        log_path=tmp_path / "train.log",
        events_path=tmp_path / "events.csv",
        run_id="member-2-run",
    )
    args = Namespace(
        member_id=2,
        experiment_seed=0,
        member_seed=9876,
        method="ewc",
        memory_per_class=50,
    )
    export_member_evaluation(
        EvaluationResult(metrics={}, outputs=_outputs()),
        _metadata(),
        run_paths={"outdir": tmp_path},
        run_id="member-2-run",
        args=args,
        task_id=1,
        split="validation",
        logger=logger,
    )

    artifact_path = tmp_path / "member_predictions" / "task_1" / "validation.npz"
    artifact = load_member_prediction_artifact(artifact_path)
    assert artifact.context.member_id == 2
    assert artifact.context.experiment_seed == 0
    assert artifact.context.member_seed == 9876
    with (tmp_path / "events.csv").open(newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    assert [event["event_type"] for event in events] == ["member_predictions_exported"]


def test_member_prediction_rejects_duplicates_and_shape_mismatches(tmp_path: Path) -> None:
    duplicate_metadata = _metadata()
    duplicate_metadata[DRUG_ID_A_COLUMN][1] = duplicate_metadata[DRUG_ID_A_COLUMN][0]
    duplicate_metadata[DRUG_ID_B_COLUMN][1] = duplicate_metadata[DRUG_ID_B_COLUMN][0]
    with pytest.raises(ValueError, match="unique"):
        export_member_prediction_artifact(
            _outputs(),
            duplicate_metadata,
            _context(),
            tmp_path / "duplicate.npz",
        )

    short_metadata = {
        key: values[:2]
        for key, values in _metadata().items()
    }
    with pytest.raises(ValueError, match="row count"):
        export_member_prediction_artifact(
            _outputs(),
            short_metadata,
            _context(),
            tmp_path / "row-mismatch.npz",
        )

    width_mismatch = replace(
        _outputs(),
        logits=_outputs().logits[:, :1].copy(),
        probabilities=_outputs().probabilities[:, :1].copy(),
    )
    with pytest.raises(ValueError, match="probability width"):
        export_member_prediction_artifact(
            width_mismatch,
            _metadata(),
            _context(),
            tmp_path / "width-mismatch.npz",
        )


def test_member_prediction_loader_rejects_missing_class_order_and_bad_rows(
    tmp_path: Path,
) -> None:
    path = export_member_prediction_artifact(
        _outputs(),
        _metadata(),
        _context(),
        tmp_path / "valid.npz",
    )
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}

    missing_class_order = tmp_path / "missing-class-order.npz"
    with missing_class_order.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{key: value for key, value in arrays.items() if key != "raw_class_ids"},
        )
    with pytest.raises(ValueError, match="missing keys.*raw_class_ids"):
        load_member_prediction_artifact(missing_class_order)

    bad_rows = tmp_path / "bad-rows.npz"
    arrays["labels"] = arrays["labels"][:2]
    with bad_rows.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    with pytest.raises(ValueError, match="labels.*row count"):
        load_member_prediction_artifact(bad_rows)


def test_evaluation_and_export_preserve_deterministic_sample_order(tmp_path: Path) -> None:
    torch.manual_seed(19)
    model = MLP(
        MLPConfig(
            input_dim=3,
            hidden_dims=(5,),
            num_classes=2,
            dropout=0.0,
            norm="none",
        )
    ).eval()
    features = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 0, 1, 0])
    dataset = TensorDataset(features, labels)
    metadata = {
        DRUG_ID_A_COLUMN: np.asarray([f"A{index}" for index in range(5)]),
        DRUG_ID_B_COLUMN: np.asarray([f"B{index}" for index in range(5)]),
    }

    artifacts = []
    for member_id, batch_size in enumerate((2, 3)):
        result = evaluate_model(
            model,
            DataLoader(dataset, batch_size=batch_size, shuffle=False),
            nn.CrossEntropyLoss(),
            "cpu",
            {0: 10, 1: 30},
            collect_outputs=True,
        )
        assert result.outputs is not None
        context = _context(member_id=member_id, member_seed=123 + member_id)
        path = export_member_prediction_artifact(
            result.outputs,
            metadata,
            context,
            tmp_path / f"member_{member_id}.npz",
        )
        artifacts.append(load_member_prediction_artifact(path))

    expected_ids = np.asarray([f"A{index}|B{index}" for index in range(5)])
    np.testing.assert_array_equal(artifacts[0].sample_ids, expected_ids)
    np.testing.assert_array_equal(artifacts[1].sample_ids, expected_ids)
    np.testing.assert_array_equal(artifacts[0].labels, artifacts[1].labels)
    # Different batch widths may select different GEMM kernels; row alignment
    # remains deterministic up to ordinary float32 round-off.
    np.testing.assert_allclose(artifacts[0].logits, artifacts[1].logits, rtol=1e-6, atol=1e-7)
