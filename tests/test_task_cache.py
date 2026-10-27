from pathlib import Path

import numpy as np
import pytest

from src.data.ddi_dataset import DDIBatchArrays
from src.data.task_cache import (
    TASK_CACHE_KIND,
    assignment_arrays_sha256,
    build_task_cache_contract,
    load_task_cache,
    load_validation_trajectory,
    task_cache_lock,
    write_task_cache,
)


def _arrays(offset: float) -> DDIBatchArrays:
    return DDIBatchArrays(
        features=np.asarray([[offset, 1.0], [offset + 1.0, 2.0]], dtype=np.float32),
        labels=np.asarray([3, 4], dtype=np.int64),
        metadata={"drugid-drug_a": np.asarray(["a", "b"]), "source_row_index": np.asarray([0, 1])},
    )


def _contract() -> dict:
    return build_task_cache_contract(
        protocol_id="p4_seed0_fold42",
        task_file_sha256="task-hash",
        feature_columns=["f0", "f1"],
        scaler_sha256="scaler-hash",
        member_id=0,
        fold_id=1,
        fold_count=3,
        fold_seed=42,
        fold_assignments_sha256="assignment-hash",
    )


def test_task_cache_round_trip_is_method_neutral(tmp_path: Path) -> None:
    contract = _contract()
    manifest = write_task_cache(
        tmp_path,
        member_id=0,
        task_id=2,
        contract=contract,
        current_class_ids=[3],
        validation_class_ids=[1, 2, 3, 4],
        feature_columns=["f0", "f1"],
        train=_arrays(0.0),
        validation=_arrays(10.0),
        model_train=_arrays(100.0),
        model_validation=_arrays(110.0),
    )
    assert manifest["kind"] == TASK_CACHE_KIND
    assert "method" not in manifest
    train, validation, loaded = load_task_cache(
        tmp_path,
        member_id=0,
        task_id=2,
        expected_contract=contract,
        feature_columns=["f0", "f1"],
    )
    np.testing.assert_allclose(train.features, _arrays(0.0).features)
    np.testing.assert_array_equal(validation.labels, _arrays(10.0).labels)
    assert loaded["roles"]["train"]["rows"] == 2
    model_train, _, _ = load_task_cache(
        tmp_path,
        member_id=0,
        task_id=2,
        expected_contract=contract,
        feature_columns=["f0", "f1"],
        feature_view="model",
    )
    np.testing.assert_allclose(model_train.features, _arrays(100.0).features)


def test_task_cache_rejects_contract_mismatch(tmp_path: Path) -> None:
    contract = _contract()
    write_task_cache(
        tmp_path,
        member_id=0,
        task_id=0,
        contract=contract,
        current_class_ids=[3],
        validation_class_ids=[3],
        feature_columns=["f0", "f1"],
        train=_arrays(0.0),
        validation=_arrays(10.0),
    )
    wrong = {**contract, "scaler_sha256": "different"}
    with pytest.raises(ValueError, match="contract mismatch"):
        load_task_cache(
            tmp_path,
            member_id=0,
            task_id=0,
            expected_contract=wrong,
            feature_columns=["f0", "f1"],
        )


def test_task_cache_rejects_tampered_role(tmp_path: Path) -> None:
    contract = _contract()
    write_task_cache(
        tmp_path,
        member_id=0,
        task_id=0,
        contract=contract,
        current_class_ids=[3],
        validation_class_ids=[3],
        feature_columns=["f0", "f1"],
        train=_arrays(0.0),
        validation=_arrays(10.0),
    )
    path = tmp_path / "member_0" / "task_0" / "train.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_task_cache(
            tmp_path,
            member_id=0,
            task_id=0,
            expected_contract=contract,
            feature_columns=["f0", "f1"],
        )


def test_assignment_hash_is_order_independent() -> None:
    left = assignment_arrays_sha256(["b", "a"], [1, 0], fold_count=3, seed=42)
    right = assignment_arrays_sha256(["a", "b"], [0, 1], fold_count=3, seed=42)
    assert left == right


def test_validation_trajectory_reads_only_reached_shards(tmp_path: Path) -> None:
    contract = _contract()
    for task_id, class_ids, offset in ((0, [1], 10.0), (1, [2], 20.0)):
        write_task_cache(
            tmp_path,
            member_id=0,
            task_id=task_id,
            contract=contract,
            current_class_ids=class_ids,
            validation_class_ids=class_ids,
            feature_columns=["f0", "f1"],
            train=_arrays(float(task_id)),
            validation=_arrays(offset),
        )
    trajectory = load_validation_trajectory(
        tmp_path,
        member_id=0,
        task_class_ids=[[1], [2]],
        upto_task_id=1,
        expected_contract=contract,
        feature_columns=["f0", "f1"],
    )
    np.testing.assert_allclose(trajectory.features[:, 0], [10.0, 11.0, 20.0, 21.0])


def test_task_cache_lock_is_released_after_materialization(tmp_path: Path) -> None:
    with task_cache_lock(tmp_path, member_id=0, task_id=0, poll_s=0.01):
        lock_path = tmp_path / "member_0" / "task_0" / ".materialize.lock"
        assert lock_path.is_file()
    assert not lock_path.exists()
