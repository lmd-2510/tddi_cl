"""Focused contract selection tests for completed versus resumable runs."""
import json

import pytest

from src.training.fold_run_setup import _validation_contract_for_existing_run


def _write_run_config(root, contract):
    root.mkdir()
    (root / "run_config.json").write_text(
        json.dumps({"checkpoint_contract": contract}),
        encoding="utf-8",
    )


def test_completed_scope_accepts_only_implementation_hash_drift(tmp_path):
    root = tmp_path / "run"
    stored = {"method": "ewc", "sources": {"train": "same"},
              "implementation_sha256": {"train.py": "old"}}
    fresh = {**stored, "implementation_sha256": {"train.py": "new"}}
    _write_run_config(root, stored)
    selected, accepted = _validation_contract_for_existing_run(
        root, root / "checkpoints/task_7.pt", fresh,
        requested_stop_after_task=7,
    )
    assert selected == stored
    assert accepted is True


def test_incomplete_scope_does_not_accept_implementation_drift(tmp_path):
    root = tmp_path / "run"
    stored = {"method": "ewc", "implementation_sha256": {"train.py": "old"}}
    fresh = {"method": "ewc", "implementation_sha256": {"train.py": "new"}}
    _write_run_config(root, stored)
    selected, accepted = _validation_contract_for_existing_run(
        root, root / "checkpoints/task_6.pt", fresh,
        requested_stop_after_task=7,
    )
    assert selected == fresh
    assert accepted is False


def test_completed_scope_still_rejects_learning_or_data_drift(tmp_path):
    root = tmp_path / "run"
    stored = {"method": "ewc", "sources": {"train": "old"},
              "implementation_sha256": {"train.py": "old"}}
    fresh = {"method": "ewc", "sources": {"train": "changed"},
             "implementation_sha256": {"train.py": "new"}}
    _write_run_config(root, stored)
    with pytest.raises(ValueError, match="learning/data contract"):
        _validation_contract_for_existing_run(
            root, root / "checkpoints/task_7.pt", fresh,
            requested_stop_after_task=7,
        )


def test_singleton_training_policy_change_requires_explicit_migration(tmp_path):
    root = tmp_path / "run"
    stored = {"method": "upcl", "model": {"variant": "ddinet_multiclass_cil"},
              "implementation_sha256": {"train.py": "old"}}
    fresh = {**stored, "model": {**stored["model"],
             "singleton_batchnorm_policy": "running_stats_for_singleton_v1"}}
    _write_run_config(root, stored)
    with pytest.raises(ValueError, match="learning/data contract.*model"):
        _validation_contract_for_existing_run(
            root, root / "checkpoints/task_7.pt", fresh, requested_stop_after_task=7,
        )
