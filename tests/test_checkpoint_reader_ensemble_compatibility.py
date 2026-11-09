"""Reviewed loader changes must not block old prediction artifacts."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from tests.test_stratified_ensemble_mode import _provenanced_fold_members
from src.eval.ensemble_ue import aggregate_member_predictions, export_offline_ensemble_artifact, load_offline_ensemble_artifact
from src.eval.study_contract import (
    CHECKPOINT_READER_HASHES, contract_digest, prediction_study_contract,
)
from src.training.fold_training import _prediction_provenance


@pytest.mark.parametrize("change", [None, "training", "unknown_reader", "model", "tampered_config"])
@pytest.mark.parametrize("split", ["validation", "test"])
def test_existing_predictions_reviewed_reader_compatibility(tmp_path, change, split):
    members = _provenanced_fold_members()
    hashes = sorted(CHECKPOINT_READER_HASHES)
    paths = []
    updated = []
    for member, artifact in enumerate(members):
        if split == "test":
            import numpy as np
            from src.eval.predictions import partition_identity_sha256
            sample_ids = members[0].sample_ids
            fold_ids = np.full(artifact.row_count, -1, dtype=np.int16)
            artifact = replace(
                artifact, context=replace(artifact.context, split="test"),
                sample_ids=sample_ids, fold_ids=fold_ids,
                provenance=replace(artifact.provenance, expected_partition_sha256=partition_identity_sha256(
                    sample_ids, artifact.labels, fold_ids,
                )),
            )
        p = artifact.provenance
        contract = {
            "training_policy": "policy", "method": artifact.context.method,
            "method_protocol": artifact.context.method_protocol, "fold_seed": 42,
            "task_protocol": "head_to_tail", "task_file_sha256": p.task_file_sha256,
            "task_layout": [38, 20, 20, 20, 20, 20, 20, 20],
            "assignment_sha256": p.assignment_sha256, "fold_manifest_sha256": p.fold_manifest_sha256,
            "model": {"variant": "ddinet_multiclass_cil"},
            "head_policy": "fixed_global_178_seen_class_mask_v1", "global_raw_class_order": list(range(178)),
            "microbatch": 64, "effective_batch_target": 1024, "loss_scope": "ewc",
            "hyperparameters": {"lr": .001}, "sampler": {"policy": "same"},
            "implementation_sha256": {"training/replay_checkpoint.py": hashes[member % 2], "training/train_cil.py": "same"},
            "seeds": {"experiment_seed": 0}, "preprocessing": {"policy": p.preprocessing_policy},
            "buffer": {"ranking": {"policy": p.ranking_policy}, "policy": p.buffer_policy},
            "global_slot_budget": p.global_memory_budget,
        }
        if member == 2:
            if change == "training":
                contract["implementation_sha256"]["training/train_cil.py"] = "changed"
            elif change == "unknown_reader":
                contract["implementation_sha256"]["training/replay_checkpoint.py"] = "unreviewed"
            elif change == "model":
                contract["model"]["variant"] = "other-model"
        digest = contract_digest(prediction_study_contract(contract))
        updated.append(replace(artifact, provenance=replace(p, study_contract_sha256=digest)))
        root = tmp_path / f"member_{member}"
        path = root / f"member_predictions/task_0/{split}.npz"
        path.parent.mkdir(parents=True)
        if member == 2 and change == "tampered_config":
            contract["hyperparameters"]["lr"] = .5
        (root / "run_config.json").write_text(json.dumps({
            "run_id": artifact.context.run_id, "checkpoint_contract": contract,
        }), encoding="utf-8")
        paths.append(path)
    if change is not None:
        with pytest.raises(ValueError, match="contract"):
            aggregate_member_predictions(updated, source_artifact_paths=paths)
        return
    result = aggregate_member_predictions(updated, source_artifact_paths=paths)
    assert result.row_count == (6 if split == "validation" else 2)
    assert result.member_provenance == tuple(a.provenance for a in updated)
    saved = export_offline_ensemble_artifact(result, tmp_path / "ensemble.npz")
    assert load_offline_ensemble_artifact(saved).member_provenance == result.member_provenance


def test_reconstructed_contract_matches_training_export(monkeypatch):
    """Compare against the trainer's actual payload construction."""
    from types import SimpleNamespace
    from src.training import fold_training
    payloads = []
    monkeypatch.setattr(fold_training, "_digest", lambda payload: payloads.append(deepcopy(payload)) or "a" * 64)
    c = {key: key for key in (
        "training_policy", "method", "method_protocol", "task_protocol", "loss_scope", "head_policy",
    )}
    c.update(fold_seed=42, task_file_sha256="b" * 64, task_layout=[38],
             assignment_sha256="c" * 64, fold_manifest_sha256="d" * 64,
             model={}, global_raw_class_order=[0], microbatch=64, effective_batch_target=1024,
             hyperparameters={}, sampler={}, implementation_sha256={}, seeds={"experiment_seed": 0},
             preprocessing={"policy": "raw_identity"}, buffer={"ranking": {"policy": "none"}, "policy": "none"},
             global_slot_budget=0)
    import numpy as np
    prepared = dict(resolved=c, buffer=SimpleNamespace(metadata=c["buffer"]),
                    seeds=SimpleNamespace(member_id=0, experiment_seed=0), task_hash=c["task_file_sha256"],
                    manifest={"assignment_sha256": c["assignment_sha256"]},
                    context=SimpleNamespace(manifest_sha256=c["fold_manifest_sha256"]),
                    budgets={0: 0, 1: 0, 2: 0}, contract=c, prep_hash="e" * 64)
    _prediction_provenance(prepared, metadata={"sample_id": np.array(["a"]), "fold_id": np.array([0])}, labels=np.array([0]))
    assert prediction_study_contract(c) == payloads[0]
