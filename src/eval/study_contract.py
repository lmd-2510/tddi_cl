"""Compatibility for reviewed checkpoint-reader fixes in completed studies."""
from __future__ import annotations

import hashlib
import json


CHECKPOINT_READER_HASHES = frozenset({
    # Before and after commit 7c304b9. Only backbone acceptance and EWC
    # BatchNorm-buffer validation changed in replay_checkpoint.py.
    "b48fef6276952abc220bfc0961519d13583505276f2ab33f5204ee06d2347085",
    "0d6c5208fc899c4b3a2795f50b2be3a3ab3ac154c539ab47f990a6c63b13cfc0",
})


def compatible_implementation(implementation):
    result = dict(implementation)
    key = "training/replay_checkpoint.py"
    if result.get(key) in CHECKPOINT_READER_HASHES:
        result[key] = "reviewed_ddinet_checkpoint_reader_fix_v1"
    return result


def contract_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def prediction_study_contract(contract):
    """Reconstruct the shared provenance payload written by fold_training."""
    keys = (
        "training_policy", "method", "method_protocol", "fold_seed",
        "task_protocol", "task_file_sha256", "task_layout", "assignment_sha256",
        "fold_manifest_sha256", "model", "head_policy", "global_raw_class_order",
        "microbatch", "effective_batch_target", "loss_scope", "hyperparameters",
        "sampler", "implementation_sha256",
    )
    result = {key: contract[key] for key in keys}
    result.update(
        experiment_seed=contract["seeds"]["experiment_seed"],
        preprocessing_policy=contract["preprocessing"]["policy"],
        ranking_policy=contract["buffer"]["ranking"]["policy"],
        buffer_policy=contract["buffer"]["policy"],
        global_memory_budget=contract["global_slot_budget"],
    )
    return result
