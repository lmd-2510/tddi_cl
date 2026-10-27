"""Task-0-only frozen preprocessing for the original-split workflow."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from src.data.original_split import OriginalSplitContext, load_original_split_arrays, sha256_file
from src.data.task_protocols import validate_full_task_spec

KIND = "ddi_cil_original_split_preprocessing"
VERSION = 1
POLICIES = ("raw_identity", "task0_standard_frozen")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _payload_contract(context, task_file, feature_columns, policy):
    spec = json.loads(Path(task_file).read_text(encoding="utf-8"))
    tasks = validate_full_task_spec(spec)
    return {
        "policy": policy,
        "task_file_sha256": sha256_file(task_file),
        "protocol": spec["protocol"],
        "feature_columns": list(feature_columns),
        "feature_order_sha256": _digest(feature_columns),
        "sources": {name: {key: info[key] for key in ("sha256", "row_count")}
                    for name, info in context.source_info.items()},
        "fit_scope": {"split": "train", "task_id": 0,
                      "raw_class_ids": list(tasks[0]["classes"])},
    }


@dataclass(frozen=True)
class SinglePreprocessing:
    payload: dict

    @property
    def metadata(self) -> dict:
        return json.loads(_canonical(self.payload))

    def transform(self, values, *, feature_columns: list[str]) -> np.ndarray:
        if list(feature_columns) != self.payload["contract"]["feature_columns"]:
            raise ValueError("Single-model preprocessing feature order mismatch.")
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_columns) or not np.isfinite(matrix).all():
            raise ValueError("Single-model preprocessing requires a finite 2D feature matrix.")
        if self.payload["contract"]["policy"] == "raw_identity":
            result = matrix.copy()
        else:
            result = (matrix - np.asarray(self.payload["statistics"]["mean"])) / np.asarray(self.payload["statistics"]["scale"])
        if not np.isfinite(result).all():
            raise ValueError("Single-model preprocessing produced nonfinite values.")
        return result


def prepare_single_preprocessing(context: OriginalSplitContext, *, task_file, feature_columns, policy):
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}.")
    contract = _payload_contract(context, task_file, feature_columns, policy)
    arrays = load_original_split_arrays(
        context, feature_columns, split="train",
        class_ids=contract["fit_scope"]["raw_class_ids"],
    )
    x = np.asarray(arrays.features, dtype=np.float64)
    statistics = None
    if policy == "task0_standard_frozen":
        mean = x.mean(axis=0)
        variance = np.square(x - mean).mean(axis=0)
        scale = np.where(variance == 0, 1.0, np.sqrt(variance))
        statistics = {"mean": mean.tolist(), "variance": variance.tolist(), "scale": scale.tolist()}
    payload = {
        "kind": KIND, "schema_version": VERSION, "contract": contract,
        "fit_row_count": int(len(arrays.labels)),
        "fit_sample_ids_sha256": _digest(arrays.metadata["sample_id"].tolist()),
        "statistics": statistics, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["payload_sha256"] = _digest(payload)
    return SinglePreprocessing(payload)


def save_single_preprocessing(artifact: SinglePreprocessing, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_canonical(artifact.payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_single_preprocessing(path, *, context, task_file, feature_columns, policy):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    checksum = payload.pop("payload_sha256", None)
    if payload.get("kind") != KIND or payload.get("schema_version") != VERSION or checksum != _digest(payload):
        raise ValueError("Invalid single-model preprocessing artifact.")
    payload["payload_sha256"] = checksum
    expected = _payload_contract(context, task_file, feature_columns, policy)
    if payload.get("contract") != expected:
        raise ValueError("Single-model preprocessing provenance mismatch.")
    stats = payload.get("statistics")
    if policy == "task0_standard_frozen":
        if not isinstance(stats, dict) or any(len(stats[key]) != len(feature_columns) for key in ("mean", "variance", "scale")):
            raise ValueError("Invalid frozen scaler statistics.")
    elif stats is not None:
        raise ValueError("raw_identity cannot contain fitted statistics.")
    return SinglePreprocessing(payload)


def load_or_prepare_single_preprocessing(path, *, context, task_file, feature_columns, policy):
    path = Path(path)
    if path.is_file():
        return load_single_preprocessing(path, context=context, task_file=task_file,
                                         feature_columns=feature_columns, policy=policy)
    artifact = prepare_single_preprocessing(context, task_file=task_file,
                                            feature_columns=feature_columns, policy=policy)
    try:
        save_single_preprocessing(artifact, path)
    except FileExistsError:
        pass
    return load_single_preprocessing(path, context=context, task_file=task_file,
                                     feature_columns=feature_columns, policy=policy)
