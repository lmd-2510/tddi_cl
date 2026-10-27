"""Method-neutral, on-demand task cache for the original source split."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from src.data.ddi_dataset import DDIBatchArrays
from src.data.original_split import load_original_split_arrays


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def _lock(path: Path, timeout: float = 600.0):
    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Timed out waiting for task-cache lock: {path}")
            time.sleep(0.2)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def cache_namespace(root, *, context, task_file_sha256, preprocessing_sha256, feature_columns):
    contract = {
        "kind": "ddi_cil_original_split_task_cache", "schema_version": 1,
        "sources": {name: {key: info[key] for key in ("sha256", "row_count")}
                    for name, info in context.source_info.items()},
        "task_file_sha256": task_file_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "feature_columns": list(feature_columns),
    }
    return Path(root) / _digest(contract), contract


def _write_npz(path: Path, arrays: DDIBatchArrays) -> None:
    metadata = arrays.metadata or {}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    serializable_metadata = {}
    for key, value in metadata.items():
        array = np.asarray(value)
        # ``allow_pickle=False`` is a deliberate cache trust boundary.
        if array.dtype.kind in "OUS":
            array = array.astype(np.str_)
        serializable_metadata[f"meta__{key}"] = array
    np.savez(temporary, features=np.asarray(arrays.features, dtype=np.float32),
             labels=np.asarray(arrays.labels, dtype=np.int64), **serializable_metadata)
    os.replace(temporary, path)


def _read_npz(path: Path) -> DDIBatchArrays:
    with np.load(path, allow_pickle=False) as payload:
        metadata = {name[6:]: payload[name].copy() for name in payload.files if name.startswith("meta__")}
        return DDIBatchArrays(payload["features"].copy(), payload["labels"].copy(), metadata)


def ensure_single_task_cache(
    root, *, context, task_file_sha256, preprocessing_sha256, feature_columns,
    preprocessing, task_id: int, class_ids: list[int],
):
    namespace, contract = cache_namespace(
        root, context=context, task_file_sha256=task_file_sha256,
        preprocessing_sha256=preprocessing_sha256, feature_columns=feature_columns,
    )
    task_dir = namespace / f"task_{task_id}"
    manifest_path = task_dir / "manifest.json"
    if not manifest_path.is_file():
        with _lock(namespace / "locks" / f"task_{task_id}.lock"):
            if not manifest_path.is_file():
                task_dir.mkdir(parents=True, exist_ok=True)
                rows = {}
                for split in ("train", "validation", "test"):
                    raw = load_original_split_arrays(context, feature_columns, split=split, class_ids=class_ids)
                    raw.features = preprocessing.transform(raw.features, feature_columns=feature_columns).astype(np.float32)
                    target = task_dir / f"{split}.npz"
                    _write_npz(target, raw)
                    rows[split] = int(len(raw.labels))
                manifest = {**contract, "task_id": int(task_id), "class_ids": list(class_ids), "rows": rows}
                temporary = task_dir / f".manifest.{uuid.uuid4().hex}.tmp"
                temporary.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temporary, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in contract.items()) or manifest.get("task_id") != task_id or manifest.get("class_ids") != list(class_ids):
        raise ValueError("Original-split task-cache contract mismatch.")
    loaded = {split: _read_npz(task_dir / f"{split}.npz") for split in ("train", "validation", "test")}
    if any(len(loaded[split].labels) != manifest["rows"][split] for split in loaded):
        raise ValueError("Original-split task-cache row-count mismatch.")
    return loaded, manifest


def concatenate_arrays(parts: list[DDIBatchArrays]) -> DDIBatchArrays:
    if not parts:
        raise ValueError("Cannot concatenate an empty task trajectory.")
    keys = set((parts[0].metadata or {}).keys())
    return DDIBatchArrays(
        np.concatenate([part.features for part in parts]),
        np.concatenate([part.labels for part in parts]),
        {key: np.concatenate([(part.metadata or {})[key] for part in parts]) for key in keys},
    )
