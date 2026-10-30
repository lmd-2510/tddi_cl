"""Method-neutral, task-scoped caches for continual-learning runs.

This module intentionally stores *data only*.  It is separate from a method
output directory, so CB-Replay, EWC and A-GEM can reuse the
same transformed task arrays when their contracts are identical.  A cache is
materialised on demand at a task boundary; it must never be pre-created for
future tasks by a trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ddi_dataset import DDIBatchArrays, DEFAULT_LABEL_COL


TASK_CACHE_SCHEMA_VERSION = 2
TASK_CACHE_KIND = "ddi_cil_common_task_cache"
TASK_CACHE_MANIFEST = "manifest.json"
TASK_CACHE_ROLES = ("train", "validation")
TASK_CACHE_LAYOUT = "incremental_task_class_shards_v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_columns_sha256(columns: Iterable[str]) -> str:
    return sha256_json(list(columns))


def fold_assignments_sha256(assignments: Any) -> str:
    """Hash the validated runtime fold lookup without depending on Parquet bytes."""

    sample_ids = np.asarray(assignments.sorted_sample_ids, dtype=np.str_)
    fold_ids = np.asarray(assignments.sorted_fold_ids, dtype=np.int16)
    return assignment_arrays_sha256(
        sample_ids,
        fold_ids,
        fold_count=int(assignments.fold_count),
        seed=int(assignments.seed),
    )


def assignment_arrays_sha256(
    sample_ids: Iterable[str],
    fold_ids: Iterable[int],
    *,
    fold_count: int,
    seed: int,
) -> str:
    """Hash a canonical sorted sample-id/fold mapping for cross-run reuse."""

    sample_ids = np.asarray(sample_ids, dtype=np.str_)
    fold_ids = np.asarray(fold_ids, dtype=np.int16)
    if sample_ids.shape[0] != fold_ids.shape[0]:
        raise ValueError("Fold assignment sample/fold arrays have different lengths")
    order = np.argsort(sample_ids, kind="stable")
    sample_ids = sample_ids[order]
    fold_ids = fold_ids[order]
    digest = hashlib.sha256()
    digest.update(np.asarray([int(fold_count), int(seed)], dtype=np.int64).tobytes())
    for sample_id, fold_id in zip(sample_ids, fold_ids, strict=True):
        encoded = str(sample_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(np.asarray([int(fold_id)], dtype=np.int16).tobytes())
    return digest.hexdigest()


def task_cache_path(root: str | Path, *, member_id: int, task_id: int) -> Path:
    """Return the method-neutral directory for one member/task boundary."""

    if int(member_id) < 0 or int(task_id) < 0:
        raise ValueError("member_id and task_id must be non-negative")
    return Path(root) / f"member_{int(member_id)}" / f"task_{int(task_id)}"


@contextmanager
def task_cache_lock(
    root: str | Path,
    *,
    member_id: int,
    task_id: int,
    wait_timeout_s: float = 24 * 60 * 60,
    stale_after_s: float = 24 * 60 * 60,
    poll_s: float = 2.0,
):
    """Serialize common-cache materialization for one member/task.

    The lock is intentionally scoped only to the method-neutral cache.  It is
    acquired atomically with ``O_EXCL``; a second method waits and then
    re-checks the manifest, so it can reuse the completed shard instead of
    rereading and transforming the same source data.  A lock older than
    ``stale_after_s`` is treated as abandoned after a crashed process.
    """

    task_dir = task_cache_path(root, member_id=member_id, task_id=task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    lock_path = task_dir / ".materialize.lock"
    token = f"pid={os.getpid()} host={socket.gethostname()} token={uuid.uuid4().hex}\n"
    deadline = time.monotonic() + float(wait_timeout_s)
    acquired = False
    while not acquired:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(token)
                handle.flush()
                os.fsync(handle.fileno())
            acquired = True
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > float(stale_after_s):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    continue
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for common task-cache lock: {lock_path}"
                )
            time.sleep(max(0.1, float(poll_s)))
    try:
        yield lock_path
    finally:
        try:
            if lock_path.read_text(encoding="utf-8") == token:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def _normalise_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(dict(contract), ensure_ascii=False, sort_keys=True))
    if value.get("kind") not in {None, TASK_CACHE_KIND}:
        raise ValueError("Task-cache contract kind is not method-neutral")
    value["kind"] = TASK_CACHE_KIND
    value["schema_version"] = int(value.get("schema_version", TASK_CACHE_SCHEMA_VERSION))
    if value["schema_version"] != TASK_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported task-cache schema version")
    # A common cache must not silently encode method-specific state.
    if value.get("method") not in (None, "common"):
        raise ValueError("Method-specific state cannot be stored in common task cache")
    value.pop("method", None)
    return value


def build_task_cache_contract(
    *,
    protocol_id: str,
    task_file_sha256: str,
    feature_columns: Iterable[str],
    scaler_sha256: str,
    member_id: int,
    fold_id: int | None = None,
    fold_count: int | None = None,
    fold_seed: int | None = None,
    fold_assignments_sha256: str | None = None,
    preprocessing_policy: str = "task0_standard_frozen",
    source_hashes: Mapping[str, str] | None = None,
    row_limits: Mapping[str, int | None] | None = None,
) -> dict[str, Any]:
    """Build the provenance contract shared by every method using the cache."""

    if not protocol_id:
        raise ValueError("protocol_id must be non-empty")
    columns = [str(column) for column in feature_columns]
    contract: dict[str, Any] = {
        "kind": TASK_CACHE_KIND,
        "schema_version": TASK_CACHE_SCHEMA_VERSION,
        "cache_layout": TASK_CACHE_LAYOUT,
        "protocol_id": str(protocol_id),
        "task_file_sha256": str(task_file_sha256),
        "feature_columns_sha256": feature_columns_sha256(columns),
        "feature_column_count": len(columns),
        "scaler_sha256": str(scaler_sha256),
        "preprocessing_policy": str(preprocessing_policy),
        "member_id": int(member_id),
    }
    if fold_id is not None:
        contract["fold_id"] = int(fold_id)
    if fold_count is not None:
        contract["fold_count"] = int(fold_count)
    if fold_seed is not None:
        contract["fold_seed"] = int(fold_seed)
    if fold_assignments_sha256 is not None:
        contract["fold_assignments_sha256"] = str(fold_assignments_sha256)
    if source_hashes:
        contract["source_hashes"] = {str(k): str(v) for k, v in sorted(source_hashes.items())}
    if row_limits:
        contract["row_limits"] = {
            str(k): (None if v is None else int(v))
            for k, v in sorted(row_limits.items())
        }
    return _normalise_contract(contract)


def _contract_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    differences: list[str] = []
    actual_norm = _normalise_contract(actual)
    expected_norm = _normalise_contract(expected)
    for key, expected_value in expected_norm.items():
        if actual_norm.get(key) != expected_value:
            differences.append(key)
    return differences


def _manifest_path(task_dir: Path) -> Path:
    return task_dir / TASK_CACHE_MANIFEST


def _role_path(task_dir: Path, role: str) -> Path:
    if role not in TASK_CACHE_ROLES:
        raise ValueError(f"Unsupported task-cache role: {role!r}")
    return task_dir / f"{role}.parquet"


def _array_to_arrow(values: np.ndarray) -> pa.Array:
    values = np.asarray(values)
    if values.dtype.kind in {"U", "S", "O"}:
        return pa.array([None if value is None else str(value) for value in values], type=pa.string())
    if values.dtype.kind in {"i", "u"}:
        return pa.array(values.astype(np.int64, copy=False), type=pa.int64())
    if values.dtype.kind == "b":
        return pa.array(values.astype(bool, copy=False), type=pa.bool_())
    return pa.array(values.astype(np.float32, copy=False), type=pa.float32())


def _arrays_to_table(arrays: DDIBatchArrays, feature_columns: list[str]) -> pa.Table:
    features = np.asarray(arrays.features, dtype=np.float32)
    labels = np.asarray(arrays.labels, dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != len(feature_columns):
        raise ValueError("Task-cache features do not match feature_columns")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Task-cache feature/label row count mismatch")
    columns: dict[str, pa.Array] = {
        name: pa.array(features[:, index], type=pa.float32())
        for index, name in enumerate(feature_columns)
    }
    columns[DEFAULT_LABEL_COL] = pa.array(labels, type=pa.int64())
    metadata = arrays.metadata or {}
    for name, values in metadata.items():
        if name in columns:
            raise ValueError(f"Metadata column collides with feature/label column: {name}")
        values = np.asarray(values)
        if len(values) != len(labels):
            raise ValueError(f"Metadata row count mismatch: {name}")
        columns[str(name)] = _array_to_arrow(values)
    return pa.table(columns)


def _table_to_arrays(table: pa.Table, feature_columns: list[str]) -> DDIBatchArrays:
    required = [*feature_columns, DEFAULT_LABEL_COL]
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"Task cache is missing columns: {missing}")
    features = np.column_stack([
        table[name].to_numpy(zero_copy_only=False) for name in feature_columns
    ]).astype(np.float32, copy=False)
    labels = table[DEFAULT_LABEL_COL].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    metadata = {
        name: table[name].to_numpy(zero_copy_only=False)
        for name in table.column_names
        if name not in required
    }
    return DDIBatchArrays(features=features, labels=labels, metadata=metadata or None)


def write_task_cache(
    root: str | Path,
    *,
    member_id: int,
    task_id: int,
    contract: Mapping[str, Any],
    current_class_ids: Iterable[int],
    validation_class_ids: Iterable[int],
    feature_columns: Iterable[str],
    train: DDIBatchArrays,
    validation: DDIBatchArrays,
    model_train: DDIBatchArrays | None = None,
    model_validation: DDIBatchArrays | None = None,
) -> dict[str, Any]:
    """Atomically write one task cache and return its committed manifest.

    The manifest is written last.  A killed process therefore leaves either a
    complete cache or a directory without a commit marker, which the loader
    rejects safely.
    """

    task_dir = task_cache_path(root, member_id=member_id, task_id=task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    contract = _normalise_contract(contract)
    if int(contract.get("member_id", member_id)) != int(member_id):
        raise ValueError("Task-cache contract member_id mismatch")
    columns = [str(name) for name in feature_columns]
    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns must be unique")
    roles = {"train": train, "validation": validation}
    model_roles = {"train": model_train, "validation": model_validation}
    temporary_paths: dict[str, Path] = {}
    final_paths: dict[str, Path] = {}
    role_manifest: dict[str, Any] = {}
    try:
        for role, arrays in roles.items():
            target = _role_path(task_dir, role)
            temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
            table = _arrays_to_table(arrays, columns)
            pq.write_table(table, temporary, compression="zstd")
            temporary_paths[role] = temporary
            final_paths[role] = target
            role_manifest[role] = {
                "path": target.name,
                "sha256": sha256_file(temporary),
                "rows": int(table.num_rows),
                "columns": table.column_names,
            }
            model_arrays = model_roles[role]
            if model_arrays is not None:
                model_target = task_dir / f"{role}_model.parquet"
                model_temporary = model_target.with_name(
                    f".{model_target.name}.tmp.{os.getpid()}"
                )
                model_table = _arrays_to_table(model_arrays, columns)
                if len(model_arrays.labels) != len(arrays.labels):
                    raise ValueError(f"Model-view row count mismatch: {role}")
                pq.write_table(model_table, model_temporary, compression="zstd")
                temporary_paths[f"{role}_model"] = model_temporary
                final_paths[f"{role}_model"] = model_target
                role_manifest[role]["model_path"] = model_target.name
                role_manifest[role]["model_sha256"] = sha256_file(model_temporary)
                role_manifest[role]["model_rows"] = int(model_table.num_rows)
        for role, temporary in temporary_paths.items():
            os.replace(temporary, final_paths[role])
        manifest = {
            **contract,
            "member_id": int(member_id),
            "task_id": int(task_id),
            "current_class_ids": sorted({int(value) for value in current_class_ids}),
            "validation_class_ids": sorted({int(value) for value in validation_class_ids}),
            "feature_columns": columns,
            "roles": role_manifest,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        temporary_manifest = _manifest_path(task_dir).with_name(
            f".{TASK_CACHE_MANIFEST}.tmp.{os.getpid()}"
        )
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, _manifest_path(task_dir))
        return manifest
    finally:
        for temporary in [*temporary_paths.values(), _manifest_path(task_dir).with_name(
            f".{TASK_CACHE_MANIFEST}.tmp.{os.getpid()}"
        )]:
            temporary.unlink(missing_ok=True)


def load_task_cache(
    root: str | Path,
    *,
    member_id: int,
    task_id: int,
    expected_contract: Mapping[str, Any],
    feature_columns: Iterable[str],
    current_class_ids: Iterable[int] | None = None,
    validation_class_ids: Iterable[int] | None = None,
    feature_view: str = "raw",
) -> tuple[DDIBatchArrays, DDIBatchArrays, dict[str, Any]]:
    """Load and verify a previously committed common task cache."""

    task_dir = task_cache_path(root, member_id=member_id, task_id=task_id)
    manifest_path = _manifest_path(task_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing task-cache commit marker: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    differences = _contract_matches(manifest, expected_contract)
    if int(manifest.get("member_id", -1)) != int(member_id):
        differences.append("member_id")
    if int(manifest.get("task_id", -1)) != int(task_id):
        differences.append("task_id")
    columns = [str(name) for name in feature_columns]
    if manifest.get("feature_columns") != columns:
        differences.append("feature_columns")
    if current_class_ids is not None and manifest.get("current_class_ids") != sorted({int(v) for v in current_class_ids}):
        differences.append("current_class_ids")
    if validation_class_ids is not None and manifest.get("validation_class_ids") != sorted({int(v) for v in validation_class_ids}):
        differences.append("validation_class_ids")
    if differences:
        raise ValueError(f"Common task-cache contract mismatch: {sorted(set(differences))}")
    if feature_view not in {"raw", "model"}:
        raise ValueError("feature_view must be 'raw' or 'model'")
    loaded: dict[str, DDIBatchArrays] = {}
    for role in TASK_CACHE_ROLES:
        role_key = "path" if feature_view == "raw" else "model_path"
        path = task_dir / manifest.get("roles", {}).get(role, {}).get(role_key, "")
        role_info = manifest.get("roles", {}).get(role, {})
        if not path.is_file():
            raise FileNotFoundError(f"Missing task-cache {feature_view} role: {path}")
        expected_hash = role_info.get("sha256" if feature_view == "raw" else "model_sha256")
        if expected_hash and expected_hash != sha256_file(path):
            raise ValueError(f"Common task-cache {feature_view} hash mismatch: {role}")
        loaded[role] = _table_to_arrays(pq.read_table(path), columns)
        expected_rows = role_info.get("rows" if feature_view == "raw" else "model_rows")
        if expected_rows is not None and int(expected_rows) != len(loaded[role].labels):
            raise ValueError(f"Common task-cache row-count mismatch: {role}")
    return loaded["train"], loaded["validation"], manifest


def load_task_cache_role(
    root: str | Path,
    *,
    member_id: int,
    task_id: int,
    role: str,
    expected_contract: Mapping[str, Any],
    feature_columns: Iterable[str],
    current_class_ids: Iterable[int] | None = None,
    validation_class_ids: Iterable[int] | None = None,
    feature_view: str = "raw",
) -> tuple[DDIBatchArrays, dict[str, Any]]:
    """Load one role without reading the other role's Parquet file."""

    if role not in TASK_CACHE_ROLES:
        raise ValueError(f"Unsupported task-cache role: {role!r}")
    task_dir = task_cache_path(root, member_id=member_id, task_id=task_id)
    manifest_path = _manifest_path(task_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing task-cache commit marker: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    differences = _contract_matches(manifest, expected_contract)
    if int(manifest.get("member_id", -1)) != int(member_id):
        differences.append("member_id")
    if int(manifest.get("task_id", -1)) != int(task_id):
        differences.append("task_id")
    columns = [str(name) for name in feature_columns]
    if manifest.get("feature_columns") != columns:
        differences.append("feature_columns")
    if current_class_ids is not None and manifest.get("current_class_ids") != sorted({int(v) for v in current_class_ids}):
        differences.append("current_class_ids")
    if validation_class_ids is not None and manifest.get("validation_class_ids") != sorted({int(v) for v in validation_class_ids}):
        differences.append("validation_class_ids")
    if differences:
        raise ValueError(f"Common task-cache contract mismatch: {sorted(set(differences))}")
    if feature_view not in {"raw", "model"}:
        raise ValueError("feature_view must be 'raw' or 'model'")
    role_info = manifest.get("roles", {}).get(role, {})
    role_key = "path" if feature_view == "raw" else "model_path"
    path = task_dir / role_info.get(role_key, "")
    if not path.is_file():
        raise FileNotFoundError(f"Missing task-cache {feature_view} role: {path}")
    expected_hash = role_info.get("sha256" if feature_view == "raw" else "model_sha256")
    if expected_hash and expected_hash != sha256_file(path):
        raise ValueError(f"Common task-cache {feature_view} hash mismatch: {role}")
    arrays = _table_to_arrays(pq.read_table(path), columns)
    expected_rows = role_info.get("rows" if feature_view == "raw" else "model_rows")
    if expected_rows is not None and int(expected_rows) != len(arrays.labels):
        raise ValueError(f"Common task-cache row-count mismatch: {role}")
    return arrays, manifest


def concatenate_task_arrays(parts: Iterable[DDIBatchArrays]) -> DDIBatchArrays:
    """Concatenate task shards while preserving row order and metadata."""

    materialized = list(parts)
    if not materialized:
        raise ValueError("At least one task-cache shard is required")
    metadata_keys = set((materialized[0].metadata or {}).keys())
    if any(set((part.metadata or {}).keys()) != metadata_keys for part in materialized[1:]):
        raise ValueError("Task-cache shard metadata columns do not match")
    return DDIBatchArrays(
        features=np.concatenate([part.features for part in materialized], axis=0),
        labels=np.concatenate([part.labels for part in materialized], axis=0),
        metadata=(
            {
                key: np.concatenate([(part.metadata or {})[key] for part in materialized])
                for key in sorted(metadata_keys)
            }
            if metadata_keys else None
        ),
    )


def load_validation_trajectory(
    root: str | Path,
    *,
    member_id: int,
    task_class_ids: Iterable[Iterable[int]],
    upto_task_id: int,
    expected_contract: Mapping[str, Any],
    feature_columns: Iterable[str],
    feature_view: str = "raw",
) -> DDIBatchArrays:
    """Load validation shards for tasks ``0..upto_task_id`` and concatenate them.

    Only already-reached task shards are requested.  This is deliberately
    task-boundary scoped: no future validation class is materialised or read.
    """

    class_lists = list(task_class_ids)
    if int(upto_task_id) < 0 or int(upto_task_id) >= len(class_lists):
        raise ValueError("upto_task_id is outside task_class_ids")
    parts = [
        load_task_cache_role(
            root,
            member_id=member_id,
            task_id=task_id,
            role="validation",
            expected_contract=expected_contract,
            feature_columns=feature_columns,
            validation_class_ids=class_lists[task_id],
            feature_view=feature_view,
        )[0]
        for task_id in range(int(upto_task_id) + 1)
    ]
    return concatenate_task_arrays(parts)
