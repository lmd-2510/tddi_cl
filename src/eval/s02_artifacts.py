"""Validated, provenance-aware artifact export for diagnostic milestone S02."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.eval.cil_evaluation import PredictionOutputs


S02_SCHEMA_VERSION = 1
DRUG_ID_A_COLUMN = "drugid-drug_a"
DRUG_ID_B_COLUMN = "drugid-drug_b"
SUPPORTED_SPLITS = {"validation", "test"}


@dataclass(frozen=True)
class S02ExportContext:
    run_id: str
    seed: int
    method: str
    method_protocol: str
    train_task: int
    split: str
    checkpoint_path: Path
    run_config_path: Path


@dataclass(frozen=True)
class S02ArtifactPaths:
    predictions_path: Path
    latent_features_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class S02ValidationSummary:
    row_count: int
    class_count: int
    latent_dim: int
    sample_ids: np.ndarray
    labels: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run_config(context: S02ExportContext) -> dict[str, object]:
    payload = json.loads(context.run_config_path.read_text(encoding="utf-8"))
    if payload.get("run_id") != context.run_id:
        raise ValueError("S02 context run_id does not match run_config.json.")
    arguments = payload.get("arguments")
    resolved = payload.get("resolved")
    if not isinstance(arguments, dict) or not isinstance(resolved, dict):
        raise ValueError("run_config.json is missing arguments or resolved provenance.")
    expected_values = {
        "seed": context.seed,
        "method": context.method,
    }
    for key, expected in expected_values.items():
        if arguments.get(key) != expected:
            raise ValueError(f"S02 {key} does not match run_config.json.")
    if resolved.get("method_protocol") != context.method_protocol:
        raise ValueError("S02 method_protocol does not match run_config.json.")
    return payload


def _build_sample_ids(
    metadata: Mapping[str, np.ndarray],
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = sorted({DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN} - set(metadata))
    if missing:
        raise ValueError(f"Missing S02 drug-pair metadata columns: {missing}")
    raw_drug_ids_a = np.asarray(metadata[DRUG_ID_A_COLUMN])
    raw_drug_ids_b = np.asarray(metadata[DRUG_ID_B_COLUMN])
    for column, values in (
        (DRUG_ID_A_COLUMN, raw_drug_ids_a),
        (DRUG_ID_B_COLUMN, raw_drug_ids_b),
    ):
        if values.dtype.kind not in {"O", "S", "U"}:
            raise TypeError(f"{column} must contain string identifiers.")
        if values.dtype.kind == "O" and not all(
            isinstance(value, (str, np.str_)) for value in values.tolist()
        ):
            raise TypeError(f"{column} contains a non-string or missing identifier.")
    drug_ids_a = raw_drug_ids_a.astype(np.str_, copy=False)
    drug_ids_b = raw_drug_ids_b.astype(np.str_, copy=False)
    if drug_ids_a.ndim != 1 or drug_ids_b.ndim != 1:
        raise ValueError("Drug ID metadata must be one-dimensional.")
    if drug_ids_a.shape[0] != expected_rows or drug_ids_b.shape[0] != expected_rows:
        raise ValueError("Drug ID metadata row count does not match prediction outputs.")
    if np.any(drug_ids_a == "") or np.any(drug_ids_b == ""):
        raise ValueError("Drug IDs must be non-empty.")
    sample_ids = np.char.add(np.char.add(drug_ids_a, "|"), drug_ids_b)
    unique_count = np.unique(sample_ids).shape[0]
    if unique_count != expected_rows:
        raise ValueError(
            f"S02 sample IDs must be unique; found {expected_rows - unique_count} duplicates."
        )
    return sample_ids, drug_ids_a, drug_ids_b


def build_stable_sample_ids(
    metadata: Mapping[str, np.ndarray],
    expected_rows: int,
) -> np.ndarray:
    """Build validated, order-preserving drug-pair IDs for prediction artifacts."""

    sample_ids, _, _ = _build_sample_ids(metadata, expected_rows)
    return sample_ids


def _validate_outputs(outputs: PredictionOutputs) -> tuple[int, int, int]:
    logits = np.asarray(outputs.logits)
    probabilities = np.asarray(outputs.probabilities)
    predictions = np.asarray(outputs.predictions)
    labels = np.asarray(outputs.labels)
    latent = np.asarray(outputs.latent_features)
    class_ids = np.asarray(outputs.class_ids)

    expected_dtypes = {
        "logits": (logits.dtype, np.dtype(np.float32)),
        "probabilities": (probabilities.dtype, np.dtype(np.float32)),
        "latent_features": (latent.dtype, np.dtype(np.float32)),
        "predictions": (predictions.dtype, np.dtype(np.int64)),
        "labels": (labels.dtype, np.dtype(np.int64)),
        "class_ids": (class_ids.dtype, np.dtype(np.int64)),
    }
    for name, (actual, expected) in expected_dtypes.items():
        if actual != expected:
            raise TypeError(f"{name} must use dtype {expected}, got {actual}.")

    if logits.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("Logits and probabilities must be two-dimensional.")
    if logits.shape != probabilities.shape:
        raise ValueError("Logits and probabilities must have identical shapes.")
    row_count, class_count = logits.shape
    if row_count == 0 or class_count == 0:
        raise ValueError("S02 outputs must contain at least one row and one class.")
    if predictions.shape != (row_count,) or labels.shape != (row_count,):
        raise ValueError("Labels and predictions must match the output row count.")
    if class_ids.shape != (class_count,):
        raise ValueError("class_ids width must match logit/probability columns.")
    if np.unique(class_ids).shape[0] != class_count:
        raise ValueError("class_ids must be unique.")
    if latent.ndim != 2 or latent.shape[0] != row_count or latent.shape[1] == 0:
        raise ValueError("Latent features must have shape [rows, positive_latent_dim].")
    if not np.isfinite(logits).all():
        raise ValueError("Logits contain NaN or infinite values.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinite values.")
    if not np.isfinite(latent).all():
        raise ValueError("Latent features contain NaN or infinite values.")
    if np.any(probabilities < -1e-7) or np.any(probabilities > 1.0 + 1e-7):
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Probability rows must sum to one.")
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    expected_probabilities = np.exp(shifted_logits)
    expected_probabilities /= expected_probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities, expected_probabilities, rtol=1e-5, atol=1e-6):
        raise ValueError("Probabilities do not match softmax(logits).")
    expected_predictions = class_ids[probabilities.argmax(axis=1)]
    if not np.array_equal(predictions, expected_predictions):
        raise ValueError("Raw predictions do not match probability argmax and class_ids.")
    if not np.isin(labels, class_ids).all() or not np.isin(predictions, class_ids).all():
        raise ValueError("Labels and predictions must belong to seen raw class IDs.")
    return row_count, class_count, latent.shape[1]


def _prediction_schema(class_count: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("seed", pa.int64(), nullable=False),
            pa.field("method", pa.string(), nullable=False),
            pa.field("method_protocol", pa.string(), nullable=False),
            pa.field("train_task", pa.int32(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("drug_id_a", pa.string(), nullable=False),
            pa.field("drug_id_b", pa.string(), nullable=False),
            pa.field("label", pa.int64(), nullable=False),
            pa.field("prediction", pa.int64(), nullable=False),
            pa.field("confidence", pa.float32(), nullable=False),
            pa.field("entropy", pa.float32(), nullable=False),
            pa.field("class_ids", pa.list_(pa.int64(), class_count), nullable=False),
            pa.field("logits", pa.list_(pa.float32(), class_count), nullable=False),
            pa.field("probabilities", pa.list_(pa.float32(), class_count), nullable=False),
        ]
    )


def _fixed_size_list_array(values: np.ndarray, value_type: pa.DataType) -> pa.Array:
    values = np.ascontiguousarray(values)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=value_type),
        values.shape[1],
    )


def _prediction_table(
    outputs: PredictionOutputs,
    context: S02ExportContext,
    sample_ids: np.ndarray,
    drug_ids_a: np.ndarray,
    drug_ids_b: np.ndarray,
) -> pa.Table:
    row_count, class_count = outputs.logits.shape
    probabilities = np.asarray(outputs.probabilities, dtype=np.float32)
    safe_probabilities = np.clip(probabilities, np.finfo(np.float32).tiny, 1.0)
    confidence = probabilities.max(axis=1).astype(np.float32, copy=False)
    entropy = -np.sum(probabilities * np.log(safe_probabilities), axis=1).astype(
        np.float32,
        copy=False,
    )
    repeated_class_ids = np.broadcast_to(
        np.asarray(outputs.class_ids, dtype=np.int64),
        (row_count, class_count),
    ).copy()
    arrays = [
        pa.array([context.run_id] * row_count, type=pa.string()),
        pa.array(np.full(row_count, context.seed, dtype=np.int64), type=pa.int64()),
        pa.array([context.method] * row_count, type=pa.string()),
        pa.array([context.method_protocol] * row_count, type=pa.string()),
        pa.array(np.full(row_count, context.train_task, dtype=np.int32), type=pa.int32()),
        pa.array([context.split] * row_count, type=pa.string()),
        pa.array(sample_ids, type=pa.string()),
        pa.array(drug_ids_a, type=pa.string()),
        pa.array(drug_ids_b, type=pa.string()),
        pa.array(np.asarray(outputs.labels, dtype=np.int64), type=pa.int64()),
        pa.array(np.asarray(outputs.predictions, dtype=np.int64), type=pa.int64()),
        pa.array(confidence, type=pa.float32()),
        pa.array(entropy, type=pa.float32()),
        _fixed_size_list_array(repeated_class_ids, pa.int64()),
        _fixed_size_list_array(np.asarray(outputs.logits, dtype=np.float32), pa.float32()),
        _fixed_size_list_array(probabilities, pa.float32()),
    ]
    return pa.Table.from_arrays(arrays, schema=_prediction_schema(class_count))


def _write_latent_npz(
    path: Path,
    outputs: PredictionOutputs,
    context: S02ExportContext,
    sample_ids: np.ndarray,
    drug_ids_a: np.ndarray,
    drug_ids_b: np.ndarray,
    checkpoint_sha256: str,
) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            latent_features=np.asarray(outputs.latent_features, dtype=np.float32),
            sample_ids=sample_ids,
            labels=np.asarray(outputs.labels, dtype=np.int64),
            drug_ids_a=drug_ids_a,
            drug_ids_b=drug_ids_b,
            class_ids=np.asarray(outputs.class_ids, dtype=np.int64),
            run_id=np.asarray(context.run_id),
            seed=np.asarray(context.seed, dtype=np.int64),
            method=np.asarray(context.method),
            method_protocol=np.asarray(context.method_protocol),
            train_task=np.asarray(context.train_task, dtype=np.int32),
            split=np.asarray(context.split),
            checkpoint_sha256=np.asarray(checkpoint_sha256),
            schema_version=np.asarray(S02_SCHEMA_VERSION, dtype=np.int32),
        )


def validate_s02_artifacts(
    predictions_path: str | Path,
    latent_features_path: str | Path,
    *,
    expected_context: S02ExportContext | None = None,
) -> S02ValidationSummary:
    """Read an O03/O04 pair and verify schema, provenance, and row alignment."""

    predictions_path = Path(predictions_path)
    latent_features_path = Path(latent_features_path)
    table = pq.read_table(predictions_path)
    if table.schema != _prediction_schema(table.schema.field("logits").type.list_size):
        raise ValueError("predictions.parquet does not match the S02 schema.")
    row_count = table.num_rows
    class_count = table.schema.field("logits").type.list_size
    parquet_sample_ids = np.asarray(table["sample_id"].to_pylist(), dtype=np.str_)
    parquet_drug_ids_a = np.asarray(table["drug_id_a"].to_pylist(), dtype=np.str_)
    parquet_drug_ids_b = np.asarray(table["drug_id_b"].to_pylist(), dtype=np.str_)
    parquet_labels = table["label"].to_numpy().astype(np.int64, copy=False)
    logits = np.asarray(table["logits"].to_pylist(), dtype=np.float32)
    probabilities = np.asarray(table["probabilities"].to_pylist(), dtype=np.float32)
    predictions = table["prediction"].to_numpy().astype(np.int64, copy=False)
    class_ids_by_row = np.asarray(table["class_ids"].to_pylist(), dtype=np.int64)
    confidence = table["confidence"].to_numpy().astype(np.float32, copy=False)
    entropy = table["entropy"].to_numpy().astype(np.float32, copy=False)
    if row_count == 0 or class_ids_by_row.shape != (row_count, class_count):
        raise ValueError("Invalid class_ids shape in predictions.parquet.")
    if not np.all(class_ids_by_row == class_ids_by_row[0]):
        raise ValueError("class_ids must be identical for every prediction row.")

    with np.load(latent_features_path, allow_pickle=False) as latent_payload:
        required_keys = {
            "latent_features",
            "sample_ids",
            "labels",
            "drug_ids_a",
            "drug_ids_b",
            "class_ids",
            "run_id",
            "seed",
            "method",
            "method_protocol",
            "train_task",
            "split",
            "checkpoint_sha256",
            "schema_version",
        }
        missing = sorted(required_keys - set(latent_payload.files))
        if missing:
            raise ValueError(f"latent_features.npz is missing keys: {missing}")
        npz_numeric_dtypes = {
            "latent_features": np.dtype(np.float32),
            "labels": np.dtype(np.int64),
            "class_ids": np.dtype(np.int64),
            "seed": np.dtype(np.int64),
            "train_task": np.dtype(np.int32),
            "schema_version": np.dtype(np.int32),
        }
        for key, expected_dtype in npz_numeric_dtypes.items():
            if latent_payload[key].dtype != expected_dtype:
                raise TypeError(
                    f"O04 {key} must use dtype {expected_dtype}, "
                    f"got {latent_payload[key].dtype}."
                )
        latent = np.asarray(latent_payload["latent_features"])
        npz_sample_ids = np.asarray(latent_payload["sample_ids"], dtype=np.str_)
        npz_labels = np.asarray(latent_payload["labels"])
        npz_drug_ids_a = np.asarray(latent_payload["drug_ids_a"], dtype=np.str_)
        npz_drug_ids_b = np.asarray(latent_payload["drug_ids_b"], dtype=np.str_)
        npz_class_ids = np.asarray(latent_payload["class_ids"])
        npz_run_id = str(latent_payload["run_id"].item())
        npz_seed = int(latent_payload["seed"].item())
        npz_method = str(latent_payload["method"].item())
        npz_protocol = str(latent_payload["method_protocol"].item())
        npz_task = int(latent_payload["train_task"].item())
        npz_split = str(latent_payload["split"].item())
        npz_checkpoint_sha256 = str(latent_payload["checkpoint_sha256"].item())
        npz_schema_version = int(latent_payload["schema_version"].item())

    outputs = PredictionOutputs(
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        labels=parquet_labels,
        latent_features=latent,
        class_ids=class_ids_by_row[0],
    )
    checked_rows, checked_classes, latent_dim = _validate_outputs(outputs)
    if checked_rows != row_count or checked_classes != class_count:
        raise ValueError("S02 artifact shape validation failed.")
    if not np.array_equal(parquet_sample_ids, npz_sample_ids):
        raise ValueError("O03/O04 sample IDs are not aligned.")
    if not np.array_equal(parquet_drug_ids_a, npz_drug_ids_a):
        raise ValueError("O03/O04 drug_id_a values are not aligned.")
    if not np.array_equal(parquet_drug_ids_b, npz_drug_ids_b):
        raise ValueError("O03/O04 drug_id_b values are not aligned.")
    if not np.array_equal(parquet_labels, npz_labels):
        raise ValueError("O03/O04 labels are not aligned.")
    if not np.array_equal(class_ids_by_row[0], npz_class_ids):
        raise ValueError("O03/O04 class IDs are not aligned.")
    if np.unique(parquet_sample_ids).shape[0] != row_count:
        raise ValueError("O03 contains duplicate sample IDs.")
    if npz_schema_version != S02_SCHEMA_VERSION:
        raise ValueError("Unsupported S02 schema version in latent_features.npz.")
    expected_confidence = probabilities.max(axis=1)
    safe_probabilities = np.clip(probabilities, np.finfo(np.float32).tiny, 1.0)
    expected_entropy = -np.sum(probabilities * np.log(safe_probabilities), axis=1)
    if not np.isfinite(confidence).all() or not np.isfinite(entropy).all():
        raise ValueError("O03 confidence or entropy contains non-finite values.")
    if not np.allclose(confidence, expected_confidence, rtol=1e-5, atol=1e-6):
        raise ValueError("O03 confidence does not match probabilities.")
    if not np.allclose(entropy, expected_entropy, rtol=1e-5, atol=1e-6):
        raise ValueError("O03 entropy does not match probabilities.")

    parquet_run_ids = set(table["run_id"].to_pylist())
    parquet_seeds = set(table["seed"].to_pylist())
    parquet_methods = set(table["method"].to_pylist())
    parquet_protocols = set(table["method_protocol"].to_pylist())
    parquet_tasks = set(table["train_task"].to_pylist())
    parquet_splits = set(table["split"].to_pylist())
    if parquet_run_ids != {npz_run_id} or parquet_seeds != {npz_seed}:
        raise ValueError("O03/O04 run provenance is not aligned.")
    if parquet_methods != {npz_method} or parquet_protocols != {npz_protocol}:
        raise ValueError("O03/O04 method provenance is not aligned.")
    if parquet_tasks != {npz_task} or parquet_splits != {npz_split}:
        raise ValueError("O03/O04 task/split provenance is not aligned.")
    if expected_context is not None:
        _load_run_config(expected_context)
        expected = (
            expected_context.run_id,
            expected_context.seed,
            expected_context.method,
            expected_context.method_protocol,
            expected_context.train_task,
            expected_context.split,
        )
        actual = (
            npz_run_id,
            npz_seed,
            npz_method,
            npz_protocol,
            npz_task,
            npz_split,
        )
        if actual != expected:
            raise ValueError("S02 artifacts do not match the expected export context.")
        if npz_checkpoint_sha256 != _sha256_file(expected_context.checkpoint_path):
            raise ValueError("O04 checkpoint hash does not match the expected checkpoint.")

    return S02ValidationSummary(
        row_count=row_count,
        class_count=class_count,
        latent_dim=latent_dim,
        sample_ids=parquet_sample_ids,
        labels=parquet_labels,
    )


def _update_manifest(
    manifest_path: Path,
    *,
    context: S02ExportContext,
    summary: S02ValidationSummary,
    checkpoint_sha256: str,
    predictions_path: Path,
    latent_features_path: Path,
) -> None:
    root = manifest_path.parent
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != S02_SCHEMA_VERSION:
            raise ValueError("Existing S02 manifest has an unsupported schema version.")
        if manifest.get("run_id") != context.run_id:
            raise ValueError("Existing S02 manifest belongs to another run_id.")
    else:
        manifest = {
            "schema_version": S02_SCHEMA_VERSION,
            "run_id": context.run_id,
            "seed": context.seed,
            "method": context.method,
            "method_protocol": context.method_protocol,
            "exports": [],
        }
    export_key = (context.train_task, context.split)
    for existing in manifest["exports"]:
        if (existing["train_task"], existing["split"]) == export_key:
            raise FileExistsError(f"S02 manifest already contains export {export_key}.")
    manifest["exports"].append(
        {
            "train_task": context.train_task,
            "split": context.split,
            "row_count": summary.row_count,
            "class_count": summary.class_count,
            "class_ids": np.asarray(
                pq.read_table(predictions_path, columns=["class_ids"])["class_ids"][0].as_py(),
                dtype=np.int64,
            ).tolist(),
            "latent_dim": summary.latent_dim,
            "checkpoint": str(context.checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "predictions": str(predictions_path.relative_to(root)),
            "predictions_bytes": predictions_path.stat().st_size,
            "latent_features": str(latent_features_path.relative_to(root)),
            "latent_features_bytes": latent_features_path.stat().st_size,
        }
    )
    manifest["exports"].sort(key=lambda row: (row["train_task"], row["split"]))
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(manifest_path)


def _validate_manifest_export(
    manifest_path: Path,
    context: S02ExportContext,
    summary: S02ValidationSummary,
    checkpoint_sha256: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_header = {
        "schema_version": S02_SCHEMA_VERSION,
        "run_id": context.run_id,
        "seed": context.seed,
        "method": context.method,
        "method_protocol": context.method_protocol,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"S02 manifest {key} does not match export context.")
    exports = manifest.get("exports")
    if not isinstance(exports, list):
        raise ValueError("S02 manifest exports must be a list.")
    matches = [
        row
        for row in exports
        if row.get("train_task") == context.train_task and row.get("split") == context.split
    ]
    if len(matches) != 1:
        raise ValueError("S02 manifest must contain exactly one current task/split entry.")
    entry = matches[0]
    expected_entry = {
        "row_count": summary.row_count,
        "class_count": summary.class_count,
        "latent_dim": summary.latent_dim,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for key, expected in expected_entry.items():
        if entry.get(key) != expected:
            raise ValueError(f"S02 manifest {key} does not match validated artifacts.")
    for key in ("predictions", "latent_features"):
        relative_path = entry.get(key)
        if not isinstance(relative_path, str) or not (manifest_path.parent / relative_path).is_file():
            raise ValueError(f"S02 manifest points to a missing {key} artifact.")


def export_s02_artifacts(
    outputs: PredictionOutputs,
    metadata: Mapping[str, np.ndarray],
    context: S02ExportContext,
    output_root: str | Path,
) -> S02ArtifactPaths:
    """Validate and atomically publish one task/split O03/O04 artifact pair."""

    if context.split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported S02 split: {context.split}")
    if not context.run_id:
        raise ValueError("S02 export requires a non-empty run_id.")
    if context.train_task < 0:
        raise ValueError("train_task must be non-negative.")
    if not context.checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing S02 source checkpoint: {context.checkpoint_path}")
    _load_run_config(context)
    row_count, _, _ = _validate_outputs(outputs)
    sample_ids, drug_ids_a, drug_ids_b = _build_sample_ids(metadata, row_count)

    output_root = Path(output_root)
    manifest_path = output_root / "manifest.json"
    task_dir = output_root / f"task_{context.train_task}"
    shard_dir = task_dir / context.split
    if shard_dir.exists():
        raise FileExistsError(f"S02 shard already exists: {shard_dir}")
    task_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = task_dir / f".{context.split}.{uuid.uuid4().hex}.tmp"
    temporary_dir.mkdir()
    temporary_predictions = temporary_dir / "predictions.parquet"
    temporary_latent = temporary_dir / "latent_features.npz"
    checkpoint_sha256 = _sha256_file(context.checkpoint_path)

    try:
        table = _prediction_table(
            outputs,
            context,
            sample_ids,
            drug_ids_a,
            drug_ids_b,
        )
        pq.write_table(table, temporary_predictions, compression="zstd")
        _write_latent_npz(
            temporary_latent,
            outputs,
            context,
            sample_ids,
            drug_ids_a,
            drug_ids_b,
            checkpoint_sha256,
        )
        validate_s02_artifacts(
            temporary_predictions,
            temporary_latent,
            expected_context=context,
        )
        temporary_dir.replace(shard_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    predictions_path = shard_dir / "predictions.parquet"
    latent_features_path = shard_dir / "latent_features.npz"
    summary = validate_s02_artifacts(
        predictions_path,
        latent_features_path,
        expected_context=context,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    previous_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
    try:
        _update_manifest(
            manifest_path,
            context=context,
            summary=summary,
            checkpoint_sha256=checkpoint_sha256,
            predictions_path=predictions_path,
            latent_features_path=latent_features_path,
        )
        _validate_manifest_export(
            manifest_path,
            context,
            summary,
            checkpoint_sha256,
        )
    except Exception:
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        if previous_manifest is None:
            manifest_path.unlink(missing_ok=True)
        else:
            recovery_path = manifest_path.with_name(
                f".{manifest_path.name}.{uuid.uuid4().hex}.recovery"
            )
            recovery_path.write_bytes(previous_manifest)
            recovery_path.replace(manifest_path)
        raise
    return S02ArtifactPaths(
        predictions_path=predictions_path,
        latent_features_path=latent_features_path,
        manifest_path=manifest_path,
    )
