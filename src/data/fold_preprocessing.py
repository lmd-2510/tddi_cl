"""Opt-in raw / task-0-only frozen preprocessing for persisted development folds.

No legacy scaler payloads, trainer imports, fitting during transform, or RNG draws.
All policies reject nonfinite descriptors; no hidden imputation/drop/clip policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from src.data.ddi_dataset import DevelopmentFoldContext, iter_development_fold_arrays
from src.utils.seed import resolve_seed_configuration


POLICIES = ("raw_identity", "task0_standard_frozen")
SCHEMA_VERSION = 1
ARTIFACT_KIND = "ddi_cil_fold_preprocessing"
ARTIFACT_FILENAME = "fold_preprocessing.json"
NUMERICS = {
    "accumulator_dtype": "float64", "output_dtype": "float64",
    "variance_ddof": 0, "zero_variance_scale": 1.0,
    "nonfinite_policy": "error", "statistics_algorithm": "chan_batch_merge_v1",
    "near_zero_policy": "no_epsilon_floor", "input_layernorm": "unchanged_in_model",
}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _task0(path: str | Path) -> tuple[str, list[int]]:
    content = Path(path).read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("protocol") != "tail_to_head":
        raise ValueError("Expected P3 task-file protocol=tail_to_head.")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task file must contain tasks starting at task 0.")
    all_classes = []
    for index, task in enumerate(tasks):
        if (not isinstance(task, dict) or type(task.get("task_id")) is not int
                or task["task_id"] != index):
            raise ValueError("Task IDs must be ordered and contiguous starting at 0.")
        classes = task.get("classes")
        if (not isinstance(classes, list) or not classes
                or any(type(c) is not int or not -(2**63) <= c < 2**63 for c in classes)):
            raise ValueError("Task classes must be nonempty raw int64 class IDs.")
        all_classes.extend(classes)
    if len(all_classes) != len(set(all_classes)):
        raise ValueError("Duplicate class IDs in task file.")
    # Small synthetic/P3-prefix files are supported here; full-study layout guards
    # belong to the later study config, not this general preprocessing primitive.
    return hashlib.sha256(content).hexdigest(), tasks[0]["classes"]


def _provenance(
    context: DevelopmentFoldContext, task_file: str | Path, feature_columns: list[str],
    member_id: int, validation_fold: int, experiment_seed: int,
) -> dict:
    context.assert_unchanged()
    manifest = context.manifest
    if type(member_id) is not int or member_id not in (0, 1, 2):
        raise ValueError("member_id must be 0, 1 or 2.")
    if type(validation_fold) is not int or validation_fold != manifest["member_to_validation_fold"][str(member_id)]:
        raise ValueError("validation_fold/member mapping mismatch.")
    if type(experiment_seed) is not int or not 0 <= experiment_seed < 2**32:
        raise ValueError("experiment_seed must be a uint32 integer.")
    if (not feature_columns or not all(isinstance(c, str) and c for c in feature_columns)
            or len(set(feature_columns)) != len(feature_columns)):
        raise ValueError("feature_columns must be an ordered list of unique names.")
    task_hash, classes = _task0(task_file)
    records = []
    for rows in context._rows:
        mask = ((rows["fold_id"].to_numpy() != validation_fold)
                & np.isin(rows["raw_class_id"].to_numpy(), classes))
        for index in np.flatnonzero(mask):
            records.append([rows[key][int(index)].as_py() for key in
                            ("source_split", "source_row_index", "sample_id", "raw_class_id", "fold_id")])
    if not records or set(r[3] for r in records) != set(classes):
        raise ValueError("Task 0 has no training rows for one or more requested classes.")
    return {
        "seeds": asdict(resolve_seed_configuration(experiment_seed, member_id)),
        "validation_fold": validation_fold, "fold_seed": manifest["fold_seed"],
        "assignment_sha256": manifest["assignment_sha256"],
        "fold_manifest_sha256": context.manifest_sha256,
        # Physical paths are deliberately not part of compatibility checks.
        "sources": {s: {k: v[k] for k in ("sha256", "row_count")}
                    for s, v in manifest["sources"].items()},
        "task_file_sha256": task_hash, "protocol": "tail_to_head",
        "label_col": context.label_col,
        "feature_columns": list(feature_columns), "feature_order_sha256": _digest(feature_columns),
        "reference_selection": {
            "role": "train", "task_id": 0, "raw_class_ids": classes,
            "row_count": len(records),
            "row_fields": ["source_split", "source_row_index", "sample_id", "raw_class_id", "fold_id"],
            "rows": records, "rows_sha256": _digest(records),
            "sample_ids_sha256": _digest([r[2] for r in records]),
        },
    }


def _finite_matrix(values, width: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != width or array.dtype.kind not in "fiu":
        raise ValueError("Expected a numeric 2D descriptor matrix in artifact feature order.")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        row, col = np.argwhere(~np.isfinite(array))[0]
        raise ValueError(f"nonfinite descriptor at batch row={row}, feature column={col}; policy=error.")
    return array


def _validate_payload(payload: dict) -> None:
    if (payload.get("artifact_kind") != ARTIFACT_KIND
            or type(payload.get("schema_version")) is not int or payload["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Unsupported fold preprocessing schema.")
    if payload.get("policy") not in POLICIES or type(payload.get("policy_version")) is not int or payload["policy_version"] != 1:
        raise ValueError("Unsupported preprocessing policy/version.")
    if payload.get("numerics") != NUMERICS:
        raise ValueError("Preprocessing numeric conventions mismatch.")
    checksum = payload.get("payload_sha256")
    if checksum != _digest({k: v for k, v in payload.items() if k != "payload_sha256"}):
        raise ValueError("Preprocessing payload SHA256 mismatch.")
    provenance = payload["provenance"]
    selection = provenance["reference_selection"]
    if payload["policy"] == "raw_identity":
        if payload.get("fit") is not None or payload.get("statistics") is not None:
            raise ValueError("raw_identity must not claim fitted statistics.")
    else:
        expected_fit = {k: selection[k] for k in ("role", "task_id", "raw_class_ids", "row_count", "rows_sha256", "sample_ids_sha256")}
        if payload.get("fit") != expected_fit:
            raise ValueError("Fitted row provenance mismatch.")
        stats = payload["statistics"]
        width = len(provenance["feature_columns"])
        mean, var, scale = [np.asarray(stats[k], dtype=np.float64) for k in ("mean", "variance", "scale")]
        if any(a.shape != (width,) or not np.isfinite(a).all() for a in (mean, var, scale)):
            raise ValueError("Invalid preprocessing statistics shape/nonfinite values.")
        if (var < 0).any() or not np.array_equal(scale, np.where(var == 0, 1.0, np.sqrt(var))):
            raise ValueError("Invalid variance/scale convention.")


@dataclass(frozen=True)
class FoldPreprocessing:
    """Immutable frozen transform; reconstruct through prepare/load APIs only."""

    _payload_json: str
    _member_id: int = field(init=False, repr=False)
    _policy: str = field(init=False, repr=False)
    _columns: tuple[str, ...] = field(init=False, repr=False)
    _mean: tuple[float, ...] = field(init=False, repr=False)
    _scale: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        payload = json.loads(self._payload_json)
        _validate_payload(payload)
        object.__setattr__(self, "_member_id", payload["provenance"]["seeds"]["member_id"])
        object.__setattr__(self, "_policy", payload["policy"])
        object.__setattr__(self, "_columns", tuple(payload["provenance"]["feature_columns"]))
        stats = payload["statistics"] or {"mean": [], "scale": []}
        object.__setattr__(self, "_mean", tuple(stats["mean"]))
        object.__setattr__(self, "_scale", tuple(stats["scale"]))

    @property
    def metadata(self) -> dict:
        return json.loads(self._payload_json)

    def transform(self, values, *, feature_columns: list[str], member_id: int, policy: str) -> np.ndarray:
        if type(member_id) is not int or member_id != self._member_id:
            raise ValueError("Preprocessing member mismatch.")
        if policy != self._policy:
            raise ValueError("Preprocessing policy mismatch.")
        if tuple(feature_columns) != self._columns:
            raise ValueError("Preprocessing feature order mismatch.")
        values = _finite_matrix(values, len(feature_columns))
        if policy == "raw_identity":
            return values.copy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = (values - np.asarray(self._mean)) / np.asarray(self._scale)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite transform output; no clipping or imputation is allowed.")
        return result


def prepare_fold_preprocessing(
    context: DevelopmentFoldContext, *, task_file: str | Path, feature_columns: list[str],
    member_id: int, validation_fold: int, policy: str, experiment_seed: int = 0,
    batch_size: int = 2048,
) -> FoldPreprocessing:
    """Scan ONLY task-0 training rows. Raw validates those rows but fits nothing."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}.")
    provenance = _provenance(context, task_file, feature_columns, member_id, validation_fold, experiment_seed)
    selection = provenance["reference_selection"]
    count = 0
    mean = np.zeros(len(feature_columns), dtype=np.float64)
    m2 = np.zeros_like(mean)
    for batch in iter_development_fold_arrays(
        context, feature_columns, role="train", member_id=member_id,
        validation_fold=validation_fold, class_ids=selection["raw_class_ids"], batch_size=batch_size,
    ):
        x = _finite_matrix(batch.features, len(feature_columns))
        n = len(x)
        # Verify exact IDs/order reaching statistics, not just the final count.
        expected = selection["rows"][count:count + n]
        actual = [[batch.metadata[k][i].item() if isinstance(batch.metadata[k][i], np.generic)
                   else batch.metadata[k][i] for k in ("source_split", "source_row_index", "sample_id")]
                  + [int(batch.labels[i]), int(batch.metadata["fold_id"][i])] for i in range(n)]
        if actual != expected:
            raise ValueError("Preprocessing input rows differ from frozen task-0 training selection.")
        if policy == "task0_standard_frozen" and n:
            with np.errstate(over="ignore", invalid="ignore"):
                batch_mean = x.mean(axis=0)
                batch_m2 = np.square(x - batch_mean).sum(axis=0)
                delta = batch_mean - mean
                m2 += batch_m2 + np.square(delta) * (count * n / (count + n))
                mean += delta * (n / (count + n))
        count += n
    if count != selection["row_count"]:
        raise ValueError("Fitted row coverage mismatch.")
    if _task0(task_file)[0] != provenance["task_file_sha256"]:
        raise ValueError("Task file changed during preprocessing.")
    context.assert_unchanged()
    statistics = fit = None
    if policy == "task0_standard_frozen":
        variance = m2 / count
        scale = np.where(variance == 0, 1.0, np.sqrt(variance))
        if not all(np.isfinite(a).all() for a in (mean, variance, scale)):
            raise ValueError("Nonfinite fitted statistics; descriptor magnitudes overflow float64.")
        statistics = {"mean": mean.tolist(), "variance": variance.tolist(), "scale": scale.tolist()}
        fit = {k: selection[k] for k in ("role", "task_id", "raw_class_ids", "row_count", "rows_sha256", "sample_ids_sha256")}
    payload = {
        "artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION,
        "policy": policy, "policy_version": 1, "provenance": provenance,
        "numerics": NUMERICS, "fit": fit, "statistics": statistics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "scan_batch_size": batch_size,
    }
    payload["payload_sha256"] = _digest(payload)
    _validate_payload(payload)
    return FoldPreprocessing(_json(payload))


def save_fold_preprocessing(artifact: FoldPreprocessing, outdir: str | Path) -> Path:
    """Publish one JSON atomically in a NEW namespace; never overwrite old artifacts."""
    _validate_payload(artifact.metadata)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    temporary = outdir / f".{uuid.uuid4().hex}.tmp"
    destination = outdir / ARTIFACT_FILENAME
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(artifact._payload_json + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic no-clobber publication on the same filesystem (including NTFS).
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)  # only this call's staging file
    return destination


def load_fold_preprocessing(
    path: str | Path, *, context: DevelopmentFoldContext, task_file: str | Path,
    feature_columns: list[str], member_id: int, validation_fold: int, policy: str,
    experiment_seed: int = 0, expected_sha256: str | None = None,
) -> FoldPreprocessing:
    """Validate disk artifact against a freshly validated run/resume context; NO fit.

    ``expected_sha256`` is the file-byte digest to pin from a future checkpoint.
    Returning the frozen state never recalculates mean/variance from data.
    """
    content = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Preprocessing file SHA256 mismatch.")
    try:
        payload = json.loads(content)
        _validate_payload(payload)
        if payload["policy"] != policy:
            raise ValueError("Preprocessing policy mismatch.")
        expected = _provenance(context, task_file, feature_columns, member_id, validation_fold, experiment_seed)
        for key, value in expected.items():
            if payload["provenance"].get(key) != value:
                raise ValueError(f"Preprocessing provenance mismatch: {key}.")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed fold preprocessing artifact.") from error
    context.assert_unchanged()
    return FoldPreprocessing(_json(payload))
