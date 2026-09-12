from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.stratified_folds import (
    FOLD_ASSIGNMENT_FILENAME,
    FOLD_ASSIGNMENT_SCHEMA,
    FOLD_MANIFEST_FILENAME,
    build_stratified_fold_assignments,
    describe_fold_source,
    fold_file_sha256,
    load_fold_artifact,
    save_fold_artifact,
    validate_fold_artifact,
)


@pytest.fixture
def partition(tmp_path: Path):
    paths = {}
    sources = {}
    split_names, row_indices, ids, labels = [], [], [], []
    for split, count in (("train", 8), ("validation", 4), ("test", 2)):
        path = tmp_path / f"{split}.parquet"
        source = pa.table({
            "class": [10, 30] * (count // 2),
            "drugid-drug_a": [f"{split}_A{i}" for i in range(count)],
            "drugid-drug_b": [f"{split}_B{i}" for i in range(count)],
        })
        pq.write_table(source, path)
        paths[split] = path
        sources[split] = describe_fold_source(path)
        if split != "test":
            split_names.extend([split] * count)
            row_indices.extend(range(count))
            ids.extend(f"{split}_A{i}|{split}_B{i}" for i in range(count))
            labels.extend(source["class"].to_pylist())
    lookup = build_stratified_fold_assignments((paths["train"], paths["validation"]), seed=42)
    table = pa.Table.from_pydict({
        "source_split": split_names,
        "source_row_index": row_indices,
        "sample_id": ids,
        "raw_class_id": labels,
        "fold_id": lookup.lookup(np.asarray(ids)),
    }, schema=FOLD_ASSIGNMENT_SCHEMA)
    # Persistence must preserve supplied row order, even if it differs from lookup order.
    table = table.take(np.arange(table.num_rows - 1, -1, -1))
    outdir = tmp_path / "partition"
    saved = save_fold_artifact(outdir, table, sources=sources, fold_seed=42)
    return outdir, table, saved.manifest, paths, lookup


def _load(outdir, **kwargs):
    return load_fold_artifact(outdir / FOLD_ASSIGNMENT_FILENAME, outdir / FOLD_MANIFEST_FILENAME, **kwargs)


def _replace_column(table, name, values, dtype=None):
    dtype = dtype or table.schema.field(name).type
    return table.set_column(table.column_names.index(name), name, pa.array(values, type=dtype))


def test_round_trip_preserves_row_order_dtypes_and_runtime_lookup(partition):
    outdir, original, manifest, paths, legacy = partition
    loaded = _load(outdir, source_paths=paths, expected_metadata={"fold_seed": 42})
    assert loaded.assignments.equals(original, check_metadata=True)
    assert loaded.assignments.schema == FOLD_ASSIGNMENT_SCHEMA
    assert loaded.manifest == manifest
    assert manifest["assignment_sha256"] == fold_file_sha256(outdir / FOLD_ASSIGNMENT_FILENAME)
    assert set(p.name for p in outdir.iterdir()) == {FOLD_ASSIGNMENT_FILENAME, FOLD_MANIFEST_FILENAME}
    actual = loaded.to_lookup()
    np.testing.assert_array_equal(actual.sorted_sample_ids, legacy.sorted_sample_ids)
    np.testing.assert_array_equal(actual.sorted_fold_ids, legacy.sorted_fold_ids)
    with pytest.raises(ValueError, match="absent"):
        actual.lookup(np.asarray(["unknown|pair"]))


@pytest.mark.parametrize("field", [
    "schema_version", "artifact_kind", "fold_seed", "n_splits", "sources",
    "split_strategy", "member_to_validation_fold", "assignment_file",
    "assignment_sha256", "assignment_row_count", "created_at_utc", "creation_command",
])
def test_missing_manifest_field_is_rejected(partition, field):
    _, table, manifest, _, _ = partition
    broken = copy.deepcopy(manifest)
    del broken[field]
    with pytest.raises(ValueError, match="missing fields"):
        validate_fold_artifact(table, broken)


@pytest.mark.parametrize(("field", "value", "error"), [
    ("schema_version", 99, "schema_version"),
    ("schema_version", True, "integer"),
    ("artifact_kind", "another_kind", "artifact_kind"),
    ("n_splits", 4, "n_splits"),
    ("fold_seed", -1, "fold_seed"),
    ("fold_seed", 2**32, "uint32"),
    ("assignment_row_count", 13, "row_count"),
    ("assignment_sha256", "invalid", "SHA256"),
    ("assignment_file", "../elsewhere.parquet", "assignment_file"),
    ("split_strategy", {"name": "random"}, "split_strategy"),
    ("member_to_validation_fold", {"0": 1, "1": 0, "2": 2}, "member_to_validation_fold"),
    ("created_at_utc", "2026-09-12", "UTC"),
])
def test_invalid_manifest_is_rejected(partition, field, value, error):
    _, table, manifest, _, _ = partition
    broken = {**manifest, field: value}
    with pytest.raises(ValueError, match=error):
        validate_fold_artifact(table, broken)


@pytest.mark.parametrize("field", FOLD_ASSIGNMENT_SCHEMA.names)
def test_missing_assignment_column_is_rejected(partition, field):
    _, table, manifest, _, _ = partition
    with pytest.raises(ValueError, match="columns"):
        validate_fold_artifact(table.drop([field]), manifest)


@pytest.mark.parametrize("case", ["duplicate_id", "duplicate_row", "negative_row", "test_row", "fold_range", "missing_fold", "null_label", "wrong_dtype", "empty_id"])
def test_invalid_assignment_rows_are_rejected(partition, case):
    _, table, manifest, _, _ = partition
    if case == "duplicate_id":
        values = table["sample_id"].to_pylist()
        values[1] = values[0]
        table = _replace_column(table, "sample_id", values)
        error = "duplicate sample_id"
    elif case in {"duplicate_row", "negative_row"}:
        values = table["source_row_index"].to_pylist()
        values[1] = values[0] if case == "duplicate_row" else -1
        table = _replace_column(table, "source_row_index", values)
        error = "source_row_index coverage"
    elif case == "test_row":
        values = table["source_split"].to_pylist()
        values[0] = "test"
        table = _replace_column(table, "source_split", values)
        error = "test is excluded"
    elif case in {"fold_range", "missing_fold"}:
        values = table["fold_id"].to_pylist()
        if case == "fold_range":
            values[0] = 3
        else:
            values = [0 if value == 2 else value for value in values]
        table = _replace_column(table, "fold_id", values)
        error = "fold_id|all three folds"
    elif case == "null_label":
        values = table["raw_class_id"].to_pylist()
        values[0] = None
        table = _replace_column(table, "raw_class_id", values)
        error = "null"
    elif case == "wrong_dtype":
        table = _replace_column(table, "raw_class_id", table["raw_class_id"].to_pylist(), pa.float64())
        error = "dtype"
    else:
        values = table["sample_id"].to_pylist()
        values[0] = ""
        table = _replace_column(table, "sample_id", values)
        error = "non-empty"
    with pytest.raises(ValueError, match=error):
        validate_fold_artifact(table, manifest)


def test_assignment_byte_corruption_and_expected_metadata_mismatch(partition):
    outdir, _, _, _, _ = partition
    with pytest.raises(ValueError, match="metadata mismatch: fold_seed"):
        _load(outdir, expected_metadata={"fold_seed": 0})
    path = outdir / FOLD_ASSIGNMENT_FILENAME
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="assignment SHA256 mismatch"):
        _load(outdir)


def test_structural_validation_runs_after_valid_file_hash(partition):
    outdir, table, manifest, _, _ = partition
    path = outdir / FOLD_ASSIGNMENT_FILENAME
    pq.write_table(table.drop(["raw_class_id"]), path)
    manifest["assignment_sha256"] = fold_file_sha256(path)
    (outdir / FOLD_MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        _load(outdir)


def test_source_verification_allows_relocation_but_rejects_changed_bytes(partition, tmp_path):
    outdir, _, _, paths, _ = partition
    relocated = {}
    for split, path in paths.items():
        relocated[split] = tmp_path / f"relocated_{split}.parquet"
        relocated[split].write_bytes(path.read_bytes())
    _load(outdir, source_paths=relocated)
    source = pq.read_table(relocated["train"])
    pq.write_table(source.take(np.arange(source.num_rows - 1, -1, -1)), relocated["train"])
    with pytest.raises(ValueError, match="source SHA256 mismatch: train"):
        _load(outdir, source_paths=relocated)
    # Offline loading does not pretend to audit the source dataset.
    _load(outdir)


def test_source_row_count_and_missing_source_metadata(partition):
    outdir, table, manifest, paths, _ = partition
    broken = copy.deepcopy(manifest)
    del broken["sources"]["train"]["sha256"]
    with pytest.raises(ValueError, match="source train missing"):
        validate_fold_artifact(table, broken)
    broken = copy.deepcopy(manifest)
    broken["sources"]["test"]["row_count"] = 3
    (outdir / FOLD_MANIFEST_FILENAME).write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="source row_count mismatch: test"):
        _load(outdir, source_paths=paths)
    with pytest.raises(ValueError, match="source_paths"):
        _load(outdir, source_paths={"train": paths["train"]})


def test_save_never_overwrites_existing_output(partition):
    outdir, table, manifest, _, _ = partition
    original = {p.name: p.read_bytes() for p in outdir.iterdir()}
    with pytest.raises(FileExistsError):
        save_fold_artifact(outdir, table, sources=manifest["sources"], fold_seed=42)
    assert original == {p.name: p.read_bytes() for p in outdir.iterdir()}


def test_failed_save_does_not_publish_manifest_or_leave_staging_files(partition, tmp_path):
    _, table, manifest, _, _ = partition
    outdir = tmp_path / "interrupted"
    import src.data.stratified_folds as module
    original_replace = module.os.replace

    def interrupt_manifest(source, destination):
        if Path(destination).name == FOLD_MANIFEST_FILENAME:
            raise OSError("simulated interruption")
        original_replace(source, destination)

    with patch.object(module.os, "replace", side_effect=interrupt_manifest):
        with pytest.raises(OSError, match="simulated interruption"):
            save_fold_artifact(outdir, table, sources=manifest["sources"], fold_seed=42)
    assert {p.name for p in outdir.iterdir()} == {FOLD_ASSIGNMENT_FILENAME}
    with pytest.raises(FileNotFoundError):
        _load(outdir)
    with pytest.raises(FileExistsError):
        save_fold_artifact(outdir, table, sources=manifest["sources"], fold_seed=42)
