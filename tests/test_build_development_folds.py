from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import build_development_folds as script
from src.data.stratified_folds import (
    FOLD_ASSIGNMENT_SCHEMA, build_stratified_fold_assignments, load_fold_artifact,
)


@pytest.fixture
def inputs(tmp_path):
    paths = {}
    for split, labels in (
        ("train", [10] * 17 + [30] * 8 + [90] * 2),
        ("validation", [10] * 6 + [30] * 4 + [90]),
        ("test", ["unused_test_label"] * 3),
    ):
        paths[split] = tmp_path / f"{split}.parquet"
        pq.write_table(pa.table({
            "class": labels,
            "drugid-drug_a": [f"{split}_A{i}" for i in range(len(labels))],
            "drugid-drug_b": [f"{split}_B{i}" for i in range(len(labels))],
            "feature_not_to_read": [1.5] * len(labels),
        }), paths[split])
    return paths


def test_builder_is_deterministic_balanced_and_matches_legacy(inputs, tmp_path):
    first = script.build_development_folds(**inputs, fold_seed=42, outdir=tmp_path / "first")
    second = script.build_development_folds(**inputs, fold_seed=42, outdir=tmp_path / "second")
    other = script.build_development_folds(**inputs, fold_seed=43, outdir=tmp_path / "other")
    assert first.assignments.equals(second.assignments)
    assert first.manifest["assignment_sha256"] == second.manifest["assignment_sha256"]
    assert not first.assignments["fold_id"].equals(other.assignments["fold_id"])
    table = first.assignments
    assert table.schema == FOLD_ASSIGNMENT_SCHEMA
    assert table.num_rows == 38
    assert table["source_split"].to_pylist() == ["train"] * 27 + ["validation"] * 11
    assert table["source_row_index"].to_pylist() == list(range(27)) + list(range(11))
    assert len(set(table["sample_id"].to_pylist())) == 38
    assert all(not value.startswith("test_") for value in table["sample_id"].to_pylist())
    labels, folds = table["raw_class_id"].to_numpy(), table["fold_id"].to_numpy()
    for label in np.unique(labels):
        counts = np.bincount(folds[labels == label], minlength=3)
        assert counts.min() > 0
        assert counts.max() - counts.min() <= 1
        for fold in range(3):
            assert first.manifest["fold_summary"][str(fold)]["class_counts"][str(label)] == counts[fold]
    legacy = build_stratified_fold_assignments((inputs["train"], inputs["validation"]), seed=42)
    np.testing.assert_array_equal(first.to_lookup().sorted_fold_ids, legacy.sorted_fold_ids)
    loaded = load_fold_artifact(
        tmp_path / "first/fold_assignments.parquet", tmp_path / "first/fold_manifest.json",
        source_paths=inputs,
    )
    assert loaded.assignments.equals(table)
    assert {p.name for p in (tmp_path / "first").iterdir()} == {"fold_assignments.parquet", "fold_manifest.json"}


def test_never_reads_features_or_test_labels(inputs, tmp_path, monkeypatch):
    real_read = pq.read_table
    calls = []

    def projected_read(path, *, columns, **kwargs):
        assert "feature_not_to_read" not in columns
        if Path(path) == inputs["test"]:
            assert "class" not in columns
        calls.append((Path(path), columns))
        return real_read(path, columns=columns, **kwargs)

    monkeypatch.setattr(pq, "read_table", projected_read)
    script.build_development_folds(**inputs, fold_seed=0, outdir=tmp_path / "out")
    assert {path for path, _ in calls} == set(inputs.values())


@pytest.mark.parametrize("case", ["overlap_test", "duplicate_dev", "too_rare", "null_label", "float_label", "missing_id", "empty"])
def test_invalid_sources_fail_before_output(inputs, tmp_path, case):
    split = "test" if case == "overlap_test" else "train"
    table = pq.read_table(inputs[split])
    values = table.to_pydict()
    if case in {"overlap_test", "duplicate_dev"}:
        source = pq.read_table(inputs["validation"])
        for col in ("drugid-drug_a", "drugid-drug_b"):
            values[col][0] = source[col][0].as_py()
        table = pa.table(values)
    elif case == "too_rare":
        values["class"][0] = 999
        table = pa.table(values)
    elif case == "null_label":
        values["class"][0] = None
        table = pa.table(values)
    elif case == "float_label":
        values["class"] = [float(x) + 0.5 for x in values["class"]]
        table = pa.table(values)
    elif case == "missing_id":
        table = table.drop(["drugid-drug_a"])
    else:
        table = table.slice(0, 0)
    pq.write_table(table, inputs[split])
    with pytest.raises((ValueError, TypeError, pa.ArrowException)):
        script.build_development_folds(**inputs, fold_seed=42, outdir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_output_is_rejected_before_reading(inputs, tmp_path, monkeypatch):
    outdir = tmp_path / "out"
    outdir.mkdir()
    marker = outdir / "user_artifact.txt"
    marker.write_text("keep", encoding="utf-8")
    def unexpected_read(*args, **kwargs):
        pytest.fail("Existing output should fail before source I/O")
    monkeypatch.setattr(script, "describe_fold_source", unexpected_read)
    with pytest.raises(FileExistsError):
        script.build_development_folds(**inputs, fold_seed=0, outdir=outdir)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_source_changed_during_build_is_rejected(inputs, tmp_path, monkeypatch):
    real_builder = script.build_stratified_fold_assignments
    def changing_source(*args, **kwargs):
        result = real_builder(*args, **kwargs)
        table = pq.read_table(inputs["train"])
        pq.write_table(table.take(np.arange(table.num_rows - 1, -1, -1)), inputs["train"])
        return result
    monkeypatch.setattr(script, "build_stratified_fold_assignments", changing_source)
    with pytest.raises(ValueError, match="Source changed"):
        script.build_development_folds(**inputs, fold_seed=0, outdir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kwargs", [{"fold_seed": -1}, {"fold_seed": 2**32}, {"fold_seed": 0, "n_splits": 2}])
def test_invalid_seed_or_split_count(inputs, tmp_path, kwargs):
    with pytest.raises(ValueError):
        script.build_development_folds(**inputs, outdir=tmp_path / "out", **kwargs)


def test_cli_from_outside_repo_and_explicit_seed(inputs, tmp_path):
    command = [sys.executable, str(Path(script.__file__).resolve())]
    for split, path in inputs.items():
        command.extend([f"--{split}", str(path)])
    command.extend(["--outdir", str(tmp_path / "cli_out")])
    missing = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert missing.returncode != 0 and "--fold-seed" in missing.stderr
    result = subprocess.run(command + ["--fold-seed", "0"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "[done]" in result.stdout
    loaded = load_fold_artifact(tmp_path / "cli_out/fold_assignments.parquet", tmp_path / "cli_out/fold_manifest.json")
    assert "--fold-seed 0" in loaded.manifest["creation_command"]
