from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import audit_development_folds as audit
from scripts.build_development_folds import build_development_folds
from src.data.stratified_folds import FOLD_ASSIGNMENT_SCHEMA, describe_fold_source, fold_file_sha256


@pytest.fixture
def bundle(tmp_path):
    sources = {}
    for split, labels in (("train", [10] * 13 + [50] * 5),
                          ("validation", [10] * 4 + [50] * 4), ("test", ["not_used"] * 4)):
        path = tmp_path / f"{split}.parquet"
        pq.write_table(pa.table({"class": labels,
                                "drugid-drug_a": [f"{split}A{i}" for i in range(len(labels))],
                                "drugid-drug_b": [f"{split}B{i}" for i in range(len(labels))],
                                "feature": [1.0] * len(labels)}), path)
        sources[split] = path
    build_development_folds(**sources, fold_seed=42, outdir=tmp_path / "folds")
    return {**sources, "assignments": tmp_path / "folds/fold_assignments.parquet",
            "manifest": tmp_path / "folds/fold_manifest.json", "outdir": tmp_path / "audit"}


def _change_manifest(bundle, update):
    path = bundle["manifest"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    update(metadata)
    path.write_text(json.dumps(metadata), encoding="utf-8")


def _change_table(bundle, update):
    table = update(pq.read_table(bundle["assignments"]))
    pq.write_table(table, bundle["assignments"])
    _change_manifest(bundle, lambda m: m.update(assignment_sha256=fold_file_sha256(bundle["assignments"])))


def _source_metadata(bundle, split):
    _change_manifest(bundle, lambda m: m["sources"].update({split: describe_fold_source(bundle[split])}))


def test_valid_reports_member_sizes_and_read_only(bundle):
    hashes = {key: fold_file_sha256(path) for key, path in bundle.items() if key != "outdir"}
    result = audit.audit_development_folds(**bundle)
    assert result["status"] == "PASSED"
    assert result["development_count"] == 26
    assert result["fold_seed"] == 42
    assert all(c["status"] == "PASS" for c in result["checks"])
    assert len(result["member_views"]) == 3
    for m in result["member_views"]:
        assert m["validation_fold"] == m["member_id"]
        assert m["train_count"] + m["validation_count"] == 26
        assert m["train_count"] in (17, 18)
        assert m["validation_count"] in (8, 9)
        assert m["test_count"] == 4
        assert m["train_validation_overlap"] == 0
        assert m["train_fraction"] + m["validation_fraction"] == pytest.approx(1)
    assert {r["raw_class_id"]: r["development_count"] for r in result["class_counts"]} == {10: 17, 50: 9}
    assert {p.name for p in bundle["outdir"].iterdir()} == {
        "fold_audit.json", "fold_audit.md", "fold_class_counts.csv", "fold_member_views.csv"}
    assert json.loads((bundle["outdir"] / "fold_audit.json").read_text(encoding="utf-8")) == result
    with (bundle["outdir"] / "fold_member_views.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert "PASSED" in (bundle["outdir"] / "fold_audit.md").read_text(encoding="utf-8")
    assert all(fold_file_sha256(bundle[key]) == value for key, value in hashes.items())


@pytest.mark.parametrize("case, check_text", [
    ("missing_row", "coverage_disjointness"), ("duplicate_row", "coverage_disjointness"),
    ("duplicate_id", "coverage_disjointness"), ("invalid_fold", "coverage_disjointness"),
    ("wrong_mapping", "coverage_disjointness"), ("bad_hash", "coverage_disjointness"),
    ("wrong_source_count", "source_test"), ("source_changed", "source_train"),
    ("wrong_id", "row_identity"), ("wrong_label", "row_labels"),
    ("imbalance", "class_stratification"), ("wrong_summary", "manifest_fold_summary"),
    ("test_overlap", "development_test_ordered_overlap"),
    ("duplicate_source", "source_train"),
])
def test_bad_artifacts_fail_and_still_report(bundle, case, check_text):
    if case == "missing_row":
        _change_table(bundle, lambda t: t.slice(1))
    elif case == "duplicate_row":
        _change_table(bundle, lambda t: pa.concat_tables([t.slice(0, 1), t.slice(0, t.num_rows - 1)]))
    elif case in {"duplicate_id", "invalid_fold", "wrong_id", "wrong_label", "imbalance"}:
        def alter(t):
            values = t.to_pydict()
            if case == "duplicate_id":
                values["sample_id"][0] = values["sample_id"][1]
            elif case == "invalid_fold":
                values["fold_id"][0] = 3
            elif case == "wrong_id":
                values["sample_id"][0] = "wrong|pair"
            elif case == "wrong_label":
                values["raw_class_id"][0] = 123
            else:
                values["fold_id"] = [0 if label == 10 else (1 + i % 2)
                                     for i, label in enumerate(values["raw_class_id"])]
            return pa.Table.from_pydict(values, schema=FOLD_ASSIGNMENT_SCHEMA)
        _change_table(bundle, alter)
    elif case == "wrong_mapping":
        _change_manifest(bundle, lambda m: m["member_to_validation_fold"].update({"0": 1}))
    elif case == "bad_hash":
        _change_manifest(bundle, lambda m: m.update(assignment_sha256="0" * 64))
    elif case == "wrong_source_count":
        _change_manifest(bundle, lambda m: m["sources"]["test"].update(row_count=100))
    elif case == "wrong_summary":
        _change_manifest(bundle, lambda m: m["fold_summary"]["0"].update(row_count=0))
    else:
        split = "test" if case == "test_overlap" else "train"
        values = pq.read_table(bundle[split]).to_pydict()
        if case == "source_changed":
            values["feature"][0] = 99
        elif case == "duplicate_source":
            for col in ("drugid-drug_a", "drugid-drug_b"):
                values[col][0] = values[col][1]
        else:
            train = pq.read_table(bundle["train"]).to_pydict()
            for col in ("drugid-drug_a", "drugid-drug_b"):
                values[col][0] = train[col][0]
        pq.write_table(pa.table(values), bundle[split])
        if case != "source_changed":
            _source_metadata(bundle, split)
    result = audit.audit_development_folds(**bundle)
    assert result["status"] == "FAILED"
    assert any(check_text in c["name"] and c["status"] == "FAIL" for c in result["checks"])
    assert (bundle["outdir"] / "fold_audit.json").is_file()


def test_no_features_test_labels_or_builder_used(bundle, monkeypatch):
    real_read = pq.read_table
    def projected_read(path, **kwargs):
        if Path(path) in [bundle[s] for s in ("train", "validation", "test")]:
            assert "feature" not in kwargs["columns"]
            if Path(path) == bundle["test"]:
                assert "class" not in kwargs["columns"]
        return real_read(path, **kwargs)
    monkeypatch.setattr(pq, "read_table", projected_read)
    import src.data.stratified_folds as shared
    def forbidden(*args, **kwargs):
        pytest.fail("Audit must not rebuild folds")
    monkeypatch.setattr(shared, "build_stratified_fold_assignments", forbidden)
    assert audit.audit_development_folds(**bundle)["status"] == "PASSED"


def test_unordered_test_overlap_warns_without_changing_policy(bundle):
    values = pq.read_table(bundle["test"]).to_pydict()
    source = pq.read_table(bundle["train"]).to_pydict()
    values["drugid-drug_a"][0] = source["drugid-drug_b"][0]
    values["drugid-drug_b"][0] = source["drugid-drug_a"][0]
    pq.write_table(pa.table(values), bundle["test"])
    _source_metadata(bundle, "test")
    result = audit.audit_development_folds(**bundle)
    assert result["status"] == "PASSED"
    assert result["ordered_test_overlap_count"] == 0
    assert result["unordered_test_overlap_count"] == 1
    assert result["warnings"]


def test_shuffled_assignment_rows_and_legacy_optional_summary(bundle):
    _change_table(bundle, lambda t: t.take(np.arange(t.num_rows - 1, -1, -1)))
    _change_manifest(bundle, lambda m: m.pop("fold_summary"))
    assert audit.audit_development_folds(**bundle)["status"] == "PASSED"


def test_existing_output_preserved(bundle):
    bundle["outdir"].mkdir()
    marker = bundle["outdir"] / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit.audit_development_folds(**bundle)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_exit_codes(bundle):
    argv = [value for k, path in bundle.items() for value in (f"--{k}", str(path))]
    assert audit.main(argv) is None
    _change_manifest(bundle, lambda m: m.update(assignment_sha256="0" * 64))
    argv[-1] = str(bundle["outdir"].with_name("failed_audit"))
    with pytest.raises(SystemExit) as error:
        audit.main(argv)
    assert error.value.code == 1
    with pytest.raises(SystemExit) as error:
        audit.main(argv)
    assert error.value.code == 2


def test_input_changed_during_audit_fails(bundle, monkeypatch):
    original = audit._read_source
    def change(path, **kwargs):
        result = original(path, **kwargs)
        if path == bundle["train"]:
            table = pq.read_table(path)
            pq.write_table(table.take(np.arange(table.num_rows - 1, -1, -1)), path)
        return result
    monkeypatch.setattr(audit, "_read_source", change)
    report = audit.audit_development_folds(**bundle)
    assert report["status"] == "FAILED"
    assert any(c["name"] == "train_unchanged" and c["status"] == "FAIL" for c in report["checks"])
