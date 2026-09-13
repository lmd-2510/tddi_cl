"""Frozen loader contracts: tiny Parquet sources only, no model training."""

import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.ddi_dataset import (
    load_development_fold_arrays, load_split_arrays, prepare_development_fold_context,
)
from src.data import stratified_folds as folds


@pytest.fixture
def frozen(tmp_path):
    paths, records = {}, []
    for split, count in (("train", 12), ("validation", 6), ("test", 3)):
        paths[split] = tmp_path / f"{split}.parquet"
        labels = [10, 57, 901] * (count // 3)
        # Test deliberately has neither descriptors nor labels.
        values = {"drugid-drug_a": [f"{split}{i}" for i in range(count)],
                  "drugid-drug_b": ["B"] * count}
        if split != "test":
            values.update({"class": labels, "x": np.arange(count, dtype=float) + (100 if split == "validation" else 0),
                           "y": np.arange(count, dtype=float) * -1000})
            for i in range(count):
                records.append({"source_split": split, "source_row_index": i,
                                "sample_id": f"{split}{i}|B", "raw_class_id": labels[i],
                                "fold_id": (i // 3) % 3})
        pq.write_table(pa.table(values), paths[split], row_group_size=4)
    # Frozen partition deliberately stored in reverse order: source indices win.
    folds.save_fold_artifact(
        tmp_path / "folds", pa.Table.from_pylist(records[::-1], schema=folds.FOLD_ASSIGNMENT_SCHEMA),
        sources={s: folds.describe_fold_source(p) for s, p in paths.items()}, fold_seed=42,
    )
    return paths, tmp_path / "folds/fold_assignments.parquet", tmp_path / "folds/fold_manifest.json", records


def prepare(frozen, **kwargs):
    paths, assignment, manifest, _ = frozen
    return prepare_development_fold_context(assignment, manifest, source_paths=paths, **kwargs)


def load(context, member=0, role="train", **kwargs):
    return load_development_fold_arrays(
        context, ["y", "x"], role=role, member_id=member, validation_fold=member, **kwargs,
    )


@pytest.mark.parametrize("member", [0, 1, 2])
@pytest.mark.parametrize("role", ["train", "validation"])
@pytest.mark.parametrize("classes", [None, [901, 10], [57], [], [99999]])
def test_all_views_alignment_and_raw_class_filters(frozen, member, role, classes):
    context = prepare(frozen)
    a = load(context, member, role, class_ids=classes, batch_size=2)
    b = load(context, member, role, class_ids=classes, batch_size=7)
    expected = [r for r in frozen[3]
                if ((r["fold_id"] == member) == (role == "validation"))
                and (classes is None or r["raw_class_id"] in classes)]
    assert a.features.shape == (len(expected), 2)
    assert a.features.dtype == np.float64
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.labels, [r["raw_class_id"] for r in expected])
    for key in a.metadata:
        np.testing.assert_array_equal(a.metadata[key], b.metadata[key])
    for key in ["sample_id", "source_split", "source_row_index", "fold_id"]:
        assert a.metadata[key].tolist() == [r[key] for r in expected]
    for i, row in enumerate(expected):
        index, split = row["source_row_index"], row["source_split"]
        assert a.features[i].tolist() == [-1000 * index, index + (100 if split == "validation" else 0)]
        assert a.metadata["source_path"][i] == str(frozen[0][split].resolve())
        assert a.metadata["drugid-drug_a"][i] + "|" + a.metadata["drugid-drug_b"][i] == row["sample_id"]
        assert not row["sample_id"].startswith("test")


def test_train_heldout_disjoint_complete_and_no_refolding_or_rehash(frozen, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No refolding or repeat byte hashes during per-task reads")
    monkeypatch.setattr(folds, "StratifiedKFold", forbidden)
    monkeypatch.setattr(folds, "build_stratified_fold_assignments", forbidden)
    context = prepare(frozen)
    monkeypatch.setattr(folds, "fold_file_sha256", forbidden)
    all_validation = []
    for member in range(3):
        train = set(load(context, member).metadata["sample_id"])
        valid = set(load(context, member, "validation").metadata["sample_id"])
        assert train.isdisjoint(valid)
        assert train | valid == {r["sample_id"] for r in frozen[3]}
        all_validation.extend(valid)
    assert len(all_validation) == len(set(all_validation)) == 18


def test_context_only_reads_identity_and_labels_and_protects_manifest(frozen, monkeypatch):
    read_table = pq.read_table
    def projected_read(path, **kwargs):
        if path in frozen[0].values():
            columns = kwargs["columns"]
            assert set(columns) <= {"drugid-drug_a", "drugid-drug_b", "class"}
            if path == frozen[0]["test"]:
                assert "class" not in columns
        return read_table(path, **kwargs)
    monkeypatch.setattr(pq, "read_table", projected_read)
    context = prepare(frozen)
    context.manifest["member_to_validation_fold"]["0"] = 2
    assert context.manifest["member_to_validation_fold"]["0"] == 0


def test_change_during_identity_read_is_rejected(frozen, monkeypatch):
    read_table = pq.read_table
    def changing_read(path, **kwargs):
        result = read_table(path, **kwargs)
        if path == frozen[0]["test"]:
            with path.open("ab") as handle:
                handle.write(b"changed during validation")
        return result
    monkeypatch.setattr(pq, "read_table", changing_read)
    with pytest.raises(ValueError, match="context file changed"):
        prepare(frozen)


def update_manifest(frozen, change):
    manifest_path = frozen[2]
    manifest = json.loads(manifest_path.read_text())
    change(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def rewrite_assignment(frozen, change):
    data = pq.read_table(frozen[1]).to_pydict()
    change(data)
    pq.write_table(pa.table(data, schema=folds.FOLD_ASSIGNMENT_SCHEMA), frozen[1])
    update_manifest(frozen, lambda m: m.update(assignment_sha256=folds.fold_file_sha256(frozen[1])))


@pytest.mark.parametrize("case,match", [
    ("assignment_hash", "SHA256"), ("source_hash", "SHA256"),
    ("source_count", "row_count"), ("coverage", "coverage"),
    ("id", "sample ID agreement"), ("label", "label agreement"),
    ("mapping", "member_to_validation_fold"), ("test_assignment", "test is excluded"),
])
def test_integrity_failures(frozen, case, match):
    if case == "assignment_hash":
        with frozen[1].open("ab") as handle:
            handle.write(b"changed")
    elif case == "source_hash":
        with frozen[0]["train"].open("ab") as handle:
            handle.write(b"changed")
    elif case == "source_count":
        update_manifest(frozen, lambda m: m["sources"]["test"].update(row_count=4))
    elif case == "mapping":
        update_manifest(frozen, lambda m: m["member_to_validation_fold"].update({"0": 1}))
    else:
        field, value = {"coverage": ("source_row_index", 999), "id": ("sample_id", "wrong|ID"),
                        "label": ("raw_class_id", 123), "test_assignment": ("source_split", "test")}[case]
        rewrite_assignment(frozen, lambda data: data[field].__setitem__(0, value))
    with pytest.raises(ValueError, match=match):
        prepare(frozen)


def test_relocated_sources_and_checkpoint_metadata_pin(frozen, tmp_path):
    first = prepare(frozen)
    moved = tmp_path / "moved"
    moved.mkdir()
    paths = {}
    for split, path in frozen[0].items():
        paths[split] = moved / path.name
        shutil.copyfile(path, paths[split])
    second = prepare_development_fold_context(
        frozen[1], frozen[2], source_paths=paths,
        expected_metadata={"assignment_sha256": first.manifest["assignment_sha256"], "fold_seed": 42},
    )
    np.testing.assert_array_equal(load(first).features, load(second).features)
    np.testing.assert_array_equal(load(first).metadata["sample_id"], load(second).metadata["sample_id"])
    with pytest.raises(ValueError, match="metadata mismatch"):
        prepare(frozen, expected_metadata={"assignment_sha256": "0" * 64})


@pytest.mark.parametrize("target", ["train", "validation", "test", "assignment", "manifest"])
def test_context_detects_changes_and_resume_revalidates(frozen, target):
    context = prepare(frozen)
    path = frozen[1] if target == "assignment" else frozen[2] if target == "manifest" else frozen[0][target]
    with path.open("ab") as handle:
        handle.write(b" " if target == "manifest" else b"changed")
    with pytest.raises(ValueError, match="context file changed"):
        load(context)
    if target != "manifest":
        with pytest.raises(ValueError, match="SHA256"):
            prepare(frozen)
    else:
        new = prepare(frozen)  # JSON whitespace valid, but provenance hash differs.
        assert new.manifest_sha256 != context.manifest_sha256


def test_test_overlap_even_when_source_hash_matches(frozen):
    path = frozen[0]["test"]
    values = pq.read_table(path).to_pydict()
    values["drugid-drug_a"][0] = "train0"
    pq.write_table(pa.table(values), path)
    update_manifest(frozen, lambda m: m["sources"].update(test=folds.describe_fold_source(path)))
    with pytest.raises(ValueError, match="Development/test sample ID overlap"):
        prepare(frozen)


@pytest.mark.parametrize("labels", [[1.5] * 12, [None] * 12, [2**63] * 12])
def test_source_label_types_not_silently_coerced(frozen, labels):
    path = frozen[0]["train"]
    table = pq.read_table(path)
    array = pa.array(labels, type=pa.uint64() if labels[0] == 2**63 else None)
    table = table.set_column(table.schema.get_field_index("class"), "class", array)
    pq.write_table(table, path)
    update_manifest(frozen, lambda m: m["sources"].update(train=folds.describe_fold_source(path)))
    with pytest.raises(ValueError, match="labels"):
        prepare(frozen)


@pytest.mark.parametrize("kwargs,match", [
    ({"role": "test"}, "role"), ({"member_id": 3}, "member_id"),
    ({"member_id": True}, "member_id"), ({"validation_fold": 1}, "mapping"),
    ({"class_ids": [10.5]}, "class_ids"), ({"class_ids": [True]}, "class_ids"),
    ({"class_ids": [2**64]}, "class_ids"), ({"batch_size": 0}, "batch_size"),
    ({"feature_columns": ["class"]}, "feature_columns"),
    ({"feature_columns": ["x", "x"]}, "feature_columns"),
    ({"feature_columns": ["missing"]}, "descriptor"),
])
def test_invalid_requests(frozen, kwargs, match):
    args = dict(feature_columns=["x"], role="train", member_id=0, validation_fold=0)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        load_development_fold_arrays(prepare(frozen), **args)


def test_raw_nonfinite_no_imputation_and_legacy_unchanged(frozen):
    path = frozen[0]["train"]
    table = pq.read_table(path)
    values = table["x"].to_pylist()
    values[3:6] = [None, float("inf"), float("nan")]
    table = table.set_column(table.schema.get_field_index("x"), "x", pa.array(values))
    pq.write_table(table, path)
    update_manifest(frozen, lambda m: m["sources"].update(train=folds.describe_fold_source(path)))
    result = load(prepare(frozen))
    assert np.isnan(result.features[0, 1])
    assert np.isposinf(result.features[1, 1])
    assert np.isnan(result.features[2, 1])
    legacy = load_split_arrays(path, ["y"], class_ids=[10])
    assert legacy.features.dtype == np.float32
    assert legacy.metadata is None
    np.testing.assert_array_equal(legacy.features[:, 0], [0, -3000, -6000, -9000])
