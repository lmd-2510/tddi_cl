"""Task-0-only frozen preprocessing: synthetic files, never real data/training."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.ddi_dataset import prepare_development_fold_context, transform_features
from src.data import fold_preprocessing as prep
from src.data.stratified_folds import FOLD_ASSIGNMENT_SCHEMA, describe_fold_source, fold_file_sha256, save_fold_artifact


@pytest.fixture
def data(tmp_path):
    paths, rows = {}, []
    for split in ("train", "validation", "test"):
        paths[split] = tmp_path / f"{split}.parquet"
        n = 9 if split != "test" else 1
        values = {"drugid-drug_a": [f"{split}{i}" for i in range(n)], "drugid-drug_b": ["B"] * n}
        if split != "test":
            values.update({"class": [10, 57, 901] * 3,
                           "x": [float(2 * (i // 3) + (6 if split == "validation" else 0)) for i in range(n)],
                           "constant": [5.0] * n})
            rows.extend({"source_split": split, "source_row_index": i,
                         "sample_id": f"{split}{i}|B", "raw_class_id": values["class"][i],
                         "fold_id": i // 3} for i in range(n))
        pq.write_table(pa.table(values), paths[split], row_group_size=3)
    save_fold_artifact(tmp_path / "folds", pa.Table.from_pylist(rows, schema=FOLD_ASSIGNMENT_SCHEMA),
                       sources={s: describe_fold_source(p) for s, p in paths.items()}, fold_seed=42)
    task = tmp_path / "tasks.json"
    task.write_text(json.dumps({"protocol": "tail_to_head", "tasks": [
        {"task_id": 0, "classes": [10]}, {"task_id": 1, "classes": [57, 901]}]}))
    return paths, tmp_path / "folds", task


def context(data):
    paths, folder, _ = data
    return prepare_development_fold_context(folder / "fold_assignments.parquet", folder / "fold_manifest.json", source_paths=paths)


def args(data, member=0, policy="task0_standard_frozen"):
    return dict(context=context(data), task_file=data[2], feature_columns=["x", "constant"],
                member_id=member, validation_fold=member, policy=policy, experiment_seed=0)


def transform(artifact, values, member=0):
    return artifact.transform(np.asarray(values, dtype=np.float64), feature_columns=["x", "constant"],
                              member_id=member, policy=artifact.metadata["policy"])


@pytest.mark.parametrize("member,mean,var", [(0, 6, 10), (1, 5, 13), (2, 4, 10)])
@pytest.mark.parametrize("batch_size", [1, 4, 100])
def test_hand_statistics_and_exact_fit_ids(data, member, mean, var, batch_size):
    artifact = prep.prepare_fold_preprocessing(**args(data, member), batch_size=batch_size)
    metadata = artifact.metadata
    np.testing.assert_allclose(metadata["statistics"]["mean"], [mean, 5], atol=1e-14)
    np.testing.assert_allclose(metadata["statistics"]["variance"], [var, 0], atol=1e-14)
    np.testing.assert_allclose(metadata["statistics"]["scale"], [np.sqrt(var), 1], atol=1e-14)
    rows = metadata["provenance"]["reference_selection"]["rows"]
    expected_ids = [f"{split}{i}|B" for split in ("train", "validation")
                    for i in (0, 3, 6) if i // 3 != member]
    assert [r[2] for r in rows] == expected_ids
    assert all(r[3] == 10 and r[4] != member for r in rows)
    assert metadata["fit"]["row_count"] == 4
    assert metadata["fit"]["sample_ids_sha256"] == prep._digest(expected_ids)
    assert metadata["provenance"]["seeds"]["member_id"] == member
    np.testing.assert_allclose(transform(artifact, [[mean, 5], [mean + np.sqrt(var), 7]], member), [[0, 0], [1, 2]])


def test_raw_identity_no_fake_fit_and_same_ab_rows(data):
    raw = prep.prepare_fold_preprocessing(**args(data, policy="raw_identity"))
    scaled = prep.prepare_fold_preprocessing(**args(data))
    assert raw.metadata["fit"] is None and raw.metadata["statistics"] is None
    assert raw.metadata["provenance"] == scaled.metadata["provenance"]
    x = np.array([[3.0, 5.0], [1234.0, 8.0]])
    result = transform(raw, x)
    np.testing.assert_array_equal(result, x)
    result[0, 0] = -999
    assert x[0, 0] == 3  # transform does not return a mutable view of input


def test_p4_task_file_fits_and_loads_with_p4_provenance(data, tmp_path):
    payload = json.loads(data[2].read_text())
    payload["protocol"] = "constrained_mass_balanced"
    payload["seed"] = 0
    p4_task = tmp_path / "p4_tasks.json"
    p4_task.write_text(json.dumps(payload))
    p4_data = data[0], data[1], p4_task

    artifact = prep.prepare_fold_preprocessing(**args(p4_data))
    assert artifact.metadata["provenance"]["protocol"] == "constrained_mass_balanced"
    path = prep.save_fold_preprocessing(artifact, tmp_path / "p4_preprocessing")
    loaded = prep.load_fold_preprocessing(path, **args(p4_data))
    assert loaded.metadata == artifact.metadata


@pytest.mark.parametrize("kwargs,match", [
    ({"member_id": 2}, "member"), ({"policy": "raw_identity"}, "policy"),
    ({"feature_columns": ["constant", "x"]}, "feature order"),
])
def test_transform_guards(data, kwargs, match):
    artifact = prep.prepare_fold_preprocessing(**args(data))
    call = dict(feature_columns=["x", "constant"], member_id=0, policy="task0_standard_frozen")
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        artifact.transform([[1.0, 5.0]], **call)


@pytest.mark.parametrize("policy", prep.POLICIES)
def test_transform_nonfinite_shape_and_empty(data, policy):
    artifact = prep.prepare_fold_preprocessing(**args(data, policy=policy))
    for bad in ([[np.inf, 5]], [[np.nan, 5]], [[1, 2, 3]]):
        with pytest.raises(ValueError):
            transform(artifact, bad)
    empty = transform(artifact, np.empty((0, 2)))
    assert empty.shape == (0, 2)


def test_streaming_fit_batches_are_bounded(data, monkeypatch):
    original = prep.iter_development_fold_arrays
    sizes = []
    def tracked(*a, **kw):
        for batch in original(*a, **kw):
            sizes.append(len(batch.labels))
            assert len(batch.labels) <= 2
            assert set(batch.labels) == {10}
            assert not np.any(batch.metadata["fold_id"] == 0)
            yield batch
    monkeypatch.setattr(prep, "iter_development_fold_arrays", tracked)
    prep.prepare_fold_preprocessing(**args(data), batch_size=2)
    assert sum(sizes) == 4 and len(sizes) > 1


@pytest.mark.parametrize("kind", ["standard", "robust"])
def test_legacy_scaler_still_imputes_and_returns_float32(kind):
    payload = {"scaler_type": kind, "impute_values": [4.0, 6.0],
               "scale": [2.0, 3.0], "mean": [2.0, 3.0], "center": [2.0, 3.0]}
    result = transform_features(np.array([[np.nan, np.inf], [6.0, 9.0]]), payload)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[1, 1], [2, 2]])


def rewrite_source(data, split, edit):
    path = data[0][split]
    values = pq.read_table(path).to_pydict()
    edit(values)
    pq.write_table(pa.table(values), path)
    manifest_path = data[1] / "fold_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"][split] = describe_fold_source(path)
    manifest_path.write_text(json.dumps(manifest))


def test_future_heldout_values_do_not_affect_fit_but_change_hashes(data):
    before = prep.prepare_fold_preprocessing(**args(data))
    for split in ("train", "validation"):
        def edit(values):
            for i in range(9):
                if i not in (3, 6):  # not member0 task0 training
                    values["x"][i] = float("nan") if i == 0 else 1e30
        rewrite_source(data, split, edit)
    after = prep.prepare_fold_preprocessing(**args(data))
    assert before.metadata["statistics"] == after.metadata["statistics"]
    assert before.metadata["fit"] == after.metadata["fit"]
    assert before.metadata["provenance"]["sources"] != after.metadata["provenance"]["sources"]
    # Transform is the same frozen operator for validation/test/future/replay;
    # nonfinite values at those later calls still fail, not silently impute.
    with pytest.raises(ValueError, match="nonfinite"):
        transform(after, [[float("nan"), 5]])


@pytest.mark.parametrize("policy", prep.POLICIES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None])
def test_nonfinite_fit_rows_fail_for_both_policies(data, policy, bad):
    rewrite_source(data, "train", lambda v: v["x"].__setitem__(3, bad))
    with pytest.raises(ValueError, match="nonfinite"):
        prep.prepare_fold_preprocessing(**args(data, policy=policy))


@pytest.mark.parametrize("policy", prep.POLICIES)
def test_roundtrip_load_never_fits_and_relocation(data, tmp_path, monkeypatch, policy):
    artifact = prep.prepare_fold_preprocessing(**args(data, policy=policy))
    path = prep.save_fold_preprocessing(artifact, tmp_path / policy)
    original_bytes = path.read_bytes()
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    moved = {}
    for split, source in data[0].items():
        moved[split] = relocated / source.name
        shutil.copyfile(source, moved[split])
    relocated_data = moved, data[1], data[2]
    def forbidden(*a, **kw):
        pytest.fail("Frozen load/transform must never fit or scan descriptors")
    monkeypatch.setattr(prep, "iter_development_fold_arrays", forbidden)
    frozen = prep.load_fold_preprocessing(path, **args(relocated_data, policy=policy), expected_sha256=fold_file_sha256(path))
    assert frozen.metadata == artifact.metadata
    np.testing.assert_array_equal(transform(frozen, [[8, 5]]), transform(artifact, [[8, 5]]))
    assert path.read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        prep.save_fold_preprocessing(artifact, path.parent)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("change,match", [
    ({"member_id": 1, "validation_fold": 1}, "seeds"),
    ({"validation_fold": 1}, "mapping"),
    ({"policy": "raw_identity"}, "policy"),
    ({"experiment_seed": 1}, "seeds"),
    ({"feature_columns": ["constant", "x"]}, "feature_columns"),
    ({"expected_sha256": "0" * 64}, "SHA256"),
])
def test_frozen_wrong_member_policy_seed_feature_hash(data, tmp_path, change, match):
    artifact = prep.prepare_fold_preprocessing(**args(data))
    path = prep.save_fold_preprocessing(artifact, tmp_path / "out")
    kw = args(data)
    kw.update(change)
    with pytest.raises(ValueError, match=match):
        prep.load_fold_preprocessing(path, **kw)


@pytest.mark.parametrize("target", ["task", "source", "assignment"])
def test_frozen_provenance_mismatch(data, tmp_path, target):
    path = prep.save_fold_preprocessing(prep.prepare_fold_preprocessing(**args(data)), tmp_path / "out")
    if target == "task":
        with data[2].open("a") as handle:
            handle.write(" ")
    elif target == "source":
        rewrite_source(data, "train", lambda v: v["x"].__setitem__(3, 100.0))
    else:
        assignment_path = data[1] / "fold_assignments.parquet"
        table = pq.read_table(assignment_path)
        pq.write_table(table, assignment_path, compression="none")
        manifest_path = data[1] / "fold_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["assignment_sha256"] = fold_file_sha256(assignment_path)
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance mismatch"):
        prep.load_fold_preprocessing(path, **args(data))


def test_corrupt_artifact_and_invalid_stats(data, tmp_path):
    artifact = prep.prepare_fold_preprocessing(**args(data))
    path = prep.save_fold_preprocessing(artifact, tmp_path / "out")
    payload = artifact.metadata
    payload["statistics"]["mean"][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="payload SHA256"):
        prep.load_fold_preprocessing(path, **args(data))
    payload["statistics"]["scale"][0] = 0
    payload["payload_sha256"] = prep._digest({k: v for k, v in payload.items() if k != "payload_sha256"})
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="variance/scale"):
        prep.load_fold_preprocessing(path, **args(data))


def test_exact_rows_guard_not_only_count(data, monkeypatch):
    original = prep.iter_development_fold_arrays
    def swapped(*a, **kw):
        for batch in original(*a, **kw):
            batch.metadata["sample_id"] = batch.metadata["sample_id"][::-1]
            yield batch
    monkeypatch.setattr(prep, "iter_development_fold_arrays", swapped)
    with pytest.raises(ValueError, match="input rows differ"):
        prep.prepare_fold_preprocessing(**args(data))


def test_atomic_publish_failure_leaves_no_final_file(data, tmp_path, monkeypatch):
    artifact = prep.prepare_fold_preprocessing(**args(data))
    def fail(*a, **kw):
        raise OSError("simulated publication failure")
    monkeypatch.setattr(prep.os, "link", fail)
    with pytest.raises(OSError, match="simulated"):
        prep.save_fold_preprocessing(artifact, tmp_path / "new")
    assert list((tmp_path / "new").iterdir()) == []


def test_cli_synthetic_and_no_overwrite(data, tmp_path):
    columns = tmp_path / "features.json"
    columns.write_text(json.dumps(["x", "constant"]))
    cmd = [sys.executable, "scripts/prepare_fold_preprocessing.py",
           "--assignments", str(data[1] / "fold_assignments.parquet"),
           "--manifest", str(data[1] / "fold_manifest.json"),
           "--task-file", str(data[2]), "--feature-cols", str(columns),
           "--member-id", "0", "--validation-fold", "0", "--policy", "task0_standard_frozen",
           "--batch-size", "2", "--outdir", str(tmp_path / "cli")]
    for split, path in data[0].items():
        cmd.extend([f"--{split}", str(path)])
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert run.returncode == 0, run.stderr
    assert "reference_rows=4 fitted=True" in run.stdout
    path = tmp_path / "cli" / prep.ARTIFACT_FILENAME
    before = path.read_bytes()
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert run.returncode != 0 and "already exists" in run.stderr
    assert path.read_bytes() == before
