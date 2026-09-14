"""Tiny synthetic allocation/ranking/fold-buffer tests; no model training."""

from copy import deepcopy
import json
import pickle
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.ddi_dataset import DDIBatchArrays, load_development_fold_arrays, prepare_development_fold_context
from src.data.fold_preprocessing import prepare_fold_preprocessing
from src.data import fold_replay_buffer as replay
from src.data.stratified_folds import FOLD_ASSIGNMENT_SCHEMA, describe_fold_source, save_fold_artifact


@pytest.mark.parametrize("observed,caps,budget,q,expected", [
    ({10: 100, 20: 400}, {10: 100, 20: 400}, 50, 10, {10: 20, 20: 30}),
    ({10: 100, 20: 400, 30: 900}, {10: 11, 20: 100, 30: 100}, 60, 10, {10: 11, 20: 22, 30: 27}),
    ({10: 4, 20: 100, 30: 100}, {10: 4, 20: 100, 30: 100}, 34, 10, {10: 4, 20: 15, 30: 15}),
    ({10: 100, 20: 100, 30: 1}, {10: 2, 20: 6, 30: 1}, 8, 10, {10: 2, 20: 5, 30: 1}),
    ({10: 100, 20: 100, 30: 1}, {10: 100, 20: 100, 30: 1}, 8, 10, {10: 4, 20: 3, 30: 1}),
    ({10: 1, 20: 9, 30: 9}, {10: 1, 20: 9, 30: 9}, 5, 10, {10: 1, 20: 2, 30: 2}),
    ({10: 5, 20: 5, 30: 5}, {10: 5, 20: 5, 30: 5}, 2, 10, {10: 1, 20: 1, 30: 0}),
    ({10: 100, 20: 4}, {10: 0, 20: 2}, 100, 10, {10: 0, 20: 2}),
    ({10: 100, 20: 100}, {10: 100, 20: 100}, 5, 0, {10: 3, 20: 2}),
    ({10: 4}, {10: 3}, 0, 10, {10: 0}),
    ({}, {}, 0, 10, {}),
])
def test_allocation_by_hand(observed, caps, budget, q, expected):
    assert replay.min_quota_sqrt_allocation(observed, caps, budget, base_quota=q) == expected
    assert replay.min_quota_sqrt_allocation(dict(reversed(list(observed.items()))), dict(reversed(list(caps.items()))), budget, base_quota=q) == expected


def test_allocation_invariants_random_small_cases():
    rng = np.random.default_rng(12)
    for _ in range(200):
        observed = {int(c): int(n) for c, n in zip([9, 57, 901, -2], rng.integers(1, 500, 4))}
        caps = {c: int(rng.integers(0, n + 1)) for c, n in observed.items()}
        budget = int(rng.integers(0, 1000))
        result = replay.min_quota_sqrt_allocation(observed, caps, budget, base_quota=int(rng.integers(0, 20)))
        assert sum(result.values()) == min(budget, sum(caps.values()))
        assert all(0 <= result[c] <= caps[c] for c in caps)


@pytest.mark.parametrize("observed,caps,budget,q", [
    ({1: 0}, {1: 0}, 10, 10), ({1: 5}, {1: 6}, 10, 10),
    ({1: 5}, {2: 5}, 10, 10), ({1: 5}, {1: -1}, 10, 10),
    ({1: 5}, {1: 2}, -1, 10), ({1: 5}, {1: 2}, 10, -1),
    ({1: 5.2}, {1: 2}, 10, 10), ({True: 5}, {True: 2}, 10, 10),
])
def test_allocation_invalid_inputs(observed, caps, budget, q):
    with pytest.raises(ValueError):
        replay.min_quota_sqrt_allocation(observed, caps, budget, base_quota=q)


def test_ranking_by_hand_and_input_permutation():
    x = np.array([[0., 2.], [0., 2.], [2., 0.]])
    ids = np.array(["b", "a", "c"])
    order, distances = replay.rank_raw_class_exemplars(x, ids)
    assert ids[order].tolist() == ["a", "b", "c"]
    # z=(+-1/sqrt(1+eps), -+1/sqrt(1+eps)); two positive, one negative.
    d = np.sqrt(2) / np.sqrt(1 + replay.RANKING_EPSILON)
    np.testing.assert_allclose(distances, [2*d/3, 2*d/3, 4*d/3])
    for perm in ([2, 0, 1], [1, 2, 0]):
        p, ds = replay.rank_raw_class_exemplars(x[perm], ids[perm])
        assert ids[perm][p].tolist() == ["a", "b", "c"]
        np.testing.assert_array_equal(ds, distances)
    p, ds = replay.rank_raw_class_exemplars(np.array([[3., 3.], [99., 99.]]), np.array(["z", "a"]))
    assert p.tolist() == [1, 0]
    np.testing.assert_array_equal(ds, [0, 0])


def test_pipeline_input_ranking_by_hand_and_permutation():
    x = np.array([[0., 0.], [2., 0.], [10., 0.]])
    ids = np.array(["b", "a", "c"])
    order, distances = replay.rank_input_class_exemplars(x, ids)
    assert ids[order].tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(distances, [2., 4., 6.])
    perm = np.array([2, 0, 1])
    other, other_distances = replay.rank_input_class_exemplars(x[perm], ids[perm])
    assert ids[perm][other].tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(other_distances, distances)
    # The two candidate spaces are genuinely different ranking functions.
    raw = np.array([
        [1.1626641008, 6.4952563575, 1.0248931554],
        [-4.3842692045, 7.1572402525, 1.3844882337],
        [-1.8064953921, 4.5939967254, 1.1307682473],
        [0.9895628984, 0.2246904417, 1.6957007501],
    ])
    names = np.array(["a", "b", "c", "d"])
    sample_order, _ = replay.rank_raw_class_exemplars(raw, names)
    input_order, _ = replay.rank_input_class_exemplars(raw, names)
    assert names[sample_order].tolist() != names[input_order].tolist()


@pytest.mark.parametrize("x,ids", [([[np.nan, 2]], ["a"]), ([[np.inf, 2]], ["a"]),
    ([[1, 2], [2, 3]], ["a", "a"]), ([[1, 2]], [""]), ([[1e308, -1e308]], ["a"])])
def test_ranking_invalid(x, ids):
    with pytest.raises(ValueError):
        replay.rank_raw_class_exemplars(np.asarray(x), np.asarray(ids))


@pytest.fixture
def data(tmp_path):
    paths, rows = {}, []
    for split in ("train", "validation", "test"):
        paths[split] = tmp_path / f"{split}.parquet"
        count = 9 if split != "test" else 1
        values = {"drugid-drug_a": [f"{split}{i}" for i in range(count)], "drugid-drug_b": ["B"] * count}
        if split != "test":
            values.update({"class": [10, 57, 901] * 3, "x": [float(i + (10 if split == "validation" else 0)) for i in range(count)], "y": [5.] * count})
            rows.extend({"source_split": split, "source_row_index": i, "sample_id": f"{split}{i}|B",
                         "raw_class_id": values["class"][i], "fold_id": i // 3} for i in range(count))
        pq.write_table(pa.table(values), paths[split])
    save_fold_artifact(tmp_path / "folds", pa.Table.from_pylist(rows, schema=FOLD_ASSIGNMENT_SCHEMA),
                       sources={s: describe_fold_source(p) for s, p in paths.items()}, fold_seed=42)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"protocol": "tail_to_head", "tasks": [
        {"task_id": i, "classes": [c]} for i, c in enumerate((10, 57, 901))]}))
    return paths, tmp_path / "folds", tasks


def settings(data, member=0, budget=5):
    ctx = prepare_development_fold_context(data[1] / "fold_assignments.parquet", data[1] / "fold_manifest.json", source_paths=data[0])
    return dict(context=ctx, task_file=data[2], feature_columns=["x", "y"], member_id=member, total_memory_budget=budget)


def current(kwargs, task, role="train"):
    return load_development_fold_arrays(kwargs["context"], ["x", "y"], member_id=kwargs["member_id"],
        validation_fold=kwargs["member_id"], role=role, class_ids=[[10], [57], [901]][task])


def update(buffer, kwargs, task):
    return buffer.update(current(kwargs, task), task_id=task, feature_columns=["x", "y"])


def assert_arrays_equal(a, b):
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert set(a.metadata) == set(b.metadata)
    for k in a.metadata:
        np.testing.assert_array_equal(a.metadata[k], b.metadata[k])


def test_buffer_arrival_counts_trim_prefix_no_old_feature_read(data, monkeypatch):
    kw = settings(data)
    buffer = replay.FoldSqrtReplayBuffer(**kw)
    incoming = [current(kw, task) for task in range(3)]
    def no_read(*a, **kw):
        pytest.fail("Buffer must not reread current/discarded Parquet features")
    monkeypatch.setattr(pq, "read_table", no_read)
    saved = {}
    for task in range(3):
        before = buffer.memory_counts
        audit = buffer.update(incoming[task], task_id=task, feature_columns=["x", "y"])
        assert buffer.observed_counts == {c: 4 for c in [10, 57, 901][:task + 1]}
        assert audit["feasible_capacities"] == {**before, [10, 57, 901][task]: 4}
        arrays = buffer.get_all()
        assert buffer.total_size == min(5, sum(audit["feasible_capacities"].values()))
        for c in buffer.memory_counts:
            ids = arrays.metadata["sample_id"][arrays.labels == c].tolist()
            if c in saved:
                assert ids == saved[c][:len(ids)]
            saved[c] = ids
        assert 0 not in arrays.metadata["fold_id"]
    assert buffer.memory_counts == {10: 2, 57: 2, 901: 1}
    arrays.features[:] = -100  # caller mutations cannot corrupt retained storage
    assert np.all(buffer.get_all().features >= 0)
    incoming[0].features[:] = -200
    assert np.all(buffer.get_all().features >= 0)
    state = buffer.state_dict()
    assert sum(len(e["labels"]) for e in state["entries"].values()) == 5
    assert all(set(e) == {"features", "labels", "metadata"} for e in state["entries"].values())


@pytest.mark.parametrize("case", ["heldout", "future", "test", "partial", "duplicate", "label", "source_row", "nan", "wrong_order"])
def test_invalid_current_rejected_transactionally(data, case):
    kw = settings(data)
    buffer = replay.FoldSqrtReplayBuffer(**kw)
    rows = current(kw, 0)
    if case == "heldout":
        rows = current(kw, 0, "validation")
    elif case == "future":
        rows = current(kw, 1)
    elif case == "partial":
        rows = replay._copy_arrays(rows, slice(1, None))
    else:
        rows = deepcopy(rows)
        if case == "test":
            rows.metadata["drugid-drug_a"][0] = "test0"
            rows.metadata["sample_id"][0] = "test0|B"
            rows.metadata["source_split"][0] = "test"
        elif case == "duplicate":
            rows = replay._copy_arrays(rows, [0, 0, 2, 3])
        elif case == "label":
            rows.labels = rows.labels.copy()
            rows.labels[0] = 57
        elif case == "source_row":
            rows.metadata["source_row_index"] = rows.metadata["source_row_index"].copy()
            rows.metadata["source_row_index"][0] += 1
        elif case == "nan":
            rows.features[0, 0] = np.nan
    before = buffer.state_dict()["state_sha256"]
    with pytest.raises(ValueError):
        buffer.update(rows, task_id=0, feature_columns=["y", "x"] if case == "wrong_order" else ["x", "y"])
    assert buffer.state_dict()["state_sha256"] == before


def test_ab_ids_equal_model_features_can_differ(data):
    kw = settings(data)
    a, b = replay.FoldSqrtReplayBuffer(**kw), replay.FoldSqrtReplayBuffer(**kw)
    common = {k: v for k, v in kw.items() if k != "total_memory_budget"}
    raw = prepare_fold_preprocessing(**common, validation_fold=0, policy="raw_identity")
    scaler = prepare_fold_preprocessing(**common, validation_fold=0, policy="task0_standard_frozen")
    for task in range(3):
        rows = current(kw, task)
        a.update(rows, task_id=task, feature_columns=["x", "y"])
        # Reordering incoming rows must not change tie break or class mean.
        b.update(replay._copy_arrays(rows, np.arange(len(rows.labels))[::-1]), task_id=task, feature_columns=["x", "y"])
        assert_arrays_equal(a.get_all(), b.get_all())
        av, bv = a.model_arrays(raw), b.model_arrays(scaler)
        np.testing.assert_array_equal(av.metadata["sample_id"], bv.metadata["sample_id"])
        assert not np.allclose(av.features, bv.features)


def test_two_ranking_spaces_provenance_selection_and_roundtrip(data):
    kw = settings(data, budget=1)
    prep_kwargs = {k: v for k, v in kw.items() if k != "total_memory_budget"}
    scaler = prepare_fold_preprocessing(**prep_kwargs, validation_fold=0,
                                        policy="task0_standard_frozen")
    pipeline_kw = {
        **kw,
        "ranking_policy": replay.PIPELINE_INPUT_RANKING_POLICY,
        "ranking_preprocessing": scaler,
        "ranking_preprocessing_sha256": "a" * 64,
    }
    sample_buffer = replay.FoldSqrtReplayBuffer(**kw)
    pipeline_buffer = replay.FoldSqrtReplayBuffer(**pipeline_kw)
    rows = current(kw, 0)
    sample_buffer.update(rows, task_id=0, feature_columns=["x", "y"])
    pipeline_buffer.update(rows, task_id=0, feature_columns=["x", "y"])
    metadata = pipeline_buffer.metadata["ranking"]
    assert metadata["policy"] == replay.PIPELINE_INPUT_RANKING_POLICY
    assert metadata["preprocessing_policy"] == "task0_standard_frozen"
    assert metadata["preprocessing_sha256"] == "a" * 64
    assert metadata["feature_space"] == "frozen_preprocessing_output_before_model_layernorm"
    class_rows = rows.features[rows.labels == 10]
    class_ids = rows.metadata["sample_id"][rows.labels == 10]
    transformed = scaler.transform(class_rows, feature_columns=["x", "y"], member_id=0,
                                   policy="task0_standard_frozen")
    expected, _ = replay.rank_input_class_exemplars(transformed, class_ids)
    assert pipeline_buffer.get_all().metadata["sample_id"].tolist() == [class_ids[expected[0]]]
    # Candidate spaces may select the same IDs for a particular class; equality is
    # not forced either way. Provenance, not artificial divergence, distinguishes them.
    snapshot = pickle.loads(pickle.dumps(pipeline_buffer.state_dict()))
    restored = replay.FoldSqrtReplayBuffer.from_state_dict(snapshot, **pipeline_kw)
    assert_arrays_equal(restored.get_all(), pipeline_buffer.get_all())
    with pytest.raises(ValueError, match="metadata mismatch"):
        replay.FoldSqrtReplayBuffer.from_state_dict(snapshot, **kw)


def test_pipeline_ranking_requires_approved_frozen_scaler(data):
    kw = settings(data)
    common = {k: v for k, v in kw.items() if k != "total_memory_budget"}
    raw = prepare_fold_preprocessing(**common, validation_fold=0, policy="raw_identity")
    with pytest.raises(ValueError, match="task0_standard_frozen"):
        replay.FoldSqrtReplayBuffer(
            **kw, ranking_policy=replay.PIPELINE_INPUT_RANKING_POLICY,
            ranking_preprocessing=raw, ranking_preprocessing_sha256="a" * 64,
        )


@pytest.mark.parametrize("save_after,budget", [(-1, 5), (0, 5), (1, 5), (2, 5), (0, 0), (1, 1)])
def test_state_roundtrip_continuation(data, save_after, budget):
    kw = settings(data, budget=budget)
    continuous = replay.FoldSqrtReplayBuffer(**kw)
    for task in range(save_after + 1):
        update(continuous, kw, task)
    # Exercise serialization rather than relying on shared array references.
    snapshot = pickle.loads(pickle.dumps(continuous.state_dict()))
    resumed = replay.FoldSqrtReplayBuffer.from_state_dict(snapshot, **settings(data, budget=budget))
    assert resumed.next_task_id == save_after + 1
    for task in range(save_after + 1, 3):
        assert update(continuous, kw, task) == update(resumed, kw, task)
        assert_arrays_equal(continuous.get_all(), resumed.get_all())
        assert continuous.state_dict()["state_sha256"] == resumed.state_dict()["state_sha256"]


@pytest.mark.parametrize("change", [{"member_id": 1}, {"total_memory_budget": 6}, {"base_quota": 9},
                                    {"experiment_seed": 1}, {"feature_columns": ["y", "x"]}])
def test_resume_incompatible_config(data, change):
    kw = settings(data)
    buffer = replay.FoldSqrtReplayBuffer(**kw)
    update(buffer, kw, 0)
    kw.update(change)
    with pytest.raises(ValueError, match="metadata mismatch"):
        replay.FoldSqrtReplayBuffer.from_state_dict(buffer.state_dict(), **kw)


def test_resume_corruption_and_capacity_guards(data):
    kw = settings(data)
    buffer = replay.FoldSqrtReplayBuffer(**kw)
    update(buffer, kw, 0)
    snapshot = buffer.state_dict()
    snapshot["entries"][10]["features"][0, 0] += 100
    with pytest.raises(ValueError, match="SHA256"):
        replay.FoldSqrtReplayBuffer.from_state_dict(snapshot, **kw)
    snapshot = buffer.state_dict()
    snapshot["observed_counts"][10] = 999
    snapshot["state_sha256"] = replay._state_digest(snapshot)
    with pytest.raises(ValueError, match="observed"):
        replay.FoldSqrtReplayBuffer.from_state_dict(snapshot, **kw)


def test_restore_does_not_rerank_or_read_discarded_features(data, monkeypatch):
    kw = settings(data)
    buffer = replay.FoldSqrtReplayBuffer(**kw)
    update(buffer, kw, 0)
    update(buffer, kw, 1)
    snapshot = buffer.state_dict()
    def forbidden(*a, **kw):
        pytest.fail("Restore must not rerank or read old feature Parquet")
    monkeypatch.setattr(replay, "rank_raw_class_exemplars", forbidden)
    monkeypatch.setattr(pq, "read_table", forbidden)
    resumed = replay.FoldSqrtReplayBuffer.from_state_dict(snapshot, **kw)
    assert_arrays_equal(buffer.get_all(), resumed.get_all())
    snapshot["entries"][10]["features"][:] = -999
    assert np.all(resumed.get_all().features >= 0)


@pytest.mark.parametrize("target", ["task", "source"])
def test_source_or_task_mutation_blocks_update(data, target):
    kw = settings(data)
    buffer = replay.FoldSqrtReplayBuffer(**kw)
    rows = current(kw, 0)
    path = data[2] if target == "task" else data[0]["train"]
    with path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="changed"):
        buffer.update(rows, task_id=0, feature_columns=["x", "y"])
    assert buffer.next_task_id == 0 and buffer.total_size == 0


def test_global_budget_overlap_and_no_three_times_budget(data):
    assert replay.four_percent_member_budgets(694455) == {0: 9260, 1: 9259, 2: 9259}
    assert replay.four_percent_member_budgets(10) == {0: 0, 1: 0, 2: 0}
    buffers = []
    for member in range(3):
        kw = settings(data, member=member, budget=4)
        b = replay.FoldSqrtReplayBuffer(**kw)
        update(b, kw, 0)
        buffers.append(b)
    audit = replay.ensemble_buffer_accounting(buffers, global_budget=12)
    assert audit["stored_slots"] == 12 and audit["unique_sample_ids"] == 6
    assert audit["duplicate_stored_copies"] == 6
    assert audit["pairwise_overlap"] == {"0_1": 2, "0_2": 2, "1_2": 2}
    with pytest.raises(ValueError, match="global budget"):
        replay.ensemble_buffer_accounting(buffers, global_budget=4)
    with pytest.raises(ValueError, match="exactly members"):
        replay.ensemble_buffer_accounting([buffers[0]] * 3, global_budget=12)


def test_relocated_sources_resume_and_wrong_preprocessing(data, tmp_path):
    kw = settings(data)
    b = replay.FoldSqrtReplayBuffer(**kw)
    update(b, kw, 0)
    moved = tmp_path / "relocated"
    moved.mkdir()
    paths = {}
    for split, path in data[0].items():
        paths[split] = moved / path.name
        shutil.copyfile(path, paths[split])
    restored = replay.FoldSqrtReplayBuffer.from_state_dict(b.state_dict(), **settings((paths, data[1], data[2])))
    assert_arrays_equal(b.get_all(), restored.get_all())
    other = settings(data, member=1)
    artifact = prepare_fold_preprocessing(**{k: v for k, v in other.items() if k != "total_memory_budget"},
                                          validation_fold=1, policy="raw_identity")
    with pytest.raises(ValueError, match="preprocessing provenance"):
        b.model_arrays(artifact)
