from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_single_model_data import main as prepare_main
from src.data.original_split import prepare_original_split_context
from src.data.single_preprocessing import prepare_single_preprocessing
from src.data.single_task_cache import ensure_single_task_cache


def _sources(tmp_path: Path, task_file: Path, columns: list[str]):
    spec = json.loads(task_file.read_text())
    classes = [raw for task in spec["tasks"] for raw in task["classes"]]
    paths = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        data = {name: np.asarray([raw * .01 + index + split_index for raw in classes], dtype=np.float32)
                for index, name in enumerate(columns)}
        data.update({
            "class": np.asarray(classes, dtype=np.int64),
            "drugid-drug_a": [f"{split}_a_{index}" for index in range(len(classes))],
            "drugid-drug_b": [f"{split}_b_{index}" for index in range(len(classes))],
        })
        path = tmp_path / f"{split}.parquet"
        pq.write_table(pa.table(data), path)
        paths[split] = path
    return paths, spec


def test_original_split_preprocessing_and_cache_are_task_scoped(tmp_path):
    task_file = Path("study_assets/task_protocols/head_to_tail_tasks.json")
    columns = ["x0", "x1"]
    paths, spec = _sources(tmp_path, task_file, columns)
    context = prepare_original_split_context(**paths)
    prep = prepare_single_preprocessing(
        context, task_file=task_file, feature_columns=columns,
        policy="task0_standard_frozen",
    )
    assert prep.metadata["contract"]["fit_scope"]["split"] == "train"
    assert prep.metadata["fit_row_count"] == 38
    cache = tmp_path / "cache"
    loaded, _ = ensure_single_task_cache(
        cache, context=context, task_file_sha256="task-hash",
        preprocessing_sha256="prep-hash", feature_columns=columns,
        preprocessing=prep, task_id=0, class_ids=spec["tasks"][0]["classes"],
    )
    assert len(loaded["train"].labels) == 38
    assert not list(cache.rglob("task_1"))


def test_original_split_overlap_is_rejected(tmp_path):
    task_file = Path("study_assets/task_protocols/head_to_tail_tasks.json")
    paths, _ = _sources(tmp_path, task_file, ["x"])
    paths["validation"].write_bytes(paths["train"].read_bytes())
    with pytest.raises(ValueError, match="leakage"):
        prepare_original_split_context(**paths)


def test_preprocessing_check_never_starts_training(capsys):
    prepare_main(["--config", "configs/ordered_preprocessing.json", "--check"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_training"] is False
    assert payload["through_task"] == 7
