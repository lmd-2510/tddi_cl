import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.source_shard_cache import (
    SOURCE_SHARD_KIND,
    SOURCE_SHARD_SCHEMA_VERSION,
    load_source_shard,
)


def _write_shard(root: Path, source: str, fold: int, values: list[float]) -> None:
    path = root / source / "task_0" / f"fold_{fold}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "f0": pa.array(values, type=pa.float64()),
        "f1": pa.array([1.0] * len(values), type=pa.float64()),
        "class": pa.array([1] * len(values), type=pa.int64()),
        "sample_id": pa.array([f"{source}-{fold}-{i}" for i in range(len(values))]),
        "source_split": pa.array([source] * len(values)),
        "source_row_index": pa.array(list(range(len(values))), type=pa.int64()),
        "source_path": pa.array([f"{source}.parquet"] * len(values)),
        "fold_id": pa.array([fold] * len(values), type=pa.int16()),
        "drugid-drug_a": pa.array([f"a{i}" for i in range(len(values))]),
        "drugid-drug_b": pa.array([f"b{i}" for i in range(len(values))]),
    }), path)


def test_load_source_shard_selects_member_role(tmp_path: Path) -> None:
    _write_shard(tmp_path, "train", 0, [0.0])
    _write_shard(tmp_path, "train", 1, [1.0])
    _write_shard(tmp_path, "validation", 0, [2.0])
    _write_shard(tmp_path, "validation", 1, [3.0])
    (tmp_path / "manifest.json").write_text(json.dumps({
        "kind": SOURCE_SHARD_KIND,
        "schema_version": SOURCE_SHARD_SCHEMA_VERSION,
        "fold_count": 2,
    }), encoding="utf-8")

    validation = load_source_shard(
        tmp_path, member_id=0, task_id=0, validation_fold=0,
        role="validation", feature_columns=["f0", "f1"],
    )
    train = load_source_shard(
        tmp_path, member_id=0, task_id=0, validation_fold=0,
        role="train", feature_columns=["f0", "f1"],
    )
    np.testing.assert_allclose(validation.features[:, 0], [0.0, 2.0])
    np.testing.assert_allclose(train.features[:, 0], [1.0, 3.0])
