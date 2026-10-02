from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.prepared_cache import (
    CACHE_MANIFEST,
    CACHE_SCHEMA_VERSION,
    feature_columns_sha256,
    materialize_split,
    sha256_file,
    validate_cache_manifest,
)


def _write_source(path, *, offset: int) -> None:
    table = pa.table(
        {
            "f0": np.asarray([1.0 + offset, 2.0 + offset], dtype=np.float64),
            "f1": np.asarray([3.0 + offset, 4.0 + offset], dtype=np.float64),
            "class": np.asarray([10, 11], dtype=np.int64),
            "drugid-drug_a": [f"a{offset}", f"a{offset + 1}"],
            "drugid-drug_b": [f"b{offset}", f"b{offset + 1}"],
        }
    )
    pq.write_table(table, path)


def test_materialize_and_validate_prepared_cache(tmp_path):
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    _write_source(train, offset=0)
    _write_source(validation, offset=10)
    scaler_path = tmp_path / "scaler.pkl"
    import pickle

    scaler = {
        "scaler_type": "standard",
        "mean": [1.0, 3.0],
        "scale": [1.0, 1.0],
        "impute_values": [0.0, 0.0],
    }
    scaler_path.write_bytes(pickle.dumps(scaler))
    root = tmp_path / "cache"
    materialize_split(train, root / "train.parquet", feature_columns=["f0", "f1"], scaler=scaler, split="train")
    materialize_split(validation, root / "validation.parquet", feature_columns=["f0", "f1"], scaler=scaler, split="validation")
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "ddi_cil_prepared_feature_cache",
        "policy": "task0_standard_frozen",
        "feature_columns_sha256": feature_columns_sha256(["f0", "f1"]),
        "scaler_sha256": sha256_file(scaler_path),
        "splits": {
            split: {
                "source_sha256": sha256_file(source),
                "sha256": sha256_file(root / f"{split}.parquet"),
            }
            for split, source in (("train", train), ("validation", validation))
        },
    }
    root.mkdir(exist_ok=True)
    (root / CACHE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    validate_cache_manifest(
        root,
        feature_columns=["f0", "f1"],
        scaler_path=scaler_path,
        source_paths={"train": train, "validation": validation},
    )
    cached = pq.read_table(root / "train.parquet")
    assert cached["f0"].to_numpy().tolist() == [0.0, 1.0]
    assert cached["f1"].to_numpy().tolist() == [0.0, 1.0]
