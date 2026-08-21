"""Backbone-specific input loading while preserving shared split/task rows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.data.ddi_dataset import DDIBatchArrays, load_split_frame, transform_features
from src.data.molecular_graphs import MolecularGraphBank


DRUG_ID_A = "drugid-drug_a"
DRUG_ID_B = "drugid-drug_b"


def load_ddi_gcn_split_arrays(
    parquet_path: str | Path,
    graph_bank: MolecularGraphBank,
    *,
    class_ids: Iterable[int] | None = None,
    max_rows: int | None = None,
    ranking_feature_columns: list[str] | None = None,
    scaler_payload: dict[str, object] | None = None,
) -> tuple[DDIBatchArrays, np.ndarray | None]:
    feature_columns = ranking_feature_columns or []
    frame = load_split_frame(
        parquet_path,
        feature_columns,
        class_ids=class_ids,
        include_metadata=True,
        meta_cols=[DRUG_ID_A, DRUG_ID_B],
        max_rows=max_rows,
    )
    try:
        left = np.asarray(
            [graph_bank.drug_id_to_index[str(value)] for value in frame[DRUG_ID_A]],
            dtype=np.float32,
        )
        right = np.asarray(
            [graph_bank.drug_id_to_index[str(value)] for value in frame[DRUG_ID_B]],
            dtype=np.float32,
        )
    except KeyError as error:
        raise ValueError(f"Drug ID is missing from the molecular graph cache: {error}") from error
    model_inputs = np.column_stack([left, right]).astype(np.float32, copy=False)
    labels = frame["class"].to_numpy(dtype=np.int64, copy=False)
    ranking_features = None
    if feature_columns:
        if scaler_payload is None:
            raise ValueError("Descriptor ranking features require a scaler payload.")
        ranking_features = transform_features(
            frame[feature_columns].to_numpy(dtype=np.float64, copy=False),
            scaler_payload,
        )
    arrays = DDIBatchArrays(
        features=model_inputs,
        labels=labels,
        metadata={DRUG_ID_A: left, DRUG_ID_B: right},
    )
    return arrays, ranking_features
