"""Stable sample identity rules for DDI drug-pair data."""

from __future__ import annotations

from typing import Mapping

import numpy as np


DRUG_ID_A_COLUMN = "drugid-drug_a"
DRUG_ID_B_COLUMN = "drugid-drug_b"
SUPPORTED_PREDICTION_SPLITS = frozenset({"validation", "test"})


def build_stable_sample_ids(
    metadata: Mapping[str, np.ndarray],
    expected_rows: int,
) -> np.ndarray:
    """Build validated, order-preserving ``drug_a|drug_b`` identifiers."""

    missing = sorted({DRUG_ID_A_COLUMN, DRUG_ID_B_COLUMN} - set(metadata))
    if missing:
        raise ValueError(f"Missing DDI drug-pair metadata columns: {missing}")

    raw_drug_ids_a = np.asarray(metadata[DRUG_ID_A_COLUMN])
    raw_drug_ids_b = np.asarray(metadata[DRUG_ID_B_COLUMN])
    for column, values in (
        (DRUG_ID_A_COLUMN, raw_drug_ids_a),
        (DRUG_ID_B_COLUMN, raw_drug_ids_b),
    ):
        if values.dtype.kind not in {"O", "S", "U"}:
            raise TypeError(f"{column} must contain string identifiers.")
        if values.dtype.kind == "O" and not all(
            isinstance(value, (str, np.str_)) for value in values.tolist()
        ):
            raise TypeError(f"{column} contains a non-string or missing identifier.")

    drug_ids_a = raw_drug_ids_a.astype(np.str_, copy=False)
    drug_ids_b = raw_drug_ids_b.astype(np.str_, copy=False)
    if drug_ids_a.ndim != 1 or drug_ids_b.ndim != 1:
        raise ValueError("Drug ID metadata must be one-dimensional.")
    if drug_ids_a.shape[0] != expected_rows or drug_ids_b.shape[0] != expected_rows:
        raise ValueError("Drug ID metadata row count does not match prediction outputs.")
    if np.any(drug_ids_a == "") or np.any(drug_ids_b == ""):
        raise ValueError("Drug IDs must be non-empty.")

    sample_ids = np.char.add(np.char.add(drug_ids_a, "|"), drug_ids_b)
    unique_count = np.unique(sample_ids).shape[0]
    if unique_count != expected_rows:
        raise ValueError(
            "DDI sample IDs must be unique; "
            f"found {expected_rows - unique_count} duplicates."
        )
    return sample_ids
