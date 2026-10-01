#!/usr/bin/env python3
"""Convert a validated task-0 fold-preprocessing JSON to legacy scaler payload.

The legacy ``train_cil.py`` EWC engine accepts a pickle payload, while the P4
frozen-fold pipeline stores the same task-0 statistics in a provenance-rich
JSON artifact.  This adapter copies only the frozen mean/scale statistics; it
does not fit, impute, clip, or read any future-task rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.input.read_bytes()
    payload = json.loads(source_bytes)
    if payload.get("policy") != "task0_standard_frozen":
        raise ValueError("Expected a task0_standard_frozen preprocessing artifact.")
    stats = payload.get("statistics")
    if not isinstance(stats, dict) or not stats.get("mean") or not stats.get("scale"):
        raise ValueError("Preprocessing artifact has no frozen mean/scale statistics.")
    mean = list(stats["mean"])
    scale = list(stats["scale"])
    if len(mean) != len(scale) or not all(float(x) == float(x) for x in mean + scale):
        raise ValueError("Invalid non-finite or mismatched preprocessing statistics.")
    if any(float(x) <= 0.0 for x in scale):
        raise ValueError("Frozen scales must be strictly positive.")

    converted = {
        "scaler_type": "standard",
        "impute_strategy": "error",
        "rows_fitted": int(payload["provenance"]["reference_selection"]["row_count"]),
        "mean": mean,
        "var": list(stats.get("variance", [float(x) ** 2 for x in scale])),
        "scale": scale,
        # The fold-preprocessing policy rejects nonfinite descriptors.  These
        # values are present only for the legacy payload schema and are never
        # used on finite inputs.
        "impute_values": [0.0] * len(mean),
        "source_fold_preprocessing_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {args.output} ({len(mean)} features)")


if __name__ == "__main__":
    main()
