from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import analyze_p3_task6_task7 as diagnostic


def test_aligned_boundary_diagnostic_counts_old_to_new_errors(tmp_path: Path, monkeypatch) -> None:
    classes = list(range(178))
    groups = [classes[:38]] + [classes[start:start + 20] for start in range(38, 178, 20)]
    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({
        "protocol": "tail_to_head",
        "tasks": [{"task_id": task_id, "classes": group} for task_id, group in enumerate(groups)],
    }), encoding="utf-8")

    def fake_prediction(root: Path, scope: str, task_id: int, split: str):
        labels = np.arange(158 if task_id == 6 else 178, dtype=np.int64)
        predictions = labels.copy()
        if task_id == 7:
            predictions[0] = 158
        return {
            "sample_ids": np.asarray([f"{split}-{raw}" for raw in labels]),
            "labels": labels,
            "predictions": predictions,
            "class_ids": np.arange(158 if task_id == 6 else 178, dtype=np.int64),
        }

    monkeypatch.setattr(diagnostic, "_load_prediction", fake_prediction)
    outputs = diagnostic.analyze(tmp_path / "run", task_file, tmp_path / "report")
    classwise = pd.read_csv(outputs["classwise"])
    confusion = pd.read_csv(outputs["cross_group"])

    class_zero = classwise[
        (classwise.scope == "ensemble")
        & (classwise.split == "validation")
        & (classwise.raw_class_id == 0)
    ].iloc[0]
    assert class_zero.task6_f1 == 1.0
    assert class_zero.task7_f1 == 0.0
    old_to_new = confusion[
        (confusion.scope == "ensemble")
        & (confusion.split == "validation")
        & (confusion.comparison == "old_true_predicted_as_task7_class")
    ].iloc[0]
    assert old_to_new.error_count == 1
    assert np.isclose(old_to_new.error_rate, 1 / 158)
    assert outputs["report"].is_file()
