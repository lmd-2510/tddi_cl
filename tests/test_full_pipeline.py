"""Tiny CPU trajectory through all eight tasks, three members and final report."""
import hashlib
import json
import sys
import pytest
import numpy as np

from tests.test_fold_training import data, tiny, cli
from src.training import fold_training, train_cil as engine
from src.data.fold_preprocessing import prepare_fold_preprocessing, save_fold_preprocessing
from src.eval.predictions import load_member_prediction_artifact
from src.eval.ensemble_ue import aggregate_member_predictions, export_offline_ensemble_artifact
from src.eval.threshold import (
    load_threshold_selection_config, select_confidence_threshold,
    export_frozen_threshold_artifact, load_frozen_threshold_artifact,
    evaluate_with_frozen_threshold, export_threshold_report,
)
from src.eval.report import build_final_report


@pytest.mark.parametrize("data", ["full_test"], indirect=True)
@pytest.mark.parametrize("protocol", ["head_to_tail", "shuffled"])
def test_eight_task_ensemble_threshold_and_auto_protocol_report(data, tiny, monkeypatch, protocol):
    if protocol == "shuffled":
        spec = json.loads(data["tasks"].read_text())
        classes = [c for t in spec["tasks"] for c in t["classes"]]
        classes = np.random.default_rng(0).permutation(classes).tolist()
        offset = 0
        for task in spec["tasks"]:
            width = len(task["classes"])
            task["classes"] = classes[offset:offset + width]
            offset += width
        spec.update(protocol="shuffled", seed=0)
        data["tasks"].write_text(json.dumps(spec))
    monkeypatch.setattr(fold_training, "DEFAULT_META_COLS", ["drugid-drug_a", "drugid-drug_b"])
    root = data["root"] / "ensemble"
    for member in range(3):
        prep = prepare_fold_preprocessing(
            data["context"], task_file=data["tasks"], feature_columns=["x", "y"],
            member_id=member, validation_fold=member, policy="raw_identity")
        path = save_fold_preprocessing(prep, data["root"] / f"prep_member_{member}")
        command = cli(data, extra=[
            "--member-id", str(member), "--fold-preprocessing", str(path),
            "--outdir", str(root / f"member_{member}"), "--stop-after-task", "7",
            "--epochs", "1", "--export-member-predictions",
        ])
        command.remove("--validation-only")
        monkeypatch.setattr(sys, "argv", command)
        engine.main()
    config = load_threshold_selection_config("configs/eval_macro_f1_entropy.json")
    for task in range(8):
        output = root / "offline_evaluation" / f"task_{task}"
        ensembles = {}
        for split in ("validation", "test"):
            paths = [root / f"member_{m}/member_predictions/task_{task}/{split}.npz" for m in range(3)]
            artifact = aggregate_member_predictions(
                [load_member_prediction_artifact(p) for p in paths],
                source_artifact_paths=paths, ensemble_mode="stratified_3fold")
            target = output / ("oof.npz" if split == "validation" else "test.npz")
            export_offline_ensemble_artifact(artifact, target)
            ensembles[split] = artifact
        selected = select_confidence_threshold(
            ensembles["validation"], config, source_ensemble_path=output / "oof.npz")
        frozen_path = export_frozen_threshold_artifact(selected, output / "frozen_threshold.json")
        frozen = load_frozen_threshold_artifact(frozen_path, config_path=config.source_path)
        export_threshold_report(
            evaluate_with_frozen_threshold(ensembles["test"], frozen), output / "test_threshold_report.json")
    spec = json.loads(data["tasks"].read_text())
    (root / "full_manifest.json").write_text(json.dumps({"protocol": {
        "name": spec["protocol"], "tasks": spec["tasks"],
        "sha256": hashlib.sha256(data["tasks"].read_bytes()).hexdigest(),
    }}))
    paths = build_final_report(full_root=root)
    label = "ORDERED" if protocol == "head_to_tail" else "SHUFFLED"
    assert f"CB-Replay {label} Final Report" in paths["report"].read_text(encoding="utf-8")
