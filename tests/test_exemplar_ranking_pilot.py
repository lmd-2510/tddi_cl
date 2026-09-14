"""Prompt 14 configs/orchestration/comparison; synthetic CPU data only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts import compare_exemplar_ranking_pilots as comparison
from src.data.fold_replay_buffer import (
    PIPELINE_INPUT_RANKING_POLICY,
    SAMPLE_NORMALIZED_RANKING_POLICY,
)
from src.data.stratified_folds import fold_file_sha256
from src.training import fold_ab_study as study, fold_pilot_training as pilot
from tests.test_fold_pilot_training import data, tiny


CONFIG_NAMES = (
    "smoke_tddi_p3_fold_ranking_sample_normalized_seed0.json",
    "smoke_tddi_p3_fold_ranking_pipeline_input_seed0.json",
)


def production_configs(phase="smoke", overrides=None):
    names = tuple(name.replace("smoke_", f"{phase}_") for name in CONFIG_NAMES)
    return [study.load_pilot_config(study.PROJECT_ROOT / "configs" / name, overrides=overrides)
            for name in names]


def test_ranking_configs_hold_preprocessing_and_only_change_space():
    pair = production_configs()
    study.validate_ab_group(pair)
    assert [config["case"] for config in pair] == ["sample_normalized", "pipeline_input"]
    assert all(config["kind"] == study.RANKING_KIND for config in pair)
    assert all(config["preprocessing"]["policy"] == "task0_standard_frozen" for config in pair)
    assert [config["replay"]["ranking_policy"] for config in pair] == [
        SAMPLE_NORMALIZED_RANKING_POLICY, PIPELINE_INPUT_RANKING_POLICY]
    plan = study.build_pilot_plan(pair[::-1], inspector=lambda *a, **k: {
        "status": "fresh", "run_id": None, "completed_task_id": None,
        "resume_checkpoint": None,
    })
    assert [(entry["case"], entry["member_id"]) for entry in plan["entries"]] == [
        ("sample_normalized", 0), ("pipeline_input", 0)]
    assert plan["comparison_axis"] == "ranking" and plan["kind"] == study.RANKING_KIND
    for entry, policy in zip(plan["entries"],
                             (SAMPLE_NORMALIZED_RANKING_POLICY, PIPELINE_INPUT_RANKING_POLICY), strict=True):
        args = study.engine.parse_args(entry["command"][2:])
        assert args.preprocessing_policy == "task0_standard_frozen"
        assert args.exemplar_ranking_policy == policy
        assert args.stop_after_task == 1 and args.validation_only


def test_ranking_pair_drift_and_alignment_rules():
    pair = production_configs()
    bad = deepcopy(pair)
    bad[1]["training"]["lr"] = 0.2
    with pytest.raises(ValueError, match="differ beyond"):
        study.validate_ab_group(bad)
    bad = deepcopy(pair)
    bad[1]["preprocessing"]["policy"] = "raw_identity"
    with pytest.raises(ValueError, match="differ beyond"):
        study.validate_ab_group(bad)

    def proof(retained, replay_hash):
        return {"0": {
            "alignment": {"current_ids": "same", "validation_ids": "same",
                          "seen_class_map": "same", "replay_ids_before": replay_hash,
                          "sampler": replay_hash},
            "retained_ids_sha256": retained, "retained_labels_sha256": "same-labels",
            "epochs": {"1": {"draw_order_ids_sha256": replay_hash}},
        }}
    entries = [
        {"kind": study.RANKING_KIND, "case": "sample_normalized", "member_id": 0,
         "order_proofs": proof("ids-a", "draw-a")},
        {"kind": study.RANKING_KIND, "case": "pipeline_input", "member_id": 0,
         "order_proofs": proof("ids-b", "draw-b")},
    ]
    # Exemplar and replay IDs are the independent variable and may differ.
    study.validate_ab_alignment(entries)
    entries[1]["order_proofs"]["0"]["alignment"]["validation_ids"] = "wrong"
    with pytest.raises(ValueError, match="validation_ids mismatch"):
        study.validate_ab_alignment(entries)


def synthetic_configs(dataset):
    prep_root = dataset["root"] / "ranking_preprocessing/member_0/B"
    shutil.copytree(dataset["preps"]["task0_standard_frozen"].parent, prep_root)
    overrides = {
        **dataset["paths"], "feature_cols": dataset["cols"],
        "fold_assignments": dataset["folds"] / "fold_assignments.parquet",
        "fold_manifest": dataset["folds"] / "fold_manifest.json",
        "device": "cpu",
    }
    result = []
    for source in production_configs():
        raw = json.loads(Path(source["config_path"]).read_text())
        raw.update(development_count=1068, global_slot_budget=42,
                   member_budgets={str(member): 14 for member in range(3)})
        raw["protocol"].update(task_file=str(dataset["tasks"]),
                               sha256=fold_file_sha256(dataset["tasks"]))
        raw["training"]["epochs"] = 2
        raw["preprocessing"]["artifact_template"] = str(
            dataset["root"] / "ranking_preprocessing/member_{member_id}/B/fold_preprocessing.json")
        raw["output_root"] = str(dataset["root"] / "ranking_runs" / raw["case"])
        path = dataset["root"] / f"ranking_{raw['case']}.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        result.append(study.load_pilot_config(path, overrides=overrides))
    return result


@pytest.fixture(scope="module")
def completed_ranking(tmp_path_factory):
    root = tmp_path_factory.mktemp("ranking_pilot")
    dataset = data.__wrapped__(root, SimpleNamespace())
    patcher = pytest.MonkeyPatch()
    small = tiny.__wrapped__(patcher)
    next(small)
    try:
        configs = synthetic_configs(dataset)
        for config in configs:
            args = study.engine.parse_args(study.member_command(config, 0)[2:])
            pilot.run_fold_training(args, engine=study.engine)
        return configs, dataset
    finally:
        next(small, None)
        patcher.undo()


def test_synthetic_training_checkpoint_provenance_and_report(completed_ranking, tmp_path):
    configs, dataset = completed_ranking
    roots = [Path(config["output_root"]) / "member_0" for config in configs]
    for root, expected in zip(roots,
                              (SAMPLE_NORMALIZED_RANKING_POLICY, PIPELINE_INPUT_RANKING_POLICY), strict=True):
        run_config = json.loads((root / "run_config.json").read_text())
        ranking = run_config["checkpoint_contract"]["buffer"]["ranking"]
        assert ranking["policy"] == expected
        if expected == PIPELINE_INPUT_RANKING_POLICY:
            assert ranking["preprocessing_sha256"] == run_config["checkpoint_contract"]["preprocessing_sha256"]
    report = comparison.compare(roots[0], roots[1], dataset["tasks"])
    assert report["alignment_status"] == "pass" and report["winner"] is None
    assert report["preprocessing_policy"] == "task0_standard_frozen"
    assert set(report["retained_id_overlap"]) == {"0", "1"}
    assert "pipeline_input" in comparison.markdown(report)

    out = tmp_path / "review"
    comparison.main([
        "--sample-normalized-run", str(roots[0]), "--pipeline-input-run", str(roots[1]),
        "--task-file", str(dataset["tasks"]), "--outdir", str(out), "--bundle",
    ])
    assert (out / "comparison.json").is_file() and (out / "ranking_review.tar.gz").is_file()


def test_pipeline_ranking_resume_with_wrong_policy_fails(completed_ranking, monkeypatch):
    configs, _ = completed_ranking
    monkeypatch.setattr(pilot, "TDDI_PAPER_INPUT_DIM", 2)
    pipeline_config = configs[1]
    args = study.engine.parse_args(study.member_command(pipeline_config, 0)[2:])
    prepared = pilot.prepare_fold_run(args, engine=study.engine)
    assert prepared["contract"]["buffer"]["ranking"]["policy"] == PIPELINE_INPUT_RANKING_POLICY
    args.exemplar_ranking_policy = SAMPLE_NORMALIZED_RANKING_POLICY
    other = pilot.prepare_fold_run(args, engine=study.engine)
    assert other["contract"] != prepared["contract"]
