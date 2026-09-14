"""Prompt 11: read-only planning/fake dispatch, plus tiny CPU checkpoint proof."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from src.data.stratified_folds import fold_file_sha256
from src.training import fold_ab_study as study, fold_pilot_training as pilot
from tests.test_fold_pilot_training import data, tiny


def configs(phase="smoke", overrides=None):
    return [study.load_pilot_config(study.PROJECT_ROOT / "configs" / f"{phase}_tddi_p3_fold_ab_{name}_seed0.json",
                                    overrides=overrides)
            for name in ("A_raw", "B_task0_scaler")]


def fresh(config, member, command, **kwargs):
    return dict(status="fresh", run_id=None, completed_task_id=None, resume_checkpoint=None)


@pytest.mark.parametrize("phase,epochs", [("smoke", 3), ("pilot", 20)])
def test_config_controls_full_budget_and_member_command(phase, epochs):
    pair = configs(phase)
    study.validate_ab_group(pair)
    plan = study.build_pilot_plan(pair, inspector=fresh)
    assert [(e["case"], e["member_id"]) for e in plan["entries"]] == [("A", 0), ("B", 0)]
    assert plan["scope"] == dict(tasks=[0, 1], full_protocol_tasks=8, validation_only=True)
    assert not plan["offline_ensemble"] and not plan["threshold_selection"]
    for config, entry in zip(pair, plan["entries"]):
        args = study.engine.parse_args(entry["command"][2:])
        assert args.epochs == epochs and args.patience == 5
        assert args.member_id == 0 and args.seed == 0 and args.fold_seed == 42
        assert args.stop_after_task == 1 and args.validation_only
        assert args.fold_replay_policy == "stratified_fraction_v1"
        assert args.batch_size == 64 and args.effective_batch_size == 1024
        assert config["training"]["gradient_accumulation_steps"] == 16
        assert config["development_count"] == 694455 and config["global_slot_budget"] == 27778
        assert config["member_budgets"] == {"0": 9260, "1": 9259, "2": 9259}
        assert "--total-memory-budget" not in entry["command"] and "--scaler" not in entry["command"]
        assert not args.export_member_predictions
        assert args.preprocessing_policy == study.POLICIES[config["case"]]
    assert plan["entries"][0]["seeds"] == plan["entries"][1]["seeds"]


@pytest.mark.parametrize("field,value", [
    (("global_slot_budget",), 6800), (("development_count",), 10000),
    (("training", "lr"), 0.01), (("training", "epochs"), 20),
    (("training", "epochs"), 3.0), (("training", "method"), "ewc"),
    (("execution", "stop_after_task"), 7), (("execution", "validation_only"), False),
    (("preprocessing", "policy"), "task0_standard_frozen"),
    (("preprocessing", "artifact_template"), "{member_id}/{unknown}.json"),
    (("protocol", "sha256"), "0" * 64), (("schema_version",), True),
])
def test_bad_config_rejected(tmp_path, field, value):
    original = study.PROJECT_ROOT / "configs/smoke_tddi_p3_fold_ab_A_raw_seed0.json"
    raw = json.loads(original.read_text())
    target = raw
    for key in field[:-1]: target = target[key]
    target[field[-1]] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        study.load_pilot_config(path)


def test_ab_shared_controls_and_outputs_cannot_drift():
    pair = configs()
    for section, key, value in [("inputs", "train", "other.parquet"), ("training", "lr", 0.2),
                                ("replay", "ranking_policy", "different")]:
        bad = deepcopy(pair)
        bad[1][section][key] = value
        with pytest.raises(ValueError, match="differ beyond"):
            study.validate_ab_group(bad)
    bad = deepcopy(pair)
    bad[1]["output_root"] = str(Path(bad[0]["output_root"]) / "nested")
    with pytest.raises(ValueError, match="namespaces"):
        study.validate_ab_group(bad)
    with pytest.raises(ValueError, match="differ beyond"):
        study.validate_ab_group([pair[0], configs("pilot")[1]])


def test_server_overrides_and_task_hash(tmp_path):
    original = Path(configs()[0]["protocol"]["task_file"])
    moved = tmp_path / "moved.json"
    shutil.copy2(original, moved)
    overrides = {key: tmp_path / f"{key}.parquet" for key in study.INPUT_KEYS}
    overrides.update(task_file=moved, preprocessing_root=tmp_path / "prep", output_root=tmp_path / "runs", device="cpu")
    pair = configs(overrides=overrides)
    study.validate_ab_group(pair)
    for config in pair:
        assert config["device"] == "cpu" and config["protocol"]["task_file"] == str(moved)
        assert Path(config["output_root"]) == tmp_path / "runs/smoke" / config["case"]
        assert Path(config["preprocessing"]["artifact_template"].format(member_id=2)) == tmp_path / "prep/member_2" / config["case"] / "fold_preprocessing.json"
    moved.write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        configs(overrides=overrides)


def test_cli_dry_run_has_no_model_or_writes(tmp_path, monkeypatch, capsys):
    def forbidden(*a, **kw): pytest.fail("Dry-run must never train or construct a model")
    monkeypatch.setattr(study, "execute_pilot", forbidden)
    monkeypatch.setattr(study.engine, "expand_model_for_seen_classes", forbidden)
    paths = [c["config_path"] for c in configs()]
    study.main(["--config", *paths, "--train", str(tmp_path / "missing.parquet"),
                "--output-root", str(tmp_path / "runs"), "--member-id", "2"])
    output = capsys.readouterr().out
    assert "DRY RUN" in output and "UNVERIFIED" in output and "member=2" in output
    output.encode("cp1252")  # Windows redirected console must not fail on a decorative arrow.
    assert not (tmp_path / "runs").exists()


def test_missing_inputs_and_empty_existing_output_fail_without_overwrite(tmp_path):
    pair = configs(overrides={"output_root": tmp_path / "runs", "train": tmp_path / "missing"})
    with pytest.raises(FileNotFoundError, match="Missing pilot inputs"):
        study.execute_pilot(pair)
    assert not (tmp_path / "runs").exists()
    root = Path(pair[0]["output_root"]) / "member_0"
    root.mkdir(parents=True)
    sentinel = root / "user.txt"
    sentinel.write_text("keep")
    with pytest.raises(RuntimeError, match="Incomplete output"):
        study.build_pilot_plan(pair)
    assert sentinel.read_text() == "keep"


class FakeJobs:
    def __init__(self):
        self.states, self.calls, self.busy = {}, [], False

    def inspect(self, config, member, command, **kwargs):
        status = self.states.get((config["case"], member), "fresh")
        root = str(Path(config["output_root"]) / f"member_{member}")
        return dict(status=status, run_id=None if status == "fresh" else f"run_{config['case']}_{member}",
                    completed_task_id=1 if status == "complete" else 0 if status == "resume" else None,
                    full_trajectory_complete=False, run_config=root + "/run_config.json",
                    resume_checkpoint=root + "/checkpoints/task_0.pt" if status == "resume" else None)

    def run(self, command, cwd):
        assert not self.busy and cwd == study.PROJECT_ROOT
        self.busy = True
        args = study.engine.parse_args(command[2:])
        case = args.outdir.parent.name
        if self.states.get((case, args.member_id)) == "resume":
            assert args.resume_fold_checkpoint == args.outdir / "checkpoints/task_0.pt"
        else:
            assert args.resume_fold_checkpoint is None
        self.calls.append((case, args.member_id))
        self.states[(case, args.member_id)] = "complete"
        self.busy = False


@pytest.mark.parametrize("selected,expected", [(None, [("A", 0), ("B", 0)]),
    ([2], [("A", 2), ("B", 2)]), ([2, 0, 1], [(c, m) for c in ("A", "B") for m in range(3)])])
def test_blocking_sequential_selected_member_and_manifest(tmp_path, selected, expected):
    pair = configs(overrides={"output_root": tmp_path / "runs"})
    jobs = FakeJobs()
    path = study.execute_pilot(pair[::-1], member_ids=selected, runner=jobs.run, inspector=jobs.inspect)
    assert jobs.calls == expected
    manifest = json.loads(path.read_text())
    assert all(e["status"] == "complete" and e["run_id"] and not e["full_trajectory_complete"] for e in manifest["entries"])
    assert all(e["seeds"]["experiment_seed"] == 0 and e["task_file_sha256"] for e in manifest["entries"])
    assert not manifest["offline_ensemble"] and not manifest["threshold_selection"]
    saved = path.read_bytes()
    jobs.calls.clear()
    next_path = study.execute_pilot(pair, member_ids=selected, runner=jobs.run, inspector=jobs.inspect)
    assert not jobs.calls and path != next_path and path.read_bytes() == saved
    assert all(e["action"] == "skipped_verified_complete" for e in json.loads(next_path.read_text())["entries"])


def test_skip_complete_and_resume_incomplete(tmp_path):
    pair = configs(overrides={"output_root": tmp_path})
    jobs = FakeJobs()
    jobs.states.update({("A", 0): "complete", ("B", 0): "resume"})
    result = study.execute_pilot(pair, runner=jobs.run, inspector=jobs.inspect)
    assert jobs.calls == [("B", 0)]
    assert "--resume-fold-checkpoint" in json.loads(result.read_text())["entries"][1]["command"]


def test_child_missing_boundary_fails_and_stops_next_case(tmp_path):
    pair = configs(overrides={"output_root": tmp_path})
    calls = []
    with pytest.raises(RuntimeError, match="without a verified"):
        study.execute_pilot(pair, runner=lambda command, cwd: calls.append(command), inspector=fresh)
    assert len(calls) == 1
    failed = list(tmp_path.rglob("failed_0.json"))
    assert len(failed) == 1 and "error" in json.loads(failed[0].read_text())["entries"][0]


def test_alignment_guard_and_different_early_stopping():
    proof = {"0": {"alignment": {"sampler": "same"}, "retained_ids_sha256": "same",
                    "epochs": {"1": {"draw_order_ids_sha256": "same"}}}}
    entries = [dict(case=c, member_id=0, order_proofs=deepcopy(proof)) for c in ("A", "B")]
    entries[1]["order_proofs"]["0"]["epochs"]["2"] = {"draw_order_ids_sha256": "later"}
    study.validate_ab_alignment(entries)
    entries[1]["order_proofs"]["0"]["epochs"]["1"]["draw_order_ids_sha256"] = "different"
    with pytest.raises(ValueError, match="sampler order"):
        study.validate_ab_alignment(entries)
    entries[1]["order_proofs"]["0"]["retained_ids_sha256"] = "different"
    with pytest.raises(ValueError, match="exemplar alignment"):
        study.validate_ab_alignment(entries)


def synthetic_configs(data):
    overrides = {**data["paths"], "feature_cols": data["cols"], "fold_assignments": data["folds"] / "fold_assignments.parquet",
                 "fold_manifest": data["folds"] / "fold_manifest.json", "device": "cpu", "output_root": data["root"] / "runs"}
    result = []
    for base in configs():
        raw = json.loads(Path(base["config_path"]).read_text())
        raw.update(development_count=1068, global_slot_budget=42, member_budgets={str(m): 14 for m in range(3)})
        raw["protocol"].update(task_file=str(data["tasks"]), sha256=fold_file_sha256(data["tasks"]))
        raw["training"]["epochs"] = 2
        folder = data["root"] / "prep/member_0" / raw["case"]
        # Preserve sidecars too: B contains an immutable scaler archive.
        shutil.copytree(data["preps"][raw["preprocessing"]["policy"]].parent, folder)
        raw["preprocessing"]["artifact_template"] = str(data["root"] / "prep/member_{member_id}" / raw["case"] / "fold_preprocessing.json")
        path = data["root"] / f"{raw['case']}.json"
        path.write_text(json.dumps(raw))
        result.append(study.load_pilot_config(path, overrides=overrides))
    return result


def test_real_preflight_tiny_resume_and_ab_order_proofs(data, tiny):
    pair = synthetic_configs(data)
    # Start A with task0 only, then orchestrator must resume A and start B.
    command = study.member_command(pair[0], 0)
    command[command.index("--stop-after-task") + 1] = "0"
    pilot.run_fold_training(study.engine.parse_args(command[2:]), engine=study.engine)
    root = Path(pair[0]["output_root"]) / "member_0"
    old = {p: p.read_bytes() for p in (root / "task_0").rglob("*") if p.is_file()}
    plan = study.build_pilot_plan(pair, require_inputs=True)
    assert [e["status"] for e in plan["entries"]] == ["resume", "fresh"]
    def run(command, cwd):
        pilot.run_fold_training(study.engine.parse_args(command[2:]), engine=study.engine)
    path = study.execute_pilot(pair, runner=run)
    manifest = json.loads(path.read_text())
    assert [e["completed_task_id"] for e in manifest["entries"]] == [1, 1]
    assert all(e["order_proofs"]["1"]["epochs"]["1"]["draw_order_ids_sha256"] for e in manifest["entries"])
    study.validate_ab_alignment(manifest["entries"])
    assert all(p.read_bytes() == content for p, content in old.items())
    assert all(not (Path(e["outdir"]) / "task_2").exists() for e in manifest["entries"])
    # No GPU models in plan/skip: factory call count stays unchanged.
    calls = list(tiny)
    study.execute_pilot(pair, runner=lambda *a: pytest.fail("Complete pilot must skip"))
    assert tiny == calls
    (root / "task_0/metrics.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        study.build_pilot_plan(pair)


def test_legacy_cli_budget_behavior_unchanged():
    required = [item for key in ("train", "validation", "test", "feature-cols", "task-file", "outdir", "scaler")
                for item in ("--" + key, "unused")]
    args = study.engine.parse_args([*required, "--seed", "0"])
    assert args.total_memory_budget == args.replay_draws_per_epoch == 6800
    assert args.fold_replay_policy is None and args.member_id is None


def test_shared_preflight_context_reuse_and_path_guard(data, tiny, monkeypatch):
    pair = synthetic_configs(data)
    args = study.engine.parse_args(study.member_command(pair[0], 0)[2:])
    original = pilot.prepare_fold_run(args, engine=study.engine)
    def forbidden(*a, **kw): pytest.fail("Validated context reuse must not re-read source data")
    monkeypatch.setattr(pilot, "prepare_development_fold_context", forbidden)
    reused = pilot.prepare_fold_run(args, engine=study.engine, context=original["context"])
    assert reused["contract"] == original["contract"]
    args.train = data["root"] / "other.parquet"
    with pytest.raises(ValueError, match="Cached fold context"):
        pilot.prepare_fold_run(args, engine=study.engine, context=original["context"])
    assert not tiny  # preflight, including scaler validation, constructs no model.
