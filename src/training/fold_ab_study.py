"""Plan/explicitly run frozen-fold smoke or preprocessing A/B pilots, never full8.

No model is constructed here. Training happens only in one blocking child process
at a time; checkpoint validation uses the same read-only preflight as the trainer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from string import Formatter
import subprocess
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fold_replay_buffer import (
    BUFFER_POLICY,
    PIPELINE_INPUT_RANKING_POLICY,
    RANKING_POLICY,
    SAMPLE_NORMALIZED_RANKING_POLICY,
    four_percent_member_budgets,
)
from src.data.stratified_folds import fold_file_sha256
from src.training import train_cil as engine
from src.training.fold_pilot_training import P3_LAYOUT, _digest, _publish_json, prepare_fold_run
from src.training.replay_checkpoint import load_fold_replay_checkpoint
from src.utils.seed import resolve_seed_configuration

KIND = "tddi_frozen_fold_preprocessing_pilot"
RANKING_KIND = "tddi_frozen_fold_exemplar_ranking_pilot"
POLICIES = {"A": "raw_identity", "B": "task0_standard_frozen"}
RANKING_CASES = {
    "sample_normalized": SAMPLE_NORMALIZED_RANKING_POLICY,
    "pipeline_input": PIPELINE_INPUT_RANKING_POLICY,
}
TRAINING = dict(method="replay_distill_fixed_budget_uniform", fold_replay_policy="stratified_fraction_v1",
    optimizer="adamw", batch_size=64, effective_batch_size=1024, gradient_accumulation_steps=16,
    lr=0.001, weight_decay=0.0001, focal_gamma=1.0, distill_alpha=1.0, temperature=2.0,
    feature_distill_weight=0.5, weight_alignment="none")
MODEL = dict(variant="tddi_paper_member", input_dim=3780, hidden_dims=[7560, 7560],
             activation="gelu", dropout=0.2, norm="layernorm")
REPLAY = dict(buffer_policy=BUFFER_POLICY, ranking_policy=RANKING_POLICY,
              base_quota=10, fraction=0.125, repeat_cap=3, replay_starts_at_task=1)
EXECUTION = dict(stop_after_task=1, validation_only=True, sequential=True,
                 checkpoint_resume_policy="immutable_frozen_fold_task_boundary_v1")
INPUT_KEYS = {"train", "validation", "test", "feature_cols", "fold_assignments", "fold_manifest"}


def _default_runner(command, cwd: Path) -> None:
    """Run one member command synchronously; callers enforce sequential execution."""
    subprocess.run(list(command), cwd=cwd, check=True)


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name}: required keys {sorted(expected)}; unknown/missing keys are not ignored.")


def _path(value, root):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty explicit path.")
    return str((root / value).resolve())


def _comparison_axis(config):
    return "preprocessing" if config.get("kind", KIND) == KIND else "ranking"


def _case_order(config):
    order = list(POLICIES) if _comparison_axis(config) == "preprocessing" else list(RANKING_CASES)
    return order.index(config["case"])


def _preprocessing_case(config):
    return config["case"] if _comparison_axis(config) == "preprocessing" else "B"


def load_pilot_config(path, *, project_root=PROJECT_ROOT, overrides=None):
    """Strict new schema. Never routes old configs through new budget validation."""
    path, project_root = Path(path).resolve(), Path(project_root).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    _keys(value, {"schema_version", "kind", "phase", "case", "experiment_seed", "fold_seed", "member_ids",
        "development_count", "global_slot_budget", "member_budgets", "protocol", "inputs", "preprocessing",
        "model", "training", "replay", "execution", "device", "output_root"}, "pilot config")
    for key in ("schema_version", "experiment_seed", "fold_seed", "development_count", "global_slot_budget"):
        if type(value[key]) is not int:
            raise ValueError(f"{key} must be an integer, not a boolean/float.")
    if (value["schema_version"] != 1 or value["kind"] not in (KIND, RANKING_KIND)
            or value["phase"] not in ("smoke", "pilot")):
        raise ValueError("Unsupported pilot kind/schema/phase/case.")
    cases = POLICIES if value["kind"] == KIND else RANKING_CASES
    if value["case"] not in cases:
        raise ValueError("Unsupported pilot case for the selected comparison kind.")
    if type(value["experiment_seed"]) is not int or value["experiment_seed"] != 0 or value["fold_seed"] != 42:
        raise ValueError("Keep experiment seed0/fold_seed42.")
    members = value["member_ids"]
    if (not isinstance(members, list) or not members or any(type(m) is not int or m not in (0, 1, 2) for m in members)
            or len(members) != len(set(members))):
        raise ValueError("member_ids must be unique members 0/1/2; defaults are [0].")
    budgets = {str(k): v for k, v in four_percent_member_budgets(value["development_count"]).items()}
    if value["member_budgets"] != budgets or value["global_slot_budget"] != sum(budgets.values()):
        raise ValueError("Budget must use floor(4% of FULL development count), split across three planned members.")
    _keys(value["training"], {*TRAINING, "epochs", "patience"}, "training")
    for key in ("epochs", "patience", "batch_size", "effective_batch_size", "gradient_accumulation_steps"):
        if type(value["training"][key]) is not int:
            raise ValueError(f"training.{key} must be an integer.")
    microbatch = value["training"]["batch_size"]
    if microbatch not in (8, 16, 32, 64):
        raise ValueError("microbatch must be 64, or explicit OOM fallback 32/16/8.")
    expected_training = {**TRAINING, "batch_size": microbatch, "gradient_accumulation_steps": 1024 // microbatch}
    if {k: value["training"][k] for k in TRAINING} != expected_training:
        raise ValueError("Keep the agreed training/loss/batch hyperparameters; no legacy 6800 budget flags.")
    if value["training"]["patience"] != 5 or (value["training"]["epochs"] not in (2, 3) if value["phase"] == "smoke"
                                               else value["training"]["epochs"] != 20):
        raise ValueError("Smoke needs 2–3 epochs, pilot20; both use patience5.")
    expected_replay = dict(REPLAY)
    if value["kind"] == RANKING_KIND:
        expected_replay["ranking_policy"] = RANKING_CASES[value["case"]]
    if value["model"] != MODEL or value["replay"] != expected_replay or value["execution"] != EXECUTION:
        raise ValueError("Model/replay/execution policy mismatch; this entrypoint stops at task1, validation-only.")
    _keys(value["inputs"], INPUT_KEYS, "inputs")
    _keys(value["preprocessing"], {"policy", "artifact_template"}, "preprocessing")
    expected_preprocessing = (POLICIES[value["case"]] if value["kind"] == KIND
                              else "task0_standard_frozen")
    if value["preprocessing"]["policy"] != expected_preprocessing:
        raise ValueError("Preprocessing policy does not match the controlled pilot case.")
    template = value["preprocessing"]["artifact_template"]
    if not isinstance(template, str) or "{member_id}" not in template:
        raise ValueError("Preprocessing template must explicitly contain {member_id}.")
    if any(field is not None and (field != "member_id" or spec or conversion)
           for _, field, spec, conversion in Formatter().parse(template)):
        raise ValueError("Only plain {member_id} is supported in preprocessing paths.")
    _keys(value["protocol"], {"id", "name", "task_file", "sha256", "layout", "class_count"}, "protocol")
    protocol = value["protocol"]
    if protocol["id"] != "P3" or protocol["name"] != "tail_to_head" or protocol["layout"] != P3_LAYOUT or protocol["class_count"] != 178:
        raise ValueError("Use full P3 8-task /178-class protocol, not the smoke task file.")
    overrides = overrides or {}
    for name in INPUT_KEYS:
        value["inputs"][name] = _path(str(overrides[name]) if overrides.get(name) else value["inputs"][name], project_root)
    protocol["task_file"] = _path(str(overrides["task_file"]) if overrides.get("task_file") else protocol["task_file"], project_root)
    if fold_file_sha256(protocol["task_file"]) != protocol["sha256"]:
        raise ValueError("P3 task-file SHA256 mismatch (path may move; contents may not).")
    tasks = engine.load_task_spec(Path(protocol["task_file"]))
    if (tasks.get("protocol") != "tail_to_head" or [t["task_id"] for t in tasks["tasks"]] != list(range(8))
            or [len(t["classes"]) for t in tasks["tasks"]] != P3_LAYOUT
            or len({c for t in tasks["tasks"] for c in t["classes"]}) != 178):
        raise ValueError("Task file does not contain the full P3 layout/raw class map.")
    if overrides.get("preprocessing_root"):
        template = str(Path(overrides["preprocessing_root"]) / "member_{member_id}" /
                       _preprocessing_case(value) / "fold_preprocessing.json")
    value["preprocessing"]["artifact_template"] = _path(template, project_root)
    for member in (0, 1, 2):
        try:
            value["preprocessing"]["artifact_template"].format(member_id=member)
        except (KeyError, ValueError) as error:
            raise ValueError("Only {member_id} is supported in preprocessing paths.") from error
    value["output_root"] = _path(str(Path(overrides["output_root"]) / value["phase"] / value["case"])
                                 if overrides.get("output_root") else value["output_root"], project_root)
    value["device"] = str(overrides.get("device") or value["device"])
    value["config_path"], value["config_sha256"] = str(path), fold_file_sha256(path)
    return value


def validate_ab_group(configs):
    if (not configs or len(configs) > 2 or len({c["case"] for c in configs}) != len(configs)
            or len({c["kind"] for c in configs}) != 1):
        raise ValueError("Supply one case or exactly one matched pair of a single comparison kind.")
    allowed = POLICIES if configs[0]["kind"] == KIND else RANKING_CASES
    if any(c["case"] not in allowed for c in configs):
        raise ValueError("Pilot cases do not match the comparison kind.")
    def control(config):
        shared = deepcopy(config)
        for key in ("case", "output_root", "config_path", "config_sha256"):
            shared.pop(key)
        if _comparison_axis(config) == "preprocessing":
            shared.pop("preprocessing")
        else:
            shared["replay"] = dict(shared["replay"])
            shared["replay"].pop("ranking_policy")
        return shared
    if any(control(c) != control(configs[0]) for c in configs[1:]):
        raise ValueError("Pilot cases differ beyond the declared comparison axis/output.")
    roots = [Path(c["output_root"]).resolve() for c in configs]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError("A/B output namespaces must be separate and non-nested.")


def member_command(config, member, *, python=sys.executable):
    args = {**config["inputs"], "task_file": config["protocol"]["task_file"],
        "fold_preprocessing": config["preprocessing"]["artifact_template"].format(member_id=member),
        "preprocessing_policy": config["preprocessing"]["policy"],
        "exemplar_ranking_policy": config["replay"]["ranking_policy"],
        "member_id": member, "seed": config["experiment_seed"], "fold_seed": config["fold_seed"],
        "device": config["device"],
        "stop_after_task": config["execution"]["stop_after_task"],
        "outdir": str(Path(config["output_root"]) / f"member_{member}")}
    args.update({k: v for k, v in config["training"].items() if k not in ("optimizer", "gradient_accumulation_steps")})
    args.update({k: v for k, v in config["model"].items() if k not in ("input_dim", "hidden_dims")})
    command = [str(python), str(PROJECT_ROOT / "src/training/train_cil.py")]
    for key, value in args.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if config["execution"]["validation_only"]:
        command.append("--validation-only")
    prediction_splits = config["execution"].get("export_member_predictions", [])
    if prediction_splits:
        command.extend(("--export-member-predictions", "--member-prediction-splits"))
        command.extend(str(value) for value in prediction_splits)
    return command


def _order_proofs(root, state):
    """Summarize already checkpoint-validated audits, not raw descriptors or weights."""
    proofs = {}
    for summary in state["progress"]["task_summaries"]:
        task_dir = root / summary["artifact_directory"]
        inputs = json.loads((task_dir / "input_audit.json").read_text(encoding="utf-8"))
        buffer = json.loads((task_dir / "buffer_audit.json").read_text(encoding="utf-8"))
        proofs[str(summary["task_id"])] = {
            "input_audit_path": str(task_dir / "input_audit.json"),
            "input_audit_sha256": state["artifact_hashes"][(task_dir / "input_audit.json").relative_to(root).as_posix()],
            "alignment": {key: _digest(inputs[key]) for key in
                          ("current_ids", "replay_ids_before", "validation_ids", "seen_class_map", "sampler")},
            "retained_ids_sha256": _digest(buffer["retained_ids"]),
            "retained_labels_sha256": _digest(buffer["retained_raw_labels"]),
            "epochs": {str(epoch): json.loads((task_dir / f"epoch_{epoch}_audit.json").read_text(encoding="utf-8"))["sampling"]
                       for epoch in range(1, summary["epochs_trained"] + 1)},
        }
    return proofs


def validate_ab_alignment(entries):
    """Only compare the same member and shared completed tasks/epochs (early stopping can differ)."""
    for member in {e["member_id"] for e in entries}:
        pair = {e["case"]: e for e in entries if e["member_id"] == member}
        kind = next((e.get("kind", KIND) for e in entries if e["member_id"] == member), KIND)
        expected = set(POLICIES if kind == KIND else RANKING_CASES)
        if set(pair) != expected:
            continue
        ordered = sorted((pair[c] for c in expected), key=lambda e: _case_order(e))
        left, right = [entry.get("order_proofs", {}) for entry in ordered]
        for task in left.keys() & right.keys():
            a, b = left[task], right[task]
            if kind == KIND:
                if a["alignment"] != b["alignment"] or a["retained_ids_sha256"] != b["retained_ids_sha256"]:
                    raise ValueError(f"A/B input/class/exemplar alignment mismatch: member={member} task={task}.")
                for epoch in a["epochs"].keys() & b["epochs"].keys():
                    if a["epochs"][epoch] != b["epochs"][epoch]:
                        raise ValueError(f"A/B sampler order mismatch: member={member} task={task} epoch={epoch}.")
            else:
                for key in ("current_ids", "validation_ids", "seen_class_map"):
                    if a["alignment"][key] != b["alignment"][key]:
                        raise ValueError(f"Ranking-pilot {key} mismatch: member={member} task={task}.")
                if a.get("retained_labels_sha256") != b.get("retained_labels_sha256"):
                    raise ValueError(f"Ranking-pilot allocation/retained-label mismatch: member={member} task={task}.")


def inspect_member(config, member, command, *, require_inputs=False):
    """No model allocation, no files written. Existing runs must pass full validation."""
    args = engine.parse_args(command[2:])
    root = Path(args.outdir)
    candidates = [root / "checkpoints" / f"task_{i}.pt" for i in range(8)]
    existing = [p for p in candidates if p.is_file()]
    if root.exists() and not existing:
        raise RuntimeError(f"Incomplete output without a valid frozen-fold checkpoint: {root}; never overwrite.")
    inputs = [getattr(args, k) for k in (*INPUT_KEYS, "task_file", "fold_preprocessing")]
    missing = sorted(str(p) for p in inputs if not p.is_file())
    if missing:
        if require_inputs or root.exists():
            raise FileNotFoundError(f"Missing pilot inputs: {missing}")
        return {"status": "unverified_missing_inputs", "missing_inputs": missing,
                "run_id": None, "completed_task_id": None, "resume_checkpoint": None}
    prepared = prepare_fold_run(args, engine=engine)
    if prepared["manifest"]["assignment_row_count"] != config["development_count"]:
        raise ValueError("Development count differs from config; never recompute budget from the task0–1 prefix.")
    contract = prepared["contract"]
    info = {"status": "fresh", "run_id": None, "completed_task_id": None, "resume_checkpoint": None,
            "contract_sha256": _digest(contract), "assignment_sha256": contract["assignment_sha256"],
            "fold_manifest_sha256": contract["fold_manifest_sha256"], "source_hashes": contract["sources"],
            "preprocessing_sha256": contract["preprocessing_sha256"], "buffer": contract["buffer"],
            "sampler": contract["sampler"], "task_file_sha256": contract["task_file_sha256"]}
    if existing:
        state, _ = load_fold_replay_checkpoint(existing[-1], root=root, expected_contract=contract,
            tasks=prepared["tasks"], context=prepared["context"], buffer_kwargs=prepared["buffer_kwargs"])
        complete = state["completed_task_id"] >= config["execution"]["stop_after_task"]
        info.update(status="complete" if complete else "resume", run_id=state["run_id"],
            completed_task_id=state["completed_task_id"], full_trajectory_complete=state["full_trajectory_complete"],
            resume_checkpoint=str(existing[-1]), checkpoint_sha256=fold_file_sha256(existing[-1]),
            task_summaries=state["progress"]["task_summaries"], run_config=str(root / "run_config.json"),
            run_config_sha256=state["run_config_sha256"], order_proofs=_order_proofs(root, state))
    return info


def build_pilot_plan(configs, *, member_ids=None, python=sys.executable, require_inputs=False, inspector=inspect_member):
    validate_ab_group(configs)
    selected = list(configs[0]["member_ids"] if member_ids is None else member_ids)
    if not selected or len(selected) != len(set(selected)) or any(type(m) is not int or m not in (0, 1, 2) for m in selected):
        raise ValueError("Select unique member IDs 0/1/2.")
    entries = []
    for config in sorted(configs, key=_case_order):
        for member in sorted(selected):
            command = member_command(config, member, python=python)
            info = inspector(config, member, command, require_inputs=require_inputs)
            if info["status"] == "resume":
                command += ["--resume-fold-checkpoint", info["resume_checkpoint"]]
            entries.append({"case": config["case"], "kind": config["kind"],
                "comparison_axis": _comparison_axis(config), "phase": config["phase"], "member_id": member,
                "seeds": asdict(resolve_seed_configuration(config["experiment_seed"], member)),
                "validation_fold": member, "planned_member_budgets": config["member_budgets"],
                "global_slot_budget": config["global_slot_budget"], "development_count": config["development_count"],
                "config_path": config["config_path"], "config_sha256": config["config_sha256"],
                "task_file_sha256": config["protocol"]["sha256"], "command": command,
                "outdir": str(Path(config["output_root"]) / f"member_{member}"), **info})
    validate_ab_alignment(entries)
    return {"kind": configs[0]["kind"], "schema_version": 1,
            "comparison_axis": _comparison_axis(configs[0]),
            "execution_policy": "blocking_sequential_cases_then_members",
            "scope": {"tasks": [0, 1], "full_protocol_tasks": 8, "validation_only": True},
            "offline_ensemble": False, "threshold_selection": False, "selected_members": sorted(selected),
            "configs": configs, "entries": entries}


def execute_pilot(configs, *, member_ids=None, python=sys.executable, runner=_default_runner, inspector=inspect_member):
    """Only explicitly called by --execute; fake runner/inspector seams are for tests."""
    plan = build_pilot_plan(configs, member_ids=member_ids, python=python, require_inputs=True, inspector=inspector)
    dispatch = uuid.uuid4().hex
    manifest_dir = Path(configs[0]["output_root"]).parent / "manifests" / dispatch
    def snapshot(stage):
        path = manifest_dir / f"{stage}.json"
        _publish_json(path, {**plan, "dispatch_id": dispatch, "stage": stage,
                             "created_at_utc": datetime.now(timezone.utc).isoformat()})
        return path
    snapshot("planned")
    by_case = {c["case"]: c for c in configs}
    for i, entry in enumerate(plan["entries"]):
        config = by_case[entry["case"]]
        try:
            # Recheck immediately before launch; never trust a stale dry-run plan.
            command = member_command(config, entry["member_id"], python=python)
            current = inspector(config, entry["member_id"], command, require_inputs=True)
            entry.update(current)
            if current["status"] == "complete":
                entry["action"] = "skipped_verified_complete"
            else:
                if current["status"] not in ("fresh", "resume"):
                    raise RuntimeError("Execution requires validated fresh/resume state.")
                if current["status"] == "resume":
                    command += ["--resume-fold-checkpoint", current["resume_checkpoint"]]
                entry["command"] = command
                runner(command, PROJECT_ROOT)  # synchronous; do NOT use Popen/background/GPU models here
                after = inspector(config, entry["member_id"], member_command(config, entry["member_id"], python=python), require_inputs=True)
                if after["status"] != "complete":
                    raise RuntimeError("Child exited without a verified task1 boundary and artifacts.")
                entry.update(after, action="executed")
            validate_ab_alignment(plan["entries"])
        except Exception as error:
            entry["error"] = str(error)
            snapshot(f"failed_{i}")
            raise
        snapshot(f"completed_{i}")
    return snapshot("complete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="+", required=True, help="One A/B config or a matched A+B pair of the same phase.")
    parser.add_argument("--execute", action="store_true", help="Omit for read-only dry-run. Never auto-runs full8/ensemble/threshold.")
    parser.add_argument("--member-id", type=int, choices=(0, 1, 2), action="append", dest="member_ids")
    parser.add_argument("--python", default=sys.executable)
    for key in sorted(INPUT_KEYS | {"task_file", "preprocessing_root", "output_root"}):
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    overrides = {k: getattr(args, k) for k in INPUT_KEYS | {"task_file", "preprocessing_root", "output_root", "device"}}
    configs = [load_pilot_config(p, overrides=overrides) for p in args.config]
    if args.execute:
        print(f"Completed manifest: {execute_pilot(configs, member_ids=args.member_ids, python=args.python)}", flush=True)
    else:
        plan = build_pilot_plan(configs, member_ids=args.member_ids, python=args.python)
        labels = "->".join(c["case"] for c in sorted(configs, key=_case_order))
        print(f"DRY RUN: no model/training/writes. Sequential cases {labels}; members in ascending order. Task0-1 only.")
        for entry in plan["entries"]:
            print(f"{entry['phase']} {entry['case']} member={entry['member_id']} seed={entry['seeds']['member_seed']} "
                  f"status={entry['status']} budget={entry['planned_member_budgets'][str(entry['member_id'])]}")
            if entry.get("missing_inputs"): print("UNVERIFIED inputs:", ", ".join(entry["missing_inputs"]))
            print(shlex.join(entry["command"]))
        print("No test evaluation, ensemble, threshold, or pilot winner selection is scheduled.")


if __name__ == "__main__":
    main()
