"""Shared locked configuration, command construction and read-only run inspection."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import train_cil as engine
from src.training.fold_training import _digest, prepare_fold_run
from src.training.replay_checkpoint import load_fold_replay_checkpoint
from src.data.stratified_folds import fold_file_sha256

TRAINING = dict(method="class_balanced_fixed_budget_replay", fold_replay_policy="stratified_fraction_v1",
    optimizer="adamw", batch_size=64, effective_batch_size=1024, gradient_accumulation_steps=16,
    lr=0.001, weight_decay=0.0001, focal_gamma=1.0, loss_variant="focal_current_ce_replay")
MODEL = dict(variant="tddi_paper_member", input_dim=3780, hidden_dims=[7560, 7560],
             activation="gelu", dropout=0.2, norm="layernorm")
REPLAY = dict(buffer_policy="equal_class_fixed_budget", ranking_policy="sample_normalized_class_mean",
              fraction=0.125, repeat_cap=3, replay_starts_at_task=1)
INPUT_KEYS = {"train", "validation", "test", "feature_cols", "fold_assignments", "fold_manifest"}

def _default_runner(command, cwd: Path) -> None:
    """Run one member command synchronously; callers enforce sequential execution."""
    subprocess.run(list(command), cwd=cwd, check=True)

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
    args.update({k: v for k, v in config["model"].items()
                 if k in ("variant", "activation", "dropout", "norm")})
    if config.get("budget_policy") == "per_member_4_percent":
        args["fold_member_budget"] = config["member_budgets"][str(member)]
        args["fold_global_budget"] = config["global_slot_budget"]
    args["replay_fraction"] = config["replay"]["fraction"]
    args["replay_repeat_cap"] = config["replay"]["repeat_cap"]
    args["buffer_policy"] = config["replay"]["buffer_policy"]
    if config["training"]["method"] != engine.FIXED_BUDGET_METHOD:
        for key in ("exemplar_ranking_policy", "buffer_policy", "replay_fraction",
                    "replay_repeat_cap", "fold_replay_policy", "focal_gamma"):
            args.pop(key, None)
    common_task_cache_root = os.environ.get("CIL_COMMON_TASK_CACHE_ROOT")
    if common_task_cache_root:
        # Explicit opt-in keeps the cache namespace outside any method output.
        args["task_cache_root"] = str(Path(common_task_cache_root).resolve())
    command = [str(python), str(PROJECT_ROOT / "src/training/train_cil.py")]
    for key, value in args.items():
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    if config["execution"]["validation_only"]:
        command.append("--validation-only")
    prediction_splits = config["execution"].get("export_member_predictions", [])
    if prediction_splits:
        command.extend(("--export-member-predictions", "--member-prediction-splits"))
        command.extend(str(value) for value in prediction_splits)
    return command

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
            raise FileNotFoundError(f"Missing fold inputs: {missing}")
        return {"status": "unverified_missing_inputs", "missing_inputs": missing,
                "run_id": None, "completed_task_id": None, "resume_checkpoint": None}
    prepared = prepare_fold_run(args, engine=engine)
    if prepared["manifest"]["assignment_row_count"] != config["development_count"]:
        raise ValueError("Development count differs from config; never recompute budget from the task prefix.")
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
            run_config_sha256=state["run_config_sha256"], )
    return info
