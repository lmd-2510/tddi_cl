"""Frozen-fold replay trainer with task-boundary resume.

The numerical training, expansion and evaluation kernels remain in train_cil.
This module composes the validated fold, preprocessing, replay and cache APIs.
It never fits preprocessing during training or rebuilds frozen fold assignments.
"""
from __future__ import annotations

from dataclasses import asdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.class_mapping import build_seen_class_map, invert_class_map, remap_labels
from src.data.ddi_dataset import (
    DEFAULT_META_COLS, DDIBatchArrays, load_development_fold_arrays, load_development_fold_identity,
    load_feature_columns, load_split_frame,
    prepare_development_fold_context,
)
from src.data.task_cache import (
    assignment_arrays_sha256,
    build_task_cache_contract,
    concatenate_task_arrays,
    load_task_cache_role,
    load_validation_trajectory,
    task_cache_lock,
    write_task_cache,
)
from src.data.source_shard_cache import load_source_shard, validate_source_shard_cache
from src.data.fold_preprocessing import load_fold_preprocessing
from src.data.class_balanced_replay_buffer import BUFFER_POLICY, RANKING_POLICY, four_percent_member_budgets
from src.data.replay_sampler import FoldReplayFractionSampler, RNG_DERIVATION, SAMPLER_POLICY
from src.data.stratified_folds import fold_file_sha256
from src.data.task_protocols import (
    PROTOCOL_IDS,
    TASK_LAYOUT,
    validate_full_task_spec,
)
from src.models.tddi_paper_member import TDDI_PAPER_INPUT_DIM, paper_member_manifest
from src.methods.ewc import compute_fisher, grow_head_state
from src.training.regularization_memory import create_memory
from src.eval.predictions import (
    MemberPredictionContext,
    PredictionProvenance,
    export_member_prediction_artifact,
    partition_identity_sha256,
)
from src.eval.metrics import compute_classification_metrics
from src.utils.logging import RunLogger
from src.utils.seed import resolve_seed_configuration, set_configured_seeds
from src.training.replay_checkpoint import (
    atomic_publish_fold_file, fold_artifact_inventory, load_fold_replay_checkpoint,
    save_fold_replay_checkpoint, restore_rng_state,
)

TRAINING_POLICY = "frozen_fold_class_balanced_replay_v1"
FULL_TASK_LAYOUT = TASK_LAYOUT
SUPPORTED_TASK_PROTOCOLS = PROTOCOL_IDS


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _publish_json(path, payload):
    # A fresh run owns these files. Exclusive create never clobbers old artifacts.
    atomic_publish_fold_file(path, lambda handle: handle.write((_json(payload) + "\n").encode("utf-8")))


def _publish_csv(path, rows):
    atomic_publish_fold_file(path, lambda handle: handle.write(pd.DataFrame(rows).to_csv(index=False).encode("utf-8")))


def engine_methods():
    return {"class_balanced_fixed_budget_replay", "ewc", "agem"}


def method_policy(method):
    return {
        "class_balanced_fixed_budget_replay": TRAINING_POLICY,
        "ewc": "frozen_fold_online_ewc_v1",
        "agem": "frozen_fold_agem_v1",
    }[method]


def validate_fold_options(args):
    if args.method not in engine_methods():
        raise ValueError("Supported methods are CB-Replay, EWC and A-GEM.")
    if args.variant != "tddi_paper_member" or args.norm != "layernorm":
        raise ValueError("Unsupported frozen-fold model/normalization.")
    if args.member_id not in (0, 1, 2) or args.seed != 0 or args.fold_seed != 42:
        raise ValueError("Study requires experiment seed 0, fold seed 42 and members 0/1/2.")
    if not all((args.fold_assignments, args.fold_manifest, args.fold_preprocessing, args.preprocessing_policy)):
        raise ValueError("Provide frozen fold assignments, manifest and preprocessing.")
    if args.export_member_predictions and not args.member_prediction_splits:
        raise ValueError("Prediction export requires at least one validation/test split.")
    if args.batch_size not in (8, 16, 32, 64) or args.effective_batch_size != 1024:
        raise ValueError("Frozen-fold training uses microbatch 64 (OOM: 32/16/8), effective target 1024.")
    if args.gradient_clip_norm is not None:
        raise ValueError("Gradient clipping is not enabled for the locked T-DDI recipe.")
    if args.label_smoothing != 0.:
        raise ValueError("The locked T-DDI recipe uses no label smoothing.")
    if args.evaluation_batch_size <= 0:
        raise ValueError("--evaluation-batch-size must be positive.")
    if args.stop_after_task is not None and args.stop_after_task not in range(8):
        raise ValueError("--stop-after-task must be between 0 and 7.")
    if args.epochs is None or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("Epochs and patience must be positive.")
    expected_loss = "focal_current_ce_replay" if args.method == "class_balanced_fixed_budget_replay" else "cross_entropy"
    if args.loss_variant != expected_loss:
        raise ValueError("Loss must match the selected method.")
    if args.buffer_policy != BUFFER_POLICY or args.exemplar_ranking_policy != RANKING_POLICY:
        raise ValueError("Use the official equal-class and sample-normalized ranking policies.")
    if not math.isfinite(args.replay_fraction) or not 0 <= args.replay_fraction < 1:
        raise ValueError("replay_fraction must be finite and in [0,1).")
    if args.replay_repeat_cap <= 0 or args.agem_reference_batch_size <= 0:
        raise ValueError("Replay cap and A-GEM reference batch must be positive.")
    for name in ("lr", "weight_decay", "focal_gamma", "ewc_lambda"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "lr" and value == 0):
            raise ValueError(f"Invalid hyperparameter {name}.")
    if not 0 <= args.dropout < 1:
        raise ValueError("Dropout must be in [0,1).")


def validate_fold_task_spec(spec, context):
    """Validate one of the two canonical full task maps."""
    tasks = validate_full_task_spec(spec)
    classes = [int(raw) for task in tasks for raw in task["classes"]]
    # Identity-only tables already validated against both source labels. No old
    # descriptors or future features are read for this protocol integrity check.
    assigned = {int(c) for table in context._rows for c in table["raw_class_id"].to_pylist()}
    if len(set(classes)) != 178 or set(classes) != assigned:
        raise ValueError("Task protocol must cover exactly the 178 assignment raw classes.")
    return tasks


def _model_values(raw, prep, columns, member):
    values = prep.transform(raw, feature_columns=columns, member_id=member, policy=prep.metadata["policy"])
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Model inputs overflow float32 after frozen preprocessing.")
    return values


def _ids(arrays):
    return arrays.metadata["sample_id"].tolist()


def _prediction_provenance(prepared, *, metadata, labels, all_oof=None):
    """Bind one prediction to frozen assignment/preprocess/ranking/budget state."""

    fold_ids = np.asarray(metadata["fold_id"], dtype=np.int16)
    sample_ids = np.asarray(metadata["sample_id"]).astype(np.str_, copy=False)
    oof_rows = oof_sha = None
    if all_oof is not None:
        oof_labels, oof_metadata = all_oof
        oof_rows = int(len(oof_labels))
        oof_sha = partition_identity_sha256(
            oof_metadata["sample_id"], oof_labels, oof_metadata["fold_id"]
        )
    resolved, buffer = prepared["resolved"], prepared["buffer"]
    member_id = prepared["seeds"].member_id
    # This digest intentionally excludes member seed/fold, scaler bytes and the
    # one-slot budget remainder. Those are recorded below but legitimately vary.
    shared_contract = {
        "training_policy": resolved["training_policy"],
        "method": resolved["method"],
        "method_protocol": resolved["method_protocol"],
        "experiment_seed": prepared["seeds"].experiment_seed,
        "fold_seed": resolved["fold_seed"],
        "task_protocol": resolved["task_protocol"],
        "task_file_sha256": prepared["task_hash"],
        "task_layout": resolved["task_layout"],
        "assignment_sha256": prepared["manifest"]["assignment_sha256"],
        "fold_manifest_sha256": prepared["context"].manifest_sha256,
        "preprocessing_policy": resolved["preprocessing"]["policy"],
        "ranking_policy": buffer.metadata["ranking"]["policy"],
        "buffer_policy": buffer.metadata["policy"],
        "global_memory_budget": sum(prepared["budgets"].values()),
        "model": resolved["model"],
        "microbatch": resolved["microbatch"],
        "effective_batch_target": resolved["effective_batch_target"],
        "loss_scope": resolved["loss_scope"],
        "hyperparameters": prepared["contract"]["hyperparameters"],
        "sampler": prepared["contract"]["sampler"],
        "implementation_sha256": resolved["implementation_sha256"],
    }
    return PredictionProvenance(
        assignment_sha256=prepared["manifest"]["assignment_sha256"],
        fold_manifest_sha256=prepared["context"].manifest_sha256,
        task_file_sha256=prepared["task_hash"],
        study_contract_sha256=_digest(shared_contract),
        preprocessing_policy=resolved["preprocessing"]["policy"],
        preprocessing_sha256=prepared["prep_hash"],
        preprocessing_member_id=member_id,
        ranking_policy=buffer.metadata["ranking"]["policy"],
        buffer_policy=buffer.metadata["policy"],
        member_memory_budget=prepared["budgets"][member_id],
        global_memory_budget=sum(prepared["budgets"].values()),
        expected_partition_rows=int(len(labels)),
        expected_partition_sha256=partition_identity_sha256(sample_ids, labels, fold_ids),
        expected_oof_rows=oof_rows,
        expected_oof_sha256=oof_sha,
    )


def _export_fold_prediction(
    prepared, *, outputs, labels, metadata, run_id, task_id, split, root,
    all_oof=None,
):
    """Publish outputs already collected by the single reporting pass."""
    member_id = prepared["seeds"].member_id
    artifact_path = root / "member_predictions" / f"task_{task_id}" / f"{split}.npz"
    export_member_prediction_artifact(
        outputs, metadata,
        MemberPredictionContext(
            run_id=run_id, method=prepared["resolved"]["method"],
            method_protocol=prepared["resolved"]["method_protocol"], task_id=task_id, split=split,
            member_id=member_id, experiment_seed=prepared["seeds"].experiment_seed,
            member_seed=prepared["seeds"].member_seed,
            ensemble_mode="stratified_3fold", fold_id=member_id,
            fold_count=3, fold_seed=prepared["resolved"]["fold_seed"],
        ), artifact_path,
        provenance=_prediction_provenance(
            prepared, metadata=metadata, labels=labels, all_oof=all_oof
        ),
    )
    return artifact_path


def _group_metrics_from_outputs(outputs, seen_map, new_classes, criterion):
    """Compute reporting groups without running the model again.

    ``old`` and ``current`` partition ``seen_all``.  Metrics are therefore
    derived by masking the one aligned prediction table.  Only the inexpensive
    loss reduction is repeated; no feature tensor is sent through the model.
    """
    results = []
    old_classes = sorted(set(seen_map) - set(new_classes))
    for group, classes in (("seen_all", sorted(seen_map)), ("old", old_classes), ("current", new_classes)):
        keep = np.isin(outputs.labels, classes)
        if not keep.any():
            results.append({"group": group, "sample_count": 0, "status": "empty"})
            continue
        labels = np.asarray(outputs.labels[keep], dtype=np.int64)
        predictions = np.asarray(outputs.predictions[keep], dtype=np.int64)
        local_labels = remap_labels(labels, seen_map)
        with torch.no_grad():
            loss = float(criterion(
                torch.from_numpy(np.asarray(outputs.logits[keep], dtype=np.float32)),
                torch.from_numpy(local_labels),
            ).item())
        group_metrics = compute_classification_metrics(labels, predictions, labels=classes)
        results.append({
            "group": group,
            "sample_count": int(keep.sum()),
            "status": "ok",
            **group_metrics,
            "loss": loss,
        })
    return results


def _evaluate_fold_split(
    *, engine, model, values, labels, seen_map, new_classes, criterion,
    device, batch_size,
):
    """Run one reporting inference pass and reuse it for metrics and export."""
    loader = DataLoader(
        engine.build_tensor_dataset(values, remap_labels(labels, seen_map)),
        batch_size=batch_size, shuffle=False, drop_last=False,
        generator=torch.Generator().manual_seed(0),
    )
    result = engine.evaluate_model(
        model, loader, criterion, device, invert_class_map(seen_map),
        evaluation_class_indices=list(range(len(seen_map))),
        collect_outputs=True, collect_latent=False,
    )
    if result.outputs is None:
        raise RuntimeError("Fold reporting evaluation did not return prediction outputs.")
    return _group_metrics_from_outputs(
        result.outputs, seen_map, new_classes, criterion
    ), result.outputs


def prepare_fold_run(args, *, engine, context=None):
    """Read-only shared trainer/orchestrator preflight: never create a model.

    A context may be shared within a single planning call. Each actual trainer
    invocation/resume creates and validates its own fresh context.
    """
    validate_fold_options(args)
    columns = load_feature_columns(args.feature_cols)
    if len(columns) != TDDI_PAPER_INPUT_DIM or len(set(columns)) != len(columns):
        raise ValueError("Paper member requires exactly 3780 distinct descriptors in recorded order.")
    task_hash = fold_file_sha256(args.task_file)
    spec = engine.load_task_spec(args.task_file)
    print("Frozen-fold preflight: validating assignment/source hashes and IDs; no model created yet.", flush=True)
    if context is None:
        context = prepare_development_fold_context(args.fold_assignments, args.fold_manifest,
            source_paths={"train": args.train, "validation": args.validation, "test": args.test})
    else:
        requested = (args.fold_assignments, args.fold_manifest, args.train, args.validation, args.test)
        if tuple(Path(p).resolve() for p in requested) != tuple(p for p, _ in context._file_stamps):
            raise ValueError("Cached fold context does not match requested source/assignment paths.")
        context.assert_unchanged()
    tasks = validate_fold_task_spec(spec, context)
    manifest = context.manifest
    if manifest["fold_seed"] != 42:
        raise ValueError("Assignment fold seed must be 42.")
    prep_hash = fold_file_sha256(args.fold_preprocessing)
    prep = load_fold_preprocessing(args.fold_preprocessing, context=context, task_file=args.task_file,
        feature_columns=columns, member_id=args.member_id, validation_fold=args.member_id,
        policy=args.preprocessing_policy, experiment_seed=args.seed, expected_sha256=prep_hash)
    if args.fold_member_budget is None:
        budgets = four_percent_member_budgets(manifest["assignment_row_count"])
    else:
        if args.fold_member_budget < 0:
            raise ValueError("--fold-member-budget must be non-negative.")
        if args.fold_global_budget is None:
            raise ValueError("--fold-global-budget is required with --fold-member-budget.")
        budgets = {member: int(args.fold_member_budget) for member in (0, 1, 2)}
        if int(args.fold_global_budget) != sum(budgets.values()):
            raise ValueError("Explicit frozen-fold global budget does not equal three member budgets.")
    if args.method == "ewc":
        budgets = {member: 0 for member in (0, 1, 2)}
    buffer_kwargs = dict(method=args.method, context=context, task_file=args.task_file, feature_columns=columns,
        member_id=args.member_id, total_memory_budget=budgets[args.member_id],
        experiment_seed=args.seed, buffer_policy=args.buffer_policy, ranking_policy=args.exemplar_ranking_policy)
    buffer = create_memory(**buffer_kwargs)
    seeds = resolve_seed_configuration(args.seed, args.member_id)
    device = engine.resolve_device(args.device)
    stop = 7 if args.stop_after_task is None else args.stop_after_task
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    resolved = {
        "training_policy": method_policy(args.method), "schema_version": 2, "method": args.method,
        "method_protocol": method_policy(args.method), "ensemble_mode": "frozen_stratified_3fold",
        "seeds": asdict(seeds), "validation_fold": args.member_id, "fold_seed": 42,
        "task_protocol": spec["protocol"], "task_spec": spec,
        "task_protocol_id": SUPPORTED_TASK_PROTOCOLS[spec["protocol"]],
        "task_file_sha256": task_hash,
        "task_layout": FULL_TASK_LAYOUT, "protocol_task_count": 8, "stop_after_task": stop,
        "assignment_sha256": manifest["assignment_sha256"], "fold_manifest_sha256": context.manifest_sha256,
        "sources": buffer.metadata["sources"], "preprocessing_sha256": prep_hash,
        "preprocessing": prep.metadata, "buffer": buffer.metadata,
        "member_budgets": budgets, "global_slot_budget": sum(budgets.values()),
        "ranking_seed_role": buffer.metadata["ranking"]["policy"],
        "microbatch": args.batch_size, "gradient_accumulation": args.effective_batch_size // args.batch_size,
        "effective_batch_target": args.effective_batch_size,
        "reporting_batch_size": args.evaluation_batch_size,
        "reporting_policy": "single_inference_pass_metrics_and_prediction_export_v1",
        "core_reporting_metrics": [
            "macro_f1", "balanced_accuracy", "accuracy", "weighted_f1",
            "sample_count",
        ],
        "drop_last": False, "optimizer": "AdamW_new_each_task", "scheduler": None,
        "classification_weight": 1.0, "loss_variant": args.loss_variant,
        "loss_scope": ("focal_current_cross_entropy_replay" if args.method == "class_balanced_fixed_budget_replay"
                       else "cross_entropy_plus_online_ewc" if args.method == "ewc" else "cross_entropy_with_agem_projection"),
        "early_stopping": "held_out_all_seen_macro_f1", "validation_only": args.validation_only,
        "test_policy": "integrity_hash_and_pair_IDs_only" if args.validation_only else "report_only_after_best_validation",
        "prediction_export": {
            "enabled": bool(args.export_member_predictions),
            "splits": list(args.member_prediction_splits) if args.export_member_predictions else [],
            "schema": "member_prediction_v3_assignment_bound",
            "oof_policy": "concatenate_one_held_out_fold_prediction_per_sample",
        },
        "checkpoint_policy": "immutable_frozen_fold_task_boundary_v1",
        "model": engine.model_manifest(args), "device": device,
        "implementation_sha256": {name: fold_file_sha256(Path(__file__).parents[1] / name) for name in (
            "training/fold_training.py", "training/train_cil.py", "eval/evaluation.py",
            "eval/metrics.py", "eval/predictions.py", "data/ddi_dataset.py",
            "data/fold_preprocessing.py", "data/class_balanced_replay_buffer.py", "data/replay_sampler.py",
            "methods/ewc.py", "methods/agem.py", "methods/replay.py", "models/tddi_paper_member.py",
            "training/regularization_memory.py", "training/replay_checkpoint.py")},
    }
    # Scope/path relocation may change; learning/provenance contract may not.
    contract = {k: v for k, v in resolved.items() if k != "stop_after_task"}
    contract["hyperparameters"] = {k: getattr(args, k) for k in (
        "epochs", "patience", "lr", "weight_decay", "dropout", "activation", "norm", "focal_gamma",
        "loss_variant", "ewc_lambda", "agem_reference_batch_size")}
    contract["sampler"] = {
        "policy": SAMPLER_POLICY, "schema_version": 1, "rng_derivation": RNG_DERIVATION,
        "replay_fraction": args.replay_fraction if args.method == "class_balanced_fixed_budget_replay" else 0.0,
        "repeat_cap": args.replay_repeat_cap, "task_boundary_epoch_reset": 0,
    }
    contract = json.loads(_json(contract))
    task_cache_contract = None
    source_shard_root = getattr(args, "source_shard_cache_root", None)
    if source_shard_root is not None:
        source_shard_root = Path(source_shard_root).resolve()
        validate_source_shard_cache(
            source_shard_root,
            task_file=args.task_file,
            feature_columns=columns,
            fold_manifest_sha256=context.manifest_sha256,
            assignment_sha256=context.manifest.get("assignment_sha256"),
            source_paths={"train": args.train, "validation": args.validation},
        )
        resolved["raw_source_shard_cache_root"] = str(source_shard_root)
    if getattr(args, "task_cache_root", None) is not None:
        args.task_cache_root = Path(args.task_cache_root).resolve()
        assignment_sample_ids = np.concatenate([
            table["sample_id"].to_numpy(zero_copy_only=False) for table in context._rows
        ])
        assignment_fold_ids = np.concatenate([
            table["fold_id"].to_numpy(zero_copy_only=False) for table in context._rows
        ])
        assignment_hash = assignment_arrays_sha256(
            assignment_sample_ids,
            assignment_fold_ids,
            fold_count=int(manifest["n_splits"]),
            seed=int(manifest["fold_seed"]),
        )
        task_cache_contract = build_task_cache_contract(
            protocol_id=str(spec.get("protocol") or "unspecified"),
            task_file_sha256=task_hash,
            feature_columns=columns,
            scaler_sha256=prep_hash,
            member_id=args.member_id,
            fold_id=args.member_id,
            fold_count=manifest["n_splits"],
            fold_seed=manifest["fold_seed"],
            fold_assignments_sha256=assignment_hash,
            preprocessing_policy=args.preprocessing_policy,
            source_hashes={
                "train": manifest["sources"]["train"]["sha256"],
                "validation": manifest["sources"]["validation"]["sha256"],
            },
        )
        resolved["common_task_cache_root"] = str(args.task_cache_root)
    return dict(columns=columns, tasks=tasks, task_hash=task_hash, context=context, manifest=manifest,
                prep=prep, prep_hash=prep_hash, budgets=budgets, buffer_kwargs=buffer_kwargs, buffer=buffer,
                seeds=seeds, device=device, stop=stop, arguments=arguments, resolved=resolved, contract=contract,
                task_cache_contract=task_cache_contract,
                task_cache_root=getattr(args, "task_cache_root", None),
                source_shard_cache_root=source_shard_root)


def run_fold_training(args, *, engine):
    """Execute one member with shared preflight and immutable task-boundary resume."""
    validate_fold_options(args)
    root = Path(args.outdir)
    resume_path = getattr(args, "resume_fold_checkpoint", None)
    if root.exists() and not resume_path:
        raise FileExistsError(f"Frozen-fold run output already exists; never overwrite/restart: {root}")
    prepared = prepare_fold_run(args, engine=engine)
    columns, tasks, task_hash, context, manifest = (prepared[k] for k in
        ("columns", "tasks", "task_hash", "context", "manifest"))
    prep, prep_hash, budgets, buffer_kwargs, buffer = (prepared[k] for k in
        ("prep", "prep_hash", "budgets", "buffer_kwargs", "buffer"))
    seeds, device, stop, arguments, resolved, contract = (prepared[k] for k in
        ("seeds", "device", "stop", "arguments", "resolved", "contract"))
    task_cache_contract = prepared.get("task_cache_contract")
    task_cache_root = prepared.get("task_cache_root")
    source_shard_root = prepared.get("source_shard_cache_root")
    loaded = None
    if resume_path:
        loaded, buffer = load_fold_replay_checkpoint(resume_path, root=root, expected_contract=contract,
            tasks=tasks, context=context, buffer_kwargs=buffer_kwargs)
        run_id = loaded["run_id"]
    else:
        run_id = uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=False)  # claim namespace, also protects concurrent starts
        _publish_json(root / "run_config.json", {"run_id": run_id, "arguments": arguments, "resolved": resolved,
            "checkpoint_contract": contract,
            "config_sha256": _digest({"arguments": arguments, "resolved": resolved}),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "git": engine._git_state(engine.PROJECT_ROOT)})
    config_hash = fold_file_sha256(root / "run_config.json")
    if loaded and loaded["completed_task_id"] >= stop:
        print(f"Verified requested scope 0..{stop} already complete; full trajectory complete={loaded['full_trajectory_complete']}", flush=True)
        return loaded["progress"]["task_summaries"][:stop + 1]
    logger = RunLogger(root / "train.log", root / "events.csv", root / "stdout.log", run_id)
    logger.log(f"Frozen-fold training started policy={method_policy(args.method)} member={args.member_id} stop_after_task={stop}")
    set_configured_seeds(seeds)
    previous, previous_map = None, {}
    fisher, theta_star = None, None
    all_metrics, all_epochs, task_summaries = [], [], []
    artifact_hashes, start_task = {}, 0
    if loaded:
        previous_map = loaded["seen_class_map"]
        previous = engine.expand_model_for_seen_classes(None, None, previous_map,
            variant=args.variant, input_dim=len(columns), dropout=args.dropout,
            activation=args.activation, norm=args.norm)
        previous.load_state_dict(loaded["model_state"], strict=True)
        previous.eval().requires_grad_(False)
        all_metrics, all_epochs, task_summaries = (loaded["progress"][k] for k in
                                                   ("metric_rows", "epoch_rows", "task_summaries"))
        artifact_hashes, start_task = dict(loaded["artifact_hashes"]), loaded["next_task_id"]
        # Reconstruction consumes initialization RNG. Restore AFTER it and before
        # expansion of the next student, matching a continuous task boundary.
        restore_rng_state(loaded["rng_state"])
        fisher, theta_star = loaded["method_state"].get("fisher"), loaded["method_state"].get("theta_star")
        del loaded
        logger.log(f"Resumed task boundary; next_task={start_task}, preprocessing not refit")
    for task in tasks[start_task:stop + 1]:
        task_id, new_classes = task["task_id"], task["classes"]
        started = time.perf_counter()
        if fold_file_sha256(args.task_file) != task_hash:
            raise ValueError("Task-file changed during run.")
        context.assert_unchanged()
        task_root = root / f"task_{task_id}"
        if task_root.exists():
            # Preserve any interrupted attempt, including all partial artifacts.
            task_root = root / "attempts" / f"task_{task_id}_{uuid.uuid4().hex}"
            task_root.parent.mkdir(exist_ok=True)
        task_root.mkdir(exist_ok=False)
        seen_map = build_seen_class_map(set(previous_map) | set(new_classes))
        logger.log(f"task={task_id} loading current/held-out seen_classes={len(seen_map)}")
        cache_loaded = False
        cache_manifest = (
            task_cache_root / f"member_{args.member_id}" / f"task_{task_id}" / "manifest.json"
            if task_cache_root is not None else None
        )
        if task_cache_contract is not None and cache_manifest is not None and cache_manifest.is_file():
            current, _ = load_task_cache_role(
                task_cache_root, member_id=args.member_id, task_id=task_id,
                expected_contract=task_cache_contract, feature_columns=columns,
                role="train", current_class_ids=new_classes, feature_view="raw",
            )
            validation = load_validation_trajectory(
                task_cache_root, member_id=args.member_id,
                task_class_ids=[item["classes"] for item in tasks], upto_task_id=task_id,
                expected_contract=task_cache_contract, feature_columns=columns, feature_view="raw",
            )
            current_model, _ = load_task_cache_role(
                task_cache_root, member_id=args.member_id, task_id=task_id,
                expected_contract=task_cache_contract, feature_columns=columns,
                role="train", current_class_ids=new_classes, feature_view="model",
            )
            validation_model = load_validation_trajectory(
                task_cache_root, member_id=args.member_id,
                task_class_ids=[item["classes"] for item in tasks], upto_task_id=task_id,
                expected_contract=task_cache_contract, feature_columns=columns, feature_view="model",
            )
            if (
                not np.array_equal(current.labels, current_model.labels)
                or not np.array_equal(validation.labels, validation_model.labels)
                or _ids(current) != _ids(current_model)
                or _ids(validation) != _ids(validation_model)
            ):
                raise ValueError("Common task-cache raw/model views are not row-aligned.")
            cache_loaded = True
            logger.log(f"task={task_id} common_task_cache=hit root={task_cache_root}")
        if not cache_loaded:
            cache_guard = (
                task_cache_lock(task_cache_root, member_id=args.member_id, task_id=task_id)
                if task_cache_contract is not None else nullcontext()
            )
            with cache_guard:
                if task_cache_contract is not None and cache_manifest.is_file():
                    current, _ = load_task_cache_role(
                        task_cache_root, member_id=args.member_id, task_id=task_id,
                        expected_contract=task_cache_contract, feature_columns=columns,
                        role="train", current_class_ids=new_classes, feature_view="raw",
                    )
                    validation = load_validation_trajectory(
                        task_cache_root, member_id=args.member_id,
                        task_class_ids=[item["classes"] for item in tasks], upto_task_id=task_id,
                        expected_contract=task_cache_contract, feature_columns=columns, feature_view="raw",
                    )
                    current_model, _ = load_task_cache_role(
                        task_cache_root, member_id=args.member_id, task_id=task_id,
                        expected_contract=task_cache_contract, feature_columns=columns,
                        role="train", current_class_ids=new_classes, feature_view="model",
                    )
                    validation_model = load_validation_trajectory(
                        task_cache_root, member_id=args.member_id,
                        task_class_ids=[item["classes"] for item in tasks], upto_task_id=task_id,
                        expected_contract=task_cache_contract, feature_columns=columns, feature_view="model",
                    )
                    cache_loaded = True
                    logger.log(f"task={task_id} common_task_cache=hit_after_wait root={task_cache_root}")
                else:
                    if source_shard_root is not None:
                        current = load_source_shard(
                            source_shard_root, member_id=args.member_id, task_id=task_id,
                            validation_fold=args.member_id, role="train", feature_columns=columns)
                        validation_shard = load_source_shard(
                            source_shard_root, member_id=args.member_id, task_id=task_id,
                            validation_fold=args.member_id, role="validation", feature_columns=columns)
                        if task_cache_contract is None:
                            validation = concatenate_task_arrays([
                                load_source_shard(
                                    source_shard_root, member_id=args.member_id, task_id=seen_task["task_id"],
                                    validation_fold=args.member_id, role="validation", feature_columns=columns)
                                for seen_task in tasks[: task_id + 1]
                            ])
                        else:
                            validation = validation_shard
                    else:
                        current = load_development_fold_arrays(
                            context, columns, role="train", member_id=args.member_id,
                            validation_fold=args.member_id, class_ids=new_classes)
                        validation = load_development_fold_arrays(
                            context, columns, role="validation", member_id=args.member_id,
                            validation_fold=args.member_id,
                            class_ids=new_classes if task_cache_contract is not None else list(seen_map),
                        )
                    current_model = DDIBatchArrays(
                        _model_values(current.features, prep, columns, args.member_id),
                        current.labels.copy(),
                        {key: value.copy() for key, value in (current.metadata or {}).items()},
                    )
                    validation_model = DDIBatchArrays(
                        _model_values(validation.features, prep, columns, args.member_id),
                        validation.labels.copy(),
                        {key: value.copy() for key, value in (validation.metadata or {}).items()},
                    )
                    if task_cache_contract is not None:
                        write_task_cache(
                            task_cache_root, member_id=args.member_id, task_id=task_id,
                            contract=task_cache_contract, current_class_ids=new_classes,
                            validation_class_ids=new_classes, feature_columns=columns,
                            train=current, validation=validation,
                            model_train=current_model, model_validation=validation_model,
                        )
                        logger.log(f"task={task_id} common_task_cache=written root={task_cache_root}")
                    if task_cache_contract is not None:
                        validation = load_validation_trajectory(
                            task_cache_root, member_id=args.member_id,
                            task_class_ids=[item["classes"] for item in tasks], upto_task_id=task_id,
                            expected_contract=task_cache_contract, feature_columns=columns, feature_view="raw",
                        )
                        validation_model = load_validation_trajectory(
                            task_cache_root, member_id=args.member_id,
                            task_class_ids=[item["classes"] for item in tasks], upto_task_id=task_id,
                            expected_contract=task_cache_contract, feature_columns=columns, feature_view="model",
                        )
        retained = buffer.get_all()
        if not len(current.labels) or not len(validation.labels):
            raise ValueError("Empty current or held-out all-seen view.")
        if set(retained.labels) - set(previous_map):
            raise ValueError("Replay contains non-old classes.")
        if set(_ids(current) + _ids(retained)) & set(_ids(validation)):
            raise ValueError("Training/replay overlaps held-out validation IDs.")
        replay_ids = _ids(retained) if args.method == "class_balanced_fixed_budget_replay" else []
        sampler = FoldReplayFractionSampler(
            current_sample_ids=_ids(current), replay_sample_ids=replay_ids,
            replay_raw_labels=retained.labels if args.method == "class_balanced_fixed_budget_replay" else np.empty(0, dtype=np.int64), experiment_seed=args.seed,
            member_id=args.member_id, task_id=task_id,
            replay_fraction=contract["sampler"]["replay_fraction"],
            repeat_cap=args.replay_repeat_cap,
        )
        _publish_json(task_root / "input_audit.json", {"task_id": task_id, "current_ids": _ids(current),
            "replay_ids_before": replay_ids, "replay_raw_labels_before": retained.labels.tolist() if replay_ids else [],
            "validation_ids": _ids(validation),
            "current_raw_classes": new_classes, "seen_class_map": seen_map,
            "head_size": len(seen_map), "sampler": sampler.metadata,
            "assignment_sha256": manifest["assignment_sha256"], "preprocessing_sha256": prep_hash,
            "buffer_policy": buffer.metadata["policy"], "ranking": buffer.metadata["ranking"]})
        current_inputs = current_model.features
        replay_inputs = (_model_values(retained.features, prep, columns, args.member_id)
                         if args.method == "class_balanced_fixed_budget_replay" else np.empty((0, len(columns)), dtype=np.float32))
        replay_labels = retained.labels if args.method == "class_balanced_fixed_budget_replay" else np.empty(0, dtype=np.int64)
        val_inputs = validation_model.features
        training = engine.build_tensor_dataset(np.concatenate((current_inputs, replay_inputs)),
            remap_labels(np.concatenate((current.labels, replay_labels)), seen_map))
        loader = DataLoader(training, batch_size=args.batch_size, sampler=sampler, drop_last=False,
                            generator=torch.Generator().manual_seed(seeds.member_seed + task_id))
        val_loader = DataLoader(engine.build_tensor_dataset(val_inputs, remap_labels(validation.labels, seen_map)),
                               batch_size=args.batch_size, shuffle=False, drop_last=False,
                               generator=torch.Generator().manual_seed(seeds.member_seed + task_id))
        del current_inputs, replay_inputs
        model = engine.expand_model_for_seen_classes(previous, previous_map, seen_map,
            variant=args.variant, input_dim=len(columns), dropout=args.dropout,
            activation=args.activation, norm=args.norm).to(device)
        if args.method == "ewc" and fisher is not None:
            fisher = {k: v.to(device) for k, v in grow_head_state(fisher, previous_map, seen_map, model.state_dict(), zero_new_rows=True).items()}
            theta_star = {k: v.to(device) for k, v in grow_head_state(theta_star, previous_map, seen_map, model.state_dict(), zero_new_rows=False).items()}
        optimizer = engine.make_optimizer(model, args)
        criterion = engine.training_criterion(args)
        evaluation_criterion = criterion
        best_score, best_state, best_epoch, best_validation_metrics, stale = (
            -float("inf"), None, None, None, 0
        )
        epoch_rows = []
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            projector = None
            schedule_kwargs = {}
            if args.method == "agem" and len(retained.labels):
                def projector(step):
                    rng = np.random.default_rng(np.random.SeedSequence([seeds.member_seed, task_id, epoch, step]))
                    idx = rng.choice(len(retained.labels), min(args.agem_reference_batch_size, len(retained.labels)), replace=False)
                    engine.apply_agem_projection(model, criterion,
                        _model_values(retained.features[idx], prep, columns, args.member_id),
                        retained.labels[idx], seen_map, device)
            losses = engine.train_one_epoch(model, loader, optimizer, criterion, device,
                old_class_indices=[seen_map[c] for c in previous_map],
                fisher=fisher, theta_star=theta_star, ewc_lambda=args.ewc_lambda,
                gradient_accumulation_steps=args.effective_batch_size // args.batch_size,
                gradient_projector=projector, loss_variant=args.loss_variant,
                return_loss_components=True, gradient_clip_norm=args.gradient_clip_norm, **schedule_kwargs)
            audit = sampler.last_audit
            expected_replay = (
                0 if task_id == 0 or args.method != "class_balanced_fixed_budget_replay" else min(
                    int(len(current.labels) * args.replay_fraction / (1.0 - args.replay_fraction)),
                    args.replay_repeat_cap * len(retained.labels),
                )
            )
            expected_current = len(current.labels)
            if (audit is None or audit["epoch"] != epoch or audit["current_draws"] != expected_current
                    or audit["max_repeat"] > args.replay_repeat_cap or audit["actual_replay_draws"] !=
                    expected_replay
                    or losses.examples_seen != len(sampler)
                    or losses.optimizer_steps != math.ceil(len(sampler) / args.effective_batch_size)
                    or losses.tail_effective_batch != (len(sampler) - 1) % args.effective_batch_size + 1):
                raise RuntimeError("Runtime sampler/current coverage/cap/optimizer-step audit mismatch.")
            result = engine.evaluate_model(model, val_loader, evaluation_criterion, device, invert_class_map(seen_map),
                                           evaluation_class_indices=list(range(len(seen_map))))
            row = {"task": task_id, "epoch": epoch + 1, **asdict(losses),
                   "classification_weight": 1.0,
                   "validation_macro_f1": result.metrics["macro_f1"],
                   "validation_balanced_accuracy": result.metrics["balanced_accuracy"],
                   "validation_loss": result.metrics["loss"], "seconds": time.perf_counter() - epoch_start}
            if not all(math.isfinite(v) for v in row.values()):
                raise FloatingPointError("Nonfinite frozen-fold training/validation metrics.")
            epoch_rows.append(row)
            _publish_json(task_root / f"epoch_{epoch + 1}_audit.json", {**row, "sampling": audit})
            logger.log_event("epoch", f"task={task_id} epoch={epoch + 1} classification_loss={losses.classification_loss:.6f} "
                f"total_loss={losses.total_loss:.6f} "
                f"val_macro_f1={row['validation_macro_f1']:.6f} steps={losses.optimizer_steps}", _json(row))
            score = result.metrics["macro_f1"]
            if score > best_score:
                best_score, best_epoch, stale = score, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_validation_metrics = dict(result.metrics)
            else:
                stale += 1
                if stale >= args.patience:
                    break
        if best_state is None or best_validation_metrics is None:
            raise RuntimeError(f"Task {task_id} did not produce a finite best checkpoint.")
        model.load_state_dict(best_state)
        reporting_timings = {}
        stage_started = time.perf_counter()
        validation_group_metrics, validation_outputs = _evaluate_fold_split(
            engine=engine, model=model, values=val_inputs,
            labels=validation.labels, seen_map=seen_map,
            new_classes=new_classes, criterion=evaluation_criterion, device=device,
            batch_size=args.evaluation_batch_size,
        )
        reporting_timings["validation_inference_seconds"] = time.perf_counter() - stage_started
        metrics = [
            {"task": task_id, "split": "validation", **row}
            for row in validation_group_metrics
        ]
        selected_seen_metrics = next(row for row in metrics if row["group"] == "seen_all")
        # Model-only inference snapshot, deliberately not a resumable checkpoint.
        atomic_publish_fold_file(task_root / "best_model.pt", lambda handle:
            torch.save({"model_state": best_state, "seen_class_map": seen_map, "task_id": task_id,
                        "run_id": run_id, "resumable": False, "policy": method_policy(args.method)}, handle))
        prediction_paths = []
        if args.export_member_predictions and "validation" in args.member_prediction_splits:
            stage_started = time.perf_counter()
            all_oof = load_development_fold_identity(
                context, role="all", member_id=args.member_id,
                validation_fold=args.member_id, class_ids=list(seen_map),
            )
            prediction_paths.append(_export_fold_prediction(
                prepared, outputs=validation_outputs,
                labels=validation.labels, metadata=validation.metadata,
                run_id=run_id, task_id=task_id, split="validation",
                root=root, all_oof=all_oof,
            ))
            reporting_timings["validation_export_seconds"] = time.perf_counter() - stage_started
        if not args.validation_only:
            stage_started = time.perf_counter()
            context.assert_unchanged()
            frame = load_split_frame(
                args.test, columns, class_ids=list(seen_map),
                include_metadata=True, meta_cols=DEFAULT_META_COLS,
            )
            test_values = _model_values(frame[columns].to_numpy(dtype=np.float64), prep, columns, args.member_id)
            test_labels = frame["class"].to_numpy(dtype=np.int64)
            reporting_timings["test_load_transform_seconds"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            test_group_metrics, test_outputs = _evaluate_fold_split(
                engine=engine, model=model, values=test_values,
                labels=test_labels, seen_map=seen_map,
                new_classes=new_classes, criterion=evaluation_criterion, device=device,
                batch_size=args.evaluation_batch_size,
            )
            reporting_timings["test_inference_seconds"] = time.perf_counter() - stage_started
            metrics.extend(
                {"task": task_id, "split": "test", **row}
                for row in test_group_metrics
            )
            if args.export_member_predictions and "test" in args.member_prediction_splits:
                stage_started = time.perf_counter()
                test_metadata = {column: frame[column].to_numpy(copy=False) for column in DEFAULT_META_COLS}
                test_metadata["sample_id"] = np.char.add(
                    np.char.add(test_metadata[DEFAULT_META_COLS[0]].astype(np.str_), "|"),
                    test_metadata[DEFAULT_META_COLS[1]].astype(np.str_),
                )
                test_metadata["fold_id"] = np.full(len(frame), -1, dtype=np.int16)
                prediction_paths.append(_export_fold_prediction(
                    prepared, outputs=test_outputs,
                    labels=test_labels, metadata=test_metadata, run_id=run_id,
                    task_id=task_id, split="test", root=root,
                ))
                reporting_timings["test_export_seconds"] = time.perf_counter() - stage_started
            del frame, test_values, test_outputs
        del validation_outputs
        stage_started = time.perf_counter()
        buffer_audit = buffer.update(current, task_id=task_id, feature_columns=columns)
        reporting_timings["buffer_update_seconds"] = time.perf_counter() - stage_started
        logger.log(
            f"task={task_id} reporting_timings="
            f"{_json({key: round(value, 6) for key, value in reporting_timings.items()})}"
        )
        after = buffer.get_all()
        if buffer.total_size > budgets[args.member_id] or set(_ids(after)) - set(_ids(current) + _ids(retained)):
            raise RuntimeError("Retained quota/identity/source invariant failed.")
        _publish_json(task_root / "buffer_audit.json", {**buffer_audit, "retained_ids": _ids(after),
            "retained_raw_labels": after.labels.tolist(), "ranking": buffer.metadata["ranking"]})
        _publish_json(task_root / "metrics.json", metrics)
        _publish_csv(task_root / "metrics.csv", metrics)
        _publish_csv(task_root / "training_audit.csv", epoch_rows)
        summary = {"task_id": task_id, "head_size": len(seen_map), "seen_class_map": seen_map,
            "best_epoch": best_epoch, "best_validation_macro_f1": best_score, "epochs_trained": len(epoch_rows),
            "selection_validation_macro_f1": best_validation_metrics["macro_f1"],
            "selection_validation_balanced_accuracy": best_validation_metrics["balanced_accuracy"],
            "validation_macro_f1": selected_seen_metrics["macro_f1"],
            "validation_balanced_accuracy": selected_seen_metrics["balanced_accuracy"],
            "reporting_timings": reporting_timings,
            "memory_before": len(retained.labels), "memory_after": buffer.total_size,
            "runtime_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if str(device).startswith("cuda") else None,
            "checkpoint_size_bytes": (task_root / "best_model.pt").stat().st_size,
            "artifact_directory": task_root.relative_to(root).as_posix(),
            "boundary_checkpoint": f"checkpoints/task_{task_id}.pt",
            "prediction_artifacts": [path.relative_to(root).as_posix() for path in prediction_paths],
            "completion_policy": "requires_valid_boundary_checkpoint_and_hashed_artifacts"}
        summary = json.loads(_json(summary))
        _publish_json(task_root / "completed_task.json", summary)
        all_metrics.extend(metrics)
        all_epochs.extend(epoch_rows)
        task_summaries.append(summary)
        artifact_hashes.update(fold_artifact_inventory(root, task_root))
        if prediction_paths:
            artifact_hashes.update(fold_artifact_inventory(root, prediction_paths[0].parent))
        if args.method == "ewc":
            fisher_loader = DataLoader(engine.build_tensor_dataset(current_model.features, remap_labels(current.labels, seen_map)),
                                       batch_size=args.batch_size, shuffle=False,
                                       generator=torch.Generator().manual_seed(seeds.member_seed + task_id))
            new_fisher = compute_fisher(model, fisher_loader, device)
            fisher = {k: (new_fisher[k] + (fisher[k] if fisher is not None else 0)).detach().cpu() for k in new_fisher}
            theta_star = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
        checkpoint_started = time.perf_counter()
        boundary_path = save_fold_replay_checkpoint(root / "checkpoints" / f"task_{task_id}.pt", run_id=run_id,
            completed_task_id=task_id, model_state=best_state, seen_class_map=seen_map, contract=contract,
            buffer=buffer, sampler=sampler, method_state={"fisher": fisher, "theta_star": theta_star} if args.method == "ewc" else {}, run_config_sha256=config_hash, artifact_hashes=artifact_hashes,
            progress={"metric_rows": all_metrics, "epoch_rows": all_epochs, "task_summaries": task_summaries})
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        logger.log(f"task={task_id} complete head={len(seen_map)} best_epoch={best_epoch} retained={buffer.total_size} "
                   f"checkpoint={boundary_path} checkpoint_bytes={boundary_path.stat().st_size} "
                   f"checkpoint_seconds={checkpoint_seconds:.6f}")
        previous, previous_map = model.cpu(), seen_map
        del optimizer, model, best_state, current, validation, retained, after, training, loader, val_loader, val_inputs
    report_root = root if not resume_path else root / "reports" / f"through_task_{stop}_{uuid.uuid4().hex}"
    report_root.mkdir(parents=True, exist_ok=True)
    _publish_csv(report_root / "metrics.csv", all_metrics)
    _publish_csv(report_root / "training_audit.csv", all_epochs)
    _publish_json(report_root / "run_summary.json", {"run_id": run_id, "completed_task_id": stop,
        "full_protocol_complete": stop == 7, "task_file_sha256": task_hash, "tasks": task_summaries,
        "requested_scope_complete": True, "validation_only": args.validation_only, "resumable": True})
    with (report_root / "run_summary.md").open("x", encoding="utf-8") as handle:
        handle.write(f"# Frozen-fold study\n\nPolicy: `{method_policy(args.method)}`. Member {args.member_id}. "
                     f"Preprocessing: `{args.preprocessing_policy}`.\n\nCompleted tasks 0–{stop} of full "
                     f"{resolved['task_protocol_id']} (8 tasks). "
                     f"Validation-only: {args.validation_only}. Resume via checkpoints/task_{stop}.pt; not a legacy checkpoint.\n\n")
        for s in task_summaries:
            handle.write(f"- Task {s['task_id']}: head {s['head_size']}, best epoch {s['best_epoch']}, "
                         f"selection validation Macro-F1 {s['selection_validation_macro_f1']:.6f}, "
                         f"validation Macro-F1 {s['validation_macro_f1']:.6f}.\n")
    logger.log(f"Requested execution scope complete; report={report_root}.")
    return task_summaries
