"""Opt-in frozen-fold replay trainer with task-boundary resume (Prompts 9--10).

The numerical training/expansion/evaluation kernels remain in train_cil. This
module only composes the validated APIs from Prompts 5--8. Never fits a scaler,
rebuilds folds, or rereads discarded old training descriptors.
"""
from __future__ import annotations

from dataclasses import asdict
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
    DEFAULT_META_COLS, load_development_fold_arrays, load_development_fold_identity,
    load_feature_columns, load_split_frame,
    prepare_development_fold_context,
)
from src.data.fold_preprocessing import load_fold_preprocessing
from src.data.fold_replay_buffer import (
    PIPELINE_INPUT_RANKING_POLICY,
    RANKING_POLICIES,
    SAMPLE_NORMALIZED_RANKING_POLICY,
    FoldSqrtReplayBuffer,
    four_percent_member_budgets,
)
from src.data.fold_replay_sampler import FoldReplayFractionSampler, SAMPLER_POLICY, RNG_DERIVATION
from src.data.stratified_folds import fold_file_sha256
from src.models.tddi_paper_member import TDDI_PAPER_INPUT_DIM, paper_member_manifest
from src.eval.predictions import (
    MemberPredictionContext,
    PredictionProvenance,
    export_member_prediction_artifact,
    partition_identity_sha256,
)
from src.utils.logging import RunLogger
from src.utils.seed import resolve_seed_configuration, set_configured_seeds
from src.training.replay_checkpoint import (
    atomic_publish_fold_file, fold_artifact_inventory, load_fold_replay_checkpoint,
    save_fold_replay_checkpoint, restore_rng_state,
)

TRAINING_POLICY = "frozen_fold_replay_distill_v1"
P3_LAYOUT = [38, 20, 20, 20, 20, 20, 20, 20]


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _publish_json(path, payload):
    # A fresh run owns these files. Exclusive create never clobbers old artifacts.
    atomic_publish_fold_file(path, lambda handle: handle.write((_json(payload) + "\n").encode("utf-8")))


def _publish_csv(path, rows):
    atomic_publish_fold_file(path, lambda handle: handle.write(pd.DataFrame(rows).to_csv(index=False).encode("utf-8")))


def validate_fold_options(args):
    if args.method != "replay_distill_fixed_budget_uniform":
        raise ValueError("Frozen-fold replay policy requires replay_distill_fixed_budget_uniform.")
    if args.variant != "tddi_paper_member" or args.norm != "layernorm":
        raise ValueError("Frozen-fold training requires tddi_paper_member with input LayerNorm.")
    if args.member_id not in (0, 1, 2) or args.seed != 0 or args.fold_seed != 42 or args.fold_count != 3:
        raise ValueError("Study requires experiment seed 0, fold seed 42, members 0/1/2 and three folds.")
    if args.fold_id not in (None, args.member_id) or args.ensemble_mode != "seeded":
        raise ValueError("Frozen assignments imply member=held-out fold; do not mix legacy ensemble-mode.")
    if not all((args.fold_assignments, args.fold_manifest, args.fold_preprocessing, args.preprocessing_policy)):
        raise ValueError("Provide fold assignments, manifest, frozen preprocessing and explicit preprocessing policy.")
    if args.scaler is not None:
        raise ValueError("Do not pass the legacy --scaler with the frozen-fold policy.")
    if args.resume_replay_checkpoint or args.resume_ewc_checkpoint:
        raise ValueError("Prompt 10 uses --resume-fold-checkpoint; legacy checkpoints are incompatible.")
    if args.export_member_predictions and not args.member_prediction_splits:
        raise ValueError("Prediction export requires at least one validation/test split.")
    if any(v is not None for v in (args.max_train_rows_per_task, args.max_validation_rows_per_task,
                                   args.max_test_rows_per_task)):
        raise ValueError("Frozen-fold policy requires complete current/validation rows; no row limits.")
    if args.total_memory_budget is not None or args.replay_draws_per_epoch is not None:
        raise ValueError("Legacy fixed 6800 budgets cannot be used in frozen-fold mode.")
    if args.batch_size not in (8, 16, 32, 64) or args.effective_batch_size != 1024:
        raise ValueError("This pilot contract uses microbatch 64 (OOM: 32/16/8), effective target 1024.")
    if args.stop_after_task is not None and args.stop_after_task not in range(8):
        raise ValueError("--stop-after-task must be between 0 and 7.")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("Epochs and patience must be positive.")
    if args.exemplar_ranking_policy not in RANKING_POLICIES:
        raise ValueError(f"Unsupported exemplar ranking policy: {args.exemplar_ranking_policy}.")
    if (args.exemplar_ranking_policy == PIPELINE_INPUT_RANKING_POLICY
            and args.preprocessing_policy != "task0_standard_frozen"):
        raise ValueError("Pipeline-input exemplar ranking requires task0_standard_frozen preprocessing.")
    for name in ("lr", "weight_decay", "distill_alpha", "temperature", "feature_distill_weight", "focal_gamma"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in ("lr", "temperature") and value == 0):
            raise ValueError(f"Invalid hyperparameter {name}.")
    if not 0 <= args.dropout < 1:
        raise ValueError("Dropout must be in [0,1).")


def validate_p3_spec(spec, context):
    tasks = spec.get("tasks", [])
    if spec.get("protocol") != "tail_to_head" or len(tasks) != 8:
        raise ValueError("Use the full P3 tail_to_head file, not a two-task smoke protocol.")
    classes = []
    for task_id, (task, width) in enumerate(zip(tasks, P3_LAYOUT, strict=True)):
        raw = task.get("classes", [])
        if type(task.get("task_id")) is not int or task["task_id"] != task_id or len(raw) != width:
            raise ValueError("P3 task IDs/layout must be 0..7 and [38,20,20,20,20,20,20,20].")
        if any(type(c) is not int for c in raw):
            raise ValueError("P3 requires integer raw class IDs.")
        classes.extend(raw)
    # Identity-only tables already validated against both source labels. No old
    # descriptors or future features are read for this protocol integrity check.
    assigned = {int(c) for table in context._rows for c in table["raw_class_id"].to_pylist()}
    if len(set(classes)) != 178 or set(classes) != assigned:
        raise ValueError("P3 must cover exactly the 178 assignment raw classes without duplicates.")
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
    prepared, *, engine, model, values, labels, metadata, seen_map, criterion,
    run_id, task_id, split, root, device, batch_size, all_oof=None,
):
    loader = DataLoader(
        engine.build_tensor_dataset(values, remap_labels(labels, seen_map)),
        batch_size=batch_size, shuffle=False, drop_last=False,
        generator=torch.Generator().manual_seed(0),
    )
    result = engine.evaluate_model(
        model, loader, criterion, device, invert_class_map(seen_map),
        evaluation_class_indices=list(range(len(seen_map))), collect_outputs=True,
    )
    if result.outputs is None:
        raise RuntimeError("Fold prediction collection did not return outputs.")
    member_id = prepared["seeds"].member_id
    artifact_path = root / "member_predictions" / f"task_{task_id}" / f"{split}.npz"
    export_member_prediction_artifact(
        result.outputs, metadata,
        MemberPredictionContext(
            run_id=run_id, method=prepared["resolved"]["method"],
            method_protocol=TRAINING_POLICY, task_id=task_id, split=split,
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


def _eval_groups(engine, model, values, labels, seen_map, new_classes, criterion, device, batch_size=64):
    results = []
    old_classes = sorted(set(seen_map) - set(new_classes))
    for group, classes in (("seen_all", sorted(seen_map)), ("old", old_classes), ("current", new_classes)):
        keep = np.isin(labels, classes)
        if not keep.any():
            results.append({"group": group, "sample_count": 0, "status": "empty"})
            continue
        loader = DataLoader(engine.build_tensor_dataset(values[keep], remap_labels(labels[keep], seen_map)),
                            batch_size=batch_size, shuffle=False, drop_last=False,
                            generator=torch.Generator().manual_seed(0))
        result = engine.evaluate_model(model, loader, criterion, device, invert_class_map(seen_map),
                                       evaluation_class_indices=[seen_map[c] for c in classes])
        results.append({"group": group, "sample_count": int(keep.sum()), "status": "ok", **result.metrics})
    return results


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
    tasks = validate_p3_spec(spec, context)
    manifest = context.manifest
    if manifest["fold_seed"] != 42:
        raise ValueError("Assignment fold seed must be 42.")
    prep_hash = fold_file_sha256(args.fold_preprocessing)
    prep = load_fold_preprocessing(args.fold_preprocessing, context=context, task_file=args.task_file,
        feature_columns=columns, member_id=args.member_id, validation_fold=args.member_id,
        policy=args.preprocessing_policy, experiment_seed=args.seed, expected_sha256=prep_hash)
    budgets = four_percent_member_budgets(manifest["assignment_row_count"])
    buffer_kwargs = dict(context=context, task_file=args.task_file, feature_columns=columns,
        member_id=args.member_id, total_memory_budget=budgets[args.member_id], base_quota=10,
        experiment_seed=args.seed, ranking_policy=args.exemplar_ranking_policy,
        ranking_preprocessing=(prep if args.exemplar_ranking_policy == PIPELINE_INPUT_RANKING_POLICY else None),
        ranking_preprocessing_sha256=(prep_hash if args.exemplar_ranking_policy == PIPELINE_INPUT_RANKING_POLICY else None))
    buffer = FoldSqrtReplayBuffer(**buffer_kwargs)
    seeds = resolve_seed_configuration(args.seed, args.member_id)
    device = engine.resolve_device(args.device)
    stop = 7 if args.stop_after_task is None else args.stop_after_task
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    resolved = {
        "training_policy": TRAINING_POLICY, "schema_version": 1, "method": args.method,
        "method_protocol": TRAINING_POLICY, "ensemble_mode": "frozen_stratified_3fold",
        "seeds": asdict(seeds), "validation_fold": args.member_id, "fold_seed": 42,
        "task_protocol": "tail_to_head", "task_file_sha256": task_hash,
        "task_layout": P3_LAYOUT, "protocol_task_count": 8, "stop_after_task": stop,
        "assignment_sha256": manifest["assignment_sha256"], "fold_manifest_sha256": context.manifest_sha256,
        "sources": buffer.metadata["sources"], "preprocessing_sha256": prep_hash,
        "preprocessing": prep.metadata, "buffer": buffer.metadata,
        "member_budgets": budgets, "global_slot_budget": sum(budgets.values()),
        "ranking_seed_role": (
            "no_rng_per_sample_normalized_class_mean_sample_ID_tie"
            if args.exemplar_ranking_policy == SAMPLE_NORMALIZED_RANKING_POLICY
            else "no_rng_frozen_preprocessed_input_class_mean_sample_ID_tie"
        ),
        "microbatch": args.batch_size, "gradient_accumulation": 1024 // args.batch_size, "effective_batch_target": 1024,
        "drop_last": False, "optimizer": "AdamW_new_each_task", "scheduler": None,
        "classification_weight": 1.0, "logit_distillation_weight": args.distill_alpha,
        "feature_distillation_weight": args.feature_distill_weight, "temperature": args.temperature,
        "loss_scope": "focal_and_old_column_KL_T2_and_latent_MSE_on_all_current_plus_replay_draws",
        "early_stopping": "held_out_all_seen_macro_f1", "validation_only": args.validation_only,
        "test_policy": "integrity_hash_and_pair_IDs_only" if args.validation_only else "report_only_after_best_validation",
        "prediction_export": {
            "enabled": bool(args.export_member_predictions),
            "splits": list(args.member_prediction_splits) if args.export_member_predictions else [],
            "schema": "member_prediction_v3_assignment_bound",
            "oof_policy": "concatenate_one_held_out_fold_prediction_per_sample",
        },
        "checkpoint_policy": "immutable_frozen_fold_task_boundary_v1",
        "model": paper_member_manifest(dropout=args.dropout, activation=args.activation), "device": device,
        "implementation_sha256": {name: fold_file_sha256(Path(__file__).parents[1] / name) for name in (
            "training/fold_pilot_training.py", "training/train_cil.py", "data/ddi_dataset.py",
            "data/fold_preprocessing.py", "data/fold_replay_buffer.py", "data/fold_replay_sampler.py")},
    }
    # Scope/path relocation may change; learning/provenance contract may not.
    contract = {k: v for k, v in resolved.items() if k != "stop_after_task"}
    contract["hyperparameters"] = {k: getattr(args, k) for k in (
        "epochs", "patience", "lr", "weight_decay", "dropout", "activation", "norm", "focal_gamma",
        "distill_alpha", "temperature", "feature_distill_weight")}
    contract["sampler"] = {"policy": SAMPLER_POLICY, "schema_version": 1, "rng_derivation": RNG_DERIVATION,
                           "replay_fraction": 0.125, "repeat_cap": 3, "task_boundary_epoch_reset": 0}
    contract = json.loads(_json(contract))
    return dict(columns=columns, tasks=tasks, task_hash=task_hash, context=context, manifest=manifest,
                prep=prep, prep_hash=prep_hash, budgets=budgets, buffer_kwargs=buffer_kwargs, buffer=buffer,
                seeds=seeds, device=device, stop=stop, arguments=arguments, resolved=resolved, contract=contract)


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
    logger.log(f"Frozen-fold training started policy={TRAINING_POLICY} member={args.member_id} stop_after_task={stop}")
    set_configured_seeds(seeds)
    previous, previous_map = None, {}
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
        current = load_development_fold_arrays(context, columns, role="train", member_id=args.member_id,
            validation_fold=args.member_id, class_ids=new_classes)
        validation = load_development_fold_arrays(context, columns, role="validation", member_id=args.member_id,
            validation_fold=args.member_id, class_ids=list(seen_map))
        retained = buffer.get_all()
        if not len(current.labels) or not len(validation.labels):
            raise ValueError("Empty current or held-out all-seen view.")
        if set(retained.labels) - set(previous_map):
            raise ValueError("Replay contains non-old classes.")
        if set(_ids(current) + _ids(retained)) & set(_ids(validation)):
            raise ValueError("Training/replay overlaps held-out validation IDs.")
        sampler = FoldReplayFractionSampler(current_sample_ids=_ids(current), replay_sample_ids=_ids(retained),
            replay_raw_labels=retained.labels, experiment_seed=args.seed, member_id=args.member_id, task_id=task_id)
        _publish_json(task_root / "input_audit.json", {"task_id": task_id, "current_ids": _ids(current),
            "replay_ids_before": _ids(retained), "replay_raw_labels_before": retained.labels.tolist(),
            "validation_ids": _ids(validation),
            "current_raw_classes": new_classes, "seen_class_map": seen_map,
            "head_size": len(seen_map), "sampler": sampler.metadata,
            "assignment_sha256": manifest["assignment_sha256"], "preprocessing_sha256": prep_hash,
            "buffer_policy": buffer.metadata["policy"], "ranking": buffer.metadata["ranking"]})
        current_inputs = _model_values(current.features, prep, columns, args.member_id)
        replay_inputs = _model_values(retained.features, prep, columns, args.member_id)
        val_inputs = _model_values(validation.features, prep, columns, args.member_id)
        training = engine.build_tensor_dataset(np.concatenate((current_inputs, replay_inputs)),
            remap_labels(np.concatenate((current.labels, retained.labels)), seen_map))
        loader = DataLoader(training, batch_size=args.batch_size, sampler=sampler, drop_last=False,
                            generator=torch.Generator().manual_seed(seeds.member_seed + task_id))
        val_loader = DataLoader(engine.build_tensor_dataset(val_inputs, remap_labels(validation.labels, seen_map)),
                               batch_size=args.batch_size, shuffle=False, drop_last=False,
                               generator=torch.Generator().manual_seed(seeds.member_seed + task_id))
        del current_inputs, replay_inputs
        model = engine.expand_model_for_seen_classes(previous, previous_map, seen_map,
            variant=args.variant, input_dim=len(columns), dropout=args.dropout,
            activation=args.activation, norm=args.norm).to(device)
        teacher = previous.to(device).eval().requires_grad_(False) if previous is not None else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = engine.FocalLoss(args.focal_gamma)
        best_score, best_state, best_epoch, stale = -float("inf"), None, None, 0
        epoch_rows = []
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            losses = engine.train_one_epoch(model, loader, optimizer, criterion, device,
                teacher_model=teacher, teacher_raw_classes=engine.ordered_raw_classes(previous_map),
                current_seen_map=seen_map, distill_alpha=args.distill_alpha, temperature=args.temperature,
                feature_distill_weight=args.feature_distill_weight, gradient_accumulation_steps=1024 // args.batch_size,
                return_loss_components=True)
            audit = sampler.last_audit
            if (audit is None or audit["epoch"] != epoch or audit["current_draws"] != len(current.labels)
                    or audit["max_repeat"] > 3 or audit["actual_replay_draws"] !=
                    (0 if task_id == 0 else min(len(current.labels) // 7, 3 * len(retained.labels)))
                    or losses.examples_seen != len(sampler)
                    or losses.optimizer_steps != math.ceil(len(sampler) / 1024)
                    or losses.tail_effective_batch != (len(sampler) - 1) % 1024 + 1):
                raise RuntimeError("Runtime sampler/current coverage/cap/optimizer-step audit mismatch.")
            result = engine.evaluate_model(model, val_loader, criterion, device, invert_class_map(seen_map),
                                           evaluation_class_indices=list(range(len(seen_map))))
            row = {"task": task_id, "epoch": epoch + 1, **asdict(losses),
                   "classification_weight": 1.0, "logit_distillation_weight": args.distill_alpha,
                   "feature_distillation_weight": args.feature_distill_weight, "temperature": args.temperature,
                   "validation_macro_f1": result.metrics["macro_f1"],
                   "validation_balanced_accuracy": result.metrics["balanced_accuracy"],
                   "validation_loss": result.metrics["loss"], "seconds": time.perf_counter() - epoch_start}
            if not all(math.isfinite(v) for v in row.values()):
                raise FloatingPointError("Nonfinite training/validation metrics; pilot is not accepted.")
            epoch_rows.append(row)
            _publish_json(task_root / f"epoch_{epoch + 1}_audit.json", {**row, "sampling": audit})
            logger.log_event("epoch", f"task={task_id} epoch={epoch + 1} classification_loss={losses.classification_loss:.6f} "
                f"logit_distillation_loss={losses.logit_distillation_loss:.6f} "
                f"feature_distillation_loss={losses.feature_distillation_loss:.6f} total_loss={losses.total_loss:.6f} "
                f"val_macro_f1={row['validation_macro_f1']:.6f} steps={losses.optimizer_steps}", _json(row))
            score = result.metrics["macro_f1"]
            if score > best_score:
                best_score, best_epoch, stale = score, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        # Model-only inference snapshot, deliberately not a resumable checkpoint.
        atomic_publish_fold_file(task_root / "best_model.pt", lambda handle:
            torch.save({"model_state": best_state, "seen_class_map": seen_map, "task_id": task_id,
                        "run_id": run_id, "resumable": False, "policy": TRAINING_POLICY}, handle))
        metrics = [{"task": task_id, "split": "validation", **m} for m in
                   _eval_groups(engine, model, val_inputs, validation.labels, seen_map, new_classes, criterion, device, args.batch_size)]
        prediction_paths = []
        if args.export_member_predictions and "validation" in args.member_prediction_splits:
            all_oof = load_development_fold_identity(
                context, role="all", member_id=args.member_id,
                validation_fold=args.member_id, class_ids=list(seen_map),
            )
            prediction_paths.append(_export_fold_prediction(
                prepared, engine=engine, model=model, values=val_inputs,
                labels=validation.labels, metadata=validation.metadata,
                seen_map=seen_map, criterion=criterion, run_id=run_id,
                task_id=task_id, split="validation", root=root, device=device,
                batch_size=args.batch_size, all_oof=all_oof,
            ))
        if not args.validation_only:
            context.assert_unchanged()
            frame = load_split_frame(
                args.test, columns, class_ids=list(seen_map),
                include_metadata=True, meta_cols=DEFAULT_META_COLS,
            )
            test_values = _model_values(frame[columns].to_numpy(dtype=np.float64), prep, columns, args.member_id)
            test_labels = frame["class"].to_numpy(dtype=np.int64)
            metrics.extend({"task": task_id, "split": "test", **m} for m in
                _eval_groups(engine, model, test_values, test_labels, seen_map, new_classes, criterion, device, args.batch_size))
            if args.export_member_predictions and "test" in args.member_prediction_splits:
                test_metadata = {column: frame[column].to_numpy(copy=False) for column in DEFAULT_META_COLS}
                test_metadata["sample_id"] = np.char.add(
                    np.char.add(test_metadata[DEFAULT_META_COLS[0]].astype(np.str_), "|"),
                    test_metadata[DEFAULT_META_COLS[1]].astype(np.str_),
                )
                test_metadata["fold_id"] = np.full(len(frame), -1, dtype=np.int16)
                prediction_paths.append(_export_fold_prediction(
                    prepared, engine=engine, model=model, values=test_values,
                    labels=test_labels, metadata=test_metadata, seen_map=seen_map,
                    criterion=criterion, run_id=run_id, task_id=task_id,
                    split="test", root=root, device=device,
                    batch_size=args.batch_size,
                ))
            del frame, test_values
        buffer_audit = buffer.update(current, task_id=task_id, feature_columns=columns)
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
        boundary_path = save_fold_replay_checkpoint(root / "checkpoints" / f"task_{task_id}.pt", run_id=run_id,
            completed_task_id=task_id, model_state=best_state, seen_class_map=seen_map, contract=contract,
            buffer=buffer, sampler=sampler, run_config_sha256=config_hash, artifact_hashes=artifact_hashes,
            progress={"metric_rows": all_metrics, "epoch_rows": all_epochs, "task_summaries": task_summaries})
        logger.log(f"task={task_id} complete head={len(seen_map)} best_epoch={best_epoch} retained={buffer.total_size} "
                   f"checkpoint={boundary_path} checkpoint_bytes={boundary_path.stat().st_size}")
        previous, previous_map = model.cpu(), seen_map
        del teacher, optimizer, model, best_state, current, validation, retained, after, training, loader, val_loader, val_inputs
    report_root = root if not resume_path else root / "reports" / f"through_task_{stop}_{uuid.uuid4().hex}"
    report_root.mkdir(parents=True, exist_ok=True)
    _publish_csv(report_root / "metrics.csv", all_metrics)
    _publish_csv(report_root / "training_audit.csv", all_epochs)
    _publish_json(report_root / "run_summary.json", {"run_id": run_id, "completed_task_id": stop,
        "full_protocol_complete": stop == 7, "task_file_sha256": task_hash, "tasks": task_summaries,
        "requested_scope_complete": True, "validation_only": args.validation_only, "resumable": True})
    with (report_root / "run_summary.md").open("x", encoding="utf-8") as handle:
        handle.write(f"# Frozen-fold pilot\n\nPolicy: `{TRAINING_POLICY}`. Member {args.member_id}. "
                     f"Preprocessing: `{args.preprocessing_policy}`.\n\nCompleted tasks 0–{stop} of full P3 (8 tasks). "
                     f"Validation-only: {args.validation_only}. Resume via checkpoints/task_{stop}.pt; not a legacy checkpoint.\n\n")
        for s in task_summaries:
            handle.write(f"- Task {s['task_id']}: head {s['head_size']}, best epoch {s['best_epoch']}, "
                         f"validation Macro-F1 {s['best_validation_macro_f1']:.6f}.\n")
    logger.log(f"Requested execution scope complete; report={report_root}; no preprocessing winner selected.")
    return task_summaries
