"""Expert training and causal merging for Task Arithmetic, TIES, OPCM, and MagMax.

The ordinary frozen-fold trainer is causal/sequential.  Model merging needs a
different trajectory: Task Arithmetic/TIES use task-0 as their common base,
while paper-faithful OPCM creates task experts from one shared initialization
and merges them online. MagMax fine-tunes sequentially and merges online after
each task. Only backbone tensors are merged; fixed-head rows are stitched by
ownership.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.class_mapping import build_global_class_map, invert_class_map, remap_labels
from src.data.ddi_dataset import (
    DEFAULT_META_COLS,
    DDIBatchArrays,
    load_development_fold_arrays,
    load_feature_columns,
    load_split_frame,
    prepare_development_fold_context,
)
from src.data.sample_identity import build_stable_sample_ids
from src.data.fold_preprocessing import load_fold_preprocessing
from src.data.source_shard_cache import load_source_shard
from src.data.stratified_folds import fold_file_sha256
from src.eval.metrics import compute_classification_metrics
from src.methods.model_merging import (
    ModelMergingRecipe,
    MAGMAX_METHOD,
    OPCM_METHOD,
    TASK_ARITHMETIC_METHOD,
    TIES_METHOD,
    _state_vector_norm,
    inspect_neural_merge_scope,
    merge_magmax_task_boundary,
    merge_opcm_task_boundary,
    merge_task_boundary,
)
from src.training.fold_training import _model_values, validate_fold_task_spec
from src.utils.seed import resolve_seed_configuration, set_configured_seeds


def _log(message: str) -> None:
    """Flush progress immediately so nohup/main.log is useful while running."""

    print(f"[MODEL-MERGING] {message}", flush=True)


def _dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_task_arrays(args, *, context, prep, columns, task_classes, role: str, source_shard_root: Path | None = None):
    source_kind = "development_parquet"
    if role == "test":
        # Test is deliberately outside DevelopmentFoldContext.  That context
        # only owns the train/validation frozen folds, so passing role="test"
        # to load_development_fold_arrays would (correctly) raise.  Load the
        # immutable test split through the ordinary split loader and attach
        # the same identity metadata used by fold evaluation/export.
        source_kind = "test_parquet"
        frame = load_split_frame(
            args.test,
            columns,
            class_ids=list(task_classes),
            include_metadata=True,
            meta_cols=DEFAULT_META_COLS,
        )
        metadata = {
            column: frame[column].to_numpy(copy=False)
            for column in DEFAULT_META_COLS
            if column in frame.columns
        }
        metadata["sample_id"] = build_stable_sample_ids(
            {
                "drugid-drug_a": metadata["drugid-drug_a"],
                "drugid-drug_b": metadata["drugid-drug_b"],
            },
            len(frame),
        )
        metadata["source_split"] = np.full(len(frame), "test", dtype=np.str_)
        metadata["source_row_index"] = np.arange(len(frame), dtype=np.int64)
        metadata["fold_id"] = np.full(len(frame), -1, dtype=np.int16)
        arrays = DDIBatchArrays(
            features=frame[columns].to_numpy(dtype=np.float64, copy=False),
            labels=frame["class"].to_numpy(dtype=np.int64, copy=False),
            metadata=metadata,
        )
    elif source_shard_root is not None and role in {"train", "validation"}:
        source_kind = "source_shard_cache"
        arrays = load_source_shard(
            source_shard_root,
            member_id=args.member_id,
            task_id=args._merging_task_id,
            validation_fold=args.member_id,
            role=role,
            feature_columns=columns,
        )
        if set(np.unique(arrays.labels).tolist()) != set(int(c) for c in task_classes):
            raise ValueError("Source shard labels do not match the requested task classes.")
    else:
        arrays = load_development_fold_arrays(
            context,
            columns,
            role=role,
            member_id=args.member_id,
            validation_fold=args.member_id,
            class_ids=list(task_classes),
        )
    values = _model_values(arrays.features, prep, columns, args.member_id)
    _log(
        f"loaded role={role} task={getattr(args, '_merging_task_id', '?')} "
        f"rows={len(arrays.labels)} classes={len(np.unique(arrays.labels))} "
        f"source={source_kind} transformed_shape={tuple(values.shape)}"
    )
    return values, arrays.labels.astype(np.int64, copy=False), arrays.metadata


def _owned_optimizer_step(model, optimizer, owned_rows: list[int]):
    """Step AdamW while restoring every fixed-head row outside the expert task."""

    head = model.classifier_head
    all_rows = torch.arange(head.out_features, device=head.weight.device)
    owned = torch.as_tensor(owned_rows, dtype=torch.long, device=head.weight.device)
    mask = torch.ones(head.out_features, dtype=torch.bool, device=head.weight.device)
    mask[owned] = False
    protected = all_rows[mask]
    weight = head.weight.detach()[protected].clone()
    bias = None if head.bias is None else head.bias.detach()[protected].clone()
    optimizer.step()
    with torch.no_grad():
        head.weight.index_copy_(0, protected, weight)
        if head.bias is not None and bias is not None:
            head.bias.index_copy_(0, protected, bias)


def _evaluate(engine, model, values, labels, *, global_map, active_rows, device, batch_size):
    inverse = invert_class_map(global_map)
    loader = DataLoader(
        engine.build_tensor_dataset(values, remap_labels(labels, global_map)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    result = engine.evaluate_model(
        model,
        loader,
        engine.nn.CrossEntropyLoss(),
        device,
        inverse,
        evaluation_class_indices=active_rows,
        collect_outputs=True,
        collect_latent=False,
        active_class_indices=active_rows,
    )
    if result.outputs is None:
        raise RuntimeError("Model-merging evaluation did not return outputs.")
    return result.metrics, result.outputs


def _train_expert(
    args,
    *,
    task_id: int,
    engine,
    initial_state: dict[str, torch.Tensor] | None,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    global_map: dict[int, int],
    owned_rows: list[int],
    device: str,
):
    started = time.perf_counter()
    _log(
        f"expert_start task={task_id} rows={len(train_labels)} "
        f"validation_rows={len(validation_labels)} owned_classes={len(owned_rows)} "
        f"initial_state_loaded={initial_state is not None}"
    )
    model = engine.expand_model_for_seen_classes(
        None,
        None,
        global_map,
        variant=args.variant,
        input_dim=train_values.shape[1],
        dropout=args.dropout,
        activation=args.activation,
        norm=args.norm,
        method=None,
        active_class_indices=owned_rows,
    ).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    optimizer = engine.make_optimizer(model, args)
    criterion = engine.nn.CrossEntropyLoss()
    labels = remap_labels(train_labels, global_map)
    loader = DataLoader(
        engine.build_tensor_dataset(train_values, labels),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(args.seed) + args.member_id + len(owned_rows)),
    )
    best_state = _state_cpu(model)
    best_score = -float("inf")
    stale = 0
    epoch_rows: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        loss = engine.train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
            loss_variant="cross_entropy",
            active_class_indices=owned_rows,
            gradient_accumulation_steps=max(1, args.effective_batch_size // args.batch_size),
            optimizer_step=lambda: _owned_optimizer_step(model, optimizer, owned_rows),
        )
        metrics, _ = _evaluate(
            engine,
            model,
            validation_values,
            validation_labels,
            global_map=global_map,
            active_rows=owned_rows,
            device=device,
            batch_size=args.evaluation_batch_size,
        )
        row = {"epoch": epoch, "train_loss": float(loss), "val_macro_f1": float(metrics["macro_f1"])}
        epoch_rows.append(row)
        if row["val_macro_f1"] > best_score:
            best_score = float(row["val_macro_f1"])
            best_state = _state_cpu(model)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                _log(
                    f"expert_early_stop task={task_id} epoch={epoch} "
                    f"best_val_macro_f1={best_score:.6f} patience={args.patience}"
                )
                break
        _log(
            f"expert_epoch task={task_id} epoch={epoch}/{args.epochs} "
            f"train_loss={float(loss):.6f} val_macro_f1={float(metrics['macro_f1']):.6f} "
            f"best_val_macro_f1={best_score:.6f} elapsed_s={time.perf_counter() - started:.1f}"
        )
    model.load_state_dict(best_state, strict=True)
    _log(
        f"expert_complete task={task_id} epochs={len(epoch_rows)} "
        f"best_val_macro_f1={best_score:.6f} elapsed_s={time.perf_counter() - started:.1f}"
    )
    return model, best_state, epoch_rows


def run_model_merging_member(args, *, engine) -> None:
    if args.method not in {TASK_ARITHMETIC_METHOD, TIES_METHOD, OPCM_METHOD, MAGMAX_METHOD}:
        raise ValueError(f"Unsupported model-merging method: {args.method}")
    root = Path(args.outdir)
    if root.exists():
        raise FileExistsError(f"Model-merging output already exists; never overwrite: {root}")
    root.mkdir(parents=True)
    set_configured_seeds(resolve_seed_configuration(args.seed, args.member_id))
    device = engine.resolve_device(args.device)
    columns = load_feature_columns(args.feature_cols)
    _log(
        f"member_start member={args.member_id} method={args.method} device={device} "
        f"feature_count={len(columns)}"
    )
    context = prepare_development_fold_context(
        args.fold_assignments,
        args.fold_manifest,
        source_paths={"train": args.train, "validation": args.validation, "test": args.test},
    )
    tasks = validate_fold_task_spec(engine.load_task_spec(args.task_file), context)
    global_map = build_global_class_map(raw for task in tasks for raw in task["classes"])
    _log(
        f"fold_ready member={args.member_id} tasks={len(tasks)} "
        f"global_classes={len(global_map)} validation_fold={args.member_id}"
    )
    prep_hash = fold_file_sha256(args.fold_preprocessing)
    prep = load_fold_preprocessing(
        args.fold_preprocessing,
        context=context,
        task_file=args.task_file,
        feature_columns=columns,
        member_id=args.member_id,
        validation_fold=args.member_id,
        policy=args.preprocessing_policy,
        experiment_seed=args.seed,
        expected_sha256=prep_hash,
    )
    source_shard_root = (
        Path(args.source_shard_cache_root).resolve()
        if getattr(args, "source_shard_cache_root", None) is not None
        and Path(args.source_shard_cache_root).is_dir()
        else None
    )
    _log(
        f"preprocessing_ready policy={args.preprocessing_policy} "
        f"cache={'source_shard' if source_shard_root is not None else 'direct_parquet'} "
        f"prep_sha256={prep_hash[:12]}"
    )
    if args.method == TASK_ARITHMETIC_METHOD:
        recipe = ModelMergingRecipe.paper_task_arithmetic()
    elif args.method == TIES_METHOD:
        recipe = ModelMergingRecipe.paper_ties()
    elif args.method == OPCM_METHOD:
        recipe = ModelMergingRecipe.paper_opcm()
    else:
        recipe = ModelMergingRecipe.paper_magmax()
    opcm_mode = args.method == OPCM_METHOD
    magmax_mode = args.method == MAGMAX_METHOD
    _dump_json(root / "run_config.json", {
        "method": args.method,
        "method_policy": (
            "frozen_fold_fixed_head_opcm_sequential_paper_v1"
            if opcm_mode
            else "frozen_fold_fixed_head_magmax_sequential_paper_v1"
            if magmax_mode
            else f"frozen_fold_fixed_head_{args.method}_independent_experts_v1"
        ),
        "protocol": "head_to_tail",
        "task_layout": [len(task["classes"]) for task in tasks],
        "head_policy": "task_owned_fixed_head_rows_v1",
        "merge_scope": "learned_backbone_parameters_only_v1",
        "expert_generation": (
            "sequential_finetuning_from_previous_task"
            if magmax_mode
            else "independent_task_experts_from_shared_initialization"
            if opcm_mode
            else "independent_task_experts_from_task0_checkpoint"
        ),
        "merge_trajectory": (
            "sequential_magmax" if magmax_mode
            else "sequential_opcm" if opcm_mode
            else "causal_boundary_recompute"
        ),
        "execution_schedule": (
            "online_train_current_expert_then_merge" if opcm_mode
            else "online_sequential_finetune_then_magmax_merge" if magmax_mode
            else "train_experts_then_causal_boundary_recompute"
        ),
        "pretrained_reference": "model_initialization_before_task0" if (opcm_mode or magmax_mode) else None,
        "recipe": recipe.manifest(),
        "experiment_seed": args.seed,
        "member_id": args.member_id,
        "fold_seed": args.fold_seed,
        "preprocessing_policy": args.preprocessing_policy,
        "preprocessing_sha256": prep_hash,
        "task_file_sha256": fold_file_sha256(args.task_file),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    train_data: dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    validation_data: dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    (root / "experts").mkdir(parents=True, exist_ok=True)
    sample_counts: dict[int, int] = {}
    if opcm_mode or magmax_mode:
        # Both paper-faithful sequential methods expose only the current task
        # to the training loop.  Future train/validation arrays are loaded at
        # the corresponding boundary, immediately before that task is trained.
        _log("online_task_data_ready policy=current_task_only")
    else:
        for task in tasks:
            task_id = int(task["task_id"])
            args._merging_task_id = task_id
            train_data[task_id] = _load_task_arrays(
                args, context=context, prep=prep, columns=columns,
                task_classes=task["classes"], role="train", source_shard_root=source_shard_root,
            )
            validation_data[task_id] = _load_task_arrays(
                args, context=context, prep=prep, columns=columns,
                task_classes=task["classes"], role="validation", source_shard_root=source_shard_root,
            )
            sample_counts[task_id] = int(len(train_data[task_id][1]))
        _log(
            "all_task_data_ready "
            + ", ".join(
                f"task_{task_id}=train:{len(train_data[task_id][1])}/val:{len(validation_data[task_id][1])}"
                for task_id in range(len(tasks))
            )
        )
    pretrained_reference_state: dict[str, torch.Tensor] | None = None
    if opcm_mode or magmax_mode:
        reference_model = engine.expand_model_for_seen_classes(
            None, None, global_map,
            variant=args.variant, input_dim=len(columns), dropout=args.dropout,
            activation=args.activation, norm=args.norm, method=None,
            active_class_indices=list(range(len(global_map))),
        )
        pretrained_reference_state = _state_cpu(reference_model)
        _log("shared_reference_ready initialization_before_task0=True")

    experts: dict[int, dict[str, torch.Tensor]] = {}
    epochs_by_task: dict[str, object] = {}
    task_states: dict[int, dict[str, torch.Tensor]] = {}
    if opcm_mode:
        # Expert training is intentionally deferred to the online boundary
        # loop below: train task t, merge it, then expose task t+1.
        base_state = None
    elif magmax_mode:
        # MagMax expert training is deferred to the online boundary loop below.
        # The next task starts from the raw sequential checkpoint, while the
        # merged model is produced for evaluation/serving at each boundary.
        assert pretrained_reference_state is not None
        base_state = None
        sequential_state: dict[str, torch.Tensor] | None = pretrained_reference_state
    else:
        _, base_state, base_epochs = _train_expert(
            args,
            task_id=0,
            engine=engine,
            initial_state=None,
            train_values=train_data[0][0], train_labels=train_data[0][1],
            validation_values=validation_data[0][0], validation_labels=validation_data[0][1],
            global_map=global_map,
            owned_rows=[global_map[c] for c in tasks[0]["classes"]],
            device=device,
        )
        epochs_by_task["0"] = base_epochs
        torch.save({"model_state": base_state, "task_id": 0}, root / "experts" / "task_0.pt")
        _log("expert_saved task=0 path=experts/task_0.pt")
        for task in tasks[1:]:
            task_id = int(task["task_id"])
            owned_rows = [global_map[c] for c in task["classes"]]
            _, expert_state, epoch_rows = _train_expert(
                args,
                task_id=task_id,
                engine=engine,
                initial_state=base_state,
                train_values=train_data[task_id][0], train_labels=train_data[task_id][1],
                validation_values=validation_data[task_id][0], validation_labels=validation_data[task_id][1],
                global_map=global_map,
                owned_rows=owned_rows,
                device=device,
            )
            experts[task_id] = expert_state
            epochs_by_task[str(task_id)] = epoch_rows
            torch.save({"model_state": expert_state, "task_id": task_id, "owned_rows": owned_rows}, root / "experts" / f"task_{task_id}.pt")
            _log(f"expert_saved task={task_id} path=experts/task_{task_id}.pt")

    args._merging_task_id = 0
    test_values, test_labels, test_metadata = _load_task_arrays(
        args, context=context, prep=prep, columns=columns,
        task_classes=list(global_map), role="test",
    )
    _log(f"test_ready rows={len(test_labels)}")
    rows_by_task = {
        int(task["task_id"]): [global_map[c] for c in task["classes"]]
        for task in tasks
    }
    metrics_rows: list[dict[str, object]] = []
    merged_dir = root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    opcm_previous_state: dict[str, torch.Tensor] | None = None
    opcm_previous_lambda = 1.0
    opcm_task_vector_norms: dict[int, float] = {}
    for boundary in range(len(tasks)):
        if opcm_mode or magmax_mode:
            # Paper-faithful online schedule: train only the current task,
            # merge it immediately, evaluate the boundary, then expose t+1.
            task_id = boundary
            task = tasks[task_id]
            args._merging_task_id = task_id
            train_data[task_id] = _load_task_arrays(
                args, context=context, prep=prep, columns=columns,
                task_classes=task["classes"], role="train", source_shard_root=source_shard_root,
            )
            validation_data[task_id] = _load_task_arrays(
                args, context=context, prep=prep, columns=columns,
                task_classes=task["classes"], role="validation", source_shard_root=source_shard_root,
            )
            sample_counts[task_id] = int(len(train_data[task_id][1]))
            owned_rows = [global_map[c] for c in task["classes"]]
            if opcm_mode:
                assert pretrained_reference_state is not None
                initial_state = pretrained_reference_state
            else:
                assert sequential_state is not None
                initial_state = sequential_state
            _, expert_state, epoch_rows = _train_expert(
                args,
                task_id=task_id,
                engine=engine,
                initial_state=initial_state,
                train_values=train_data[task_id][0], train_labels=train_data[task_id][1],
                validation_values=validation_data[task_id][0], validation_labels=validation_data[task_id][1],
                global_map=global_map,
                owned_rows=owned_rows,
                device=device,
            )
            task_states[task_id] = expert_state
            epochs_by_task[str(task_id)] = epoch_rows
            if magmax_mode:
                # Keep the raw sequential checkpoint for the next task;
                # MagMax's merged model is not the fine-tuning initializer.
                sequential_state = expert_state
            torch.save(
                {"model_state": expert_state, "task_id": task_id, "owned_rows": owned_rows},
                root / "experts" / f"task_{task_id}.pt",
            )
            _log(
                f"expert_saved task={task_id} path=experts/task_{task_id}.pt "
                f"online_before_merge=True method={args.method}"
            )
            if base_state is None:
                base_state = expert_state
        _log(
            f"merge_start boundary={boundary} "
            f"experts_used={boundary + 1 if (opcm_mode or magmax_mode) else boundary} "
            f"seen_classes={sum(len(task['classes']) for task in tasks[:boundary + 1])}"
        )
        model = engine.expand_model_for_seen_classes(
            None, None, global_map,
            variant=args.variant, input_dim=len(columns), dropout=args.dropout,
            activation=args.activation, norm=args.norm, method=None,
            active_class_indices=[global_map[c] for task in tasks[: boundary + 1] for c in task["classes"]],
        ).to(device)
        if opcm_mode:
            assert pretrained_reference_state is not None
            if boundary == 0:
                # ``task_states`` already stores CPU state-dicts.  Do not call
                # _state_cpu here: that helper is for nn.Module instances.
                merged_state = {
                    key: value.detach().cpu().clone()
                    for key, value in task_states[0].items()
                }
                opcm_previous_state = merged_state
                opcm_previous_lambda = 1.0
                scope = inspect_neural_merge_scope(model)
                opcm_task_vector_norms[0] = float(
                    _state_vector_norm(
                        pretrained_reference_state,
                        task_states[0],
                        scope.backbone_parameter_keys,
                    )
                )
                merge_audit = {
                    "artifact_kind": "ddi_cil_sequential_opcm_boundary",
                    "base_task_id": 0,
                    "upto_task_id": 0,
                    "expert_task_ids": [0],
                    "future_experts_used": False,
                    "recipe": recipe.manifest(),
                    "scope": scope.manifest(),
                    "head_policy": "task_owned_fixed_head_rows_v1",
                    "seen_head_rows": rows_by_task[0],
                    "batchnorm_policy": "expert_task0_state",
                    "projection": {
                        "applied": False,
                        "reason": "task_0_is_the_first_sequential_expert",
                        "task_vector_norm": opcm_task_vector_norms[0],
                    },
                }
                model.load_state_dict(merged_state, strict=True)
            else:
                assert opcm_previous_state is not None
                scope = inspect_neural_merge_scope(model)
                # Online OPCM only knows the current expert at this point.
                # Register its task-vector norm before constructing the
                # boundary average; future-task norms must not be required.
                opcm_task_vector_norms[boundary] = float(
                    _state_vector_norm(
                        pretrained_reference_state,
                        task_states[boundary],
                        scope.backbone_parameter_keys,
                    )
                )
                opcm_result = merge_opcm_task_boundary(
                    model,
                    pretrained_state=pretrained_reference_state,
                    previous_state=opcm_previous_state,
                    task_state=task_states[boundary],
                    task_states_by_task={task_id: task_states[task_id] for task_id in range(boundary + 1)},
                    rows_by_task=rows_by_task,
                    train_sample_counts_by_task={task_id: sample_counts[task_id] for task_id in range(boundary + 1)},
                    task_vector_norms_by_task={task_id: opcm_task_vector_norms[task_id] for task_id in range(boundary + 1)},
                    upto_task_id=boundary,
                    previous_lambda=opcm_previous_lambda,
                    recipe=recipe,
                )
                merged_state = opcm_result.state_dict
                merge_audit = opcm_result.audit
                opcm_previous_state = merged_state
                opcm_previous_lambda = opcm_result.lambda_t
                _log(
                    f"opcm_projection boundary={boundary} lambda_t={opcm_previous_lambda:.6f} "
                    f"avg_task_vector_norm={merge_audit['projection']['average_task_vector_norm']:.6f}"
                )
        elif magmax_mode:
            assert pretrained_reference_state is not None
            result = merge_magmax_task_boundary(
                model,
                pretrained_state=pretrained_reference_state,
                task_states_by_task={task_id: task_states[task_id] for task_id in range(boundary + 1)},
                rows_by_task=rows_by_task,
                train_sample_counts_by_task={task_id: sample_counts[task_id] for task_id in range(boundary + 1)},
                upto_task_id=boundary,
                recipe=recipe,
            )
            merged_state = result.state_dict
            merge_audit = result.audit
            _log(
                f"magmax_selection boundary={boundary} tasks_used={boundary + 1} "
                f"scale={recipe.scale:.3f}"
            )
        else:
            result = merge_task_boundary(
                model,
                base_state=base_state,
                experts_by_task={task_id: experts[task_id] for task_id in range(1, boundary + 1)},
                rows_by_task=rows_by_task,
                train_sample_counts_by_task={task_id: sample_counts[task_id] for task_id in range(boundary + 1)},
                upto_task_id=boundary,
                recipe=recipe,
            )
            merged_state = result.state_dict
            merge_audit = result.audit
        seen_rows = [row for task_id in range(boundary + 1) for row in rows_by_task[task_id]]
        seen_raw = [raw for raw, index in global_map.items() if index in set(seen_rows)]
        keep = np.isin(test_labels, np.asarray(seen_raw, dtype=np.int64))
        metric, outputs = _evaluate(
            engine, model, test_values[keep], test_labels[keep],
            global_map=global_map, active_rows=sorted(seen_rows), device=device,
            batch_size=args.evaluation_batch_size,
        )
        task_row = {
            "task_id": boundary,
            "seen_class_count": len(seen_rows),
            "test_samples": int(keep.sum()),
            **{key: float(value) for key, value in metric.items()},
        }
        metrics_rows.append(task_row)
        task_dir = merged_dir / f"task_{boundary}"
        task_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": merged_state, "task_id": boundary, "audit": merge_audit}, task_dir / "model.pt")
        _dump_json(task_dir / "merge_audit.json", merge_audit)
        np.savez_compressed(
            task_dir / "test_predictions.npz",
            sample_id=np.asarray(test_metadata["sample_id"])[keep],
            labels=outputs.labels,
            predictions=outputs.predictions,
            probabilities=outputs.probabilities,
            class_ids=outputs.class_ids,
        )
        _log(
            f"merge_complete boundary={boundary} test_samples={int(keep.sum())} "
            f"macro_f1={float(metric['macro_f1']):.6f}"
        )
    pd.DataFrame(metrics_rows).to_csv(root / "merged_metrics.csv", index=False)
    _dump_json(root / "expert_epochs.json", epochs_by_task)
    _dump_json(root / "run_summary.json", {
        "method": args.method,
        "member_id": args.member_id,
        "final_task": len(tasks) - 1,
        "final_macro_f1": metrics_rows[-1]["macro_f1"],
        "mean_task_macro_f1": float(np.mean([row["macro_f1"] for row in metrics_rows])),
        "expert_generation": (
            "sequential_finetuning_from_previous_task" if magmax_mode
            else "independent_from_shared_initialization" if opcm_mode
            else "independent_from_task0_checkpoint"
        ),
        "sequential_merge": opcm_mode or magmax_mode,
        "execution_schedule": (
            "online_train_current_expert_then_merge" if opcm_mode
            else "online_sequential_finetune_then_magmax_merge" if magmax_mode
            else "train_experts_then_causal_boundary_recompute"
        ),
        "future_experts_used": False,
    })
    _log(f"member_complete member={args.member_id} output={root}")
    print(f"MODEL MERGING MEMBER COMPLETE: method={args.method} member={args.member_id} output={root}", flush=True)


__all__ = ["run_model_merging_member"]
