"""One frozen-fold P3 member trained with online DER++ and a fixed 178-way head."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.class_mapping import build_seen_class_map
from src.data.ddi_dataset import (
    DEFAULT_META_COLS, load_development_fold_arrays, load_development_fold_identity,
    load_feature_columns, load_split_frame, prepare_development_fold_context,
)
from src.data.fold_preprocessing import load_fold_preprocessing
from src.data.sample_identity import build_stable_sample_ids
from src.data.stratified_folds import fold_file_sha256
from src.eval.evaluation import PredictionOutputs
from src.eval.metrics import compute_classification_metrics
from src.eval.predictions import (
    MemberPredictionContext, PredictionProvenance, export_member_prediction_artifact,
    partition_identity_sha256,
)
from src.methods.derpp import ClassBalancedReservoirLogitBuffer, ReservoirLogitBuffer, observe
from src.models.tddi_paper_member import TDDIPaperMember, TDDIPaperMemberConfig
from src.utils.seed import resolve_seed_configuration, set_configured_seeds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD = "derpp"
METHOD_PROTOCOL = "p3_derpp_online_fixed_head_v1"
BALANCED_METHOD_PROTOCOL = "p3_derpp_online_equal_class_buffer_v1"
BALANCED_BUFFER_POLICY = ClassBalancedReservoirLogitBuffer.policy


def _sha(data: object) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _log(root: Path, message: str) -> None:
    row = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(row, flush=True)
    with (root / "train.log").open("a", encoding="utf-8") as stream:
        stream.write(row + "\n")


def _values(raw: np.ndarray, prep, columns: list[str], member: int) -> np.ndarray:
    transformed = prep.transform(raw, feature_columns=columns, member_id=member,
                                 policy="task0_standard_frozen")
    values = np.asarray(transformed, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(columns) or not np.isfinite(values).all():
        raise ValueError("DER++ model inputs are malformed or nonfinite.")
    return values


def _predict(model: torch.nn.Module, values: np.ndarray, labels: np.ndarray,
             metadata: dict[str, np.ndarray], seen_raw: list[int], full_map: dict[int, int],
             device: str, batch_size: int) -> tuple[PredictionOutputs, dict[str, float]]:
    """Mask future output columns for task-boundary class-incremental evaluation."""
    selected = [full_map[raw] for raw in seen_raw]
    logits_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy(values[start:start + batch_size]).to(device)
            logits = model(x)[:, selected]
            logits_chunks.append(logits.cpu().numpy().astype(np.float32, copy=False))
    scores = np.concatenate(logits_chunks)
    probabilities = F.softmax(torch.from_numpy(scores), dim=1).numpy()
    classes = np.asarray(seen_raw, dtype=np.int64)
    predicted = classes[np.argmax(scores, axis=1)]
    outputs = PredictionOutputs(scores, probabilities.astype(np.float32, copy=False),
                                predicted, np.asarray(labels, dtype=np.int64),
                                np.empty((0, 0), dtype=np.float32), classes)
    metrics = compute_classification_metrics(outputs.labels, outputs.predictions, labels=seen_raw)
    return outputs, metrics


def _groups(outputs: PredictionOutputs, new_classes: list[int], seen_classes: list[int]) -> list[dict]:
    old_classes = sorted(set(seen_classes) - set(new_classes))
    rows = []
    for group, classes in (("seen_all", seen_classes), ("old", old_classes),
                           ("current", new_classes)):
        mask = np.isin(outputs.labels, classes)
        if not mask.any():
            rows.append({"group": group, "sample_count": 0, "status": "empty"})
        else:
            rows.append({"group": group, "sample_count": int(mask.sum()), "status": "ok",
                         **compute_classification_metrics(outputs.labels[mask], outputs.predictions[mask],
                                                          labels=classes)})
    return rows


def _prediction_provenance(args, *, context, prep_hash: str, task_hash: str,
                           config_hash: str, labels: np.ndarray,
                           metadata: dict[str, np.ndarray], member: int,
                           method_protocol: str, buffer_policy: str, member_count: int,
                           all_identity=None) -> PredictionProvenance:
    ids = np.asarray(metadata["sample_id"]).astype(np.str_, copy=False)
    folds = np.asarray(metadata["fold_id"], dtype=np.int16)
    expected_oof_rows = expected_oof_sha = None
    if all_identity is not None:
        all_labels, all_meta = all_identity
        expected_oof_rows = len(all_labels)
        expected_oof_sha = partition_identity_sha256(
            all_meta["sample_id"], all_labels, all_meta["fold_id"])
    shared = {
        "method": METHOD, "method_protocol": method_protocol, "config_sha256": config_hash,
        "task_file_sha256": task_hash, "assignment_sha256": context.manifest["assignment_sha256"],
        "fold_manifest_sha256": context.manifest_sha256,
    }
    return PredictionProvenance(
        assignment_sha256=context.manifest["assignment_sha256"],
        fold_manifest_sha256=context.manifest_sha256, task_file_sha256=task_hash,
        study_contract_sha256=_sha(shared), preprocessing_policy="task0_standard_frozen",
        preprocessing_sha256=prep_hash, preprocessing_member_id=member,
        ranking_policy=("equal_class_quota_random_within_class" if buffer_policy == BALANCED_BUFFER_POLICY
                        else "reservoir_uniform_online"), buffer_policy=buffer_policy,
        member_memory_budget=args.memory_budget,
        global_memory_budget=member_count * args.memory_budget,
        expected_partition_rows=len(labels),
        expected_partition_sha256=partition_identity_sha256(ids, labels, folds),
        expected_oof_rows=expected_oof_rows, expected_oof_sha256=expected_oof_sha,
    )


def _export(args, *, root: Path, context, task_id: int, split: str, outputs: PredictionOutputs,
            metadata: dict[str, np.ndarray], run_id: str, seeds, prep_hash: str,
            task_hash: str, config_hash: str, method_protocol: str,
            buffer_policy: str, member_count: int, all_identity=None) -> Path:
    member = args.member_id
    provenance = _prediction_provenance(
        args, context=context, prep_hash=prep_hash, task_hash=task_hash,
        config_hash=config_hash, labels=outputs.labels, metadata=metadata,
        member=member, method_protocol=method_protocol, buffer_policy=buffer_policy,
        member_count=member_count, all_identity=all_identity)
    return export_member_prediction_artifact(
        outputs, metadata,
        MemberPredictionContext(run_id=run_id, method=METHOD, method_protocol=method_protocol,
                                task_id=task_id, split=split, member_id=member,
                                experiment_seed=0, member_seed=seeds.member_seed,
                                ensemble_mode="stratified_3fold", fold_id=member,
                                fold_count=3, fold_seed=42),
        root / "member_predictions" / f"task_{task_id}" / f"{split}.npz",
        provenance=provenance,
    )


def _checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_member_summary(root: Path, member_id: int, run_id: str,
                          summary_rows: list[dict]) -> None:
    if len(summary_rows) != 8 or [row["task_id"] for row in summary_rows] != list(range(8)):
        raise ValueError("DER++ member summary requires eight completed tasks.")
    _json(root / "run_summary.json", {"method": METHOD, "member_id": member_id,
           "completed_task_id": 7, "head_size": 178, "tasks": summary_rows,
           "full_protocol_complete": True, "run_id": run_id})
    pd.DataFrame(summary_rows).to_csv(root / "metrics.csv", index=False)
    (root / "run_summary.md").write_text(
        "# DER++ P3 member " + str(member_id) + "\n\n"
        + "Online one pass per task; fixed 178-way head; CE + alpha*MSE(buffer logits) + beta*CE(buffer labels).\n\n"
        + "\n".join(f"- Task {row['task_id']}: validation Macro-F1 {row['validation_macro_f1']:.6f}; "
                     f"test Macro-F1 {row['test_macro_f1']:.6f}; reservoir {row['memory_after']}."
                     for row in summary_rows) + "\n", encoding="utf-8")


def _preserve_incomplete_task(root: Path, task_id: int) -> None:
    """Keep interrupted artifacts for audit, then rerun only the incomplete task."""
    task_root = root / f"task_{task_id}"
    prediction_root = root / "member_predictions" / f"task_{task_id}"
    if not task_root.exists() and not prediction_root.exists():
        return
    attempt_root = root / "interrupted_attempts" / f"task_{task_id}_{uuid.uuid4().hex}"
    attempt_root.mkdir(parents=True)
    if task_root.exists():
        task_root.rename(attempt_root / task_root.name)
    if prediction_root.exists():
        prediction_root.rename(attempt_root / "member_predictions")
    _log(root, f"Preserved incomplete task={task_id} artifacts at {attempt_root}")


def run_member(args: argparse.Namespace) -> None:
    if args.member_id not in (0, 1, 2) or args.memory_budget != 27778:
        raise ValueError("DER++ requires member 0/1/2 and 4%=27,778 slots per member.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    balanced_pilot = config.get("kind") == "p3_derpp_equal_class_pilot"
    buffer_policy = config.get("replay", {}).get("policy")
    expected_policy = BALANCED_BUFFER_POLICY if balanced_pilot else "online_reservoir_logits_v1"
    method_protocol = BALANCED_METHOD_PROTOCOL if balanced_pilot else METHOD_PROTOCOL
    member_ids = config.get("member_ids")
    if (config.get("kind") not in {"p3_derpp_online_full8", "p3_derpp_equal_class_pilot"}
            or config.get("training", {}).get("epochs_per_task") != 1
            or buffer_policy != expected_policy
            or member_ids != ([0] if balanced_pilot else [0, 1, 2])
            or args.member_id not in member_ids):
        raise ValueError("DER++ config must select one online pass and reservoir logit memory.")
    if (config["model"].get("variant") != "tddi_paper_member"
            or config["model"].get("head") != "fixed_178_from_task0"
            or config["model"].get("input_dim") != 3780
            or config["model"].get("hidden_dims") != [7560, 7560]
            or config["model"].get("norm") != "layernorm"):
        raise ValueError("DER++ fixed-head T-DDI architecture contract changed.")
    train_cfg = config["training"]
    for key in ("alpha", "beta", "lr", "weight_decay"):
        if not math.isfinite(float(train_cfg[key])) or float(train_cfg[key]) < 0:
            raise ValueError(f"DER++ {key} is invalid.")
    batch_size = int(train_cfg["batch_size"])
    replay_batch_size = int(train_cfg["replay_batch_size"])
    if batch_size <= 0 or replay_batch_size <= 0:
        raise ValueError("DER++ batch sizes must be positive.")
    task_hash = fold_file_sha256(args.task_file)
    if task_hash != config["protocol"]["sha256"]:
        raise ValueError("P3 task schedule hash does not match the frozen config.")
    spec = json.loads(args.task_file.read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if (spec.get("protocol") != "tail_to_head" or len(tasks) != 8
            or [len(task["classes"]) for task in tasks] != [38] + [20] * 7):
        raise ValueError("Expected the frozen eight-task P3 schedule.")
    all_classes = [int(raw) for task in tasks for raw in task["classes"]]
    if len(set(all_classes)) != 178:
        raise ValueError("P3 schedule must contain 178 distinct raw classes.")
    full_map = build_seen_class_map(all_classes)
    columns = load_feature_columns(args.feature_cols)
    if len(columns) != 3780:
        raise ValueError("T-DDI member requires 3,780 descriptors.")
    context = prepare_development_fold_context(
        args.fold_assignments, args.fold_manifest,
        source_paths={"train": args.train, "validation": args.validation, "test": args.test})
    prep_path = args.preprocessing_root / f"member_{args.member_id}" / "B" / "fold_preprocessing.json"
    prep_hash = fold_file_sha256(prep_path)
    prep = load_fold_preprocessing(
        prep_path, context=context, task_file=args.task_file, feature_columns=columns,
        member_id=args.member_id, validation_fold=args.member_id,
        policy="task0_standard_frozen", experiment_seed=0, expected_sha256=prep_hash)
    seeds = resolve_seed_configuration(0, args.member_id)
    root = args.output_root / f"member_{args.member_id}"
    config_hash = fold_file_sha256(args.config)
    config_file = root / "run_config.json"
    if root.exists() and not config_file.is_file():
        raise FileExistsError(f"DER++ member output exists without run_config.json: {root}")
    if not root.exists():
        root.mkdir(parents=True)
        run_id = uuid.uuid4().hex
        _json(config_file, {
            "run_id": run_id, "method": METHOD, "method_protocol": method_protocol,
            "config_sha256": config_hash, "task_file_sha256": task_hash,
            "assignment_sha256": context.manifest["assignment_sha256"],
            "preprocessing_sha256": prep_hash, "member_id": args.member_id,
            "arguments": {key: str(value) for key, value in vars(args).items()},
            "training": train_cfg, "replay": config["replay"],
            "head_raw_class_order": sorted(all_classes),
        })
    else:
        prior = json.loads(config_file.read_text(encoding="utf-8"))
        if (prior["config_sha256"] != config_hash or prior["task_file_sha256"] != task_hash
                or prior["assignment_sha256"] != context.manifest["assignment_sha256"]
                or prior["preprocessing_sha256"] != prep_hash
                or prior["member_id"] != args.member_id):
            raise ValueError("DER++ resume provenance changed; refusing to mix runs.")
        run_id = prior["run_id"]
    set_configured_seeds(seeds)
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model = TDDIPaperMember(TDDIPaperMemberConfig(
        input_dim=len(columns), num_classes=178, dropout=float(config["model"]["dropout"]),
        activation=config["model"]["activation"])).to(device)
    buffer_type = ClassBalancedReservoirLogitBuffer if balanced_pilot else ReservoirLogitBuffer
    buffer = buffer_type(args.memory_budget, len(columns), 178, seed=seeds.member_seed + 13579)
    checkpoints = sorted((root / "checkpoints").glob("task_*.pt"),
                         key=lambda path: int(path.stem.split("_")[1])) if (root / "checkpoints").exists() else []
    completed = -1
    summary_rows: list[dict] = []
    observed_counts: Counter[int] = Counter()
    if checkpoints:
        checkpoint = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        completed = int(checkpoint["task_id"])
        if (checkpoint["kind"] != method_protocol or checkpoint["run_id"] != run_id
                or checkpoint["config_sha256"] != config_hash
                or checkpoint["task_file_sha256"] != task_hash
                or checkpoint["member_id"] != args.member_id
                or checkpoints[-1].name != f"task_{completed}.pt"):
            raise ValueError("DER++ checkpoint provenance mismatch.")
        for task_id in range(completed + 1):
            for split in ("validation", "test"):
                if not (root / "member_predictions" / f"task_{task_id}" / f"{split}.npz").is_file():
                    raise ValueError("DER++ resume checkpoint lacks a completed prediction artifact.")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        buffer.load_state_dict(checkpoint["buffer_state"])
        summary_rows = checkpoint["summary_rows"]
        observed_counts = Counter({int(k): int(v) for k, v in checkpoint["observed_counts"].items()})
        torch.random.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        _log(root, f"Resumed after task={completed}; reservoir={buffer.size}/{buffer.capacity}")
    else:
        _log(root, f"Started DER++ member={args.member_id}; P3; fixed_head=178; one_pass_per_task")
    if completed == 7:
        _write_member_summary(root, args.member_id, run_id, summary_rows)
        _log(root, "Verified task 7 checkpoint already exists; no retraining.")
        return
    _preserve_incomplete_task(root, completed + 1)
    seen: set[int] = {raw for task in tasks[:completed + 1] for raw in task["classes"]}
    for task in tasks[completed + 1:]:
        task_id = int(task["task_id"])
        new_classes = [int(raw) for raw in task["classes"]]
        seen.update(new_classes)
        seen_raw = sorted(seen)
        task_root = root / f"task_{task_id}"
        task_root.mkdir()
        started = time.perf_counter()
        current = load_development_fold_arrays(context, columns, role="train",
            member_id=args.member_id, validation_fold=args.member_id, class_ids=new_classes)
        if not len(current.labels):
            raise ValueError(f"Task {task_id} has no training rows for member {args.member_id}.")
        current_inputs = _values(current.features, prep, columns, args.member_id)
        raw_labels = np.asarray(current.labels, dtype=np.int64)
        local_labels = np.asarray([full_map[int(raw)] for raw in raw_labels], dtype=np.int64)
        memory_before = buffer.size
        if balanced_pilot:
            quotas = buffer.begin_task(dict(Counter(local_labels.tolist())))
            _log(root, f"task={task_id} balanced quotas set for {len(quotas)} observed classes; "
                  f"retained_old={buffer.size}/{buffer.capacity}")
        memory_after_rebalance = buffer.size
        sample_ids = np.asarray(current.metadata["sample_id"]).astype(np.str_, copy=False)
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError("Current DER++ stream contains duplicate sample IDs.")
        order = np.random.default_rng(seeds.member_seed + task_id + 24680).permutation(len(raw_labels))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                                      weight_decay=float(train_cfg["weight_decay"]))
        model.train()
        totals = defaultdict(float)
        exposure: dict[int, int] = {}
        for start in range(0, len(order), batch_size):
            picked = order[start:start + batch_size]
            x = torch.from_numpy(current_inputs[picked]).to(device)
            y = torch.from_numpy(local_labels[picked]).to(device)
            losses = observe(model, optimizer, x, y, sample_ids[picked].tolist(), buffer,
                alpha=float(train_cfg["alpha"]), beta=float(train_cfg["beta"]),
                replay_batch_size=replay_batch_size, exposure=exposure)
            for name in ("current_ce", "replay_logit_mse", "replay_label_ce", "total"):
                totals[name] += getattr(losses, name) * len(picked)
            totals["replay_logit_rows"] += losses.replay_logit_rows
            totals["replay_label_rows"] += losses.replay_label_rows
            if (start // batch_size + 1) % 250 == 0:
                _log(root, f"task={task_id} step={start // batch_size + 1} rows={start + len(picked)}/{len(order)} reservoir={buffer.size}")
        for raw, count in Counter(raw_labels.tolist()).items():
            observed_counts[int(raw)] += int(count)
        if balanced_pilot:
            buffer.finish_task()
        del current_inputs, current
        validation = load_development_fold_arrays(context, columns, role="validation",
            member_id=args.member_id, validation_fold=args.member_id, class_ids=seen_raw)
        val_inputs = _values(validation.features, prep, columns, args.member_id)
        val_meta = {key: np.asarray(value) for key, value in validation.metadata.items()}
        val_outputs, val_metrics = _predict(model, val_inputs, validation.labels, val_meta,
                                            seen_raw, full_map, device, batch_size)
        all_identity = load_development_fold_identity(context, role="all",
            member_id=args.member_id, validation_fold=args.member_id, class_ids=seen_raw)
        _export(args, root=root, context=context, task_id=task_id, split="validation",
            outputs=val_outputs, metadata=val_meta, run_id=run_id, seeds=seeds,
            prep_hash=prep_hash, task_hash=task_hash, config_hash=config_hash,
            method_protocol=method_protocol, buffer_policy=buffer_policy,
            member_count=len(member_ids),
            all_identity=all_identity)
        frame = load_split_frame(args.test, columns, class_ids=seen_raw,
                                 include_metadata=True, meta_cols=DEFAULT_META_COLS)
        test_values = _values(frame[columns].to_numpy(dtype=np.float64), prep, columns, args.member_id)
        test_labels = frame["class"].to_numpy(dtype=np.int64)
        test_meta = {column: frame[column].to_numpy(copy=False) for column in DEFAULT_META_COLS}
        test_meta["sample_id"] = build_stable_sample_ids(test_meta, len(test_labels))
        test_meta["fold_id"] = np.full(len(test_labels), -1, dtype=np.int16)
        test_outputs, test_metrics = _predict(model, test_values, test_labels, test_meta,
                                             seen_raw, full_map, device, batch_size)
        _export(args, root=root, context=context, task_id=task_id, split="test",
            outputs=test_outputs, metadata=test_meta, run_id=run_id, seeds=seeds,
            prep_hash=prep_hash, task_hash=task_hash, config_hash=config_hash,
            method_protocol=method_protocol, buffer_policy=buffer_policy,
            member_count=len(member_ids))
        metrics = ([{"task": task_id, "split": "validation", **row}
                    for row in _groups(val_outputs, new_classes, seen_raw)]
                   + [{"task": task_id, "split": "test", **row}
                      for row in _groups(test_outputs, new_classes, seen_raw)])
        pd.DataFrame(metrics).to_csv(task_root / "metrics.csv", index=False)
        _json(task_root / "metrics.json", metrics)
        rows = len(raw_labels)
        audit = {"task_id": task_id, "train_rows": rows, "optimizer_steps": math.ceil(rows / batch_size),
                 "epochs": 1, "memory_before": memory_before,
                 "memory_after_rebalance": memory_after_rebalance,
                 "memory_after": buffer.size,
                 "stream_rows_seen_total": buffer.seen,
                 **{f"mean_{name}": totals[name] / rows for name in
                    ("current_ce", "replay_logit_mse", "replay_label_ce", "total")},
                 "replay_logit_rows": int(totals["replay_logit_rows"]),
                 "replay_label_rows": int(totals["replay_label_rows"]),
                 "validation_macro_f1": val_metrics["macro_f1"],
                 "test_macro_f1": test_metrics["macro_f1"],
                 "seconds": time.perf_counter() - started}
        _json(task_root / "training_audit.json", audit)
        pd.DataFrame([audit]).to_csv(task_root / "training_audit.csv", index=False)
        raw_order = sorted(all_classes)
        counts = Counter(raw_order[int(index)] for index in buffer.labels[:buffer.size])
        _json(task_root / "buffer_audit.json", {
            "policy": buffer_policy, "budget": buffer.capacity,
            "retained_count": buffer.size, "stream_rows_seen": buffer.seen,
            "allocation": {str(raw): counts[raw] for raw in seen_raw},
            "observed_counts": {str(raw): observed_counts[raw] for raw in seen_raw},
            "feasible_capacities": {str(raw): observed_counts[raw] for raw in seen_raw},
            "target_quotas": ({str(raw_order[label]): quota for label, quota in buffer.quotas.items()}
                              if balanced_pilot else None),
            "retained_ids": buffer.sample_ids[:buffer.size].tolist(),
            "retained_raw_labels": [raw_order[int(index)] for index in buffer.labels[:buffer.size]],
            "stored_logits_width": 178,
        })
        pd.DataFrame([{"raw_class_id": raw_order[label], "replay_draws": count}
                      for label, count in sorted(exposure.items())],
                     columns=["raw_class_id", "replay_draws"]).to_csv(
                          task_root / "replay_exposure.csv", index=False)
        summary = {"task_id": task_id, "head_size": 178, "seen_class_count": len(seen_raw),
                   "train_rows": rows, "memory_before": memory_before,
                   "memory_after_rebalance": memory_after_rebalance,
                   "memory_after": buffer.size,
                   "validation_macro_f1": val_metrics["macro_f1"], "test_macro_f1": test_metrics["macro_f1"],
                   "runtime_seconds": audit["seconds"]}
        _json(task_root / "completed_task.json", summary)
        summary_rows.append(summary)
        _checkpoint(root / "checkpoints" / f"task_{task_id}.pt", {
            "kind": method_protocol, "run_id": run_id, "member_id": args.member_id,
            "task_id": task_id, "config_sha256": config_hash, "task_file_sha256": task_hash,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "buffer_state": buffer.state_dict(), "summary_rows": summary_rows,
            "observed_counts": dict(observed_counts),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        })
        _log(root, f"task={task_id} complete val_macro_f1={val_metrics['macro_f1']:.6f} "
              f"test_macro_f1={test_metrics['macro_f1']:.6f} reservoir={buffer.size}")
        del validation, val_inputs, val_outputs, frame, test_values, test_outputs
    _write_member_summary(root, args.member_id, run_id, summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "task_file", "train", "validation", "test", "feature_cols",
                 "fold_assignments", "fold_manifest", "preprocessing_root", "output_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--member-id", type=int, required=True)
    parser.add_argument("--memory-budget", type=int, default=27778)
    parser.add_argument("--device", default="cuda")
    run_member(parser.parse_args())


if __name__ == "__main__":
    main()
