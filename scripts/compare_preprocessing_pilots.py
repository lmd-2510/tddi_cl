"""Validation-only frozen-fold A/B report. No torch, dataset scan or winner selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile

LAYOUT = [38, 20, 20, 20, 20, 20, 20, 20]
LOSSES = ("classification_loss", "logit_distillation_loss", "scaled_logit_distillation_loss",
          "feature_distillation_loss", "scaled_feature_distillation_loss", "total_loss")
TELEMETRY = ("runtime_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "checkpoint_size_bytes")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_run(root, case, protocol, task_hash, *, expected_preprocessing_policy=None,
             expected_ranking_policy=None, comparison_name="preprocessing"):
    root = Path(root).resolve()
    inventory, warnings = {}, []
    def load(relative):
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Artifact path escapes run root")
        payload = read_json(path)
        inventory[path.relative_to(root).as_posix()] = sha256(path)
        return payload
    config = load("run_config.json")
    contract = config["checkpoint_contract"]
    resolved = config["resolved"]
    require(contract["method"] == "replay_distill_fixed_budget_uniform" and contract["model"]["variant"] == "tddi_paper_member", f"Wrong method/backbone for {comparison_name} pilot")
    require(config["config_sha256"] == digest({"arguments": config["arguments"], "resolved": resolved}), "run_config checksum mismatch")
    require(contract["training_policy"] == "frozen_fold_replay_distill_v1" and contract["validation_only"], "Only frozen-fold validation-only pilots are accepted; no test selection")
    require(contract["task_file_sha256"] == task_hash and contract["task_protocol"] == "tail_to_head"
            and contract["task_layout"] == LAYOUT, "P3 task-file hash/layout mismatch")
    require(contract["seeds"]["experiment_seed"] == 0 and contract["seeds"]["member_id"] == 0
            and contract["validation_fold"] == 0 and contract["fold_seed"] == 42, "Report scope is seed0/fold42/member0")
    if expected_preprocessing_policy is None:
        expected_preprocessing_policy = {"A": "raw_identity", "B": "task0_standard_frozen"}[case]
    require(contract["preprocessing"]["policy"] == expected_preprocessing_policy,
            f"{comparison_name} preprocessing policy mismatch")
    if expected_ranking_policy is not None:
        require(contract["buffer"]["ranking"]["policy"] == expected_ranking_policy,
                f"{comparison_name} ranking policy mismatch")
    require(resolved["stop_after_task"] in (0, 1), "Not a full8 report")
    require(all(resolved.get(k) == v for k, v in contract.items() if k not in ("hyperparameters", "sampler")), "Resolved/contract mismatch")
    require(all(config["arguments"].get(k) == v for k, v in contract["hyperparameters"].items()), "Argument/hyperparameter mismatch")
    folders = {}
    for path in root.rglob("completed_task.json"):
        summary = load(path.relative_to(root))
        task = summary["task_id"]
        require(task in (0, 1) and task not in folders, "Missing/ambiguous completion scope: only one completed artifact directory per task0/1")
        require(summary["artifact_directory"] == path.parent.relative_to(root).as_posix(), "Task artifact directory mismatch")
        folders[task] = (path.parent.relative_to(root), summary)
    require(set(folders) == {0, 1}, "Both task0 and task1 must be complete before comparison")
    tasks = []
    for task in (0, 1):
        folder, summary = folders[task]
        inputs, buffer, metrics = [load(folder / name) for name in ("input_audit.json", "buffer_audit.json", "metrics.json")]
        raw = sorted(c for t in protocol["tasks"][:task + 1] for c in t["classes"])
        expected_map = {str(c): i for i, c in enumerate(raw)}
        require(inputs["seen_class_map"] == summary["seen_class_map"] == expected_map
                and summary["head_size"] == len(raw), "Class map/head mismatch")
        require(inputs["current_raw_classes"] == protocol["tasks"][task]["classes"], "Current P3 classes mismatch")
        require(inputs["assignment_sha256"] == contract["assignment_sha256"]
                and inputs["preprocessing_sha256"] == contract["preprocessing_sha256"], "Task provenance mismatch")
        for key in ("current_ids", "replay_ids_before", "validation_ids"):
            require(len(inputs[key]) == len(set(inputs[key])), f"Duplicate {key}")
        require(not set(inputs["current_ids"] + inputs["replay_ids_before"]) & set(inputs["validation_ids"]), "Held-out IDs entered training/replay")
        retained = buffer["retained_ids"]
        require(len(retained) == len(set(retained)) == summary["memory_after"]
                and set(retained) <= set(inputs["current_ids"] + inputs["replay_ids_before"]), "Retained exemplar IDs/count/source mismatch")
        require(len(buffer["retained_raw_labels"]) == len(retained) == buffer["stored_slots"] <= buffer["budget"], "Buffer labels/slots/budget mismatch")
        require(len(inputs["replay_ids_before"]) == summary["memory_before"], "Memory-before count mismatch")
        require(len(metrics) == 3 and {r["group"] for r in metrics} == {"seen_all", "old", "current"}, "Missing/duplicate validation groups")
        for row in metrics:
            require(row["task"] == task and row["split"] == "validation", "Test metrics cannot enter this report")
            if row["sample_count"]:
                require(row["status"] == "ok" and all(number(row.get(k)) and 0 <= row[k] <= 1 for k in ("macro_f1", "balanced_accuracy", "accuracy")), "Invalid/nonfinite validation metrics")
        by_group = {r["group"]: r for r in metrics}
        require(by_group["seen_all"]["sample_count"] == len(inputs["validation_ids"])
                and sum(by_group[g]["sample_count"] for g in ("old", "current")) == len(inputs["validation_ids"]), "Validation sample-count mismatch")
        curves = []
        for epoch in range(1, summary["epochs_trained"] + 1):
            audit = load(folder / f"epoch_{epoch}_audit.json")
            require(audit["task"] == task and audit["epoch"] == epoch, "Epoch audit alignment mismatch")
            sampling = audit["sampling"]
            require(sampling["current_draws"] == len(inputs["current_ids"])
                    and sampling["max_repeat"] <= sampling["repeat_cap"] == 3
                    and sampling["memory_size"] == summary["memory_before"], "Replay cap/current coverage mismatch")
            replay = 0 if task == 0 else min(len(inputs["current_ids"]) // 7, 3 * summary["memory_before"])
            require(sampling["actual_replay_draws"] == replay and sampling["total_draws"] == replay + len(inputs["current_ids"])
                    and math.isclose(sampling["actual_fraction"], replay / sampling["total_draws"], abs_tol=1e-12), "Replay fraction/draw count mismatch")
            require(all(number(audit.get(k)) for k in (*LOSSES, "validation_macro_f1", "validation_balanced_accuracy")), "Missing/nonfinite losses or validation epoch metrics")
            require(math.isclose(audit["total_loss"], audit["classification_loss"] + audit["scaled_logit_distillation_loss"]
                                 + audit["scaled_feature_distillation_loss"], rel_tol=1e-6, abs_tol=1e-8), "Loss components do not sum to total")
            curves.append({k: audit[k] for k in ("epoch", *LOSSES, "validation_macro_f1", "validation_balanced_accuracy", "sampling")})
            for key in ("seconds", "optimizer_steps"):
                value = audit.get(key)
                curves[-1][key] = value if number(value) and value >= 0 else None
                if curves[-1][key] is None: warnings.append(f"{case} task{task} epoch{epoch} {key}: unavailable")
        require(curves and 1 <= summary["best_epoch"] <= len(curves), "Invalid best epoch")
        best = curves[summary["best_epoch"] - 1]["validation_macro_f1"]
        require(math.isclose(best, summary["best_validation_macro_f1"], abs_tol=1e-9)
                and math.isclose(best, by_group["seen_all"]["macro_f1"], abs_tol=1e-9)
                and math.isclose(best, max(r["validation_macro_f1"] for r in curves), abs_tol=1e-9), "Best checkpoint/validation score mismatch")
        telemetry = {}
        for key in TELEMETRY:
            value = summary.get(key)
            telemetry[key] = value if number(value) and value >= 0 else None
            if telemetry[key] is None: warnings.append(f"{case} task{task} {key}: unavailable")
        boundary = root / summary["boundary_checkpoint"]
        require(boundary.resolve().is_relative_to(root), "Boundary path escapes root")
        telemetry["boundary_checkpoint_size_bytes"] = boundary.stat().st_size if boundary.is_file() else None
        if not boundary.is_file(): warnings.append(f"{case} task{task} boundary checkpoint size: unavailable (report-only bundle is allowed)")
        tasks.append(dict(task_id=task, artifact_directory=folder.as_posix(),
            best_epoch=summary["best_epoch"], epochs_trained=len(curves),
            raw_classes={"seen_all": raw, "old": sorted(set(raw) - set(inputs["current_raw_classes"])), "current": inputs["current_raw_classes"]},
            metrics=by_group, epoch_curves=curves, telemetry=telemetry,
            memory_before=summary["memory_before"], memory_after=summary["memory_after"],
            alignment={k: digest(inputs[k]) for k in ("current_ids", "replay_ids_before", "validation_ids", "seen_class_map", "sampler")},
            retained_ids_sha256=digest(retained), retained_labels_sha256=digest(buffer["retained_raw_labels"])))
    return dict(root=str(root), run_id=config["run_id"], contract=contract, tasks=tasks,
                input_sha256=inventory, warnings=warnings)


def compare(run_a, run_b, task_file):
    protocol = read_json(task_file)
    require(protocol.get("protocol") == "tail_to_head" and [t["task_id"] for t in protocol["tasks"]] == list(range(8))
            and [len(t["classes"]) for t in protocol["tasks"]] == LAYOUT
            and len({c for t in protocol["tasks"] for c in t["classes"]}) == 178, "Full P3 protocol required")
    runs = {case: read_run(root, case, protocol, sha256(task_file)) for case, root in (("A", run_a), ("B", run_b))}
    controls = []
    for case in ("A", "B"):
        control = deepcopy(runs[case]["contract"])
        prep = control.pop("preprocessing")
        control.pop("preprocessing_sha256")
        control["preprocessing_reference"] = prep["provenance"]
        controls.append(control)
    require(controls[0] == controls[1], "A/B fold/member/task/hyper/data/ranking contract mismatch; no score comparison")
    for a, b in zip(runs["A"]["tasks"], runs["B"]["tasks"], strict=True):
        for key in ("alignment", "retained_ids_sha256", "retained_labels_sha256"):
            require(a[key] == b[key], f"A/B task{a['task_id']} {key} mismatch; no score comparison")
        for x, y in zip(a["epoch_curves"], b["epoch_curves"]):
            require(x["sampling"] == y["sampling"], f"A/B sampler order/exposure mismatch at task{a['task_id']} epoch{x['epoch']}")
    final = {case: runs[case]["tasks"][1]["metrics"]["seen_all"] for case in runs}
    return dict(schema="ddi_cil_preprocessing_ab_report", version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        validation_only=True, scope={"member_id": 0, "completed_task_id": 1, "full_protocol_tasks": 8},
        alignment_status="pass", winner=None, task_file_sha256=sha256(task_file),
        metric_semantics="Task1 model evaluated on ALL seen validation classes; NOT a mean across stages; no test metrics.",
        integrity_scope="JSON audit consistency and source hashes; not checkpoint/resume validation or dataset re-audit.",
        final_task1_seen_all=final, delta_B_minus_A={k: final["B"][k] - final["A"][k] for k in ("macro_f1", "balanced_accuracy", "accuracy")},
        warnings=[w for r in runs.values() for w in r["warnings"]], runs=runs)


def markdown(report):
    def fmt(v): return "unavailable" if v is None else f"{v:.6f}" if isinstance(v, float) else str(v)
    lines = ["# Preprocessing A/B — validation pilot", "", report["metric_semantics"], "",
             "Scope: member 0, task 0–1 / P3 full8. Alignment: PASS. **Không tự chọn winner.**", "",
             "## Metrics theo task (không lấy mean stage thành final)", "",
             "| Case | Task | Group | N | Accuracy | Macro-F1 | Balanced Accuracy |", "|---|---:|---|---:|---:|---:|---:|"]
    for case, run in report["runs"].items():
        for task in run["tasks"]:
            for group, row in task["metrics"].items():
                lines.append(f"| {case} | {task['task_id']} | {group} | {row['sample_count']} | {fmt(row.get('accuracy'))} | {fmt(row.get('macro_f1'))} | {fmt(row.get('balanced_accuracy'))} |")
    lines += ["", "Task1 seen_all delta B − A: " + "; ".join(f"{k}={v:+.6f}" for k, v in report["delta_B_minus_A"].items()),
              "", "Old = class task0; current = class task1. Argmax vẫn trên tất cả seen classes, không dùng task ID để thu hẹp head."]
    for case, run in report["runs"].items():
        lines += ["", f"## {case} — run `{run['run_id']}`", ""]
        for task in run["tasks"]:
            lines += [f"### Task {task['task_id']}", "", f"Best epoch: {task['best_epoch']}; trained: {task['epochs_trained']}; memory {task['memory_before']} → {task['memory_after']}.", "",
                      "; ".join(f"{k}: {fmt(v)}" for k, v in task["telemetry"].items()), "",
                      "| Epoch | Val F1 | Val BA | Focal | Logit raw/scaled | Feature raw/scaled | Total | Steps | Seconds | Replay fraction | Class/exemplar coverage | Max repeat |",
                      "|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---:|"]
            for row in task["epoch_curves"]:
                s = row["sampling"]
                lines.append("| " + " | ".join(fmt(v) for v in (row["epoch"], row["validation_macro_f1"], row["validation_balanced_accuracy"], row["classification_loss"],
                    f"{row['logit_distillation_loss']:.6f}/{row['scaled_logit_distillation_loss']:.6f}", f"{row['feature_distillation_loss']:.6f}/{row['scaled_feature_distillation_loss']:.6f}",
                    row["total_loss"], row["optimizer_steps"], row["seconds"], s["actual_fraction"], f"{s['class_coverage']:.6f}/{s['exemplar_coverage']:.6f}", s["max_repeat"])) + " |")
    lines += ["", "## Giới hạn và telemetry thiếu", "", report["integrity_scope"], "",
              "Chỉ so preprocessing với ranking tạm thời; một member/hai task chưa chứng minh hiệu quả task7. Early stopping có thể khác; sampler chỉ so epoch chung.", ""]
    lines += [f"- {w}" for w in report["warnings"]] or ["Không thiếu telemetry đã yêu cầu."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True, help="NEW report directory; never overwrite")
    parser.add_argument("--bundle", action="store_true", help="Create report-only review.tar.gz; excludes model/data/assignment")
    args = parser.parse_args(argv)
    require(not args.outdir.exists(), "Report output exists; choose a new --outdir")
    report = compare(args.run_a, args.run_b, args.task_file)
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    (args.outdir / "comparison.md").write_text(markdown(report), encoding="utf-8")
    if args.bundle:
        with tarfile.open(args.outdir / "review.tar.gz", "x:gz") as archive:
            for name in ("comparison.json", "comparison.md"):
                archive.add(args.outdir / name, arcname=name)
            archive.add(args.task_file, arcname="tail_to_head_tasks.json")
            for case, run in report["runs"].items():
                for relative in run["input_sha256"]:
                    archive.add(Path(run["root"]) / relative, arcname=f"{case}/{relative}")
                for path in Path(run["root"]).glob("*.log"):
                    archive.add(path, arcname=f"{case}/{path.name}")
    print(f"Validation comparison saved: {args.outdir}. No winner selected.")


if __name__ == "__main__":
    main()
