"""Validation-only comparison of two exemplar-ranking pilots; never selects a winner."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tarfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_preprocessing_pilots import (
    LAYOUT,
    read_json,
    read_run,
    require,
    sha256,
)
from src.data.fold_replay_buffer import (
    PIPELINE_INPUT_RANKING_POLICY,
    SAMPLE_NORMALIZED_RANKING_POLICY,
)

CASES = {
    "sample_normalized": SAMPLE_NORMALIZED_RANKING_POLICY,
    "pipeline_input": PIPELINE_INPUT_RANKING_POLICY,
}


def _sampling_control(audit):
    """Remove only hashes that are expected to change with exemplar identity."""
    value = deepcopy(audit)
    value.pop("replay_ids_sha256", None)
    value.pop("draw_order_ids_sha256", None)
    return value


def compare(sample_normalized_run, pipeline_input_run, task_file):
    protocol = read_json(task_file)
    require(protocol.get("protocol") == "tail_to_head"
            and [t["task_id"] for t in protocol["tasks"]] == list(range(8))
            and [len(t["classes"]) for t in protocol["tasks"]] == LAYOUT
            and len({c for task in protocol["tasks"] for c in task["classes"]}) == 178,
            "Full P3 protocol required")
    task_hash = sha256(task_file)
    roots = {"sample_normalized": sample_normalized_run, "pipeline_input": pipeline_input_run}
    runs = {
        case: read_run(
            root, case, protocol, task_hash,
            expected_preprocessing_policy="task0_standard_frozen",
            expected_ranking_policy=CASES[case], comparison_name="exemplar-ranking",
        )
        for case, root in roots.items()
    }
    controls = []
    for case in CASES:
        control = deepcopy(runs[case]["contract"])
        control["buffer"] = dict(control["buffer"])
        control["buffer"].pop("ranking")
        control.pop("ranking_seed_role")
        controls.append(control)
    require(controls[0] == controls[1],
            "Ranking pilots differ beyond ranking policy/output; no score comparison")

    overlaps = {}
    left, right = (runs[case] for case in CASES)
    for a, b in zip(left["tasks"], right["tasks"], strict=True):
        task = a["task_id"]
        for key in ("current_ids", "validation_ids", "seen_class_map"):
            require(a["alignment"][key] == b["alignment"][key],
                    f"Ranking task{task} {key} mismatch; no score comparison")
        require(a["retained_labels_sha256"] == b["retained_labels_sha256"],
                f"Ranking task{task} allocation/retained-label mismatch")
        for x, y in zip(a["epoch_curves"], b["epoch_curves"]):
            require(_sampling_control(x["sampling"]) == _sampling_control(y["sampling"]),
                    f"Ranking replay exposure mismatch at task{task} epoch{x['epoch']}")
        # Full retained IDs are intentionally not embedded in the report. Their
        # digests prove identity; overlap is recovered from validated buffer audits.
        a_folder = Path(left["root"]) / a["artifact_directory"] / "buffer_audit.json"
        b_folder = Path(right["root"]) / b["artifact_directory"] / "buffer_audit.json"
        require(a_folder.is_file() and b_folder.is_file(), "Missing ranking buffer audit")
        a_ids = set(read_json(a_folder)["retained_ids"])
        b_ids = set(read_json(b_folder)["retained_ids"])
        require(len(a_ids) == a["memory_after"] and len(b_ids) == b["memory_after"],
                "Ranking retained-ID count mismatch")
        union = a_ids | b_ids
        overlaps[str(task)] = {
            "sample_normalized_count": len(a_ids), "pipeline_input_count": len(b_ids),
            "intersection_count": len(a_ids & b_ids), "union_count": len(union),
            "jaccard": len(a_ids & b_ids) / len(union) if union else 1.0,
            "identical_ids": a_ids == b_ids,
        }
    final = {case: runs[case]["tasks"][1]["metrics"]["seen_all"] for case in CASES}
    delta = {key: final["pipeline_input"][key] - final["sample_normalized"][key]
             for key in ("macro_f1", "balanced_accuracy", "accuracy")}
    return {
        "schema": "ddi_cil_exemplar_ranking_pilot_report", "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "validation_only": True,
        "scope": {"member_id": 0, "completed_task_id": 1, "full_protocol_tasks": 8},
        "alignment_status": "pass", "winner": None, "task_file_sha256": task_hash,
        "preprocessing_policy": "task0_standard_frozen",
        "metric_semantics": "Task1 model evaluated on ALL seen validation classes; not a mean across stages; no test metrics.",
        "integrity_scope": "JSON audit consistency and source hashes; not checkpoint tensor validation or dataset re-audit.",
        "retained_id_overlap": overlaps, "final_task1_seen_all": final,
        "delta_pipeline_input_minus_sample_normalized": delta,
        "warnings": [warning for run in runs.values() for warning in run["warnings"]],
        "runs": runs,
    }


def markdown(report):
    def fmt(value):
        return "unavailable" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
    lines = [
        "# Exemplar ranking — validation pilot", "", report["metric_semantics"], "",
        "Preprocessing cố định: `task0_standard_frozen`. Scope: member 0, task 0–1 / P3 full8.",
        "Alignment: PASS. **Không tự chọn winner.**", "",
        "## Metrics", "",
        "| Ranking | Task | Group | N | Accuracy | Macro-F1 | Balanced Accuracy |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for case, run in report["runs"].items():
        for task in run["tasks"]:
            for group, row in task["metrics"].items():
                lines.append(f"| {case} | {task['task_id']} | {group} | {row['sample_count']} | "
                             f"{fmt(row.get('accuracy'))} | {fmt(row.get('macro_f1'))} | "
                             f"{fmt(row.get('balanced_accuracy'))} |")
    lines += ["", "Task1 seen_all delta pipeline_input − sample_normalized: "
              + "; ".join(f"{key}={value:+.6f}" for key, value in
                          report["delta_pipeline_input_minus_sample_normalized"].items()),
              "", "## Exemplar ID overlap", "",
              "IDs được phép khác; quota và raw-label allocation phải giống nhau.", "",
              "| Task | Sample-normalized | Pipeline-input | Intersection | Union | Jaccard | Identical |",
              "|---:|---:|---:|---:|---:|---:|---|" ]
    for task, row in report["retained_id_overlap"].items():
        lines.append(f"| {task} | {row['sample_normalized_count']} | {row['pipeline_input_count']} | "
                     f"{row['intersection_count']} | {row['union_count']} | {row['jaccard']:.6f} | "
                     f"{row['identical_ids']} |")
    for case, run in report["runs"].items():
        lines += ["", f"## {case} — run `{run['run_id']}`", ""]
        for task in run["tasks"]:
            lines += [f"### Task {task['task_id']}", "",
                      f"Best epoch: {task['best_epoch']}; trained: {task['epochs_trained']}; "
                      f"memory {task['memory_before']} → {task['memory_after']}.", "",
                      "; ".join(f"{key}: {fmt(value)}" for key, value in task["telemetry"].items()), "",
                      "| Epoch | Val F1 | Val BA | Focal | Logit raw/scaled | Feature raw/scaled | Total | Replay fraction | Max repeat |",
                      "|---:|---:|---:|---:|---|---|---:|---:|---:|"]
            for row in task["epoch_curves"]:
                sampling = row["sampling"]
                lines.append(f"| {row['epoch']} | {row['validation_macro_f1']:.6f} | "
                             f"{row['validation_balanced_accuracy']:.6f} | {row['classification_loss']:.6f} | "
                             f"{row['logit_distillation_loss']:.6f}/{row['scaled_logit_distillation_loss']:.6f} | "
                             f"{row['feature_distillation_loss']:.6f}/{row['scaled_feature_distillation_loss']:.6f} | "
                             f"{row['total_loss']:.6f} | {sampling['actual_fraction']:.6f} | {sampling['max_repeat']} |")
    lines += ["", "## Giới hạn", "", report["integrity_scope"], "",
              "Chỉ là ranking tạm trên validation member0/task0–1. Không dùng test, không tự chọn winner, chưa chứng minh task7.", ""]
    lines += [f"- {warning}" for warning in report["warnings"]] or ["Không thiếu telemetry đã yêu cầu."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-normalized-run", type=Path, required=True)
    parser.add_argument("--pipeline-input-run", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True, help="NEW report directory; never overwrite")
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args(argv)
    require(not args.outdir.exists(), "Report output exists; choose a new --outdir")
    report = compare(args.sample_normalized_run, args.pipeline_input_run, args.task_file)
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    (args.outdir / "comparison.md").write_text(markdown(report), encoding="utf-8")
    if args.bundle:
        with tarfile.open(args.outdir / "ranking_review.tar.gz", "x:gz") as archive:
            for name in ("comparison.json", "comparison.md"):
                archive.add(args.outdir / name, arcname=name)
            archive.add(args.task_file, arcname="tail_to_head_tasks.json")
            for case, run in report["runs"].items():
                for relative in run["input_sha256"]:
                    archive.add(Path(run["root"]) / relative, arcname=f"{case}/{relative}")
                for path in Path(run["root"]).glob("*.log"):
                    archive.add(path, arcname=f"{case}/{path.name}")
    print(f"Ranking validation comparison saved: {args.outdir}. No winner selected.")


if __name__ == "__main__":
    main()
