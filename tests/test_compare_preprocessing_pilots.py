"""Offline report-only comparison and tiny CPU end-to-end input artifacts."""
import ast
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from scripts import compare_preprocessing_pilots as report
from src.training import fold_ab_study as study, fold_pilot_training as pilot
from tests.test_fold_ab_study import synthetic_configs, configs
from tests.test_fold_pilot_training import data, tiny


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("ab_report_source")
    dataset = data.__wrapped__(root, SimpleNamespace())
    patcher = pytest.MonkeyPatch()
    small = tiny.__wrapped__(patcher)
    next(small)
    try:
        pair = synthetic_configs(dataset)
        for config in pair:
            args = study.engine.parse_args(study.member_command(config, 0)[2:])
            pilot.run_fold_training(args, engine=study.engine)
        return [Path(c["output_root"]) / "member_0" for c in pair], dataset["tasks"]
    finally:
        next(small, None)
        patcher.undo()


@pytest.fixture
def copied(completed, tmp_path):
    roots, protocol = completed
    for name, root in zip(("A", "B"), roots):
        shutil.copytree(root, tmp_path / name)
    return [tmp_path / "A", tmp_path / "B"], protocol


def change(path, update):
    value = report.read_json(path)
    update(value)
    path.write_text(json.dumps(value))


def test_end_to_end_cli_and_report_only_bundle(completed, tmp_path):
    (a, b), tasks = completed
    out = tmp_path / "report"
    command = [sys.executable, str(study.PROJECT_ROOT / "scripts/compare_preprocessing_pilots.py"),
               "--run-a", str(a), "--run-b", str(b), "--task-file", str(tasks), "--outdir", str(out), "--bundle"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = report.read_json(out / "comparison.json")
    assert result["winner"] is None and result["alignment_status"] == "pass"
    for case in ("A", "B"):
        assert result["final_task1_seen_all"][case] == result["runs"][case]["tasks"][1]["metrics"]["seen_all"]
        assert len(result["runs"][case]["tasks"][1]["epoch_curves"]) == 2
    text = (out / "comparison.md").read_text(encoding="utf-8")
    assert "NOT a mean" in text and "unavailable" in text and "Logit raw/scaled" in text
    unpacked = tmp_path / "unpacked"
    with tarfile.open(out / "review.tar.gz") as bundle:
        assert not any(n.endswith((".pt", ".parquet")) for n in bundle.getnames())
        bundle.extractall(unpacked, filter="data")
    portable = report.compare(unpacked / "A", unpacked / "B", unpacked / "tail_to_head_tasks.json")
    assert portable["final_task1_seen_all"] == result["final_task1_seen_all"]
    before = (out / "comparison.json").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (out / "comparison.json").read_bytes() == before


@pytest.mark.parametrize("relative,update,match", [
    ("task_1/input_audit.json", lambda v: v["validation_ids"].reverse(), "alignment mismatch"),
    ("task_1/buffer_audit.json", lambda v: v["retained_ids"].reverse(), "retained_ids_sha256"),
    ("task_1/epoch_1_audit.json", lambda v: v["sampling"].update(draw_order_ids_sha256="wrong"), "sampler order"),
    ("task_1/epoch_1_audit.json", lambda v: v["sampling"].update(max_repeat=4), "Replay cap"),
    ("task_1/metrics.json", lambda v: v[0].update(split="test"), "Test metrics"),
    ("task_1/metrics.json", lambda v: v[0].update(macro_f1=float("nan")), "nonfinite"),
    ("task_1/completed_task.json", lambda v: v.update(head_size=99), "Class map"),
])
def test_mismatch_fails_before_export(copied, tmp_path, relative, update, match):
    (a, b), tasks = copied
    change(b / relative, update)
    out = tmp_path / "no_report"
    with pytest.raises(ValueError, match=match):
        report.main(["--run-a", str(a), "--run-b", str(b), "--task-file", str(tasks), "--outdir", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("key,value", [("member_id", 1), ("lr", 0.002)])
def test_member_or_hyper_drift(copied, key, value):
    (a, b), tasks = copied
    def update(c):
        if key == "member_id":
            c["checkpoint_contract"]["seeds"][key] = value
            c["resolved"]["seeds"][key] = value
        else:
            c["checkpoint_contract"]["hyperparameters"][key] = value
            c["arguments"][key] = value
        c["config_sha256"] = report.digest({"arguments": c["arguments"], "resolved": c["resolved"]})
    change(b / "run_config.json", update)
    with pytest.raises(ValueError, match="scope|contract mismatch"):
        report.compare(a, b, tasks)


def test_missing_telemetry_unavailable_and_resumed_attempt_directory(copied):
    (a, b), tasks = copied
    source = b / "task_1"
    dest = b / "attempts/task_1_resumed"
    dest.parent.mkdir()
    shutil.move(str(source), str(dest))
    def update(v):
        v["artifact_directory"] = "attempts/task_1_resumed"
        for key in report.TELEMETRY: v.pop(key, None)
    change(dest / "completed_task.json", update)
    change(dest / "epoch_1_audit.json", lambda v: [v.pop(k, None) for k in ("seconds", "optimizer_steps")])
    result = report.compare(a, b, tasks)
    t = result["runs"]["B"]["tasks"][1]
    assert t["telemetry"]["runtime_seconds"] is None
    assert t["epoch_curves"][0]["optimizer_steps"] is None
    assert result["warnings"] and "unavailable" in report.markdown(result)


def test_missing_task_or_wrong_protocol_fails(copied, tmp_path):
    (a, b), tasks = copied
    bad = tmp_path / "tasks.json"
    bad.write_bytes(tasks.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        report.compare(a, b, bad)
    (b / "task_1/completed_task.json").unlink()
    with pytest.raises(ValueError, match="must be complete"):
        report.compare(a, b, tasks)


def test_help_is_standalone_without_training_imports():
    done = subprocess.run([sys.executable, str(study.PROJECT_ROOT / "scripts/compare_preprocessing_pilots.py"), "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "--bundle" in done.stdout


def test_runbook_bash_and_embedded_python_syntax_only():
    """Parse documentation commands; never execute data preparation or training."""
    doc = (study.PROJECT_ROOT / "docs/TDDI_PREPROCESSING_AB_PILOT_RUNBOOK.md").read_text(encoding="utf-8")
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if bash is None:
        pytest.skip("Bash unavailable for shell syntax validation")
    blocks = re.findall(r"```bash\n(.*?)\n```", doc, re.S)
    assert len(blocks) >= 10
    for block in blocks:
        result = subprocess.run([bash, "-n"], input=block, text=True, capture_output=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr
    for block in re.findall(r"python - <<'PY'[^\n]*\n(.*?)\nPY", doc, re.S):
        ast.parse(block)


@pytest.mark.parametrize("micro", [8, 16, 32])
def test_oom_config_and_dry_run_preserves_effective_batch(tmp_path, micro):
    pair = []
    for old in configs():
        raw = report.read_json(old["config_path"])
        raw["training"].update(batch_size=micro, gradient_accumulation_steps=1024 // micro)
        path = tmp_path / f"{old['case']}.json"
        path.write_text(json.dumps(raw))
        pair.append(study.load_pilot_config(path))
    study.validate_ab_group(pair)
    plan = study.build_pilot_plan(pair)
    assert all("--effective-batch-size" in e["command"] for e in plan["entries"])
    args = study.engine.parse_args(plan["entries"][0]["command"][2:])
    pilot.validate_fold_options(args)
    assert args.batch_size * pair[0]["training"]["gradient_accumulation_steps"] == 1024
    pair[1]["training"]["batch_size"] = 64
    with pytest.raises(ValueError, match="differ beyond"):
        study.validate_ab_group(pair)


def test_oom_tiny_training_uses_requested_microbatch(data, tiny):
    config = synthetic_configs(data)[0]
    config["training"].update(batch_size=32, gradient_accumulation_steps=32)
    args = study.engine.parse_args(study.member_command(config, 0)[2:])
    pilot.run_fold_training(args, engine=study.engine)
    saved = report.read_json(args.outdir / "run_config.json")
    assert saved["resolved"]["microbatch"] == saved["resolved"]["gradient_accumulation"] == 32
    audit = report.read_json(args.outdir / "task_1/epoch_1_audit.json")
    assert audit["optimizer_steps"] == 1 and audit["examples_seen"] == 91
