from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

from scripts import package_p3_experiment_review as package


def test_review_package_keeps_reports_and_figures_but_excludes_arrays(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    (run_root / "visualizations").mkdir(parents=True)
    (run_root / "member_0" / "checkpoints").mkdir(parents=True)
    (run_root / "run_summary.md").write_text("summary", encoding="utf-8")
    (run_root / "visualizations" / "01.png").write_bytes(b"png")
    (run_root / "member_0" / "run_config.json").write_text("{}", encoding="utf-8")
    (run_root / "member_0" / "member_predictions.npz").write_bytes(b"npz")
    (run_root / "member_0" / "checkpoints" / "task_7.pt").write_bytes(b"weights")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    log = tmp_path / "run.log"
    log.write_text("failed after task 3", encoding="utf-8")
    archive = tmp_path / "review.zip"
    monkeypatch.setattr(sys, "argv", [
        "package", "--label", "sample", "--run-root", str(run_root),
        "--config", str(config), "--status", "train_exit=1", "--log", str(log),
        "--archive", str(archive),
    ])
    package.main()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "run/run_summary.md" in names
    assert "run/visualizations/01.png" in names
    assert "run/member_0/run_config.json" in names
    assert "logs/run.log" in names
    assert not any(name.endswith((".npz", ".pt")) for name in names)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.with_suffix(".zip.sha256").read_text(encoding="utf-8").startswith(digest)
