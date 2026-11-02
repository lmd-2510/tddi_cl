from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from src.eval.package import build_metrics_review_package


FINAL_SUFFIXES = (
    "final_report.md",
    "task_summary.csv",
    "paper_table.csv",
    "task_matrix.csv",
    "forgetting_summary.csv",
    "bwt_summary.csv",
    "member_forgetting_summary.csv",
    "diversity_summary.csv",
)


def _completed_run(tmp_path: Path) -> Path:
    root = tmp_path / "completed_run"
    final = root / "final_results"
    final.mkdir(parents=True)
    for suffix in FINAL_SUFFIXES:
        (final / f"ensemble3_ordered_{suffix}").write_text("metric\n", encoding="utf-8")
    for member_id in range(3):
        member = root / f"member_{member_id}"
        member.mkdir()
        (member / "metrics.csv").write_text("macro_f1\n0.5\n", encoding="utf-8")
        (member / "training_audit.csv").write_text("loss\n1.0\n", encoding="utf-8")
        (member / "run_summary.json").write_text("{}\n", encoding="utf-8")
        (member / "run_config.json").write_text("{}\n", encoding="utf-8")
        (member / "train.log").write_text("completed\n", encoding="utf-8")
        (member / "events.csv").write_text("event\n", encoding="utf-8")
        task = member / "task_7"
        task.mkdir()
        (task / "metrics.csv").write_text("macro_f1\n0.5\n", encoding="utf-8")
        (task / "buffer_audit.json").write_text("{}\n", encoding="utf-8")
        (member / "checkpoints").mkdir()
        (member / "checkpoints" / "task_7.pt").write_bytes(b"model")
        (member / "member_predictions").mkdir()
        (member / "member_predictions" / "test.npz").write_bytes(b"prediction")
    visualization = root / "visualizations"
    visualization.mkdir()
    (visualization / "forgetting.png").write_bytes(b"png")
    threshold = root / "offline_evaluation" / "task_7"
    threshold.mkdir(parents=True)
    (threshold / "test_threshold_report.json").write_text("{}\n", encoding="utf-8")
    (threshold / "test.npz").write_bytes(b"prediction")
    return root


def test_builds_validated_metrics_zip_and_checksum(tmp_path: Path) -> None:
    run_root = _completed_run(tmp_path)

    archive, checksum = build_metrics_review_package(run_root)

    assert archive.is_file() and checksum.is_file()
    expected_digest, expected_name = checksum.read_text(encoding="utf-8").strip().split()
    assert expected_name == archive.name
    assert expected_digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        assert "PACKAGE_MANIFEST.json" in names
        assert "PACKAGE_SUMMARY.md" in names
        assert "final_results/ensemble3_ordered_bwt_summary.csv" in names
        assert "visualizations/forgetting.png" in names
        assert "member_0/run_config.json" in names
        assert "member_0/task_7/metrics.csv" in names
        assert "offline_evaluation/task_7/test_threshold_report.json" in names
        assert not any(name.endswith((".pt", ".npz")) for name in names)
        assert not any(
            name.endswith((".log", "training_audit.csv", "events.csv", "buffer_audit.json"))
            for name in names
        )
        manifest = json.loads(package.read("PACKAGE_MANIFEST.json"))
    assert manifest["artifact_kind"] == "ddi_cil_metrics_review_package"
    assert "bwt_summary.csv" in manifest["required_artifacts"]
    assert manifest["included_file_count"] == len(manifest["files"])


def test_refuses_incomplete_run_and_existing_archive(tmp_path: Path) -> None:
    run_root = _completed_run(tmp_path)
    (run_root / "final_results" / "ensemble3_ordered_bwt_summary.csv").unlink()
    with pytest.raises(ValueError, match="bwt_summary"):
        build_metrics_review_package(run_root)

    run_root = _completed_run(tmp_path / "second")
    build_metrics_review_package(run_root)
    with pytest.raises(FileExistsError, match="--overwrite"):
        build_metrics_review_package(run_root)
