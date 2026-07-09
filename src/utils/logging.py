"""Minimal run logging helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunLogger:
    """Simple dual logger for stdout-like messages and structured events."""

    log_path: Path
    events_path: Path
    mirror_log_path: Path | None = None

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.mirror_log_path is not None:
            self.mirror_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.events_path.exists():
            with self.events_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "event_type", "message", "payload_json"],
                )
                writer.writeheader()

    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.mirror_log_path is not None:
            with self.mirror_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def event(self, event_type: str, message: str, payload_json: str = "") -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.events_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["timestamp", "event_type", "message", "payload_json"],
            )
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "message": message,
                    "payload_json": payload_json,
                }
            )

    def log_event(self, event_type: str, message: str, payload_json: str = "") -> None:
        self.log(message)
        self.event(event_type, message, payload_json=payload_json)


def ensure_run_paths(outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return {
        "outdir": outdir,
        "train_log": outdir / "train.log",
        "stdout_log": outdir / "stdout.log",
        "events_csv": outdir / "events.csv",
        "metrics_csv": outdir / "metrics.csv",
        "per_class_metrics_csv": outdir / "per_class_metrics.csv",
        "checkpoint_pt": outdir / "best_model.pt",
        "checkpoint_paths_txt": outdir / "checkpoint_paths.txt",
        "run_summary_md": outdir / "run_summary.md",
    }
