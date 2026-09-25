#!/usr/bin/env python3
"""Create a compact review ZIP: reports/logs/CSVs/figures, never model arrays."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


EXCLUDED_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".parquet", ".pq", ".pkl", ".pickle", ".ckpt"}
INCLUDED_SUFFIXES = {".md", ".json", ".csv", ".log", ".txt", ".png", ".yaml", ".yml"}


def _ignored(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in {"checkpoints", "__pycache__"} for part in relative.parts):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return path.suffix.lower() not in INCLUDED_SUFFIXES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--status", required=True)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    config = args.config.resolve()
    archive = args.archive.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not _ignored(p, root)) if root.exists() else []
    offline_task_ids = []
    offline_root = root / "offline_evaluation"
    if offline_root.is_dir():
        for task_dir in offline_root.glob("task_*"):
            try:
                task_id = int(task_dir.name.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if (task_dir / "oof.npz").is_file() and (task_dir / "test.npz").is_file():
                offline_task_ids.append(task_id)
    offline_task_ids.sort()
    summary = [
        f"# {args.label} — experiment review package",
        "",
        f"- Run root: `{root}`",
        f"- Config: `{config}`",
        f"- Status: `{args.status}`",
        (
            "- Offline OOF/test ensemble artifacts present for tasks: "
            f"`{offline_task_ids}` (NPZ arrays are intentionally excluded from this ZIP)."
            if offline_task_ids
            else "- Offline OOF/test ensemble artifacts: none found; this package does not claim an ensemble completed."
        ),
        f"- Full manifest present: `{(root / 'full_manifest.json').is_file()}`.",
        "- Included: configs, run/task summaries, logs, metric/audit CSV/JSON, visualization PNG/CSV.",
        "- Excluded: NPZ/NPY prediction arrays, checkpoints/model weights, and Parquet sources.",
        "",
        "## Included files",
        "",
    ]
    summary.extend(f"- `{p.relative_to(root).as_posix()}`" for p in files)
    if not files:
        summary.append("- No run artifacts were present; config and failure metadata are still included.")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("REVIEW_SUMMARY.md", "\n".join(summary) + "\n")
        zf.writestr(f"config/{config.name}", config.read_bytes())
        zf.writestr("experiment_status.txt", args.status.rstrip() + "\n")
        if args.log is not None and args.log.is_file():
            zf.write(args.log, f"logs/{args.log.name}")
        for path in files:
            zf.write(path, f"run/{path.relative_to(root).as_posix()}")
    hasher = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    print(f"[PACKAGE] {archive} ({archive.stat().st_size} bytes)")
    print(f"[SHA256] {digest}")


if __name__ == "__main__":
    main()
