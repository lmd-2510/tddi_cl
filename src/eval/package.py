"""Build a validated, lightweight metrics-review archive for a completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence
import uuid
import zipfile


PACKAGE_SCHEMA_VERSION = 1
PACKAGE_KIND = "ddi_cil_metrics_review_package"
INCLUDED_SUFFIXES = frozenset(
    {
        ".csv",
        ".json",
        ".md",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".yaml",
        ".yml",
    }
)
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {"checkpoints", "member_predictions"}
)
DIRECT_RESULT_FILENAMES = frozenset(
    {
        "metrics.csv",
        "metrics.json",
        "per_class_metrics.csv",
        "run_summary.json",
        "run_summary.md",
        "completed_task.json",
        "forgetting.csv",
        "class_forgetting.csv",
        "class_trajectory.csv",
    }
)
VISUALIZATION_SUFFIXES = frozenset({".csv", ".json", ".md", ".png", ".jpg", ".jpeg", ".svg"})
REQUIRED_FINAL_RESULT_SUFFIXES = (
    "_final_report.md",
    "_task_summary.csv",
    "_paper_table.csv",
    "_task_matrix.csv",
    "_forgetting_summary.csv",
    "_bwt_summary.csv",
    "_member_forgetting_summary.csv",
    "_diversity_summary.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_completed_run(run_root: Path) -> dict[str, str]:
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    required: dict[str, str] = {}
    final_results = run_root / "final_results"
    if not final_results.is_dir():
        raise FileNotFoundError(
            f"Missing final_results directory: {final_results}. Build the final report first."
        )
    reports = sorted(final_results.glob("*_final_report.md"))
    if len(reports) != 1:
        raise ValueError(
            f"Expected exactly one final_results/*_final_report.md, found {len(reports)}."
        )
    prefix = reports[0].name.removesuffix("_final_report.md")
    for suffix in REQUIRED_FINAL_RESULT_SUFFIXES:
        path = final_results / f"{prefix}{suffix}"
        if not path.is_file():
            raise ValueError(f"Missing required final result: {path}")
        required[suffix.removeprefix("_")] = path.relative_to(run_root).as_posix()

    for member_id in range(3):
        member_root = run_root / f"member_{member_id}"
        for filename in ("metrics.csv", "run_summary.json", "run_config.json"):
            path = member_root / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing required member artifact: {path}")
            required[f"member_{member_id}_{filename}"] = path.relative_to(run_root).as_posix()
    return required


def _is_review_artifact(path: Path, *, run_root: Path) -> bool:
    relative = path.relative_to(run_root)
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts[:-1]):
        return False
    if path.suffix.casefold() not in INCLUDED_SUFFIXES:
        return False
    if path.name.startswith(".") or ".tmp" in path.name:
        return False
    if path.name in {"PACKAGE_MANIFEST.json", "PACKAGE_SUMMARY.md"}:
        return False
    if relative.parts[0] == "final_results":
        return True
    if any(part.casefold().startswith("visualization") for part in relative.parts[:-1]):
        return path.suffix.casefold() in VISUALIZATION_SUFFIXES
    if path.name in DIRECT_RESULT_FILENAMES:
        return True

    stem = path.stem.casefold()
    suffix = path.suffix.casefold()
    if suffix in {".json", ".yaml", ".yml"} and (
        "config" in stem or "manifest" in stem
    ):
        return True
    if suffix == ".json" and ("threshold" in stem or "report" in stem):
        return True
    return False


def _selected_files(run_root: Path) -> list[Path]:
    return sorted(
        (path for path in run_root.rglob("*") if path.is_file() and _is_review_artifact(path, run_root=run_root)),
        key=lambda path: path.relative_to(run_root).as_posix(),
    )


def _manifest(
    run_root: Path,
    files: Sequence[Path],
    *,
    required: dict[str, str],
) -> dict[str, Any]:
    entries = [
        {
            "path": path.relative_to(run_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "artifact_kind": PACKAGE_KIND,
        "run_root_name": run_root.name,
        "selection_policy": {
            "included_suffixes": sorted(INCLUDED_SUFFIXES),
            "included_categories": [
                "final_results",
                "member_and_task_metrics",
                "run_summaries",
                "threshold_reports",
                "visualizations",
                "configs_and_manifests",
            ],
            "excluded_directories": sorted(EXCLUDED_DIRECTORY_NAMES),
            "excluded_artifacts": [
                "*.npz",
                "*.pt",
                "*.pth",
                "*.pkl",
                "*.parquet",
                "*.log",
                "training_audit.csv",
                "events.csv",
                "epoch_*_audit.json",
                "buffer_audit.json",
            ],
        },
        "required_artifacts": required,
        "included_file_count": len(entries),
        "included_bytes": int(sum(entry["size_bytes"] for entry in entries)),
        "files": entries,
    }


def _summary(manifest: dict[str, Any]) -> str:
    required = manifest["required_artifacts"]
    return "\n".join(
        [
            "# DDI-CIL metrics review package",
            "",
            f"- Run: `{manifest['run_root_name']}`",
            f"- Included files: `{manifest['included_file_count']}`",
            f"- Included bytes before ZIP: `{manifest['included_bytes']}`",
            "- Checkpoints, prediction NPZ files, preprocessing data, parquet files, long logs and detailed audits are excluded.",
            "- Every included file is listed with SHA256 in `PACKAGE_MANIFEST.json`.",
            "",
            "## Required final metrics",
            "",
            *[f"- `{name}`: `{path}`" for name, path in sorted(required.items())],
            "",
        ]
    )


def build_metrics_review_package(
    run_root: str | Path,
    *,
    archive_path: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Validate a completed full run and atomically write ZIP plus SHA256."""

    run_root = Path(run_root).resolve()
    required = _validate_completed_run(run_root)
    files = _selected_files(run_root)
    selected_relatives = {path.relative_to(run_root).as_posix() for path in files}
    missing_from_selection = sorted(set(required.values()) - selected_relatives)
    if missing_from_selection:
        raise RuntimeError(f"Required artifacts were excluded from the package: {missing_from_selection}")

    if archive_path is None:
        archive_path = run_root / f"{run_root.name}_metrics_review.zip"
    archive_path = Path(archive_path).resolve()
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if not overwrite and (archive_path.exists() or checksum_path.exists()):
        raise FileExistsError(
            f"Package already exists: {archive_path} or {checksum_path}. Use --overwrite."
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = _manifest(run_root, files, required=required)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    summary_bytes = _summary(manifest).encode("utf-8")
    temporary = archive_path.with_name(f".{archive_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr("PACKAGE_MANIFEST.json", manifest_bytes)
            archive.writestr("PACKAGE_SUMMARY.md", summary_bytes)
            for path in files:
                archive.write(path, path.relative_to(run_root).as_posix())
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)

    checksum = _sha256(archive_path)
    checksum_temporary = checksum_path.with_name(
        f".{checksum_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        checksum_temporary.write_text(
            f"{checksum}  {archive_path.name}\n", encoding="utf-8", newline="\n"
        )
        os.replace(checksum_temporary, checksum_path)
    finally:
        checksum_temporary.unlink(missing_ok=True)
    return archive_path, checksum_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package all lightweight metrics/report artifacts from one completed full run."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    archive, checksum = build_metrics_review_package(
        args.run_root,
        archive_path=args.archive,
        overwrite=args.overwrite,
    )
    print(f"Review package: {archive}")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
