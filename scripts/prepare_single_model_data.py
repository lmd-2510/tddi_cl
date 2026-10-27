"""Prepare the frozen Task-0 scaler and reusable original-split task cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ddi_dataset import load_feature_columns
from src.data.original_split import prepare_original_split_context, sha256_file
from src.data.single_preprocessing import load_or_prepare_single_preprocessing
from src.data.single_task_cache import ensure_single_task_cache
from src.data.task_protocols import validate_full_task_spec


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def prepare(config_path: str | Path, *, through_task: int = 7, scaler_only: bool = False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("kind") != "ddi_cil_original_split_preprocessing":
        raise ValueError("Expected an original-split preprocessing config.")
    if through_task not in range(8):
        raise ValueError("through-task must be between 0 and 7.")
    inputs = {key: _path(value) for key, value in config["inputs"].items()}
    task_file = _path(config["protocol"]["task_file"])
    columns = load_feature_columns(inputs["feature_cols"])
    context = prepare_original_split_context(
        train=inputs["train"], validation=inputs["validation"], test=inputs["test"],
    )
    tasks = validate_full_task_spec(json.loads(task_file.read_text(encoding="utf-8")))
    artifact_path = _path(config["preprocessing"]["artifact"])
    preprocessing = load_or_prepare_single_preprocessing(
        artifact_path, context=context, task_file=task_file, feature_columns=columns,
        policy=config["preprocessing"]["policy"],
    )
    print(f"[OK] Frozen Task-0 preprocessing: {artifact_path}", flush=True)
    if scaler_only:
        return
    task_hash = sha256_file(task_file)
    prep_hash = sha256_file(artifact_path)
    cache_root = _path(config["preprocessing"]["task_cache_root"])
    for task_id in range(through_task + 1):
        _, manifest = ensure_single_task_cache(
            cache_root, context=context, task_file_sha256=task_hash,
            preprocessing_sha256=prep_hash, feature_columns=columns,
            preprocessing=preprocessing, task_id=task_id,
            class_ids=[int(value) for value in tasks[task_id]["classes"]],
        )
        print(f"[OK] Task {task_id} cache: {manifest['rows']}", flush=True)
    print(f"[DONE] Shared method-neutral cache: {cache_root}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--through-task", type=int, default=7)
    parser.add_argument("--scaler-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.check:
        print(json.dumps({
            "config": str(args.config.resolve()),
            "preprocessing_artifact": str(_path(config["preprocessing"]["artifact"])),
            "task_cache_root": str(_path(config["preprocessing"]["task_cache_root"])),
            "through_task": args.through_task,
            "scaler_only": args.scaler_only,
            "model_training": False,
        }, indent=2))
        return
    prepare(args.config, through_task=args.through_task, scaler_only=args.scaler_only)


if __name__ == "__main__":
    main()
