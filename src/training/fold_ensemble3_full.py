#!/usr/bin/env python3
"""Sequential full P3 frozen-fold ensemble study orchestration.

The command is a read-only dry run unless ``--execute`` is supplied. Training is
delegated to ``train_cil.py`` one member at a time. Offline OOF/test ensemble,
uncertainty and frozen-threshold evaluation start only after all three members
have a verified task-7 boundary and prediction artifacts for tasks 0--7.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.fold_ab_study import INPUT_KEYS  # noqa: E402
from src.training.fold_ensemble3_pilot import (  # noqa: E402
    Inspector,
    MEMBER_IDS,
    Runner,
    _default_runner,
    _load_locked_config,
    build_pilot_plan,
    execute_ensemble_study,
    inspect_member,
)


CONFIG_KIND = "tddi_frozen_fold_ensemble3_full"
STOP_AFTER_TASK = 7
MANIFEST_NAME = "full_manifest.json"
MANIFEST_KIND = "ddi_cil_frozen_fold_ensemble3_full_manifest"


def load_full_config(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Load the locked approved eight-task study configuration."""

    return _load_locked_config(
        path,
        expected_kind=CONFIG_KIND,
        expected_phase="full",
        expected_stop_after_task=STOP_AFTER_TASK,
        project_root=project_root,
        overrides=overrides,
    )


def execute_full(
    config: Mapping[str, Any],
    *,
    member_ids: Sequence[int] | None = None,
    python: str = sys.executable,
    runner: Runner = _default_runner,
    inspector: Inspector = inspect_member,
) -> Path | None:
    """Run selected full trajectories sequentially and evaluate when all finish."""

    return execute_ensemble_study(
        config,
        member_ids=member_ids,
        python=python,
        runner=runner,
        inspector=inspector,
        manifest_name=MANIFEST_NAME,
        manifest_kind=MANIFEST_KIND,
        limitations=(
            "This is one experiment seed; three ensemble members are not three independent experiment seeds.",
            "P3 tail-to-head results must be reported with per-task/old-class forgetting, not final accuracy alone.",
            "OOF target accuracy does not guarantee the same selected test accuracy.",
        ),
    )


def print_dry_run(config: Mapping[str, Any], *, member_ids: Sequence[int] | None, python: str) -> None:
    plan = build_pilot_plan(config, member_ids=member_ids, python=python)
    print(
        "DRY RUN: no model/training/writes. Members run synchronously 0->1->2; "
        "full scope task 0-7."
    )
    for member in plan.members:
        print(
            f"member={member.member_id} member_seed={member.member_seed} "
            f"status={member.status} completed_task={member.completed_task_id}"
        )
        if member.command:
            print(shlex.join(member.command))
    print("Offline OOF/test ensemble, UE and frozen thresholds wait for all three members.")
    for evaluation in plan.evaluations:
        print(f"task={evaluation.task_id} evaluation_status={evaluation.status}")
        for command in evaluation.commands:
            print(shlex.join(command))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--member-id", type=int, choices=MEMBER_IDS, action="append", dest="member_ids"
    )
    parser.add_argument("--python", default=sys.executable)
    for name in sorted(
        INPUT_KEYS
        | {"task_file", "preprocessing_root", "output_root", "threshold_config"}
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    override_names = INPUT_KEYS | {
        "task_file",
        "preprocessing_root",
        "output_root",
        "threshold_config",
        "device",
    }
    config = load_full_config(
        args.config,
        overrides={name: getattr(args, name) for name in override_names},
    )
    if not args.execute:
        print_dry_run(config, member_ids=args.member_ids, python=args.python)
        return
    result = execute_full(config, member_ids=args.member_ids, python=args.python)
    if result is None:
        print("Selected member(s) complete; offline evaluation waits for all three members.")
    else:
        print(f"Full study complete: {result}")


if __name__ == "__main__":
    main()
