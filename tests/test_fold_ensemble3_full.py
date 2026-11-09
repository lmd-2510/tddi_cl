from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from src.training import fold_ensemble3_runner as shared
from src.training.fold_ensemble3_full import (
    MANIFEST_KIND,
    MEMBER_IDS,
    execute_full,
    load_full_config,
)


ORDERED_CONFIG = Path("configs/ordered_baseline.json")
SHUFFLED_CONFIG = Path("configs/shuffled_seed0_baseline.json")


def _arg(command: Sequence[str], name: str) -> str:
    return command[command.index(name) + 1]


def _config(tmp_path: Path):
    return load_full_config(
        ORDERED_CONFIG,
        overrides={"output_root": tmp_path / "full"},
    )


def _inspector(states: dict[int, str], resume: Path | None = None):
    def inspect(config, member, command, *, require_inputs=False):
        status = states[member]
        return {
            "status": status,
            "completed_task_id": 7 if status == "complete" else (4 if status == "resume" else None),
            "resume_checkpoint": str(resume) if status == "resume" else None,
        }

    return inspect


def test_canonical_baseline_is_locked_to_eight_tasks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    plan = shared.build_ensemble_plan(
        config,
        python="full-python",
        inspector=_inspector({member: "fresh" for member in MEMBER_IDS}),
    )

    assert config["phase"] == "full"
    assert config["protocol"]["name"] == "head_to_tail"
    assert config["execution"]["stop_after_task"] == 7
    assert config["training"]["epochs"] == 30
    assert config["training"]["loss_variant"] == "focal_current_ce_replay"
    assert config["replay"]["buffer_policy"] == "equal_class_fixed_budget"
    assert config["member_budgets"] == {"0": 27778, "1": 27778, "2": 27778}
    assert config["evaluation"]["threshold_config"].endswith(
        "eval_macro_f1_entropy.json"
    )
    assert len(plan.evaluations) == 8
    for member in plan.members:
        assert member.command is not None
        assert _arg(member.command, "--stop-after-task") == "7"
        assert _arg(member.command, "--epochs") == "30"
        assert _arg(member.command, "--loss-variant") == "focal_current_ce_replay"
        assert _arg(member.command, "--buffer-policy") == "equal_class_fixed_budget"


def test_shuffled_changes_only_protocol_preprocessing_and_output(tmp_path: Path) -> None:
    ordered = load_full_config(
        ORDERED_CONFIG,
        overrides={"output_root": tmp_path / "ordered"},
    )
    shuffled = load_full_config(
        SHUFFLED_CONFIG,
        overrides={"output_root": tmp_path / "shuffled"},
    )

    assert shuffled["protocol"]["id"] == "SHUFFLED"
    assert shuffled["protocol"]["name"] == "shuffled"
    assert shuffled["training"] == ordered["training"]
    assert shuffled["replay"] == ordered["replay"]
    assert shuffled["model"] == ordered["model"]
    assert shuffled["evaluation"] == ordered["evaluation"]
    assert "preprocessing_shuffled_seed0_fold42" in shuffled["preprocessing"][
        "artifact_template"
    ]


def test_ider_full_override_uses_paper_defaults_and_equal_class_memory(tmp_path: Path) -> None:
    config = load_full_config(
        ORDERED_CONFIG,
        overrides={"output_root": tmp_path / "ider", "method": "ider"},
    )
    plan = shared.build_ensemble_plan(
        config,
        python="full-python",
        inspector=_inspector({member: "fresh" for member in MEMBER_IDS}),
    )
    command = plan.members[0].command
    assert command is not None
    assert config["training"]["loss_variant"] == "ider"
    assert config["training"]["ider_weight_a"] == 0.5
    assert config["training"]["ider_weight_b"] == 0.4
    assert config["training"]["ider_weight_c"] == 0.5
    assert _arg(command, "--buffer-policy") == "equal_class_fixed_budget"
    assert _arg(command, "--ider-uniform-probability") == "1.0"
    assert _arg(command, "--ider-replay-batch-size") == "64"


@pytest.mark.parametrize("method", ["agem", "er", "ider", "upcl"])
def test_ddinet_full_plan_reuses_fingerprints_and_method_defaults(tmp_path, monkeypatch, method):
    from src.training import train_cil
    fingerprint_root = tmp_path / "fingerprints"
    monkeypatch.setenv("CIL_DDINET_FINGERPRINT_CACHE", str(fingerprint_root))
    monkeypatch.setenv("CIL_COMMON_TASK_CACHE_ROOT", str(tmp_path / "tddi-cache"))
    config = load_full_config(
        "configs/ordered_ddinet_ewc.json",
        overrides={"method": method, "output_root": tmp_path / method},
    )
    plan = shared.build_ensemble_plan(
        config, inspector=_inspector({member: "fresh" for member in MEMBER_IDS}),
    )
    assert len(plan.evaluations) == 8
    assert config["global_slot_budget"] == 83334
    for member in plan.members:
        args = train_cil.parse_args(member.command[2:])
        assert args.method == method
        assert args.variant == "ddinet_multiclass_cil"
        assert args.fold_member_budget == 27778
        assert args.agem_reference_batch_size == 256
        assert args.loss_variant == (method if method in {"ider", "upcl"} else "cross_entropy")
        assert args.ddinet_fingerprint_cache == fingerprint_root
        assert args.task_cache_root is None
        assert args.preprocessing_policy == "raw_identity"
        assert args.stop_after_task == 7
        if method == "er":
            assert args.buffer_policy == "reservoir_uniform_v1"
            assert args.exemplar_ranking_policy == "reservoir_uniform_v1"
            assert args.fold_replay_policy == "er_uniform_reservoir_equal_batch_v1"
            assert args.replay_fraction == 0.5
            assert args.replay_repeat_cap == 1
        if method == "ider":
            assert (args.ider_weight_a, args.ider_weight_b, args.ider_weight_c) == (0.5, 0.4, 0.5)
            assert args.ider_replay_batch_size == 64
            assert args.ider_uniform_probability == 1.0
            assert args.buffer_policy == "equal_class_fixed_budget"
        if method == "upcl":
            assert config["model"]["hidden_dims"] == [1024, 512, 256, 256]
            assert train_cil.model_manifest(args)["hidden_dims"] == config["model"]["hidden_dims"]
            assert args.upcl_temperature == 0.07
            assert args.replay_fraction == 0.0
            assert args.buffer_policy == "equal_class_fixed_budget"


def test_ddinet_upcl_production_head_supports_all_178_prototypes():
    import torch
    from src.training import train_cil
    from src.methods.upcl import initialise_prototype_state, _orthogonal_candidates
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = train_cil.expand_model_for_seen_classes(
            None, None, {i: i for i in range(178)}, variant="ddinet_multiclass_cil",
            input_dim=2048, dropout=.3, activation="relu", norm="none", method="upcl",
        )
        head = model.classifier_head
        assert (head.in_features, head.out_features) == (256, 178)
        assert not head.weight.requires_grad
        state = initialise_prototype_state(total_classes=178, latent_dim=head.in_features)
        prototypes = _orthogonal_candidates(
            state["prototypes"][state["assigned"]], count=178, latent_dim=head.in_features,
            generator=torch.Generator().manual_seed(0),
        )
        torch.testing.assert_close(prototypes @ prototypes.T, torch.eye(178), atol=1e-5, rtol=1e-5)
        assert model.eval()(torch.zeros(2, 2048)).shape == (2, 178)
    finally:
        torch.set_num_threads(old_threads)


def test_full_selected_member_skip_and_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    resume = tmp_path / "task_4.pt"
    states = {0: "complete", 1: "resume", 2: "fresh"}
    plan = shared.build_ensemble_plan(
        config,
        member_ids=[0, 1],
        python="full-python",
        inspector=_inspector(states, resume),
    )

    assert plan.members[0].command is None
    assert plan.members[1].command is not None
    assert _arg(plan.members[1].command, "--resume-fold-checkpoint") == str(resume)
    assert plan.members[2].command is None


def test_offline_ensemble_rejects_different_completed_member_contracts(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def inspect(config, member, command, *, require_inputs=False):
        return {
            "status": "complete",
            "completed_task_id": 7,
            "resume_checkpoint": None,
            "implementation_sha256_digest": f"implementation-{member}",
        }

    with pytest.raises(ValueError, match="different frozen contracts"):
        shared.build_ensemble_plan(config, inspector=inspect)


def test_offline_plan_accepts_reviewed_checkpoint_reader_fix(tmp_path):
    from src.eval.study_contract import CHECKPOINT_READER_HASHES, compatible_implementation, contract_digest
    hashes = sorted(CHECKPOINT_READER_HASHES)

    def inspect(config, member, command, *, require_inputs=False):
        implementation = {
            "training/replay_checkpoint.py": hashes[member % 2],
            "training/train_cil.py": "same-training-code",
        }
        return {
            "status": "complete", "completed_task_id": 7, "resume_checkpoint": None,
            "implementation_sha256_digest": contract_digest(implementation),
            "compatible_implementation_sha256_digest": contract_digest(compatible_implementation(implementation)),
        }

    plan = shared.build_ensemble_plan(_config(tmp_path), inspector=inspect)
    assert all(member.status == "complete" for member in plan.members)


def test_full_members_are_sequential_then_all_eight_tasks_are_evaluated(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    states = {member: "fresh" for member in MEMBER_IDS}
    calls: list[tuple[str, int, str | None]] = []

    def runner(command: Sequence[str], cwd: Path) -> None:
        member = int(_arg(command, "--member-id"))
        calls.append(("train", member, None))
        root = Path(_arg(command, "--outdir"))
        root.mkdir(parents=True)
        (root / "run_config.json").write_text(
            json.dumps({"run_id": f"full-member-{member}"}), encoding="utf-8"
        )
        states[member] = "complete"

    def fake_validate(config, member):
        calls.append(("validate", member, None))

    def fake_ensemble(config, task, split, python, runner):
        calls.append(("ensemble", task, split))
        root = Path(config["output_root"]) / "offline_evaluation" / f"task_{task}"
        root.mkdir(parents=True, exist_ok=True)
        path = root / ("oof.npz" if split == "validation" else "test.npz")
        path.write_bytes(b"synthetic ensemble")
        return path

    def fake_thresholds(config, task, python, runner):
        calls.append(("threshold", task, None))
        root = Path(config["output_root"]) / "offline_evaluation" / f"task_{task}"
        for name in (
            "frozen_threshold.json",
            "oof_threshold_report.json",
            "test_threshold_report.json",
        ):
            (root / name).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(shared, "_validate_member_predictions", fake_validate)
    monkeypatch.setattr(shared, "_run_ensemble", fake_ensemble)
    monkeypatch.setattr(shared, "_run_thresholds", fake_thresholds)

    manifest_path = execute_full(
        config,
        python="full-python",
        runner=runner,
        inspector=_inspector(states),
    )

    assert [call for call in calls if call[0] == "train"] == [
        ("train", 0, None),
        ("train", 1, None),
        ("train", 2, None),
    ]
    first_eval = next(index for index, call in enumerate(calls) if call[0] == "ensemble")
    last_train = max(index for index, call in enumerate(calls) if call[0] == "train")
    assert first_eval > last_train
    assert [call[1] for call in calls if call[0] == "threshold"] == list(range(8))
    assert manifest_path is not None and manifest_path.name == "full_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == MANIFEST_KIND
    assert manifest["study_scope"] == {
        "completed_tasks": list(range(8)),
        "full_protocol_task_count": 8,
    }
    assert len(manifest["evaluations"]) == 8
