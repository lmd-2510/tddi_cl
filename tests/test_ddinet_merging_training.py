"""Small real-data-flow runs for DDINet and TDDI merging methods."""
import json
from zipfile import ZipFile

import pytest
import torch

from tests.test_fold_training import data, tiny, cli
from src.data.ddinet_fingerprints import build_ddinet_fingerprint_cache
from src.data.fold_preprocessing import prepare_fold_preprocessing, save_fold_preprocessing
from src.models.ddinet_member import DDINetMember, DDINetMemberConfig, DDINET_SINGLETON_BN_POLICY
from src.models.tddi_paper_member import TDDIPaperMember, TDDIPaperMemberConfig
from src.training import train_cil as engine
from src.training import model_merging_training as merging
from src.training.model_merging_ensemble import build_ensemble
from src.training.replay_checkpoint import fold_state_digest


@pytest.mark.parametrize("data", ["full_test"], indirect=True)
@pytest.mark.parametrize("method", ["task_arithmetic", "ties", "opcm", "magmax"])
@pytest.mark.parametrize("variant", ["ddinet_multiclass_cil", "tddi_paper_member"])
def test_merging_eight_tasks_three_members_and_review(data, monkeypatch, tiny, method, variant):
    ddinet = variant == "ddinet_multiclass_cil"
    cache_root = data["root"] / "fingerprints"
    records = {"B": "CCO"}
    for split, count in (("train", 534), ("validation", 534), ("test", 178)):
        records.update({f"{split}_{index}": "CC" for index in range(count)})
    build_ddinet_fingerprint_cache(cache_root, records)
    monkeypatch.setattr(merging, "DEFAULT_META_COLS", ["drugid-drug_a", "drugid-drug_b"])

    def expand(previous, previous_map, class_map, **kwargs):
        if not ddinet:
            assert kwargs["input_dim"] == 2
            assert kwargs["method"] is None
            return TDDIPaperMember(TDDIPaperMemberConfig(
                input_dim=2, hidden_dims=(8, 4), num_classes=178, dropout=0.0,
            ))
        assert kwargs["input_dim"] == 2048
        assert kwargs["method"] == method
        return DDINetMember(DDINetMemberConfig(
            num_classes=178, hidden_dims=(8, 8, 4, 4), dropout=0.0,
            singleton_batchnorm_policy=DDINET_SINGLETON_BN_POLICY,
        ))

    monkeypatch.setattr(engine, "expand_model_for_seen_classes", expand)
    original_train = merging._train_expert
    expert_inputs = []
    events = []
    original_log = merging._log

    def log(message):
        if message.startswith("merge_complete boundary="):
            events.append(("merge", int(message.split("boundary=")[1].split()[0])))
        original_log(message)

    def train(*args, **kwargs):
        initial = kwargs["initial_state"]
        events.append(("train", kwargs["task_id"]))
        expert_inputs.append((kwargs["task_id"], None if initial is None else fold_state_digest(initial)))
        return original_train(*args, **kwargs)

    monkeypatch.setattr(merging, "_train_expert", train)
    monkeypatch.setattr(merging, "_log", log)
    root = data["root"] / method
    policy = "raw_identity" if ddinet else "task0_standard_frozen"
    for member in (0, 1, 2):
        prep = prepare_fold_preprocessing(
            data["context"], task_file=data["tasks"], feature_columns=["x", "y"],
            member_id=member, validation_fold=member, policy=policy,
        )
        prep_path = save_fold_preprocessing(prep, data["root"] / f"prep_{member}")
        command = cli(data, extra=[
            "--method", method, "--variant", variant,
            "--norm", "none" if ddinet else "layernorm", "--activation", "relu", "--dropout", "0.3",
            "--preprocessing-policy", policy,
            "--ddinet-fingerprint-cache", str(cache_root), "--member-id", str(member),
            "--fold-preprocessing", str(prep_path), "--epochs", "1", "--batch-size", "79",
            "--outdir", str(root / f"member_{member}"), "--stop-after-task", "7",
        ])
        args = engine.parse_args(command[1:])
        merging.run_model_merging_member(args, engine=engine)
        member_root = root / f"member_{member}"
        base = torch.load(member_root / "experts/task_0.pt", weights_only=True)["model_state"]
        calls = expert_inputs[member * 8:(member + 1) * 8]
        if method == "opcm":
            assert calls[0][1] is not None
            assert all(digest == calls[0][1] for _, digest in calls)
        elif method == "magmax":
            assert calls[0][1] is not None
            for task, digest in calls[1:]:
                previous = torch.load(member_root / f"experts/task_{task - 1}.pt", weights_only=True)["model_state"]
                assert digest == fold_state_digest(previous)
        else:
            assert calls[0] == (0, None)
            assert all(digest == fold_state_digest(base) for _, digest in calls[1:])
        if method in {"opcm", "magmax"}:
            assert events[member * 16:(member + 1) * 16] == [
                (phase, task) for task in range(8) for phase in ("train", "merge")
            ]
        for task in range(8):
            audit = json.loads((member_root / f"merged/task_{task}/merge_audit.json").read_text())
            assert audit["future_experts_used"] is False
            if ddinet:
                assert audit["model"]["singleton_batchnorm_policy"] == DDINET_SINGLETON_BN_POLICY
            else:
                assert "model" not in audit
            if method == "opcm" and task == 0:
                assert audit["batchnorm_policy"] == "expert_task0_state"
            else:
                assert audit["batchnorm_uses_stored_samples"] is False
        config = json.loads((member_root / "run_config.json").read_text())
        if ddinet:
            assert config["model"]["hidden_dims"] == [1024, 512, 256, 128]
            assert config["ddinet_fingerprint_manifest_sha256"]
        else:
            assert "ddinet_fingerprint_manifest_sha256" not in config

    archive = build_ensemble(root, method=method)
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert not any(name.endswith((".npz", ".pt")) for name in names)
        assert all(f"member_{member}/run_config.json" in names for member in (0, 1, 2))
        assert "ensemble_summary.json" in names
    summary = json.loads((root / "ensemble_summary.json").read_text())
    assert summary["tasks"] == 8 and summary["members"] == [0, 1, 2]
    assert summary["sequential_merge"] == (method in {"opcm", "magmax"})
    if ddinet:
        assert summary["model"]["variant"] == "ddinet_multiclass_cil"
    assert 0 <= summary["final_macro_f1"] <= 1

    if not ddinet:
        return
    config_path = root / "member_2/run_config.json"
    config["model"]["hidden_dims"][-1] = 256
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration mismatch"):
        build_ensemble(root, method=method)
