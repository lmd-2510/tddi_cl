"""MFP method recipes, task-boundary resume and full metrics packaging."""
from copy import deepcopy
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tests.test_fold_training import data, tiny
from scripts.build_mfp_fingerprint_cache import main as build_cache
from src.data.fold_preprocessing import prepare_fold_preprocessing, save_fold_preprocessing
from src.data.morgan_fingerprints import MorganFingerprintConfig, build_morgan_fingerprint_cache
from src.data.stratified_folds import fold_file_sha256
from src.eval import ensemble_ue, threshold
from src.eval.package import build_metrics_review_package
from src.eval.report import build_final_report
from src.models.mfp_siamese_member import (
    MFPSiameseMember, MFPSiameseMemberConfig, MFP_SIAMESE_SINGLETON_BN_POLICY,
)
from src.training import fold_training, train_cil as engine
from src.training.fold_ensemble3_full import load_full_config, execute_full
from src.training.fold_run_setup import member_command


@pytest.mark.parametrize("method", ["er", "agem", "ewc"])
def test_mfp_recipe_and_production_model(tmp_path, monkeypatch, method):
    monkeypatch.setenv("CIL_MFP_FINGERPRINT_CACHE", str(tmp_path / "mfp-cache"))
    config = load_full_config(f"configs/ordered_mfp_{method}.json")
    args = engine.parse_args(member_command(config, 0, python=sys.executable)[2:])
    assert args.method == method and args.variant == "mfp_siamese_member"
    assert args.batch_size == args.effective_batch_size == 512
    assert args.lr == 0.0001 and args.weight_decay == 0
    assert args.epochs == 200 and args.patience == 5
    if method == "er":
        assert args.replay_fraction == 0.5
    elif method == "agem":
        assert args.agem_reference_batch_size == 256
        assert config["replay"]["buffer_policy"] == "agem_task_uniform_v1"
        assert config["replay"]["fraction"] == 0
    else:
        assert args.ewc_lambda == 1000
        assert args.fold_member_budget == args.fold_global_budget == 0
        assert config["budget_policy"] == "no_exemplar_memory"
        assert config["replay"]["buffer_policy"] == "ewc_no_examples_v1"
        assert config["replay"]["fraction"] == 0
    assert args.mfp_fingerprint_cache == tmp_path / "mfp-cache"
    fold_training.validate_fold_options(args)
    model = engine.expand_model_for_seen_classes(
        None, None, {raw: raw for raw in range(178)},
        variant=args.variant, input_dim=4096, dropout=.3, activation="relu", norm="none", method=method,
    )
    assert model.config.drug_input_dim == 2048
    assert model.config.singleton_batchnorm_policy == MFP_SIAMESE_SINGLETON_BN_POLICY
    optimizer = engine.make_optimizer(model, args)
    assert type(optimizer) is torch.optim.Adam
    logits = model(torch.randn(1, 4096))
    assert logits.shape == (1, 178) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert model.encoder[0].weight.grad is not None
    manifest = engine.model_manifest(args)
    assert manifest["variant"] == "mfp_siamese_member"
    assert manifest["shared_encoder"]["dimensions"] == [2048, 512, 256]


@pytest.mark.parametrize("method", ["er", "agem", "ewc"])
@pytest.mark.parametrize("field,value", [("fraction", .25), ("repeat_cap", 7)])
def test_mfp_config_rejects_inconsistent_memory_recipe(tmp_path, method, field, value):
    config = json.loads(Path(f"configs/ordered_mfp_{method}.json").read_text())
    config["replay"][field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="memory/replay configuration"):
        load_full_config(path)


@pytest.mark.parametrize("method", ["er", "agem", "ewc"])
def test_mfp_launcher_worker_dispatches_method_and_postprocess(tmp_path, method):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if sys.platform == "win32" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("run_mfp_head_to_tail_full.sh", f"run_{method}_mfp_head_to_tail_full.sh"):
        shutil.copyfile(Path("scripts") / name, scripts / name)
    fake_python = tmp_path / "python-stub"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MFP_TEST_CALLS"\n', newline="\n")
    fake_python.chmod(0o755)
    run_home, output = tmp_path / "monitor", tmp_path / "output"
    run_home.mkdir()
    output.mkdir()
    calls = tmp_path / "calls.log"
    result = subprocess.run(
        [str(bash), str(scripts / f"run_{method}_mfp_head_to_tail_full.sh"),
         "_worker", run_home.as_posix(), output.as_posix()],
        env={**os.environ, "CIL_PYTHON": fake_python.as_posix(), "MFP_TEST_CALLS": calls.as_posix()},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = calls.read_text().splitlines()
    assert len(commands) == 4
    assert f"ordered_mfp_{method}.json" in commands[0]
    assert f"--method {method}" in commands[0] and "--execute" in commands[0]
    assert "src/eval/report.py" in commands[1]
    assert "visualize_cil_run.py" in commands[2]
    assert "src.eval.package" in commands[3]
    assert (run_home / "exit_code.txt").read_text().strip() == "0"


def test_mfp_cache_builder_reads_pair_columns_and_reuses_cache(tmp_path):
    paths = []
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.parquet"
        pq.write_table(pa.table({
            "drugid-drug_a": ["A"], "drugsmiles-drug_a": ["CC"],
            "drugid-drug_b": ["B"], "drugsmiles-drug_b": ["CCO"],
        }), path)
        paths.append(path)
    root = tmp_path / "cache"
    command = ["--train", str(paths[0]), "--validation", str(paths[1]),
               "--test", str(paths[2]), "--outdir", str(root)]
    build_cache(command)
    first = (root / "manifest.json").read_bytes()
    build_cache(command)
    assert (root / "manifest.json").read_bytes() == first
    manifest = json.loads(first)
    assert manifest["config"]["n_bits"] == 2048
    assert manifest["config"]["consumer"] == "mfp_siamese"


@pytest.mark.parametrize("data", ["full_test"], indirect=True)
@pytest.mark.parametrize("method", ["er", "agem", "ewc"])
def test_mfp_full_trajectory_resume_ensemble_and_zip(data, monkeypatch, tiny, method):
    records = {"B": "CCO"}
    for split, count in (("train", 534), ("validation", 534), ("test", 178)):
        records.update({f"{split}_{index}": "CC" for index in range(count)})
    cache_root = data["root"] / "mfp-cache"
    build_morgan_fingerprint_cache(
        cache_root, records, config=MorganFingerprintConfig(n_bits=2048, consumer="mfp_siamese"),
    )
    monkeypatch.setenv("CIL_MFP_FINGERPRINT_CACHE", str(cache_root))
    monkeypatch.delenv("CIL_SOURCE_SHARD_CACHE_ROOT", raising=False)
    monkeypatch.setattr(fold_training, "DEFAULT_META_COLS", ["drugid-drug_a", "drugid-drug_b"])

    def expand(previous, previous_map, class_map, **kwargs):
        assert kwargs["variant"] == "mfp_siamese_member" and kwargs["input_dim"] == 4096
        model = MFPSiameseMember(MFPSiameseMemberConfig(
            encoder_hidden_dim=8, drug_latent_dim=4, classifier_hidden_dim=8,
            num_classes=len(class_map), dropout=0,
            singleton_batchnorm_policy=MFP_SIAMESE_SINGLETON_BN_POLICY,
        ))
        if previous is not None:
            model.load_state_dict(previous.state_dict())
        return model

    monkeypatch.setattr(engine, "expand_model_for_seen_classes", expand)
    reference_shapes = []
    original_projection = engine.apply_agem_projection

    def project(model, criterion, memory_features, *args, **kwargs):
        reference_shapes.append(memory_features.shape)
        return original_projection(model, criterion, memory_features, *args, **kwargs)

    monkeypatch.setattr(engine, "apply_agem_projection", project)
    config = load_full_config(f"configs/ordered_mfp_{method}.json", overrides={
        "output_root": data["root"] / "full", "device": "cpu",
    })
    config["inputs"] = {**{name: str(path) for name, path in data["paths"].items()},
                        "feature_cols": str(data["cols"]),
                        "fold_assignments": str(data["folds"] / "fold_assignments.parquet"),
                        "fold_manifest": str(data["folds"] / "fold_manifest.json")}
    config["development_count"] = data["context"].manifest["assignment_row_count"]
    config["protocol"].update(task_file=str(data["tasks"]), sha256=fold_file_sha256(data["tasks"]))
    config["tasks"] = json.loads(data["tasks"].read_text())["tasks"]
    config["training"]["epochs"] = 1
    config["preprocessing"]["artifact_template"] = str(data["root"] / "prep_{member_id}/fold_preprocessing.json")
    for member in (0, 1, 2):
        prep = prepare_fold_preprocessing(
            data["context"], task_file=data["tasks"], feature_columns=["x", "y"],
            member_id=member, validation_fold=member, policy="raw_identity",
        )
        save_fold_preprocessing(prep, data["root"] / f"prep_{member}")

    def runner(command, cwd):
        script = Path(command[1]).name
        if script == "train_cil.py":
            args = engine.parse_args(command[2:])
            if args.member_id == 0:
                first = deepcopy(args)
                first.stop_after_task = 0
                fold_training.run_fold_training(first, engine=engine)
                args.resume_fold_checkpoint = Path(args.outdir) / "checkpoints/task_0.pt"
            fold_training.run_fold_training(args, engine=engine)
        elif script == "ensemble_ue.py":
            with monkeypatch.context() as patch:
                patch.setattr(sys, "argv", list(command[1:]))
                ensemble_ue.main()
        elif script == "threshold.py":
            threshold.main(command[2:])
        else:
            raise AssertionError(f"Unexpected script: {script}")

    manifest_path = execute_full(config, runner=runner)
    assert manifest_path.is_file()
    root = Path(config["output_root"])
    for member in (0, 1, 2):
        state = torch.load(root / f"member_{member}/checkpoints/task_7.pt", weights_only=False)
        assert state["contract"]["model"]["variant"] == "mfp_siamese_member"
        assert state["contract"]["optimizer"] == "Adam_new_each_task"
        assert state["model_state"]["head.weight"].shape[0] == 178
        memory_policy = {"er": "reservoir_uniform_v1", "agem": "agem_task_uniform_v1", "ewc": "ewc_no_examples_v1"}
        assert state["buffer_state"]["metadata"]["policy"] == memory_policy[method]
        if method == "agem":
            assert state["contract"]["sampler"]["replay_fraction"] == 0
            audit = json.loads((root / f"member_{member}/task_1/input_audit.json").read_text())
            assert audit["sampler_replay_ids_before"] == []
            assert state["progress"]["task_summaries"][1]["memory_before"] > 0
        if method == "ewc":
            assert state["contract"]["global_slot_budget"] == 0
            assert set(state["method_state"]) == {"fisher", "theta_star"}
            fisher = state["method_state"]["fisher"]
            assert fisher and all(torch.isfinite(tensor).all() for tensor in fisher.values())
            assert not any(key.endswith(("running_mean", "running_var", "num_batches_tracked")) for key in fisher)
            assert all(task["memory_after"] == 0 for task in state["progress"]["task_summaries"])
        assert state["contract"]["mfp_fingerprint_manifest_sha256"]
    if method == "agem":
        assert reference_shapes
        assert all(0 < shape[0] <= 256 and shape[1] == 4096 for shape in reference_shapes)
    build_final_report(full_root=root, task_file=data["tasks"])
    task_metrics = pd.read_csv(root / "final_results/ensemble3_ordered_task_summary.csv")
    assert task_metrics["task_id"].tolist() == list(range(8))
    assert np.isfinite(task_metrics["full_macro_f1"]).all()
    archive, checksum = build_metrics_review_package(root)
    assert checksum.is_file()
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert any(name.endswith("_bwt_summary.csv") for name in names)
        assert not any(name.endswith((".npz", ".pt", ".npy")) for name in names)
