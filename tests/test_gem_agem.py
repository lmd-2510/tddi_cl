from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.methods.agem import project_agem_gradient
from src.methods.gem import (
    TaskEpisodicMemory,
    assign_gradients,
    flatten_gradients,
    project_gem_gradient,
    trainable_parameters,
)
from src.training.train_cil import (
    FocalLoss,
    apply_episodic_gradient_projection,
    compute_episodic_reference_gradient,
    main,
    train_one_epoch,
)


class GradientProjectionTest(unittest.TestCase):
    def test_gem_is_identity_when_all_constraints_are_satisfied(self) -> None:
        current = torch.tensor([2.0, 1.0])
        references = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        projected = project_gem_gradient(current, references)
        torch.testing.assert_close(projected, current)

    def test_gem_projects_onto_all_conflicting_constraints(self) -> None:
        current = torch.tensor([-1.0, -2.0])
        references = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        )
        projected = project_gem_gradient(current, references)
        self.assertTrue(bool(torch.all(references @ projected >= -1e-6)))
        torch.testing.assert_close(projected, torch.zeros(2), atol=1e-5, rtol=0)

    def test_agem_uses_closed_form_projection_and_handles_zero_reference(self) -> None:
        current = torch.tensor([1.0, -2.0])
        reference = torch.tensor([0.0, 1.0])
        projected = project_agem_gradient(current, reference)
        torch.testing.assert_close(projected, torch.tensor([1.0, 0.0]))
        self.assertGreaterEqual(float(torch.dot(projected, reference)), -1e-7)
        torch.testing.assert_close(
            project_agem_gradient(current, torch.zeros_like(reference)),
            current,
        )

    def test_projections_reject_nonfinite_gradients(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            project_gem_gradient(
                torch.tensor([float("nan")]),
                torch.tensor([[1.0]]),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            project_agem_gradient(
                torch.tensor([1.0]),
                torch.tensor([float("inf")]),
            )

    def test_flatten_and_assign_preserve_parameter_layout_with_none_gradients(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        parameters = trainable_parameters(model)
        parameters[0].grad = torch.ones_like(parameters[0])
        parameters[1].grad = None
        parameters[2].grad = torch.full_like(parameters[2], 2.0)
        parameters[3].grad = None

        flattened = flatten_gradients(parameters)
        replacement = torch.arange(flattened.numel(), dtype=flattened.dtype)
        assign_gradients(parameters, replacement)
        torch.testing.assert_close(flatten_gradients(parameters), replacement)


class EpisodicMemoryTest(unittest.TestCase):
    def test_memory_is_task_separated_deterministic_and_within_budget(self) -> None:
        features = np.arange(40, dtype=np.float32).reshape(10, 4)
        labels = np.repeat(np.asarray([10, 20], dtype=np.int64), 5)
        first = TaskEpisodicMemory(total_budget=5, total_tasks=2, random_seed=7)
        second = TaskEpisodicMemory(total_budget=5, total_tasks=2, random_seed=7)

        for memory in [first, second]:
            memory.update(0, features, labels)
            memory.update(1, features + 100, labels + 100)

        self.assertEqual(first.quota_for_task(0), 3)
        self.assertEqual(first.quota_for_task(1), 2)
        self.assertEqual(first.total_size, 5)
        self.assertEqual(first.task_ids, [0, 1])
        np.testing.assert_array_equal(first.get_task(0)[0], second.get_task(0)[0])
        np.testing.assert_array_equal(first.get_task(1)[1], second.get_task(1)[1])

        sample_a = first.sample(3, seed_components=[1, 2, 3])
        sample_b = first.sample(3, seed_components=[1, 2, 3])
        np.testing.assert_array_equal(sample_a[0], sample_b[0])
        np.testing.assert_array_equal(sample_a[1], sample_b[1])

    def test_memory_artifacts_include_task_ids(self) -> None:
        memory = TaskEpisodicMemory(total_budget=4, total_tasks=2)
        memory.update(
            0,
            np.arange(12, dtype=np.float32).reshape(4, 3),
            np.asarray([4, 4, 8, 8], dtype=np.int64),
        )
        with tempfile.TemporaryDirectory() as tempdir:
            summary_path = Path(tempdir) / "summary.csv"
            snapshot_path = Path(tempdir) / "snapshot.parquet"
            memory.save_summary(summary_path)
            memory.save_snapshot(snapshot_path)
            summary = pd.read_csv(summary_path)
            snapshot = pd.read_parquet(snapshot_path)
        self.assertEqual(summary["memory_count"].sum(), 2)
        self.assertEqual(snapshot["task_id"].unique().tolist(), [0])


class TrainingIntegrationTest(unittest.TestCase):
    def test_chunked_reference_gradient_matches_full_batch_and_preserves_current_grad(self) -> None:
        torch.manual_seed(2)
        model = nn.Linear(3, 2)
        criterion = FocalLoss(gamma=0.0)
        parameters = trainable_parameters(model)
        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)
        current_before = flatten_gradients(parameters)
        features = np.arange(15, dtype=np.float32).reshape(5, 3) / 10
        labels = np.asarray([10, 20, 10, 20, 10], dtype=np.int64)
        class_map = {10: 0, 20: 1}

        chunked = compute_episodic_reference_gradient(
            model,
            criterion,
            features,
            labels,
            class_map,
            "cpu",
            parameters,
            reference_batch_size=2,
        )
        full = compute_episodic_reference_gradient(
            model,
            criterion,
            features,
            labels,
            class_map,
            "cpu",
            parameters,
            reference_batch_size=10,
        )
        torch.testing.assert_close(chunked, full, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(flatten_gradients(parameters), current_before)
        self.assertTrue(model.training)

    def test_projection_hook_runs_once_per_optimizer_step_under_accumulation(self) -> None:
        model = nn.Linear(3, 2)
        loader = DataLoader(
            TensorDataset(torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 1])),
            batch_size=2,
            shuffle=False,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        calls: list[int] = []
        train_one_epoch(
            model,
            loader,
            optimizer,
            FocalLoss(gamma=0.0),
            "cpu",
            gradient_accumulation_steps=2,
            gradient_projector=calls.append,
        )
        self.assertEqual(calls, [0, 1])

    def test_gem_and_agem_orchestration_replace_conflicting_gradients(self) -> None:
        for method in ["gem", "agem"]:
            model = nn.Linear(1, 2)
            criterion = FocalLoss(gamma=0.0)
            memory = TaskEpisodicMemory(total_budget=2, total_tasks=2)
            memory.update(
                0,
                np.asarray([[1.0], [2.0]], dtype=np.float32),
                np.asarray([10, 10], dtype=np.int64),
            )
            parameters = trainable_parameters(model)
            reference = compute_episodic_reference_gradient(
                model,
                criterion,
                *memory.get_task(0),
                {10: 0, 20: 1},
                "cpu",
                parameters,
                reference_batch_size=2,
            )
            assign_gradients(parameters, -reference)
            audit = apply_episodic_gradient_projection(
                model,
                criterion,
                memory,
                {10: 0, 20: 1},
                "cpu",
                method=method,
                gem_reference_batch_size=2,
                agem_reference_batch_size=2,
                seed=0,
                task_id=1,
                epoch=1,
                optimizer_step=0,
            )
            self.assertTrue(audit.violated)
            self.assertGreaterEqual(audit.minimum_dot_after, -1e-6)


class TinyCLISmokeTest(unittest.TestCase):
    def _write_smoke_inputs(self, root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
        feature_columns = ["f0", "f1"]
        train = pd.DataFrame(
            {
                "f0": [-1.2, -1.0, -0.8, -0.6, 0.6, 0.8, 1.0, 1.2],
                "f1": [-1.0, -0.9, -0.7, -0.5, 0.5, 0.7, 0.9, 1.0],
                "class": [10, 10, 10, 10, 20, 20, 20, 20],
            }
        )
        validation = train.groupby("class", sort=True).head(2).reset_index(drop=True)
        test = train.groupby("class", sort=True).tail(2).reset_index(drop=True)
        split_paths = []
        for name, frame in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]:
            path = root / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            split_paths.append(path)

        feature_path = root / "features.json"
        feature_path.write_text(json.dumps(feature_columns), encoding="utf-8")
        scaler_path = root / "scaler.pkl"
        with scaler_path.open("wb") as handle:
            pickle.dump(
                {
                    "scaler_type": "standard",
                    "mean": np.zeros(2),
                    "scale": np.ones(2),
                    "impute_values": np.zeros(2),
                },
                handle,
            )
        task_path = root / "smoke_tasks.json"
        task_path.write_text(
            json.dumps(
                {
                    "protocol": "gem_agem_smoke",
                    "seed": 0,
                    "tasks": [
                        {"task_id": 0, "classes": [10]},
                        {"task_id": 1, "classes": [20]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return (*split_paths, feature_path, scaler_path, task_path)

    def test_gem_and_agem_smoke_write_complete_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            train, validation, test, features, scaler, tasks = self._write_smoke_inputs(root)
            for method in ["gem", "agem"]:
                outdir = root / method
                argv = [
                    "train_cil.py",
                    "--train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--test",
                    str(test),
                    "--feature-cols",
                    str(features),
                    "--scaler",
                    str(scaler),
                    "--task-file",
                    str(tasks),
                    "--outdir",
                    str(outdir),
                    "--method",
                    method,
                    "--variant",
                    "small",
                    "--batch-size",
                    "4",
                    "--epochs",
                    "1",
                    "--patience",
                    "1",
                    "--device",
                    "cpu",
                    "--dropout",
                    "0",
                    "--norm",
                    "none",
                    "--focal-gamma",
                    "0",
                    "--episodic-memory-budget",
                    "4",
                    "--agem-reference-batch-size",
                    "2",
                ]
                with patch.object(sys, "argv", argv):
                    main()

                required = [
                    "run_config.json",
                    "run_summary.md",
                    "task_matrix.csv",
                    "forgetting.csv",
                    "metrics.csv",
                    "training_audit.csv",
                    "checkpoints/task_0_model.pt",
                    "checkpoints/task_1_model.pt",
                    "memory/memory_summary.csv",
                    "memory/memory_after_task_1.parquet",
                ]
                for relative_path in required:
                    self.assertTrue((outdir / relative_path).is_file(), relative_path)
                config = json.loads((outdir / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual(config["arguments"]["method"], method)
                self.assertEqual(config["resolved"]["task_protocol"], "gem_agem_smoke")
                self.assertEqual(config["resolved"]["memory_budget"], 4)
                audit = pd.read_csv(outdir / "training_audit.csv")
                self.assertEqual(audit.shape[0], 2)
                self.assertTrue(bool(audit["gradient_projection_active"].all()))
                task_one = audit[audit["task"] == 1].iloc[0]
                self.assertGreater(int(task_one["gradient_projection_steps_checked"]), 0)
                self.assertGreater(int(task_one["gradient_constraint_checks"]), 0)
                self.assertGreater(int(task_one["memory_before"]), 0)
                final_memory = pd.read_parquet(
                    outdir / "memory" / "memory_after_task_1.parquet"
                )
                self.assertEqual(final_memory.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
