from __future__ import annotations

import json
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.class_mapping import build_seen_class_map
from src.data.fixed_budget_replay import FixedBudgetReplayBuffer, FixedReplaySampler
from src.models.mlp import MLP, preset_config
from src.training.train_cil import (
    FIXED_BUDGET_METHOD,
    fixed_replay_seed_provenance,
    parse_args,
    seed_provenance,
    write_run_config,
)
from src.utils.seed import (
    LEGACY_SEED_DERIVATION,
    MEMBER_SEED_DERIVATION,
    derive_member_seed,
    resolve_seed_configuration,
    set_configured_seeds,
    set_global_seed,
)


class SeedDerivationTest(unittest.TestCase):
    def test_same_experiment_and_member_produce_same_seed(self) -> None:
        first = derive_member_seed(17, 2)
        second = derive_member_seed(17, 2)
        self.assertEqual(first, second)
        self.assertEqual(
            resolve_seed_configuration(17, 2).derivation,
            MEMBER_SEED_DERIVATION,
        )

    def test_different_member_ids_change_model_rng_not_experiment_rng(self) -> None:
        first = resolve_seed_configuration(23, 0)
        second = resolve_seed_configuration(23, 1)
        self.assertNotEqual(first.member_seed, second.member_seed)

        set_configured_seeds(first)
        first_python = random.random()
        first_numpy = np.random.random()
        first_model = MLP(
            preset_config("small", input_dim=4, num_classes=2, dropout=0.0)
        )
        first_weight = first_model.head.weight.detach().clone()

        set_configured_seeds(second)
        second_python = random.random()
        second_numpy = np.random.random()
        second_model = MLP(
            preset_config("small", input_dim=4, num_classes=2, dropout=0.0)
        )
        second_weight = second_model.head.weight.detach().clone()

        self.assertEqual(first_python, second_python)
        self.assertEqual(first_numpy, second_numpy)
        self.assertFalse(torch.equal(first_weight, second_weight))

        dataset = TensorDataset(torch.arange(20))
        set_configured_seeds(first)
        first_order = torch.cat(
            [batch[0] for batch in DataLoader(dataset, batch_size=5, shuffle=True)]
        )
        set_configured_seeds(second)
        second_order = torch.cat(
            [batch[0] for batch in DataLoader(dataset, batch_size=5, shuffle=True)]
        )
        self.assertFalse(torch.equal(first_order, second_order))

    def test_legacy_seed_configuration_matches_previous_global_behavior(self) -> None:
        configuration = resolve_seed_configuration(31)
        self.assertIsNone(configuration.member_id)
        self.assertEqual(configuration.experiment_seed, 31)
        self.assertEqual(configuration.member_seed, 31)
        self.assertEqual(configuration.derivation, LEGACY_SEED_DERIVATION)

        set_global_seed(31)
        expected = (random.random(), np.random.random(), torch.rand(4))
        set_configured_seeds(configuration)
        actual = (random.random(), np.random.random(), torch.rand(4))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)


class FixedReplayMemberSeedTest(unittest.TestCase):
    _REPLAY_LABELS = np.repeat(np.asarray([10, 30, 50], dtype=np.int64), 4)

    @classmethod
    def _sampler_orders(cls, member_id: int) -> list[list[int]]:
        configuration = resolve_seed_configuration(0, member_id)
        sampler = FixedReplaySampler(
            current_count=12,
            replay_raw_labels=cls._REPLAY_LABELS,
            replay_draws_per_epoch=12,
            seed=configuration.member_seed,
            task_id=1,
        )
        return [list(iter(sampler)), list(iter(sampler))]

    def test_same_experiment_seed_and_member_produce_same_sampler_order(self) -> None:
        self.assertEqual(self._sampler_orders(0), self._sampler_orders(0))

    def test_different_members_produce_different_sampler_orders(self) -> None:
        first_order = self._sampler_orders(0)[0]
        second_order = self._sampler_orders(1)[0]
        self.assertNotEqual(first_order, second_order)
        self.assertNotEqual(
            [index for index in first_order if index < 12],
            [index for index in second_order if index < 12],
        )
        self.assertNotEqual(
            [index for index in first_order if index >= 12],
            [index for index in second_order if index >= 12],
        )

    def test_replay_buffer_identity_remains_shared_between_members(self) -> None:
        features = np.arange(96, dtype=np.float32).reshape(24, 4)
        labels = np.repeat(np.asarray([10, 30, 50], dtype=np.int64), 8)
        buffers = []
        for member_id in (0, 1, 2):
            configuration = resolve_seed_configuration(7, member_id)
            buffer = FixedBudgetReplayBuffer(
                total_memory_budget=12,
                random_seed=configuration.experiment_seed,
            )
            buffer.update(features, labels)
            buffers.append(buffer)

        expected_features, expected_labels = buffers[0].get_all()
        for buffer in buffers[1:]:
            actual_features, actual_labels = buffer.get_all()
            self.assertEqual(buffer.memory_counts, buffers[0].memory_counts)
            np.testing.assert_array_equal(actual_features, expected_features)
            np.testing.assert_array_equal(actual_labels, expected_labels)

    def test_legacy_identity_seed_preserves_fixed_sampler_order(self) -> None:
        legacy = resolve_seed_configuration(19)
        self.assertEqual(legacy.member_seed, legacy.experiment_seed)
        previous_sampler = FixedReplaySampler(
            current_count=12,
            replay_raw_labels=self._REPLAY_LABELS,
            replay_draws_per_epoch=12,
            seed=legacy.experiment_seed,
            task_id=1,
        )
        resolved_sampler = FixedReplaySampler(
            current_count=12,
            replay_raw_labels=self._REPLAY_LABELS,
            replay_draws_per_epoch=12,
            seed=legacy.member_seed,
            task_id=1,
        )
        self.assertEqual(list(iter(previous_sampler)), list(iter(resolved_sampler)))


class SeedCliAndProvenanceTest(unittest.TestCase):
    _REQUIRED = [
        "--train", "train.parquet",
        "--validation", "validation.parquet",
        "--test", "test.parquet",
        "--feature-cols", "features.json",
        "--scaler", "scaler.pkl",
        "--task-file", "tasks.json",
        "--outdir", "run",
    ]

    def test_legacy_cli_with_only_seed_still_resolves_identity_mode(self) -> None:
        with patch("sys.argv", ["train_cil.py", *self._REQUIRED, "--seed", "41"]):
            args = parse_args()
        configuration = resolve_seed_configuration(args.seed, args.member_id)
        self.assertEqual(args.seed, 41)
        self.assertIsNone(args.member_id)
        self.assertIsNone(args.resume_ewc_checkpoint)
        self.assertFalse(args.export_member_predictions)
        self.assertEqual(configuration.experiment_seed, 41)
        self.assertEqual(configuration.member_seed, 41)
        self.assertEqual(configuration.mode, "legacy")

    @staticmethod
    def _run_args(task_file: Path, member_id: int) -> Namespace:
        configuration = resolve_seed_configuration(5, member_id)
        return Namespace(
            seed=5,
            member_id=member_id,
            experiment_seed=configuration.experiment_seed,
            member_seed=configuration.member_seed,
            seed_mode=configuration.mode,
            member_seed_derivation=configuration.derivation,
            method="ewc",
            memory_per_class=50,
            variant="small",
            graph_cache=None,
            task_file=task_file,
        )

    def test_protocol_and_class_metadata_remain_shared_between_members(self) -> None:
        task_spec = {
            "protocol": "constrained_mass_balanced",
            "seed": 5,
            "tasks": [
                {"task_id": 0, "classes": [30, 10]},
                {"task_id": 1, "classes": [70]},
            ],
        }
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            task_file = root / "tasks.json"
            task_file.write_text(json.dumps(task_spec), encoding="utf-8")
            paths = [root / "member0.json", root / "member1.json"]
            for member_id, path in enumerate(paths):
                write_run_config(
                    path,
                    args=self._run_args(task_file, member_id),
                    run_id=f"member-{member_id}",
                    device="cpu",
                    task_spec=task_spec,
                )
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

        first_resolved = payloads[0]["resolved"]
        second_resolved = payloads[1]["resolved"]
        for key in (
            "experiment_seed",
            "order_seed",
            "task_protocol",
            "num_tasks",
            "task_file_sha256",
        ):
            self.assertEqual(first_resolved[key], second_resolved[key])
        self.assertNotEqual(first_resolved["member_id"], second_resolved["member_id"])
        self.assertNotEqual(first_resolved["member_seed"], second_resolved["member_seed"])
        self.assertEqual(
            seed_provenance(self._run_args(Path("tasks.json"), 0)),
            {
                "experiment_seed": 5,
                "member_id": 0,
                "member_seed": derive_member_seed(5, 0),
                "member_seed_derivation": MEMBER_SEED_DERIVATION,
                "seed_mode": "member",
            },
        )
        self.assertEqual(
            build_seen_class_map([30, 10, 70]),
            {10: 0, 30: 1, 70: 2},
        )

    def test_fixed_replay_run_config_records_shared_and_member_seed_roles(self) -> None:
        task_spec = {
            "protocol": "constrained_mass_balanced",
            "seed": 5,
            "tasks": [{"task_id": 0, "classes": [10, 30]}],
        }
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            task_file = root / "tasks.json"
            task_file.write_text(json.dumps(task_spec), encoding="utf-8")
            args = self._run_args(task_file, member_id=1)
            args.method = FIXED_BUDGET_METHOD
            args.total_memory_budget = 6800
            args.replay_draws_per_epoch = 6800
            path = root / "run_config.json"
            write_run_config(
                path,
                args=args,
                run_id="fixed-member-1",
                device="cpu",
                task_spec=task_spec,
            )
            resolved = json.loads(path.read_text(encoding="utf-8"))["resolved"]

        expected = {
            "replay_buffer_seed": 5,
            "replay_buffer_seed_role": "experiment_seed",
            "sampler_seed": derive_member_seed(5, 1),
            "sampler_seed_role": "member_seed",
            "sampler_seed_derivation": MEMBER_SEED_DERIVATION,
        }
        self.assertEqual(
            fixed_replay_seed_provenance(args),
            expected,
        )
        for key, value in expected.items():
            self.assertEqual(resolved[key], value)


if __name__ == "__main__":
    unittest.main()
