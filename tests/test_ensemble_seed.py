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
from src.models.mlp import MLP, preset_config
from src.training.train_cil import parse_args, seed_provenance, write_run_config
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
            variant="tddi",
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


if __name__ == "__main__":
    unittest.main()
