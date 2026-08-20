from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "protocol_study_tddi.json"
RUNNER_PATH = ROOT / "scripts" / "run_protocol_study.sh"


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class ProtocolStudyContractTest(unittest.TestCase):
    def test_locked_model_and_training_contract(self) -> None:
        config = load_config()
        self.assertEqual(config["status"], "locked")
        self.assertEqual(
            config["scope"],
            {
                "independent_variable": "task_protocol",
                "fixed_backbone": "tddi",
                "fixed_method": "replay_distill_fixed_budget_uniform",
                "main_protocol": "P4",
                "reference_protocol": "P0",
                "stress_protocols": ["P2", "P3"],
                "advanced_protocols": ["P5", "P6", "P7", "P8"],
            },
        )
        self.assertEqual(config["task_layout"]["classes_per_task"], [38] + [20] * 7)
        self.assertEqual(
            config["model"],
            {
                "variant": "tddi",
                "architecture": "MLP",
                "input_dim": 3780,
                "hidden_dims": [1024, 512],
                "classifier": "expandable_linear_head",
                "normalization": "layernorm",
                "activation": "gelu",
                "dropout": 0.2,
            },
        )
        training = config["training"]
        self.assertEqual(training["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(training["batch_size"], 1024)
        self.assertEqual(training["max_epochs_per_task"], 20)
        self.assertEqual(training["early_stopping_patience"], 5)
        self.assertEqual(training["optimizer"], "AdamW")
        self.assertEqual(training["learning_rate"], 0.001)
        self.assertEqual(training["weight_decay"], 0.0001)
        self.assertEqual(config["replay"]["total_memory_budget"], 6800)
        self.assertEqual(config["replay"]["replay_draws_per_epoch_after_task0"], 6800)

    def test_protocol_task_files_cover_each_class_exactly_once(self) -> None:
        config = load_config()
        for protocol_id, protocol in config["protocols"].items():
            schedule_seeds = protocol["schedule_seeds"] or [None]
            for seed in schedule_seeds:
                relative_path = protocol["task_file"]
                if seed is not None:
                    relative_path = relative_path.format(seed=seed)
                task_path = ROOT / relative_path
                payload = json.loads(task_path.read_text(encoding="utf-8"))
                tasks = payload["tasks"]
                classes = [class_id for task in tasks for class_id in task["classes"]]
                self.assertEqual(
                    [task["num_classes"] for task in tasks],
                    [38] + [20] * 7,
                    protocol_id,
                )
                self.assertEqual(len(classes), 178, protocol_id)
                self.assertEqual(len(set(classes)), 178, protocol_id)

    def test_runner_is_tddi_only_and_pins_locked_hyperparameters(self) -> None:
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        required_fragments = [
            'METHOD="replay_distill_fixed_budget_uniform"',
            'VARIANT="tddi"',
            "--batch-size 1024",
            "--epochs 20",
            "--lr 0.001",
            "--weight-decay 0.0001",
            "--dropout 0.2",
            "--activation gelu",
            "--norm layernorm",
            "--patience 5",
            "--total-memory-budget 6800",
            "--replay-draws-per-epoch 6800",
            "--distill-alpha 1.0",
            "--temperature 2.0",
            "--feature-distill-weight 0.5",
            "--focal-gamma 1.0",
        ]
        for fragment in required_fragments:
            self.assertIn(fragment, runner)
        self.assertNotIn("small|base|large", runner)


if __name__ == "__main__":
    unittest.main()
