from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fixed_budget_replay import (
    FixedBudgetReplayBuffer,
    FixedReplaySampler,
    max_min_uniform_allocation,
)
from src.training.train_cil import (
    FIXED_BUDGET_METHOD,
    build_fixed_budget_audit_rows,
    method_protocol_name,
    sampler_policy_name,
)


class FixedBudgetAllocationTest(unittest.TestCase):
    def test_capacity_constrained_water_filling_is_exact_and_deterministic(self) -> None:
        capacities = {30: 2, 10: 100, 20: 100, 40: 100}
        allocation = max_min_uniform_allocation(capacities, 20)
        self.assertEqual(allocation, {10: 6, 20: 6, 30: 2, 40: 6})
        self.assertEqual(sum(allocation.values()), 20)
        self.assertLessEqual(max(allocation[c] for c in [10, 20, 40]) - min(allocation[c] for c in [10, 20, 40]), 1)

    def test_buffer_reaches_budget_and_shrinks_old_classes(self) -> None:
        buffer = FixedBudgetReplayBuffer(total_memory_budget=12, random_seed=7)
        features = np.arange(80, dtype=np.float32).reshape(20, 4)
        labels = np.repeat(np.asarray([10, 30]), 10)
        buffer.update(features, labels)
        self.assertEqual(buffer.total_size, 12)
        self.assertEqual(buffer.memory_counts, {10: 6, 30: 6})
        old_blocks = {class_id: block.copy() for class_id, block in buffer.features_by_class.items()}

        new_features = np.arange(80, 160, dtype=np.float32).reshape(20, 4)
        new_labels = np.repeat(np.asarray([50, 70]), 10)
        buffer.update(new_features, new_labels)
        self.assertEqual(buffer.memory_counts, {10: 3, 30: 3, 50: 3, 70: 3})
        np.testing.assert_array_equal(buffer.features_by_class[10], old_blocks[10][:3])
        np.testing.assert_array_equal(buffer.features_by_class[30], old_blocks[30][:3])

    def test_buffer_uses_all_available_samples_below_budget_and_rejects_duplicate_classes(self) -> None:
        buffer = FixedBudgetReplayBuffer(total_memory_budget=50)
        features = np.arange(24, dtype=np.float32).reshape(6, 4)
        labels = np.asarray([1, 1, 2, 2, 2, 2])
        buffer.update(features, labels)
        self.assertEqual(buffer.total_size, 6)
        with self.assertRaisesRegex(ValueError, "received classes twice"):
            buffer.update(features[:2], np.asarray([1, 1]))

    def test_summary_and_snapshot_match_memory(self) -> None:
        buffer = FixedBudgetReplayBuffer(total_memory_budget=5)
        features = np.arange(24, dtype=np.float32).reshape(6, 4)
        labels = np.asarray([4, 4, 4, 9, 9, 9])
        buffer.update(features, labels)
        with tempfile.TemporaryDirectory() as tempdir:
            summary_path = Path(tempdir) / "summary.csv"
            snapshot_path = Path(tempdir) / "snapshot.parquet"
            buffer.save_summary(summary_path)
            buffer.save_snapshot(snapshot_path)
            summary = pd.read_csv(summary_path)
            snapshot = pd.read_parquet(snapshot_path)
        self.assertEqual(int(summary["memory_count"].sum()), 5)
        self.assertEqual(snapshot.shape[0], 5)


class FixedReplaySamplerTest(unittest.TestCase):
    def test_current_once_and_exact_uniform_replay_draws(self) -> None:
        replay_labels = np.asarray([10, 10, 20, 20, 30, 30], dtype=np.int64)
        sampler = FixedReplaySampler(
            current_count=5,
            replay_raw_labels=replay_labels,
            replay_draws_per_epoch=8,
            seed=11,
            task_id=3,
        )
        indices = list(iter(sampler))
        audit = sampler.last_audit
        assert audit is not None
        self.assertEqual(len(indices), 13)
        self.assertEqual(sorted(index for index in indices if index < 5), list(range(5)))
        self.assertEqual(audit.current_draws, 5)
        self.assertEqual(audit.replay_draws, 8)
        self.assertEqual(sum(audit.per_class_draws.values()), 8)
        self.assertLessEqual(max(audit.per_class_draws.values()) - min(audit.per_class_draws.values()), 1)

    def test_remainder_rotates_and_sampling_is_reproducible(self) -> None:
        labels = np.asarray([10, 20, 30], dtype=np.int64)
        first = FixedReplaySampler(
            current_count=2,
            replay_raw_labels=labels,
            replay_draws_per_epoch=4,
            seed=5,
            task_id=1,
        )
        second = FixedReplaySampler(
            current_count=2,
            replay_raw_labels=labels,
            replay_draws_per_epoch=4,
            seed=5,
            task_id=1,
        )
        self.assertEqual(list(iter(first)), list(iter(second)))
        self.assertEqual(first.history[0].per_class_draws, {10: 2, 20: 1, 30: 1})
        list(iter(first))
        self.assertEqual(first.history[1].per_class_draws, {10: 1, 20: 2, 30: 1})

    def test_task_zero_current_only(self) -> None:
        sampler = FixedReplaySampler(
            current_count=7,
            replay_raw_labels=np.empty((0,), dtype=np.int64),
            replay_draws_per_epoch=0,
            seed=0,
            task_id=0,
        )
        self.assertEqual(sorted(iter(sampler)), list(range(7)))
        assert sampler.last_audit is not None
        self.assertEqual(sampler.last_audit.replay_draws, 0)

    def test_o06_counts_match_the_sampler_indices(self) -> None:
        labels = np.asarray([2, 2, 5, 5], dtype=np.int64)
        sampler = FixedReplaySampler(
            current_count=3,
            replay_raw_labels=labels,
            replay_draws_per_epoch=7,
            seed=9,
            task_id=1,
        )
        for _ in range(2):
            list(iter(sampler))

        rows = build_fixed_budget_audit_rows(
            run_id="run-1",
            seed=9,
            task_id=1,
            seen_raw_classes=[2, 5, 8],
            current_raw_classes=[8],
            available_count_by_class={2: 10, 5: 10, 8: 3},
            memory_before_by_class={2: 2, 5: 2},
            memory_after_by_class={2: 2, 5: 2, 8: 3},
            total_memory_budget=7,
            replay_draws_per_epoch=7,
            current_dataset_size=3,
            epoch_audits=sampler.history,
            optimizer_steps=4,
        )
        by_class = {row["raw_class_id"]: row for row in rows}
        self.assertEqual(sum(row["replay_draws_total"] for row in rows), 14)
        self.assertEqual(by_class[2]["unique_replay_support"], 2)
        self.assertEqual(by_class[5]["unique_replay_support"], 2)
        self.assertEqual(by_class[8]["replay_draws_total"], 0)
        self.assertEqual(by_class[8]["class_role"], "current")


class S04ProtocolTest(unittest.TestCase):
    def test_fixed_budget_and_standard_replay_protocol_names(self) -> None:
        self.assertEqual(method_protocol_name("replay", 50), "replay_balanced_per_class_cap50")
        self.assertEqual(method_protocol_name(FIXED_BUDGET_METHOD, 50), FIXED_BUDGET_METHOD)
        self.assertEqual(
            sampler_policy_name(FIXED_BUDGET_METHOD),
            "current_once_plus_fixed_class_uniform_replay",
        )


if __name__ == "__main__":
    unittest.main()
