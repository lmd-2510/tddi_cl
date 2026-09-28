from __future__ import annotations

import unittest
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.build_cil_tasks import (
    FREQUENCY_BINS,
    build_constrained_mass_balanced_tasks,
    build_frequency_order,
    proportional_bin_quotas,
)
from scripts.run_p4_derpp_cb_replay_full import audit_p4


class FrequencyDirectionProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.counts = pd.DataFrame(
            {
                "class_id": [40, 10, 30, 20],
                "count": [2, 100, 2, 7],
            }
        )

    def test_p2_orders_head_to_tail_with_stable_class_id_ties(self) -> None:
        self.assertEqual(
            build_frequency_order(self.counts, descending=True),
            [10, 20, 30, 40],
        )

    def test_p3_is_the_opposing_tail_to_head_stress_test(self) -> None:
        self.assertEqual(
            build_frequency_order(self.counts, descending=False),
            [30, 40, 20, 10],
        )


class ConstrainedMassBalancedProtocolTest(unittest.TestCase):
    def test_frozen_p4_has_expected_layout_and_no_overlap(self) -> None:
        spec = json.loads(Path(
            "study_assets/task_protocols/constrained_mass_balanced_seed0_tasks.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(spec["protocol"], "constrained_mass_balanced")
        self.assertEqual([len(task["classes"]) for task in spec["tasks"]], [38] + [20] * 7)
        assigned = [raw for task in spec["tasks"] for raw in task["classes"]]
        self.assertEqual(len(assigned), 178)
        self.assertEqual(len(set(assigned)), 178)

    def test_p4_quotas_preserve_rows_and_frequency_bin_totals(self) -> None:
        sizes = [38] + [20] * 7
        bin_counts = {"ultra_tail": 45, "tail": 39, "medium": 55, "head": 39}
        quotas = proportional_bin_quotas(bin_counts, sizes, seed=0)
        self.assertEqual([sum(row.values()) for row in quotas], sizes)
        self.assertEqual(
            {name: sum(row[name] for row in quotas) for name in FREQUENCY_BINS},
            bin_counts,
        )

    def test_p4_is_reproducible_and_assigns_every_class_once(self) -> None:
        counts = pd.DataFrame(
            {
                "class_id": list(range(12)),
                "count": [2, 5, 10, 20, 25, 50, 75, 100, 150, 500, 1200, 5000],
            }
        )
        sizes = [4, 4, 4]
        first = build_constrained_mass_balanced_tasks(counts, sizes, seed=7)
        second = build_constrained_mass_balanced_tasks(counts, sizes, seed=7)
        self.assertEqual(first, second)
        assigned = [class_id for task in first for class_id in task["classes"]]
        self.assertEqual(sorted(assigned), list(range(12)))
        self.assertEqual([task["num_classes"] for task in first], sizes)

    def test_p4_every_task_has_tail_mid_and_head_from_train_counts(self) -> None:
        # Four train-only frequency strata with enough members for every task.
        counts = pd.DataFrame({
            "class_id": list(range(178)),
            "count": ([10] * 45) + ([50] * 39) + ([500] * 55) + ([5000] * 39),
        })
        tasks = build_constrained_mass_balanced_tasks(counts, [38] + [20] * 7, seed=0)
        lookup = dict(zip(counts.class_id, counts["count"], strict=True))
        for task in tasks:
            values = [lookup[raw] for raw in task["classes"]]
            self.assertTrue(any(value <= 100 for value in values))
            self.assertTrue(any(101 <= value <= 1000 for value in values))
            self.assertTrue(any(value > 1000 for value in values))

    def test_frozen_p4_audit_uses_train_frequency_groups(self) -> None:
        task_file = Path("study_assets/task_protocols/constrained_mass_balanced_seed0_tasks.json")
        spec = json.loads(task_file.read_text(encoding="utf-8"))
        counts = {}
        for task in spec["tasks"]:
            for index, raw in enumerate(task["classes"]):
                counts[int(raw)] = (10, 500, 2000)[index % 3]
        with patch("scripts.run_p4_derpp_cb_replay_full._counts",
                   side_effect=[counts, counts, counts]):
            audit = audit_p4(task_file, Path("train"), Path("validation"), Path("test"))
        self.assertEqual(audit["assignment_statistics_source"], "TRAIN only")
        self.assertEqual(len(audit["tasks"]), 8)
        for row in audit["tasks"]:
            self.assertGreater(row["tail_classes"], 0)
            self.assertGreater(row["mid_classes"], 0)
            self.assertGreater(row["head_classes"], 0)


if __name__ == "__main__":
    unittest.main()
