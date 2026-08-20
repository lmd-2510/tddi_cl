from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_cil_tasks import (
    FREQUENCY_BINS,
    build_constrained_mass_balanced_tasks,
    build_frequency_order,
    proportional_bin_quotas,
)


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


if __name__ == "__main__":
    unittest.main()
