from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.build_advanced_protocols import (
    assign_initial,
    controlled_drift_quotas,
    relative_mse,
)


class AdvancedProtocolConstructionTest(unittest.TestCase):
    def test_controlled_drift_quotas_are_exact_and_monotonic_at_extremes(self) -> None:
        sizes = [38] + [20] * 7
        counts = {"ultra_tail": 45, "tail": 39, "medium": 55, "head": 39}
        quotas = controlled_drift_quotas(sizes, counts)
        self.assertEqual([sum(row.values()) for row in quotas], sizes)
        self.assertEqual(
            {name: sum(row[name] for row in quotas) for name in counts},
            counts,
        )
        self.assertGreater(
            quotas[0]["head"] / sizes[0],
            quotas[-1]["head"] / sizes[-1],
        )
        self.assertLess(
            quotas[0]["ultra_tail"] / sizes[0],
            quotas[-1]["ultra_tail"] / sizes[-1],
        )

    def test_relative_mse_is_zero_at_target(self) -> None:
        values = np.asarray([3.0, 4.0])
        self.assertEqual(relative_mse(values, values.copy()), 0.0)

    def test_initial_assignment_respects_frequency_quotas(self) -> None:
        frame = pd.DataFrame(
            {
                "class_id": list(range(8)),
                "count": [1000, 900, 100, 90, 20, 19, 5, 4],
                "frequency_bin": [
                    "medium", "medium", "tail", "tail",
                    "ultra_tail", "ultra_tail", "ultra_tail", "ultra_tail",
                ],
            }
        )
        sizes = [4, 4]
        quotas = [
            {"ultra_tail": 2, "tail": 1, "medium": 1, "head": 0},
            {"ultra_tail": 2, "tail": 1, "medium": 1, "head": 0},
        ]
        buckets = assign_initial(frame, sizes, quotas, seed=3)
        by_class = frame.set_index("class_id")["frequency_bin"].to_dict()
        for bucket in buckets:
            observed = {name: sum(by_class[c] == name for c in bucket) for name in quotas[0]}
            self.assertEqual(observed, quotas[0])


if __name__ == "__main__":
    unittest.main()
