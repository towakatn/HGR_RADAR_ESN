"""Checks for fair, training-only choices in the HAR diagnostic experiment."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.har_fusion_diagnostics import LAMBDA_GRID, map_redundancy, select_common_regularization


class DiagnosticSelectionTests(unittest.TestCase):
    def test_eight_baselines_do_not_outvote_the_other_four_families(self):
        curve = []
        for value in LAMBDA_GRID:
            for i in range(8):
                score = .99 if value == .001 else .8 if value == .1 else .1
                curve.append(dict(method=f"Single-map/map{i}", regularization=value,
                                  validation_accuracy=score, test_accuracy=0.0))
            for name in ("Single-Early", "Parallel-Early", "Parallel-Intermediate", "Parallel-Late/mean"):
                score = .6 if value == .001 else .8 if value == .1 else .1
                curve.append(dict(method=name, regularization=value, validation_accuracy=score,
                                  test_accuracy=1.0 if value == .001 else 0.0))
            # Alternative Late pools must not contribute additional family votes.
            for name in ("product", "geometric", "max"):
                curve.append(dict(method=f"Parallel-Late/{name}", regularization=value,
                                  validation_accuracy=1.0 if value == .001 else 0.0))
        selected, _ = select_common_regularization(curve)
        self.assertEqual(selected, .1)

    def test_redundancy_measurement_ignores_held_out_samples(self):
        rng = np.random.default_rng(812)
        values = [rng.normal(size=(5, 3)) for _ in range(6)]
        maps = {"RTM_ch0": values, "RTM_ch1": [2 * array + 7 for array in values]}
        first = map_redundancy(maps, np.arange(4))
        maps["RTM_ch0"][4:] = [rng.normal(1e6, 1e6, (5, 3)) for _ in range(2)]
        maps["RTM_ch1"][4:] = [rng.normal(-1e6, 1e6, (5, 3)) for _ in range(2)]
        self.assertEqual(first, map_redundancy(maps, np.arange(4)))
        self.assertAlmostEqual(first[0]["linear_cka"], 1.0)
        self.assertAlmostEqual(first[0]["standardized_binwise_correlation"], 1.0)


if __name__ == "__main__":
    unittest.main()
