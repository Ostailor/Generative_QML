from __future__ import annotations

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.benchmarking.run_m7_benchmarks import run_analysis  # noqa: E402


class M7RegressionTests(unittest.TestCase):
    def test_run_analysis_reproducible_at_fixed_seed(self) -> None:
        seed = 20260211
        first = run_analysis(seed)["metrics"]
        second = run_analysis(seed)["metrics"]

        for key in (
            "real_label_efficiency_gain",
            "qsvr_rmse_gain",
            "dft_delta_vs_classical_rmse",
            "qgpr_coverage",
            "novelty_gap",
            "sensitivity_index",
            "hardware_mean_fidelity",
            "quantum_vs_classical_gap",
        ):
            self.assertAlmostEqual(float(first[key]), float(second[key]), places=10, msg=f"metric drifted: {key}")


if __name__ == "__main__":
    unittest.main()
