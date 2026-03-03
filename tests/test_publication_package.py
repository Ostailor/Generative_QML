from __future__ import annotations

import json
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


class PublicationPackageTests(unittest.TestCase):
    def test_manuscript_core_files_exist(self) -> None:
        required = [
            BASE_DIR / "docs" / "manuscript" / "outline.md",
            BASE_DIR / "docs" / "manuscript" / "conference_paper.md",
            BASE_DIR / "docs" / "manuscript" / "appendix_reproducibility.md",
            BASE_DIR / "docs" / "manuscript" / "internal_review_log.md",
            BASE_DIR / "docs" / "submission" / "submission_manifest.json",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")

    def test_generated_figures_exist(self) -> None:
        fig_dir = BASE_DIR / "docs" / "manuscript" / "figures"
        expected = {
            "fig_real_dft_kpi.png",
            "fig_hardware_cost_fidelity.png",
            "fig_m7_benchmark_summary.png",
        }
        existing = {p.name for p in fig_dir.glob("*.png")}
        self.assertTrue(expected.issubset(existing), "missing manuscript figures")

    def test_submission_manifest_claim_policy(self) -> None:
        manifest = json.loads((BASE_DIR / "docs" / "submission" / "submission_manifest.json").read_text())
        policy = manifest["claims_policy"]
        self.assertTrue(policy["real_only_headline_claims"])
        self.assertEqual(policy["simulated_results_position"], "ablation_appendix_only")


if __name__ == "__main__":
    unittest.main()
