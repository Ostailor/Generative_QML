from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.hpc.run_real_dft_campaign import (  # noqa: E402
    compute_label_efficiency,
    parse_composition,
    write_handoff_package,
)


class CampaignMathTests(unittest.TestCase):
    def test_parse_composition_normalizes_counts(self) -> None:
        parsed = parse_composition("Al2 Co1 Ni1")
        self.assertAlmostEqual(sum(parsed.values()), 1.0, places=6)
        self.assertAlmostEqual(parsed["Al"], 0.5, places=6)
        self.assertAlmostEqual(parsed["Co"], 0.25, places=6)
        self.assertAlmostEqual(parsed["Ni"], 0.25, places=6)

    def test_label_efficiency_gain(self) -> None:
        gain = compute_label_efficiency(120, 24)
        self.assertAlmostEqual(gain, 0.8, places=6)

    def test_label_efficiency_clamped(self) -> None:
        gain = compute_label_efficiency(100, 140)
        self.assertEqual(gain, 0.0)

    def test_label_efficiency_invalid_budget_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_label_efficiency(0, 10)

    def test_request_packaging_creates_metadata_and_structure(self) -> None:
        candidate = pd.Series(
            {
                "candidate_id": "CAND-001",
                "composition": "Al2 Co1 Ni1",
                "phase": "fcc",
                "predicted_density_g_cm3": 7.3,
                "target_density_g_cm3": 7.5,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            handoff = write_handoff_package(
                candidate,
                request_id="REQ-PACK-001",
                job_class="scf_screen",
                iteration_index=1,
                output_root=out_root,
                random_state=42,
                base_lattice_constant=3.65,
                supercell=(2, 2, 2),
                calculation="scf",
                relaxation={},
            )
            self.assertTrue((handoff / "metadata.json").exists())
            self.assertTrue((handoff / "structure.cif").exists())

            payload = json.loads((handoff / "metadata.json").read_text())
            self.assertEqual(payload["request_id"], "REQ-PACK-001")
            self.assertEqual(payload["job_class"], "scf_screen")
            self.assertEqual(payload["calculation"], "scf")


if __name__ == "__main__":
    unittest.main()
