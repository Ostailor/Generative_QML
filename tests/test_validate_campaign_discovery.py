from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.dft.validate_production_outputs import _discover_requests_from_campaign  # noqa: E402


class CampaignDiscoveryTests(unittest.TestCase):
    def test_discover_from_candidate_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "camp-001"
            campaign.mkdir(parents=True)
            csv_path = campaign / "candidate_library.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["request_id"])
                writer.writeheader()
                writer.writerow({"request_id": "REQ-001"})
                writer.writerow({"request_id": "REQ-002"})

            requests, meta = _discover_requests_from_campaign("camp-001", root)
            self.assertEqual(requests, ["REQ-001", "REQ-002"])
            self.assertIn("request_source", meta)

    def test_discover_from_iteration_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "camp-002"
            campaign.mkdir(parents=True)
            manifest = campaign / "iteration_01_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["request_id", "candidate_id"])
                writer.writeheader()
                writer.writerow({"request_id": "REQ-A", "candidate_id": "C-1"})
                writer.writerow({"request_id": "REQ-B", "candidate_id": "C-2"})

            requests, _ = _discover_requests_from_campaign("camp-002", root)
            self.assertEqual(requests, ["REQ-A", "REQ-B"])


if __name__ == "__main__":
    unittest.main()
