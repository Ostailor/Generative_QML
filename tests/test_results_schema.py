from __future__ import annotations

import json
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = BASE_DIR / "data" / "dft_workflow"
REQUIRED_KEYS = {"schema_version", "engine", "dft_settings_hash", "uncertainty", "evidence_tier"}


class ResultsSchemaTests(unittest.TestCase):
    def test_all_results_have_required_schema_keys(self) -> None:
        result_files = list(WORKFLOW_DIR.glob("*/results.json"))
        self.assertGreater(len(result_files), 0, "no results files found")
        for path in result_files:
            payload = json.loads(path.read_text())
            missing = REQUIRED_KEYS - set(payload.keys())
            self.assertFalse(missing, f"{path} missing keys: {sorted(missing)}")


if __name__ == "__main__":
    unittest.main()
