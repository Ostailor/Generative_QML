from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.publication.validate_publication_build import run_build  # noqa: E402
from scripts.repro.run_repro_check import run_checks  # noqa: E402


class ReproAndPublicationBuildTests(unittest.TestCase):
    def test_repro_check_passes(self) -> None:
        report = run_checks()
        self.assertTrue(report["checks"]["required_files_present"])
        self.assertTrue(report["checks"]["release_manifest_valid"])
        self.assertTrue(report["reproduction_success"])
        self.assertTrue((BASE_DIR / "data" / "reproducibility" / "reproduction_report.json").exists())

    def test_publication_build_pipeline_passes(self) -> None:
        report = run_build()
        self.assertEqual(report["status"], "pass")
        steps = report["steps"]
        self.assertGreater(int(steps["bibliography_build"]["entry_count"]), 0)
        self.assertEqual(steps["figure_regeneration"]["missing_figures"], [])
        self.assertTrue(Path(steps["manuscript_compile"]["bundle"]).exists())
        self.assertTrue(Path(steps["poster_export"]["poster_manifest"]).exists())
        build_report = BASE_DIR / "docs" / "submission" / "build" / "publication_build_report.json"
        self.assertTrue(build_report.exists())
        payload = json.loads(build_report.read_text())
        self.assertEqual(payload["status"], "pass")


if __name__ == "__main__":
    unittest.main()
