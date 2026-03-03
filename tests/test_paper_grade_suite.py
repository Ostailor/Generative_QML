from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


class PaperGradeSuiteTests(unittest.TestCase):
    def test_smoke_run_with_cpu_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            summary = tmp_dir / "summary.json"
            runs_csv = tmp_dir / "runs.csv"

            cmd = [
                sys.executable,
                str(BASE_DIR / "scripts" / "benchmarking" / "run_paper_grade_gpu_suite.py"),
                "--total-runs",
                "6",
                "--qsvr-runs",
                "2",
                "--qgpr-runs",
                "2",
                "--classical-runs",
                "2",
                "--fastest",
                "--allow-cpu-fallback",
                "--max-qsvr-relative-gap",
                "1.0",
                "--max-abs-qgpr-coverage-gap",
                "1.0",
                "--qsvr-max-train",
                "400",
                "--qsvr-max-test",
                "120",
                "--qgpr-max-train",
                "200",
                "--qgpr-max-test",
                "80",
                "--classical-iterations",
                "1",
                "--classical-pool-subsample",
                "256",
                "--classical-max-eval-size",
                "64",
                "--summary-path",
                str(summary),
                "--runs-path",
                str(runs_csv),
            ]
            subprocess.run(cmd, cwd=str(BASE_DIR), check=True, capture_output=True, text=True)

            self.assertTrue(summary.exists(), "summary JSON missing")
            self.assertTrue(runs_csv.exists(), "runs CSV missing")

            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(int(payload["execution"]["total_runs_completed"]), 6)
            self.assertEqual(payload["plan"]["speed_profile"], "fastest")


if __name__ == "__main__":
    unittest.main()
