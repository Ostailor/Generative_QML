from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.dft import run_dft_workflow as workflow  # noqa: E402


def _fake_result(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "total_energy_eV": -12.5,
        "formation_energy_eV": -3.1,
        "max_force_eV_A": 0.01,
        "strain_results": [],
    }


class QueueRetryAndIntegrationTests(unittest.TestCase):
    def test_retry_then_success(self) -> None:
        attempts = {"REQ-A": 0}

        def _run(request_id: str, resume: bool = False) -> dict:
            del resume
            attempts[request_id] += 1
            if attempts[request_id] == 1:
                raise RuntimeError("transient failure")
            return _fake_result(request_id)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monitor_dir = tmp_path / "monitor"
            tracking_uri = f"file://{(tmp_path / 'mlruns').resolve()}"
            run_name = "queue-retry-success"

            with patch.object(workflow, "run_workflow", side_effect=_run), patch.object(workflow.time, "sleep", return_value=None):
                summary = workflow.run_queue(
                    ["REQ-A"],
                    max_workers=1,
                    max_retries=1,
                    tracking_uri=tracking_uri,
                    experiment="test_queue_retry",
                    run_name=run_name,
                    monitor_dir=monitor_dir,
                )

            self.assertEqual(summary["completed_jobs"], 1)
            self.assertEqual(summary["failed_jobs"], 0)
            jobs_log = Path(summary["artefacts"]["jobs_log"])
            self.assertTrue(jobs_log.exists())
            line = json.loads(jobs_log.read_text().strip())
            self.assertEqual(line["attempts"], 2)
            self.assertEqual(line["status"], "completed")
            self.assertEqual(len(line["errors"]), 1)

    def test_three_request_queue_logs_latency_and_mlflow_artifacts(self) -> None:
        attempts = {"REQ-A": 0, "REQ-B": 0, "REQ-C": 0}

        def _run(request_id: str, resume: bool = False) -> dict:
            del resume
            attempts[request_id] += 1
            if request_id == "REQ-B" and attempts[request_id] == 1:
                raise RuntimeError("retry-me")
            return _fake_result(request_id)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monitor_dir = tmp_path / "monitor"
            tracking_uri = f"file://{(tmp_path / 'mlruns').resolve()}"
            experiment_name = "test_queue_integration_3req"
            run_name = "queue-3req-retry-latency"

            with patch.object(workflow, "run_workflow", side_effect=_run), patch.object(workflow.time, "sleep", return_value=None):
                summary = workflow.run_queue(
                    ["REQ-A", "REQ-B", "REQ-C"],
                    max_workers=2,
                    max_retries=2,
                    tracking_uri=tracking_uri,
                    experiment=experiment_name,
                    run_name=run_name,
                    monitor_dir=monitor_dir,
                )

            self.assertEqual(summary["total_jobs"], 3)
            self.assertEqual(summary["completed_jobs"], 3)
            self.assertEqual(summary["failed_jobs"], 0)
            self.assertTrue(summary["mlflow_run"])
            self.assertIsNotNone(summary["latency"]["avg"])
            self.assertTrue(Path(summary["artefacts"]["summary_path"]).exists())

            jobs_log = Path(summary["artefacts"]["jobs_log"])
            rows = [json.loads(line) for line in jobs_log.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 3)
            by_id = {row["request_id"]: row for row in rows}
            self.assertEqual(by_id["REQ-B"]["attempts"], 2)

            client = MlflowClient(tracking_uri=tracking_uri)
            exp = client.get_experiment_by_name(experiment_name)
            self.assertIsNotNone(exp)
            runs = client.search_runs(
                [exp.experiment_id],
                filter_string=f"attributes.run_name = '{run_name}'",
            )
            self.assertGreaterEqual(len(runs), 1)
            run_id = runs[0].info.run_id
            root_artifacts = {item.path for item in client.list_artifacts(run_id)}
            self.assertIn("queue_summary.json", root_artifacts)
            queue_artifacts = {item.path for item in client.list_artifacts(run_id, "queue_monitoring")}
            self.assertIn("queue_monitoring/jobs.jsonl", queue_artifacts)


if __name__ == "__main__":
    unittest.main()
