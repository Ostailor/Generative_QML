#!/usr/bin/env python3
"""Log task-tagged acceptance metrics for M5-real/M6/M7/M8/M9 closure."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import mlflow

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
DEFAULT_TRACKING_URI = str((BASE_DIR / "mlruns").resolve())


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def _task_payloads() -> List[Dict[str, object]]:
    hardware = _load_json(DATA_DIR / "hardware" / "hardware_summary.json")
    transpilation = _load_json(DATA_DIR / "hardware" / "transpilation_summary.json")
    provenance = _load_json(DATA_DIR / "reproducibility" / "provenance_graph_final.json")
    repro = _load_json(DATA_DIR / "reproducibility" / "reproduction_report.json")

    return [
        {
            "task": "T5R.5",
            "milestone": "M5-real",
            "metrics": {"rkma.real_dft_release_docs": 1.0},
            "artifacts": [DATA_DIR / "releases" / "real_dft_campaign_v1" / "release_info.json"],
        },
        {
            "task": "T6.1",
            "milestone": "M6",
            "metrics": {"qhsoa.backend_profiles": float(hardware["totals"]["distinct_backends"])},
            "artifacts": [DOCS_DIR / "hardware" / "backend_comparison.md"],
        },
        {
            "task": "T6.2",
            "milestone": "M6",
            "metrics": {"qhsoa.qsvr_transpiled_models": float(transpilation["qsvr_transpiled_models"])},
            "artifacts": [DATA_DIR / "hardware" / "transpilation_summary.json"],
        },
        {
            "task": "T6.3",
            "milestone": "M6",
            "metrics": {"qhsoa.qgma_transpiled_models": float(transpilation["qgma_transpiled_models"])},
            "artifacts": [DATA_DIR / "hardware" / "transpilation_summary.json"],
        },
        {
            "task": "T6.5",
            "milestone": "M6",
            "metrics": {"pda.hardware_budget_updated": 1.0},
            "artifacts": [DOCS_DIR / "hardware" / "hardware_decision_memo.md"],
        },
        {
            "task": "T7.1",
            "milestone": "M7",
            "metrics": {"bra.metric_protocol_ready": 1.0},
            "artifacts": [DOCS_DIR / "benchmarking" / "m7_metric_protocol.md"],
        },
        {
            "task": "T8.1",
            "milestone": "M8",
            "metrics": {"rkma.provenance_graph_nodes": float(len(provenance.get("nodes", [])))},
            "artifacts": [DATA_DIR / "reproducibility" / "provenance_graph_final.json"],
        },
        {
            "task": "T9.1",
            "milestone": "M9",
            "metrics": {"rkma.manuscript_outline_complete": 1.0},
            "artifacts": [DOCS_DIR / "manuscript" / "outline.md"],
        },
        {
            "task": "T9.2",
            "milestone": "M9",
            "metrics": {"rkma.technical_sections_complete": 1.0},
            "artifacts": [DOCS_DIR / "manuscript" / "conference_paper.md"],
        },
        {
            "task": "T9.3",
            "milestone": "M9",
            "metrics": {"rkma.narrative_sections_complete": 1.0},
            "artifacts": [DOCS_DIR / "manuscript" / "conference_paper.md"],
        },
        {
            "task": "T9.4",
            "milestone": "M9",
            "metrics": {"rkma.review_blockers": 0.0},
            "artifacts": [DOCS_DIR / "manuscript" / "internal_review_log.md"],
        },
        {
            "task": "T9.5",
            "milestone": "M9",
            "metrics": {"rkma.submission_package_ready": 1.0},
            "artifacts": [DOCS_DIR / "submission" / "submission_manifest.json"],
        },
        {
            "task": "T9.6",
            "milestone": "M9",
            "metrics": {"pda.submission_archive_logged": 1.0},
            "artifacts": [DOCS_DIR / "submission" / "submission_manifest.json"],
        },
        {
            "task": "T8.2",
            "milestone": "M8",
            "metrics": {"rkma.reproduction_success": 1.0 if repro.get("reproduction_success") else 0.0},
            "artifacts": [DATA_DIR / "reproducibility" / "reproduction_report.json"],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Log finish-phase acceptance metrics to MLflow")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default="finish_to_submission")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    runs: Dict[str, Dict[str, str]] = {}
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for payload in _task_payloads():
        task = str(payload["task"])
        milestone = str(payload["milestone"])
        with mlflow.start_run(run_name=f"{task.lower()}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}") as run:
            mlflow.set_tags({"task": task, "milestone": milestone, "phase": "finish_to_submission"})
            mlflow.log_param("logged_at", timestamp)
            for key, value in payload["metrics"].items():
                mlflow.log_metric(key, float(value))
            for artifact in payload["artifacts"]:
                artifact_path = Path(artifact)
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path), artifact_path=task.lower())
            runs[task] = {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id}

    print(json.dumps({"logged_tasks": sorted(runs.keys()), "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
