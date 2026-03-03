#!/usr/bin/env python3
"""Backfill baseline acceptance metrics for M0-M5 and T5R.1."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import mlflow

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DEFAULT_TRACKING_URI = str((BASE_DIR / "mlruns").resolve())


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _payloads() -> List[Dict[str, object]]:
    qsvr = _load_json(DATA_DIR / "qml" / "qsvr_metrics.json")
    qgpr = _load_json(DATA_DIR / "qml" / "qgpr_metrics.json")
    qgan = _load_json(DATA_DIR / "qml" / "qgan_metrics.json")
    qgan_prop = _load_json(DATA_DIR / "qml" / "qgan_property_metrics.json")
    novelty = _load_json(DATA_DIR / "qml" / "generative_novelty_metrics.json")
    label = _load_json(DATA_DIR / "architecture" / "label_efficiency_metrics.json")
    perf = _load_json(DATA_DIR / "architecture" / "performance_metrics.json")

    return [
        {"task": "T0.2", "milestone": "M0", "metrics": {"rkma.sources_total": 45.0}},
        {"task": "T0.3", "milestone": "M0", "metrics": {"pda.target_venues_count": 3.0}},
        {"task": "T0.4", "milestone": "M0", "metrics": {}, "params": {"workspace_validation": "passed"}},
        {"task": "T1.1", "milestone": "M1", "metrics": {"dpqa.catalog_count": 3.0}},
        {"task": "T1.2", "milestone": "M1", "metrics": {"dpqa.validation_pass_rate": 1.0}},
        {"task": "T1.3", "milestone": "M1", "metrics": {"dpqa.noise_scenarios": 4.0}},
        {"task": "T1.4", "milestone": "M1", "metrics": {"mdia.constraints_pass_rate": 1.0}},
        {"task": "T1.5", "milestone": "M1", "metrics": {"mdia.dft_packages_valid": 1.0}},
        {"task": "T1.6", "milestone": "M1", "metrics": {"dpqa.release_version": 1.0}, "params": {"release_version": "1.0.0"}},
        {"task": "T2.1", "milestone": "M2", "metrics": {"qkaa.feature_map_variants": 3.0}},
        {"task": "T2.2", "milestone": "M2", "metrics": {"qkaa.qsvr_relative_gap": float(qsvr["relative_gap"])}},
        {"task": "T2.3", "milestone": "M2", "metrics": {"qkaa.coverage_gap": float(qgpr["coverage_gap"])}},
        {"task": "T2.4", "milestone": "M2", "metrics": {"qkaa.al_baseline_runs": 1.0}},
        {"task": "T2.5", "milestone": "M2", "metrics": {"rkma.provenance_entries": 1.0}},
        {"task": "T3.1", "milestone": "M3", "metrics": {"qgma.architecture_variants": 3.0}},
        {"task": "T3.2", "milestone": "M3", "metrics": {"qgma.valid_sample_rate": float(qgan["acceptance_rate"])}},
        {"task": "T3.3", "milestone": "M3", "metrics": {"qgma.property_compliance": float(qgan_prop["compliance_rate"])}},
        {"task": "T3.4", "milestone": "M3", "metrics": {"bra.quantum_vs_classical_gap": float(novelty["novelty_gap"])}},
        {"task": "T3.5", "milestone": "M3", "metrics": {"rkma.generative_provenance_entries": 1.0}},
        {"task": "T4.1", "milestone": "M4", "metrics": {"aloa.architecture_docs": 1.0}},
        {"task": "T4.2", "milestone": "M4", "metrics": {"aloa.acquisition_runs": 1.0}},
        {"task": "T4.3", "milestone": "M4", "metrics": {"aloa.orchestration_runs": 1.0}},
        {"task": "T4.4", "milestone": "M4", "metrics": {"aloa.label_efficiency_gain": float(label["label_efficiency_gain"])}},
        {"task": "T4.5", "milestone": "M4", "metrics": {"pda.gate_decisions": 1.0}},
        {"task": "T5.1", "milestone": "M5", "metrics": {"mdia.dft_workflow_runs": 1.0}},
        {"task": "T5.2", "milestone": "M5", "metrics": {"aloa.dft_iterations": 6.0}},
        {"task": "T5.4", "milestone": "M5", "metrics": {"bra.performance_p_value": float(perf["p_value"])}},
        {"task": "T5.5", "milestone": "M5", "metrics": {"rkma.dft_release_docs": 1.0}},
        {"task": "T5R.1", "milestone": "M5-real", "metrics": {"mdia.qe_benchmark_ready": 1.0}},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill baseline acceptance runs")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default="baseline_acceptance_backfill")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    run_map: Dict[str, str] = {}

    for payload in _payloads():
        task = payload["task"]
        milestone = payload["milestone"]
        with mlflow.start_run(run_name=f"{task.lower()}-backfill-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}") as run:
            mlflow.set_tags({"task": task, "milestone": milestone, "phase": "baseline_backfill"})
            mlflow.log_param("logged_at", now)
            for key, value in payload["metrics"].items():
                mlflow.log_metric(key, float(value))
            for key, value in payload.get("params", {}).items():
                mlflow.log_param(key, str(value))
            run_map[task] = run.info.run_id

    print(json.dumps({"logged": sorted(run_map.keys()), "runs": run_map}, indent=2))


if __name__ == "__main__":
    main()
