#!/usr/bin/env python3
"""Estimate storage and CPU requirements for QE campaigns (T5R.1 helper)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "dft_workflow"
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or f"file://{(REPO_ROOT / 'tracking' / 'mlflow' / 'benchmarks').resolve()}"
DEFAULT_SUMMARY_NAME = "benchmark_summary.json"


def _bytes_to_gib(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def _collect_disk_usage(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            total += (Path(root) / fname).stat().st_size
    return total


def _load_summary(request_id: str, summary_name: str = DEFAULT_SUMMARY_NAME) -> Dict[str, float]:
    run_dir = OUTPUT_DIR / request_id
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Expected QE output directory at {run_dir}. Run the benchmark first."
        )

    summary_path = run_dir / summary_name
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        # Fallback: approximate from directory contents.
        per_run_bytes = float(_collect_disk_usage(run_dir))
        summary = {
            "disk_bytes": per_run_bytes,
            "disk_gib": _bytes_to_gib(per_run_bytes),
        }
    summary.setdefault("disk_bytes", 0.0)
    summary.setdefault("disk_gib", _bytes_to_gib(summary["disk_bytes"]))
    return summary


def _estimate_single_job(
    request_id: str,
    num_candidates: int,
    safety_factor: float,
    summary_name: str = DEFAULT_SUMMARY_NAME,
) -> Dict[str, float]:
    summary = _load_summary(request_id, summary_name)

    per_run_bytes = float(summary.get("disk_bytes", 0.0))
    if per_run_bytes <= 0:
        raise ValueError(
            f"Per-run disk usage is zero for {request_id}; verify the benchmark outputs."
        )

    wall_time = float(summary.get("wall_time_s", 0.0))
    cpu_time = float(summary.get("cpu_time_s", 0.0))
    job_label = summary.get("job_label")

    total_bytes = per_run_bytes * num_candidates * safety_factor
    total_gib = _bytes_to_gib(total_bytes)

    payload = {
        "request_id": request_id,
        "job_label": job_label,
        "runs_planned": num_candidates,
        "safety_factor": safety_factor,
        "per_run_bytes": per_run_bytes,
        "per_run_gib": _bytes_to_gib(per_run_bytes),
        "total_bytes": total_bytes,
        "total_gib": total_gib,
        "per_run_wall_time_s": wall_time,
        "per_run_cpu_time_s": cpu_time,
        "estimated_wall_time_hours": (wall_time * num_candidates * safety_factor) / 3600
        if wall_time
        else None,
    }

    report_path = (OUTPUT_DIR / request_id) / "storage_forecast.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_config(config_path: Path) -> List[Dict[str, object]]:
    data = json.loads(config_path.read_text())
    if isinstance(data, dict):
        jobs = data.get("jobs")
        if jobs is None:
            raise ValueError("Config JSON must contain a 'jobs' list.")
    elif isinstance(data, list):
        jobs = data
    else:
        raise ValueError("Unsupported config format. Provide a list or an object with 'jobs'.")

    normalized: List[Dict[str, object]] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each job entry must be a JSON object.")
        for key in ("request_id", "runs"):
            if key not in job:
                raise ValueError(f"Job entry missing required key '{key}'.")
        entry = {
            "request_id": job["request_id"],
            "runs": int(job["runs"]),
            "safety_factor": float(job.get("safety_factor", 1.0)),
            "summary_name": job.get("summary_name", DEFAULT_SUMMARY_NAME),
            "job_label": job.get("job_label"),
        }
        normalized.append(entry)
    return normalized


def estimate_storage(
    request_id: Optional[str],
    num_candidates: Optional[int],
    safety_factor: float,
    tracking_uri: str | None,
    experiment: str | None,
    summary_name: str,
    config_path: Optional[Path],
) -> Dict[str, object]:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)

    if config_path is not None:
        jobs = _load_config(config_path)
    else:
        if request_id is None or num_candidates is None:
            raise ValueError("Provide --request-id/--runs or a --config file.")
        jobs = [
            {
                "request_id": request_id,
                "runs": num_candidates,
                "safety_factor": safety_factor,
                "summary_name": summary_name,
            }
        ]

    aggregated: List[Dict[str, float]] = []
    total_bytes = 0.0
    total_wall_hours = 0.0

    with mlflow.start_run(run_name="qe-storage-plan"):
        mlflow.set_tags({"task": "T5R.1", "agent": "MDIA"})

        for job in jobs:
            job_payload = _estimate_single_job(
                request_id=job["request_id"],
                num_candidates=job["runs"],
                safety_factor=job["safety_factor"],
                summary_name=job["summary_name"],
            )
            job_label = job.get("job_label") or job_payload.get("job_label")
            if job_label:
                mlflow.log_param(f"job::{job_label}::request_id", job_payload["request_id"])
                mlflow.log_param(f"job::{job_label}::runs", job_payload["runs_planned"])
                mlflow.log_metric(
                    f"job::{job_label}::storage_gib",
                    job_payload["total_gib"],
                )
            else:
                mlflow.log_param(f"job::{job_payload['request_id']}::runs", job_payload["runs_planned"])
                mlflow.log_metric(
                    f"job::{job_payload['request_id']}::storage_gib",
                    job_payload["total_gib"],
                )

            total_bytes += job_payload["total_bytes"]
            if job_payload.get("estimated_wall_time_hours"):
                total_wall_hours += job_payload["estimated_wall_time_hours"]

            aggregated.append(job_payload)

        total_gib = _bytes_to_gib(total_bytes)
        mlflow.log_metric("qe_storage_gib_total", total_gib)
        if total_wall_hours:
            mlflow.log_metric("qe_wall_time_hours_total", total_wall_hours)

        aggregate_payload = {
            "jobs": aggregated,
            "totals": {
                "storage_bytes": total_bytes,
                "storage_gib": total_gib,
                "wall_time_hours": total_wall_hours if total_wall_hours else None,
            },
        }

        report_path = REPO_ROOT / "docs" / "hpc" / "storage_forecast_plan.json"
        report_path.write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(report_path), artifact_path="storage_forecast")

    return aggregate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate QE storage footprint")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--runs", type=int, default=None, help="Number of candidates to plan for")
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=1.5,
        help="Multiplier to cover retries and checkpoints (default: 1.5)",
    )
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default="dft_benchmarks")
    parser.add_argument(
        "--summary-name",
        default=DEFAULT_SUMMARY_NAME,
        help="Override benchmark summary filename (default: benchmark_summary.json)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file describing multiple job classes to aggregate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = estimate_storage(
        request_id=args.request_id,
        num_candidates=args.runs,
        safety_factor=args.safety_factor,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        summary_name=args.summary_name,
        config_path=args.config,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
