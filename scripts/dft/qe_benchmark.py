#!/usr/bin/env python3
"""Quantum ESPRESSO benchmark runner with MLflow logging.

This utility executes (or replays) a Quantum ESPRESSO job using the
`run_workflow` helper, records runtime/resource statistics, and logs artefacts
to MLflow so the T5R.1 acceptance criteria can be satisfied.

Usage examples:

  # Run a real SCF calculation on the default handoff package
  python scripts/dft/qe_benchmark.py --request-id QAL-0001 \
      --experiment dft_benchmarks

  # Capture metrics from an existing QE output without rerunning (for smoke tests)
  python scripts/dft/qe_benchmark.py --from-log espresso.pwo --dry-run

All runs create a `benchmark_summary.json` artefact alongside the QE outputs
and flip the `mdia.qe_benchmark_ready` metric in MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import mlflow

try:
    import psutil
except ImportError as exc:  # pragma: no cover - psutil is expected in prod envs
    raise SystemExit(
        "psutil is required for QE benchmarking. Install with `pip install psutil`."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or f"file://{(REPO_ROOT / 'tracking' / 'mlflow' / 'benchmarks').resolve()}"
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from scripts.dft.run_dft_workflow import OUTPUT_DIR, run_workflow  # noqa: E402


def _bytes_to_gib(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def _collect_disk_usage(path: Path) -> Dict[str, float]:
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            total += (Path(root) / fname).stat().st_size
    return {
        "bytes": float(total),
        "giB": _bytes_to_gib(total),
    }


def _parse_pw_log(log_path: Path) -> Dict[str, float]:
    wall_time = None
    cpu_time = None
    energy_ry = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "PWSCF" in line and "starts on" in line:
                continue
            if line.strip().startswith("!    total energy"):
                tokens = line.strip().split()
                # !    total energy              =    -4667.32631396 Ry
                try:
                    energy_ry = float(tokens[-2])
                except (ValueError, IndexError):
                    continue
            if "PWSCF" in line and "CPU" in line and "WALL" in line:
                tokens = line.replace(":", " ").split()
                for idx, token in enumerate(tokens):
                    if token.upper() == "CPU" and idx >= 1:
                        try:
                            cpu_time = float(tokens[idx - 1].rstrip("s"))
                        except ValueError:
                            continue
                    if token.upper() == "WALL" and idx >= 1:
                        try:
                            wall_time = float(tokens[idx - 1].rstrip("s"))
                        except ValueError:
                            continue
    metrics: Dict[str, float] = {}
    if wall_time is not None:
        metrics["wall_time_s"] = wall_time
    if cpu_time is not None:
        metrics["cpu_time_s"] = cpu_time
    if energy_ry is not None:
        metrics["total_energy_ry"] = energy_ry
    return metrics


def run_benchmark(
    request_id: str,
    experiment: Optional[str],
    dry_run: bool,
    tracking_uri: Optional[str],
    from_log: Optional[Path],
    job_label: Optional[str],
) -> Dict[str, float]:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)

    summary: Dict[str, float] = {}
    tags = {
        "task": "T5R.1",
        "agent": "MDIA",
        "request_id": request_id,
        "benchmark_mode": "dry" if dry_run else "full",
    }
    if job_label:
        tags["job_label"] = job_label

    with mlflow.start_run(run_name=f"qe-benchmark-{request_id}"):
        mlflow.set_tags(tags)

        pw_command = os.environ.get("QE_PW_COMMAND", os.environ.get("QE_BIN", "pw.x"))
        if pw_command:
            mlflow.log_param("pw_command", pw_command)
            mlflow.log_param("pw_version", Path(pw_command).name)
        if job_label:
            mlflow.log_param("job_label", job_label)

        if dry_run and from_log is None:
            raise ValueError("--dry-run requires --from-log to supply metrics")

        output_dir = OUTPUT_DIR / request_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if from_log:
            metrics = _parse_pw_log(from_log)
            summary.update(metrics)
            mlflow.log_params({"log_source": str(from_log)})
        elif dry_run:
            metrics = {}
        else:
            if output_dir.exists():
                shutil.rmtree(output_dir)

            proc = psutil.Process()
            start_rusage_self = resource.getrusage(resource.RUSAGE_SELF)
            start_rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)

            start_wall = time.perf_counter()
            results = run_workflow(request_id)
            wall_time = time.perf_counter() - start_wall

            end_rusage_self = resource.getrusage(resource.RUSAGE_SELF)
            end_rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)

            cpu_user = (
                end_rusage_children.ru_utime
                + end_rusage_self.ru_utime
                - start_rusage_children.ru_utime
                - start_rusage_self.ru_utime
            )
            cpu_sys = (
                end_rusage_children.ru_stime
                + end_rusage_self.ru_stime
                - start_rusage_children.ru_stime
                - start_rusage_self.ru_stime
            )

            rss_bytes = proc.memory_info().rss
            max_rss_bytes = max(
                end_rusage_self.ru_maxrss,
                end_rusage_children.ru_maxrss,
            )
            # ru_maxrss is kilobytes on Linux, bytes on macOS.
            if sys.platform == "darwin":
                max_rss_bytes = max_rss_bytes
            else:
                max_rss_bytes *= 1024

            metrics = {
                "wall_time_s": wall_time,
                "cpu_time_s": cpu_user + cpu_sys,
                "cpu_user_s": cpu_user,
                "cpu_sys_s": cpu_sys,
                "rss_bytes": float(rss_bytes),
                "max_rss_bytes": float(max_rss_bytes),
            }
            summary.update(results)
            mlflow.log_dict(results, "qe_output/results.json")

        if output_dir.exists():
            disk_usage = _collect_disk_usage(output_dir)
            summary["disk_bytes"] = disk_usage["bytes"]
            summary["disk_gib"] = disk_usage["giB"]
            mlflow.log_artifacts(str(output_dir), artifact_path="qe_output")

        summary.update(metrics)
        if job_label:
            summary["job_label"] = job_label

        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        if "disk_gib" in summary:
            mlflow.log_metric("qe_output_gib", summary["disk_gib"])

        mlflow.log_metric("mdia.qe_benchmark_ready", 1.0)

        summary_path = output_dir / "benchmark_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path), artifact_path="qe_output")

        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or replay a QE benchmark")
    parser.add_argument("--request-id", default="QAL-0001")
    parser.add_argument("--job-label", default=None, help="Logical name for the DFT job class (e.g., scf, relax)")
    parser.add_argument(
        "--experiment",
        default="dft_benchmarks",
        help="MLflow experiment name (default: dft_benchmarks)",
    )
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--dry-run", action="store_true", help="Skip QE execution")
    parser.add_argument(
        "--from-log",
        type=Path,
        help="Path to a Quantum ESPRESSO output (.pwo) for metric extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(
        request_id=args.request_id,
        experiment=args.experiment,
        dry_run=args.dry_run,
        tracking_uri=args.tracking_uri,
        from_log=args.from_log,
        job_label=args.job_label,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
