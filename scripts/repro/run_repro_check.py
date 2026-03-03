#!/usr/bin/env python3
"""Run reproducibility checks for M8 and log acceptance metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import mlflow

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
REPRO_DIR = DATA_DIR / "reproducibility"
DEFAULT_TRACKING_URI = str((BASE_DIR / "mlruns").resolve())

REAL_RELEASE_DIR = DATA_DIR / "releases" / "real_dft_campaign_v1"
REAL_RELEASE_MANIFEST = REAL_RELEASE_DIR / "release_manifest.json"
M7_SUMMARY = DATA_DIR / "benchmarks" / "m7" / "m7_summary.json"
HARDWARE_SUMMARY = DATA_DIR / "hardware" / "hardware_summary.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_manifest(root: Path, manifest_path: Path) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    checked = 0
    mismatches: List[str] = []
    for entry in manifest:
        rel = entry["path"]
        path = root / rel
        if not path.exists():
            mismatches.append(f"missing:{rel}")
            continue
        digest = _sha256(path)
        if digest != entry["sha256"]:
            mismatches.append(f"checksum:{rel}")
        checked += 1
    return {"checked": checked, "mismatches": mismatches, "ok": len(mismatches) == 0}


def run_checks() -> Dict[str, object]:
    REPRO_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        REAL_RELEASE_MANIFEST,
        REAL_RELEASE_DIR / "campaign" / "closed_loop_summary.json",
        DATA_DIR / "benchmarks" / "m7" / "m7_metrics_table.csv",
        HARDWARE_SUMMARY,
        M7_SUMMARY,
    ]
    missing = [str(path) for path in required_files if not path.exists()]

    manifest_result = _verify_manifest(REAL_RELEASE_DIR, REAL_RELEASE_MANIFEST) if REAL_RELEASE_MANIFEST.exists() else {"checked": 0, "mismatches": ["manifest_missing"], "ok": False}
    reproduction_success = (len(missing) == 0) and bool(manifest_result["ok"])

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checks": {
            "required_files_present": len(missing) == 0,
            "release_manifest_valid": manifest_result["ok"],
        },
        "required_files": [str(path) for path in required_files],
        "missing_files": missing,
        "manifest_verification": manifest_result,
        "reproduction_success": reproduction_success,
    }
    (REPRO_DIR / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def log_mlflow(report: Dict[str, object], tracking_uri: str) -> Dict[str, str]:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("m8_reproducibility")
    with mlflow.start_run(run_name=f"m8-repro-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}") as run:
        mlflow.set_tags({"task": "T8.2", "milestone": "M8", "agent": "RKMA"})
        mlflow.log_metric("rkma.reproduction_success", 1.0 if report["reproduction_success"] else 0.0)
        mlflow.log_metric("rkma.repro_missing_files", float(len(report["missing_files"])))
        mlflow.log_metric("rkma.repro_manifest_mismatches", float(len(report["manifest_verification"]["mismatches"])))
        mlflow.log_dict(report, "reproduction_report.json")
        return {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M8 reproducibility checks")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    report = run_checks()
    mlflow_run = log_mlflow(report, args.tracking_uri)
    report["mlflow_run"] = mlflow_run
    (REPRO_DIR / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
