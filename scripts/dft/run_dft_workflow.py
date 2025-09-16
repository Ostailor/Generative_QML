#!/usr/bin/env python3
"""Mock automated DFT workflow for T5.1."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "data" / "dft_handoff" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "dft_workflow"
REPORT_PATH = OUTPUT_DIR / "workflow_report.json"


def run_workflow(request_id: str) -> dict:
    input_path = INPUT_DIR / request_id
    output_path = OUTPUT_DIR / request_id
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((input_path / "metadata.json").read_text())
    scale = 0.01
    simulated_energy = -0.1 + scale * np.random.randn()

    results = {
        "request_id": request_id,
        "status": "completed",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "total_energy_eV": simulated_energy,
        "formation_energy_eV": simulated_energy / 10,
        "properties": {
            "yield_strength_mpa": 900 + 50 * np.random.randn(),
            "exp_density_g_cm3": 7.8 + 0.1 * np.random.randn()
        }
    }

    (output_path / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_path / "log.txt").write_text("DFT workflow completed.\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mock DFT workflow")
    parser.add_argument("request_id", default="QAL-0001", nargs="?")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_workflow(args.request_id)
    REPORT_PATH.write_text(json.dumps({"last_run": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
