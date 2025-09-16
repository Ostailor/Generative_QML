#!/usr/bin/env python3
"""Closed-loop execution combining orchestrator and DFT workflow (T5.2)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dft.run_dft_workflow import run_workflow
from qal_orchestrator import orchestrate

BASE_DIR = Path(__file__).resolve().parents[1]
LOOP_SUMMARY_PATH = BASE_DIR / "data" / "architecture" / "closed_loop_summary.json"


def run_closed_loop(random_state: int = 42, top_k: int = 10, iterations: int = 3) -> dict:
    orchestrator_runs = []
    dft_results_all = []
    for itr in range(iterations):
        orchestrator_runs.append(orchestrate(random_state=random_state + itr, top_k=top_k))
        dft_results_all.append(run_workflow("QAL-0001"))

    summary = {
        "run_id": f"qal-closed-loop-{random_state}",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "iterations_completed": iterations,
        "orchestrator_runs": orchestrator_runs,
        "dft_results": dft_results_all,
    }
    LOOP_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run closed-loop QAL simulation")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    summary = run_closed_loop(args.random_state, args.top_k, args.iterations)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
