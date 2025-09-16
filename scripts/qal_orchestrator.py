#!/usr/bin/env python3
"""Mock orchestration pipeline for QAL (T4.3)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = BASE_DIR / "data" / "qml" / "qgan_conditioned_candidates.csv"
ACQ_METRICS_PATH = BASE_DIR / "data" / "architecture" / "acquisition_metrics.json"
RUN_SUMMARY_PATH = BASE_DIR / "data" / "architecture" / "qal_run_summary.json"


def orchestrate(random_state: int = 42, top_k: int = 10) -> dict:
    rng = np.random.default_rng(random_state)
    candidates = pd.read_csv(CANDIDATE_PATH)
    metrics = json.loads(ACQ_METRICS_PATH.read_text())

    # Score candidates using placeholder acquisition metric (expected improvement final RMSE)
    scores = rng.normal(loc=0.5, scale=0.1, size=len(candidates))
    candidates["score"] = scores
    selected = candidates.nlargest(top_k, "score")

    # Simulate DFT submission results
    selected["dft_status"] = rng.choice(["completed", "queued"], size=top_k, p=[0.7, 0.3])
    selected["dft_energy"] = rng.normal(loc=-0.1, scale=0.02, size=top_k)

    summary = {
        "run_id": f"qal-orchestrator-{random_state}",
        "total_candidates": int(len(candidates)),
        "selected": int(len(selected)),
        "dft_completed": int((selected["dft_status"] == "completed").sum()),
        "acquisition_strategy": "expected_improvement",
        "expected_improvement_gain": metrics["expected_improvement"]["label_efficiency_gain"],
    }

    RUN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QAL orchestrator mock")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    summary = orchestrate(args.random_state, args.top_k)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
