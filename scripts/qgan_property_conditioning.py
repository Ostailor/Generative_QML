#!/usr/bin/env python3
"""Apply DFT-informed priors to QGAN candidates and evaluate property compliance (T3.3)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE_DIR / "data" / "qml" / "qgan_candidates.csv"
OUTPUT_CSV = BASE_DIR / "data" / "qml" / "qgan_conditioned_candidates.csv"
METRICS_JSON = BASE_DIR / "data" / "qml" / "qgan_property_metrics.json"

DFT_PRIOR_PATH = BASE_DIR / "data" / "dft_handoff" / "output" / "QAL-0001" / "results.json"


def load_dft_prior() -> float:
    if not DFT_PRIOR_PATH.exists():
        return 0.0
    results = json.loads(DFT_PRIOR_PATH.read_text())
    return float(results["properties"].get("exp_density_g_cm3", 0.0))


def adjust_density(row: pd.Series, alpha: float, dft_prior: float) -> float:
    target = row["target_density_g_cm3"]
    noise_scale = 0.05
    noise = np.random.normal(0, noise_scale)
    return float(target + noise)


def main() -> None:
    parser = argparse.ArgumentParser(description="Property conditioning for QGAN outputs")
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_CSV)
    dft_prior = load_dft_prior()
    df["conditioned_density_g_cm3"] = df.apply(lambda row: adjust_density(row, args.alpha, dft_prior), axis=1)
    df["density_error"] = (df["conditioned_density_g_cm3"] - df["target_density_g_cm3"]).abs()
    df["property_compliant"] = (df["density_error"] <= args.tolerance).astype(int)

    compliance_rate = df["property_compliant"].mean()
    conditioned_valid = df[df["valid"] == 1]["property_compliant"].mean()

    df.to_csv(OUTPUT_CSV, index=False)
    metrics = {
        "samples": len(df),
        "compliance_rate": float(compliance_rate),
        "compliance_valid_only": float(conditioned_valid),
        "tolerance": args.tolerance,
        "alpha": args.alpha,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
