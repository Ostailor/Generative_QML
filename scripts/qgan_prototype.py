#!/usr/bin/env python3
"""Simulated property-conditional quantum generative prototype (T3.2)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONSTRAINT_PATH = BASE_DIR / "data" / "metadata" / "hea_constraints.yaml"
OUTPUT_CSV = BASE_DIR / "data" / "qml" / "qgan_candidates.csv"
METRICS_JSON = BASE_DIR / "data" / "qml" / "qgan_metrics.json"

random.seed(42)
np.random.seed(42)


def load_constraints() -> Dict:
    return yaml.safe_load(CONSTRAINT_PATH.read_text())


def sample_composition(elements: List[str], num_elements: int, min_frac: float, max_frac: float) -> Dict[str, float]:
    selected = np.random.choice(elements, size=num_elements, replace=False)
    fractions = np.random.dirichlet(np.ones(num_elements))
    # rescale to enforce min/max bounds
    fractions = np.clip(fractions, min_frac, None)
    fractions /= fractions.sum()
    fractions = np.clip(fractions, None, max_frac)
    fractions /= fractions.sum()
    return dict(zip(selected, fractions))


def synthesize_candidates(n: int = 100) -> Tuple[pd.DataFrame, float]:
    constraints = load_constraints()
    allowed = constraints["composition_rules"]["allowed_elements"]
    min_elems = constraints["composition_rules"]["min_unique_elements"]
    max_elems = constraints["composition_rules"]["max_unique_elements"]
    min_frac = constraints["composition_rules"]["min_atomic_fraction"]
    max_frac = constraints["composition_rules"]["max_atomic_fraction"]

    records = []
    valid = 0
    for idx in range(n):
        num_elements = np.random.randint(min_elems, max_elems + 1)
        comp = sample_composition(allowed, num_elements, min_frac, max_frac)
        phase = random.choice(["FCC", "BCC", "other"])
        target_density = np.random.uniform(6.0, 9.0)
        predicted_density = target_density + np.random.normal(0, 0.2)
        entry = {
            "candidate_id": f"QGAN-{idx:03d}",
            "composition": " ".join(f"{el}{frac:.2f}" for el, frac in comp.items()),
            "phase": phase,
            "predicted_density_g_cm3": round(float(predicted_density), 3),
            "target_density_g_cm3": round(float(target_density), 3),
            "valid": 1,
        }
        # mark some invalid deliberately
        if random.random() < 0.1:
            entry["valid"] = 0
        else:
            valid += 1
        records.append(entry)
    df = pd.DataFrame(records)
    acceptance = valid / len(df)
    return df, acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated QGAN candidates")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    df, acceptance = synthesize_candidates(args.num_samples)
    df.to_csv(OUTPUT_CSV, index=False)
    metrics = {
        "samples_generated": len(df),
        "valid_samples": int(df["valid"].sum()),
        "acceptance_rate": acceptance,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
