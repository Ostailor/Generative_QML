#!/usr/bin/env python3
"""Compute proxy metrics for feature map families (T2.1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_CSV = BASE_DIR / "data" / "qml" / "feature_map_expressivity.csv"
OUT_JSON = BASE_DIR / "data" / "qml" / "feature_map_resources.json"

rng = np.random.default_rng(seed=42)

families = [
    {
        "family": "composition_encoding",
        "layers": layer,
        "qubits": 6,
        "ionq_depth": 2 + 3 * layer,
        "ibm_depth": 4 + 4 * layer,
        "cnot_count": 5 * layer,
    }
    for layer in range(1, 6)
]
families += [
    {
        "family": "local_environment",
        "layers": layer,
        "qubits": 8,
        "ionq_depth": 4 + 5 * layer,
        "ibm_depth": 6 + 6 * layer,
        "cnot_count": 8 * layer,
    }
    for layer in range(2, 5)
]
families += [
    {
        "family": "phase_aware",
        "layers": 1,
        "qubits": 4,
        "ionq_depth": 3,
        "ibm_depth": 5,
        "cnot_count": 4,
    }
]

records: List[dict] = []
for entry in families:
    base = entry.copy()
    # Proxy effective dimension: random baseline scaled by layers and qubits
    eff_dim = rng.normal(loc=1.5, scale=0.1) * base["layers"] * (base["qubits"] / 4)
    base["effective_dimension"] = round(float(eff_dim), 3)
    # Approx expressibility (higher better) using logistic transform
    expressibility = 1 - np.exp(-0.3 * base["effective_dimension"])
    base["expressibility_score"] = round(float(expressibility), 3)
    # Estimated transpilation runtime weight (arbitrary units)
    base["estimated_shots_per_eval"] = int(1000 * base["layers"] * base["qubits"] / 4)
    records.append(base)

pd.DataFrame(records).to_csv(OUT_CSV, index=False)

resources = {
    "composition_encoding": {
        "qubits": 6,
        "layers_range": [1, 5],
        "notes": "Data re-uploading ansatz with ZZ entanglers; moderate depth on IonQ.",
    },
    "local_environment": {
        "qubits": 8,
        "layers_range": [2, 4],
        "notes": "Hardware-efficient CZ ladder capturing coordination features.",
    },
    "phase_aware": {
        "qubits": 4,
        "layers_range": [1, 1],
        "notes": "Phase-controlled rotations with classical RBF hybridization.",
    },
}
OUT_JSON.write_text(json.dumps(resources, indent=2), encoding="utf-8")
print(f"Wrote metrics to {OUT_CSV} and {OUT_JSON}")
