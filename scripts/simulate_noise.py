#!/usr/bin/env python3
"""Noise and perturbation simulation module for M1 T1.3.

Generates perturbed datasets according to predefined scenarios to support
robustness evaluations and imputation stress tests.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SIM_OUTPUT_DIR = BASE_DIR / "data" / "simulations"
METADATA_DIR = BASE_DIR / "data" / "metadata"
SIM_REPORT_PATH = METADATA_DIR / "qa_reports" / "noise_simulations_summary.json"

SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ScenarioResult:
    name: str
    description: str
    rows: int
    perturbation_params: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "rows": self.rows,
            "perturbation_params": self.perturbation_params,
        }


def load_dataset(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}_features.parquet"
    return pd.read_parquet(path)


def apply_gaussian_noise(series: pd.Series, sigma: float) -> pd.Series:
    noise = np.random.normal(0, sigma, size=len(series))
    return series + noise


def simulate_perovskite_noise(df: pd.DataFrame) -> ScenarioResult:
    sigma_band_gap = 0.05
    sigma_energy = 0.01
    perturbed = df.copy()
    perturbed["band_gap_eV_noise"] = apply_gaussian_noise(perturbed["band_gap_eV"], sigma_band_gap)
    perturbed["energy_above_hull_eV_noise"] = apply_gaussian_noise(
        perturbed["energy_above_hull_eV"], sigma_energy
    )
    out_path = SIM_OUTPUT_DIR / "perovskites_noise.parquet"
    perturbed.to_parquet(out_path, index=False)
    return ScenarioResult(
        name="perovskites_noise",
        description="Gaussian perturbations on band gap and energy above hull",
        rows=len(perturbed),
        perturbation_params={"sigma_band_gap": sigma_band_gap, "sigma_energy": sigma_energy},
    )


def simulate_hea_dropout(df: pd.DataFrame) -> ScenarioResult:
    dropout_rate = 0.3
    perturbed = df.copy()
    rng = np.random.default_rng()
    numeric_cols = [
        "vickers_hardness",
        "yield_strength_mpa",
        "uts_mpa",
        "elongation_pct",
        "elongation_plastic_pct",
        "exp_youngs_gpa",
        "calc_youngs_gpa",
    ]
    mask = rng.random((len(perturbed), len(numeric_cols))) < dropout_rate
    for idx, col in enumerate(numeric_cols):
        perturbed.loc[mask[:, idx], col] = np.nan
    out_path = SIM_OUTPUT_DIR / "hea_dropout.parquet"
    perturbed.to_parquet(out_path, index=False)
    return ScenarioResult(
        name="hea_dropout",
        description="Random dropout of mechanical properties to stress-test imputation",
        rows=len(perturbed),
        perturbation_params={"dropout_rate": dropout_rate},
    )


def simulate_saa_noise(df: pd.DataFrame) -> ScenarioResult:
    sigma_reaction = 0.1
    perturbed = df.copy()
    perturbed["reaction_energy_eV_noise"] = apply_gaussian_noise(
        perturbed["reaction_energy_eV"], sigma_reaction
    )
    out_path = SIM_OUTPUT_DIR / "saa_noise.parquet"
    perturbed.to_parquet(out_path, index=False)
    return ScenarioResult(
        name="saa_noise",
        description="Gaussian perturbation on reaction energies for SAA dataset",
        rows=len(perturbed),
        perturbation_params={"sigma_reaction_energy": sigma_reaction},
    )


def simulate_saa_bias(df: pd.DataFrame) -> ScenarioResult:
    bias = 0.05
    perturbed = df.copy()
    perturbed["reaction_energy_eV_bias"] = perturbed["reaction_energy_eV"] + bias
    out_path = SIM_OUTPUT_DIR / "saa_bias.parquet"
    perturbed.to_parquet(out_path, index=False)
    return ScenarioResult(
        name="saa_bias",
        description="Systematic positive bias on reaction energies to emulate calibration drift",
        rows=len(perturbed),
        perturbation_params={"bias": bias},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noise/perturbation scenarios")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=["perovskites", "hea", "saa_noise", "saa_bias", "all"],
        default=["all"],
    )
    args = parser.parse_args()
    scenarios = args.scenarios
    if "all" in scenarios:
        scenarios = ["perovskites", "hea", "saa_noise", "saa_bias"]

    results: List[ScenarioResult] = []

    if "perovskites" in scenarios:
        df = load_dataset("perovskites")
        results.append(simulate_perovskite_noise(df))

    if "hea" in scenarios:
        df = load_dataset("hea")
        results.append(simulate_hea_dropout(df))

    if "saa_noise" in scenarios or "saa_bias" in scenarios:
        df = load_dataset("saa")
        if "saa_noise" in scenarios:
            results.append(simulate_saa_noise(df))
        if "saa_bias" in scenarios:
            results.append(simulate_saa_bias(df))

    with open(SIM_REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump([res.to_dict() for res in results], handle, indent=2)
    print(f"[INFO] Simulation summary written to {SIM_REPORT_PATH}")


if __name__ == "__main__":
    main()
