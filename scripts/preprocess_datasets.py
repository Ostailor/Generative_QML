#!/usr/bin/env python3
"""Feature engineering pipelines for M1 T1.2.

Generates processed datasets with standardized schemas, cleaned numeric values,
engineered descriptors, and QA summaries for perovskites, high-entropy alloys,
and doped nanoparticle datasets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
METADATA_DIR = BASE_DIR / "data" / "metadata"
QA_DIR = METADATA_DIR / "qa_reports"

PEROVSKITE_OUTPUT = PROCESSED_DIR / "perovskites_features.parquet"
HEA_OUTPUT = PROCESSED_DIR / "hea_features.parquet"
SAA_OUTPUT = PROCESSED_DIR / "saa_features.parquet"

QA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Summary:
    name: str
    rows: int
    columns: List[str]
    missing: Dict[str, int]
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": self.columns,
            "missing": self.missing,
            "notes": self.notes,
        }


def write_summary(summary: Summary) -> None:
    out_path = QA_DIR / f"{summary.name}_summary.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2)
    print(f"[INFO] QA summary written to {out_path}")


def load_perovskites() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "perovskites" / "materials_project_perovskites.csv")
    df.rename(
        columns={
            "pretty_formula": "formula",
            "band_gap": "band_gap_eV",
            "formation_energy_per_atom": "formation_energy_per_atom_eV",
            "e_above_hull": "energy_above_hull_eV",
        },
        inplace=True,
    )
    df["spacegroup"] = df["spacegroup"].fillna("UNK")
    df["is_insulator"] = df["band_gap_eV"].apply(lambda x: 1 if x and x >= 0.1 else 0)
    df["log_band_gap"] = df["band_gap_eV"].apply(lambda x: math.log(x) if x and x > 0 else None)
    return df


def load_hea() -> pd.DataFrame:
    path = RAW_DIR / "high_entropy_alloys" / "High Entropy Alloy Properties.csv"
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[/\\()$]", "", regex=True)
        .str.replace(r"\\mu", "mu", regex=True)
        .str.replace(r"\^", "", regex=True)
        .str.replace(":", "", regex=False)
        .str.replace(r"\.", "", regex=True)
    )
    df.rename(
        columns={
            "identifier_reference_id": "reference_id",
            "formula": "formula",
            "property_microstructure": "microstructure",
            "property_processing_method": "processing_method",
            "property_bccfccother": "phase_label",
            "property_grain_size_mum": "grain_size_um",
            "property_exp_density_gcm3": "exp_density_g_cm3",
            "property_calculated_density_gcm3": "calc_density_g_cm3",
            "property_hv": "vickers_hardness",
            "property_type_of_test": "test_type",
            "property_test_temperature_circc": "test_temperature_c",
            "property_ys_mpa": "yield_strength_mpa",
            "property_uts_mpa": "uts_mpa",
            "property_elongation_%": "elongation_pct",
            "property_elongation_plastic_%": "elongation_plastic_pct",
            "property_exp_young_modulus_gpa": "exp_youngs_gpa",
            "property_calculated_young_modulus_gpa": "calc_youngs_gpa",
            "property_o_content_wppm": "oxygen_wppm",
            "property_n_content_wppm": "nitrogen_wppm",
            "property_c_content_wppm": "carbon_wppm",
        },
        inplace=True,
    )
    numeric_cols = [
        "grain_size_um",
        "exp_density_g_cm3",
        "calc_density_g_cm3",
        "vickers_hardness",
        "test_temperature_c",
        "yield_strength_mpa",
        "uts_mpa",
        "elongation_pct",
        "elongation_plastic_pct",
        "exp_youngs_gpa",
        "calc_youngs_gpa",
        "oxygen_wppm",
        "nitrogen_wppm",
        "carbon_wppm",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["phase_bcc"] = df["phase_label"].str.contains("BCC", case=False, na=False).astype(int)
    df["phase_fcc"] = df["phase_label"].str.contains("FCC", case=False, na=False).astype(int)
    df["phase_other"] = (~((df["phase_bcc"] == 1) | (df["phase_fcc"] == 1))).astype(int)
    return df


def parse_sites(value: Optional[str]) -> Tuple[str, str, str]:
    if not value or not isinstance(value, str):
        return ("", "", "")
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return ("", "", "")
    if not data:
        return ("", "", "")
    site_key = next(iter(data.keys()))
    raw = data.get(site_key, "")
    parts = raw.split("|")
    if len(parts) == 3:
        return (site_key, parts[0], parts[2])
    if len(parts) == 2:
        return (site_key, parts[0], parts[1])
    return (site_key, raw, "")


def load_saa() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "doped_nanoparticles" / "catalysis_hub_single_atom_alloy.csv")
    df[["adsorbate", "site_geometry", "coordination"]] = df["sites"].apply(lambda x: pd.Series(parse_sites(x)))
    df.drop(columns=["sites"], inplace=True)
    df["reaction_energy_eV"] = pd.to_numeric(df["reaction_energy_eV"], errors="coerce")
    df["is_exothermic"] = df["reaction_energy_eV"].apply(lambda x: 1 if x is not None and x < 0 else 0)
    return df


def summarize(df: pd.DataFrame, name: str) -> Summary:
    missing = df.isna().sum().to_dict()
    notes: List[str] = []
    if df.empty:
        notes.append("Dataset is empty")
    return Summary(name=name, rows=len(df), columns=list(df.columns), missing=missing, notes=notes)


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)
    print(f"[INFO] Wrote {len(df)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing pipelines")
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=["perovskites", "hea", "saa", "all"],
        default=["all"],
    )
    args = parser.parse_args()
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["perovskites", "hea", "saa"]

    if "perovskites" in datasets:
        df = load_perovskites()
        save_dataset(df, PEROVSKITE_OUTPUT)
        write_summary(summarize(df, "perovskites"))

    if "hea" in datasets:
        df = load_hea()
        save_dataset(df, HEA_OUTPUT)
        write_summary(summarize(df, "hea"))

    if "saa" in datasets:
        df = load_saa()
        save_dataset(df, SAA_OUTPUT)
        write_summary(summarize(df, "saa"))


if __name__ == "__main__":
    main()
