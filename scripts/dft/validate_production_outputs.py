#!/usr/bin/env python3
"""Validate production DFT outputs against reference data (T5R.3)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import mlflow
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DFT_RESULTS_DIR = DATA_DIR / "dft_workflow"
REFERENCE_DATASET = DATA_DIR / "processed" / "hea_features.parquet"
DEFAULT_REPORT_PATH = DFT_RESULTS_DIR / "validation" / "validation_report.json"
DEFAULT_EXPERIMENT = "dft_validation"
DEFAULT_TRACKING_URI = str((BASE_DIR / "mlruns").resolve())


def _load_reference_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Reference dataset not found at {path}")
    df = pd.read_parquet(path)
    df = df.copy()
    df["formula_normalized"] = df["formula"].str.replace(" ", "", regex=False)
    return df.set_index("formula_normalized")


def _normalize_composition(composition: str) -> str:
    return composition.replace(" ", "")


def _load_result(request_id: str) -> Dict:
    result_path = DFT_RESULTS_DIR / request_id / "results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing DFT results for {request_id} at {result_path}")
    with result_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["_result_path"] = str(result_path)
    return data


@dataclass
class PropertyStats:
    name: str
    values: List[float] = field(default_factory=list)
    reference: Optional[float] = None

    def add(self, value: Optional[float]) -> None:
        if value is not None:
            self.values.append(float(value))

    @property
    def mean(self) -> Optional[float]:
        if not self.values:
            return None
        return float(np.mean(self.values))

    @property
    def std(self) -> Optional[float]:
        if len(self.values) <= 1:
            return 0.0 if self.values else None
        return float(np.std(self.values, ddof=1))

    @property
    def ci95(self) -> Optional[float]:
        if not self.values:
            return None
        std = self.std or 0.0
        return float(1.96 * std)

    @property
    def relative_gap(self) -> Optional[float]:
        if not self.values or self.reference in (None, 0.0):
            return None
        mean_val = self.mean
        if mean_val is None:
            return None
        return float(abs(mean_val - self.reference) / abs(self.reference))


def _aggregate_by_formula(
    request_ids: Iterable[str],
    reference_df: pd.DataFrame,
) -> Dict[str, Dict]:
    grouped: Dict[str, Dict] = {}
    for request_id in request_ids:
        data = _load_result(request_id)
        composition = data["metadata"].get("composition")
        if not composition:
            raise KeyError(f"Result {request_id} missing 'composition' in metadata.")
        formula_key = _normalize_composition(composition)
        record = grouped.setdefault(
            formula_key,
            {
                "formula": composition,
                "requests": [],
                "properties": {
                    "density_g_cm3": PropertyStats("density_g_cm3"),
                    "formation_energy_eV_per_atom": PropertyStats("formation_energy_eV_per_atom"),
                },
            },
        )
        record["requests"].append(
            {
                "request_id": request_id,
                "result_path": data["_result_path"],
                "timestamp_utc": data.get("timestamp_utc"),
            }
        )
        formation_energy = data.get("formation_energy_eV")
        if formation_energy is not None:
            record["properties"]["formation_energy_eV_per_atom"].add(float(formation_energy))
        density = data.get("properties", {}).get("exp_density_g_cm3")
        record["properties"]["density_g_cm3"].add(density)

        # Add reference values if available
        if formula_key in reference_df.index:
            ref_row = reference_df.loc[formula_key]
            ref_density = ref_row.get("calc_density_g_cm3")
            if pd.notnull(ref_density):
                record["properties"]["density_g_cm3"].reference = float(ref_density)
        else:
            record.setdefault("missing_reference", True)
    return grouped


def _build_summary_payload(grouped: Dict[str, Dict]) -> Dict:
    timestamp = datetime.utcnow().isoformat() + "Z"
    summary = {
        "timestamp_utc": timestamp,
        "formulas": [],
    }

    max_gap = 0.0
    for formula_key, payload in grouped.items():
        entry = {
            "formula": payload["formula"],
            "normalized_formula": formula_key,
            "requests": payload["requests"],
            "properties": {},
        }
        for prop_name, stats in payload["properties"].items():
            prop_entry = {
                "mean": stats.mean,
                "std": stats.std,
                "ci95": stats.ci95,
                "reference": stats.reference,
                "relative_gap": stats.relative_gap,
                "values": stats.values,
            }
            entry["properties"][prop_name] = prop_entry
            if stats.relative_gap is not None:
                max_gap = max(max_gap, stats.relative_gap)
        entry["missing_reference"] = payload.get("missing_reference", False)
        summary["formulas"].append(entry)

    summary["max_relative_gap"] = max_gap if grouped else None
    summary["status"] = "pass" if max_gap is not None and max_gap <= 0.05 else "review"
    return summary


def _log_to_mlflow(summary: Dict, tracking_uri: str, experiment: str) -> Dict[str, str]:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    run_info: Dict[str, str] = {}
    with mlflow.start_run(run_name=f"dft-validation-{datetime.utcnow():%Y%m%dT%H%M%S}") as run:
        run_info["run_id"] = run.info.run_id
        run_info["experiment_id"] = run.info.experiment_id
        mlflow.set_tags({"task": "T5R.3", "milestone": "M5-real"})
        max_gap = summary.get("max_relative_gap")
        if max_gap is not None:
            mlflow.log_metric("bra.dft_validation_gap", float(max_gap))
        for formula_entry in summary["formulas"]:
            formula_key = formula_entry["normalized_formula"]
            for prop_name, prop_entry in formula_entry["properties"].items():
                prefix = f"{formula_key}.{prop_name}"
                for metric_key in ("mean", "std", "ci95", "reference", "relative_gap"):
                    value = prop_entry.get(metric_key)
                    if value is not None:
                        mlflow.log_metric(f"{prefix}.{metric_key}", float(value))
        artifact_path = "validation_report"
        mlflow.log_dict(summary, f"{artifact_path}/summary.json")
    return run_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate production DFT results against reference data."
    )
    parser.add_argument(
        "--requests",
        nargs="+",
        required=True,
        help="DFT request identifiers to include in the validation.",
    )
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        default=REFERENCE_DATASET,
        help="Path to the reference HEA dataset (parquet).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Destination for the validation report JSON.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_TRACKING_URI}).",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_df = _load_reference_dataset(args.reference_dataset)
    grouped = _aggregate_by_formula(args.requests, reference_df)
    summary = _build_summary_payload(grouped)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_info = _log_to_mlflow(summary, args.tracking_uri, args.experiment)
    summary["mlflow_run"] = run_info
    args.report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
