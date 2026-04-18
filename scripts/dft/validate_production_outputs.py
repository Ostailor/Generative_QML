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
DEFAULT_CAMPAIGN_ROOT = DFT_RESULTS_DIR / "campaigns"


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


def _discover_requests_from_campaign(
    campaign_id: str,
    campaign_root: Path,
) -> tuple[List[str], Dict[str, object]]:
    campaign_dir = campaign_root / campaign_id
    if not campaign_dir.exists():
        raise FileNotFoundError(f"Campaign directory not found: {campaign_dir}")

    metadata: Dict[str, object] = {"campaign_id": campaign_id, "campaign_dir": str(campaign_dir)}
    request_ids: List[str] = []

    candidate_library = campaign_dir / "candidate_library.csv"
    if candidate_library.exists():
        df = pd.read_csv(candidate_library)
        if "request_id" in df.columns:
            request_ids.extend(df["request_id"].dropna().astype(str).tolist())
            metadata["request_source"] = str(candidate_library)

    if not request_ids:
        manifest_paths = sorted(campaign_dir.glob("iteration_*_manifest.csv"))
        for path in manifest_paths:
            df = pd.read_csv(path)
            if "request_id" in df.columns:
                request_ids.extend(df["request_id"].dropna().astype(str).tolist())
        if manifest_paths:
            metadata["request_source"] = [str(path) for path in manifest_paths]

    deduped = sorted(dict.fromkeys(request_ids))
    if not deduped:
        raise ValueError(
            f"Unable to discover request IDs for campaign '{campaign_id}'. "
            "Expected candidate_library.csv or iteration_*_manifest.csv."
        )
    metadata["request_count"] = len(deduped)
    return deduped, metadata


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
            if isinstance(ref_row, pd.DataFrame):
                ref_density_series = pd.to_numeric(
                    ref_row.get("calc_density_g_cm3"),
                    errors="coerce",
                ).dropna()
                ref_density = (
                    float(ref_density_series.mean())
                    if not ref_density_series.empty
                    else None
                )
            else:
                ref_density_raw = ref_row.get("calc_density_g_cm3")
                ref_density = (
                    float(ref_density_raw)
                    if pd.notnull(ref_density_raw)
                    else None
                )
            if ref_density is not None:
                record["properties"]["density_g_cm3"].reference = ref_density
        else:
            record.setdefault("missing_reference", True)
    return grouped


def _build_summary_payload(grouped: Dict[str, Dict]) -> Dict:
    timestamp = datetime.utcnow().isoformat() + "Z"
    summary = {
        "timestamp_utc": timestamp,
        "formulas": [],
    }

    max_gap: Optional[float] = None
    referenced_formulas = 0
    missing_reference_count = 0
    for formula_key, payload in grouped.items():
        entry = {
            "formula": payload["formula"],
            "normalized_formula": formula_key,
            "requests": payload["requests"],
            "properties": {},
        }
        formula_has_reference = False
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
            if stats.reference is not None:
                formula_has_reference = True
            if stats.relative_gap is not None:
                max_gap = stats.relative_gap if max_gap is None else max(max_gap, stats.relative_gap)
        entry["missing_reference"] = payload.get("missing_reference", False)
        if entry["missing_reference"]:
            missing_reference_count += 1
        if formula_has_reference:
            referenced_formulas += 1
        summary["formulas"].append(entry)

    summary["max_relative_gap"] = max_gap
    summary["referenced_formulas"] = referenced_formulas
    summary["missing_reference_count"] = missing_reference_count
    summary["status"] = (
        "pass"
        if (max_gap is not None and max_gap <= 0.05 and referenced_formulas > 0)
        else "review"
    )
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
        default=None,
        help="DFT request identifiers to include in the validation.",
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="Campaign identifier under data/dft_workflow/campaigns used to discover request IDs.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
        help=f"Root directory containing campaign folders (default: {DEFAULT_CAMPAIGN_ROOT}).",
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
    discovered_meta: Dict[str, object] = {}
    request_ids: List[str] = list(args.requests or [])
    if args.campaign_id:
        discovered_ids, discovered_meta = _discover_requests_from_campaign(
            args.campaign_id, args.campaign_root
        )
        request_ids.extend(discovered_ids)

    request_ids = sorted(dict.fromkeys(request_ids))
    if not request_ids:
        raise SystemExit("Provide at least one request via --requests or supply --campaign-id.")

    report_path = args.report_path
    if args.campaign_id and report_path == DEFAULT_REPORT_PATH:
        report_path = args.campaign_root / args.campaign_id / "validation" / "validation_report.json"

    reference_df = _load_reference_dataset(args.reference_dataset)
    grouped = _aggregate_by_formula(request_ids, reference_df)
    summary = _build_summary_payload(grouped)
    if discovered_meta:
        summary["campaign"] = discovered_meta

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_info = _log_to_mlflow(summary, args.tracking_uri, args.experiment)
    summary["mlflow_run"] = run_info
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
