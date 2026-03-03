#!/usr/bin/env python3
"""Run M7 benchmarking and robustness analyses using real campaign outputs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
BENCH_DIR = DATA_DIR / "benchmarks" / "m7"
DEFAULT_TRACKING_URI = str((BASE_DIR / "mlruns").resolve())
DEFAULT_CAMPAIGN_ID = "t5r4-20260211-fasttrack-221-mw4"
DEFAULT_CAMPAIGN_ROOT = DATA_DIR / "dft_workflow" / "campaigns"

CLASSICAL_AL = DATA_DIR / "qml" / "classical_al_metrics.json"
QSVR_METRICS = DATA_DIR / "qml" / "qsvr_metrics.json"
QGPR_METRICS = DATA_DIR / "qml" / "qgpr_metrics.json"
HARDWARE_SUMMARY = DATA_DIR / "hardware" / "hardware_summary.json"


def _resolve_campaign_inputs(campaign_id: str, campaign_root: Path) -> tuple[Path, Path]:
    campaign_dir = campaign_root / campaign_id
    summary = campaign_dir / "closed_loop_summary.json"
    library = campaign_dir / "candidate_library.csv"
    return summary, library


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return json.loads(path.read_text())


def _compute_sensitivity_index(candidate_df: pd.DataFrame, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if candidate_df.empty:
        return 0.0
    base = candidate_df["formation_energy_eV"].fillna(candidate_df["formation_energy_eV"].median()).to_numpy()
    if len(base) < 3:
        return 0.0

    perturb_scales = [0.01, 0.025, 0.05, 0.075]
    responses: List[float] = []
    for scale in perturb_scales:
        noise = rng.normal(0.0, scale, size=len(base))
        perturbed = base * (1 + noise)
        response = float(np.std(perturbed) / (abs(np.mean(perturbed)) + 1e-9))
        responses.append(response)
    return float(np.mean(responses))


def run_analysis(seed: int, campaign_id: str = DEFAULT_CAMPAIGN_ID, campaign_root: Path = DEFAULT_CAMPAIGN_ROOT) -> Dict[str, object]:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    campaign_summary_path, campaign_library_path = _resolve_campaign_inputs(campaign_id, campaign_root)
    classical_al = _load_json(CLASSICAL_AL)
    qsvr = _load_json(QSVR_METRICS)
    qgpr = _load_json(QGPR_METRICS)
    hardware = _load_json(HARDWARE_SUMMARY)
    campaign_summary = _load_json(campaign_summary_path)
    candidates = pd.read_csv(campaign_library_path)

    real_label_eff = float(campaign_summary["label_efficiency_gain"])
    quantum_rmse = float(qsvr["rmse_quantum"])
    classical_rmse = float(qsvr["rmse_classical"])
    rmse_gain = classical_rmse - quantum_rmse
    novelty_gap = float(_load_json(DATA_DIR / "qml" / "generative_novelty_metrics.json")["novelty_gap"])

    classical_final_rmse = float(classical_al["final_rmse"])
    dft_delta_vs_classical = classical_final_rmse - quantum_rmse
    sensitivity_index = _compute_sensitivity_index(candidates, seed)

    hardware_mean_fidelity = float(hardware["totals"]["mean_fidelity"])
    quantum_vs_classical_gap = float((real_label_eff - 0.30) + max(0.0, rmse_gain))

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "campaign_summary": str(campaign_summary_path),
            "campaign_library": str(campaign_library_path),
            "classical_al": str(CLASSICAL_AL),
            "qsvr_metrics": str(QSVR_METRICS),
            "qgpr_metrics": str(QGPR_METRICS),
            "hardware_summary": str(HARDWARE_SUMMARY),
        },
        "metrics": {
            "real_label_efficiency_gain": real_label_eff,
            "qsvr_rmse_gain": rmse_gain,
            "dft_delta_vs_classical_rmse": dft_delta_vs_classical,
            "qgpr_coverage": float(qgpr["coverage_quantum"]),
            "novelty_gap": novelty_gap,
            "sensitivity_index": sensitivity_index,
            "hardware_mean_fidelity": hardware_mean_fidelity,
            "quantum_vs_classical_gap": quantum_vs_classical_gap,
        },
        "headline_policy": {
            "real_only_headline_claims": True,
            "simulated_results_position": "ablation_appendix_only",
        },
    }

    (BENCH_DIR / "m7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary["metrics"]]).to_csv(BENCH_DIR / "m7_metrics_table.csv", index=False)
    return summary


def log_mlflow(summary: Dict[str, object], tracking_uri: str) -> Dict[str, Dict[str, str]]:
    mlflow.set_tracking_uri(tracking_uri)

    outputs: Dict[str, Dict[str, str]] = {}
    metrics = summary["metrics"]

    mlflow.set_experiment("m7_benchmarking")
    with mlflow.start_run(run_name=f"m7-t72-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}") as run_t72:
        mlflow.set_tags({"task": "T7.2", "milestone": "M7", "agent": "BRA"})
        mlflow.log_metric("bra.sensitivity_index", float(metrics["sensitivity_index"]))
        mlflow.log_metric("bra.hardware_mean_fidelity", float(metrics["hardware_mean_fidelity"]))
        mlflow.log_dict(summary, "m7_t72_summary.json")
        outputs["T7.2"] = {"run_id": run_t72.info.run_id, "experiment_id": run_t72.info.experiment_id}

    with mlflow.start_run(run_name=f"m7-t74-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}") as run_t74:
        mlflow.set_tags({"task": "T7.4", "milestone": "M7", "agent": "BRA"})
        mlflow.log_metric("bra.quantum_vs_classical_gap", float(metrics["quantum_vs_classical_gap"]))
        mlflow.log_metric("bra.dft_delta_vs_classical_rmse", float(metrics["dft_delta_vs_classical_rmse"]))
        mlflow.log_metric("bra.real_label_efficiency_gain", float(metrics["real_label_efficiency_gain"]))
        mlflow.log_dict(summary, "m7_t74_summary.json")
        outputs["T7.4"] = {"run_id": run_t74.info.run_id, "experiment_id": run_t74.info.experiment_id}

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M7 benchmark + robustness analyses")
    parser.add_argument("--seed", type=int, default=20260211)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    summary = run_analysis(args.seed, campaign_id=args.campaign_id, campaign_root=args.campaign_root)
    mlflow_runs = log_mlflow(summary, args.tracking_uri)
    summary["mlflow_runs"] = mlflow_runs
    (BENCH_DIR / "m7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(BENCH_DIR / "m7_summary.json"), "mlflow_runs": mlflow_runs}, indent=2))


if __name__ == "__main__":
    main()
