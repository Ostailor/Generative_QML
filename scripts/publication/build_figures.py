#!/usr/bin/env python3
"""Generate reproducible manuscript/poster figures from project artefacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MANUSCRIPT_FIG_DIR = BASE_DIR / "docs" / "manuscript" / "figures"
POSTER_FIG_DIR = BASE_DIR / "docs" / "poster" / "figures"
DEFAULT_CAMPAIGN_ID = "t5r4-20260211-fasttrack-221-mw4"
DEFAULT_CAMPAIGN_ROOT = DATA_DIR / "dft_workflow" / "campaigns"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return json.loads(path.read_text())


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 180,
        }
    )


def build_real_dft_kpi(campaign_summary: dict, out_path: Path) -> None:
    metrics = ["Label Efficiency", "Valid Candidates", "Completed Jobs"]
    values = [
        float(campaign_summary["label_efficiency_gain"]),
        float(campaign_summary["valid_candidates"]),
        float(campaign_summary["completed_jobs"]),
    ]
    targets = [0.30, 10.0, 10.0]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(metrics))
    ax.bar([i - 0.16 for i in x], values, width=0.32, label="Observed", color="#0A5E4A")
    ax.bar([i + 0.16 for i in x], targets, width=0.32, label="Target", color="#9AA6B2")
    ax.set_xticks(list(x), metrics)
    ax.set_ylabel("Value")
    ax.set_title("Real DFT Campaign KPI vs Acceptance Targets")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_hardware_cost_fidelity(hardware_summary: dict, out_path: Path) -> None:
    df = pd.DataFrame(hardware_summary["by_backend_model"])
    backend_df = (
        df.groupby("backend_id", as_index=False)
        .agg(mean_fidelity=("mean_fidelity", "mean"), total_cost_usd=("total_cost_usd", "sum"))
        .sort_values("mean_fidelity", ascending=False)
    )

    fig, ax1 = plt.subplots(figsize=(7.4, 4.4))
    ax1.bar(backend_df["backend_id"], backend_df["mean_fidelity"], color="#2364AA")
    ax1.set_ylim(0.9, 1.0)
    ax1.set_ylabel("Mean Fidelity")
    ax1.set_title("Backend Fidelity and Cost Trade-off")
    ax1.tick_params(axis="x", rotation=20)

    ax2 = ax1.twinx()
    ax2.plot(backend_df["backend_id"], backend_df["total_cost_usd"], color="#F25F5C", marker="o", linewidth=2)
    ax2.set_ylabel("Total Cost (USD)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_m7_benchmark_summary(m7_summary: dict, out_path: Path) -> None:
    metrics = m7_summary["metrics"]
    labels = [
        "Q vs C Gap",
        "Sensitivity",
        "DFT Delta RMSE",
        "Novelty Gap",
    ]
    values = [
        float(metrics["quantum_vs_classical_gap"]),
        float(metrics["sensitivity_index"]),
        float(metrics["dft_delta_vs_classical_rmse"]),
        float(metrics["novelty_gap"]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(labels, values, color=["#006D77", "#83C5BE", "#FFB703", "#8ECAE6"])
    ax.axhline(0.0, color="#374151", linewidth=1)
    ax.set_title("M7 Benchmarking and Robustness Summary")
    ax.set_ylabel("Metric Value")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manuscript/poster figures")
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    args = parser.parse_args()

    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()

    campaign_summary_path = args.campaign_root / args.campaign_id / "closed_loop_summary.json"
    campaign_summary = _load_json(campaign_summary_path)
    hardware_summary = _load_json(DATA_DIR / "hardware" / "hardware_summary.json")
    m7_summary = _load_json(DATA_DIR / "benchmarks" / "m7" / "m7_summary.json")

    build_real_dft_kpi(campaign_summary, MANUSCRIPT_FIG_DIR / "fig_real_dft_kpi.png")
    build_hardware_cost_fidelity(hardware_summary, MANUSCRIPT_FIG_DIR / "fig_hardware_cost_fidelity.png")
    build_m7_benchmark_summary(m7_summary, MANUSCRIPT_FIG_DIR / "fig_m7_benchmark_summary.png")

    # Mirror to poster bundle
    for fig in MANUSCRIPT_FIG_DIR.glob("*.png"):
        (POSTER_FIG_DIR / fig.name).write_bytes(fig.read_bytes())

    print(
        json.dumps(
            {
                "campaign_summary": str(campaign_summary_path),
                "manuscript_figures": sorted(str(path) for path in MANUSCRIPT_FIG_DIR.glob("*.png")),
                "poster_figures": sorted(str(path) for path in POSTER_FIG_DIR.glob("*.png")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
