#!/usr/bin/env python3
"""Build paper figures from server-side non-QPU DFT campaign artefacts."""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_REPO_ROOT = Path("/home/om/Projects/Generative_QML")
DEFAULT_SCF_CAMPAIGN = "top-tier-nonqpu-gpu-20260306T211650Z-scf-20260306t211651z"
DEFAULT_VC_CAMPAIGN = "top-tier-nonqpu-gpu-20260303T195828Z-vc-20260303t195829z"
DEFAULT_ELASTIC_DIR = "BENCH-ELASTIC-0001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build figures from server-side DFT campaign artefacts.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--scf-campaign", default=DEFAULT_SCF_CAMPAIGN)
    parser.add_argument("--vc-campaign", default=DEFAULT_VC_CAMPAIGN)
    parser.add_argument("--elastic-dir", default=DEFAULT_ELASTIC_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 180,
        }
    )


def campaign_dir(repo_root: Path, campaign_id: str) -> Path:
    return repo_root / "data" / "dft_workflow" / "campaigns" / campaign_id


def load_candidate_library(campaign_path: Path) -> pd.DataFrame:
    path = campaign_path / "candidate_library.csv"
    df = pd.read_csv(path)
    return df


def build_candidate_landscape(repo_root: Path, scf_campaign: str, out_path: Path) -> None:
    df = load_candidate_library(campaign_dir(repo_root, scf_campaign))
    df = df[df["status"] == "completed"].copy()
    df["formation_energy_eV"] = pd.to_numeric(df["formation_energy_eV"])
    df["density_g_cm3"] = pd.to_numeric(df["density_g_cm3"])
    df["iteration"] = pd.to_numeric(df["iteration"])

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    cmap = plt.get_cmap("viridis")
    colors = [cmap((it - 1) / max(1, df["iteration"].max() - 1)) for it in df["iteration"]]
    ax.scatter(
        df["density_g_cm3"],
        df["formation_energy_eV"],
        c=colors,
        s=52,
        edgecolor="#203040",
        linewidth=0.4,
        alpha=0.88,
    )

    ax.set_title("Server-Side Real DFT Candidate Landscape")
    ax.set_xlabel("Density (g/cm$^3$)")
    ax.set_ylabel("Formation energy (eV)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_screening_progress(repo_root: Path, scf_campaign: str, out_path: Path) -> None:
    df = load_candidate_library(campaign_dir(repo_root, scf_campaign))
    df = df[df["status"] == "completed"].copy()
    df["formation_energy_eV"] = pd.to_numeric(df["formation_energy_eV"])
    df["density_g_cm3"] = pd.to_numeric(df["density_g_cm3"])
    df["iteration"] = pd.to_numeric(df["iteration"])
    df = df.sort_values(["iteration", "request_id"]).reset_index(drop=True)
    df["job_index"] = range(1, len(df) + 1)
    df["best_energy_so_far"] = df["formation_energy_eV"].cummin()
    df["best_density_so_far"] = df["density_g_cm3"].cummin()

    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    ax1.plot(df["job_index"], df["best_energy_so_far"], color="#0F766E", linewidth=2.4, label="Best formation energy so far")
    ax1.set_xlabel("Completed screening jobs")
    ax1.set_ylabel("Best formation energy so far (eV)")
    ax1.set_title("Server-Side Screening Progress")

    ax2 = ax1.twinx()
    ax2.plot(df["job_index"], df["best_density_so_far"], color="#1D4ED8", linewidth=2.2, linestyle="--", label="Lowest density so far")
    ax2.set_ylabel("Lowest density so far (g/cm$^3$)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def collect_elastic_convergence(repo_root: Path, elastic_dir: str) -> pd.DataFrame:
    root = repo_root / "data" / "dft_workflow" / elastic_dir
    rows = []
    for xml_path in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        steps = tree.findtext(".//n_scf_steps")
        scf_error = tree.findtext(".//scf_error")
        if not steps or not scf_error:
            continue
        rel = xml_path.relative_to(root)
        state = rel.parts[0]
        rows.append(
            {
                "state": "reference" if state == "qe_tmp" else state,
                "n_scf_steps": int(steps),
                "scf_error": float(scf_error),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["state"]).sort_values("state").reset_index(drop=True)
    return out


def build_elastic_convergence(repo_root: Path, elastic_dir: str, out_path: Path) -> None:
    df = collect_elastic_convergence(repo_root, elastic_dir)
    df["rank"] = range(1, len(df) + 1)
    df["neglog10_error"] = -df["scf_error"].apply(math.log10)

    fig, ax1 = plt.subplots(figsize=(7.8, 4.8))
    colors = plt.get_cmap("plasma")(df["rank"] / max(1, len(df)))
    ax1.bar(df["rank"], df["n_scf_steps"], color=colors, width=0.78)
    ax1.set_xlabel("Elastic benchmark state")
    ax1.set_ylabel("SCF steps to convergence")
    ax1.set_title("QE Convergence Across Elastic Benchmark States")
    ax1.set_xticks(df["rank"], [state.replace("_", " ") for state in df["state"]], rotation=25, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(df["rank"], df["neglog10_error"], color="#0F172A", linewidth=2, marker="o", markersize=3)
    ax2.set_ylabel("Final -log10(SCF error)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    build_candidate_landscape(args.repo_root, args.scf_campaign, args.out_dir / "fig_server_side_candidate_landscape.png")
    build_screening_progress(args.repo_root, args.scf_campaign, args.out_dir / "fig_server_side_screening_progress.png")
    build_elastic_convergence(args.repo_root, args.elastic_dir, args.out_dir / "fig_qe_elastic_benchmark_convergence.png")
    print(
        json.dumps(
            {
                "repo_root": str(args.repo_root),
                "scf_campaign": args.scf_campaign,
                "vc_campaign": args.vc_campaign,
                "elastic_dir": args.elastic_dir,
                "out_dir": str(args.out_dir),
                "figures": sorted(p.name for p in args.out_dir.glob("*.png")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
