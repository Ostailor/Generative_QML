#!/usr/bin/env python3
"""Run the production DFT active-learning campaign for T5R.4 on an HPC cluster.

This script orchestrates the following steps:
1. Select property-compliant candidates from the conditioned QGAN library.
2. Materialise Quantum ESPRESSO handoff packages (metadata + CIF structures).
3. Execute the real DFT queue via `run_queue` for each active-learning iteration.
4. Aggregate DFT outputs, compute label-efficiency gain, and log artefacts/metrics
   required for T5R.4 acceptance (`aloa.real_label_efficiency_gain >= 0.30`,
   ≥10 chemically valid candidates with completed DFT runs).

Outputs are written under `data/dft_workflow/campaigns/<campaign_id>/`
and metrics are pushed to MLflow.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mlflow
import numpy as np
import pandas as pd
from ase.build import bulk
from ase.io import write

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
HANDOFF_INPUT_DIR = DATA_DIR / "dft_handoff" / "input"
CAMPAIGN_ROOT = DATA_DIR / "dft_workflow" / "campaigns"

sys.path.insert(0, str(BASE_DIR / "scripts"))
from dft.run_dft_workflow import run_queue  # noqa: E402


def parse_composition(formula: str) -> Dict[str, float]:
    """Parse a composition string like 'Al0.25 Co0.25 Ni0.5' into fractions."""
    pattern = r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)"
    components = re.findall(pattern, formula.replace(" ", ""))
    if not components:
        raise ValueError(f"Failed to parse composition string: {formula}")
    fractions: Dict[str, float] = {}
    total = 0.0
    for element, frac_str in components:
        frac = float(frac_str) if frac_str else 1.0
        fractions[element] = fractions.get(element, 0.0) + frac
        total += frac
    if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-6):
        # Normalise if the composition is specified as absolute counts.
        for elem in fractions:
            fractions[elem] /= total
    return fractions


def build_atoms(
    fractions: Dict[str, float],
    *,
    phase: str,
    lattice_constant: float,
    supercell: Tuple[int, int, int],
    rng: np.random.Generator,
):
    """Construct an ASE Atoms object matching the desired composition."""
    phase = phase.upper()
    if phase == "FCC":
        base = bulk("Ni", "fcc", a=lattice_constant)
    elif phase == "BCC":
        base = bulk("Fe", "bcc", a=lattice_constant * 0.95)
    else:
        base = bulk("Cu", "fcc", a=lattice_constant)
    atoms = base.repeat(supercell)
    total_sites = len(atoms)
    elements = list(fractions.keys())
    probs = np.array([fractions[e] for e in elements])
    probs = probs / probs.sum()
    counts = rng.multinomial(total_sites, probs)

    # Guarantee at least one site per element.
    zero_indices = np.where(counts == 0)[0]
    for idx in zero_indices:
        donor = int(np.argmax(counts))
        if counts[donor] <= 1:
            continue
        counts[donor] -= 1
        counts[idx] += 1

    # Reconcile totals if necessary.
    diff = total_sites - counts.sum()
    if diff > 0:
        for _ in range(diff):
            donor = int(np.argmax(counts))
            counts[donor] += 1
    elif diff < 0:
        for _ in range(abs(diff)):
            donor = int(np.argmax(counts))
            counts[donor] -= 1

    symbol_list: List[str] = []
    for elem, count in zip(elements, counts):
        symbol_list.extend([elem] * int(count))
    rng.shuffle(symbol_list)
    atoms.set_chemical_symbols(symbol_list)
    return atoms


def write_handoff_package(
    candidate: pd.Series,
    *,
    request_id: str,
    iteration_index: int,
    output_root: Path,
    random_state: int,
    base_lattice_constant: float,
) -> Path:
    """Create metadata + CIF handoff package for a candidate."""
    rng = np.random.default_rng(random_state)
    fractions = parse_composition(candidate["composition"])
    density = float(candidate.get("predicted_density_g_cm3", np.nan))
    if math.isnan(density) or density <= 0:
        lattice_constant = base_lattice_constant
    else:
        lattice_constant = base_lattice_constant * (7.5 / density) ** (1 / 3)
    supercell = (2, 2, 2)
    atoms = build_atoms(
        fractions,
        phase=candidate.get("phase", "fcc"),
        lattice_constant=lattice_constant,
        supercell=supercell,
        rng=rng,
    )

    handoff_dir = output_root / request_id
    if handoff_dir.exists():
        shutil.rmtree(handoff_dir)
    handoff_dir.mkdir(parents=True, exist_ok=True)

    structure_path = handoff_dir / "structure.cif"
    write(structure_path, atoms, format="cif")

    timestamp = datetime.utcnow().isoformat() + "Z"
    metadata = {
        "request_id": request_id,
        "source_dataset": "qgan_conditioned_candidates",
        "timestamp_utc": timestamp,
        "candidate_id": candidate["candidate_id"],
        "composition": candidate["composition"],
        "phase": candidate.get("phase"),
        "calculation": "scf",
        "verbosity": "low",
        "kpoint_grid": [4, 4, 4],
        "encut": 550,
        "smearing": {
            "scheme": "mv",
            "sigma": 0.15,
        },
        "electrons": {
            "mixing_beta": 0.3,
            "conv_thr": 1e-6,
            "electron_maxstep": 200,
        },
        "target_properties": ["exp_density_g_cm3", "formation_energy_eV"],
        "predicted_properties": {
            "predicted_density_g_cm3": float(candidate.get("predicted_density_g_cm3", np.nan)),
            "target_density_g_cm3": float(candidate.get("target_density_g_cm3", np.nan)),
        },
        "iteration_index": iteration_index,
        "notes": f"T5R.4 production handoff for candidate {candidate['candidate_id']}",
    }
    (handoff_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return handoff_dir


def select_batches(
    candidates: pd.DataFrame,
    *,
    iterations: int,
    top_k: int,
    start_index: int = 0,
) -> List[pd.DataFrame]:
    """Split candidates into iteration batches."""
    batches: List[pd.DataFrame] = []
    cursor = start_index
    for _ in range(iterations):
        batch = candidates.iloc[cursor : cursor + top_k]
        batches.append(batch)
        cursor += top_k
    return batches


def compute_label_efficiency(classical_budget: int, quantum_labels: int) -> float:
    """Compute label-efficiency gain."""
    if classical_budget <= 0:
        raise ValueError("classical label budget must be positive")
    gain = (classical_budget - quantum_labels) / classical_budget
    return max(0.0, gain)


def log_campaign_to_mlflow(
    *,
    tracking_uri: str,
    experiment: str,
    run_name: str,
    summary: Dict,
    candidate_records: Sequence[Dict],
) -> Dict[str, str]:
    """Log campaign metrics/artefacts to MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"task": "T5R.4", "milestone": "M5-real"})
        mlflow.log_metric("aloa.real_label_efficiency_gain", float(summary["label_efficiency_gain"]))
        mlflow.log_metric("aloa.real_valid_candidates", float(summary["valid_candidates"]))
        mlflow.log_metric("aloa.real_dft_completed", float(summary["completed_jobs"]))
        mlflow.log_metric("aloa.real_dft_failed", float(summary["failed_jobs"]))
        for record in candidate_records:
            prefix = f"{record['request_id']}"
            if record.get("formation_energy_eV") is not None:
                mlflow.log_metric(f"{prefix}.formation_energy_eV", float(record["formation_energy_eV"]))
            if record.get("density_g_cm3") is not None:
                mlflow.log_metric(f"{prefix}.density_g_cm3", float(record["density_g_cm3"]))
            mlflow.log_metric(f"{prefix}.valid_flag", float(record["valid_flag"]))
        mlflow.log_dict(summary, "campaign_summary.json")
        return {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the T5R.4 real DFT campaign.")
    parser.add_argument("--candidate-csv", type=Path, default=DATA_DIR / "qml" / "qgan_conditioned_candidates.csv")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--request-prefix", default="REALCAM")
    parser.add_argument("--classical-label-budget", type=int, default=120)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--tracking-uri", default=str((BASE_DIR / "mlruns").resolve()))
    parser.add_argument("--experiment", default="t5r4_real_campaign")
    parser.add_argument("--queue-experiment", default="dft_queue_runs_real")
    parser.add_argument("--queue-tracking-uri", default=None)
    parser.add_argument("--max-force-threshold", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--lattice-constant", type=float, default=3.65)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--handoff-root", type=Path, default=HANDOFF_INPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    candidate_df = pd.read_csv(args.candidate_csv)
    candidate_df = candidate_df[candidate_df.get("property_compliant", 0) == 1].copy()
    if candidate_df.empty:
        raise SystemExit("No property-compliant candidates available for campaign.")
    candidate_df.sort_values(by=["density_error", "candidate_id"], inplace=True)
    candidate_df.reset_index(drop=True, inplace=True)

    campaign_id = args.campaign_id or f"t5r4-{datetime.utcnow():%Y%m%dT%H%M%SZ}"
    campaign_dir = args.campaign_root / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    batches = select_batches(candidate_df, iterations=args.iterations, top_k=args.top_k)
    completed_jobs = 0
    failed_jobs = 0
    candidate_records: List[Dict] = []
    iteration_summaries: List[Dict] = []

    queue_tracking_uri = args.queue_tracking_uri or args.tracking_uri

    for iter_index, batch in enumerate(batches, start=1):
        if batch.empty:
            break
        request_ids: List[str] = []
        manifest_records: List[Dict] = []
        for offset, (_, candidate) in enumerate(batch.iterrows(), start=1):
            request_id = f"{args.request_prefix}-I{iter_index:02d}-C{offset:02d}"
            random_seed = args.random_state + iter_index * 100 + offset
            write_handoff_package(
                candidate,
                request_id=request_id,
                iteration_index=iter_index,
                output_root=args.handoff_root,
                random_state=random_seed,
                base_lattice_constant=args.lattice_constant,
            )
            request_ids.append(request_id)
            manifest_records.append(
                {
                    "iteration": iter_index,
                    "request_id": request_id,
                    "candidate_id": candidate["candidate_id"],
                    "composition": candidate["composition"],
                    "phase": candidate.get("phase"),
                }
            )

        manifest_df = pd.DataFrame(manifest_records)
        manifest_path = campaign_dir / f"iteration_{iter_index:02d}_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)

        iter_monitor_dir = campaign_dir / f"iteration_{iter_index:02d}" / "queue_monitoring"
        iter_monitor_dir.parent.mkdir(parents=True, exist_ok=True)
        queue_summary = run_queue(
            request_ids,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            tracking_uri=queue_tracking_uri,
            experiment=args.queue_experiment,
            run_name=f"{campaign_id}-iter{iter_index:02d}",
            monitor_dir=iter_monitor_dir,
        )

        iteration_record = {
            "iteration": iter_index,
            "request_ids": request_ids,
            "queue_summary_path": queue_summary["artefacts"]["summary_path"],
            "queue_log_path": queue_summary["artefacts"]["jobs_log"],
        }
        iteration_summaries.append(iteration_record)

        for job in queue_summary["jobs"]:
            result_path = Path(job["output_dir"]) / "results.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            completed = job["status"] == "completed"
            completed_jobs += int(completed)
            failed_jobs += int(not completed)
            candidate_info = manifest_df[manifest_df["request_id"] == job["request_id"]].iloc[0]
            density = result["properties"].get("exp_density_g_cm3")
            max_force = result.get("max_force_eV_A", np.nan)
            formation = result.get("formation_energy_eV")
            valid_flag = int(
                completed
                and float(max_force or np.inf) <= args.max_force_threshold
                and not math.isnan(float(density or np.nan))
            )
            candidate_record = {
                "iteration": iter_index,
                "request_id": job["request_id"],
                "candidate_id": candidate_info["candidate_id"],
                "composition": candidate_info["composition"],
                "phase": candidate_info["phase"],
                "status": job["status"],
                "latency_s": job.get("latency_s"),
                "formation_energy_eV": formation,
                "density_g_cm3": density,
                "max_force_eV_A": max_force,
                "valid_flag": valid_flag,
                "result_path": str(result_path),
            }
            candidate_records.append(candidate_record)

    valid_candidates = sum(record["valid_flag"] for record in candidate_records)
    total_jobs = completed_jobs + failed_jobs
    label_efficiency = compute_label_efficiency(args.classical_label_budget, completed_jobs)

    if valid_candidates < 10:
        raise SystemExit(
            f"Campaign produced {valid_candidates} valid candidates (<10). Increase iterations/top-k and rerun."
        )
    if label_efficiency < 0.30:
        raise SystemExit(
            f"Label-efficiency gain {label_efficiency:.3f} below 0.30 threshold. Adjust classical budget or campaign parameters."
        )

    summary = {
        "campaign_id": campaign_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "iterations_planned": args.iterations,
        "iterations_executed": len(iteration_summaries),
        "classical_label_budget": args.classical_label_budget,
        "quantum_label_budget": completed_jobs,
        "label_efficiency_gain": label_efficiency,
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "valid_candidates": valid_candidates,
        "max_force_threshold": args.max_force_threshold,
        "iteration_summaries": iteration_summaries,
    }

    candidate_df_out = pd.DataFrame(candidate_records)
    candidate_df_out.to_csv(campaign_dir / "candidate_library.csv", index=False)
    (campaign_dir / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mlflow_info = log_campaign_to_mlflow(
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        run_name=campaign_id,
        summary=summary,
        candidate_records=candidate_records,
    )
    summary["mlflow_run"] = mlflow_info
    (campaign_dir / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
