#!/usr/bin/env python3
"""Generate and run a single vc-relax smoke test locally."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.dft.run_dft_workflow import run_workflow  # noqa: E402
from scripts.hpc.run_real_dft_campaign import (  # noqa: E402
    DATA_DIR,
    write_handoff_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single Quantum ESPRESSO vc-relax smoke test locally.",
        epilog=(
            "Example:\n"
            "  python scripts/dev/run_relax_smoketest.py "
            "--candidate-id QGAN-029 --request-id SMOKE-RELAX-01\n\n"
            "Ensure QE binaries (`pw.x`) and pseudopotentials are available locally."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=DATA_DIR / "qml" / "qgan_conditioned_candidates.csv",
        help="Candidate library (must contain property_compliant column).",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Specific candidate_id to run; defaults to the best property-compliant entry.",
    )
    parser.add_argument(
        "--request-id",
        default="SMOKE-RELAX-01",
        help="Request identifier used for handoff/output directories.",
    )
    parser.add_argument(
        "--handoff-root",
        type=Path,
        default=DATA_DIR / "dft_handoff" / "input",
        help="Where to write the QE handoff package.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=1234,
        help="Seed used for stochastic alloy placement.",
    )
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=3.65,
        help="Base lattice constant (Angstrom) before density adjustment.",
    )
    parser.add_argument(
        "--calculation",
        choices=["vc-relax", "relax", "scf"],
        default="vc-relax",
        help="QE calculation type for the smoke test.",
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=0.03,
        help="Force convergence threshold in eV/Angstrom.",
    )
    parser.add_argument(
        "--pressure-threshold",
        type=float,
        default=2.0,
        help="Pressure convergence threshold in kbar (vc-relax only).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help="Maximum ionic steps before QE aborts.",
    )
    parser.add_argument(
        "--k-grid",
        type=int,
        nargs=3,
        default=[4, 4, 4],
        metavar=("KX", "KY", "KZ"),
        help="Monkhorst-Pack grid (use 2 2 2 for a quick smoke test).",
    )
    parser.add_argument(
        "--ecutwfc",
        type=float,
        default=40.42423968,
        help="Plane-wave cutoff in Ry (reduce to ≈30.0 for quick smoke tests).",
    )
    parser.add_argument(
        "--ecutrho",
        type=float,
        default=323.39391744,
        help="Charge-density cutoff in Ry.",
    )
    parser.add_argument('--resume', action='store_true', help='Resume from existing QE outputs if present.')
    return parser.parse_args()


def select_candidate(candidate_csv: Path, candidate_id: str | None) -> pd.Series:
    df = pd.read_csv(candidate_csv)
    df = df[df.get("property_compliant", 0) == 1].copy()
    if df.empty:
        raise SystemExit(f"No property-compliant candidates found in {candidate_csv}")
    df.sort_values(by=["density_error", "candidate_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if candidate_id is None:
        return df.iloc[0]
    matches = df[df["candidate_id"] == candidate_id]
    if matches.empty:
        raise SystemExit(f"Candidate {candidate_id} not found or not property-compliant.")
    return matches.iloc[0]


def main() -> None:
    args = parse_args()
    candidate = select_candidate(args.candidate_csv, args.candidate_id)

    relaxation_cfg = {
        "ion_dynamics": "bfgs",
        "force_threshold_ev_per_ang": args.force_threshold,
        "max_steps": args.max_steps,
    }
    if args.calculation == "vc-relax":
        relaxation_cfg["cell_dynamics"] = "bfgs"
        relaxation_cfg["pressure_threshold_kbar"] = args.pressure_threshold

    dataset = pd.read_csv(args.candidate_csv)
    property_df = dataset[dataset.get("property_compliant", 0) == 1].copy()
    property_df.sort_values(by=["density_error", "candidate_id"], inplace=True)
    property_df.reset_index(drop=True, inplace=True)

    resume_save_dir = DATA_DIR / "dft_workflow" / args.request_id / "qe_tmp" / "base"
    if not args.resume or not resume_save_dir.exists():
        handoff_dir = write_handoff_package(
            candidate,
            request_id=args.request_id,
            iteration_index=0,
            output_root=args.handoff_root,
            random_state=args.random_state,
            base_lattice_constant=args.lattice_constant,
            calculation=args.calculation,
            relaxation=relaxation_cfg,
            custom_k_grid=args.k_grid,
            custom_cutoffs=(args.ecutwfc, args.ecutrho),
        )
        print(f"[smoke] Handoff written to {handoff_dir}")
    else:
        print(f"[smoke] Resuming from existing handoff at {resume_save_dir.parent}")

    print("[smoke] Launching local QE workflow...")
    result = run_workflow(args.request_id, resume=args.resume)
    print("[smoke] Workflow completed.")
    print(f"[smoke] Total energy (eV): {result['total_energy_eV']}")
    print(f"[smoke] Max force (eV/Ang): {result['max_force_eV_A']}")
    print(f"[smoke] Output directory: {(DATA_DIR / 'dft_workflow' / args.request_id).resolve()}")


if __name__ == "__main__":
    # numpy random seeding ensures reproducibility between runs.
    np.random.seed(0)
    main()
