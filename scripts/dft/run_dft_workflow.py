#!/usr/bin/env python3
"""Automated DFT workflow using Quantum ESPRESSO (ASE calculator)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from ase.io import read
    from ase.calculators.espresso import Espresso, EspressoProfile
except ImportError as exc:
    raise SystemExit(
        "ASE with espresso support is required. Install with `pip install ase`."
    ) from exc

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "data" / "dft_handoff" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "dft_workflow"
REPORT_PATH = OUTPUT_DIR / "workflow_report.json"

_PSEUDO_ENV_CANDIDATES = [
    os.environ.get("QE_PSEUDO_DIR"),
    os.environ.get("ESPRESSO_PSEUDO"),
]
PSEUDO_DIR = Path(next((val for val in _PSEUDO_ENV_CANDIDATES if val), "qe_pseudo"))
PW_COMMAND = os.environ.get("QE_PW_COMMAND", "pw.x")

AMU_TO_G = 1.66053906660
EV_TO_RY = 0.0734986176


def _select_pseudopotentials(elements: np.ndarray) -> Dict[str, str]:
    if not PSEUDO_DIR.exists():
        raise FileNotFoundError(f"Pseudopotential directory {PSEUDO_DIR} not found.")
    mapping: Dict[str, str] = {}
    ups = [p for p in PSEUDO_DIR.iterdir() if p.suffix.lower() == ".upf"]
    for elem in elements:
        pattern = elem.lower()
        candidates = [p for p in ups if pattern in p.name.lower()]
        if not candidates:
            raise FileNotFoundError(f"No pseudopotential found for {elem} in {PSEUDO_DIR}")
        mapping[elem] = candidates[0].name
    return mapping


def _build_strain_matrices(amplitude: float) -> List[np.ndarray]:
    epsilons: List[np.ndarray] = []
    # Normal strains
    for axis in range(3):
        strain = np.zeros((3, 3))
        strain[axis, axis] = amplitude
        epsilons.append(strain)
    # Shear strains (yz, xz, xy)
    shear_pairs = [(1, 2), (0, 2), (0, 1)]
    for i, j in shear_pairs:
        strain = np.zeros((3, 3))
        strain[i, j] = amplitude
        strain[j, i] = amplitude
        epsilons.append(strain)
    return epsilons


def _apply_strain(atoms, strain_matrix: np.ndarray):
    strained = atoms.copy()
    base_cell = atoms.get_cell()
    transformation = np.eye(3) + strain_matrix
    new_cell = transformation @ base_cell
    strained.set_cell(new_cell, scale_atoms=True)
    return strained


def _run_single_calculation(atoms, calc_kwargs: dict, outdir: Path, prefix: str) -> dict:
    run_kwargs = copy.deepcopy(calc_kwargs)
    outdir.mkdir(parents=True, exist_ok=True)
    run_kwargs["outdir"] = str(outdir)
    run_kwargs["prefix"] = prefix

    atoms.calc = Espresso(**run_kwargs)
    energy = float(atoms.get_potential_energy())
    stress_voigt = atoms.get_stress(voigt=True).tolist()
    max_force = float(np.abs(atoms.get_forces()).max())

    return {
        "total_energy_eV": energy,
        "stress_voigt": stress_voigt,
        "max_force_eV_A": max_force,
    }


def run_workflow(request_id: str) -> dict:
    input_path = INPUT_DIR / request_id
    output_path = OUTPUT_DIR / request_id
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((input_path / "metadata.json").read_text())
    structure_file = input_path / "structure.cif"
    if not structure_file.exists():
        raise FileNotFoundError(
            f"Structure file {structure_file} not found. Provide a CIF in the handoff package."
        )

    atoms = read(structure_file)
    unique_elements = np.array(list(dict.fromkeys(atoms.get_chemical_symbols())))
    pseudopotentials = _select_pseudopotentials(unique_elements)

    encut_eV = metadata.get("encut", 520)
    ecutwfc = encut_eV * EV_TO_RY
    ecutrho = metadata.get("ecutrho", 8 * ecutwfc)

    smearing_cfg = metadata.get("smearing", {})
    sigma_eV = smearing_cfg.get("sigma", 0.2)
    degauss_ry = smearing_cfg.get("degauss", sigma_eV * EV_TO_RY)
    mixing_beta = metadata.get("electrons", {}).get("mixing_beta", 0.3)
    conv_thr = metadata.get("electrons", {}).get("conv_thr", 1e-6)

    profile = EspressoProfile(command=PW_COMMAND, pseudo_dir=str(PSEUDO_DIR))

    calc_kwargs = {
        "kpts": tuple(metadata.get("kpoint_grid", [4, 4, 4])),
        "occupations": "smearing",
        "smearing": smearing_cfg.get("scheme", "mv"),
        "degauss": degauss_ry,
        "pseudopotentials": pseudopotentials,
        "tstress": True,
        "tprnfor": True,
        "outdir": str(output_path / "qe_tmp" / "base"),
        "prefix": request_id.lower(),
        "profile": profile,
        "input_data": {
            "control": {
                "calculation": metadata.get("calculation", "scf"),
                "verbosity": metadata.get("verbosity", "low"),
            },
            "system": {
                "input_dft": metadata.get("dft_functional", "PBE"),
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
                "occupations": metadata.get("occupations", "smearing"),
            },
            "electrons": {
                "conv_thr": conv_thr,
                "mixing_beta": mixing_beta,
                "electron_maxstep": metadata.get("electrons", {}).get("electron_maxstep", 200),
            },
        },
    }

    base_result = _run_single_calculation(
        atoms,
        calc_kwargs,
        output_path / "qe_tmp" / "base",
        request_id.lower(),
    )

    volume = atoms.get_volume()
    mass = atoms.get_masses().sum()
    density = (mass * AMU_TO_G) / volume

    results = {
        "request_id": request_id,
        "status": "completed",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "total_energy_eV": base_result["total_energy_eV"],
        "formation_energy_eV": base_result["total_energy_eV"] / len(atoms),
        "properties": {
            "exp_density_g_cm3": density,
        },
        "metadata": metadata,
        "strain_results": [],
    }

    stress_cfg = metadata.get("stress_analysis")
    if stress_cfg:
        amplitude = float(stress_cfg.get("strain_amplitude", 0.003))
        max_directions = int(stress_cfg.get("strain_directions", 6))
        strain_matrices = _build_strain_matrices(amplitude)[:max_directions]

        relaxed_atoms = atoms.copy()
        base_cell = relaxed_atoms.get_cell()

        for idx, strain_matrix in enumerate(strain_matrices):
            for sign, label in [(1.0, "positive"), (-1.0, "negative")]:
                signed_matrix = strain_matrix * sign
                strained_atoms = _apply_strain(relaxed_atoms, signed_matrix)

                strain_dir = output_path / f"strain_{idx}_{label}"
                calc_kwargs_strain = copy.deepcopy(calc_kwargs)
                calc_kwargs_strain["input_data"]["control"]["calculation"] = "scf"

                result = _run_single_calculation(
                    strained_atoms,
                    calc_kwargs_strain,
                    strain_dir / "qe_tmp",
                    f"{request_id.lower()}_strain_{idx}_{label}",
                )

                results["strain_results"].append(
                    {
                        "index": idx,
                        "sign": label,
                        "strain_tensor": signed_matrix.tolist(),
                        "total_energy_eV": result["total_energy_eV"],
                        "stress_voigt": result["stress_voigt"],
                        "max_force_eV_A": result["max_force_eV_A"],
                        "output_subdir": str((strain_dir / "qe_tmp").relative_to(output_path)),
                    }
                )

    (output_path / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_path / "log.txt").write_text("Quantum ESPRESSO run completed.\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Quantum ESPRESSO DFT workflow")
    parser.add_argument("request_id", default="QAL-0001", nargs="?")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_workflow(args.request_id)
    REPORT_PATH.write_text(json.dumps({"last_run": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
