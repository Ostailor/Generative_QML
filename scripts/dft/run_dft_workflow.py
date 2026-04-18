#!/usr/bin/env python3
"""Automated DFT workflow using Quantum ESPRESSO (ASE calculator) with queue orchestration."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mlflow
import numpy as np

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.units import Bohr, Ry
    from ase.calculators.espresso import Espresso, EspressoProfile
except ImportError as exc:
    raise SystemExit(
        "ASE with espresso support is required. Install with `pip install ase`."
    ) from exc

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "data" / "dft_handoff" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "dft_workflow"
REPORT_PATH = OUTPUT_DIR / "workflow_report.json"
QUEUE_RUN_DIR = OUTPUT_DIR / "queue_runs"
DEFAULT_QUEUE_EXPERIMENT = "dft_queue_runs"
DEFAULT_TRACKING_URI = str((BASE_DIR / "mlruns").resolve())

_PSEUDO_ENV_CANDIDATES = [
    os.environ.get("QE_PSEUDO_DIR"),
    os.environ.get("ESPRESSO_PSEUDO"),
]
_pseudo_dir_raw = next((val for val in _PSEUDO_ENV_CANDIDATES if val), None)
if _pseudo_dir_raw:
    _pseudo_dir_path = Path(_pseudo_dir_raw).expanduser()
    if not _pseudo_dir_path.is_absolute():
        _pseudo_dir_path = (BASE_DIR / _pseudo_dir_path).resolve()
    PSEUDO_DIR = _pseudo_dir_path
else:
    PSEUDO_DIR = (BASE_DIR / "qe_pseudo").resolve()
PW_COMMAND = os.environ.get("QE_PW_COMMAND", "pw.x")

AMU_TO_G = 1.66053906660
EV_TO_RY = 0.0734986176
ANGSTROM_TO_BOHR = 1.8897259886
EV_A3_TO_KBAR = 1602.1766208


def _log(message: str) -> None:
    print(f"[run_dft_workflow] {message}", flush=True)


def _normalized_metadata_bytes(path: Path) -> bytes:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return path.read_bytes()
    for key in ("timestamp_utc", "notes"):
        payload.pop(key, None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")


def _settings_metadata_subset(metadata: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "request_id",
        "candidate_id",
        "composition",
        "phase",
        "calculation",
        "verbosity",
        "kpoint_grid",
        "encut",
        "dft_functional",
        "occupations",
        "qe_overrides",
        "smearing",
        "electrons",
        "relaxation",
        "stress_analysis",
        "control",
        "system",
        "ions",
        "cell",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _settings_metadata_signature(metadata: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _settings_metadata_subset(metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        ).encode("utf-8")
    ).hexdigest()


def _compute_settings_hash(input_path: Path) -> str:
    sha = hashlib.sha256()
    hash_candidates = [
        input_path / "metadata.json",
        input_path / "structure.cif",
        input_path / "pseudopotentials.csv",
        input_path / "vasp_settings.json",
    ]
    for path in hash_candidates:
        if not path.exists():
            continue
        sha.update(path.name.encode("utf-8"))
        sha.update(b"\n")
        if path.name == "metadata.json":
            sha.update(_normalized_metadata_bytes(path))
        else:
            sha.update(path.read_bytes())
    return sha.hexdigest()


def _compute_structure_signature(atoms: Atoms) -> str:
    payload = {
        "symbols": atoms.get_chemical_symbols(),
        "cell_ang": np.round(np.array(atoms.get_cell(), dtype=float), decimals=12).tolist(),
        "positions_ang": np.round(np.array(atoms.get_positions(), dtype=float), decimals=12).tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _infer_evidence_tier(request_id: str, metadata: Dict[str, Any]) -> str:
    rid = request_id.upper()
    if rid.startswith("REALCAM") or rid.startswith("BENCH") or rid.startswith("QUEUE"):
        return "production_dft"
    if rid.startswith("SMOKE"):
        return "smoke_test"
    source = str(metadata.get("source_dataset", "")).lower()
    if "qgan" in source or "hea" in source:
        return "pilot_dft"
    return "simulation"


def _select_pseudopotentials(elements: np.ndarray) -> Dict[str, str]:
    if not PSEUDO_DIR.exists():
        raise FileNotFoundError(f"Pseudopotential directory {PSEUDO_DIR} not found.")
    mapping: Dict[str, str] = {}
    ups = sorted((p for p in PSEUDO_DIR.iterdir() if p.suffix.lower() == ".upf"), key=lambda p: p.name.lower())
    for elem in elements:
        symbol = str(elem).strip()
        pattern = symbol.lower()

        # Prefer exact element filename match (e.g., "Y.upf").
        exact = [p for p in ups if p.stem.lower() == pattern]
        if exact:
            mapping[elem] = exact[0].name
            continue

        # Fallback: allow prefixed variants like "Y_pbe..." while preventing
        # accidental substring matches (e.g., Y -> Dy, Yb).
        prefixed = []
        for p in ups:
            stem = p.stem.lower()
            if not stem.startswith(pattern):
                continue
            if len(stem) == len(pattern):
                prefixed.append(p)
                continue
            next_char = stem[len(pattern)]
            if not next_char.isalpha():
                prefixed.append(p)

        if not prefixed:
            raise FileNotFoundError(f"No pseudopotential found for {elem} in {PSEUDO_DIR}")
        mapping[elem] = prefixed[0].name
    return mapping


def _read_upf_recommended_cutoffs(path: Path) -> tuple[float | None, float | None]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    head = text[:8192]

    def _extract(attr: str) -> float | None:
        match = re.search(rf'{attr}\s*=\s*"([0-9]+(?:\.[0-9]+)?)"', head)
        if match:
            return float(match.group(1))
        return None

    return _extract("wfc_cutoff"), _extract("rho_cutoff")


def _recommended_cutoffs_ry(pseudopotentials: Dict[str, str]) -> tuple[float | None, float | None]:
    recommended_wfc: List[float] = []
    recommended_rho: List[float] = []

    for filename in pseudopotentials.values():
        path = PSEUDO_DIR / filename
        if not path.exists():
            continue
        wfc_cutoff, rho_cutoff = _read_upf_recommended_cutoffs(path)
        if wfc_cutoff is not None and wfc_cutoff > 0:
            recommended_wfc.append(wfc_cutoff)
        if rho_cutoff is not None and rho_cutoff > 0:
            recommended_rho.append(rho_cutoff)

    return (
        max(recommended_wfc) if recommended_wfc else None,
        max(recommended_rho) if recommended_rho else None,
    )


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


def _read_qe_log_tail(outdir: Path, max_chars: int = 4000) -> str:
    chunks: List[str] = []
    for filename in ("espresso.err", "espresso.pwo", "pwscf.out"):
        path = outdir / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text:
            chunks.append(text[-max_chars:])
    return "\n".join(chunks)


def _is_scf_instability_failure(outdir: Path) -> bool:
    tail = _read_qe_log_tail(outdir).lower()
    patterns = (
        "error in routine c_bands",
        "too many bands are not converged",
        "error in routine broyden",
        "minimization algorithm failed to find fermi energy",
        "convergence not achieved after",
        "history already reset at previous step",
        "maximum number of steps exceeded",
        "error in routine move_ions",
        "error in routine cell_dynamics",
        "error in routine bfgs",
    )
    return any(pattern in tail for pattern in patterns)


def _qe_output_paths(outdir: Path) -> List[Path]:
    return [
        outdir / filename
        for filename in ("espresso.pwo", "pwscf.out", "espresso.out")
        if (outdir / filename).exists()
    ]


def _qe_output_is_recoverable(text: str) -> bool:
    lower = text.lower()
    fatal_patterns = (
        "error in routine",
        "too many bands are not converged",
        "minimization algorithm failed to find fermi energy",
        "convergence not achieved after",
    )
    return not any(pattern in lower for pattern in fatal_patterns)


def _qe_alat_angstrom(text: str) -> float | None:
    patterns = (
        r"lattice parameter \(alat\)\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"celldm\(1\)\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) * Bohr
    return None


def _qe_block_unit(line: str) -> str:
    match = re.search(r"\(([^)]+)\)", line)
    if not match:
        return "alat"
    return match.group(1).strip().lower()


def _convert_qe_cell_block(values: np.ndarray, unit: str, alat_angstrom: float | None) -> np.ndarray | None:
    if unit == "angstrom":
        return values
    if unit == "bohr":
        return values * Bohr
    if unit == "alat":
        if alat_angstrom is None:
            return None
        return values * alat_angstrom
    return None


def _extract_qe_final_structure(
    text: str,
    *,
    natoms: int,
    template_atoms: Atoms | None,
) -> Atoms | None:
    if template_atoms is None:
        return None

    lines = text.splitlines()
    if len(lines) < natoms + 1:
        return None

    alat_angstrom = _qe_alat_angstrom(text)
    last_cell_idx: int | None = None
    last_positions_idx: int | None = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("CELL_PARAMETERS"):
            last_cell_idx = idx
        elif stripped.startswith("ATOMIC_POSITIONS"):
            last_positions_idx = idx

    if last_positions_idx is None:
        return None

    cell_angstrom = np.array(template_atoms.get_cell(), dtype=float)
    if last_cell_idx is not None and last_cell_idx < last_positions_idx and last_cell_idx + 3 < len(lines):
        try:
            cell_values = np.array(
                [
                    [float(value) for value in lines[last_cell_idx + offset].split()[:3]]
                    for offset in range(1, 4)
                ],
                dtype=float,
            )
        except (ValueError, IndexError):
            cell_values = None
        if cell_values is not None:
            converted_cell = _convert_qe_cell_block(
                cell_values,
                _qe_block_unit(lines[last_cell_idx]),
                alat_angstrom,
            )
            if converted_cell is not None:
                cell_angstrom = converted_cell

    try:
        position_rows = [lines[last_positions_idx + offset].split() for offset in range(1, natoms + 1)]
    except IndexError:
        return None
    if len(position_rows) != natoms or any(len(row) < 4 for row in position_rows):
        return None

    symbols = [row[0] for row in position_rows]
    if len(symbols) != natoms:
        return None

    position_values = np.array([[float(value) for value in row[1:4]] for row in position_rows], dtype=float)
    unit = _qe_block_unit(lines[last_positions_idx])

    relaxed = template_atoms.copy()
    relaxed.set_cell(cell_angstrom, scale_atoms=False)
    relaxed.set_chemical_symbols(symbols)

    if unit == "crystal":
        relaxed.set_scaled_positions(position_values)
    elif unit == "angstrom":
        relaxed.set_positions(position_values)
    elif unit == "bohr":
        relaxed.set_positions(position_values * Bohr)
    elif unit == "alat":
        if alat_angstrom is None:
            return None
        relaxed.set_positions(position_values * alat_angstrom)
    else:
        return None

    return relaxed


def _extract_qe_final_structure_from_output(
    outdir: Path,
    *,
    natoms: int,
    template_atoms: Atoms | None,
) -> Atoms | None:
    for path in _qe_output_paths(outdir):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        relaxed_atoms = _extract_qe_final_structure(
            text,
            natoms=natoms,
            template_atoms=template_atoms,
        )
        if relaxed_atoms is not None:
            return relaxed_atoms
    return None


def _parse_qe_results_from_output(
    outdir: Path,
    natoms: int,
    template_atoms: Atoms | None = None,
) -> dict | None:
    for path in _qe_output_paths(outdir):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip() or not _qe_output_is_recoverable(text):
            continue

        lines = text.splitlines()
        energy_ry: float | None = None
        forces_ry_per_bohr: np.ndarray | None = None
        stress_ry_per_bohr3: np.ndarray | None = None

        for idx, line in enumerate(lines):
            if "!    total energy" in line:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        energy_ry = float(fields[-2])
                    except ValueError:
                        pass
                continue

            if "Forces acting on atoms" in line:
                start = idx + 4 if idx + 2 < len(lines) and not lines[idx + 2].strip() else idx + 2
                block = []
                for force_line in lines[start:start + natoms]:
                    parts = force_line.split()
                    if len(parts) < 3:
                        block = []
                        break
                    try:
                        block.append([float(value) for value in parts[-3:]])
                    except ValueError:
                        block = []
                        break
                if len(block) == natoms:
                    forces_ry_per_bohr = np.array(block, dtype=float)
                continue

            if "total   stress" in line and idx + 3 < len(lines):
                try:
                    sxx, sxy, sxz = lines[idx + 1].split()[:3]
                    _, syy, syz = lines[idx + 2].split()[:3]
                    _, _, szz = lines[idx + 3].split()[:3]
                    stress_ry_per_bohr3 = np.array([sxx, syy, szz, syz, sxz, sxy], dtype=float)
                except (ValueError, IndexError):
                    pass

        if energy_ry is None or forces_ry_per_bohr is None or stress_ry_per_bohr3 is None:
            continue

        forces = forces_ry_per_bohr * (Ry / Bohr)
        stress = stress_ry_per_bohr3 * (-1.0 * Ry / (Bohr ** 3))
        max_force = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
        payload = {
            "total_energy_eV": energy_ry * Ry,
            "stress_voigt": stress.tolist(),
            "forces_eV_A": forces.tolist(),
            "max_force_eV_A": max_force,
            "parser_fallback": str(path),
        }
        relaxed_atoms = _extract_qe_final_structure(text, natoms=natoms, template_atoms=template_atoms)
        if relaxed_atoms is not None:
            payload["final_cell_ang"] = np.array(relaxed_atoms.get_cell(), dtype=float).tolist()
            payload["final_positions_ang"] = np.array(relaxed_atoms.get_positions(), dtype=float).tolist()
            payload["final_symbols"] = relaxed_atoms.get_chemical_symbols()
        return payload

    return None


def _with_conservative_scf_settings(calc_kwargs: dict) -> dict:
    updated = copy.deepcopy(calc_kwargs)
    input_data = updated.setdefault("input_data", {})
    system = input_data.setdefault("system", {})
    electrons = input_data.setdefault("electrons", {})
    ecutwfc = float(system.get("ecutwfc", 0.0) or 0.0)
    ecutrho = float(system.get("ecutrho", 0.0) or 0.0)

    updated["occupations"] = "smearing"
    updated["degauss"] = max(float(updated.get("degauss", 0.0) or 0.0), 0.02)
    updated["smearing"] = "mv"
    system["occupations"] = "smearing"
    if ecutwfc > 0:
        system["ecutrho"] = max(ecutrho, 12.0 * ecutwfc)
    electrons["diagonalization"] = "cg"
    electrons["mixing_mode"] = "local-TF"
    electrons["mixing_beta"] = min(float(electrons.get("mixing_beta", 0.3)), 0.15)
    electrons["electron_maxstep"] = max(int(electrons.get("electron_maxstep", 200)), 400)
    electrons["mixing_ndim"] = min(max(int(electrons.get("mixing_ndim", 8)), 4), 8)
    electrons["startingwfc"] = "atomic+random"
    electrons["startingpot"] = "atomic"
    electrons["diago_full_acc"] = True
    return updated


def _build_c_bands_rescue_profiles(calc_kwargs: dict) -> List[tuple[str, dict]]:
    calc_mode = str(
        calc_kwargs.get("input_data", {}).get("control", {}).get("calculation", "scf")
    ).lower()
    is_relax_mode = calc_mode in {"relax", "vc-relax"}

    first = _with_conservative_scf_settings(calc_kwargs)

    second = copy.deepcopy(first)
    second_input = second.setdefault("input_data", {})
    second_system = second_input.setdefault("system", {})
    second_electrons = second_input.setdefault("electrons", {})
    second_ecutwfc = float(second_system.get("ecutwfc", 0.0) or 0.0)
    second_ecutrho = float(second_system.get("ecutrho", 0.0) or 0.0)
    second["degauss"] = max(float(second.get("degauss", 0.02) or 0.02), 0.03)
    second["smearing"] = "mv"
    second_system["occupations"] = "smearing"
    if second_ecutwfc > 0:
        second_system["ecutrho"] = max(second_ecutrho, 16.0 * second_ecutwfc)
    second_electrons["mixing_mode"] = "plain"
    second_electrons["mixing_beta"] = min(float(second_electrons.get("mixing_beta", 0.15)), 0.08)
    second_electrons["mixing_ndim"] = 4
    second_electrons["electron_maxstep"] = max(
        int(second_electrons.get("electron_maxstep", 400)),
        600 if is_relax_mode else 800,
    )
    second_electrons["startingwfc"] = "atomic+random"
    second_electrons["startingpot"] = "atomic"
    if is_relax_mode:
        second_control = second_input.setdefault("control", {})
        second_control["nstep"] = max(int(second_control.get("nstep", 0) or 0), 300)

    third = copy.deepcopy(second)
    third_input = third.setdefault("input_data", {})
    third_system = third_input.setdefault("system", {})
    third_electrons = third_input.setdefault("electrons", {})
    third["smearing"] = "gaussian"
    third["degauss"] = min(max(float(third.get("degauss", 0.03) or 0.03), 0.015), 0.02)
    third_system["occupations"] = "smearing"
    third_electrons["mixing_mode"] = "plain"
    third_electrons["mixing_beta"] = min(float(third_electrons.get("mixing_beta", 0.08)), 0.03)
    third_electrons["electron_maxstep"] = max(
        int(third_electrons.get("electron_maxstep", 800)),
        800 if is_relax_mode else 1200,
    )
    third_electrons["startingwfc"] = "atomic+random"
    third_electrons["startingpot"] = "atomic"
    third_electrons["mixing_ndim"] = 4
    if is_relax_mode:
        third_control = third_input.setdefault("control", {})
        third_control["nstep"] = max(int(third_control.get("nstep", 0) or 0), 400)

    fourth = copy.deepcopy(third)
    fourth_input = fourth.setdefault("input_data", {})
    fourth_system = fourth_input.setdefault("system", {})
    fourth_electrons = fourth_input.setdefault("electrons", {})
    fourth_ecutwfc = float(fourth_system.get("ecutwfc", 0.0) or 0.0)
    fourth_ecutrho = float(fourth_system.get("ecutrho", 0.0) or 0.0)
    if fourth_ecutwfc > 0:
        fourth_system["ecutwfc"] = max(fourth_ecutwfc, 60.0)
        fourth_system["ecutrho"] = max(fourth_ecutrho, 16.0 * fourth_system["ecutwfc"])
    fourth["smearing"] = "gaussian"
    fourth["degauss"] = min(max(float(fourth.get("degauss", 0.02) or 0.02), 0.01), 0.015)
    fourth_electrons["mixing_mode"] = "plain"
    fourth_electrons["mixing_beta"] = min(float(fourth_electrons.get("mixing_beta", 0.03)), 0.02)
    fourth_electrons["mixing_ndim"] = 4
    fourth_electrons["electron_maxstep"] = max(
        int(fourth_electrons.get("electron_maxstep", 1200)),
        1000 if is_relax_mode else 1600,
    )
    fourth_electrons["startingwfc"] = "random"
    fourth_electrons["startingpot"] = "atomic"
    fourth_electrons["diago_cg_maxiter"] = max(int(fourth_electrons.get("diago_cg_maxiter", 20)), 100)
    if is_relax_mode:
        fourth_control = fourth_input.setdefault("control", {})
        fourth_control["nstep"] = max(int(fourth_control.get("nstep", 0) or 0), 600)

    return [
        ("conservative_cg", first),
        ("plain_low_history", second),
        ("plain_low_smear", third),
        ("random_cg_highcutoff", fourth),
    ]


def _apply_metadata_qe_sections(calc_kwargs: dict, metadata: Dict[str, Any]) -> dict:
    updated = copy.deepcopy(calc_kwargs)
    input_data = updated.setdefault("input_data", {})
    for section_name in ("control", "system", "electrons", "ions", "cell"):
        section_values = metadata.get(section_name)
        if not isinstance(section_values, dict):
            continue
        section = input_data.setdefault(section_name, {})
        section.update(section_values)
    return updated


def _run_single_calculation(atoms, calc_kwargs: dict, outdir: Path, prefix: str) -> dict:
    base_kwargs = copy.deepcopy(calc_kwargs)
    rescue_profiles = _build_c_bands_rescue_profiles(base_kwargs)
    attempts: List[tuple[str, dict]] = [("default", base_kwargs)] + rescue_profiles

    last_exc: subprocess.CalledProcessError | None = None
    for attempt_index, (label, run_kwargs) in enumerate(attempts, start=1):
        shutil.rmtree(outdir, ignore_errors=True)
        outdir.mkdir(parents=True, exist_ok=True)
        attempt_kwargs = copy.deepcopy(run_kwargs)
        # Isolate each QE invocation in its own calculator directory so parallel queue
        # jobs do not race on shared espresso.pwi/espresso.pwo files.
        attempt_kwargs["directory"] = str(outdir)
        attempt_kwargs["outdir"] = str(outdir)
        attempt_kwargs["prefix"] = prefix
        _log(
            f"Launching pw.x for prefix '{prefix}' in {outdir} "
            f"(attempt_profile={label}, attempt_index={attempt_index}/{len(attempts)})"
        )

        try:
            atoms.calc = Espresso(**attempt_kwargs)
            energy = float(atoms.get_potential_energy())
            stress_voigt = atoms.get_stress(voigt=True)
            forces = atoms.get_forces()
            max_force = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
            payload = {
                "total_energy_eV": energy,
                "stress_voigt": stress_voigt.tolist(),
                "forces_eV_A": forces.tolist(),
                "max_force_eV_A": max_force,
            }
            relaxed_atoms = _extract_qe_final_structure_from_output(
                outdir,
                natoms=len(atoms),
                template_atoms=atoms,
            )
            if relaxed_atoms is not None:
                payload["final_cell_ang"] = np.array(relaxed_atoms.get_cell(), dtype=float).tolist()
                payload["final_positions_ang"] = np.array(relaxed_atoms.get_positions(), dtype=float).tolist()
                payload["final_symbols"] = relaxed_atoms.get_chemical_symbols()
            return payload
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if not _is_scf_instability_failure(outdir):
                raise
            if attempt_index >= len(attempts):
                raise
            next_label = attempts[attempt_index][0]
            _log(
                f"Detected QE SCF instability for prefix '{prefix}' under "
                f"profile '{label}'; retrying with '{next_label}'"
            )
        except Exception as exc:  # noqa: BLE001
            calc_mode = str(
                attempt_kwargs.get("input_data", {}).get("control", {}).get("calculation", "scf")
            ).lower()
            parsed = _parse_qe_results_from_output(outdir, len(atoms), template_atoms=atoms)
            if parsed is not None:
                _log(
                    f"Recovered QE results for prefix '{prefix}' from {parsed['parser_fallback']} "
                    f"after ASE parsing failed with {exc.__class__.__name__}: {exc} "
                    f"(calculation={calc_mode})"
                )
                return parsed
            if _is_scf_instability_failure(outdir):
                if attempt_index >= len(attempts):
                    raise
                next_label = attempts[attempt_index][0]
                _log(
                    f"Detected QE SCF instability for prefix '{prefix}' after ASE parsing failed "
                    f"under profile '{label}'; retrying with '{next_label}' "
                    f"(calculation={calc_mode})"
                )
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unexpected QE retry state for prefix '{prefix}'")


def _load_cached_result_if_valid(
    result_path: Path,
    request_id: str,
    *,
    expected_settings_hash: str | None = None,
    expected_metadata_signature: str | None = None,
) -> dict | None:
    if not result_path.exists():
        return None

    try:
        payload = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        quarantine = result_path.with_suffix(
            f"{result_path.suffix}.corrupt-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        )
        try:
            result_path.replace(quarantine)
        except OSError:
            quarantine = None
        _log(
            f"Ignoring invalid cached results for '{request_id}' at {result_path}: "
            f"{exc.__class__.__name__}: {exc}. "
            + (f"Quarantined to {quarantine}." if quarantine is not None else "Leaving file in place.")
        )
        return None

    if not isinstance(payload, dict) or payload.get("status") != "completed":
        _log(
            f"Ignoring cached results for '{request_id}' at {result_path}: "
            "missing completed status."
        )
        return None
    cached_request_id = payload.get("request_id")
    if cached_request_id not in (None, request_id):
        _log(
            f"Ignoring cached results for '{request_id}' at {result_path}: "
            f"request id mismatch ({cached_request_id!r})."
        )
        return None
    if (
        expected_settings_hash is not None
        and payload.get("dft_settings_hash") != expected_settings_hash
    ):
        cached_metadata = payload.get("metadata")
        if (
            expected_metadata_signature is None
            or not isinstance(cached_metadata, dict)
            or _settings_metadata_signature(cached_metadata) != expected_metadata_signature
        ):
            _log(
                f"Ignoring cached results for '{request_id}' at {result_path}: "
                "settings hash mismatch."
            )
            return None
    return payload


def _load_reusable_strain_results(
    *,
    strain_results_path: Path,
    strain_meta_path: Path,
    request_id: str,
    expected_settings_hash: str,
    expected_structure_signature: str,
    expected_stress_analysis: Dict[str, Any],
) -> list[dict[str, Any]] | None:
    if not strain_results_path.exists() or not strain_meta_path.exists():
        return None

    try:
        payload = json.loads(strain_results_path.read_text())
        meta = json.loads(strain_meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _log(
            f"Ignoring cached strain results for '{request_id}': "
            f"{exc.__class__.__name__}: {exc}"
        )
        return None

    if not isinstance(payload, list):
        _log(f"Ignoring cached strain results for '{request_id}': payload is not a list.")
        return None
    if not isinstance(meta, dict):
        _log(f"Ignoring cached strain results for '{request_id}': metadata is not a dict.")
        return None
    if meta.get("request_id") != request_id:
        _log(f"Ignoring cached strain results for '{request_id}': request id mismatch.")
        return None
    if meta.get("dft_settings_hash") != expected_settings_hash:
        _log(f"Ignoring cached strain results for '{request_id}': settings hash mismatch.")
        return None
    if meta.get("structure_signature") != expected_structure_signature:
        _log(f"Ignoring cached strain results for '{request_id}': relaxed structure changed.")
        return None
    if meta.get("stress_analysis") != expected_stress_analysis:
        _log(f"Ignoring cached strain results for '{request_id}': strain settings changed.")
        return None
    return payload


def run_workflow(request_id: str, resume: bool = False) -> dict:
    input_path = INPUT_DIR / request_id
    output_path = OUTPUT_DIR / request_id
    result_cache = output_path / "results.json"
    settings_hash = _compute_settings_hash(input_path)
    current_metadata = json.loads((input_path / "metadata.json").read_text(encoding="utf-8"))
    metadata_signature = _settings_metadata_signature(current_metadata)

    if resume:
        cached_result = _load_cached_result_if_valid(
            result_cache,
            request_id,
            expected_settings_hash=settings_hash,
            expected_metadata_signature=metadata_signature,
        )
        if cached_result is not None:
            _log(f"Using cached results for '{request_id}' (resume enabled)")
            return cached_result

    if resume and output_path.exists():
        _log(f"Resuming workflow for '{request_id}' using existing output at {output_path}")
    else:
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    _log(f"Starting workflow for request '{request_id}'")
    metadata = current_metadata
    evidence_tier = _infer_evidence_tier(request_id, metadata)
    overrides = metadata.get("qe_overrides") or {}
    structure_file = input_path / "structure.cif"
    if not structure_file.exists():
        raise FileNotFoundError(
            f"Structure file {structure_file} not found. Provide a CIF in the handoff package."
        )

    atoms = read(structure_file)
    unique_elements = np.array(list(dict.fromkeys(atoms.get_chemical_symbols())))
    _log(f"Selecting pseudopotentials for elements: {', '.join(unique_elements)}")
    pseudopotentials = _select_pseudopotentials(unique_elements)
    recommended_ecutwfc, recommended_ecutrho = _recommended_cutoffs_ry(pseudopotentials)

    encut_eV = metadata.get("encut", 520)
    ecutwfc_override = overrides.get("ecutwfc_ry")
    if ecutwfc_override is not None:
        ecutwfc = float(ecutwfc_override)
        encut_eV = ecutwfc / EV_TO_RY
    else:
        ecutwfc = encut_eV * EV_TO_RY
    if recommended_ecutwfc is not None and ecutwfc < recommended_ecutwfc:
        _log(
            f"Raising ecutwfc from {ecutwfc:.2f} Ry to pseudopotential recommendation "
            f"{recommended_ecutwfc:.2f} Ry"
        )
        ecutwfc = recommended_ecutwfc

    ecutrho_override = overrides.get("ecutrho_ry")
    if ecutrho_override is not None:
        ecutrho = float(ecutrho_override)
    else:
        ecutrho = metadata.get("ecutrho", 8 * ecutwfc)
    if recommended_ecutrho is not None and ecutrho < recommended_ecutrho:
        _log(
            f"Raising ecutrho from {ecutrho:.2f} Ry to pseudopotential recommendation "
            f"{recommended_ecutrho:.2f} Ry"
        )
        ecutrho = recommended_ecutrho

    smearing_cfg = metadata.get("smearing", {})
    sigma_eV = smearing_cfg.get("sigma", 0.2)
    degauss_ry = smearing_cfg.get("degauss", sigma_eV * EV_TO_RY)
    mixing_beta = metadata.get("electrons", {}).get("mixing_beta", 0.3)
    conv_thr = metadata.get("electrons", {}).get("conv_thr", 1e-6)

    profile = EspressoProfile(command=PW_COMMAND, pseudo_dir=str(PSEUDO_DIR))

    k_grid = overrides.get("k_grid", metadata.get("kpoint_grid", [4, 4, 4]))

    calc_kwargs = {
        "kpts": tuple(k_grid),
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
    calc_kwargs = _apply_metadata_qe_sections(calc_kwargs, metadata)

    relaxation_cfg = metadata.get("relaxation") or {}
    ions_section: Dict[str, float | str] = {}
    cell_section: Dict[str, float | str] = {}
    if relaxation_cfg:
        ion_dynamics = relaxation_cfg.get("ion_dynamics")
        if ion_dynamics:
            ions_section["ion_dynamics"] = ion_dynamics
        cell_dynamics = relaxation_cfg.get("cell_dynamics")
        if cell_dynamics:
            cell_section["cell_dynamics"] = cell_dynamics
        press_thr = relaxation_cfg.get("pressure_threshold_kbar")
        if press_thr is not None:
            cell_section["press_conv_thr"] = float(press_thr)
        max_steps = relaxation_cfg.get("max_steps")
        if max_steps is not None:
            calc_kwargs["input_data"]["control"]["nstep"] = int(max_steps)

    if ions_section:
        calc_kwargs["input_data"]["ions"] = ions_section
    if cell_section:
        calc_kwargs["input_data"]["cell"] = cell_section

    _log("Running base calculation")
    base_result = _run_single_calculation(
        atoms,
        calc_kwargs,
        output_path / "qe_tmp" / "base",
        request_id.lower(),
    )
    _log("Base calculation completed")

    final_cell = base_result.get("final_cell_ang")
    final_positions = base_result.get("final_positions_ang")
    final_symbols = base_result.get("final_symbols")
    if final_cell is not None and final_positions is not None:
        atoms.set_cell(np.array(final_cell, dtype=float), scale_atoms=False)
        atoms.set_positions(np.array(final_positions, dtype=float))
        if final_symbols:
            atoms.set_chemical_symbols([str(symbol) for symbol in final_symbols])

    volume = atoms.get_volume()
    mass = atoms.get_masses().sum()
    density = (mass * AMU_TO_G) / volume
    forces = np.array(base_result["forces_eV_A"], dtype=float)
    max_force_val = float(base_result["max_force_eV_A"])
    stress_voigt = np.array(base_result["stress_voigt"], dtype=float)
    stress_voigt_list = stress_voigt.tolist()
    stress_voigt_kbar = (stress_voigt * EV_A3_TO_KBAR).tolist()
    pressure_kbar = float(-np.mean(stress_voigt[:3]) * EV_A3_TO_KBAR)

    final_structure_path = output_path / "relaxed_structure.cif"
    write(final_structure_path, atoms, format="cif")

    results = {
        "schema_version": "2.0.0",
        "request_id": request_id,
        "status": "completed",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "engine": {
            "name": "quantum_espresso",
            "command": PW_COMMAND,
            "mode": metadata.get("calculation", "scf"),
        },
        "dft_settings_hash": settings_hash,
        "evidence_tier": evidence_tier,
        "total_energy_eV": base_result["total_energy_eV"],
        "formation_energy_eV": base_result["total_energy_eV"] / len(atoms),
        "max_force_eV_A": max_force_val,
        "forces_eV_A": forces.tolist(),
        "properties": {
            "exp_density_g_cm3": density,
        },
        "uncertainty": {
            "formation_energy_eV": 0.0,
            "exp_density_g_cm3": 0.02,
        },
        "metadata": metadata,
        "stress": {
            "voigt_eVA3": stress_voigt_list,
            "voigt_kbar": stress_voigt_kbar,
            "pressure_kbar": pressure_kbar,
        },
        "final_structure": {
            "lattice_vectors_ang": atoms.get_cell().tolist(),
            "atomic_symbols": atoms.get_chemical_symbols(),
            "positions_cartesian_ang": atoms.get_positions().tolist(),
            "positions_fractional": atoms.get_scaled_positions().tolist(),
            "volume_ang3": volume,
        },
        "artefacts": {
            "relaxed_structure_cif": str(final_structure_path.relative_to(output_path)),
        },
        "strain_results": [],
    }

    stress_cfg = metadata.get("stress_analysis")
    if stress_cfg:
        _log("Stress analysis enabled; preparing strain calculations")
        strain_results_path = output_path / "strain_results.json"
        strain_meta_path = output_path / "strain_results.meta.json"
        structure_signature = _compute_structure_signature(atoms)
        if resume:
            reusable_strain_results = _load_reusable_strain_results(
                strain_results_path=strain_results_path,
                strain_meta_path=strain_meta_path,
                request_id=request_id,
                expected_settings_hash=settings_hash,
                expected_structure_signature=structure_signature,
                expected_stress_analysis=stress_cfg,
            )
            if reusable_strain_results is not None:
                _log("Using existing strain results and skipping strain reruns.")
                results["strain_results"] = reusable_strain_results
            else:
                _log("Cached strain results are stale or incomplete; rerunning strain calculations.")
        if not results["strain_results"]:
            amplitude = float(stress_cfg.get("strain_amplitude", 0.003))
            max_directions = int(stress_cfg.get("strain_directions", 6))
            strain_matrices = _build_strain_matrices(amplitude)[:max_directions]

            relaxed_atoms = atoms.copy()

            for idx, strain_matrix in enumerate(strain_matrices):
                for sign, label in [(1.0, "positive"), (-1.0, "negative")]:
                    signed_matrix = strain_matrix * sign
                    strained_atoms = _apply_strain(relaxed_atoms, signed_matrix)

                    strain_dir = output_path / f"strain_{idx}_{label}"
                    _log(f"Running strain index {idx} ({label})")
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
            _log("Completed stress analysis calculations")
            strain_results_path.write_text(
                json.dumps(results["strain_results"], indent=2), encoding="utf-8"
            )
            strain_meta_path.write_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "dft_settings_hash": settings_hash,
                        "structure_signature": structure_signature,
                        "stress_analysis": stress_cfg,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    else:
        _log("No stress analysis configured; skipping strain calculations")

    (output_path / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_path / "log.txt").write_text("Quantum ESPRESSO run completed.\n", encoding="utf-8")
    _log(f"Workflow for '{request_id}' finished successfully")
    return results


def _log_queue_to_mlflow(
    summary: Dict[str, Any],
    job_records: Sequence[Dict[str, Any]],
    tracking_uri: Optional[str],
    experiment: str,
    run_name: Optional[str],
    jobs_log_path: Path,
) -> None:
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment)
    effective_run_name = run_name or summary["run_id"]
    with mlflow.start_run(run_name=effective_run_name):
        mlflow.set_tags({"task": "T5R.2", "milestone": "M5-real"})
        mlflow.log_metric("aloa.real_dft_iterations", summary["total_jobs"])
        mlflow.log_metric("mdia.dft_jobs_completed", summary["completed_jobs"])
        mlflow.log_metric("mdia.dft_jobs_failed", summary["failed_jobs"])
        latency_stats = summary.get("latency", {})
        for component in ("avg", "min", "max", "p95"):
            value = latency_stats.get(component)
            if value is not None:
                mlflow.log_metric(f"dft_queue.latency_{component}_s", float(value))
        for idx, record in enumerate(job_records):
            latency_value = record.get("latency_s")
            if latency_value is not None:
                mlflow.log_metric("dft_queue.job_latency_s", float(latency_value), step=idx)
        mlflow.log_dict(summary, "queue_summary.json")
        mlflow.log_artifact(str(jobs_log_path), artifact_path="queue_monitoring")


def run_queue(
    request_ids: Sequence[str],
    *,
    max_workers: int = 1,
    max_retries: int = 1,
    tracking_uri: Optional[str] = None,
    experiment: str = DEFAULT_QUEUE_EXPERIMENT,
    run_name: Optional[str] = None,
    monitor_dir: Optional[Path] = None,
    resume: bool = False,
    max_wall_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    if not request_ids:
        raise ValueError("No request IDs supplied for queue execution.")

    QUEUE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    queue_run_id = f"dft-queue-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    monitor_root = monitor_dir or (QUEUE_RUN_DIR / queue_run_id)
    monitor_root.mkdir(parents=True, exist_ok=True)
    jobs_log_path = monitor_root / "jobs.jsonl"
    summary_path = monitor_root / "summary.json"

    _log(
        f"Launching queue run '{queue_run_id}' for {len(request_ids)} job(s) "
        f"with max_workers={max_workers}, max_retries={max_retries}"
    )

    queue_started_at = datetime.utcnow().isoformat() + "Z"
    job_records: List[Dict[str, Any]] = []

    def _worker(request_id: str) -> Dict[str, Any]:
        def _snapshot(result_dict: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "total_energy_eV": result_dict.get("total_energy_eV"),
                "formation_energy_eV": result_dict.get("formation_energy_eV"),
                "max_force_eV_A": result_dict.get("max_force_eV_A"),
                "strain_evaluations": len(result_dict.get("strain_results", [])),
            }

        attempt = 0
        errors: List[Dict[str, Any]] = []
        job_start_wall = time.monotonic()
        job_started_at = datetime.utcnow().isoformat() + "Z"
        cached_result_path = OUTPUT_DIR / request_id / "results.json"
        expected_settings_hash = _compute_settings_hash(INPUT_DIR / request_id)
        current_metadata = json.loads((INPUT_DIR / request_id / "metadata.json").read_text(encoding="utf-8"))
        expected_metadata_signature = _settings_metadata_signature(current_metadata)

        if resume:
            cached_result = _load_cached_result_if_valid(
                cached_result_path,
                request_id,
                expected_settings_hash=expected_settings_hash,
                expected_metadata_signature=expected_metadata_signature,
            )
            if cached_result is not None:
                _log(f"Queue job '{request_id}': using cached results (resume enabled)")
                completed_at = datetime.utcnow().isoformat() + "Z"
                return {
                    "request_id": request_id,
                    "status": "completed",
                    "attempts": 0,
                    "started_at": job_started_at,
                    "completed_at": completed_at,
                    "latency_s": 0.0,
                    "latest_attempt_latency_s": 0.0,
                    "errors": errors,
                    "output_dir": str((OUTPUT_DIR / request_id).resolve()),
                    "result_snapshot": _snapshot(cached_result),
                }

        while True:
            if max_wall_seconds is not None and (time.monotonic() - job_start_wall) > max_wall_seconds:
                _log(
                    f"Queue job '{request_id}' exceeded wall time limit of {max_wall_seconds}s; marking as failed"
                )
                total_latency = time.monotonic() - job_start_wall
                timeout_record = {
                    "attempt": attempt,
                    "error": "TimeoutError",
                    "traceback": "Exceeded max wall time",
                    "failed_at": datetime.utcnow().isoformat() + "Z",
                    "attempt_latency_s": total_latency,
                }
                errors.append(timeout_record)
                return {
                    "request_id": request_id,
                    "status": "failed",
                    "attempts": attempt,
                    "started_at": job_started_at,
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                    "latency_s": total_latency,
                    "errors": errors,
                }

            attempt += 1
            _log(f"Queue job '{request_id}': starting attempt {attempt}")
            attempt_start = time.monotonic()
            try:
                result = run_workflow(request_id, resume=resume)
                total_latency = time.monotonic() - job_start_wall
                attempt_latency = time.monotonic() - attempt_start
                completed_at = datetime.utcnow().isoformat() + "Z"
                _log(
                    f"Queue job '{request_id}' completed in {total_latency:.2f}s "
                    f"after {attempt} attempt(s)"
                )
                return {
                    "request_id": request_id,
                    "status": "completed",
                    "attempts": attempt,
                    "started_at": job_started_at,
                    "completed_at": completed_at,
                    "latency_s": total_latency,
                    "latest_attempt_latency_s": attempt_latency,
                    "errors": errors,
                    "output_dir": str((OUTPUT_DIR / request_id).resolve()),
                    "result_snapshot": _snapshot(result),
                }
            except Exception as exc:  # noqa: BLE001
                attempt_latency = time.monotonic() - attempt_start
                error_record = {
                    "attempt": attempt,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at": datetime.utcnow().isoformat() + "Z",
                    "attempt_latency_s": attempt_latency,
                }
                _log(
                    f"Queue job '{request_id}' failed on attempt {attempt}: {exc!r} "
                    f"(retrying: {attempt <= max_retries})"
                )
                errors.append(error_record)
                if attempt > max_retries:
                    total_latency = time.monotonic() - job_start_wall
                    return {
                        "request_id": request_id,
                        "status": "failed",
                        "attempts": attempt,
                        "started_at": job_started_at,
                        "completed_at": datetime.utcnow().isoformat() + "Z",
                        "latency_s": total_latency,
                        "errors": errors,
                    }
                backoff = min(10.0, 1.5 * attempt)
                time.sleep(backoff)

    with jobs_log_path.open("w", encoding="utf-8") as log_handle:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_worker, rid): rid for rid in request_ids}
            for index, future in enumerate(as_completed(future_map)):
                record = future.result()
                record["queue_index"] = index
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
                job_records.append(record)

    completed_jobs = [record for record in job_records if record["status"] == "completed"]
    failed_jobs = [record for record in job_records if record["status"] == "failed"]
    latencies = [record["latency_s"] for record in completed_jobs if record.get("latency_s") is not None]
    latency_stats = {
        "avg": float(np.mean(latencies)) if latencies else None,
        "min": float(np.min(latencies)) if latencies else None,
        "max": float(np.max(latencies)) if latencies else None,
        "p95": float(np.percentile(latencies, 95)) if len(latencies) >= 2 else None,
    }

    queue_completed_at = datetime.utcnow().isoformat() + "Z"
    summary = {
        "run_id": queue_run_id,
        "started_at": queue_started_at,
        "completed_at": queue_completed_at,
        "total_jobs": len(job_records),
        "completed_jobs": len(completed_jobs),
        "failed_jobs": len(failed_jobs),
        "latency": latency_stats,
        "jobs": job_records,
        "artefacts": {
            "jobs_log": str(jobs_log_path.resolve()),
            "summary_path": str(summary_path.resolve()),
        },
    }

    try:
        _log_queue_to_mlflow(summary, job_records, tracking_uri, experiment, run_name, jobs_log_path)
        summary["mlflow_run"] = True
    except Exception as exc:  # noqa: BLE001
        _log(f"Failed to log queue run to MLflow: {exc!r}")
        summary["mlflow_run"] = False

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(
        f"Queue run '{queue_run_id}' finished: "
        f"{len(completed_jobs)} completed / {len(job_records)} total."
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quantum ESPRESSO DFT workflow or queue.")
    parser.add_argument("request_id", nargs="?", help="Single DFT handoff request identifier.")
    parser.add_argument(
        "--queue",
        nargs="+",
        help="Run an asynchronous queue for the provided request identifiers.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of concurrent DFT jobs when using --queue.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum retry attempts per job when using --queue.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help=f"Override MLflow tracking URI for queue runs (default: {DEFAULT_TRACKING_URI}).",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_QUEUE_EXPERIMENT,
        help="MLflow experiment name for queue runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional MLflow run name override for queue runs.",
    )
    parser.add_argument(
        "--monitor-dir",
        type=Path,
        default=None,
        help="Directory to store queue monitoring artefacts (defaults to data/dft_workflow/queue_runs/<run_id>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.queue and args.request_id:
        raise SystemExit("Provide either a single request_id or --queue, not both.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.queue:
        summary = run_queue(
            args.queue,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            tracking_uri=args.tracking_uri,
            experiment=args.experiment,
            run_name=args.run_name,
            monitor_dir=args.monitor_dir,
        )
        print(json.dumps(summary, indent=2))
        return

    request_id = args.request_id or "QAL-0001"
    results = run_workflow(request_id)
    REPORT_PATH.write_text(json.dumps({"last_run": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
