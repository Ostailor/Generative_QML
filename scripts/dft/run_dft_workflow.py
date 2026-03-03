#!/usr/bin/env python3
"""Automated DFT workflow using Quantum ESPRESSO (ASE calculator) with queue orchestration."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mlflow
import numpy as np

try:
    from ase.io import read, write
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
PSEUDO_DIR = Path(next((val for val in _PSEUDO_ENV_CANDIDATES if val), "qe_pseudo"))
PW_COMMAND = os.environ.get("QE_PW_COMMAND", "pw.x")

AMU_TO_G = 1.66053906660
EV_TO_RY = 0.0734986176
ANGSTROM_TO_BOHR = 1.8897259886
EV_A3_TO_KBAR = 1602.1766208


def _log(message: str) -> None:
    print(f"[run_dft_workflow] {message}", flush=True)


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
        sha.update(path.read_bytes())
    return sha.hexdigest()


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
    # Isolate each QE invocation in its own calculator directory so parallel queue
    # jobs do not race on shared espresso.pwi/espresso.pwo files.
    run_kwargs["directory"] = str(outdir)
    run_kwargs["outdir"] = str(outdir)
    run_kwargs["prefix"] = prefix
    _log(f"Launching pw.x for prefix '{prefix}' in {outdir}")

    atoms.calc = Espresso(**run_kwargs)
    energy = float(atoms.get_potential_energy())
    stress_voigt = atoms.get_stress(voigt=True)
    forces = atoms.get_forces()
    max_force = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0

    return {
        "total_energy_eV": energy,
        "stress_voigt": stress_voigt.tolist(),
        "max_force_eV_A": max_force,
    }


def run_workflow(request_id: str, resume: bool = False) -> dict:
    input_path = INPUT_DIR / request_id
    output_path = OUTPUT_DIR / request_id
    result_cache = output_path / "results.json"

    if resume and result_cache.exists():
        _log(f"Using cached results for '{request_id}' (resume enabled)")
        return json.loads(result_cache.read_text())

    if resume and output_path.exists():
        _log(f"Resuming workflow for '{request_id}' using existing output at {output_path}")
    else:
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    _log(f"Starting workflow for request '{request_id}'")
    metadata = json.loads((input_path / "metadata.json").read_text())
    settings_hash = _compute_settings_hash(input_path)
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

    encut_eV = metadata.get("encut", 520)
    ecutwfc_override = overrides.get("ecutwfc_ry")
    if ecutwfc_override is not None:
        ecutwfc = float(ecutwfc_override)
        encut_eV = ecutwfc / EV_TO_RY
    else:
        ecutwfc = encut_eV * EV_TO_RY

    ecutrho_override = overrides.get("ecutrho_ry")
    if ecutrho_override is not None:
        ecutrho = float(ecutrho_override)
    else:
        ecutrho = metadata.get("ecutrho", 8 * ecutwfc)

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

    volume = atoms.get_volume()
    mass = atoms.get_masses().sum()
    density = (mass * AMU_TO_G) / volume
    forces = atoms.get_forces()
    max_force_val = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
    stress_voigt = atoms.get_stress(voigt=True)
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
        if resume and (output_path / 'strain_results.json').exists():
            _log('Using existing strain results and skipping strain reruns.')
            results["strain_results"] = json.loads((output_path / 'strain_results.json').read_text())
            return results
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
        (output_path / "strain_results.json").write_text(
            json.dumps(results["strain_results"], indent=2), encoding="utf-8"
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

        if resume and cached_result_path.exists():
            _log(f"Queue job '{request_id}': using cached results (resume enabled)")
            cached_result = json.loads(cached_result_path.read_text())
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
