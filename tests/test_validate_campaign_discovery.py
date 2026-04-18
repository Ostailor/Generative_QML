from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.dft import run_dft_workflow as workflow  # noqa: E402
from scripts.dft.validate_production_outputs import (  # noqa: E402
    _aggregate_by_formula,
    _discover_requests_from_campaign,
)


class CampaignDiscoveryTests(unittest.TestCase):
    def test_discover_from_candidate_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "camp-001"
            campaign.mkdir(parents=True)
            csv_path = campaign / "candidate_library.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["request_id"])
                writer.writeheader()
                writer.writerow({"request_id": "REQ-001"})
                writer.writerow({"request_id": "REQ-002"})

            requests, meta = _discover_requests_from_campaign("camp-001", root)
            self.assertEqual(requests, ["REQ-001", "REQ-002"])
            self.assertIn("request_source", meta)

    def test_discover_from_iteration_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "camp-002"
            campaign.mkdir(parents=True)
            manifest = campaign / "iteration_01_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["request_id", "candidate_id"])
                writer.writeheader()
                writer.writerow({"request_id": "REQ-A", "candidate_id": "C-1"})
                writer.writerow({"request_id": "REQ-B", "candidate_id": "C-2"})

            requests, _ = _discover_requests_from_campaign("camp-002", root)
            self.assertEqual(requests, ["REQ-A", "REQ-B"])

    def test_aggregate_by_formula_handles_duplicate_reference_rows(self) -> None:
        reference_df = pd.DataFrame(
            {"calc_density_g_cm3": [7.5, 8.5]},
            index=["Al0.5Co0.5", "Al0.5Co0.5"],
        )
        fake_result = {
            "metadata": {"composition": "Al0.5Co0.5"},
            "_result_path": "/tmp/results.json",
            "formation_energy_eV": -1.25,
            "properties": {"exp_density_g_cm3": 8.0},
        }

        with patch(
            "scripts.dft.validate_production_outputs._load_result",
            return_value=fake_result,
        ):
            grouped = _aggregate_by_formula(["REQ-1"], reference_df)

        stats = grouped["Al0.5Co0.5"]["properties"]["density_g_cm3"]
        self.assertEqual(stats.reference, 8.0)

    def test_select_pseudopotentials_prefers_exact_symbol_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pseudo_dir = Path(tmp)
            for filename in ("Dy.upf", "Y.upf", "Yb.upf"):
                (pseudo_dir / filename).write_text("", encoding="utf-8")

            with patch.object(workflow, "PSEUDO_DIR", pseudo_dir):
                mapping = workflow._select_pseudopotentials(np.array(["Y"]))

        self.assertEqual(mapping["Y"], "Y.upf")

    def test_recommended_cutoffs_use_upf_header_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pseudo_dir = Path(tmp)
            (pseudo_dir / "A.upf").write_text(
                '<UPF version="2.0.1"><PP_HEADER wfc_cutoff="50.0" rho_cutoff="400.0" /></UPF>',
                encoding="utf-8",
            )
            (pseudo_dir / "B.upf").write_text(
                '<UPF version="2.0.1"><PP_HEADER wfc_cutoff="65.0" rho_cutoff="520.0" /></UPF>',
                encoding="utf-8",
            )

            with patch.object(workflow, "PSEUDO_DIR", pseudo_dir):
                wfc, rho = workflow._recommended_cutoffs_ry({"A": "A.upf", "B": "B.upf"})

        self.assertEqual(wfc, 65.0)
        self.assertEqual(rho, 520.0)

    def test_conservative_scf_settings_raise_ecutrho_and_harden_mixing(self) -> None:
        updated = workflow._with_conservative_scf_settings(
            {
                "degauss": 0.01,
                "input_data": {
                    "system": {"ecutwfc": 50.0, "ecutrho": 400.0},
                    "electrons": {"mixing_beta": 0.3, "electron_maxstep": 200},
                },
            }
        )

        self.assertEqual(updated["smearing"], "mv")
        self.assertGreaterEqual(updated["degauss"], 0.02)
        self.assertEqual(updated["input_data"]["system"]["ecutrho"], 600.0)
        self.assertEqual(updated["input_data"]["electrons"]["diagonalization"], "cg")
        self.assertEqual(updated["input_data"]["electrons"]["startingwfc"], "atomic+random")
        self.assertEqual(updated["input_data"]["electrons"]["startingpot"], "atomic")
        self.assertEqual(updated["input_data"]["electrons"]["mixing_beta"], 0.15)
        self.assertEqual(updated["input_data"]["electrons"]["mixing_ndim"], 8)

    def test_scf_instability_detection_includes_broyden_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            (outdir / "espresso.pwo").write_text(
                "Error in routine broyden (7):\n"
                "factorization\n"
                "Minimization algorithm failed to find Fermi energy\n",
                encoding="utf-8",
            )

            self.assertTrue(workflow._is_scf_instability_failure(outdir))

    def test_scf_instability_detection_includes_nonconverged_scf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            (outdir / "espresso.pwo").write_text(
                "End of self-consistent calculation\n"
                "convergence NOT achieved after 400 iterations: stopping\n",
                encoding="utf-8",
            )

            self.assertTrue(workflow._is_scf_instability_failure(outdir))

    def test_scf_instability_detection_includes_vc_relax_max_step_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            (outdir / "espresso.pwo").write_text(
                "number of bfgs steps exceeded\n"
                "maximum number of steps exceeded\n",
                encoding="utf-8",
            )

            self.assertTrue(workflow._is_scf_instability_failure(outdir))

    def test_broyden_rescue_profiles_reduce_mixing_history(self) -> None:
        profiles = workflow._build_c_bands_rescue_profiles(
            {
                "degauss": 0.02,
                "input_data": {
                    "system": {"ecutwfc": 50.0, "ecutrho": 400.0},
                    "electrons": {"mixing_beta": 0.3, "electron_maxstep": 200, "mixing_ndim": 8},
                },
            }
        )

        self.assertEqual(profiles[1][0], "plain_low_history")
        self.assertEqual(profiles[1][1]["input_data"]["electrons"]["mixing_mode"], "plain")
        self.assertEqual(profiles[1][1]["input_data"]["electrons"]["mixing_ndim"], 4)
        self.assertEqual(profiles[2][0], "plain_low_smear")
        self.assertEqual(profiles[2][1]["smearing"], "gaussian")
        self.assertLessEqual(profiles[2][1]["degauss"], 0.02)
        self.assertEqual(profiles[3][0], "random_cg_highcutoff")
        self.assertEqual(profiles[3][1]["input_data"]["electrons"]["startingwfc"], "random")
        self.assertEqual(profiles[3][1]["input_data"]["electrons"]["diago_cg_maxiter"], 100)
        self.assertGreaterEqual(profiles[3][1]["input_data"]["system"]["ecutwfc"], 60.0)

    def test_relax_mode_rescue_profiles_cap_electron_steps(self) -> None:
        profiles = workflow._build_c_bands_rescue_profiles(
            {
                "input_data": {
                    "control": {"calculation": "vc-relax"},
                    "system": {"ecutwfc": 60.0, "ecutrho": 720.0},
                    "electrons": {"mixing_beta": 0.3, "electron_maxstep": 400, "mixing_ndim": 8},
                },
            }
        )

        self.assertEqual(profiles[1][1]["input_data"]["electrons"]["electron_maxstep"], 600)
        self.assertEqual(profiles[2][1]["input_data"]["electrons"]["electron_maxstep"], 800)
        self.assertEqual(profiles[3][1]["input_data"]["electrons"]["electron_maxstep"], 1000)
        self.assertEqual(profiles[1][1]["input_data"]["control"]["nstep"], 300)
        self.assertEqual(profiles[2][1]["input_data"]["control"]["nstep"], 400)
        self.assertEqual(profiles[3][1]["input_data"]["control"]["nstep"], 600)

    def test_invalid_cached_result_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "results.json"
            result_path.write_text("", encoding="utf-8")

            payload = workflow._load_cached_result_if_valid(result_path, "REQ-001")

            self.assertIsNone(payload)
            self.assertFalse(result_path.exists())
            quarantined = list(Path(tmp).glob("results.json.corrupt-*"))
            self.assertEqual(len(quarantined), 1)

    def test_run_single_calculation_retries_parser_failure_with_nonconverged_output(self) -> None:
        class FakeCalc:
            def __init__(self, **kwargs):
                self.directory = kwargs["directory"]

        class FakeAtoms:
            def __init__(self):
                self.calc = None
                self.calls = 0

            def __len__(self):
                return 1

            def get_potential_energy(self):
                self.calls += 1
                outdir = Path(self.calc.directory)
                if self.calls == 1:
                    (outdir / "espresso.pwo").write_text(
                        "End of self-consistent calculation\n"
                        "convergence NOT achieved after 400 iterations: stopping\n",
                        encoding="utf-8",
                    )
                    raise AssertionError(((2, 0), 36))
                return -1.0

            def get_stress(self, voigt=True):
                return np.zeros(6)

            def get_forces(self):
                return np.zeros((1, 3))

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "qe"
            atoms = FakeAtoms()
            calc_kwargs = {
                "input_data": {"control": {"calculation": "scf"}},
            }
            with patch.object(workflow, "Espresso", FakeCalc), patch.object(
                workflow, "_parse_qe_results_from_output", return_value=None
            ):
                result = workflow._run_single_calculation(atoms, calc_kwargs, outdir, "fake")

        self.assertEqual(atoms.calls, 2)
        self.assertEqual(result["total_energy_eV"], -1.0)

    def test_run_single_calculation_recovers_vc_relax_parser_failure_from_qe_output(self) -> None:
        class FakeCalc:
            def __init__(self, **kwargs):
                self.directory = kwargs["directory"]

        class FakeAtoms:
            def __init__(self):
                self.calc = None

            def __len__(self):
                return 1

            def get_potential_energy(self):
                outdir = Path(self.calc.directory)
                (outdir / "espresso.pwo").write_text(
                    "!    total energy              =   -10.00000000 Ry\n\n"
                    "     Forces acting on atoms (cartesian axes, Ry/au):\n\n"
                    "     atom    1 type  1   force =     0.00000000    0.00000000    0.00100000\n\n"
                    "     Computing stress (Cartesian axis) and pressure\n\n"
                    "          total   stress  (Ry/bohr**3)                   (kbar)     P=    -1.00\n"
                    "  -0.00100000   0.00000000   0.00000000        -1.00       0.00       0.00\n"
                    "   0.00000000  -0.00100000   0.00000000         0.00      -1.00       0.00\n"
                    "   0.00000000   0.00000000  -0.00100000         0.00       0.00      -1.00\n\n"
                    "=------------------------------------------------------------------------------=\n"
                    "   JOB DONE.\n"
                    "=------------------------------------------------------------------------------=\n",
                    encoding="utf-8",
                )
                raise AssertionError(((2, 0), 63))

            def get_stress(self, voigt=True):
                raise AssertionError('should use parser fallback')

            def get_forces(self):
                raise AssertionError('should use parser fallback')

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "qe"
            atoms = FakeAtoms()
            calc_kwargs = {
                "input_data": {"control": {"calculation": "vc-relax"}},
            }
            with patch.object(workflow, "Espresso", FakeCalc):
                result = workflow._run_single_calculation(atoms, calc_kwargs, outdir, "fake-vc")

        self.assertIn("total_energy_eV", result)
        self.assertIn("forces_eV_A", result)
        self.assertIn("stress_voigt", result)

    def test_parse_qe_results_from_output_rejects_nonconverged_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            (outdir / "espresso.pwo").write_text(
                "!    total energy              =   -10.00000000 Ry\n\n"
                "     Forces acting on atoms (cartesian axes, Ry/au):\n\n"
                "     atom    1 type  1   force =     0.00000000    0.00000000    0.00100000\n\n"
                "     Computing stress (Cartesian axis) and pressure\n\n"
                "          total   stress  (Ry/bohr**3)                   (kbar)     P=    -1.00\n"
                "  -0.00100000   0.00000000   0.00000000        -1.00       0.00       0.00\n"
                "   0.00000000  -0.00100000   0.00000000         0.00      -1.00       0.00\n"
                "   0.00000000   0.00000000  -0.00100000         0.00       0.00      -1.00\n\n"
                "     End of self-consistent calculation\n\n"
                "     convergence NOT achieved after 400 iterations: stopping\n\n"
                "=------------------------------------------------------------------------------=\n"
                "   JOB DONE.\n"
                "=------------------------------------------------------------------------------=\n",
                encoding="utf-8",
            )

            payload = workflow._parse_qe_results_from_output(outdir, natoms=1)

            self.assertIsNone(payload)

    def test_parse_qe_results_from_output_reads_energy_forces_and_stress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            (outdir / "espresso.pwo").write_text(
                """
!    total energy              =   -1173.47315217 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.00000000    0.10000000
     atom    2 type  2   force =     0.00000000    0.20000000    0.00000000

     Total force =     0.000000     Total SCF correction =     0.000000


     Computing stress (Cartesian axis) and pressure


          total   stress  (Ry/bohr**3)                   (kbar)     P=    -2746.83
  -0.01878034   0.00019195   0.00046000        -2762.68       28.24       67.67
   0.00019195  -0.01895001   0.00024693           28.24    -2787.64       36.32
   0.00046000   0.00024693  -0.01828738           67.67       36.32    -2690.17

   This run was terminated on:  22:40:55  10Feb2026

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
""",
                encoding="utf-8",
            )

            payload = workflow._parse_qe_results_from_output(outdir, natoms=2)

            self.assertIsNotNone(payload)
            self.assertAlmostEqual(payload["total_energy_eV"], -1173.47315217 * workflow.Ry)
            self.assertEqual(len(payload["forces_eV_A"]), 2)
            self.assertEqual(len(payload["stress_voigt"]), 6)
            self.assertIn("espresso.pwo", payload["parser_fallback"])

    def test_parse_qe_results_from_output_reads_final_relaxed_structure(self) -> None:
        template_atoms = Atoms(
            symbols=["Al", "Co"],
            positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
            cell=np.eye(3) * 3.0,
            pbc=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            (outdir / "espresso.pwo").write_text(
                """
     lattice parameter (alat)  =      7.0000  a.u.

CELL_PARAMETERS (angstrom)
  3.20000000  0.00000000  0.00000000
  0.00000000  3.30000000  0.00000000
  0.00000000  0.00000000  3.40000000
ATOMIC_POSITIONS (angstrom)
Al 0.10000000 0.20000000 0.30000000
Co 1.60000000 1.70000000 1.80000000

!    total energy              =   -10.00000000 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.00000000    0.10000000
     atom    2 type  2   force =     0.00000000    0.20000000    0.00000000

     Total force =     0.000000     Total SCF correction =     0.000000

     Computing stress (Cartesian axis) and pressure

          total   stress  (Ry/bohr**3)                   (kbar)     P=    -1.00
  -0.00100000   0.00000000   0.00000000        -1.00       0.00       0.00
   0.00000000  -0.00100000   0.00000000         0.00      -1.00       0.00
   0.00000000   0.00000000  -0.00100000         0.00       0.00      -1.00

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
""",
                encoding="utf-8",
            )

            payload = workflow._parse_qe_results_from_output(
                outdir,
                natoms=2,
                template_atoms=template_atoms,
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["final_symbols"], ["Al", "Co"])
            self.assertEqual(payload["final_cell_ang"][0], [3.2, 0.0, 0.0])
            self.assertEqual(payload["final_positions_ang"][1], [1.6, 1.7, 1.8])

    def test_parse_qe_results_from_output_accepts_truncated_log_without_job_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            (outdir / "espresso.pwo").write_text(
                """
!    total energy              =   -20.00000000 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.10000000    0.00000000

     Computing stress (Cartesian axis) and pressure

          total   stress  (Ry/bohr**3)                   (kbar)     P=    -2.00
  -0.00100000   0.00000000   0.00000000        -2.00       0.00       0.00
   0.00000000  -0.00100000   0.00000000         0.00      -2.00       0.00
   0.00000000   0.00000000  -0.00100000         0.00       0.00      -2.00
""",
                encoding="utf-8",
            )

            payload = workflow._parse_qe_results_from_output(outdir, natoms=1)

            self.assertIsNotNone(payload)
            self.assertAlmostEqual(payload["total_energy_eV"], -20.0 * workflow.Ry)

    def test_run_workflow_uses_fallback_relaxed_geometry_in_results(self) -> None:
        input_atoms = Atoms(
            symbols=["Al", "Co"],
            positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
            cell=np.eye(3) * 3.0,
            pbc=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input" / "REQ-VC-001"
            output_dir = root / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            write(input_dir / "structure.cif", input_atoms, format="cif")
            (input_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "request_id": "REQ-VC-001",
                        "composition": "Al0.5Co0.5",
                        "calculation": "vc-relax",
                        "encut": 520,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(workflow, "INPUT_DIR", root / "input"), patch.object(
                workflow, "OUTPUT_DIR", output_dir
            ), patch.object(
                workflow, "_select_pseudopotentials", return_value={}
            ), patch.object(
                workflow, "EspressoProfile", side_effect=lambda **_: object()
            ), patch.object(
                workflow,
                "_run_single_calculation",
                return_value={
                    "total_energy_eV": -2.5,
                    "stress_voigt": [0.0] * 6,
                    "forces_eV_A": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "max_force_eV_A": 0.0,
                    "parser_fallback": "espresso.pwo",
                    "final_cell_ang": [[3.2, 0.0, 0.0], [0.0, 3.3, 0.0], [0.0, 0.0, 3.4]],
                    "final_positions_ang": [[0.1, 0.2, 0.3], [1.6, 1.7, 1.8]],
                    "final_symbols": ["Al", "Co"],
                },
            ):
                result = workflow.run_workflow("REQ-VC-001")

            self.assertEqual(result["final_structure"]["lattice_vectors_ang"][0], [3.2, 0.0, 0.0])
            self.assertEqual(result["final_structure"]["positions_cartesian_ang"][1], [1.6, 1.7, 1.8])
            relaxed_atoms = read(output_dir / "REQ-VC-001" / "relaxed_structure.cif")
            np.testing.assert_allclose(relaxed_atoms.get_cell().array[0], [3.2, 0.0, 0.0])
            np.testing.assert_allclose(relaxed_atoms.get_positions()[1], [1.6, 1.7, 1.8])

    def test_run_single_calculation_attaches_relaxed_geometry_after_successful_ase_path(self) -> None:
        class FakeCalc:
            def __init__(self, **kwargs):
                self.directory = kwargs["directory"]

        class FakeAtoms:
            def __init__(self):
                self.calc = None

            def __len__(self):
                return 1

            def get_cell(self):
                return np.eye(3) * 3.0

            def copy(self):
                return Atoms(symbols=["Al"], positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 3.0, pbc=True)

            def get_potential_energy(self):
                outdir = Path(self.calc.directory)
                (outdir / "espresso.pwo").write_text(
                    """
CELL_PARAMETERS (angstrom)
  3.20000000  0.00000000  0.00000000
  0.00000000  3.30000000  0.00000000
  0.00000000  0.00000000  3.40000000
ATOMIC_POSITIONS (angstrom)
Al 0.10000000 0.20000000 0.30000000
""",
                    encoding="utf-8",
                )
                return -1.5

            def get_stress(self, voigt=True):
                return np.zeros(6)

            def get_forces(self):
                return np.zeros((1, 3))

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "qe"
            atoms = FakeAtoms()
            calc_kwargs = {
                "input_data": {"control": {"calculation": "vc-relax"}},
            }
            with patch.object(workflow, "Espresso", FakeCalc):
                result = workflow._run_single_calculation(atoms, calc_kwargs, outdir, "fake-success")

        self.assertEqual(result["final_cell_ang"][0], [3.2, 0.0, 0.0])
        self.assertEqual(result["final_positions_ang"][0], [0.1, 0.2, 0.3])

    def test_run_single_calculation_recovers_from_called_process_error_with_qe_output(self) -> None:
        class FakeCalc:
            def __init__(self, **kwargs):
                self.directory = kwargs["directory"]

        class FakeAtoms:
            def __init__(self):
                self.calc = None

            def __len__(self):
                return 1

            def get_potential_energy(self):
                outdir = Path(self.calc.directory)
                (outdir / "espresso.pwo").write_text(
                    """
!    total energy              =   -10.00000000 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.00000000    0.00100000

     Computing stress (Cartesian axis) and pressure

          total   stress  (Ry/bohr**3)                   (kbar)     P=    -1.00
  -0.00100000   0.00000000   0.00000000        -1.00       0.00       0.00
   0.00000000  -0.00100000   0.00000000         0.00      -1.00       0.00
   0.00000000   0.00000000  -0.00100000         0.00       0.00      -1.00
""",
                    encoding="utf-8",
                )
                raise subprocess.CalledProcessError(returncode=1, cmd="pw.x")

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "qe"
            atoms = FakeAtoms()
            calc_kwargs = {
                "input_data": {"control": {"calculation": "vc-relax"}},
            }
            with patch.object(workflow, "Espresso", FakeCalc):
                with self.assertRaises(subprocess.CalledProcessError):
                    workflow._run_single_calculation(atoms, calc_kwargs, outdir, "fake-called-process")

    def test_valid_cached_result_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "results.json"
            result_path.write_text(
                '{"status": "completed", "total_energy_eV": -1.23}',
                encoding="utf-8",
            )

            payload = workflow._load_cached_result_if_valid(result_path, "REQ-002")

            self.assertIsNotNone(payload)
            self.assertEqual(payload["status"], "completed")

    def test_compute_settings_hash_ignores_metadata_timestamp_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            structure_path = input_dir / "structure.cif"
            structure_path.write_text("data", encoding="utf-8")
            metadata_path = input_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "request_id": "REQ-001",
                        "composition": "Al0.5Co0.5",
                        "timestamp_utc": "2026-04-14T00:00:00Z",
                        "notes": "first",
                    }
                ),
                encoding="utf-8",
            )
            first_hash = workflow._compute_settings_hash(input_dir)

            metadata_path.write_text(
                json.dumps(
                    {
                        "request_id": "REQ-001",
                        "composition": "Al0.5Co0.5",
                        "timestamp_utc": "2026-04-15T00:00:00Z",
                        "notes": "second",
                    }
                ),
                encoding="utf-8",
            )
            second_hash = workflow._compute_settings_hash(input_dir)

            self.assertEqual(first_hash, second_hash)

    def test_cached_result_is_reused_when_metadata_signature_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "results.json"
            cached_metadata = {
                "request_id": "REQ-003",
                "candidate_id": "CAND-003",
                "composition": "Al0.5Co0.5",
                "phase": "FCC",
                "calculation": "vc-relax",
                "kpoint_grid": [5, 5, 5],
                "qe_overrides": {"ecutwfc_ry": 60.0, "ecutrho_ry": 480.0},
                "electrons": {"mixing_beta": 0.15},
                "relaxation": {"max_steps": 200},
                "timestamp_utc": "2026-03-03T00:00:00Z",
            }
            result_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "request_id": "REQ-003",
                        "dft_settings_hash": "old-mismatch",
                        "metadata": cached_metadata,
                    }
                ),
                encoding="utf-8",
            )

            payload = workflow._load_cached_result_if_valid(
                result_path,
                "REQ-003",
                expected_settings_hash="new-mismatch",
                expected_metadata_signature=workflow._settings_metadata_signature(
                    {
                        **cached_metadata,
                        "timestamp_utc": "2026-04-15T00:00:00Z",
                        "notes": "regenerated",
                        "handoff_signature": "newer-format",
                    }
                ),
            )

            self.assertIsNotNone(payload)
            self.assertEqual(payload["request_id"], "REQ-003")

    def test_run_queue_resume_uses_settings_hash_when_validating_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request_id = "REQ-QUEUE-001"
            input_dir = root / "input" / request_id
            output_dir = root / "output"
            monitor_dir = root / "monitor"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "metadata.json").write_text("{}", encoding="utf-8")
            (input_dir / "structure.cif").write_text("", encoding="utf-8")

            with patch.object(workflow, "INPUT_DIR", root / "input"), patch.object(
                workflow, "OUTPUT_DIR", output_dir
            ), patch.object(
                workflow, "QUEUE_RUN_DIR", root / "queue_runs"
            ), patch.object(
                workflow, "_compute_settings_hash", return_value="hash-queue-001"
            ), patch.object(
                workflow, "_load_cached_result_if_valid", return_value=None
            ) as mock_cache_check, patch.object(
                workflow,
                "run_workflow",
                return_value={
                    "total_energy_eV": -1.0,
                    "formation_energy_eV": -0.5,
                    "max_force_eV_A": 0.01,
                    "strain_results": [],
                },
            ) as mock_run_workflow:
                workflow.run_queue([request_id], resume=True, monitor_dir=monitor_dir)

            mock_cache_check.assert_called_once_with(
                output_dir / request_id / "results.json",
                request_id,
                expected_settings_hash="hash-queue-001",
                expected_metadata_signature=workflow._settings_metadata_signature({}),
            )
            mock_run_workflow.assert_called_once_with(request_id, resume=True)

    def test_cached_result_with_mismatched_settings_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "results.json"
            result_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "request_id": "REQ-003",
                        "dft_settings_hash": "stale-hash",
                    }
                ),
                encoding="utf-8",
            )

            payload = workflow._load_cached_result_if_valid(
                result_path,
                "REQ-003",
                expected_settings_hash="fresh-hash",
            )

            self.assertIsNone(payload)

    def test_run_workflow_resume_invalidates_stale_strain_results(self) -> None:
        input_atoms = Atoms(
            symbols=["Al"],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3) * 3.0,
            pbc=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input" / "REQ-VC-STRAIN"
            output_dir = root / "output" / "REQ-VC-STRAIN"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            write(input_dir / "structure.cif", input_atoms, format="cif")
            (input_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "request_id": "REQ-VC-STRAIN",
                        "composition": "Al1",
                        "calculation": "vc-relax",
                        "encut": 520,
                        "stress_analysis": {"strain_amplitude": 0.01, "strain_directions": 1},
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "strain_results.json").write_text(
                json.dumps([{"index": 99, "sign": "stale"}]),
                encoding="utf-8",
            )
            (output_dir / "strain_results.meta.json").write_text(
                json.dumps(
                    {
                        "request_id": "REQ-VC-STRAIN",
                        "dft_settings_hash": "old-hash",
                        "structure_signature": "old-structure",
                        "stress_analysis": {"strain_amplitude": 0.01, "strain_directions": 1},
                    }
                ),
                encoding="utf-8",
            )

            call_counter = {"count": 0}

            def fake_run_single_calculation(*args, **kwargs):
                call_counter["count"] += 1
                if call_counter["count"] == 1:
                    return {
                        "total_energy_eV": -2.5,
                        "stress_voigt": [0.0] * 6,
                        "forces_eV_A": [[0.0, 0.0, 0.0]],
                        "max_force_eV_A": 0.0,
                        "final_cell_ang": [[3.2, 0.0, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 3.2]],
                        "final_positions_ang": [[0.1, 0.2, 0.3]],
                        "final_symbols": ["Al"],
                    }
                return {
                    "total_energy_eV": -2.0,
                    "stress_voigt": [0.0] * 6,
                    "forces_eV_A": [[0.0, 0.0, 0.0]],
                    "max_force_eV_A": 0.0,
                }

            with patch.object(workflow, "INPUT_DIR", root / "input"), patch.object(
                workflow, "OUTPUT_DIR", root / "output"
            ), patch.object(
                workflow, "_select_pseudopotentials", return_value={}
            ), patch.object(
                workflow, "EspressoProfile", side_effect=lambda **_: object()
            ), patch.object(
                workflow, "_run_single_calculation", side_effect=fake_run_single_calculation
            ):
                result = workflow.run_workflow("REQ-VC-STRAIN", resume=True)

            self.assertEqual(call_counter["count"], 3)
            self.assertEqual(len(result["strain_results"]), 2)
            self.assertNotEqual(result["strain_results"][0]["index"], 99)

    def test_run_queue_resume_rejects_stale_cached_result(self) -> None:
        input_atoms = Atoms(
            symbols=["Al"],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3) * 3.0,
            pbc=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input" / "REQ-QUEUE-001"
            output_dir = root / "output" / "REQ-QUEUE-001"
            monitor_dir = root / "monitor"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            write(input_dir / "structure.cif", input_atoms, format="cif")
            (input_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "request_id": "REQ-QUEUE-001",
                        "composition": "Al1",
                        "calculation": "vc-relax",
                        "encut": 520,
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "request_id": "REQ-QUEUE-001",
                        "dft_settings_hash": "stale-hash",
                        "total_energy_eV": -1.0,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(workflow, "INPUT_DIR", root / "input"), patch.object(
                workflow, "OUTPUT_DIR", root / "output"
            ), patch.object(
                workflow, "QUEUE_RUN_DIR", root / "queue"
            ), patch.object(
                workflow,
                "run_workflow",
                return_value={"total_energy_eV": -2.0, "formation_energy_eV": -2.0, "max_force_eV_A": 0.0, "strain_results": []},
            ) as mock_run_workflow, patch.object(
                workflow, "_log_queue_to_mlflow"
            ):
                summary = workflow.run_queue(
                    ["REQ-QUEUE-001"],
                    max_workers=1,
                    max_retries=0,
                    monitor_dir=monitor_dir,
                    resume=True,
                )

            self.assertEqual(summary["completed_jobs"], 1)
            self.assertEqual(summary["jobs"][0]["attempts"], 1)
            mock_run_workflow.assert_called_once_with("REQ-QUEUE-001", resume=True)


if __name__ == "__main__":
    unittest.main()
