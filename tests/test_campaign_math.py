from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.io import read, write

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.hpc.run_real_dft_campaign import (  # noqa: E402
    _build_handoff_metadata,
    _candidate_passes_quality_gate,
    _handoff_package_is_valid,
    compute_label_efficiency,
    parse_composition,
    write_handoff_package,
)


class CampaignMathTests(unittest.TestCase):
    def test_parse_composition_normalizes_counts(self) -> None:
        parsed = parse_composition("Al2 Co1 Ni1")
        self.assertAlmostEqual(sum(parsed.values()), 1.0, places=6)
        self.assertAlmostEqual(parsed["Al"], 0.5, places=6)
        self.assertAlmostEqual(parsed["Co"], 0.25, places=6)
        self.assertAlmostEqual(parsed["Ni"], 0.25, places=6)

    def test_label_efficiency_gain(self) -> None:
        gain = compute_label_efficiency(120, 24)
        self.assertAlmostEqual(gain, 0.8, places=6)

    def test_label_efficiency_clamped(self) -> None:
        gain = compute_label_efficiency(100, 140)
        self.assertEqual(gain, 0.0)

    def test_label_efficiency_invalid_budget_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_label_efficiency(0, 10)

    def test_request_packaging_creates_metadata_and_structure(self) -> None:
        candidate = pd.Series(
            {
                "candidate_id": "CAND-001",
                "composition": "Al2 Co1 Ni1",
                "phase": "fcc",
                "predicted_density_g_cm3": 7.3,
                "target_density_g_cm3": 7.5,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            handoff = write_handoff_package(
                candidate,
                request_id="REQ-PACK-001",
                job_class="scf_screen",
                iteration_index=1,
                output_root=out_root,
                random_state=42,
                base_lattice_constant=3.65,
                supercell=(2, 2, 2),
                calculation="scf",
                relaxation={},
            )
            self.assertTrue((handoff / "metadata.json").exists())
            self.assertTrue((handoff / "structure.cif").exists())

            payload = json.loads((handoff / "metadata.json").read_text())
            self.assertEqual(payload["request_id"], "REQ-PACK-001")
            self.assertEqual(payload["job_class"], "scf_screen")
            self.assertEqual(payload["calculation"], "scf")

    def test_handoff_package_validation_rejects_zero_byte_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handoff_dir = Path(tmp) / "REQ-001"
            handoff_dir.mkdir(parents=True)
            (handoff_dir / "metadata.json").write_text("", encoding="utf-8")
            (handoff_dir / "structure.cif").write_text("", encoding="utf-8")

            self.assertFalse(_handoff_package_is_valid(handoff_dir))

    def test_handoff_package_validation_accepts_written_package(self) -> None:
        candidate = pd.Series(
            {
                "candidate_id": "CAND-002",
                "composition": "Al2 Co1 Ni1",
                "phase": "fcc",
                "predicted_density_g_cm3": 7.3,
                "target_density_g_cm3": 7.5,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            handoff_dir = write_handoff_package(
                candidate,
                request_id="REQ-PACK-002",
                job_class="scf_screen",
                iteration_index=1,
                output_root=Path(tmp),
                random_state=42,
                base_lattice_constant=3.65,
                supercell=(2, 2, 2),
                calculation="scf",
                relaxation={},
            )

            self.assertTrue(_handoff_package_is_valid(handoff_dir))

    def test_handoff_package_validation_rejects_signature_mismatch(self) -> None:
        candidate = pd.Series(
            {
                "candidate_id": "CAND-002B",
                "composition": "Al2 Co1 Ni1",
                "phase": "fcc",
                "predicted_density_g_cm3": 7.3,
                "target_density_g_cm3": 7.3,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            handoff_dir = write_handoff_package(
                candidate,
                request_id="REQ-PACK-002B",
                job_class="scf_screen",
                iteration_index=1,
                output_root=Path(tmp),
                random_state=42,
                base_lattice_constant=3.65,
                supercell=(2, 2, 2),
                calculation="scf",
                relaxation={},
            )

            self.assertFalse(
                _handoff_package_is_valid(
                    handoff_dir,
                    expected_signature="not-the-real-signature",
                )
            )

    def test_write_handoff_package_reuses_relaxed_structure_from_prior_result(self) -> None:
        candidate = pd.Series(
            {
                "candidate_id": "CAND-003",
                "composition": "Al0.5Co0.5",
                "phase": "fcc",
                "predicted_density_g_cm3": 7.3,
                "target_density_g_cm3": 7.3,
            }
        )
        relaxed_atoms = Atoms(
            symbols=["Al", "Co"],
            positions=[[0.1, 0.2, 0.3], [1.6, 1.7, 1.8]],
            cell=[[3.2, 0.0, 0.0], [0.0, 3.3, 0.0], [0.0, 0.0, 3.4]],
            pbc=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prior_dir = tmp_path / "prior"
            prior_dir.mkdir(parents=True)
            write(prior_dir / "relaxed_structure.cif", relaxed_atoms, format="cif")
            result_path = prior_dir / "results.json"
            result_path.write_text(
                json.dumps(
                    {
                        "artefacts": {"relaxed_structure_cif": "relaxed_structure.cif"},
                        "final_structure": {
                            "lattice_vectors_ang": relaxed_atoms.get_cell().tolist(),
                            "atomic_symbols": relaxed_atoms.get_chemical_symbols(),
                            "positions_cartesian_ang": relaxed_atoms.get_positions().tolist(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate["source_result_path"] = str(result_path)

            handoff_dir = write_handoff_package(
                candidate,
                request_id="REQ-PACK-003",
                job_class="elastic_eval",
                iteration_index=1,
                output_root=tmp_path / "handoff",
                random_state=42,
                base_lattice_constant=3.65,
                supercell=(2, 2, 2),
                calculation="scf",
                relaxation={},
            )

            handoff_atoms = read(handoff_dir / "structure.cif")
            self.assertEqual(handoff_atoms.get_chemical_symbols(), ["Al", "Co"])
            self.assertEqual(json.loads((handoff_dir / "metadata.json").read_text())["source_result_path"], str(result_path))
            self.assertAlmostEqual(handoff_atoms.get_positions()[1][0], 1.6)
            self.assertAlmostEqual(handoff_atoms.get_cell().array[2][2], 3.4)

    def test_handoff_signature_changes_when_source_result_changes(self) -> None:
        candidate = pd.Series(
            {
                "candidate_id": "CAND-003B",
                "composition": "Al0.5Co0.5",
                "phase": "fcc",
                "predicted_density_g_cm3": 7.3,
                "target_density_g_cm3": 7.3,
            }
        )
        relaxed_atoms = Atoms(
            symbols=["Al", "Co"],
            positions=[[0.1, 0.2, 0.3], [1.6, 1.7, 1.8]],
            cell=[[3.2, 0.0, 0.0], [0.0, 3.3, 0.0], [0.0, 0.0, 3.4]],
            pbc=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prior_dir = tmp_path / "prior"
            prior_dir.mkdir(parents=True)
            write(prior_dir / "relaxed_structure.cif", relaxed_atoms, format="cif")
            result_path = prior_dir / "results.json"
            result_path.write_text(
                json.dumps(
                    {
                        "artefacts": {"relaxed_structure_cif": "relaxed_structure.cif"},
                        "final_structure": {
                            "lattice_vectors_ang": relaxed_atoms.get_cell().tolist(),
                            "atomic_symbols": relaxed_atoms.get_chemical_symbols(),
                            "positions_cartesian_ang": relaxed_atoms.get_positions().tolist(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate["source_result_path"] = str(result_path)

            handoff_dir = write_handoff_package(
                candidate,
                request_id="REQ-PACK-003B",
                job_class="elastic_eval",
                iteration_index=1,
                output_root=tmp_path / "handoff",
                random_state=42,
                base_lattice_constant=3.65,
                supercell=(2, 2, 2),
                calculation="scf",
                relaxation={},
            )

            updated_atoms = relaxed_atoms.copy()
            updated_atoms.set_positions([[0.2, 0.3, 0.4], [1.7, 1.8, 1.9]])
            write(prior_dir / "relaxed_structure.cif", updated_atoms, format="cif")
            result_path.write_text(
                json.dumps(
                    {
                        "artefacts": {"relaxed_structure_cif": "relaxed_structure.cif"},
                        "final_structure": {
                            "lattice_vectors_ang": updated_atoms.get_cell().tolist(),
                            "atomic_symbols": updated_atoms.get_chemical_symbols(),
                            "positions_cartesian_ang": updated_atoms.get_positions().tolist(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            expected_signature = _build_handoff_metadata(
                candidate=candidate,
                request_id="REQ-PACK-003B",
                job_class="elastic_eval",
                iteration_index=1,
                calculation="scf",
                relaxation={},
                stress_analysis=None,
                custom_k_grid=None,
                custom_cutoffs=None,
                random_state=42,
                base_lattice_constant=3.65,
                supercell=(2, 2, 2),
                source_result_path=result_path,
            )["handoff_signature"]

            self.assertFalse(
                _handoff_package_is_valid(
                    handoff_dir,
                    expected_signature=expected_signature,
                )
            )

    def test_elastic_eval_handoff_requires_source_result_path(self) -> None:
        candidate = pd.Series(
            {
                "candidate_id": "CAND-004",
                "composition": "Al0.5Co0.5",
                "phase": "fcc",
                "predicted_density_g_cm3": 7.3,
                "target_density_g_cm3": 7.3,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_handoff_package(
                    candidate,
                    request_id="REQ-PACK-004",
                    job_class="elastic_eval",
                    iteration_index=1,
                    output_root=Path(tmp),
                    random_state=42,
                    base_lattice_constant=3.65,
                    supercell=(2, 2, 2),
                    calculation="scf",
                    relaxation={},
                )

    def test_candidate_quality_gate_rejects_high_pressure_vc_relax_results(self) -> None:
        self.assertFalse(
            _candidate_passes_quality_gate(
                completed=True,
                result={
                    "max_force_eV_A": 0.01,
                    "properties": {"exp_density_g_cm3": 7.5},
                    "stress": {"pressure_kbar": 3.5},
                },
                max_force_threshold=0.03,
                pressure_threshold_kbar=1.0,
            )
        )

    def test_candidate_quality_gate_accepts_pressure_within_threshold(self) -> None:
        self.assertTrue(
            _candidate_passes_quality_gate(
                completed=True,
                result={
                    "max_force_eV_A": 0.01,
                    "properties": {"exp_density_g_cm3": 7.5},
                    "stress": {"pressure_kbar": 0.8},
                },
                max_force_threshold=0.03,
                pressure_threshold_kbar=1.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
