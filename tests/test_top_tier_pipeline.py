from __future__ import annotations

import json
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.publication.run_top_tier_non_qpu_pipeline import (  # noqa: E402
    ALL_STAGE_NAMES,
    PipelineError,
    _build_followup_candidates,
    _campaign_command,
    _check_gpu_backend_smoke,
    _check_qe_command_smoke,
    _evaluate_campaign_quality_gate,
    _enforce_campaign_quality_gate,
    _normalize_stage_selection,
    _parse_accepted_failed_request_ids,
    main,
)


class TopTierPipelineTests(unittest.TestCase):
    def test_stage_selection_normalizes_and_respects_skip_flags(self) -> None:
        self.assertEqual(
            _normalize_stage_selection(
                None,
                skip_paper_grade_suite=False,
                skip_m7=False,
                skip_release=False,
            ),
            list(ALL_STAGE_NAMES),
        )
        self.assertEqual(
            _normalize_stage_selection(
                ["discovery,reference", "release"],
                skip_paper_grade_suite=False,
                skip_m7=False,
                skip_release=False,
            ),
            ["discovery", "reference", "release"],
        )
        self.assertEqual(
            _normalize_stage_selection(
                ["all"],
                skip_paper_grade_suite=True,
                skip_m7=False,
                skip_release=True,
            ),
            ["discovery", "vc_relax", "elastic_eval", "reference", "m7"],
        )

    def test_campaign_quality_gate_passes_for_clean_summary(self) -> None:
        _enforce_campaign_quality_gate(
            "discovery",
            {
                "failed_jobs": 0,
                "valid_candidates": 12,
                "label_efficiency_gain": 0.5,
            },
        )

    def test_campaign_quality_gate_fails_immediately_on_failed_jobs(self) -> None:
        with self.assertRaises(PipelineError):
            _enforce_campaign_quality_gate(
                "discovery",
                {
                    "failed_jobs": 1,
                    "valid_candidates": 12,
                    "label_efficiency_gain": 0.5,
                },
            )

    def test_evaluate_campaign_quality_gate_accepts_explicit_failed_request_waiver(self) -> None:
        library = pd.DataFrame(
            [
                {"request_id": "REQ-001", "status": "completed"},
                {"request_id": "REQ-002", "status": "failed"},
            ]
        )

        result = _evaluate_campaign_quality_gate(
            "vc_relax",
            {
                "failed_jobs": 1,
                "valid_candidates": 12,
                "label_efficiency_gain": 0.5,
            },
            library=library,
            accepted_failed_request_ids=["REQ-002"],
        )

        self.assertEqual(result["waived_failed_request_ids"], ["REQ-002"])
        self.assertEqual(result["unwaived_failed_request_ids"], [])
        self.assertEqual(result["effective_failed_jobs"], 0)

    def test_evaluate_campaign_quality_gate_rejects_unknown_waiver_id(self) -> None:
        library = pd.DataFrame(
            [
                {"request_id": "REQ-001", "status": "completed"},
                {"request_id": "REQ-002", "status": "failed"},
            ]
        )

        with self.assertRaises(PipelineError):
            _evaluate_campaign_quality_gate(
                "vc_relax",
                {
                    "failed_jobs": 1,
                    "valid_candidates": 12,
                    "label_efficiency_gain": 0.5,
                },
                library=library,
                accepted_failed_request_ids=["REQ-404"],
            )

    def test_evaluate_campaign_quality_gate_rejects_unwaived_failed_request(self) -> None:
        library = pd.DataFrame(
            [
                {"request_id": "REQ-001", "status": "completed"},
                {"request_id": "REQ-002", "status": "failed"},
            ]
        )

        with self.assertRaises(PipelineError):
            _evaluate_campaign_quality_gate(
                "vc_relax",
                {
                    "failed_jobs": 1,
                    "valid_candidates": 12,
                    "label_efficiency_gain": 0.5,
                },
                library=library,
                accepted_failed_request_ids=[],
            )

    def test_evaluate_campaign_quality_gate_rejects_summary_library_mismatch(self) -> None:
        library = pd.DataFrame(
            [
                {"request_id": "REQ-001", "status": "completed"},
            ]
        )

        with self.assertRaises(PipelineError):
            _evaluate_campaign_quality_gate(
                "vc_relax",
                {
                    "failed_jobs": 1,
                    "valid_candidates": 12,
                    "label_efficiency_gain": 0.5,
                },
                library=library,
                accepted_failed_request_ids=[],
            )

    def test_parse_accepted_failed_request_ids_infers_vc_relax_stage_from_prefix(self) -> None:
        accepted_by_stage, normalized = _parse_accepted_failed_request_ids(["VCRLX-I01-C04"])

        self.assertEqual(accepted_by_stage["discovery"], [])
        self.assertEqual(accepted_by_stage["vc_relax"], ["VCRLX-I01-C04"])
        self.assertEqual(accepted_by_stage["elastic_eval"], [])
        self.assertEqual(normalized, ["vc_relax:VCRLX-I01-C04"])

    def test_parse_accepted_failed_request_ids_accepts_explicit_stage_syntax(self) -> None:
        accepted_by_stage, normalized = _parse_accepted_failed_request_ids(["elastic_eval:ELAST-I02-C03"])

        self.assertEqual(accepted_by_stage["elastic_eval"], ["ELAST-I02-C03"])
        self.assertEqual(normalized, ["elastic_eval:ELAST-I02-C03"])

    def test_parse_accepted_failed_request_ids_rejects_reference_stage(self) -> None:
        with self.assertRaises(ValueError):
            _parse_accepted_failed_request_ids(["reference:REF1-001"])

    @patch("scripts.publication.run_top_tier_non_qpu_pipeline.subprocess.run")
    def test_qe_command_smoke_uses_help_and_succeeds(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Program PWSCF"

        result = _check_qe_command_smoke("pw.x")

        self.assertEqual(result, "ok")
        mock_run.assert_called_once()
        argv = mock_run.call_args.args[0]
        self.assertEqual(argv, ["pw.x", "-h"])

    @patch("scripts.publication.run_top_tier_non_qpu_pipeline.subprocess.run")
    def test_qe_command_smoke_accepts_timeout_after_banner(self, mock_run) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["pw.x", "-h"],
            timeout=30,
            output="Program PWSCF v.7.4 starts on  3Mar2026",
        )

        result = _check_qe_command_smoke("pw.x")

        self.assertEqual(result, "ok")

    @patch("scripts.publication.run_top_tier_non_qpu_pipeline.subprocess.run")
    def test_qe_command_smoke_rejects_non_qe_zero_exit(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "usage: wrapper [options]"

        with self.assertRaises(PipelineError):
            _check_qe_command_smoke("pw.x")

    @patch("scripts.publication.run_top_tier_non_qpu_pipeline.rbf_kernel_backend")
    def test_gpu_backend_smoke_surfaces_runtime_error(self, mock_kernel) -> None:
        mock_kernel.side_effect = OSError("libcublas.so missing")

        with self.assertRaises(PipelineError):
            _check_gpu_backend_smoke()

    def test_preflight_only_writes_pass_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            log_root = root / "logs"
            discovery_csv = root / "candidates.csv"
            discovery_csv.write_text(
                "candidate_id,composition,property_compliant\n"
                "C-001,Al0.5Co0.5,1\n"
                "C-002,Fe0.5Ni0.5,1\n"
                "C-003,Cu0.5Zn0.5,1\n"
                "C-004,Ti0.5V0.5,1\n"
                "C-005,Nb0.5Mo0.5,1\n"
                "C-006,Cr0.5Mn0.5,1\n"
                "C-007,Al0.25Co0.25Fe0.25Ni0.25,1\n"
                "C-008,Ag0.5Pd0.5,1\n"
                "C-009,Li0.5Mg0.5,1\n"
                "C-010,Ca0.5Sr0.5,1\n",
                encoding="utf-8",
            )

            argv = [
                "run_top_tier_non_qpu_pipeline.py",
                "--preflight-only",
                "--skip-paper-grade-suite",
                "--tracking-uri",
                f"file://{root / 'mlruns'}",
                "--queue-tracking-uri",
                f"file://{root / 'mlruns'}",
                "--campaign-root",
                str(root / "campaigns"),
                "--report-path",
                str(report_path),
                "--log-root",
                str(log_root),
                "--discovery-candidate-csv",
                str(discovery_csv),
            ]

            with patch.object(sys, "argv", argv), patch(
                "scripts.publication.run_top_tier_non_qpu_pipeline._check_prereqs_for_modes",
                return_value={"qe_pw_command": "pw.x"},
            ), patch(
                "scripts.publication.run_top_tier_non_qpu_pipeline._check_reference_dataset",
                return_value={"reference_rows_with_density": "42"},
            ):
                main()

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["steps"], [])
            self.assertTrue(payload["configuration"]["preflight_only"])

    def test_campaign_command_forwards_resume(self) -> None:
        cmd = _campaign_command(
            campaign_id="camp-001",
            candidate_csv=Path("/tmp/candidates.csv"),
            iterations=3,
            top_k=4,
            max_workers=2,
            max_retries=1,
            job_class="vc_relax",
            tracking_uri="file:///tmp/mlruns",
            queue_tracking_uri="file:///tmp/mlruns",
            classical_label_budget=100,
            k_grid=(5, 5, 5),
            ecutwfc=60.0,
            ecutrho=720.0,
            supercell=(2, 2, 2),
            random_state=42,
            max_wall_seconds=3600,
            campaign_root=Path("/tmp/campaigns"),
            request_prefix="VCRLX",
            resume=True,
        )

        self.assertIn("--resume", cmd)

    def test_preflight_only_fails_when_vc_relax_lacks_discovery_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            log_root = root / "logs"
            argv = [
                "run_top_tier_non_qpu_pipeline.py",
                "--preflight-only",
                "--stages",
                "vc_relax",
                "--tracking-uri",
                f"file://{root / 'mlruns'}",
                "--queue-tracking-uri",
                f"file://{root / 'mlruns'}",
                "--campaign-root",
                str(root / "campaigns"),
                "--report-path",
                str(report_path),
                "--log-root",
                str(log_root),
            ]

            with patch.object(sys, "argv", argv), patch(
                "scripts.publication.run_top_tier_non_qpu_pipeline._check_prereqs_for_modes",
                return_value={"qe_pw_command": "pw.x"},
            ):
                with self.assertRaises(SystemExit):
                    main()

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "fail")
            self.assertIn("--discovery-campaign-id", payload["error"])

    def test_preflight_only_passes_for_vc_relax_with_existing_discovery_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            log_root = root / "logs"
            campaign_root = root / "campaigns"
            discovery_root = campaign_root / "disc-001"
            discovery_root.mkdir(parents=True, exist_ok=True)
            (discovery_root / "closed_loop_summary.json").write_text(
                json.dumps({"status": "completed", "failed_jobs": 0}),
                encoding="utf-8",
            )
            (discovery_root / "candidate_library.csv").write_text(
                "candidate_id,composition,phase,density_g_cm3,formation_energy_eV,valid_flag,status,max_force_eV_A,latency_s\n"
                "C-001,Al0.5Co0.5,FCC,7.0,-1.0,1,completed,0.01,1.0\n",
                encoding="utf-8",
            )

            argv = [
                "run_top_tier_non_qpu_pipeline.py",
                "--preflight-only",
                "--stages",
                "vc_relax",
                "--discovery-campaign-id",
                "disc-001",
                "--tracking-uri",
                f"file://{root / 'mlruns'}",
                "--queue-tracking-uri",
                f"file://{root / 'mlruns'}",
                "--campaign-root",
                str(campaign_root),
                "--report-path",
                str(report_path),
                "--log-root",
                str(log_root),
                "--resume",
            ]

            with patch.object(sys, "argv", argv), patch(
                "scripts.publication.run_top_tier_non_qpu_pipeline._check_prereqs_for_modes",
                return_value={"qe_pw_command": "pw.x"},
            ):
                main()

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["campaigns"]["discovery"], "disc-001")
            self.assertEqual(payload["configuration"]["stages"], ["vc_relax"])
            self.assertTrue(payload["configuration"]["resume"])

    def test_preflight_only_reports_accepted_failed_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            log_root = root / "logs"
            discovery_csv = root / "candidates.csv"
            discovery_csv.write_text(
                "candidate_id,composition,property_compliant\n"
                "C-001,Al0.5Co0.5,1\n"
                "C-002,Fe0.5Ni0.5,1\n"
                "C-003,Cu0.5Zn0.5,1\n"
                "C-004,Ti0.5V0.5,1\n"
                "C-005,Nb0.5Mo0.5,1\n"
                "C-006,Cr0.5Mn0.5,1\n"
                "C-007,Al0.25Co0.25Fe0.25Ni0.25,1\n"
                "C-008,Ag0.5Pd0.5,1\n"
                "C-009,Li0.5Mg0.5,1\n"
                "C-010,Ca0.5Sr0.5,1\n",
                encoding="utf-8",
            )
            argv = [
                "run_top_tier_non_qpu_pipeline.py",
                "--preflight-only",
                "--skip-paper-grade-suite",
                "--accepted-failed-request-id",
                "VCRLX-I01-C04",
                "--tracking-uri",
                f"file://{root / 'mlruns'}",
                "--queue-tracking-uri",
                f"file://{root / 'mlruns'}",
                "--campaign-root",
                str(root / "campaigns"),
                "--report-path",
                str(report_path),
                "--log-root",
                str(log_root),
                "--discovery-candidate-csv",
                str(discovery_csv),
            ]

            with patch.object(sys, "argv", argv), patch(
                "scripts.publication.run_top_tier_non_qpu_pipeline._check_prereqs_for_modes",
                return_value={"qe_pw_command": "pw.x"},
            ), patch(
                "scripts.publication.run_top_tier_non_qpu_pipeline._check_reference_dataset",
                return_value={"reference_rows_with_density": "42"},
            ):
                main()

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["configuration"]["accepted_failed_request_ids"], ["vc_relax:VCRLX-I01-C04"])

    def test_build_followup_candidates_propagates_source_result_path(self) -> None:
        source_library = json.loads(
            '[{"candidate_id":"C-001","composition":"Al0.5Co0.5","phase":"FCC","density_g_cm3":7.0,"formation_energy_eV":-1.0,"valid_flag":1,"status":"completed","max_force_eV_A":0.01,"latency_s":1.0,"result_path":"/tmp/vc/results.json"},'
            '{"candidate_id":"C-002","composition":"Fe0.5Ni0.5","phase":"BCC","density_g_cm3":7.1,"formation_energy_eV":-0.9,"valid_flag":1,"status":"completed","max_force_eV_A":0.02,"latency_s":2.0,"result_path":"/tmp/vc2/results.json"},'
            '{"candidate_id":"C-003","composition":"Cu0.5Zn0.5","phase":"FCC","density_g_cm3":7.2,"formation_energy_eV":-0.8,"valid_flag":1,"status":"completed","max_force_eV_A":0.03,"latency_s":3.0,"result_path":"/tmp/vc3/results.json"},'
            '{"candidate_id":"C-004","composition":"Ti0.5V0.5","phase":"BCC","density_g_cm3":7.3,"formation_energy_eV":-0.7,"valid_flag":1,"status":"completed","max_force_eV_A":0.04,"latency_s":4.0,"result_path":"/tmp/vc4/results.json"},'
            '{"candidate_id":"C-005","composition":"Nb0.5Mo0.5","phase":"BCC","density_g_cm3":7.4,"formation_energy_eV":-0.6,"valid_flag":1,"status":"completed","max_force_eV_A":0.05,"latency_s":5.0,"result_path":"/tmp/vc5/results.json"},'
            '{"candidate_id":"C-006","composition":"Cr0.5Mn0.5","phase":"BCC","density_g_cm3":7.5,"formation_energy_eV":-0.5,"valid_flag":1,"status":"completed","max_force_eV_A":0.06,"latency_s":6.0,"result_path":"/tmp/vc6/results.json"},'
            '{"candidate_id":"C-007","composition":"Al0.25Co0.25Fe0.25Ni0.25","phase":"FCC","density_g_cm3":7.6,"formation_energy_eV":-0.4,"valid_flag":1,"status":"completed","max_force_eV_A":0.07,"latency_s":7.0,"result_path":"/tmp/vc7/results.json"},'
            '{"candidate_id":"C-008","composition":"Ag0.5Pd0.5","phase":"FCC","density_g_cm3":7.7,"formation_energy_eV":-0.3,"valid_flag":1,"status":"completed","max_force_eV_A":0.08,"latency_s":8.0,"result_path":"/tmp/vc8/results.json"},'
            '{"candidate_id":"C-009","composition":"Li0.5Mg0.5","phase":"FCC","density_g_cm3":7.8,"formation_energy_eV":-0.2,"valid_flag":1,"status":"completed","max_force_eV_A":0.09,"latency_s":9.0,"result_path":"/tmp/vc9/results.json"},'
            '{"candidate_id":"C-010","composition":"Ca0.5Sr0.5","phase":"FCC","density_g_cm3":7.9,"formation_energy_eV":-0.1,"valid_flag":1,"status":"completed","max_force_eV_A":0.10,"latency_s":10.0,"result_path":"/tmp/vc10/results.json"}]'
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "elastic_candidates.csv"
            out_df = _build_followup_candidates(
                pd.DataFrame(source_library),
                out_path=out_path,
                target_n=10,
                id_prefix="ELAST",
            )

            self.assertIn("source_result_path", out_df.columns)
            self.assertEqual(out_df.iloc[0]["source_result_path"], "/tmp/vc/results.json")

    def test_build_followup_candidates_dedupes_duplicate_compositions_for_vc_relax(self) -> None:
        source_library = pd.DataFrame(
            [
                {"candidate_id": "C-001", "composition": "Al0.5Co0.5", "phase": "FCC", "density_g_cm3": 7.0, "formation_energy_eV": -1.0, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.01, "latency_s": 1.0, "result_path": "/tmp/r1.json"},
                {"candidate_id": "C-002", "composition": "Al0.5Co0.5", "phase": "FCC", "density_g_cm3": 7.1, "formation_energy_eV": -0.9, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.02, "latency_s": 2.0, "result_path": "/tmp/r2.json"},
                {"candidate_id": "C-003", "composition": "Fe0.5Ni0.5", "phase": "BCC", "density_g_cm3": 7.2, "formation_energy_eV": -0.8, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.03, "latency_s": 3.0, "result_path": "/tmp/r3.json"},
                {"candidate_id": "C-004", "composition": "Cu0.5Zn0.5", "phase": "FCC", "density_g_cm3": 7.3, "formation_energy_eV": -0.7, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.04, "latency_s": 4.0, "result_path": "/tmp/r4.json"},
                {"candidate_id": "C-005", "composition": "Ti0.5V0.5", "phase": "BCC", "density_g_cm3": 7.4, "formation_energy_eV": -0.6, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.05, "latency_s": 5.0, "result_path": "/tmp/r5.json"},
                {"candidate_id": "C-006", "composition": "Nb0.5Mo0.5", "phase": "BCC", "density_g_cm3": 7.5, "formation_energy_eV": -0.5, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.06, "latency_s": 6.0, "result_path": "/tmp/r6.json"},
                {"candidate_id": "C-007", "composition": "Cr0.5Mn0.5", "phase": "BCC", "density_g_cm3": 7.6, "formation_energy_eV": -0.4, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.07, "latency_s": 7.0, "result_path": "/tmp/r7.json"},
                {"candidate_id": "C-008", "composition": "Al0.25Co0.25Fe0.25Ni0.25", "phase": "FCC", "density_g_cm3": 7.7, "formation_energy_eV": -0.3, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.08, "latency_s": 8.0, "result_path": "/tmp/r8.json"},
                {"candidate_id": "C-009", "composition": "Ag0.5Pd0.5", "phase": "FCC", "density_g_cm3": 7.8, "formation_energy_eV": -0.2, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.09, "latency_s": 9.0, "result_path": "/tmp/r9.json"},
                {"candidate_id": "C-010", "composition": "Li0.5Mg0.5", "phase": "FCC", "density_g_cm3": 7.9, "formation_energy_eV": -0.1, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.10, "latency_s": 10.0, "result_path": "/tmp/r10.json"},
                {"candidate_id": "C-011", "composition": "Ca0.5Sr0.5", "phase": "FCC", "density_g_cm3": 8.0, "formation_energy_eV": 0.0, "valid_flag": 1, "status": "completed", "max_force_eV_A": 0.11, "latency_s": 11.0, "result_path": "/tmp/r11.json"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_df = _build_followup_candidates(
                source_library,
                out_path=Path(tmp) / "vc_relax_candidates.csv",
                target_n=10,
                id_prefix="VCRLX",
            )

        self.assertEqual((out_df["composition"] == "Al0.5Co0.5").sum(), 1)

    def test_build_followup_candidates_keeps_distinct_relaxed_structures_for_same_composition(self) -> None:
        rows = []
        for idx in range(10):
            rows.append(
                {
                    "candidate_id": f"C-{idx:03d}",
                    "composition": "Al0.5Co0.5",
                    "phase": "FCC",
                    "density_g_cm3": 7.0 + idx * 0.01,
                    "formation_energy_eV": -1.0 + idx * 0.01,
                    "valid_flag": 1,
                    "status": "completed",
                    "max_force_eV_A": 0.01 + idx * 0.001,
                    "latency_s": 1.0 + idx,
                    "result_path": f"/tmp/vc-{idx}/results.json",
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "elastic_candidates.csv"
            out_df = _build_followup_candidates(
                pd.DataFrame(rows),
                out_path=out_path,
                target_n=10,
                id_prefix="ELAST",
                require_source_result=True,
            )

        self.assertEqual(len(out_df), 10)
        self.assertEqual(out_df["composition"].nunique(), 1)
        self.assertEqual(out_df["source_result_path"].nunique(), 10)

    def test_build_followup_candidates_requires_result_path_for_elastic_eval(self) -> None:
        rows = []
        for idx in range(10):
            rows.append(
                {
                    "candidate_id": f"C-{idx:03d}",
                    "composition": f"Al0.{idx}Co0.{10-idx}",
                    "phase": "FCC",
                    "density_g_cm3": 7.0 + idx * 0.01,
                    "formation_energy_eV": -1.0 + idx * 0.01,
                    "valid_flag": 1,
                    "status": "completed",
                    "max_force_eV_A": 0.01 + idx * 0.001,
                    "latency_s": 1.0 + idx,
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "elastic_candidates.csv"
            with self.assertRaises(PipelineError):
                _build_followup_candidates(
                    pd.DataFrame(rows),
                    out_path=out_path,
                    target_n=10,
                    id_prefix="ELAST",
                    require_source_result=True,
                )


if __name__ == "__main__":
    unittest.main()
