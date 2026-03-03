#!/usr/bin/env python3
"""Run the paper data pipeline end-to-end with fail-fast quality gates.

This script orchestrates M5-real through M9 data-generation steps and exits non-zero
as soon as any command or acceptance gate fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
DEFAULT_TRACKING_URI = str((BASE_DIR / "mlruns").resolve())
DEFAULT_CAMPAIGN_ID = "t5r4-20260211-fasttrack-221-mw4"
DEFAULT_CAMPAIGN_ROOT = DATA_DIR / "dft_workflow" / "campaigns"
DEFAULT_RELEASE_DIR = DATA_DIR / "releases" / "real_dft_campaign_v1"
DEFAULT_REPORT_PATH = DATA_DIR / "reproducibility" / "paper_pipeline_report.json"
DEFAULT_PAPER_GRADE_SUMMARY = DATA_DIR / "benchmarks" / "paper_grade" / "paper_grade_suite_summary.json"
DEFAULT_PAPER_GRADE_RUNS = DATA_DIR / "benchmarks" / "paper_grade" / "paper_grade_suite_runs.csv"


class PipelineError(RuntimeError):
    """Raised when a pipeline command or acceptance gate fails."""


@dataclass
class StepResult:
    step: str
    command: List[str]
    returncode: int
    duration_s: float
    started_utc: str
    finished_utc: str
    stdout_tail: str
    stderr_tail: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tail(text: str, max_lines: int = 40, max_chars: int = 4000) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    clipped = "\n".join(lines[-max_lines:])
    if len(clipped) > max_chars:
        return clipped[-max_chars:]
    return clipped


def _load_json(path: Path, label: str) -> Dict[str, object]:
    if not path.exists():
        raise PipelineError(f"Missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_step(step: str, command: Sequence[str]) -> StepResult:
    start = datetime.now(timezone.utc)
    proc = subprocess.run(
        list(command),
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    finish = datetime.now(timezone.utc)

    result = StepResult(
        step=step,
        command=list(command),
        returncode=int(proc.returncode),
        duration_s=(finish - start).total_seconds(),
        started_utc=start.isoformat().replace("+00:00", "Z"),
        finished_utc=finish.isoformat().replace("+00:00", "Z"),
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
    )

    if proc.returncode != 0:
        raise PipelineError(
            f"Step '{step}' failed with exit code {proc.returncode}. "
            f"stderr tail: {result.stderr_tail or '<empty>'}"
        )

    return result


def _gate(gates: List[Dict[str, object]], name: str, passed: bool, detail: str) -> None:
    gates.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        raise PipelineError(f"Gate failed: {name}. {detail}")


def _validate_campaign_summary(
    summary: Dict[str, object],
    gates: List[Dict[str, object]],
    *,
    min_label_efficiency: float,
    min_valid_candidates: int,
    max_failed_jobs: int,
    expected_label_efficiency: float | None,
    expected_valid_candidates: int | None,
    expected_completed_jobs: int | None,
    expected_tolerance: float,
) -> None:
    label_efficiency = float(summary.get("label_efficiency_gain", -1.0))
    valid_candidates = int(summary.get("valid_candidates", -1))
    completed_jobs = int(summary.get("completed_jobs", -1))
    failed_jobs = int(summary.get("failed_jobs", -1))

    _gate(
        gates,
        "campaign_label_efficiency",
        label_efficiency >= min_label_efficiency,
        f"label_efficiency_gain={label_efficiency:.4f}, required>={min_label_efficiency:.4f}",
    )
    _gate(
        gates,
        "campaign_valid_candidates",
        valid_candidates >= min_valid_candidates,
        f"valid_candidates={valid_candidates}, required>={min_valid_candidates}",
    )
    _gate(
        gates,
        "campaign_failed_jobs",
        failed_jobs <= max_failed_jobs,
        f"failed_jobs={failed_jobs}, allowed<={max_failed_jobs}",
    )

    if expected_label_efficiency is not None:
        _gate(
            gates,
            "campaign_expected_label_efficiency",
            abs(label_efficiency - expected_label_efficiency) <= expected_tolerance,
            (
                f"label_efficiency_gain={label_efficiency:.4f}, expected={expected_label_efficiency:.4f}, "
                f"tolerance={expected_tolerance:.4f}"
            ),
        )
    if expected_valid_candidates is not None:
        _gate(
            gates,
            "campaign_expected_valid_candidates",
            valid_candidates == expected_valid_candidates,
            f"valid_candidates={valid_candidates}, expected={expected_valid_candidates}",
        )
    if expected_completed_jobs is not None:
        _gate(
            gates,
            "campaign_expected_completed_jobs",
            completed_jobs == expected_completed_jobs,
            f"completed_jobs={completed_jobs}, expected={expected_completed_jobs}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper data pipeline with fail-fast gates.")

    parser.add_argument(
        "--profile",
        choices=["paper-grade", "fasttrack"],
        default="paper-grade",
        help="Execution profile. 'paper-grade' runs the 1100-run GPU benchmark suite.",
    )

    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)

    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--queue-tracking-uri", default=None)

    parser.add_argument("--run-real-campaign", action="store_true", help="Run T5R.4 campaign before downstream steps.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing handoffs/QE outputs when running real campaign.")

    parser.add_argument("--candidate-csv", type=Path, default=DATA_DIR / "qml" / "qgan_conditioned_candidates_fasttrack.csv")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--classical-label-budget", type=int, default=120)
    parser.add_argument("--request-prefix", default="REALCAM")
    parser.add_argument("--job-class", choices=["scf_screen", "vc_relax", "elastic_eval"], default="scf_screen")
    parser.add_argument("--max-wall-seconds", type=int, default=None)

    parser.add_argument("--min-label-efficiency", type=float, default=0.30)
    parser.add_argument("--min-valid-candidates", type=int, default=10)
    parser.add_argument("--max-failed-jobs", type=int, default=0)
    parser.add_argument("--max-validation-gap", type=float, default=0.05)
    parser.add_argument("--min-hardware-runs", type=int, default=3)
    parser.add_argument("--min-qvc-gap", type=float, default=0.0)
    parser.add_argument(
        "--allow-validation-review",
        action="store_true",
        help="Treat DFT validation status='review' as pass (still enforces --max-validation-gap when present).",
    )

    parser.add_argument("--expected-label-efficiency", type=float, default=0.90)
    parser.add_argument("--expected-valid-candidates", type=int, default=12)
    parser.add_argument("--expected-completed-jobs", type=int, default=12)
    parser.add_argument("--expected-tolerance", type=float, default=1e-6)

    parser.add_argument("--skip-paper-grade-suite", action="store_true")
    parser.add_argument("--paper-grade-total-runs", type=int, default=1100)
    parser.add_argument("--paper-grade-qsvr-runs", type=int, default=None)
    parser.add_argument("--paper-grade-qgpr-runs", type=int, default=None)
    parser.add_argument("--paper-grade-classical-runs", type=int, default=None)
    paper_grade_speed_group = parser.add_mutually_exclusive_group()
    paper_grade_speed_group.add_argument(
        "--paper-grade-fast",
        action="store_true",
        help="Use the paper-grade fast preset for per-run workload sizing.",
    )
    paper_grade_speed_group.add_argument(
        "--paper-grade-fastest",
        action="store_true",
        help="Use the paper-grade fastest preset (default behavior).",
    )
    parser.add_argument("--paper-grade-seed", type=int, default=20260302)
    parser.add_argument("--paper-grade-max-runtime-seconds", type=float, default=7200.0)
    parser.set_defaults(paper_grade_enforce_runtime=True)
    parser.add_argument(
        "--paper-grade-enforce-runtime",
        dest="paper_grade_enforce_runtime",
        action="store_true",
        help="Enforce the paper-grade max-runtime gate (enabled by default).",
    )
    parser.add_argument(
        "--paper-grade-skip-runtime-gate",
        dest="paper_grade_enforce_runtime",
        action="store_false",
        help="Disable max-runtime gating for the paper-grade suite.",
    )
    parser.add_argument("--paper-grade-allow-cpu-fallback", action="store_true")
    parser.add_argument("--paper-grade-max-qsvr-relative-gap", type=float, default=0.10)
    parser.add_argument("--paper-grade-max-abs-qgpr-coverage-gap", type=float, default=0.05)
    parser.add_argument(
        "--paper-grade-qgpr-engine",
        choices=["auto", "sklearn", "backend"],
        default="sklearn",
        help="QGPR engine for paper-grade suite (default: sklearn).",
    )
    parser.add_argument(
        "--paper-grade-qgpr-optimizer",
        default="fmin_l_bfgs_b",
        help="QGPR optimizer for paper-grade suite.",
    )
    parser.add_argument(
        "--paper-grade-qgpr-catastrophic-coverage-gap",
        type=float,
        default=0.25,
        help="Immediate fail-fast per-run QGPR abs coverage gap threshold.",
    )
    parser.add_argument(
        "--paper-grade-summary-path",
        type=Path,
        default=DEFAULT_PAPER_GRADE_SUMMARY,
    )
    parser.add_argument(
        "--paper-grade-runs-path",
        type=Path,
        default=DEFAULT_PAPER_GRADE_RUNS,
    )

    parser.add_argument(
        "--skip-manuscript-campaign-check",
        action="store_true",
        help="Do not require docs/manuscript/conference_paper.md to reference --campaign-id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_path: Path = args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    steps: List[Dict[str, object]] = []
    gates: List[Dict[str, object]] = []
    artifacts: Dict[str, str] = {}

    status = "pass"
    error_detail = ""

    campaign_dir = args.campaign_root / args.campaign_id
    campaign_summary_path = campaign_dir / "closed_loop_summary.json"
    validation_report_path = campaign_dir / "validation" / "validation_report.json"

    try:
        if args.run_real_campaign:
            real_cmd = [
                sys.executable,
                "scripts/hpc/run_real_dft_campaign.py",
                "--candidate-csv",
                str(args.candidate_csv),
                "--iterations",
                str(args.iterations),
                "--top-k",
                str(args.top_k),
                "--max-workers",
                str(args.max_workers),
                "--max-retries",
                str(args.max_retries),
                "--request-prefix",
                args.request_prefix,
                "--classical-label-budget",
                str(args.classical_label_budget),
                "--campaign-id",
                args.campaign_id,
                "--tracking-uri",
                args.tracking_uri,
                "--experiment",
                "t5r4_real_campaign",
                "--queue-experiment",
                "dft_queue_runs_real",
                "--job-class",
                args.job_class,
                "--campaign-root",
                str(args.campaign_root),
            ]
            queue_tracking_uri = args.queue_tracking_uri or args.tracking_uri
            real_cmd.extend(["--queue-tracking-uri", queue_tracking_uri])
            if args.resume:
                real_cmd.append("--resume")
            if args.max_wall_seconds is not None:
                real_cmd.extend(["--max-wall-seconds", str(args.max_wall_seconds)])
            steps.append(_run_step("run_real_campaign", real_cmd).__dict__)
        else:
            _gate(
                gates,
                "campaign_exists",
                campaign_summary_path.exists(),
                f"expected existing campaign summary at {campaign_summary_path}",
            )

        campaign_summary = _load_json(campaign_summary_path, "campaign summary")
        artifacts["campaign_summary"] = str(campaign_summary_path)
        _validate_campaign_summary(
            campaign_summary,
            gates,
            min_label_efficiency=float(args.min_label_efficiency),
            min_valid_candidates=int(args.min_valid_candidates),
            max_failed_jobs=int(args.max_failed_jobs),
            expected_label_efficiency=args.expected_label_efficiency,
            expected_valid_candidates=args.expected_valid_candidates,
            expected_completed_jobs=args.expected_completed_jobs,
            expected_tolerance=float(args.expected_tolerance),
        )

        validate_cmd = [
            sys.executable,
            "scripts/dft/validate_production_outputs.py",
            "--campaign-id",
            args.campaign_id,
            "--campaign-root",
            str(args.campaign_root),
            "--report-path",
            str(validation_report_path),
            "--tracking-uri",
            args.tracking_uri,
        ]
        steps.append(_run_step("validate_production_outputs", validate_cmd).__dict__)

        validation_report = _load_json(validation_report_path, "DFT validation report")
        artifacts["validation_report"] = str(validation_report_path)
        validation_gap = validation_report.get("max_relative_gap")
        _gate(
            gates,
            "validation_status",
            validation_report.get("status") == "pass"
            or (args.allow_validation_review and validation_report.get("status") == "review"),
            f"status={validation_report.get('status')}, allow_review={args.allow_validation_review}",
        )
        _gate(
            gates,
            "validation_gap",
            (validation_gap is not None and float(validation_gap) <= float(args.max_validation_gap))
            or (args.allow_validation_review and validation_gap is None),
            (
                f"max_relative_gap={validation_gap}, required<={args.max_validation_gap} "
                f"(or null when allow_review={args.allow_validation_review})"
            ),
        )

        release_cmd = [
            sys.executable,
            "scripts/releases/create_real_dft_release.py",
            "--campaign-id",
            args.campaign_id,
            "--release-dir",
            str(args.release_dir),
        ]
        steps.append(_run_step("create_real_dft_release", release_cmd).__dict__)

        release_info_path = args.release_dir / "release_info.json"
        release_manifest_path = args.release_dir / "release_manifest.json"
        release_info = _load_json(release_info_path, "release info")
        _ = _load_json(release_manifest_path, "release manifest")
        artifacts["release_info"] = str(release_info_path)
        artifacts["release_manifest"] = str(release_manifest_path)
        _gate(
            gates,
            "release_campaign_match",
            release_info.get("campaign_id") == args.campaign_id,
            f"release campaign_id={release_info.get('campaign_id')}, expected={args.campaign_id}",
        )

        hardware_cmd = [
            sys.executable,
            "scripts/hardware/run_hardware_pilots.py",
            "--tracking-uri",
            args.tracking_uri,
        ]
        steps.append(_run_step("run_hardware_pilots", hardware_cmd).__dict__)

        hardware_summary_path = DATA_DIR / "hardware" / "hardware_summary.json"
        hardware_summary = _load_json(hardware_summary_path, "hardware summary")
        artifacts["hardware_summary"] = str(hardware_summary_path)
        hardware_runs = int(hardware_summary.get("totals", {}).get("hardware_runs", 0))
        _gate(
            gates,
            "hardware_runs",
            hardware_runs >= int(args.min_hardware_runs),
            f"hardware_runs={hardware_runs}, required>={args.min_hardware_runs}",
        )

        m7_cmd = [
            sys.executable,
            "scripts/benchmarking/run_m7_benchmarks.py",
            "--campaign-id",
            args.campaign_id,
            "--campaign-root",
            str(args.campaign_root),
            "--tracking-uri",
            args.tracking_uri,
        ]
        steps.append(_run_step("run_m7_benchmarks", m7_cmd).__dict__)

        m7_summary_path = DATA_DIR / "benchmarks" / "m7" / "m7_summary.json"
        m7_summary = _load_json(m7_summary_path, "M7 summary")
        artifacts["m7_summary"] = str(m7_summary_path)
        m7_metrics = m7_summary.get("metrics", {})
        qvc_gap = m7_metrics.get("quantum_vs_classical_gap")
        sensitivity = m7_metrics.get("sensitivity_index")
        _gate(
            gates,
            "m7_quantum_vs_classical_gap",
            qvc_gap is not None and float(qvc_gap) >= float(args.min_qvc_gap),
            f"quantum_vs_classical_gap={qvc_gap}, required>={args.min_qvc_gap}",
        )
        _gate(
            gates,
            "m7_sensitivity_present",
            sensitivity is not None,
            f"sensitivity_index={sensitivity}",
        )

        if args.profile == "paper-grade" and not args.skip_paper_grade_suite:
            paper_grade_cmd = [
                sys.executable,
                "scripts/benchmarking/run_paper_grade_gpu_suite.py",
                "--total-runs",
                str(args.paper_grade_total_runs),
                "--seed",
                str(args.paper_grade_seed),
                "--max-runtime-seconds",
                str(args.paper_grade_max_runtime_seconds),
                "--max-qsvr-relative-gap",
                str(args.paper_grade_max_qsvr_relative_gap),
                "--max-abs-qgpr-coverage-gap",
                str(args.paper_grade_max_abs_qgpr_coverage_gap),
                "--qgpr-engine",
                str(args.paper_grade_qgpr_engine),
                "--qgpr-optimizer",
                str(args.paper_grade_qgpr_optimizer),
                "--qgpr-catastrophic-coverage-gap",
                str(args.paper_grade_qgpr_catastrophic_coverage_gap),
                "--summary-path",
                str(args.paper_grade_summary_path),
                "--runs-path",
                str(args.paper_grade_runs_path),
            ]
            if args.paper_grade_fast:
                paper_grade_cmd.append("--fast")
            else:
                # Keep paper-grade behavior explicit and deterministic.
                paper_grade_cmd.append("--fastest")
            if args.paper_grade_qsvr_runs is not None:
                paper_grade_cmd.extend(["--qsvr-runs", str(args.paper_grade_qsvr_runs)])
            if args.paper_grade_qgpr_runs is not None:
                paper_grade_cmd.extend(["--qgpr-runs", str(args.paper_grade_qgpr_runs)])
            if args.paper_grade_classical_runs is not None:
                paper_grade_cmd.extend(["--classical-runs", str(args.paper_grade_classical_runs)])
            if args.paper_grade_enforce_runtime:
                paper_grade_cmd.append("--enforce-max-runtime")
            if args.paper_grade_allow_cpu_fallback:
                paper_grade_cmd.append("--allow-cpu-fallback")

            steps.append(_run_step("run_paper_grade_gpu_suite", paper_grade_cmd).__dict__)

            paper_grade_summary = _load_json(args.paper_grade_summary_path, "paper-grade suite summary")
            artifacts["paper_grade_summary"] = str(args.paper_grade_summary_path)
            artifacts["paper_grade_runs_csv"] = str(args.paper_grade_runs_path)
            execution_block = paper_grade_summary.get("execution", {})
            accelerator_block = paper_grade_summary.get("accelerator", {})
            completed_runs = execution_block.get("total_runs_completed")
            effective_accelerator = accelerator_block.get("effective")
            _gate(
                gates,
                "paper_grade_status",
                paper_grade_summary.get("status") == "pass",
                f"status={paper_grade_summary.get('status')}",
            )
            _gate(
                gates,
                "paper_grade_run_count",
                completed_runs is not None and int(completed_runs) >= int(args.paper_grade_total_runs),
                f"total_runs_completed={completed_runs}, required>={args.paper_grade_total_runs}",
            )
            if not args.paper_grade_allow_cpu_fallback:
                _gate(
                    gates,
                    "paper_grade_gpu_required",
                    effective_accelerator == "gpu",
                    f"accelerator_effective={effective_accelerator}",
                )

        publication_cmd = [
            sys.executable,
            "scripts/publication/validate_publication_build.py",
            "--campaign-id",
            args.campaign_id,
            "--campaign-root",
            str(args.campaign_root),
        ]
        steps.append(_run_step("validate_publication_build", publication_cmd).__dict__)

        publication_report_path = DOCS_DIR / "submission" / "build" / "publication_build_report.json"
        publication_report = _load_json(publication_report_path, "publication build report")
        artifacts["publication_build_report"] = str(publication_report_path)
        missing_figures = publication_report.get("steps", {}).get("figure_regeneration", {}).get("missing_figures", [])
        _gate(
            gates,
            "publication_status",
            publication_report.get("status") == "pass",
            f"status={publication_report.get('status')}",
        )
        _gate(
            gates,
            "publication_missing_figures",
            not missing_figures,
            f"missing_figures={missing_figures}",
        )

        repro_cmd = [
            sys.executable,
            "scripts/repro/run_repro_check.py",
            "--tracking-uri",
            args.tracking_uri,
        ]
        steps.append(_run_step("run_repro_check", repro_cmd).__dict__)

        repro_report_path = DATA_DIR / "reproducibility" / "reproduction_report.json"
        repro_report = _load_json(repro_report_path, "reproducibility report")
        artifacts["reproduction_report"] = str(repro_report_path)
        _gate(
            gates,
            "reproduction_success",
            bool(repro_report.get("reproduction_success")),
            f"reproduction_success={repro_report.get('reproduction_success')}",
        )

        if not args.skip_manuscript_campaign_check:
            paper_path = DOCS_DIR / "manuscript" / "conference_paper.md"
            manuscript_text = paper_path.read_text(encoding="utf-8")
            _gate(
                gates,
                "manuscript_campaign_reference",
                args.campaign_id in manuscript_text,
                f"campaign_id={args.campaign_id} must appear in {paper_path}",
            )
            artifacts["conference_paper"] = str(paper_path)

    except PipelineError as exc:
        status = "fail"
        error_detail = str(exc)
    except Exception as exc:  # pragma: no cover - defensive catch for runtime diagnostics
        status = "error"
        error_detail = f"{exc}\n{traceback.format_exc()}"

    report_payload = {
        "timestamp_utc": _utc_now(),
        "status": status,
        "campaign_id": args.campaign_id,
        "configuration": {
            "profile": args.profile,
            "run_real_campaign": bool(args.run_real_campaign),
            "tracking_uri": args.tracking_uri,
            "queue_tracking_uri": args.queue_tracking_uri or args.tracking_uri,
            "campaign_root": str(args.campaign_root),
            "release_dir": str(args.release_dir),
            "report_path": str(report_path),
            "allow_validation_review": bool(args.allow_validation_review),
            "skip_paper_grade_suite": bool(args.skip_paper_grade_suite),
            "paper_grade_speed_profile": "fast" if args.paper_grade_fast else "fastest",
            "paper_grade_total_runs": args.paper_grade_total_runs,
            "paper_grade_max_runtime_seconds": args.paper_grade_max_runtime_seconds,
            "paper_grade_enforce_runtime": bool(args.paper_grade_enforce_runtime),
            "paper_grade_allow_cpu_fallback": bool(args.paper_grade_allow_cpu_fallback),
            "paper_grade_max_qsvr_relative_gap": args.paper_grade_max_qsvr_relative_gap,
            "paper_grade_max_abs_qgpr_coverage_gap": args.paper_grade_max_abs_qgpr_coverage_gap,
            "paper_grade_qgpr_engine": args.paper_grade_qgpr_engine,
            "paper_grade_qgpr_optimizer": args.paper_grade_qgpr_optimizer,
            "paper_grade_qgpr_catastrophic_coverage_gap": args.paper_grade_qgpr_catastrophic_coverage_gap,
            "expected_label_efficiency": args.expected_label_efficiency,
            "expected_valid_candidates": args.expected_valid_candidates,
            "expected_completed_jobs": args.expected_completed_jobs,
        },
        "steps": steps,
        "gates": gates,
        "artifacts": artifacts,
        "error": error_detail,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": status,
                "report_path": str(report_path),
                "campaign_id": args.campaign_id,
                "failed_gate_count": len([g for g in gates if not g["passed"]]),
            },
            indent=2,
        )
    )

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
