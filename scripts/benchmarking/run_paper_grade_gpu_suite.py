#!/usr/bin/env python3
"""Execute the paper-grade GPU benchmark suite with fail-fast gates.

The suite repeatedly executes QSVR, QGPR, and classical AL baselines to build
high-sample-count benchmark evidence for paper-grade reporting.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import pandas as pd

try:
    from scripts.accel import detect_accelerator
    from scripts.classical_al_baselines import run_simulation
    from scripts.qgpr_benchmark import evaluate as evaluate_qgpr
    from scripts.qsvr_benchmark import evaluate_models as evaluate_qsvr
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))

    from accel import detect_accelerator
    from classical_al_baselines import run_simulation
    from qgpr_benchmark import evaluate as evaluate_qgpr
    from qsvr_benchmark import evaluate_models as evaluate_qsvr

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "data" / "benchmarks" / "paper_grade"
DEFAULT_SUMMARY = OUT_DIR / "paper_grade_suite_summary.json"
DEFAULT_RUNS = OUT_DIR / "paper_grade_suite_runs.csv"

FAST_PRESET = {
    "qsvr_max_train": 6000,
    "qsvr_max_test": 2000,
    "qgpr_max_train": 1200,
    "qgpr_max_test": 300,
    "classical_iterations": 6,
    "classical_pool_subsample": 4000,
    "classical_max_eval_size": 200,
    "classical_query_batch": 25,
}

FASTEST_PRESET = {
    "qsvr_max_train": 3000,
    "qsvr_max_test": 1000,
    "qgpr_max_train": 800,
    "qgpr_max_test": 200,
    "classical_iterations": 3,
    "classical_pool_subsample": 1000,
    "classical_max_eval_size": 96,
    "classical_query_batch": 16,
}


class SuiteError(RuntimeError):
    """Raised on failed paper-grade checks."""


@dataclass
class RunRecord:
    model: str
    run_index: int
    seed: int
    metric_primary: float
    metric_secondary: float
    runtime_s: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "run_index": self.run_index,
            "seed": self.seed,
            "metric_primary": self.metric_primary,
            "metric_secondary": self.metric_secondary,
            "runtime_s": self.runtime_s,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _validate_split(total_runs: int, qsvr_runs: int | None, qgpr_runs: int | None, classical_runs: int | None) -> tuple[int, int, int]:
    provided = [qsvr_runs, qgpr_runs, classical_runs]
    if all(v is not None for v in provided):
        qsvr_n = int(qsvr_runs)
        qgpr_n = int(qgpr_runs)
        classical_n = int(classical_runs)
        if qsvr_n + qgpr_n + classical_n != total_runs:
            raise SuiteError(
                "qsvr-runs + qgpr-runs + classical-runs must equal total-runs "
                f"({qsvr_n}+{qgpr_n}+{classical_n}!={total_runs})"
            )
        return qsvr_n, qgpr_n, classical_n

    # Default split for paper-grade plan.
    qsvr_n = total_runs // 3
    qgpr_n = total_runs // 3
    classical_n = total_runs - qsvr_n - qgpr_n
    return qsvr_n, qgpr_n, classical_n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-grade GPU benchmark suite.")
    parser.add_argument("--total-runs", type=int, default=1100)
    parser.add_argument("--qsvr-runs", type=int, default=None)
    parser.add_argument("--qgpr-runs", type=int, default=None)
    parser.add_argument("--classical-runs", type=int, default=None)

    parser.add_argument("--seed", type=int, default=20260302)

    parser.add_argument("--max-runtime-seconds", type=float, default=7200.0)
    parser.set_defaults(enforce_max_runtime=True)
    parser.add_argument(
        "--enforce-max-runtime",
        dest="enforce_max_runtime",
        action="store_true",
        help="Enforce the max-runtime gate (enabled by default).",
    )
    parser.add_argument(
        "--skip-max-runtime-gate",
        dest="enforce_max_runtime",
        action="store_false",
        help="Disable max-runtime gating.",
    )

    parser.add_argument("--require-gpu", action="store_true", default=True)
    parser.add_argument("--allow-cpu-fallback", action="store_true")

    parser.add_argument("--max-qsvr-relative-gap", type=float, default=0.10)
    parser.add_argument("--max-abs-qgpr-coverage-gap", type=float, default=0.05)

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Speed preset tuned for stronger per-run coverage (slower than --fastest).",
    )
    parser.add_argument(
        "--fastest",
        action="store_true",
        help="Aggressive speed preset (default behavior when no speed flag is provided).",
    )

    parser.add_argument("--qsvr-max-train", type=int, default=None)
    parser.add_argument("--qsvr-max-test", type=int, default=None)

    parser.add_argument("--qgpr-max-train", type=int, default=None)
    parser.add_argument("--qgpr-max-test", type=int, default=None)

    parser.add_argument("--classical-iterations", type=int, default=None)
    parser.add_argument("--classical-pool-subsample", type=int, default=None)
    parser.add_argument("--classical-max-eval-size", type=int, default=None)
    parser.add_argument("--classical-query-batch", type=int, default=None)

    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--runs-path", type=Path, default=DEFAULT_RUNS)
    return parser.parse_args()


def _resolve_speed_profile(args: argparse.Namespace) -> str:
    if args.fast and args.fastest:
        raise SuiteError("Use only one speed preset: --fast or --fastest.")

    speed_profile = "fast" if args.fast else "fastest"
    preset = FAST_PRESET if speed_profile == "fast" else FASTEST_PRESET

    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return speed_profile


def _check_gpu(args: argparse.Namespace) -> Dict[str, str]:
    requested = "gpu" if (args.require_gpu and not args.allow_cpu_fallback) else "auto"
    effective, reason = detect_accelerator(requested)
    if args.require_gpu and not args.allow_cpu_fallback and effective != "gpu":
        raise SuiteError(
            "Paper-grade suite requires GPU but backend resolved to CPU "
            f"(reason={reason})."
        )
    return {"requested": requested, "effective": effective, "reason": reason}


def _run_qsvr(args: argparse.Namespace, seed: int, run_index: int, accelerator_mode: str) -> RunRecord:
    t0 = time.perf_counter()
    metrics = evaluate_qsvr(
        random_state=seed,
        max_train=args.qsvr_max_train,
        max_test=args.qsvr_max_test,
        use_float32=True,
        accelerator=accelerator_mode,
    )
    dt = time.perf_counter() - t0
    relative_gap = float(metrics["relative_gap"])
    rmse_quantum = float(metrics["rmse_quantum"])
    if relative_gap > args.max_qsvr_relative_gap:
        raise SuiteError(
            f"QSVR run {run_index} failed gap gate: relative_gap={relative_gap:.6f} > {args.max_qsvr_relative_gap:.6f}"
        )
    return RunRecord("qsvr", run_index, seed, relative_gap, rmse_quantum, dt)


def _run_qgpr(args: argparse.Namespace, seed: int, run_index: int, accelerator_mode: str) -> RunRecord:
    t0 = time.perf_counter()
    metrics = evaluate_qgpr(
        random_state=seed,
        max_train=args.qgpr_max_train,
        max_test=args.qgpr_max_test,
        optimizer="none",
        use_float32=True,
        accelerator=accelerator_mode,
        gpr_engine="backend",
    )
    dt = time.perf_counter() - t0
    coverage_gap = float(metrics["coverage_gap"])
    rmse_quantum = float(metrics["rmse_quantum"])
    if abs(coverage_gap) > args.max_abs_qgpr_coverage_gap:
        raise SuiteError(
            f"QGPR run {run_index} failed coverage gate: abs(coverage_gap)={abs(coverage_gap):.6f} > {args.max_abs_qgpr_coverage_gap:.6f}"
        )
    return RunRecord("qgpr", run_index, seed, coverage_gap, rmse_quantum, dt)


def _run_classical(args: argparse.Namespace, seed: int, run_index: int, accelerator_mode: str) -> RunRecord:
    t0 = time.perf_counter()
    metrics = run_simulation(
        random_state=seed,
        query_batch=args.classical_query_batch,
        iterations=args.classical_iterations,
        pool_subsample=args.classical_pool_subsample,
        optimizer="none",
        max_eval_size=args.classical_max_eval_size,
        accelerator=accelerator_mode,
        gp_engine="backend",
        use_float32=True,
    )
    dt = time.perf_counter() - t0
    final_rmse = float(metrics["final_rmse"])
    iterations = float(metrics["iterations"])
    if not math.isfinite(final_rmse):
        raise SuiteError(f"Classical AL run {run_index} produced non-finite RMSE")
    return RunRecord("classical_al", run_index, seed, final_rmse, iterations, dt)


def main() -> None:
    args = parse_args()
    speed_profile = _resolve_speed_profile(args)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.runs_path.parent.mkdir(parents=True, exist_ok=True)

    if args.total_runs <= 0:
        raise SystemExit("--total-runs must be > 0")

    accelerator = _check_gpu(args)
    mode_for_models = accelerator["effective"]

    qsvr_n, qgpr_n, classical_n = _validate_split(
        total_runs=int(args.total_runs),
        qsvr_runs=args.qsvr_runs,
        qgpr_runs=args.qgpr_runs,
        classical_runs=args.classical_runs,
    )

    all_records: List[RunRecord] = []
    suite_start = time.perf_counter()
    started_utc = _utc_now()

    # QSVR block
    for idx in range(1, qsvr_n + 1):
        seed = int(args.seed + idx)
        all_records.append(_run_qsvr(args, seed, idx, mode_for_models))
        if idx % 25 == 0:
            print(f"[progress] qsvr {idx}/{qsvr_n}", flush=True)

    # QGPR block
    for idx in range(1, qgpr_n + 1):
        seed = int(args.seed + 10000 + idx)
        all_records.append(_run_qgpr(args, seed, idx, mode_for_models))
        if idx % 25 == 0:
            print(f"[progress] qgpr {idx}/{qgpr_n}", flush=True)

    # Classical AL block
    for idx in range(1, classical_n + 1):
        seed = int(args.seed + 20000 + idx)
        all_records.append(_run_classical(args, seed, idx, mode_for_models))
        if idx % 25 == 0:
            print(f"[progress] classical_al {idx}/{classical_n}", flush=True)

    total_runtime = time.perf_counter() - suite_start

    if args.enforce_max_runtime and total_runtime > float(args.max_runtime_seconds):
        raise SuiteError(
            f"Paper-grade suite exceeded runtime budget: {total_runtime:.2f}s > {args.max_runtime_seconds:.2f}s"
        )

    runs_df = pd.DataFrame([r.as_dict() for r in all_records])
    runs_df.to_csv(args.runs_path, index=False)

    per_model: Dict[str, Dict[str, object]] = {}
    for model in sorted(runs_df["model"].unique()):
        sub = runs_df[runs_df["model"] == model]
        per_model[model] = {
            "runs": int(len(sub)),
            "metric_primary": _stats(sub["metric_primary"].astype(float).tolist()),
            "metric_secondary": _stats(sub["metric_secondary"].astype(float).tolist()),
            "runtime_s": _stats(sub["runtime_s"].astype(float).tolist()),
        }

    summary = {
        "timestamp_utc": _utc_now(),
        "started_utc": started_utc,
        "status": "pass",
        "accelerator": accelerator,
        "plan": {
            "speed_profile": speed_profile,
            "total_runs": int(args.total_runs),
            "qsvr_runs": qsvr_n,
            "qgpr_runs": qgpr_n,
            "classical_runs": classical_n,
            "qsvr_max_train": int(args.qsvr_max_train),
            "qsvr_max_test": int(args.qsvr_max_test),
            "qgpr_max_train": int(args.qgpr_max_train),
            "qgpr_max_test": int(args.qgpr_max_test),
            "classical_iterations": int(args.classical_iterations),
            "classical_pool_subsample": int(args.classical_pool_subsample),
            "classical_max_eval_size": int(args.classical_max_eval_size),
            "classical_query_batch": int(args.classical_query_batch),
            "max_runtime_seconds": float(args.max_runtime_seconds),
            "enforce_max_runtime": bool(args.enforce_max_runtime),
        },
        "execution": {
            "total_runs_completed": int(len(all_records)),
            "total_runtime_seconds": float(total_runtime),
            "runs_per_second": float(len(all_records) / total_runtime) if total_runtime > 0 else math.nan,
        },
        "gates": {
            "qsvr_relative_gap_max": float(args.max_qsvr_relative_gap),
            "qgpr_abs_coverage_gap_max": float(args.max_abs_qgpr_coverage_gap),
        },
        "artifacts": {
            "runs_csv": str(args.runs_path),
            "summary_json": str(args.summary_path),
        },
        "metrics": per_model,
    }

    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "pass",
                "summary": str(args.summary_path),
                "runs_csv": str(args.runs_path),
                "total_runs_completed": len(all_records),
                "total_runtime_seconds": total_runtime,
                "speed_profile": speed_profile,
                "accelerator_effective": accelerator["effective"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
