---
entry_id: MDIA-LAB-20251001-001
entry_type: lab_notebook
task_id: T5R.1
milestone_id: M5-real
owning_agent: MDIA
collaborators: [ALOA, RKMA]
dataset_version: 1.0.0
code_revision: pending-ci-integration
mlflow_run_id: 09768ff36de0475eaff97e55765d3ea0
qpu_backend: 
submission_status: pending
reviewer_comments: 
timestamp_utc: 2025-10-01T21:45:00Z
---

## Objective
Commission the production Quantum ESPRESSO stack, capture runtime/resource metrics for `QAL-0001`, and assemble the HPC readiness bundle required by T5R.1.

## Experimental Setup
- **QE Binary**: `/Users/omtailor/Downloads/qe-7.4.1/build/bin/pw.x` (exported via `QE_PW_COMMAND`).
- **Pseudopotentials**: SSSP Efficiency PBE 1.3 (`ESPRESSO_PSEUDO=$HOME/qe/pseudos/SSSP_PBE`).
- **Inputs**: `data/dft_handoff/input/QAL-0001/structure.cif` with metadata describing `Al0.25Co1Fe1Ni1`.
- **Scripts**: `scripts/dft/qe_benchmark.py`, `scripts/dft/storage_planner.py`.
- **Tracking**: MLflow experiment `dft_benchmarks` (`tracking/mlflow/benchmarks`).

## Procedure
1. Activated project virtualenv (`source .venv/bin/activate`) and confirmed QE environment exports from `.zshrc`.
2. Executed `python scripts/dft/qe_benchmark.py --request-id QAL-0001`, allowing QE 7.4.1 to run the SCF calculation end-to-end.
3. Logged artefacts and metrics to MLflow run `09768ff36de0475eaff97e55765d3ea0` (tag `task:T5R.1`, `job_label: scf_screen`).
4. Generated storage plan using `python scripts/dft/storage_planner.py --request-id QAL-0001 --runs 12 --safety-factor 1.5`, producing MLflow run `318efd3514084c09b594307a883a2b69` and aggregated forecast `docs/hpc/storage_forecast_plan.json`.
5. Collected supporting documentation: `docs/hpc/qe_environment.md`, `docs/hpc/qe_slurm_template.sh`, and `docs/hpc/README.md`.

## Results
- **Benchmark Metrics (scf_screen)**: `wall_time_s ≈ 13.34`, `cpu_time_s ≈ 13.09`, `max_rss_bytes ≈ 2.01e8`, `qe_output_gib = 0.0166` (see `data/dft_workflow/QAL-0001/benchmark_summary.json`).
- **Storage Forecast**: For 12 runs @ 1.5× buffer → `total_gib = 0.299`, estimated total wall-time `≈0.066 h` (artefact `storage_forecast.json`, MLflow run `318efd3514084c09b594307a883a2b69`).
- **Artefacts**: QE outputs and summaries attached under MLflow artefact paths (`qe_output/`, `storage_forecast/`).

## Acceptance Check
- `mdia.qe_benchmark_ready = 1.0` recorded in MLflow (run `09768ff36de0475eaff97e55765d3ea0`).
- HPC readiness bundle assembled (benchmark summary, storage forecast, scheduler template, environment spec). T5R.1 acceptance criterion satisfied.

## Next Steps
- Integrate production QE workflow into the active-learning orchestrator (T5R.2) with async queue management.
- Prepare additional benchmark handoffs for `vc_relax` and `elastic_eval` job classes before submitting HPC allocation request (use new multi-job config support in `storage_planner.py`).
- Share benchmark + storage artefacts with PDA for HPC allocation dossier and request reviewer sign-off (`submission_status → approved`).
