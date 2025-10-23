# HPC Readiness Bundle (Milestone M5-real / Task T5R.1)

Artifacts produced for the production DFT stack bring-up:

| Asset | Purpose |
| --- | --- |
| `scripts/dft/qe_benchmark.py` | Runs Quantum ESPRESSO locally, captures runtime and resource metrics, and logs artefacts/metrics to MLflow (`mdia.qe_benchmark_ready`). |
| `scripts/dft/storage_planner.py` | Aggregates benchmark outputs to project storage + CPU requirements for single or multi-class campaigns. |
| `docs/hpc/qe_slurm_template.sh` | SLURM template illustrating resource, scratch, and pseudopotential configuration for cluster submissions. |
| `docs/hpc/qe_environment.md` | Environment specification (QE version, pseudopotentials, Python deps, environment variables). |
| `docs/hpc/job_profile_plan.json` | Planned candidate counts per job class for full-scale runs (update after benchmarking). |
| `docs/hpc/storage_forecast_plan.json` | Latest aggregate storage/runtime forecast generated via `storage_planner.py`. |
| `scripts/hpc/run_real_dft_campaign.py` | Orchestrates the production T5R.4 active-learning + real DFT loop (batch handoff prep, queue execution, MLflow logging). |
| `hpc/scripts/t5r4_real_campaign.sbatch` | Cluster submission recipe that wraps `run_real_dft_campaign.py` with QE environment modules and project settings. |

## Benchmarking Workflow

1. Activate the project environment (ensure `pw.x` and SSSP pseudopotentials are on the workstation PATH, then `source .venv/bin/activate`).
2. Benchmark each required job class with `scripts/dft/qe_benchmark.py`, tagging the run via `--job-label`:
   ```bash
   python scripts/dft/qe_benchmark.py --request-id QAL-0001 --job-label scf_screen
   python scripts/dft/qe_benchmark.py --request-id BENCH-RELAX-0001 --job-label vc_relax
   python scripts/dft/qe_benchmark.py --request-id BENCH-ELASTIC-0001 --job-label elastic_eval
   ```
   Each command logs metrics/artefacts to MLflow (experiment `dft_benchmarks`) and writes the corresponding `benchmark_summary.json` under `data/dft_workflow/<request_id>/`.
3. Estimate storage + CPU scale-out:
   - Single-job quick check:
     ```bash
     python scripts/dft/storage_planner.py --request-id QAL-0001 --runs 12 --safety-factor 1.5
     ```
   - Full campaign aggregation using the multi-job config:
     ```bash
     python scripts/dft/storage_planner.py --config docs/hpc/job_profile_plan.json
     ```
   The planner produces per-job forecasts, updates `docs/hpc/storage_forecast_plan.json`, and logs aggregate metrics (e.g., `qe_storage_gib_total`) to MLflow.
4. Bundle artefacts (`benchmark_summary.json`, `storage_forecast.json`, `storage_forecast_plan.json`, MLflow run links, and the SLURM template) with the allocation request dossier.

All scripts default to the local MLflow file store at `tracking/mlflow/benchmarks` unless `MLFLOW_TRACKING_URI` is overridden.

## Job Profiles Required for T5R and HEA Discovery

| Job Label | Role in Pipeline | Handoff Directory | Notes |
| --- | --- | --- | --- |
| `scf_screen` | Fast SCF evaluations in the active-learning loop (T5R.2) before investing in expensive relaxations. | `data/dft_handoff/input/QAL-0001/` | Already benchmarked; provides baseline footprint for frequent AL calls. |
| `vc_relax` | Full variable-cell relaxations for shortlisted HEAs prior to property validation (feeds T5R.3 and manuscript figures). | `data/dft_handoff/input/BENCH-RELAX-0001/` | Run once to capture a representative relaxation cost; reuse metrics for final campaign planning. |
| `elastic_eval` | Post-relaxation elastic/strength estimation supporting HEA property claims. | `data/dft_handoff/input/BENCH-ELASTIC-0001/` | Benchmark after relaxations are validated to quantify high-cost analysis runs. |

Update `docs/hpc/job_profile_plan.json` with final counts per job type (e.g., number of AL iterations, validation relaxations, elastic analyses) and regenerate the aggregate forecast prior to submitting the HPC allocation and final manuscript.

## Running the T5R.4 Production Campaign on HPC

1. **Prepare environment**: mirror the QE stack described in `docs/hpc/qe_environment.md` on the target cluster (pseudopotentials, Python env, MLflow tracking path).
2. **Submit the campaign job**:
   ```bash
   sbatch hpc/scripts/t5r4_real_campaign.sbatch
   ```
   The script derives a unique campaign ID (`t5r4-<jobid>`), generates production handoff packages from `data/qml/qgan_conditioned_candidates.csv`, and launches the queue-backed DFT runs for each active-learning iteration.
3. **Monitor progress**: queue logs and MLflow metrics (including `aloa.real_dft_iterations` and `aloa.real_label_efficiency_gain`) are streamed under `data/dft_workflow/campaigns/<campaign_id>/iteration_*/`.
4. **Acceptance artefacts**: on completion the job writes
   - `data/dft_workflow/campaigns/<campaign_id>/candidate_library.csv`
   - `data/dft_workflow/campaigns/<campaign_id>/closed_loop_summary.json`
   - MLflow run in experiment `t5r4_real_campaign` with `aloa.real_label_efficiency_gain ≥ 0.30` and `valid_candidates ≥ 10`

These artefacts satisfy T5R.4 exit criteria and feed BRA/RKMA for downstream benchmarking and provenance updates.
