# Quantum ESPRESSO Environment Specification (T5R.1)

- **QE Version**: 7.4.1 (compiled with OpenMPI 4.1.x and GCC ≥ 11)
- **Primary Executable**: `pw.x` (available on PATH or via `QE_PW_COMMAND`)
- **Pseudopotentials**: SSSP Efficiency PBE 1.3 (
  `ESPRESSO_PSEUDO=$HOME/qe/pseudos/SSSP_PBE`). The workflow also honours
  `QE_PSEUDO_DIR` for portability across clusters.
- **Python Orchestration Env**:
  - Python 3.11
  - Packages: `ase>=3.23`, `mlflow>=2.9`, `psutil>=5.9`, `numpy>=1.24`
  - Optional: `mpi4py` if the cluster requires Python-driven MPI launches
- **Environment Variables**:
  - `QE_PW_COMMAND` *(optional)*: Absolute path to `pw.x`
  - `QE_PSEUDO_DIR` or `ESPRESSO_PSEUDO`: Directory containing `.upf` files
  - `MLFLOW_TRACKING_URI`: Tracking server URI (defaults to
    `tracking/mlflow/mlruns` when unset)
  - `MLFLOW_EXPERIMENT_NAME`: Overrides the default MLflow experiment name
  - `SCRATCH`: High-performance filesystem path for temporary QE data
- **Containers (optional)**: Singularity/Apptainer definition files can wrap the
  same environment; ensure the container exposes `pw.x` and pseudopotentials
  via bind mounts.

This specification underpins the T5R.1 benchmark scripts:

1. `scripts/dft/qe_benchmark.py` — executes or replays QE jobs while logging to
   MLflow.
2. `scripts/dft/storage_planner.py` — consumes benchmark artefacts to estimate
   storage and CPU budgets for the closed-loop campaign.
3. `docs/hpc/qe_slurm_template.sh` — SLURM submission template that mirrors the
   environment documented here.

Before requesting HPC allocations, validate the local build:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-preprocess.txt -r requirements-noise.txt ase mlflow psutil
python scripts/dft/qe_benchmark.py --request-id QAL-0001 --experiment dft_benchmarks
```

The MLflow run produced by the command above must contain:

- Runtime metrics (`wall_time_s`, `cpu_time_s`, `max_rss_bytes`)
- Disk usage (`qe_output_gib`)
- Artefacts (`qe_output/*`, `benchmark_summary.json`)

These artefacts form the core of the HPC readiness bundle for T5R.1.
