#!/bin/bash
#SBATCH --job-name=zaratan-scf
#SBATCH --account=energybio
#SBATCH --partition=cpu-quad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load gcc/9.4.0
module load python/3.8.12/gcc/9.4.0/nocuda
module load quantum-espresso/gcc/9.4.0/openmpi/4.1.1/zen2/6.8

source /scratch/zt1/project/energybio/user/otailor/projects/Generative_QML/.venv/bin/activate
export QE_PSEUDO_DIR=/scratch/zt1/project/energybio/user/otailor/pseudos/SSSP_PBE
export MLFLOW_TRACKING_URI=file:///scratch/zt1/project/energybio/user/otailor/projects/Generative_QML/tracking/mlflow/benchmarks
export SCRATCH=${SLURM_SCRATCH:-/scratch/zt1/project/energybio/user/otailor/scratch}

cd /scratch/zt1/project/energybio/user/otailor/projects/Generative_QML
mkdir -p logs

python scripts/dft/qe_benchmark.py \
  --request-id ZAR-SMOKE-001 \
  --experiment zaratan_dft_benchmarks
