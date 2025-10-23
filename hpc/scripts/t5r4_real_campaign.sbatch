#!/bin/bash
#SBATCH --job-name=t5r4-campaign
#SBATCH --account=hpcintro
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load gcc/9.4.0
module load quantum-espresso/gcc/9.4.0/openmpi/4.1.1/zen2/6.8

PROJECT_ROOT=/scratch/zt1/project/hpcintro/user/otailor/projects/Generative_QML
VENV_PATH=${PROJECT_ROOT}/.venv
PSEUDO_ROOT=/scratch/zt1/project/hpcintro/user/otailor/pseudos/SSSP_PBE
TRACKING_ROOT=${PROJECT_ROOT}/tracking/mlflow/campaigns
NTASKS=${SLURM_NTASKS:-16}

source "${VENV_PATH}/bin/activate"
export QE_PSEUDO_DIR="${PSEUDO_ROOT}"
export ESPRESSO_PSEUDO="${PSEUDO_ROOT}"
export MLFLOW_TRACKING_URI="file://${TRACKING_ROOT}"
export SCRATCH=${SLURM_SCRATCH:-/scratch/zt1/project/hpcintro/user/otailor/scratch}
export QE_PW_COMMAND="srun --mpi=pmix -n ${NTASKS} --cpu-bind=cores --kill-on-bad-exit=1 $(which pw.x)"

cd "${PROJECT_ROOT}"
mkdir -p logs "${TRACKING_ROOT}"

CAMPAIGN_ID=t5r4-${SLURM_JOB_ID:-$(date +%Y%m%dT%H%M%S)}

python scripts/hpc/run_real_dft_campaign.py \
  --candidate-csv data/qml/qgan_conditioned_candidates.csv \
  --iterations 4 \
  --top-k 4 \
  --max-workers ${NTASKS} \
  --max-retries 2 \
  --request-prefix REALCAM \
  --classical-label-budget 120 \
  --campaign-id "${CAMPAIGN_ID}" \
  --tracking-uri "${MLFLOW_TRACKING_URI}" \
  --experiment t5r4_real_campaign \
  --queue-experiment dft_queue_runs_real \
  --queue-tracking-uri "${MLFLOW_TRACKING_URI}" \
  --max-force-threshold 0.05 \
  --random-state 314159
