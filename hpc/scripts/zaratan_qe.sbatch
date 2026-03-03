#!/bin/bash
#SBATCH --job-name=zaratan-scf
#SBATCH --partition=standard
#SBATCH --account=hpcintro-aac
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load gcc/9.4.0
module load quantum-espresso/gcc/9.4.0/openmpi/4.1.1/zen2/6.8

source /scratch/zt1/project/hpcintro/user/otailor/projects/Generative_QML/.venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export QE_PSEUDO_DIR=/scratch/zt1/project/hpcintro/user/otailor/projects/SSSP_PBE
export ESPRESSO_TMPDIR=/scratch/zt1/project/hpcintro/user/otailor/tmp_qe
mkdir -p "$ESPRESSO_TMPDIR"
NTASKS=${SLURM_NTASKS:-$(( ${SLURM_NTASKS_PER_NODE:-16} * ${SLURM_JOB_NUM_NODES:-1} ))}
export QE_PW_COMMAND="srun --mpi=pmix -n ${NTASKS} --cpu-bind=cores --kill-on-bad-exit=1 $(which pw.x)"
export MLFLOW_TRACKING_URI=file:///scratch/zt1/project/hpcintro/user/otailor/projects/Generative_QML/tracking/mlflow/benchmarks
export SCRATCH=${SLURM_SCRATCH:-/scratch/zt1/project/hpcintro/user/otailor/scratch}

cd /scratch/zt1/project/hpcintro/user/otailor/projects/Generative_QML
mkdir -p logs
echo "NTASKS=${NTASKS} CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "PW_COMMAND: ${QE_PW_COMMAND}"

python scripts/dft/qe_benchmark.py \
  --request-id ZAR-SMOKE-001 \
  --experiment zaratan_dft_benchmarks \
  --progress-interval 15
