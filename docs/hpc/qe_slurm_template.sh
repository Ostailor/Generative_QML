#!/bin/bash
# Quantum ESPRESSO SLURM submission template for T5R.1 production runs.
#
# Customize account/partition/QOS to match the target cluster. The script is
# designed to be rendered programmatically so the active learning loop can
# submit batches with different k-point meshes or candidate identifiers.

#SBATCH --job-name=qe_qal_${REQUEST_ID:-QAL-0001}
#SBATCH --account=<PROJECT_ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=0
#SBATCH --mem=120G
#SBATCH --output=${SCRATCH:-$PWD}/logs/${REQUEST_ID:-QAL-0001}_%j.out
#SBATCH --error=${SCRATCH:-$PWD}/logs/${REQUEST_ID:-QAL-0001}_%j.err

module purge
module load gcc/12.1.0 openmpi/4.1.6 quantum-espresso/7.4.1

# Use project scratch for large wavefunction files to avoid home directory quotas.
export ESPRESSO_TMPDIR=${SCRATCH:-$PWD}/qe_tmp/${REQUEST_ID:-QAL-0001}
mkdir -p "$ESPRESSO_TMPDIR"

# Ensure pseudopotential directory is available on compute nodes.
export ESPRESSO_PSEUDO=${ESPRESSO_PSEUDO:-/path/to/pseudos/SSSP_PBE}

INPUT_FILE=${INPUT_FILE:-${PWD}/data/dft_handoff/input/${REQUEST_ID:-QAL-0001}/espresso.pwi}
OUTPUT_DIR=${OUTPUT_DIR:-${PWD}/data/dft_workflow/${REQUEST_ID:-QAL-0001}}
mkdir -p "$OUTPUT_DIR"

date --utc '+%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT_DIR/start_timestamp.txt"

srun pw.x -in "$INPUT_FILE" > "$OUTPUT_DIR/pw_${REQUEST_ID:-QAL-0001}.pwo"
EXIT_CODE=$?

date --utc '+%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT_DIR/end_timestamp.txt"

exit $EXIT_CODE
