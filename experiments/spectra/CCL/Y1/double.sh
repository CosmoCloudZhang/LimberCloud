#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o logs/%x_%j.out
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH -J PYTHON_CCL_Y1_Double
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

set -eo pipefail

# Configure the project environment
REPO_ROOT="${LIMBERCLOUD_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${REPO_ROOT}/scripts/nersc/load_environment.sh"
source "${REPO_ROOT}/scripts/nersc/modules/cpu.sh"
conda activate "${LIMBERCLOUD_CONDA_ENV}"

# Set environment
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Initialize the process
TAG="Y1"
LABEL="Double"
RUNTIME_ROOT="${LIMBERCLOUD_RUNTIME_ROOT:?Set LIMBERCLOUD_RUNTIME_ROOT to the external data/results root}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Run applications
srun -n 1 -c $SLURM_CPUS_PER_TASK python -u "${REPO_ROOT}/experiments/spectra/CCL/${TAG}/${LABEL,,}.py" --tag="${TAG}" --path="${REPO_ROOT}" --label="${LABEL}" --folder="${RUNTIME_ROOT}" --number="${SLURM_CPUS_PER_TASK}"
