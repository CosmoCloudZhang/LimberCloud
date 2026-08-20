#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o logs/%x_%j.out
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
#SBATCH -J COVARIANCE_Y10_MATRIX
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load conda
module load cray-mpich
module load PrgEnv-gnu
module load cray-hdf5-parallel

# Activate the conda environment
source "${HOME}/.bashrc"
conda activate "${CosmoENV}"

# Set environment
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Initialize the process
TAG="Y10"
REPO_ROOT="${LIMBERCLOUD_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
RUNTIME_ROOT="${LIMBERCLOUD_RUNTIME_ROOT:?Set LIMBERCLOUD_RUNTIME_ROOT to the external data/results root}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LIMBERCLOUD_LAYOUT="${LIMBERCLOUD_LAYOUT:-legacy}"
ONECOVARIANCE_SCRIPT="${ONECOVARIANCE_SCRIPT:?Set ONECOVARIANCE_SCRIPT to covariance.py}"

if [ "${LIMBERCLOUD_LAYOUT}" = "canonical" ]; then
    COVARIANCE_CONFIG="${RUNTIME_ROOT}/results/covariance/${TAG}/CONFIG.ini"
else
    COVARIANCE_CONFIG="${RUNTIME_ROOT}/COVARIANCE/${TAG}/CONFIG.ini"
fi

# Run applications
python -u "${REPO_ROOT}/experiments/covariance/${TAG}/matrix.py" --tag="${TAG}" --folder="${RUNTIME_ROOT}" &&
srun -u -N 1 -n 1 -c "${SLURM_CPUS_PER_TASK}" python "${ONECOVARIANCE_SCRIPT}" "${COVARIANCE_CONFIG}"
