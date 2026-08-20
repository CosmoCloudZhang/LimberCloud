#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --mail-type=END
#SBATCH --time=04:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH -J JAX_GPU_Y10_Single
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load gpu
module load conda
module load cray-mpich
module load PrgEnv-gnu
module load cray-hdf5-parallel

# Activate the conda environment
source "${HOME}/.bashrc"
conda activate "${CosmoENV}"

# Environment variables
export JAX_PLATFORMS=cuda
export SLURM_CPU_BIND=cores

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Paths
TAG="Y10"
LABEL="Single"
REPO_ROOT="${LIMBERCLOUD_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
RUNTIME_ROOT="${LIMBERCLOUD_RUNTIME_ROOT:?Set LIMBERCLOUD_RUNTIME_ROOT to the external data/results root}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Run the script
srun -n 1 -c $SLURM_CPUS_PER_TASK -G 1 python -u "${REPO_ROOT}/experiments/spectra/JAX/GPU/${TAG}/${LABEL,,}.py" --tag="${TAG}" --path="${REPO_ROOT}" --label="${LABEL}" --folder="${RUNTIME_ROOT}" --number="${SLURM_CPUS_PER_TASK}"
