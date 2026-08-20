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
#SBATCH -J JAX_CPU_Y10_Single
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
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK"

# Initialize the process
TAG="Y10"
LABEL="Single"
REPO_ROOT="${LIMBERCLOUD_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
RUNTIME_ROOT="${LIMBERCLOUD_RUNTIME_ROOT:?Set LIMBERCLOUD_RUNTIME_ROOT to the external data/results root}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Run applications
srun -n 1 -c $SLURM_CPUS_PER_TASK python -u "${REPO_ROOT}/experiments/spectra/JAX/CPU/${TAG}/${LABEL,,}.py" --tag="${TAG}" --path="${REPO_ROOT}" --label="${LABEL}" --folder="${RUNTIME_ROOT}" --number="${SLURM_CPUS_PER_TASK}"
