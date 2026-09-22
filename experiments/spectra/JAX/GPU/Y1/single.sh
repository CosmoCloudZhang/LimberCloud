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
#SBATCH -J JAX_GPU_Y1_Single
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

set -eo pipefail

# Configure the project environment
_lc_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/load_config.sh" && -d "${SLURM_SUBMIT_DIR}/src/limbercloud" ]]; then
    PROJECT_ROOT=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd -P)
else
    _walk=${_lc_dir}
    while [[ ${_walk} != "/" ]]; do
        if [[ -f "${_walk}/scripts/load_config.sh" && -d "${_walk}/src/limbercloud" ]]; then
            PROJECT_ROOT=${_walk}
            break
        fi
        _walk=$(dirname -- "${_walk}")
    done
fi
[[ -n ${PROJECT_ROOT} ]] || { echo "LimberCloud error: could not resolve PROJECT_ROOT" >&2; exit 1; }
unset _lc_dir _walk
source "${PROJECT_ROOT}/scripts/load_config.sh"
source "${PROJECT_ROOT}/scripts/nersc/modules/gpu.sh"
source "${PROJECT_ROOT}/scripts/nersc/activate_venv.sh"

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
TAG="Y1"
LABEL="Single"
SCRIPT="single"
RUNTIME_ROOT="${LIMBERCLOUD_RUNTIME_ROOT:?Set LIMBERCLOUD_RUNTIME_ROOT to the external data/results root}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Run the script
srun -n 1 -c $SLURM_CPUS_PER_TASK -G 1 python -u "${PROJECT_ROOT}/experiments/spectra/JAX/GPU/${TAG}/${SCRIPT}.py" --tag="${TAG}" --label="${LABEL}" --folder="${RUNTIME_ROOT}" "$@"
