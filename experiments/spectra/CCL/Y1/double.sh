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
source "${PROJECT_ROOT}/scripts/nersc/modules/cpu.sh"
source "${PROJECT_ROOT}/scripts/nersc/activate_venv.sh"

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
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Run applications
srun -n 1 -c $SLURM_CPUS_PER_TASK python -u "${PROJECT_ROOT}/experiments/spectra/CCL/${TAG}/${LABEL,,}.py" --tag="${TAG}" --label="${LABEL}" --folder="${RUNTIME_ROOT}" --number="${SLURM_CPUS_PER_TASK}"
