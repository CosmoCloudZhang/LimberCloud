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
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Initialize the process
TAG="Y10"
RUNTIME_ROOT="${LIMBERCLOUD_RUNTIME_ROOT:?Set LIMBERCLOUD_RUNTIME_ROOT to the external data/results root}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
limbercloud_require_onecovariance
ONECOVARIANCE_SCRIPT="${LIMBERCLOUD_ONECOVARIANCE_ROOT%/}/covariance.py"

COVARIANCE_CONFIG="${RUNTIME_ROOT}/results/covariance/${TAG}/CONFIG.ini"

# OneCovariance runs in the interpreter selected here. A separate checkout
# directory does not isolate imports or native extensions, so its executable and
# import-time dependencies are verified before the covariance job starts rather
# than after an expensive preparation step. Set
# LIMBERCLOUD_ONECOVARIANCE_PYTHON to run it in a deliberately separate
# environment; the value is reported so the producing interpreter is recorded.
LIMBERCLOUD_ONECOVARIANCE_PYTHON="${LIMBERCLOUD_ONECOVARIANCE_PYTHON:-${LIMBERCLOUD_PYTHON:-python}}"
[[ -f ${ONECOVARIANCE_SCRIPT} ]] || \
    { echo "LimberCloud error: OneCovariance executable is missing: ${ONECOVARIANCE_SCRIPT}" >&2; exit 1; }
"${LIMBERCLOUD_ONECOVARIANCE_PYTHON}" \
    "${PROJECT_ROOT}/scripts/nersc/check_onecovariance.py" "${ONECOVARIANCE_SCRIPT}"

# Run applications
python -u "${PROJECT_ROOT}/experiments/covariance/${TAG}/matrix.py" --tag="${TAG}" --folder="${RUNTIME_ROOT}" "$@" &&
srun -u -N 1 -n 1 -c "${SLURM_CPUS_PER_TASK}" "${LIMBERCLOUD_ONECOVARIANCE_PYTHON}" "${ONECOVARIANCE_SCRIPT}" "${COVARIANCE_CONFIG}"
