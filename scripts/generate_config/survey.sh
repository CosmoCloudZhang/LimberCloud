#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH -J CONFIG_SURVEY
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o logs/%x_%j.out
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
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

# Initialize the process
RUNTIME_ROOT="${LIMBERCLOUD_RUNTIME_ROOT:?Set LIMBERCLOUD_RUNTIME_ROOT to the external data/results root}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Run applications
python -u "${PROJECT_ROOT}/scripts/generate_config/survey.py" --folder="${RUNTIME_ROOT}"
