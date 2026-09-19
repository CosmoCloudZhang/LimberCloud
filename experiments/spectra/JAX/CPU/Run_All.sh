#!/bin/bash

set -euo pipefail

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
mkdir -p "${PROJECT_ROOT}/logs"

for survey in Y1 Y10; do
    for configuration in single double triple; do
        sbatch --chdir="${PROJECT_ROOT}" "${PROJECT_ROOT}/experiments/spectra/JAX/CPU/${survey}/${configuration}.sh"
    done
done
