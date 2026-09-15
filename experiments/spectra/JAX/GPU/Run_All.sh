#!/bin/bash

set -euo pipefail

REPO_ROOT="${LIMBERCLOUD_REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
export LIMBERCLOUD_REPO_ROOT="${REPO_ROOT}"
source "${REPO_ROOT}/scripts/nersc/load_environment.sh"
mkdir -p "${REPO_ROOT}/logs"

for survey in Y1 Y10; do
    for configuration in single double triple; do
        sbatch --chdir="${REPO_ROOT}" "${REPO_ROOT}/experiments/spectra/JAX/GPU/${survey}/${configuration}.sh"
    done
done
