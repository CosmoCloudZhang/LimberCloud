#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIRECTORY=$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd -P)
PYTHON_PATH="${PROJECT_ROOT}/.venv/bin/python"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/load_config.sh"

if [[ ! -x ${PYTHON_PATH} ]]; then
    printf 'LimberCloud kernel error: Python is not executable: %s\n' \
        "${PYTHON_PATH}" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# NERSC notebook hosts that write under CFS need the same locking policy as jobs.
if [[ -n ${NERSC_HOST:-} || -d /global/cfs ]]; then
    export HDF5_USE_FILE_LOCKING=FALSE
fi

if [[ ${1:-} == "--probe" ]]; then
    exec "${PYTHON_PATH}" -c \
        'import os, sys; import limbercloud; assert os.environ.get("LIMBERCLOUD_RUNTIME_ROOT"); print(sys.executable); print(limbercloud.__file__); print("LimberCloud kernel environment passed")'
fi

exec "${PYTHON_PATH}" -m ipykernel_launcher "$@"
