#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIRECTORY=$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
)
REPOSITORY_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd -P)
PYTHON_PATH="${REPOSITORY_ROOT}/.venv/bin/python"

source "${REPOSITORY_ROOT}/scripts/nersc/load_environment.sh"

if [[ ! -x ${PYTHON_PATH} ]]; then
    printf 'LimberCloud kernel error: Python is not executable: %s\n' \
        "${PYTHON_PATH}" >&2
    exit 1
fi

export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ${1:-} == "--probe" ]]; then
    exec "${PYTHON_PATH}" -c \
        'import os, sys; import limbercloud; assert os.environ.get("LIMBERCLOUD_RUNTIME_ROOT"); print(sys.executable); print(limbercloud.__file__); print("LimberCloud kernel environment passed")'
fi

exec "${PYTHON_PATH}" -m ipykernel_launcher "$@"
