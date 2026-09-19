#!/usr/bin/env bash
# Print interpreter, prefix, LimberCloud import path, and key package versions.

set -euo pipefail

SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd -P)

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/load_config.sh"

PYTHON_PATH="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x ${PYTHON_PATH} ]]; then
    printf 'Error: Python is not executable: %s\n' "${PYTHON_PATH}" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_PATH}" - <<'PY'
import importlib.metadata as metadata
import os
import sys

import limbercloud

def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"

print(f"executable={sys.executable}")
print(f"prefix={sys.prefix}")
print(f"limbercloud={limbercloud.__file__}")
print(f"LIMBERCLOUD_RUNTIME_ROOT={os.environ.get('LIMBERCLOUD_RUNTIME_ROOT')}")
print(f"PROJECT_ROOT={os.environ.get('PROJECT_ROOT')}")
for package in (
    "numpy",
    "scipy",
    "numba",
    "astropy",
    "matplotlib",
    "pyccl",
    "camb",
    "jax",
    "h5py",
    "mpi4py",
    "ipykernel",
    "ruff",
    "cosmosis",
):
    print(f"{package}={version(package)}")
PY
