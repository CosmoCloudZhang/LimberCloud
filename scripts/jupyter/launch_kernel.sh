#!/usr/bin/env bash

set -eo pipefail

# Capture the source path before command substitution so Bash 3.2 resolves the
# script directory rather than the caller's.
_limbercloud_kernel_source=${BASH_SOURCE[0]}
SCRIPT_DIRECTORY=$(
    cd -- "$(dirname -- "${_limbercloud_kernel_source}")" && pwd -P
)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd -P)
PYTHON_PATH="${PROJECT_ROOT}/.venv/bin/python"
unset _limbercloud_kernel_source

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/load_config.sh"

# A kernel must inherit the same site modules and Conda hooks as a batch job,
# or it selects the right interpreter without the shared-library state that
# CAMB, MPI-linked HDF5 and the GPU backends need. Portable hosts keep the
# plain interpreter; only supported NERSC hosts load the site stack.
LIMBERCLOUD_KERNEL_PROFILE=${LIMBERCLOUD_KERNEL_PROFILE:-cpu}
if [[ -n ${NERSC_HOST:-} ]]; then
    case ${LIMBERCLOUD_KERNEL_PROFILE} in
        cpu|gpu) ;;
        *)
            printf 'LimberCloud kernel error: unknown LIMBERCLOUD_KERNEL_PROFILE %s\n' \
                "${LIMBERCLOUD_KERNEL_PROFILE}" >&2
            exit 1
            ;;
    esac
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/scripts/nersc/modules/${LIMBERCLOUD_KERNEL_PROFILE}.sh"
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/scripts/nersc/activate_venv.sh"
    PYTHON_PATH=${LIMBERCLOUD_PYTHON:-${PYTHON_PATH}}
fi

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

# --probe stays login-safe: it reports identity only and imports no site-linked
# or MPI-initialising library. --science is the allocated check: it reads a real
# HDF5 file and evaluates a tiny CCL background quantity, so it must run under
# an allocation on a supported host.
if [[ ${1:-} == "--probe" ]]; then
    exec "${PYTHON_PATH}" -c '
import os
import sys

import limbercloud

assert os.environ.get("LIMBERCLOUD_RUNTIME_ROOT")
print(sys.executable)
print(limbercloud.__file__)
print("interpreter and checkout identity reported; no scientific import attempted")
'
fi

if [[ ${1:-} == "--science" ]]; then
    exec "${PYTHON_PATH}" -c '
import os
import sys
import tempfile

import h5py
import numpy
import pyccl

import limbercloud
from limbercloud.validation.cosmology import solver_package_versions

print(sys.executable)
print(limbercloud.__file__)
print(solver_package_versions())
print("h5py", h5py.__version__, "mpi", h5py.get_config().mpi)

with tempfile.TemporaryDirectory(dir=os.environ["LIMBERCLOUD_RUNTIME_ROOT"]) as directory:
    path = os.path.join(directory, "kernel_probe.h5")
    payload = numpy.arange(8, dtype=numpy.float64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("payload", data=payload)
    with h5py.File(path, "r") as handle:
        assert numpy.array_equal(handle["payload"][...], payload)

cosmology = pyccl.Cosmology(
    h=0.6736, Omega_c=0.26, Omega_b=0.05, n_s=0.9649, A_s=2.083e-9,
    transfer_function="boltzmann_camb",
)
print("comoving distance at z=1:", pyccl.comoving_radial_distance(cosmology, 0.5))
print("LimberCloud scientific kernel check passed")
'
fi

exec "${PYTHON_PATH}" -m ipykernel_launcher "$@"
