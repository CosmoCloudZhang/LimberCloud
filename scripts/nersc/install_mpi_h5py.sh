#!/usr/bin/env bash
# Install mpi4py and parallel h5py against NERSC Cray MPICH / HDF5 in the
# already-activated limbercloud environment. Never run against CosmoConda.

set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: $0" \
        "" \
        "Build mpi4py and MPI-enabled h5py from source using the NERSC GNU/Cray" \
        "stack (PrgEnv-gnu, cray-mpich, cray-hdf5-parallel). The current Conda" \
        "environment must already be the dedicated limbercloud prefix; this" \
        "script refuses CosmoConda."
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

# The installer mutates an environment, so it validates its target before
# touching anything. LIMBERCLOUD_INSTALL_PREFIX overrides the discovered .venv
# target only for tests with synthetic checkouts.
_limbercloud_installer_source=${BASH_SOURCE[0]}
INSTALLER_DIRECTORY=$(cd -- "$(dirname -- "${_limbercloud_installer_source}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "${INSTALLER_DIRECTORY}/../.." && pwd -P)}
unset _limbercloud_installer_source

[[ -n ${CONDA_PREFIX:-} ]] || fail "activate the limbercloud Conda environment first"
[[ ${CONDA_PREFIX} != *CosmoConda* ]] || \
    fail "refusing to modify CosmoConda; activate limbercloud instead"
[[ ${CONDA_DEFAULT_ENV:-} != CosmoConda ]] || \
    fail "refusing to modify CosmoConda; activate limbercloud instead"
[[ ${CONDA_DEFAULT_ENV:-} != base ]] || \
    fail "refusing to modify the base environment; activate limbercloud instead"

EXPECTED_PREFIX=${LIMBERCLOUD_INSTALL_PREFIX:-}
if [[ -z ${EXPECTED_PREFIX} ]]; then
    [[ -e ${PROJECT_ROOT}/.venv ]] || \
        fail "missing ${PROJECT_ROOT}/.venv; the installer target cannot be identified"
    EXPECTED_PREFIX=$(cd -- "${PROJECT_ROOT}/.venv" && pwd -P)
fi
ACTIVE_PREFIX=$(cd -- "${CONDA_PREFIX}" && pwd -P)
[[ ${ACTIVE_PREFIX} == "${EXPECTED_PREFIX}" ]] || \
    fail "active prefix ${ACTIVE_PREFIX} is not the selected target ${EXPECTED_PREFIX}"

# The interpreter on PATH must be the one inside that prefix, not another
# environment's python that merely inherited the variables.
ACTIVE_PYTHON=$(command -v python) || fail "python is unavailable"
REPORTED_PREFIX=$(python -c 'import sys; print(sys.prefix)') || \
    fail "could not query sys.prefix from ${ACTIVE_PYTHON}"
[[ ${REPORTED_PREFIX} == "${EXPECTED_PREFIX}" ]] || \
    fail "python reports sys.prefix=${REPORTED_PREFIX}, not ${EXPECTED_PREFIX}"

command -v cc >/dev/null 2>&1 || fail "Cray compiler wrapper 'cc' is unavailable"
[[ -n ${HDF5_DIR:-} || -d /opt/cray/pe/hdf5-parallel ]] || \
    fail "load cray-hdf5-parallel before running this installer"

printf 'Installing into %s via %s\n' "${EXPECTED_PREFIX}" "${ACTIVE_PYTHON}"

if [[ ${LIMBERCLOUD_INSTALL_DRY_RUN:-0} == 1 ]]; then
    printf 'Dry run: target validated, no package was modified.\n'
    exit 0
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=2.2,<2.3" "cython>=3" pkgconfig

# Remove any conda-forge serial h5py/hdf5 so the Cray-linked build is used.
python -m pip uninstall -y h5py >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
    conda remove -y --force h5py hdf5 2>/dev/null || true
fi

# Conda pkg-config needs the system Cray xpmem metadata on the search path.
if [[ -d /usr/lib64/pkgconfig ]]; then
    export PKG_CONFIG_PATH="/usr/lib64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
fi

# An already-satisfied generic wheel would otherwise be accepted as a site
# build, so the source build is forced and its linkage is recorded.
printf 'Building mpi4py against Cray MPICH via cc -shared\n'
python -m pip uninstall -y mpi4py >/dev/null 2>&1 || true
MPI4PY_BUILD_MPICC="cc -shared" \
    python -m pip install -v --no-cache-dir --force-reinstall --no-binary=mpi4py mpi4py

printf 'Building parallel h5py against cray-hdf5-parallel\n'
[[ -n ${HDF5_DIR:-} ]] || fail "HDF5_DIR is unset; load cray-hdf5-parallel first"
export LD_LIBRARY_PATH="${HDF5_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${HDF5_DIR}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CPATH="${HDF5_DIR}/include${CPATH:+:${CPATH}}"
HDF5_MPI=ON CC=cc HDF5_DIR="${HDF5_DIR}" \
    python -m pip install -v --no-cache-dir --force-reinstall --no-binary=h5py \
    --no-build-isolation --no-deps h5py

printf 'Source MPI/HDF5 builds completed in %s\n' "${CONDA_PREFIX}"
printf 'Validate under srun on a compute allocation; do not import mpi4py.MPI on login nodes.\n'
