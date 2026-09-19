#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage:" \
        "  $0 --name ENVIRONMENT_NAME [--nersc]" \
        "  $0 --prefix /absolute/path/to/environment [--nersc]" \
        "" \
        "Create a new limbercloud environment, install this checkout editable" \
        "with --no-deps, and create .venv only when it is absent." \
        "" \
        "Without --nersc, uses portable environment.yml (CPU JAX, serial h5py," \
        "conda-forge mpi4py). With --nersc, uses environment.nersc.yml (CUDA" \
        "JAX) and leaves mpi4py/h5py for scripts/nersc/install_mpi_h5py.sh." \
        "" \
        "Safety:" \
        "  - An explicit name or absolute prefix is required." \
        "  - An existing environment or prefix is never modified." \
        "  - An existing .venv file, directory, or symlink is never modified." \
        "  - CosmoConda is never targeted by this helper."
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

nersc_recipe=0
target_mode=
target=

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --nersc)
            nersc_recipe=1
            shift
            ;;
        --name|--prefix)
            [[ $# -ge 2 ]] || fail "$1 requires a value"
            target_mode=$1
            target=$2
            shift 2
            ;;
        *)
            usage >&2
            fail "unknown argument: $1"
            ;;
    esac
done

[[ -n ${target_mode} && -n ${target} ]] || {
    usage >&2
    exit 2
}

case "${target_mode}" in
    --name)
        [[ ${target} =~ ^[A-Za-z0-9_.-]+$ ]] || \
            fail "environment names may contain only letters, numbers, '.', '_', and '-'."
        [[ ${target} != "base" && ${target} != "root" && ${target} != "CosmoConda" ]] || \
            fail "refusing to use the reserved Conda environment name '${target}'."
        create_args=(--name "${target}")
        run_args=(--name "${target}")
        ;;
    --prefix)
        [[ ${target} == /* ]] || fail "--prefix requires an absolute path."
        [[ ${target} != "/" ]] || fail "refusing to use '/' as an environment prefix."
        target=${target%/}
        [[ ${target} != *CosmoConda* ]] || fail "refusing to target a CosmoConda prefix."
        [[ ! -e ${target} && ! -L ${target} ]] || \
            fail "target prefix already exists; it will not be modified: ${target}"
        create_args=(--prefix "${target}")
        run_args=(--prefix "${target}")
        ;;
    *)
        usage >&2
        fail "first argument must be --name or --prefix."
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
if ((nersc_recipe)); then
    manifest="${repo_root}/environment.nersc.yml"
else
    manifest="${repo_root}/environment.yml"
fi
venv_link="${repo_root}/.venv"

[[ -f ${manifest} ]] || fail "environment manifest not found: ${manifest}"

if ! command -v conda >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load conda
    else
        fail "Conda is unavailable. On NERSC, initialize modules and run 'module load conda'."
    fi
fi

command -v conda >/dev/null 2>&1 || fail "the Conda command is unavailable after module setup."

if [[ ${target_mode} == "--name" ]]; then
    if conda env list | awk -v requested="${target}" \
        '$1 == requested { found = 1 } END { exit(found ? 0 : 1) }'; then
        fail "Conda environment '${target}' already exists; it will not be modified."
    fi
fi

printf 'Creating a new Conda environment from %s\n' "${manifest}"
conda env create --file "${manifest}" "${create_args[@]}"

printf 'Installing this LimberCloud checkout editable without changing dependencies\n'
conda run --no-capture-output "${run_args[@]}" \
    python -m pip install --no-deps --editable "${repo_root}"

environment_prefix=$(
    conda run "${run_args[@]}" python -c 'import sys; print(sys.prefix)'
)
[[ -d ${environment_prefix} ]] || \
    fail "created environment prefix could not be resolved: ${environment_prefix}"

printf 'Verifying the new environment (imports only; MPI runtime not initialized)\n'
if ((nersc_recipe)); then
    verify_code='import astropy; import camb; import ipykernel; import jax; import limbercloud; import matplotlib; import numba; import numpy; import pyccl; import scipy; print("LimberCloud NERSC base imports passed (mpi4py/h5py pending install_mpi_h5py.sh)")'
else
    verify_code='import astropy; import camb; import h5py; import ipykernel; import jax; import limbercloud; import matplotlib; import mpi4py; import numba; import numpy; import pyccl; import scipy; print("LimberCloud portable imports passed")'
fi
conda run --no-capture-output "${run_args[@]}" python -c "${verify_code}"

if [[ -e ${venv_link} || -L ${venv_link} ]]; then
    printf 'Leaving existing .venv unchanged: %s\n' "${venv_link}"
    printf 'Inspect with: readlink -f %s\n' "${venv_link}"
else
    ln -s -- "${environment_prefix}" "${venv_link}"
    printf 'Created %s -> %s\n' "${venv_link}" "${environment_prefix}"
fi

printf 'Environment created successfully at %s\n' "${environment_prefix}"
if ((nersc_recipe)); then
    printf 'Next: activate the environment, source scripts/nersc/modules/cpu.sh, then run scripts/nersc/install_mpi_h5py.sh\n'
fi
