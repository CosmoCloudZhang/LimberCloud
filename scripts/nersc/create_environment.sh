#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage:" \
        "  $0 --name ENVIRONMENT_NAME" \
        "  $0 --prefix /absolute/path/to/environment" \
        "" \
        "Create a new environment from environment.yml, install this checkout" \
        "editable with --no-deps, and create .venv only when it is absent." \
        "" \
        "Safety:" \
        "  - An explicit name or absolute prefix is required." \
        "  - An existing environment or prefix is never modified." \
        "  - An existing .venv file, directory, or symlink is never modified."
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

target_mode=$1
target=$2

case "${target_mode}" in
    --name)
        [[ ${target} =~ ^[A-Za-z0-9_.-]+$ ]] || \
            fail "environment names may contain only letters, numbers, '.', '_', and '-'."
        [[ ${target} != "base" && ${target} != "root" ]] || \
            fail "refusing to use the reserved Conda environment name '${target}'."
        create_args=(--name "${target}")
        run_args=(--name "${target}")
        ;;
    --prefix)
        [[ ${target} == /* ]] || fail "--prefix requires an absolute path."
        [[ ${target} != "/" ]] || fail "refusing to use '/' as an environment prefix."
        target=${target%/}
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
manifest="${repo_root}/environment.yml"
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

printf 'Verifying the new environment (imports only; no GPU allocation is required)\n'
verify_code='import astropy; import ipykernel; import jax; import limbercloud; import matplotlib; import numba; import numpy; import pyccl; import scipy; print("LimberCloud environment imports passed")'
conda run --no-capture-output "${run_args[@]}" python -c "${verify_code}"

if [[ -e ${venv_link} || -L ${venv_link} ]]; then
    printf 'Leaving existing .venv unchanged: %s\n' "${venv_link}"
else
    ln -s -- "${environment_prefix}" "${venv_link}"
    printf 'Created %s -> %s\n' "${venv_link}" "${environment_prefix}"
fi

printf 'Environment created successfully at %s\n' "${environment_prefix}"
