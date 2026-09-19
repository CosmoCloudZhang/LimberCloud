#!/usr/bin/env bash
# Activate the Conda prefix referenced by ${PROJECT_ROOT}/.venv.
# Requires PROJECT_ROOT and a working conda command (after module load conda).

_limbercloud_activate_error() {
    printf 'LimberCloud environment error: %s\n' "$*" >&2
}

if [ -z "${PROJECT_ROOT:-}" ]; then
    _limbercloud_activate_error "PROJECT_ROOT is unset; source scripts/load_config.sh first"
    return 1 2>/dev/null || exit 1
fi

_limbercloud_venv_link=${PROJECT_ROOT}/.venv
if [ ! -e "${_limbercloud_venv_link}" ] && [ ! -L "${_limbercloud_venv_link}" ]; then
    _limbercloud_activate_error "missing .venv link: ${_limbercloud_venv_link}"
    unset _limbercloud_venv_link
    return 1 2>/dev/null || exit 1
fi

_limbercloud_prefix=$(cd -- "${_limbercloud_venv_link}" && pwd -P) || {
    _limbercloud_activate_error "could not resolve .venv target: ${_limbercloud_venv_link}"
    unset _limbercloud_venv_link
    return 1 2>/dev/null || exit 1
}

if [ ! -x "${_limbercloud_prefix}/bin/python" ]; then
    _limbercloud_activate_error "Python is not executable under ${_limbercloud_prefix}/bin/python"
    unset _limbercloud_venv_link _limbercloud_prefix
    return 1 2>/dev/null || exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    _limbercloud_activate_error "conda is unavailable; load the NERSC conda module first"
    unset _limbercloud_venv_link _limbercloud_prefix
    return 1 2>/dev/null || exit 1
fi

# Ensure conda shell functions are available in batch shells.
if ! type conda 2>/dev/null | grep -q 'function'; then
    _limbercloud_conda_base=$(conda info --base 2>/dev/null) || true
    if [ -n "${_limbercloud_conda_base:-}" ] && [ -f "${_limbercloud_conda_base}/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        . "${_limbercloud_conda_base}/etc/profile.d/conda.sh"
    fi
    unset _limbercloud_conda_base
fi

conda activate "${_limbercloud_prefix}" || {
    _limbercloud_activate_error "conda activate failed for ${_limbercloud_prefix}"
    unset _limbercloud_venv_link _limbercloud_prefix
    return 1 2>/dev/null || exit 1
}

export LIMBERCLOUD_PYTHON=${_limbercloud_prefix}/bin/python
unset _limbercloud_venv_link _limbercloud_prefix
unset -f _limbercloud_activate_error 2>/dev/null || true
