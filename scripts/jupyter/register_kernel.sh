#!/usr/bin/env bash

set -euo pipefail

KERNEL_NAME=limbercloud-cosmoconda
DISPLAY_NAME='LimberCloud'

usage() {
    printf '%s\n' \
        "Usage: $0 [--replace]" \
        "" \
        "Register a per-user LimberCloud Jupyter kernel. The kernel starts" \
        "through this checkout's .venv and loads its private .env each time." \
        "No package or Conda environment is created or modified."
}

replace_requested=0
case ${1:-} in
    "") ;;
    --replace) replace_requested=1 ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
[[ $# -le 1 ]] || {
    usage >&2
    exit 2
}

SCRIPT_DIRECTORY=$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
)
REPOSITORY_ROOT=$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd -P)
PYTHON_PATH="${REPOSITORY_ROOT}/.venv/bin/python"
LAUNCHER_PATH="${SCRIPT_DIRECTORY}/launch_kernel.sh"

[[ -x ${PYTHON_PATH} ]] || {
    printf 'Error: Python is not executable: %s\n' "${PYTHON_PATH}" >&2
    exit 1
}
[[ -x ${LAUNCHER_PATH} ]] || {
    printf 'Error: kernel launcher is not executable: %s\n' \
        "${LAUNCHER_PATH}" >&2
    exit 1
}

USER_KERNEL_DIRECTORY=$(
    "${PYTHON_PATH}" -c \
        'from pathlib import Path; from jupyter_core.paths import jupyter_data_dir; print(Path(jupyter_data_dir()) / "kernels" / "limbercloud-cosmoconda")'
)
KERNEL_JSON="${USER_KERNEL_DIRECTORY}/kernel.json"
replace_argument=()

if [[ -f ${KERNEL_JSON} ]]; then
    EXISTING_LAUNCHER=$(
        "${PYTHON_PATH}" -c \
            'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["argv"][0])' \
            "${KERNEL_JSON}"
    )
    if [[ ${EXISTING_LAUNCHER} != "${LAUNCHER_PATH}" ]] && \
        ((replace_requested == 0)); then
        printf '%s\n' \
            "Error: ${KERNEL_NAME} already belongs to another checkout:" \
            "  ${EXISTING_LAUNCHER}" \
            "Run with --replace only if you intend to replace that registration." \
            >&2
        exit 1
    fi
    replace_argument=(--replace)
fi

TEMPORARY_DIRECTORY=$(mktemp -d)
cleanup() {
    rm -rf -- "${TEMPORARY_DIRECTORY}"
}
trap cleanup EXIT

"${PYTHON_PATH}" -c \
    'import json, pathlib, sys; launcher, output, display = sys.argv[1:]; spec = {"argv": [launcher, "-f", "{connection_file}"], "display_name": display, "language": "python", "metadata": {"debugger": True, "limbercloud": {"launcher": launcher}}}; pathlib.Path(output).write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")' \
    "${LAUNCHER_PATH}" \
    "${TEMPORARY_DIRECTORY}/kernel.json" \
    "${DISPLAY_NAME}"

"${PYTHON_PATH}" -m jupyter kernelspec install \
    --user \
    --name "${KERNEL_NAME}" \
    "${replace_argument[@]}" \
    "${TEMPORARY_DIRECTORY}"

printf 'Registered Jupyter kernel: %s (%s)\n' \
    "${DISPLAY_NAME}" "${KERNEL_NAME}"
