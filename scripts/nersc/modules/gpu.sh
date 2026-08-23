#!/usr/bin/env bash

module load gpu

_limbercloud_modules_directory=$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
)
source "${_limbercloud_modules_directory}/common.sh"
unset _limbercloud_modules_directory
