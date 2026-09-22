#!/usr/bin/env bash

# Capture the source path before any command substitution. Under Bash 3.2 a
# sourced file's BASH_SOURCE[0] is not preserved inside $( ... ), so resolving
# the directory in one step silently yields the caller's path instead.
_limbercloud_modules_source=${BASH_SOURCE[0]}

module load cpu

_limbercloud_modules_directory=$(
    cd -- "$(dirname -- "${_limbercloud_modules_source}")" && pwd -P
)
source "${_limbercloud_modules_directory}/common.sh"
unset _limbercloud_modules_directory _limbercloud_modules_source
