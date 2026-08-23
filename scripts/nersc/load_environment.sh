#!/usr/bin/env bash

_limbercloud_environment_error() {
    printf 'LimberCloud environment error: %s\n' "$*" >&2
}

_limbercloud_environment_warning() {
    printf 'LimberCloud environment warning: %s\n' "$*" >&2
}

_limbercloud_trim_leading_space() {
    local result=$1
    result=${result#"${result%%[![:space:]]*}"}
    printf -v "$2" '%s' "$result"
}

_limbercloud_trim_trailing_space() {
    local result=$1
    result=${result%"${result##*[![:space:]]}"}
    printf -v "$2" '%s' "$result"
}

_limbercloud_parse_dotenv_value() {
    local raw_value=$1
    local line_number=$2
    local output_name=$3
    local body character escaped parsed suffix
    local index

    _limbercloud_trim_leading_space "$raw_value" raw_value
    parsed=

    case ${raw_value:0:1} in
        "'")
            body=${raw_value:1}
            if [[ $body != *"'"* ]]; then
                _limbercloud_environment_error \
                    "unterminated single quote in .env line ${line_number}"
                return 1
            fi
            parsed=${body%%\'*}
            suffix=${body#*\'}
            _limbercloud_trim_leading_space "$suffix" suffix
            if [[ -n $suffix && ${suffix:0:1} != "#" ]]; then
                _limbercloud_environment_error \
                    "unexpected text after quoted value in .env line ${line_number}"
                return 1
            fi
            ;;
        '"')
            body=${raw_value:1}
            escaped=0
            suffix=
            for ((index = 0; index < ${#body}; index++)); do
                character=${body:index:1}
                if ((escaped)); then
                    case $character in
                        n) parsed+=$'\n' ;;
                        r) parsed+=$'\r' ;;
                        t) parsed+=$'\t' ;;
                        '"'|'\\') parsed+=$character ;;
                        *) parsed+="\\${character}" ;;
                    esac
                    escaped=0
                elif [[ $character == '\\' ]]; then
                    escaped=1
                elif [[ $character == '"' ]]; then
                    suffix=${body:index+1}
                    break
                else
                    parsed+=$character
                fi
            done
            if ((escaped)) || ((index == ${#body})); then
                _limbercloud_environment_error \
                    "unterminated double quote in .env line ${line_number}"
                return 1
            fi
            _limbercloud_trim_leading_space "$suffix" suffix
            if [[ -n $suffix && ${suffix:0:1} != "#" ]]; then
                _limbercloud_environment_error \
                    "unexpected text after quoted value in .env line ${line_number}"
                return 1
            fi
            ;;
        *)
            # In an unquoted value, a hash starts a comment only at the beginning
            # or after whitespace. No interpolation or command substitution occurs.
            for ((index = 0; index < ${#raw_value}; index++)); do
                character=${raw_value:index:1}
                if [[ $character == "#" ]] && \
                    ((index == 0 || ${raw_value:index-1:1} == " " || \
                        ${raw_value:index-1:1} == $'\t')); then
                    raw_value=${raw_value:0:index}
                    break
                fi
            done
            _limbercloud_trim_trailing_space "$raw_value" parsed
            ;;
    esac

    printf -v "$output_name" '%s' "$parsed"
}

_limbercloud_read_dotenv() {
    local dotenv_file=$1
    local output_name=$2
    local line key raw_value value
    local line_number=0
    local -n dotenv_values_ref=$output_name

    while IFS= read -r line || [[ -n $line ]]; do
        ((line_number += 1))
        line=${line%$'\r'}
        _limbercloud_trim_leading_space "$line" line

        [[ -z $line || ${line:0:1} == "#" ]] && continue
        if [[ ! $line =~ ^(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
            _limbercloud_environment_error \
                "invalid assignment in ${dotenv_file}:${line_number}"
            return 1
        fi

        key=${BASH_REMATCH[2]}
        raw_value=${BASH_REMATCH[3]}
        case $key in
            LIMBERCLOUD_RUNTIME_ROOT|LIMBERCLOUD_CONDA_ENV|\
                LIMBERCLOUD_ONECOVARIANCE_ROOT|LIMBERCLOUD_TEXLIVE_BIN|\
                CosmoENV|ONE_COVARIANCE_ROOT|ONECOVARIANCE_SCRIPT)
                if ! _limbercloud_parse_dotenv_value \
                    "$raw_value" "$line_number" value; then
                    return 1
                fi
                dotenv_values_ref["$key"]=$value
                ;;
            *)
                # Other project or tool settings may share .env. They are ignored
                # rather than exported, keeping this loader explicitly allowlisted.
                ;;
        esac
    done < "$dotenv_file"
}

_limbercloud_legacy_warning() {
    local old_name=$1
    local new_name=$2
    _limbercloud_environment_warning \
        "${old_name} is deprecated; use ${new_name} instead"
}

_limbercloud_value_is_set() {
    [[ -v $1 ]]
}

_limbercloud_dotenv_has() {
    local array_name=$1
    local key=$2
    local -n values_ref=$array_name
    [[ ${values_ref[$key]+present} == present ]]
}

_limbercloud_derive_onecovariance_root() {
    local script_path=$1
    local output_name=$2
    local root

    if [[ $script_path == */* ]]; then
        root=${script_path%/*}
        [[ -n $root ]] || root=/
    else
        root=.
    fi
    printf -v "$output_name" '%s' "$root"
}

_limbercloud_normalize_legacy_onecovariance_root() {
    local legacy_value=$1
    local output_name=$2

    # ONE_COVARIANCE_ROOT was historically documented as both a directory and
    # the full covariance.py path. Accept either interpretation during migration.
    if [[ ${legacy_value##*/} == "covariance.py" ]]; then
        _limbercloud_derive_onecovariance_root "$legacy_value" "$output_name"
    else
        printf -v "$output_name" '%s' "$legacy_value"
    fi
}

_limbercloud_load_environment() {
    local loader_directory repository_root dotenv_file
    local explicit_dotenv=0
    local runtime_root conda_environment onecovariance_root texlive_bin
    local have_runtime=0 have_conda=0 have_onecovariance=0 have_texlive=0
    local have_legacy_conda=0 have_legacy_root=0 have_legacy_script=0
    local legacy_conda legacy_root legacy_script
    local -A dotenv_values=()

    loader_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
    repository_root=$(cd -- "${loader_directory}/../.." && pwd -P)

    if _limbercloud_value_is_set LIMBERCLOUD_RUNTIME_ROOT; then
        have_runtime=1
        runtime_root=${LIMBERCLOUD_RUNTIME_ROOT}
    fi
    if _limbercloud_value_is_set LIMBERCLOUD_CONDA_ENV; then
        have_conda=1
        conda_environment=${LIMBERCLOUD_CONDA_ENV}
    fi
    if _limbercloud_value_is_set LIMBERCLOUD_ONECOVARIANCE_ROOT; then
        have_onecovariance=1
        onecovariance_root=${LIMBERCLOUD_ONECOVARIANCE_ROOT}
    fi
    if _limbercloud_value_is_set LIMBERCLOUD_TEXLIVE_BIN; then
        have_texlive=1
        texlive_bin=${LIMBERCLOUD_TEXLIVE_BIN}
    fi

    if _limbercloud_value_is_set CosmoENV; then
        have_legacy_conda=1
        legacy_conda=${CosmoENV}
    fi
    if _limbercloud_value_is_set ONE_COVARIANCE_ROOT; then
        have_legacy_root=1
        legacy_root=${ONE_COVARIANCE_ROOT}
    fi
    if _limbercloud_value_is_set ONECOVARIANCE_SCRIPT; then
        have_legacy_script=1
        legacy_script=${ONECOVARIANCE_SCRIPT}
    fi

    if _limbercloud_value_is_set LIMBERCLOUD_ENV_FILE; then
        explicit_dotenv=1
        dotenv_file=${LIMBERCLOUD_ENV_FILE}
        if [[ -z $dotenv_file ]]; then
            _limbercloud_environment_error \
                "LIMBERCLOUD_ENV_FILE is set but empty"
            return 1
        fi
    else
        dotenv_file=${repository_root}/.env
    fi

    if [[ -f $dotenv_file ]]; then
        if ! _limbercloud_read_dotenv "$dotenv_file" dotenv_values; then
            return 1
        fi
    elif ((explicit_dotenv)); then
        _limbercloud_environment_error \
            "LIMBERCLOUD_ENV_FILE does not exist: ${dotenv_file}"
        return 1
    fi

    if _limbercloud_dotenv_has dotenv_values CosmoENV; then
        _limbercloud_legacy_warning CosmoENV LIMBERCLOUD_CONDA_ENV
    fi
    if _limbercloud_dotenv_has dotenv_values ONE_COVARIANCE_ROOT; then
        _limbercloud_legacy_warning \
            ONE_COVARIANCE_ROOT LIMBERCLOUD_ONECOVARIANCE_ROOT
    fi
    if _limbercloud_dotenv_has dotenv_values ONECOVARIANCE_SCRIPT; then
        _limbercloud_legacy_warning \
            ONECOVARIANCE_SCRIPT LIMBERCLOUD_ONECOVARIANCE_ROOT
    fi
    ((have_legacy_conda)) && \
        _limbercloud_legacy_warning CosmoENV LIMBERCLOUD_CONDA_ENV
    ((have_legacy_root)) && \
        _limbercloud_legacy_warning \
            ONE_COVARIANCE_ROOT LIMBERCLOUD_ONECOVARIANCE_ROOT
    ((have_legacy_script)) && \
        _limbercloud_legacy_warning \
            ONECOVARIANCE_SCRIPT LIMBERCLOUD_ONECOVARIANCE_ROOT

    if ((!have_runtime)) && \
        _limbercloud_dotenv_has dotenv_values LIMBERCLOUD_RUNTIME_ROOT; then
        runtime_root=${dotenv_values[LIMBERCLOUD_RUNTIME_ROOT]}
        have_runtime=1
    fi
    if ((!have_conda)) && \
        _limbercloud_dotenv_has dotenv_values LIMBERCLOUD_CONDA_ENV; then
        conda_environment=${dotenv_values[LIMBERCLOUD_CONDA_ENV]}
        have_conda=1
    fi
    if ((!have_onecovariance)) && \
        _limbercloud_dotenv_has dotenv_values LIMBERCLOUD_ONECOVARIANCE_ROOT; then
        onecovariance_root=${dotenv_values[LIMBERCLOUD_ONECOVARIANCE_ROOT]}
        have_onecovariance=1
    fi
    if ((!have_texlive)) && \
        _limbercloud_dotenv_has dotenv_values LIMBERCLOUD_TEXLIVE_BIN; then
        texlive_bin=${dotenv_values[LIMBERCLOUD_TEXLIVE_BIN]}
        have_texlive=1
    fi

    if ((!have_conda)); then
        if ((have_legacy_conda)); then
            conda_environment=$legacy_conda
            have_conda=1
        elif _limbercloud_dotenv_has dotenv_values CosmoENV; then
            conda_environment=${dotenv_values[CosmoENV]}
            have_conda=1
        else
            conda_environment=CosmoConda
            have_conda=1
        fi
    fi

    if ((!have_onecovariance)); then
        if ((have_legacy_root)); then
            _limbercloud_normalize_legacy_onecovariance_root \
                "$legacy_root" onecovariance_root
            have_onecovariance=1
        elif _limbercloud_dotenv_has dotenv_values ONE_COVARIANCE_ROOT; then
            _limbercloud_normalize_legacy_onecovariance_root \
                "${dotenv_values[ONE_COVARIANCE_ROOT]}" onecovariance_root
            have_onecovariance=1
        elif ((have_legacy_script)); then
            _limbercloud_derive_onecovariance_root \
                "$legacy_script" onecovariance_root
            have_onecovariance=1
        elif _limbercloud_dotenv_has dotenv_values ONECOVARIANCE_SCRIPT; then
            _limbercloud_derive_onecovariance_root \
                "${dotenv_values[ONECOVARIANCE_SCRIPT]}" onecovariance_root
            have_onecovariance=1
        fi
    fi

    if ((!have_runtime)) || [[ -z $runtime_root ]]; then
        _limbercloud_environment_error \
            "LIMBERCLOUD_RUNTIME_ROOT is required and must not be empty"
        return 1
    fi
    if ((!have_conda)) || [[ -z $conda_environment ]]; then
        _limbercloud_environment_error \
            "LIMBERCLOUD_CONDA_ENV is required and must not be empty"
        return 1
    fi

    LIMBERCLOUD_RUNTIME_ROOT=$runtime_root
    LIMBERCLOUD_CONDA_ENV=$conda_environment
    export LIMBERCLOUD_RUNTIME_ROOT LIMBERCLOUD_CONDA_ENV

    if ((have_onecovariance)); then
        LIMBERCLOUD_ONECOVARIANCE_ROOT=$onecovariance_root
        export LIMBERCLOUD_ONECOVARIANCE_ROOT
    fi
    if ((have_texlive)); then
        LIMBERCLOUD_TEXLIVE_BIN=$texlive_bin
        export LIMBERCLOUD_TEXLIVE_BIN
    fi
}

limbercloud_require_onecovariance() {
    local root=${LIMBERCLOUD_ONECOVARIANCE_ROOT-}

    if [[ -z $root ]]; then
        printf '%s\n' \
            'LimberCloud environment error: LIMBERCLOUD_ONECOVARIANCE_ROOT is required for covariance jobs' \
            >&2
        return 1
    fi
    if [[ ! -d $root ]]; then
        printf 'LimberCloud environment error: OneCovariance root is not a directory: %s\n' \
            "$root" >&2
        return 1
    fi
    if [[ ! -f $root/covariance.py ]]; then
        printf 'LimberCloud environment error: covariance.py was not found under: %s\n' \
            "$root" >&2
        return 1
    fi
}

if _limbercloud_load_environment; then
    unset -f _limbercloud_environment_error \
        _limbercloud_environment_warning \
        _limbercloud_trim_leading_space \
        _limbercloud_trim_trailing_space \
        _limbercloud_parse_dotenv_value \
        _limbercloud_read_dotenv \
        _limbercloud_legacy_warning \
        _limbercloud_value_is_set \
        _limbercloud_dotenv_has \
        _limbercloud_derive_onecovariance_root \
        _limbercloud_normalize_legacy_onecovariance_root \
        _limbercloud_load_environment
else
    _limbercloud_status=$?
    return "$_limbercloud_status" 2>/dev/null || exit "$_limbercloud_status"
fi
unset _limbercloud_status
