#!/usr/bin/env bash
# Portable LimberCloud configuration loader (Bash 3.2+ / macOS, NERSC Bash).
# Reads fixed ${PROJECT_ROOT}/.env without executing it. Exported values win.

_limbercloud_config_error() {
    printf 'LimberCloud configuration error: %s\n' "$*" >&2
}

_limbercloud_trim_leading_space() {
    # $1 = input, $2 = output variable name
    _limbercloud_trim_tmp=$1
    _limbercloud_trim_tmp=${_limbercloud_trim_tmp#"${_limbercloud_trim_tmp%%[![:space:]]*}"}
    eval "$2=\$_limbercloud_trim_tmp"
    unset _limbercloud_trim_tmp
}

_limbercloud_trim_trailing_space() {
    _limbercloud_trim_tmp=$1
    _limbercloud_trim_tmp=${_limbercloud_trim_tmp%"${_limbercloud_trim_tmp##*[![:space:]]}"}
    eval "$2=\$_limbercloud_trim_tmp"
    unset _limbercloud_trim_tmp
}

_limbercloud_parse_dotenv_value() {
    # $1 = raw value, $2 = line number, $3 = output variable name
    _limbercloud_raw=$1
    _limbercloud_value_lineno=$2
    _limbercloud_out_name=$3
    _limbercloud_parsed=
    _limbercloud_body=
    _limbercloud_suffix=
    _limbercloud_character=
    _limbercloud_escaped=0
    _limbercloud_index=0
    _limbercloud_body_length=0

    _limbercloud_trim_leading_space "$_limbercloud_raw" _limbercloud_raw

    case ${_limbercloud_raw} in
        \'*)
            _limbercloud_body=${_limbercloud_raw#\'}
            case ${_limbercloud_body} in
                *\'*)
                    _limbercloud_parsed=${_limbercloud_body%%\'*}
                    _limbercloud_suffix=${_limbercloud_body#*\'}
                    _limbercloud_trim_leading_space "$_limbercloud_suffix" _limbercloud_suffix
                    if [ -n "$_limbercloud_suffix" ]; then
                        case ${_limbercloud_suffix} in
                            \#*) ;;
                            *)
                                _limbercloud_config_error \
                                    "unexpected text after quoted value in .env line ${_limbercloud_value_lineno}"
                                return 1
                                ;;
                        esac
                    fi
                    ;;
                *)
                    _limbercloud_config_error \
                        "unterminated single quote in .env line ${_limbercloud_value_lineno}"
                    return 1
                    ;;
            esac
            ;;
        \"*)
            _limbercloud_body=${_limbercloud_raw#\"}
            _limbercloud_body_length=${#_limbercloud_body}
            _limbercloud_index=0
            _limbercloud_escaped=0
            _limbercloud_parsed=
            _limbercloud_suffix=
            while [ "$_limbercloud_index" -lt "$_limbercloud_body_length" ]; do
                _limbercloud_character=${_limbercloud_body:$_limbercloud_index:1}
                if [ "$_limbercloud_escaped" -eq 1 ]; then
                    case ${_limbercloud_character} in
                        n) _limbercloud_parsed=${_limbercloud_parsed}$'\n' ;;
                        r) _limbercloud_parsed=${_limbercloud_parsed}$'\r' ;;
                        t) _limbercloud_parsed=${_limbercloud_parsed}$'\t' ;;
                        \"|\\) _limbercloud_parsed=${_limbercloud_parsed}${_limbercloud_character} ;;
                        *) _limbercloud_parsed=${_limbercloud_parsed}\\${_limbercloud_character} ;;
                    esac
                    _limbercloud_escaped=0
                elif [ "$_limbercloud_character" = '\\' ]; then
                    _limbercloud_escaped=1
                elif [ "$_limbercloud_character" = '"' ]; then
                    _limbercloud_suffix=${_limbercloud_body:$((_limbercloud_index + 1))}
                    break
                else
                    _limbercloud_parsed=${_limbercloud_parsed}${_limbercloud_character}
                fi
                _limbercloud_index=$((_limbercloud_index + 1))
            done
            if [ "$_limbercloud_escaped" -eq 1 ] || [ "$_limbercloud_index" -eq "$_limbercloud_body_length" ]; then
                _limbercloud_config_error \
                    "unterminated double quote in .env line ${_limbercloud_value_lineno}"
                return 1
            fi
            _limbercloud_trim_leading_space "$_limbercloud_suffix" _limbercloud_suffix
            if [ -n "$_limbercloud_suffix" ]; then
                case ${_limbercloud_suffix} in
                    \#*) ;;
                    *)
                        _limbercloud_config_error \
                            "unexpected text after quoted value in .env line ${_limbercloud_value_lineno}"
                        return 1
                        ;;
                esac
            fi
            ;;
        *)
            _limbercloud_index=0
            _limbercloud_body_length=${#_limbercloud_raw}
            while [ "$_limbercloud_index" -lt "$_limbercloud_body_length" ]; do
                _limbercloud_character=${_limbercloud_raw:$_limbercloud_index:1}
                if [ "$_limbercloud_character" = "#" ]; then
                    if [ "$_limbercloud_index" -eq 0 ]; then
                        _limbercloud_raw=
                        break
                    fi
                    _limbercloud_prev=${_limbercloud_raw:$((_limbercloud_index - 1)):1}
                    if [ "$_limbercloud_prev" = " " ] || [ "$_limbercloud_prev" = $'\t' ]; then
                        _limbercloud_raw=${_limbercloud_raw:0:$_limbercloud_index}
                        break
                    fi
                fi
                _limbercloud_index=$((_limbercloud_index + 1))
            done
            _limbercloud_trim_trailing_space "$_limbercloud_raw" _limbercloud_parsed
            ;;
    esac

    eval "$_limbercloud_out_name=\$_limbercloud_parsed"
    unset _limbercloud_raw _limbercloud_value_lineno _limbercloud_out_name \
        _limbercloud_parsed _limbercloud_body _limbercloud_suffix \
        _limbercloud_character _limbercloud_escaped _limbercloud_index \
        _limbercloud_body_length _limbercloud_prev
}

_limbercloud_is_project_root() {
    [ -f "$1/scripts/load_config.sh" ] && [ -d "$1/src/limbercloud" ]
}

limbercloud_resolve_project_root() {
    # Sets PROJECT_ROOT. Prefers SLURM submit dir, then walks from the caller,
    # then from this loader. Rejects nested manuscript Git tops without markers.
    _limbercloud_candidate=
    _limbercloud_dir=
    _limbercloud_git_top=

    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        _limbercloud_candidate=$(cd -- "${SLURM_SUBMIT_DIR}" 2>/dev/null && pwd -P) || true
        if [ -n "$_limbercloud_candidate" ] && _limbercloud_is_project_root "$_limbercloud_candidate"; then
            PROJECT_ROOT=$_limbercloud_candidate
            unset _limbercloud_candidate _limbercloud_dir _limbercloud_git_top
            return 0
        fi
    fi

    if [ -n "${BASH_SOURCE[1]:-}" ]; then
        _limbercloud_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")" 2>/dev/null && pwd -P) || true
    elif [ -n "${BASH_SOURCE[0]:-}" ]; then
        _limbercloud_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" 2>/dev/null && pwd -P) || true
    else
        _limbercloud_dir=$(pwd -P)
    fi

    while [ -n "$_limbercloud_dir" ] && [ "$_limbercloud_dir" != "/" ]; do
        if _limbercloud_is_project_root "$_limbercloud_dir"; then
            PROJECT_ROOT=$_limbercloud_dir
            unset _limbercloud_candidate _limbercloud_dir _limbercloud_git_top
            return 0
        fi
        _limbercloud_dir=$(dirname -- "$_limbercloud_dir")
    done

    _limbercloud_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
    _limbercloud_candidate=$(cd -- "${_limbercloud_dir}/.." && pwd -P)
    if _limbercloud_is_project_root "$_limbercloud_candidate"; then
        PROJECT_ROOT=$_limbercloud_candidate
        unset _limbercloud_candidate _limbercloud_dir _limbercloud_git_top
        return 0
    fi

    _limbercloud_git_top=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null) || true
    if [ -n "$_limbercloud_git_top" ] && _limbercloud_is_project_root "$_limbercloud_git_top"; then
        PROJECT_ROOT=$_limbercloud_git_top
        unset _limbercloud_candidate _limbercloud_dir _limbercloud_git_top
        return 0
    fi

    _limbercloud_config_error \
        "could not resolve PROJECT_ROOT (expected scripts/load_config.sh and src/limbercloud)"
    unset _limbercloud_candidate _limbercloud_dir _limbercloud_git_top
    return 1
}

_limbercloud_load_config() {
    _limbercloud_dotenv_file=
    _limbercloud_line=
    _limbercloud_line_number=0
    _limbercloud_key=
    _limbercloud_raw_value=
    _limbercloud_value=
    _limbercloud_have_runtime=0
    _limbercloud_have_onecovariance=0
    _limbercloud_have_texlive=0
    _limbercloud_runtime_root=
    _limbercloud_onecovariance_root=
    _limbercloud_texlive_bin=
    _limbercloud_assignment=

    if ! limbercloud_resolve_project_root; then
        return 1
    fi

    # Exported values take precedence over dotenv.
    if [ -n "${LIMBERCLOUD_RUNTIME_ROOT+x}" ]; then
        _limbercloud_have_runtime=1
        _limbercloud_runtime_root=${LIMBERCLOUD_RUNTIME_ROOT}
    fi
    if [ -n "${LIMBERCLOUD_ONECOVARIANCE_ROOT+x}" ]; then
        _limbercloud_have_onecovariance=1
        _limbercloud_onecovariance_root=${LIMBERCLOUD_ONECOVARIANCE_ROOT}
    fi
    if [ -n "${LIMBERCLOUD_TEXLIVE_BIN+x}" ]; then
        _limbercloud_have_texlive=1
        _limbercloud_texlive_bin=${LIMBERCLOUD_TEXLIVE_BIN}
    fi

    _limbercloud_dotenv_file=${PROJECT_ROOT}/.env
    if [ -f "$_limbercloud_dotenv_file" ]; then
        while IFS= read -r _limbercloud_line || [ -n "$_limbercloud_line" ]; do
            _limbercloud_line_number=$((_limbercloud_line_number + 1))
            _limbercloud_line=${_limbercloud_line%$'\r'}
            _limbercloud_trim_leading_space "$_limbercloud_line" _limbercloud_line

            case ${_limbercloud_line} in
                ''|\#*) continue ;;
            esac

            case ${_limbercloud_line} in
                export[[:space:]]*)
                    _limbercloud_assignment=${_limbercloud_line#export}
                    _limbercloud_trim_leading_space "$_limbercloud_assignment" _limbercloud_assignment
                    ;;
                *)
                    _limbercloud_assignment=$_limbercloud_line
                    ;;
            esac

            case ${_limbercloud_assignment} in
                *=*)
                    _limbercloud_key=${_limbercloud_assignment%%=*}
                    _limbercloud_raw_value=${_limbercloud_assignment#*=}
                    ;;
                *)
                    _limbercloud_config_error \
                        "invalid assignment in ${_limbercloud_dotenv_file}:${_limbercloud_line_number}"
                    return 1
                    ;;
            esac

            case ${_limbercloud_key} in
                *[!A-Za-z0-9_]*|[0-9]*|"")
                    _limbercloud_config_error \
                        "invalid assignment in ${_limbercloud_dotenv_file}:${_limbercloud_line_number}"
                    return 1
                    ;;
            esac

            case ${_limbercloud_key} in
                LIMBERCLOUD_RUNTIME_ROOT|LIMBERCLOUD_ONECOVARIANCE_ROOT|LIMBERCLOUD_TEXLIVE_BIN)
                    if ! _limbercloud_parse_dotenv_value \
                        "$_limbercloud_raw_value" \
                        "$_limbercloud_line_number" \
                        _limbercloud_value; then
                        return 1
                    fi
                    case ${_limbercloud_key} in
                        LIMBERCLOUD_RUNTIME_ROOT)
                            if [ "$_limbercloud_have_runtime" -eq 0 ]; then
                                _limbercloud_runtime_root=$_limbercloud_value
                                _limbercloud_have_runtime=1
                            fi
                            ;;
                        LIMBERCLOUD_ONECOVARIANCE_ROOT)
                            if [ "$_limbercloud_have_onecovariance" -eq 0 ]; then
                                _limbercloud_onecovariance_root=$_limbercloud_value
                                _limbercloud_have_onecovariance=1
                            fi
                            ;;
                        LIMBERCLOUD_TEXLIVE_BIN)
                            if [ "$_limbercloud_have_texlive" -eq 0 ]; then
                                _limbercloud_texlive_bin=$_limbercloud_value
                                _limbercloud_have_texlive=1
                            fi
                            ;;
                    esac
                    ;;
                LIMBERCLOUD_CONDA_ENV|LIMBERCLOUD_ENV_FILE|LIMBERCLOUD_REPO_ROOT|CosmoENV|ONE_COVARIANCE_ROOT|ONECOVARIANCE_SCRIPT)
                    _limbercloud_config_error \
                        "obsolete key ${_limbercloud_key} in ${_limbercloud_dotenv_file}; remove it and use .venv / LIMBERCLOUD_RUNTIME_ROOT / LIMBERCLOUD_ONECOVARIANCE_ROOT / LIMBERCLOUD_TEXLIVE_BIN"
                    return 1
                    ;;
                *)
                    # Other project or tool settings may share .env. They are ignored.
                    ;;
            esac
        done < "$_limbercloud_dotenv_file"
    fi

    if [ "$_limbercloud_have_runtime" -eq 0 ] || [ -z "$_limbercloud_runtime_root" ]; then
        _limbercloud_config_error \
            "LIMBERCLOUD_RUNTIME_ROOT is required and must not be empty"
        return 1
    fi

    LIMBERCLOUD_RUNTIME_ROOT=$_limbercloud_runtime_root
    export PROJECT_ROOT LIMBERCLOUD_RUNTIME_ROOT
    RUNTIME_ROOT=$LIMBERCLOUD_RUNTIME_ROOT

    if [ "$_limbercloud_have_onecovariance" -eq 1 ]; then
        LIMBERCLOUD_ONECOVARIANCE_ROOT=$_limbercloud_onecovariance_root
        export LIMBERCLOUD_ONECOVARIANCE_ROOT
    fi
    if [ "$_limbercloud_have_texlive" -eq 1 ]; then
        LIMBERCLOUD_TEXLIVE_BIN=$_limbercloud_texlive_bin
        export LIMBERCLOUD_TEXLIVE_BIN
    fi

    unset _limbercloud_dotenv_file _limbercloud_line _limbercloud_line_number \
        _limbercloud_key _limbercloud_raw_value _limbercloud_value \
        _limbercloud_have_runtime _limbercloud_have_onecovariance \
        _limbercloud_have_texlive _limbercloud_runtime_root \
        _limbercloud_onecovariance_root _limbercloud_texlive_bin \
        _limbercloud_assignment
}

limbercloud_require_onecovariance() {
    _limbercloud_root=${LIMBERCLOUD_ONECOVARIANCE_ROOT-}

    if [ -z "$_limbercloud_root" ]; then
        printf '%s\n' \
            'LimberCloud configuration error: LIMBERCLOUD_ONECOVARIANCE_ROOT is required for covariance jobs' \
            >&2
        unset _limbercloud_root
        return 1
    fi
    if [ ! -d "$_limbercloud_root" ]; then
        printf 'LimberCloud configuration error: OneCovariance root is not a directory: %s\n' \
            "$_limbercloud_root" >&2
        unset _limbercloud_root
        return 1
    fi
    if [ ! -f "$_limbercloud_root/covariance.py" ]; then
        printf 'LimberCloud configuration error: covariance.py was not found under: %s\n' \
            "$_limbercloud_root" >&2
        unset _limbercloud_root
        return 1
    fi
    unset _limbercloud_root
}

if _limbercloud_load_config; then
    unset -f _limbercloud_config_error \
        _limbercloud_trim_leading_space \
        _limbercloud_trim_trailing_space \
        _limbercloud_parse_dotenv_value \
        _limbercloud_is_project_root \
        _limbercloud_load_config 2>/dev/null || true
else
    _limbercloud_status=$?
    return "$_limbercloud_status" 2>/dev/null || exit "$_limbercloud_status"
fi
unset _limbercloud_status
