SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

## Check if this script is sourced
[[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Vars script must be sourced." && exit 1) ;
## Determine location of this file
RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
####################################

### User Configuration
export ENV_NAME="$(basename "$ABSOLUTE_PATH")"             # venv name defaults to this folder's name
export ENV_DIR="${ABSOLUTE_PATH}"/venv                    # venv lives in "./venv" (gitignored)
export SMILIFY_DIR="$(realpath "${ABSOLUTE_PATH}/../..")"  # repo root (two levels up: hpc_files/sc_venv_pytorch3d)
