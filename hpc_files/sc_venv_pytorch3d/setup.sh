#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

# Keep the pip cache off the small JURECA home quota (repo is assumed to live in project space).
export PIP_CACHE_DIR="${SMILIFY_DIR}/.pip_cache"

# Ensure CUDA is properly detected for pytorch3d build
export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 GPUs on JUWELS
export FORCE_CUDA=1

# Verify CUDA setup
echo "CUDA_HOME: $CUDA_HOME"
nvcc --version

python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"
#python3 -m venv --prompt "$ENV_NAME" "${ENV_DIR}"

source "${ABSOLUTE_PATH}"/activate.sh

python3 -m pip install --upgrade -r "${ABSOLUTE_PATH}"/requirements.txt

python3 -m pip install --upgrade --no-build-isolation -r "${ABSOLUTE_PATH}"/requirements_pt3d.txt