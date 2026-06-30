#!/usr/bin/env bash
# Install the SMILify pytorch3d conda environment on Ubuntu/Linux.
# Mirrors the steps documented in README.md.
#
# Usage:
#   bash install.sh                # full install + run tests
#   bash install.sh --skip-tests   # full install, skip pytest
#   ENV_NAME=myenv bash install.sh # use a different env name (default: pytorch3d)

# NOTE: do NOT use `set -u` — conda's own deactivate hooks (e.g. libblas_mkl_deactivate.sh)
# reference unset vars like CONDA_MKL_INTERFACE_LAYER_BACKUP and will abort the script.
set -eo pipefail

ENV_NAME="${ENV_NAME:-pytorch3d}"
PYTHON_VERSION="3.10"
SKIP_TESTS=0

for arg in "$@"; do
    case "$arg" in
        --skip-tests) SKIP_TESTS=1 ;;
        -h|--help)
            sed -n '2,9p' "$0"
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Repo root is the parent of this script's hpc_files/ folder, so the script works
# whether invoked from the repo root or from inside hpc_files/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Locate conda. Prefer $CONDA_EXE, then common install paths.
find_conda() {
    if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
        echo "$CONDA_EXE"; return
    fi
    for base in "$HOME/miniforge3" "/root/miniforge3" "$HOME/miniconda3" "/opt/miniconda3" "$HOME/anaconda3" "/opt/anaconda3"; do
        if [[ -x "$base/bin/conda" ]]; then
            echo "$base/bin/conda"; return
        fi
    done
    if command -v conda >/dev/null 2>&1; then
        command -v conda; return
    fi
    return 1
}

CONDA_BIN="$(find_conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
    echo "ERROR: conda not found. Install miniforge or miniconda first." >&2
    exit 1
fi
CONDA_BASE="$("$CONDA_BIN" info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "==> Using conda at: $CONDA_BIN"
echo "==> Target env:     $ENV_NAME (Python $PYTHON_VERSION)"

# 1. Create env (skip if it already exists)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Env '$ENV_NAME' already exists — skipping create"
else
    echo "==> Creating conda env '$ENV_NAME'"
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

# 2. PyTorch 2.3.1 + CUDA 11.8
echo "==> Installing PyTorch 2.3.1 + CUDA 11.8"
conda install -n "$ENV_NAME" -y \
    pytorch=2.3.1 torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia

# 2b. fvcore / iopath / ninja / imageio / scikit-image
echo "==> Installing iopath, ninja, imageio, scikit-image"
conda install -n "$ENV_NAME" -y \
    -c conda-forge -c fvcore \
    iopath ninja imageio scikit-image

# 2c. yacs + pycocotools, then upgrade iopath
echo "==> Installing yacs, pycocotools (pip)"
pip install yacs pycocotools
pip install --upgrade iopath

# 3. PyTorch3D (Linux: conda channel)
echo "==> Installing pytorch3d"
conda install -n "$ENV_NAME" -y -c pytorch3d pytorch3d

# 4. Extra pip dependencies
echo "==> Installing extra pip dependencies"
pip install matplotlib scipy opencv-python nibabel trimesh timm pytest h5py psutil pandas toml

# 5. Verify
if [[ "$SKIP_TESTS" -eq 1 ]]; then
    echo "==> Skipping tests (--skip-tests)"
else
    echo "==> Running tests: pytest tests/ -m 'not slow' --continue-on-collection-errors"
    # Non-fatal: the env build above is what matters; the suite currently has known
    # import-path collection errors (see tests/README.md), so don't abort the install.
    pytest tests/ -m "not slow" --continue-on-collection-errors || \
        echo "WARN: test step had failures — see tests/README.md (harness import-path fix is pending)"
fi

echo
echo "==> Install complete. Activate with: conda activate $ENV_NAME"
