#!/usr/bin/env bash
# Install the SMILify pytorch3d conda environment on Ubuntu/Linux.
# Mirrors the steps documented in README.md.
# Also installs tmux (handy for detaching long-running run.ai jobs).
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

# 1b. tmux — installed SYSTEM-WIDE via apt (independent of conda), so it can be
#     called from any shell to detach long-running run.ai training/inference jobs.
#     Non-fatal: never abort the install if tmux can't be installed here.
install_tmux() {
    if command -v tmux >/dev/null 2>&1; then
        echo "==> tmux already installed ($(tmux -V)) — skipping"
        return 0
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "==> WARNING: apt-get not found; install tmux manually (e.g. 'apt-get install -y tmux')." >&2
        return 0
    fi
    local SUDO=""
    if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
    echo "==> Installing tmux (system-wide via apt)"
    # Refresh package lists best-effort. On run.ai CUDA images the NVIDIA repo can
    # make `apt-get update` exit non-zero (GPG/key warnings), but the Ubuntu lists
    # that actually carry tmux still refresh — so do NOT gate the install on the
    # update exit code (the old `update && install` chain silently skipped install).
    $SUDO apt-get update -qq || echo "==> apt-get update reported errors (continuing anyway)" >&2
    if $SUDO apt-get install -y tmux && command -v tmux >/dev/null 2>&1; then
        echo "==> tmux installed ($(tmux -V))"
    else
        echo "==> WARNING: could not install tmux via apt; install it manually with 'apt-get install -y tmux'." >&2
    fi
    return 0
}
install_tmux

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
    echo "==> Running tests: pytest tests/ -v -s"
    # Non-fatal: the env build above is what matters, so don't abort the install if a
    # test fails (e.g. a GPU-dependent test on a node with a bad CUDA context, or a
    # data-dependent test with datasets absent on this machine).
    pytest tests/ -v -s || \
        echo "==> WARN: test step had failures — env is still installed; see tests/README.md" >&2
fi

echo
echo "==> Install complete. Activate with: conda activate $ENV_NAME"

# tmux quick-start — get a detachable training run going immediately.
cat <<EOF

============================================================
 tmux quick-start — detachable training on run.ai
============================================================
  # 1. Create a detachable session named "smil-train"
  tmux new-session -s smil-train

  # 2. Inside the session: activate the env + launch an example training run
  conda activate $ENV_NAME
  python -m smal_fitter.neuralSMIL.train_multiview_regressor \\
      --config smal_fitter/neuralSMIL/configs/examples/multiview_baseline.json

  # 3. Detach (leave training running in the background): press  Ctrl-b  then  d

  # 4. Re-attach later from any shell:
  tmux attach -t smil-train
============================================================
EOF
