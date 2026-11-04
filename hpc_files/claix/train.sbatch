#!/usr/bin/env bash
#SBATCH --job-name=smil_train
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Holotype multi-view DDP training on RWTH CLAIX (c23g / c25g: 4 GPUs per node).
#
# One job is one wall-clock link of a run. It resumes from the newest
# checkpoint_epoch_*.pth in the config's output.checkpoint_dir (falling back to
# the config's own resume_checkpoint), and exits 0 without training when the run
# already wrote final_model.pth. Chain links with submit_chain.sh.
#
# Submit from the repo root of a pinned checkout on /hpcwork (logs/ must exist: the --output path is
# resolved at submit time):
#   mkdir -p logs
#   CONFIG=configs_claix/run.json sbatch --account=<project> --export=ALL,CONFIG hpc_files/claix/train.sbatch
#
# --time: Slurm's default is 15 min. The maximum is set per project (r_wlm_usage -q -p
# <project>; Uness's rwth2151: 24 h). On c23g, jobs over 24 h may use only half the nodes.
# More GPUs: add --nodes=2 (8 ranks). batch_size is PER RANK under DDP, so the
# effective batch grows with the rank count; adjust the config if that matters.
# c25g (CLAIX-2025) is a pilot as of 2026-10-28: NHR/WestAI projects only, ML jobs
# 1 node max, limited storage bandwidth. If your project may use it, 1-node jobs can
# take --partition=c23g,c25g (Slurm starts the job where a node frees first).
#
# Env:
#   CONFIG      multi-view training config JSON (required; paths valid on CLAIX and
#               checkpoint cadence set: rebase_config.py --set output.save_checkpoint_every=N)
#   ENV_PREFIX  conda env prefix (default: $HPCWORK/miniforge3/envs/pytorch3d)
#   EXTRA_ARGS  extra trainer CLI arguments
#
# NOTE: intentionally no `set -u`: conda's activate hooks reference unset vars.
set -eo pipefail

: "${CONFIG:?set CONFIG=<config.json> and submit with --export=ALL,CONFIG}"
ENV_PREFIX="${ENV_PREFIX:-$HPCWORK/miniforge3/envs/pytorch3d}"
cd "${SLURM_SUBMIT_DIR:-.}"

# --- environment ---------------------------------------------------------------
# GPU nodes have no outbound network: backbone weights must already be in the
# cache (python hpc_files/download_backbone_weights.py on a login node, same HF_HOME).
export HF_HOME="${HF_HOME:-$HPCWORK/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HPCWORK/torch_cache}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" # INFO on a first multi-node run: check for NET/IB, not NET/Socket

if [[ -z "${CONDA_BASE:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then CONDA_BASE="$(conda info --base)"; else CONDA_BASE="${ENV_PREFIX%/envs/*}"; fi
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# --- resume point --------------------------------------------------------------
read -r CKPT_DIR NUM_EPOCHS < <(python - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg["output"]["checkpoint_dir"], cfg["training"]["num_epochs"])
PY
)
if [[ -f "$CKPT_DIR/final_model.pth" ]]; then
    echo "Run already finished ($CKPT_DIR/final_model.pth exists); nothing to do."
    exit 0
fi
# Newest periodic checkpoint that is a complete file. torch.save is not atomic,
# so a wall kill during a save leaves a truncated zip; skip it rather than let
# every later link of the chain fail on it.
RESUME="$(python - "$CKPT_DIR" <<'PY'
import glob, re, sys, zipfile
paths = glob.glob(f"{sys.argv[1]}/checkpoint_epoch_*.pth")
for p in sorted(paths, key=lambda p: int(re.search(r"(\d+)\.pth$", p).group(1)), reverse=True):
    try:
        zipfile.ZipFile(p).close()
        print(p)
        break
    except zipfile.BadZipFile:
        print(f"WARNING: skipping truncated checkpoint {p}", file=sys.stderr)
PY
)"
RESUME_ARGS=()
[[ -n "$RESUME" ]] && RESUME_ARGS=(--resume_checkpoint "$RESUME")

# --- launch --------------------------------------------------------------------
# One torchrun per node (srun --ntasks-per-node=1), torchrun spawns one rank per
# GPU. The trainer detects RANK/LOCAL_RANK/WORLD_SIZE and builds its own TCPStore
# on the MASTER_PORT torchrun hands it. Proven on CLAIX with torch 2.3.1 (Uness's
# 2-node runs, Sep 2026). torch >= 2.4 shares torchrun's store on that port
# instead: re-check the "Successfully initialized NCCL" line after an upgrade.
NNODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
RDZV_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname) - $(date)"
echo "  partition ${SLURM_JOB_PARTITION} | account ${SLURM_JOB_ACCOUNT}${SLURM_JOB_END_TIME:+ | ends $(date -d @"$SLURM_JOB_END_TIME")}"
echo "  ranks    : $NNODES node(s) x $GPUS_PER_NODE GPU(s) = $((NNODES * GPUS_PER_NODE))"
echo "  config   : $CONFIG (num_epochs $NUM_EPOCHS)"
echo "  resume   : ${RESUME:-from the config}"
echo "  env      : $CONDA_PREFIX"
python -c "import torch; print(f'  torch    : {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}')"
echo "=================================================================="

srun --nodes="$NNODES" --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-node="$GPUS_PER_NODE" \
    torchrun \
    --rdzv-backend=c10d \
    --rdzv-id="smil_$SLURM_JOB_ID" \
    --rdzv-endpoint="$HEAD_NODE:$RDZV_PORT" \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    -m smal_fitter.neuralSMIL.train_multiview_regressor --config "$CONFIG" "${RESUME_ARGS[@]}" ${EXTRA_ARGS:-}

echo "Done - $(date)"
