#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --time=01:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Single-GPU benchmark / evaluation of a trained multi-view SMIL checkpoint.
# benchmark_model.py is NOT distributed — run it as a plain single-process job (no torchrun).
# Submit from the repo root:
#   sbatch hpc_files/sc_venv_pytorch3d/run_multiview_benchmark_JURECA.sbatch

# Backbone weights are pre-fetched on the login node; keep HF/timm off the network.
export HF_DATASETS_OFFLINE=1
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# Activate the venv (config.sh exports SMILIFY_DIR = repo root), then run from the repo
# root so the absolute `smal_fitter.*` / `config` imports resolve.
source "${SLURM_SUBMIT_DIR}/hpc_files/sc_venv_pytorch3d/activate.sh"
cd "${SMILIFY_DIR}"

# --- edit for your run ---
DATASET=STICKS_full_512_3D.h5
CHECKPOINT=multiview_checkpoints_Unet_512_EffB3/checkpoint_epoch_0279.pth

srun --export=ALL --cpu_bind=none \
    python -m smal_fitter.neuralSMIL.benchmark_model \
        --dataset_path "$DATASET" \
        --checkpoint "$CHECKPOINT"
