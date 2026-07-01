#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=ias-7
#SBATCH --partition=dc-cpu
#SBATCH --time=04:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# SLEAP multi-view dataset preprocessing: video decode, keypoint extraction, HDF5 write.
# Pure CPU work (no CUDA, not distributed) — plain single-process python on the CPU partition.
# Submit from the repo root:
#   sbatch hpc_files/sc_venv_pytorch3d/run_multiview_preprocess_JURECA.sbatch

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# Activate the venv (config.sh exports SMILIFY_DIR = repo root), then run from the repo
# root so the absolute `smal_fitter.*` / `config` imports resolve.
source "${SLURM_SUBMIT_DIR}/hpc_files/sc_venv_pytorch3d/activate.sh"
cd "${SMILIFY_DIR}"

# --- edit for your run (paths are relative to the repo root) ---
DATA_DIR=data/sleap_sessions
OUTPUT=SMILymice_3D_18_cam_NEW_MODEL_FULL.h5
JOINT_LOOKUP="$DATA_DIR/lookup_table_names_Falkner_MICE_NEW.csv"
SHAPE_BETAS="$DATA_DIR/lookup_table_PCs_Falkner_MICE.csv"

srun --export=ALL \
    python -m smal_fitter.sleap_data.preprocess_sleap_multiview_dataset \
        "$DATA_DIR" \
        "$OUTPUT" \
        --joint_lookup_table "$JOINT_LOOKUP" \
        --shape_betas_table "$SHAPE_BETAS"
