#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --time=04:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Multi-node, multi-GPU distributed inference with the multi-view SMIL regressor (torchrun_jsc).
# run_multiview_inference.py shards the dataset across ranks (DistributedSampler) and gathers results.
# Submit from the repo root:
#   sbatch hpc_files/sc_venv_pytorch3d/run_multiview_inference_JURECA.sbatch

# Backbone weights are pre-fetched on the login node; keep HF/timm off the network.
export HF_DATASETS_OFFLINE=1

# One task per node; expose the task's CPU allocation to srun (DataLoader workers).
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun rendezvous: use the first node of the allocation as the master.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010

# Activate the venv (config.sh exports SMILIFY_DIR = repo root), then run from the repo
# root so the absolute `smal_fitter.*` / `config` imports resolve.
source "${SLURM_SUBMIT_DIR}/hpc_files/sc_venv_pytorch3d/activate.sh"
cd "${SMILIFY_DIR}"

# --- edit for your run ---
DATASET=SMILymice_3D_18_cam_NEW_MODEL.h5
SMAL_FILE=3D_model_prep/SMILy_Mouse_static_joints_Falkner_conv_repose_hind_legs.pkl
CHECKPOINT=multiview_checkpoints/best_model.pth

# One torchrun agent per node; the entrypoint runs as a module (python -m equivalent).
srun --export=ALL --cpu_bind=none bash -c "torchrun_jsc \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=gpu \
    --rdzv_backend c10d \
    --rdzv_id $RANDOM \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
    -m smal_fitter.neuralSMIL.run_multiview_inference \
        --dataset $DATASET \
        --smal_file $SMAL_FILE \
        --checkpoint $CHECKPOINT"
