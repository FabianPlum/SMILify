#!/bin/bash
# GPU0 queue: variants C then A, sequentially, so neither contends with B on GPU1.
# Separate file from run_partfield.sh deliberately -- editing a script bash is currently
# reading corrupts it, so queues get new files rather than edits.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
D=diagnostics/moonshot/partfield
L=diagnostics/moonshot/runs

CUDA_VISIBLE_DEVICES=0 python -u diagnostics/moonshot/train_partfield.py \
  --epochs 60 --out $D/net_C_real.pt --mirror_aug --real_weight 0.5 \
  > $L/partfield_C_real.log 2>&1
echo "[$(date +%H:%M:%S)] C_real done"

CUDA_VISIBLE_DEVICES=0 python -u diagnostics/moonshot/train_partfield.py \
  --epochs 60 --out $D/net_A_arch.pt \
  > $L/partfield_A_arch.log 2>&1
echo "[$(date +%H:%M:%S)] A_arch done"
