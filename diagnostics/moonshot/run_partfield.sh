#!/bin/bash
# Part-field training variants. DO NOT EDIT WHILE RUNNING -- bash re-reads the file as it
# executes and an in-place edit corrupts the running script (this bit twice already this
# session). Copy to a _v2 and edit that instead.
#
# Three variants, so the report can separate what the architecture does from what the
# augmentation does from what the in-domain data does:
#   A_arch   synthetic only, NO mirror augmentation  -> honest G2 number for the architecture
#   B_mirror synthetic only, mirror augmentation      -> the shippable model
#   C_real   B + fit-derived real-weak labels         -> does in-domain data help or just
#                                                        re-import the fit's own errors?
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
D=diagnostics/moonshot/partfield
L=diagnostics/moonshot/runs
EP=${EP:-60}

run () {  # run <gpu> <name> <extra args...>
  local g=$1 n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/train_partfield.py \
    --epochs $EP --out $D/net_$n.pt "$@" > $L/partfield_$n.log 2>&1
  echo "[$(date +%H:%M:%S)] $n done"
}

case "${1:-gpu1}" in
  gpu1)  run 1 B_mirror --mirror_aug & \
         run 1 C_real   --mirror_aug --real_weight 0.5 & wait ;;
  gpu0)  run 0 A_arch ;;
  *)     run 1 B_mirror --mirror_aug ;;
esac
echo "[$(date +%H:%M:%S)] partfield batch complete"
