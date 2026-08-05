#!/bin/bash
# M9e — the best-shot configuration: the full part-field package WITH the debris fix.
# DO NOT EDIT WHILE RUNNING.
#
# M9a (field, debris excluded)                    lost badly to control M8a.
# M9c (field + pf_body + pf_init, debris excluded) lost by the same amount -- so neither the
#     H0 body restriction nor a near-perfect correspondence-free pose init (anchor loss
#     0.0409 -> 3.4e-05) changed the outcome. Whatever is hurting is upstream of both.
# Probe 16 says what: 90.8% of predicted "debris" is INTERIOR to the animal and the class
#     correlates -0.879 with reproducibility, i.e. it is the network's "I don't know"
#     channel, and excluding it deletes the leg-body junction.
#
# M9e = M9c + --pf_keep_debris. Run in parallel with M9d (which is M9a + keep_debris), so
# the two together separate the debris fix from the pf_body/pf_init features:
#     M9d - M9a  =  effect of keeping debris, alone
#     M9e - M9c  =  effect of keeping debris, with the features
#     M9e - M8a  =  the honest bottom line for the whole approach
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs
NET=diagnostics/moonshot/partfield/net_B_mirror.pt
G=${1:-1}
CUDA_VISIBLE_DEVICES=$G python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir $MESH --split_distal --midline 2.0 --part_field $NET \
  --pf_keep_debris --pf_body --pf_init 400 \
  --results_dir $R/M9e_full >$R/M9e_full.log 2>&1
CUDA_VISIBLE_DEVICES=$G python -u diagnostics/moonshot/eval_run.py \
  --run_dir $R/M9e_full --mesh_dir $MESH --all_stages >>$R/M9e_full.log 2>&1
echo "[$(date +%H:%M:%S)] M9e_full done"
