#!/bin/bash
# CAUSAL TEST of why M9a lost. DO NOT EDIT WHILE RUNNING.
#
# M9a (part field, debris EXCLUDED) lost decisively against its control M8a_split_only:
# part_leg_distal -85.4% (5/50, p=4.2e-09), chamfer -47.6%, part_body -25.8%.
#
# Probe 16 supplies a candidate mechanism that has nothing to do with the partition:
# 90.8% of the points the field calls "debris" are INSIDE the animal's convex hull, a median
# 0.031 from the nearest body point. They are the leg-body junction, not debris -- the
# network learned that from the synthetic "debris bridge" augmentation, which puts filaments
# exactly there. Excluding them deletes a mean 9.5% of target points (max 55%) concentrated
# at the coxae, which is the single region the hierarchical schedule exists to place.
#
# M9d is M9a with ONE change: --pf_keep_debris reassigns those points to their nearest
# non-debris class instead of dropping them. So:
#   M9d ~= M8a  ->  the debris exclusion was the whole story; the partition itself is fine.
#   M9d still loses  ->  the frozen target-derived partition is genuinely worse, and the
#                        debris artefact was a side issue.
# Either outcome is informative, which is the point of running it.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify

MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs
NET=${NET:-diagnostics/moonshot/partfield/net_B_mirror.pt}
G=${1:-0}

CUDA_VISIBLE_DEVICES=$G python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir $MESH --split_distal --midline 2.0 --part_field $NET --pf_keep_debris \
  --results_dir $R/M9d_keepdebris >$R/M9d_keepdebris.log 2>&1
CUDA_VISIBLE_DEVICES=$G python -u diagnostics/moonshot/eval_run.py \
  --run_dir $R/M9d_keepdebris --mesh_dir $MESH --all_stages >>$R/M9d_keepdebris.log 2>&1
echo "[$(date +%H:%M:%S)] M9d_keepdebris done"
