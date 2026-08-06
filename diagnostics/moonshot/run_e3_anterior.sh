#!/bin/bash
# E3 -- THE ANTERIOR FIX. DO NOT EDIT WHILE RUNNING.
#
# REPORT §6.10: head/mandible/antenna are folded into one 'body' partition group, so covering
# the head with mandible vertices costs the chamfer nothing. Measured consequence in every
# pose-mobile arm: head deforms 24-40% LESS than thorax, antennae 44-82% MORE.
# The "too small for a data term" argument in the code was false -- head is 19.5% of surface
# area (1563 samples at n=8000), 3x any leg group, which DOES get its own term.
#
#   ANT_split  --split_anterior, limits on   vs   control LIM_1x (limits on, anterior folded)
#   Single variable.
#
# PRE-REGISTERED READING:
#   1. PRIMARY (mechanism): head deform ratio must rise from 0.72x toward 1.0x AND antenna
#      must fall from 1.72x toward 1.0x. That is the failure mode, stated as a number.
#   2. SECONDARY: probe-19 gen/spread vs LIM_1x's 0.9336. This is the first PART-AWARE
#      intervention; E1 showed a pose-space fix does nothing, so a data-term fix is a
#      genuinely different lever.
#   3. Surface metrics may worsen (§3) and that alone is not a failure.
#   4. If the deform ratios do NOT move, the partition is not what drives the anterior
#      failure and the mechanism is elsewhere (per-joint scale is the next suspect --
#      nothing bounds it, baseline antenna scale ranges 889x).
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs
CUDA_VISIBLE_DEVICES=0 python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --split_anterior \
  --results_dir $R/ANT_split_hier >$R/ANT_split.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
  --yaml_src diagnostics/moonshot/cfg/M7_anterior.yaml --init_from $R/ANT_split_hier/H2_joint.npz \
  --results_dir $R/ANT_split >>$R/ANT_split.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u diagnostics/moonshot/eval_run.py \
  --run_dir $R/ANT_split --mesh_dir $MESH --all_stages >>$R/ANT_split.log 2>&1
echo "[$(date +%H:%M:%S)] ANT_split done"
