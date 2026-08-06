#!/bin/bash
# RE-VALIDATE THE GATE ON A WORKING MODEL. DO NOT EDIT WHILE RUNNING.
#
# The gate (§6.2) was validated with w_beta_prior=0.002 in both files, i.e. with the shape
# space closed and betas sd at 0.0004. Its held-out result was strong (deform_mag +38.9%,
# p=4.5e-13) but every arm in it shared that defect. Now that the prior is off by default --
# betas sd 0.318, and the only movement in the correspondence metric this investigation has
# produced -- the gate has to be re-tested on a model that can actually use its shape range.
#
# Identical protocol to the original: heldout_easy (50 lowest radial_med) vs heldout_rand
# (50 uniform), both drawn from the 707 workers NOT in bench50, same recipe, same seed.
#
# PRE-REGISTERED: the gate holds if easy still beats rand on the TARGET-INDEPENDENT metrics
# (deform_mag, edge_logratio) at p<0.01. If the advantage vanishes on a working model, then
# radial_med was selecting for "specimens a crippled model happens to fit", not fittability,
# and §6.2 has to be withdrawn.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
R=diagnostics/moonshot/runs
arm () {
  local g=$1 t=$2
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical \
    --mesh_dir diagnostics/moonshot/heldout_$t --midline 2.0 \
    --results_dir $R/G2_${t}_hier >$R/G2_$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot \
    --mesh_dir diagnostics/moonshot/heldout_$t \
    --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
    --init_from $R/G2_${t}_hier/H2_joint.npz --results_dir $R/G2_$t >>$R/G2_$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/G2_$t --mesh_dir diagnostics/moonshot/heldout_$t --all_stages >>$R/G2_$t.log 2>&1
  echo "[$(date +%H:%M:%S)] G2_$t done"
}
arm 0 easy &
arm 1 rand &
wait
echo "[$(date +%H:%M:%S)] gate re-validation complete"
