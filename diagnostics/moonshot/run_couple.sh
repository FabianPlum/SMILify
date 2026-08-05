#!/bin/bash
# THE §6.6 FIX: drive per-joint scale/translation from the betas. DO NOT EDIT WHILE RUNNING.
#
# §6.6 measured that `scaledirs` and `transdirs` -- per-joint scale/translation blendshapes
# shipped in the model at (13,55,3) and PCA-derived jointly with `shapedirs` -- were never
# referenced by smal_torch.py. The fitter instead optimised `log_beta_scales`/`betas_trans` as
# 165 free per-specimen parameters that duplicate them, and being unregularised they carried
# 70-400x more inter-specimen variation than the betas. Shape therefore lived in a pose-space
# parameter that does not transfer, which is why no shape space could be rebuilt from the
# registrations.
#
# Now wired in behind SMILIFY_COUPLE_JOINT_BLENDSHAPES=1, with the free parameters kept as a
# RESIDUAL on top so a specimen can still deviate carefully where the shape space cannot
# reach. Verified live: beta0 at +-2 sigma moves vertices 59.3% further with coupling on.
#
#   CPL_a  coupling ON,  no residual penalty   -- does coupling alone shift the balance?
#   CPL_b  coupling ON,  residual penalty 5.0  -- forced: betas must carry the variation
#   control: M1_sym (already scored) -- coupling OFF, identical in every other respect
#
# PRE-REGISTERED READING, fixed before the run. The claim is about where shape is STORED, so
# the outcomes are about that, not about surface fit:
#   1. betas sd across specimens must RISE substantially from M1_sym's 0.00189. If it does
#      not, the coupling is not changing the optimiser's preference and the fix is inert.
#   2. the ratio sd(log_beta_scales)/sd(betas) must FALL from M1_sym's 70x.
#   3. probe-19 gen/spread must move off 0.99 toward the synthetic control's 0.42. This is the
#      one that matters -- it is the only outcome that says the registrations became usable.
#   4. surface metrics may legitimately worsen (§3), and a surface win with no change in 1-3
#      would mean the coupling is just another way to shrink-wrap.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs

arm () { # arm <gpu> <tag> <jresid>
  local g=$1 t=$2 jr=$3
  SMILIFY_COUPLE_JOINT_BLENDSHAPES=1 CUDA_VISIBLE_DEVICES=$g \
    python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH --midline 2.0 --jresid $jr \
    --results_dir $R/$t >$R/$t.log 2>&1
  SMILIFY_COUPLE_JOINT_BLENDSHAPES=1 CUDA_VISIBLE_DEVICES=$g \
    python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
    --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
    --init_from $R/$t/H2_joint.npz --results_dir $R/${t}_M7 >>$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/${t}_M7 --mesh_dir $MESH --all_stages >>$R/$t.log 2>&1
  echo "[$(date +%H:%M:%S)] $t done"
}

arm 0 CPL_a 0.0 &
arm 1 CPL_b 5.0 &
wait
echo "[$(date +%H:%M:%S)] coupling arms complete"
