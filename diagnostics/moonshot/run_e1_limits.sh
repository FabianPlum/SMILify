#!/bin/bash
# E1 -- DO THE AUTHORED JOINT LIMITS FIX CORRESPONDENCE? DO NOT EDIT WHILE RUNNING.
#
# Everything correspondence-side has failed (§8: partitioning, robust kernels, correspondence
# filters, geodesic decomposition, the learned part field, shape-space enlargement -- six
# nulls). The two interventions that DID work, M7 and A4, both work by letting pose keep
# moving. §8 concluded the remaining lever is pose. This is the first experiment that
# constrains pose rather than freeing it.
#
# THE OBSERVATION THAT MOTIVATES IT (measured, this session):
#   The arms that unfreeze pose drive 37 of 96 constrained axes out of anatomical range on
#   100% of specimens, median 19.4 deg, p95 77.5 deg, concentrated on l_3_ta_* and l_3_ti_*.
#   The stock baseline violates 3.8 axes only because §1's freeze stops it moving pose at all.
#   §4 shows why those joints in particular: tibia+tarsus+pretarsus hold 2.4% of leg-chain
#   area, so the pretarsus draws ~0.3 samples at n_sample=8000. They are unconstrained by the
#   data term, and pose freedom is spent bending them into impossible configurations.
#   The joint limits are the only term in the pipeline that acts on those joints at all.
#
# WEIGHTS ARE MEASURED, NOT INHERITED (probe_20_limit_calibration.py):
#   PR #98 ships w_limit=100 for fitter_3d/trainer.py, where the hinge is ~9e-4 and that is
#   well scaled. Here the hinge is 0.122 and the hierarchical chamfer is 0.0334, so 100 would
#   be 366x the data term. Parity is 0.273. The moonshot handoff runs a different chamfer
#   (0.00071), so its parity is 0.006 -- a 47x difference, and using one number for both would
#   silently make the term meaningless in one pipeline or dominant in the other.
#   Carrying a weight across a regime change untested is exactly the §7.2 / w_beta_prior error.
#
#   LIM_0    hier 0      / moonshot 0        -- control, same 25-PC template
#   LIM_1x   hier 0.273  / moonshot 0.006    -- limit term at parity with chamfer
#   LIM_3x   hier 0.819  / moonshot 0.018    -- limits dominate
#
# PRE-REGISTERED READING, fixed before the run:
#   1. PRIMARY: probe-19 gen/spread. The claim is about correspondence, so it is judged by the
#      correspondence instrument, not by surface metrics (§3). BPX_noprior reached 0.9353, the
#      best this investigation has achieved; the synthetic exact-correspondence control is
#      0.42 and the ALL_ANTS_CLEAN registrations reach 0.5383. A win means LIM_* drops
#      meaningfully below 0.9353. Anything >= 0.93 is a null.
#   2. MECHANISM: violating axes per specimen must fall from ~37 toward 0. If it does not, the
#      weight is too low and the arm says nothing about the hypothesis.
#   3. Surface metrics may legitimately worsen (§3). A surface win with no movement in 1 is
#      NOT a success -- it would mean the limits are just another way to restrict deformation.
#   4. If gen/spread does not move at ANY weight while violations go to zero, then
#      anatomically implausible distal pose is NOT what is destroying correspondence, and the
#      distal joints should be dropped as a hypothesis rather than re-weighted.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs

arm () { # arm <gpu> <tag> <hier_w> <yaml>
  local g=$1 t=$2 w=$3 y=$4
  echo "[$(date +%H:%M:%S)] $t start (hier w_limit=$w, $y)"
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH \
    --midline 2.0 --beta_prior 0.0 --limit $w --results_dir $R/${t}_hier >$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
    --yaml_src diagnostics/moonshot/cfg/$y --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/$t --mesh_dir $MESH --all_stages >>$R/$t.log 2>&1
  echo "[$(date +%H:%M:%S)] $t done"
}

arm 0 LIM_0  0.0   M7_limit0.yaml &
arm 1 LIM_1x 0.273 M7_limit1x.yaml &
wait
arm 0 LIM_3x 0.819 M7_limit3x.yaml
echo "[$(date +%H:%M:%S)] E1 complete"
