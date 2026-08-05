#!/bin/bash
# DOES THE GATED SHAPE SPACE HELP? DO NOT EDIT WHILE RUNNING.
#
# SMIL_OmniAnt_gated.pkl was built by PCA over the 50 GATE_easy_M7 registrations (20
# directions, fold-gated against the inputs' own p90 validity). This tests it on
# heldout_easy2 -- 50 specimens disjoint from bench50 AND from both earlier held-out sets,
# so no specimen used to BUILD the shape space is used to TEST it.
#
# Two arms, identical in every respect except the model file:
#   S0_oldmodel   SMIL_OmniAnt.pkl        13 shape directions (the original)
#   S1_gated      SMIL_OmniAnt_gated.pkl  20 shape directions (rebuilt from gated fits)
#
# Both are SCORED with the default model, because eval_run.py uses only `weights` and
# `J_names`, which are byte-identical between the two -- so the metric definitions, including
# the part segmentation, are the same for both arms.
#
# PRE-REGISTERED READING, fixed before the run:
#   * Report section 4 measured that enlarging the shape space 13 -> 37 directions moved
#     fscore@0.02 only 0.7313 -> 0.7356, i.e. shape capacity is NOT the binding constraint.
#     The prior is therefore that this does NOT help, and a null result confirms section 4
#     rather than wasting the artefact.
#   * A real win requires improvement on the TARGET-INDEPENDENT metrics (edge_logratio,
#     deform_mag), not just surface proximity -- section 3 rules out judging on the latter.
#   * A surface-only win with WORSE deform_mag means the larger space is being used to
#     shrink-wrap harder, which is the failure mode this whole investigation is about.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
MESH=diagnostics/moonshot/heldout_easy2
R=diagnostics/moonshot/runs

arm () { # arm <gpu> <tag> <pkl>
  local g=$1 t=$2 pkl=$3
  SMILIFY_SMAL_FILE=$pkl CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical \
    --mesh_dir $MESH --midline 2.0 --results_dir $R/$t >$R/$t.log 2>&1
  SMILIFY_SMAL_FILE=$pkl CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot \
    --mesh_dir $MESH --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
    --init_from $R/$t/H2_joint.npz --results_dir $R/${t}_M7 >>$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/${t}_M7 --mesh_dir $MESH --all_stages >>$R/$t.log 2>&1
  echo "[$(date +%H:%M:%S)] $t done"
}

arm 0 S0_oldmodel 3D_model_prep/SMIL_OmniAnt.pkl &
arm 1 S1_gated    3D_model_prep/SMIL_OmniAnt_gated.pkl &
wait
echo "[$(date +%H:%M:%S)] shape-space A/B complete"
