#!/bin/bash
# E5 -- USE THE SHAPE SPACE PROPERLY. DO NOT EDIT WHILE RUNNING.
#
# The model ships scaledirs/transdirs: per-joint scale and translation BLENDSHAPES indexed by
# beta, produced by the addon's entangled PCA (shape+scale+translation decomposed jointly).
# They are the disentangled shape space. The optimisation fitter has never used them: they are
# read only under config.COUPLE_JOINT_BLENDSHAPES, which defaults off.
#
# §6.7 previously tested turning that on and reported a null. THAT TEST IS INVALID. The
# transform it used (commit 32e8260) treated scaledirs.beta as already-log-scale and omitted
# translation_factor. Measured against the canonical implementations
# (neuralSMIL _transform_separate_pca_weights_to_joint_values; Unreal2Pytorch3D.py:157-159):
#   log-scale differed by up to 8.17 in log space  = a 3532x scale ratio
#   translation was exactly 100x too large
# smal_torch.py now matches both canonical paths to 5.7e-06 / 2.3e-10. This re-runs the test.
#
#   CPX_on   coupling ON (corrected) + jresid 1.0 + scale barrier
#            betas drive per-joint scale/translation; the free per-specimen parameters remain
#            as a PENALISED residual, so parts can still be nudged individually ("to a lesser
#            degree") but the bulk of the variation has to live in the shape space.
#   control: LIM_1x / SCL_1x (coupling OFF), already scored.
#
# PRE-REGISTERED READING:
#   1. WIRING: betas sd must rise. If the shape space is finally carrying the variation, this
#      is where it shows. LIM_1x sits at 0.278.
#   2. PRIMARY: probe-19 gen/spread must fall below LIM_0's 0.9221 -- the best any worker arm
#      has reached. This is the metric that says the registrations became transferable.
#   3. sd(log_beta_scales)/sd(betas) must FALL from the 70-400x measured in §6.6: that ratio
#      is the statement "shape lives in a pose-space parameter", which is what coupling fixes.
#   4. If betas sd rises but gen/spread does not move, the shape space is being used and still
#      does not transfer, which would point the remaining problem at correspondence itself
#      rather than at the parameterisation.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
export SMILIFY_COUPLE_JOINT_BLENDSHAPES=1
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs
CUDA_VISIBLE_DEVICES=0 python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --scale_cap 10.7 --jresid 1.0 \
  --results_dir $R/CPX_on_hier >$R/CPX_on.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
  --yaml_src diagnostics/moonshot/cfg/M7_couple.yaml --init_from $R/CPX_on_hier/H2_joint.npz \
  --results_dir $R/CPX_on >>$R/CPX_on.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u diagnostics/moonshot/eval_run.py \
  --run_dir $R/CPX_on --mesh_dir $MESH --all_stages >>$R/CPX_on.log 2>&1
echo "[$(date +%H:%M:%S)] CPX_on done"
