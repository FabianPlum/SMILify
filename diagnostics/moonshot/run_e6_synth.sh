#!/bin/bash
# E6 -- THE GROUND-TRUTH ROUND TRIP. DO NOT EDIT WHILE RUNNING.
#
# Every correspondence number so far is INDIRECT. probe-19 measures whether the fit's vertex
# placement is CONSISTENT across a population, never whether it is CORRECT; a systematically
# wrong but reproducible correspondence scores perfectly. Its synthetic control validated the
# METRIC (shapes drawn from the shape space) and never the PIPELINE.
#
# Here the targets are generated FROM the model, so fitted vertex i is supposed to land on
# generated vertex i and we can check that directly. CEILING TEST: if the pipeline cannot
# establish correspondence on its own geometry, it cannot on ethanol-preserved worker scans.
#
#   SYN_clean  noise 0      -- the absolute ceiling
#   SYN_noisy  noise 0.005  -- 0.25% of extent, bracketing real scan roughness
#
# PRE-REGISTERED READING:
#   1. PRIMARY: fraction of vertices whose OWN ground-truth vertex is the nearest GT vertex.
#      This is correspondence correctness with no proxy. A pipeline that registers should be
#      near 1.0 on the clean arm.
#   2. per-vertex error to GT, as % of specimen extent, overall and per anatomical part.
#   3. probe-19 gen/spread on the same fits, so the indirect metric can be CALIBRATED against
#      the direct one for the first time. If probe-19 reads ~0.9 while correctness is high,
#      probe-19 is not measuring what it has been used for and every arm ranked by it needs
#      re-reading. If both agree, probe-19 is vindicated as a proxy.
#   4. If correctness is LOW even on SYN_clean, the pipeline does not establish correspondence
#      at all, and the five preceding nulls were comparing degrees of wrongness.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
arm () {
  local g=$1 t=$2 d=$3
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir diagnostics/moonshot/$d \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --results_dir $R/${t}_hier >$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir diagnostics/moonshot/$d \
    --yaml_src diagnostics/moonshot/cfg/M7_${t}.yaml --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1
  echo "[$(date +%H:%M:%S)] $t done"
}
arm 0 SYN_clean synth_clean &
arm 1 SYN_noisy synth_noisy &
wait
echo "[$(date +%H:%M:%S)] E6 complete"
