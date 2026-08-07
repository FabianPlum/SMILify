#!/bin/bash
# E8b -- the deform ablation on the GROUND-TRUTH corpus, because probe-19 cannot judge it.
#
# E8 on workers showed D2_frozen at probe-19 gen/spread 0.398, past the 0.42 that exact
# correspondence achieves. That number is an ARTEFACT: with deform_mag = 0.000185 the
# rest-space geometry is v_template + shapedirs.betas exactly, i.e. confined to a
# 25-dimensional linear subspace BY CONSTRUCTION. probe-19 measures how low-dimensional the
# registrations are, so a near-zero gen@20 is guaranteed by the ablation whether or not the
# correspondence is anatomically correct.
#
# The instrument that cannot be fooled this way is E6's direct round trip: targets generated
# FROM the model, so fitted vertex i is supposed to land on generated vertex i.
# Baseline to beat: SYN_clean, 4.81% strict correctness, 3.48% median error (SYNTHETIC_ROUNDTRIP.md).
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean
arm () {
  local g=$1 t=$2 cfg=$3 off=$4
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset $off \
    --results_dir $R/${t}_hier >$R/$t.log 2>&1 || { echo "[$t] HIER FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
    --yaml_src diagnostics/moonshot/cfg/${cfg}.yaml --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1 || { echo "[$t] MOONSHOT FAILED"; return 1; }
  echo "[$(date +%H:%M:%S)] $t done"
}
arm 0 SYN_D1 D1_SYN 30.0 &
arm 1 SYN_D2 D2_SYN 300.0 &
wait
python -u diagnostics/moonshot/score_synth_roundtrip.py \
  --pairs SYN_clean:synth_clean SYN_D1:synth_clean SYN_D2:synth_clean --render_n 2
