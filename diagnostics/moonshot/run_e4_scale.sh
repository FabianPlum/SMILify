#!/bin/bash
# E4 -- BOUND PER-JOINT SCALE. DO NOT EDIT WHILE RUNNING.
#
# §6.10.1 left exactly one untested mechanism for the anterior failure: nothing bounds
# log_beta_scales. Measured at Stage_3 across 50 specimens (of 8250 joint-axes):
#   baseline   max 165x scale, 4035 axes >2x, 1843 >4x, 657 >8x
#   LIM_0      max 11.5x,       206 >2x,   10 >4x,   1 >8x
#   LIM_1x     max 16.7x,       378 >2x,   59 >4x,  15 >8x   <- limits made scale WORSE
# i.e. forbidding out-of-range ROTATION pushed the error into SCALE, 1 -> 15 axes past 8x.
#
# scale_barrier: zero inside a 2x free band, quadratic past it. Band from domain judgement
# (2x unremarkable, 4x concerning, 8x improbable), not fitted.
# Weights measured, not inherited: LIM_0 barrier 0.00313 against hierarchical chamfer 0.0334
# -> parity 10.7; moonshot chamfer 0.00071 against barrier 0.0136 -> parity 0.052.
#
#   SCL_1x  hier 10.7 / moonshot 0.052   -- barrier at parity with chamfer
#   SCL_3x  hier 32.0 / moonshot 0.157   -- barrier dominates
#   control: LIM_1x (limits on, no scale bound), already scored
#
# PRE-REGISTERED READING:
#   1. MECHANISM: axes >4x and >8x must fall toward 0. If not, the weight is too low.
#   2. PRIMARY (the anterior claim): head deform ratio must rise from 0.72x toward 1.0x.
#      Same outcome E3 failed on; unbounded scale is the remaining suspect.
#   3. SECONDARY: probe-19 gen/spread vs LIM_1x 0.9336 / LIM_0 0.9221.
#   4. If scale is bounded but the head ratio still does not move, the head being
#      under-carried is NOT explained by partition, rotation limits, or scale, and the cause
#      is upstream of the model's parameterisation entirely.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs
arm () {
  local g=$1 t=$2 w=$3 y=$4
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --scale_cap $w \
    --results_dir $R/${t}_hier >$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
    --yaml_src diagnostics/moonshot/cfg/$y --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/$t --mesh_dir $MESH --all_stages >>$R/$t.log 2>&1
  echo "[$(date +%H:%M:%S)] $t done"
}
arm 0 SCL_1x 10.7 M7_scale1x.yaml &
arm 1 SCL_3x 32.0 M7_scale3x.yaml &
wait
echo "[$(date +%H:%M:%S)] E4 complete"
