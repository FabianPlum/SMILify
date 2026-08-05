#!/bin/bash
# THE MEASURED FIX. DO NOT EDIT WHILE RUNNING.
#
# Measured equilibrium: with w_beta_prior=0.002 the prior gradient overtakes the chamfer
# gradient at |betas| ~ 0.02 and is 4.87x larger at 0.1, 12.94x at 0.3. That pins the shape
# space shut. The stock baseline carries NO beta prior and reaches |betas| 0.26; A3b already
# established 1.5e-4 as the tuned value, and the hierarchical pipeline never received it --
# so M1/M5/M7/M8/M9, the gate, the shape space and every arm since inherited a closed shape
# space. This is the same defect recorded in §7.2, fixed once for the moonshot YAML arms and
# never propagated.
#
#   BP_a  w_beta_prior 1.5e-4  (the A3b-tuned value)
#   BP_b  w_beta_prior 0       (no prior, matching the stock baseline)
#   control: M7_handoff_midline (0.002), already scored
#
# PRE-REGISTERED:
#   1. betas sd must rise from 0.00042 toward the non-hierarchical arms' 0.068-0.50. This one
#      is close to mechanically guaranteed by the gradient measurement, so it is a check that
#      the fix is wired, not evidence that it helps.
#   2. THE REAL TEST is probe-19 gen/spread. It must move off 0.99 toward the synthetic
#      control's 0.42. Only that says the registrations became transferable.
#   3. Surface metrics may worsen (§3). A surface win with gen/spread still at 0.99 would mean
#      the shape space is merely a new way to shrink-wrap.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs
arm () {
  local g=$1 t=$2 w=$3
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH \
    --midline 2.0 --beta_prior $w --results_dir $R/$t >$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
    --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
    --init_from $R/$t/H2_joint.npz --results_dir $R/${t}_M7 >>$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/${t}_M7 --mesh_dir $MESH --all_stages >>$R/$t.log 2>&1
  echo "[$(date +%H:%M:%S)] $t done"
}
arm 0 BP_a 0.00015 &
arm 1 BP_b 0.0 &
wait
echo "[$(date +%H:%M:%S)] beta-prior arms complete"
