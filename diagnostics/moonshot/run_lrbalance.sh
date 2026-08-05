#!/bin/bash
# Does the pose:shape learning-rate ratio explain the frozen shape space? DO NOT EDIT WHILE RUNNING.
#
# Every arm through optimise_hierarchical has betas sd 0.00038-0.00050; every arm that is not
# has 0.068-0.50. A 100-1000x split, perfectly separated by pipeline, no overlap.
# The stock baseline slows pose 10x relative to shape (ants_cfg.yaml Stage_1_default:
# lr 0.02, custom_lrs joint_rot 0.002). The hierarchical stages give joint_rot an equal or
# higher rate than betas (H0 0.01 vs 0.02, H1 0.015 vs 0.008), and pose -- 162 dof plus 165
# free per-joint scales -- out-competes 13 betas.
#
# LR_a  joint_lr x0.1  reproduces the baseline's pose:shape ratio
# LR_b  joint_lr x0.3  intermediate
# control: M1_sym / M7_handoff_midline (x1.0), already scored.
#
# PRE-REGISTERED: betas sd must rise toward the non-hierarchical arms' 0.06-0.50. If it does
# not, the lr ratio is not the mechanism either and the shape freeze is elsewhere. probe-19
# gen/spread is the outcome that decides whether it MATTERS.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs
arm () {
  local g=$1 t=$2 m=$3
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH \
    --midline 2.0 --joint_lr_mult $m --results_dir $R/$t >$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
    --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
    --init_from $R/$t/H2_joint.npz --results_dir $R/${t}_M7 >>$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/${t}_M7 --mesh_dir $MESH --all_stages >>$R/$t.log 2>&1
  echo "[$(date +%H:%M:%S)] $t done"
}
arm 0 LR_a 0.1 &
arm 1 LR_b 0.3 &
wait
echo "[$(date +%H:%M:%S)] lr-balance arms complete"
