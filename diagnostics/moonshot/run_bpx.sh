#!/bin/bash
# THE COMPLETE FIX: beta prior corrected in BOTH files. DO NOT EDIT WHILE RUNNING.
#
# --beta_prior fixed the hierarchical stages and they responded exactly as the gradient
# measurement predicted: betas sd at H3 went 0.00189 (w=0.002) -> 0.01766 (1.5e-4) -> 0.12399
# (w=0), i.e. 65x, approaching the stock baseline's 0.256.
#
# Then the moonshot handoff crushed all three back to 0.00043, because
# cfg/M7_handoff_midline.yaml carries `w_beta_prior: 0.002` in BOTH of its 1000-iteration
# stages. The stock ants_cfg.yaml has NO beta prior at all; 0.002 is a value I introduced,
# and it is the UNTUNED one -- A3b had already established 1.5e-4 as the tuned value.
#
#   BPX_noprior  hierarchical w=0      + moonshot w=0
#   BPX_tuned    hierarchical w=1.5e-4 + moonshot w=1.5e-4
#   control: M7_handoff_midline (0.002 / 0.002), already scored
#
# PRE-REGISTERED. Betas surviving the handoff is now the wiring check. The outcome that
# decides whether ANY of this mattered is probe-19 gen/spread moving off 0.99 toward the
# synthetic control's 0.42 -- i.e. whether the registrations became transferable, which is
# what every downstream product needs and what four previous fixes failed to deliver.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs
arm () {
  local g=$1 t=$2 w=$3 y=$4
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH \
    --midline 2.0 --beta_prior $w --results_dir $R/${t}_hier >$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
    --yaml_src diagnostics/moonshot/cfg/$y --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/$t --mesh_dir $MESH --all_stages >>$R/$t.log 2>&1
  echo "[$(date +%H:%M:%S)] $t done"
}
arm 0 BPX_noprior 0.0     M7_noprior.yaml &
arm 1 BPX_tuned   0.00015 M7_tuned.yaml &
wait
echo "[$(date +%H:%M:%S)] BPX arms complete"
