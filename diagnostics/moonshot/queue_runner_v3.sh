#!/bin/bash
# Sequential experiment runner for one GPU: fit, then immediately score.
# Versioned copy (see v2 header): bash reads a script incrementally while running it, so
# adding an arm type means writing a NEW file, never editing one a queue is executing.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify

GPU=$1; shift
MESH=diagnostics/moonshot/bench50
export CUDA_VISIBLE_DEVICES=$GPU

for A in "$@"; do
  echo "[$(date +%H:%M:%S)] === $A on gpu $GPU ==="
  L=diagnostics/moonshot/runs/$A.log
  case "$A" in
    M1_hier)
      python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH \
        --results_dir diagnostics/moonshot/runs/$A > $L 2>&1 ;;
    M1_sym)
      # hierarchical, but with the midline constraint active in every stage
      python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH --midline 2.0 \
        --results_dir diagnostics/moonshot/runs/$A > $L 2>&1 ;;
    M3_multistart)
      python -u -m fitter_3d.optimise_multistart --mesh_dir $MESH \
        --results_dir diagnostics/moonshot/runs/$A > $L 2>&1 ;;
    M7_handoff_midline)
      python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
        --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
        --init_from diagnostics/moonshot/runs/M1_sym/H2_joint.npz > $L 2>&1 ;;
    *)
      python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
        --yaml_src diagnostics/moonshot/cfg/$A.yaml > $L 2>&1 ;;
  esac
  echo "[$(date +%H:%M:%S)] fit done: $A -> scoring"
  python -u diagnostics/moonshot/eval_run.py --run_dir diagnostics/moonshot/runs/$A \
     --mesh_dir $MESH --all_stages >> $L 2>&1
  echo "[$(date +%H:%M:%S)] scored: $A"
done
echo "[$(date +%H:%M:%S)] QUEUE COMPLETE on gpu $GPU"
