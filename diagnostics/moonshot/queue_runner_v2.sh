#!/bin/bash
# Sequential experiment runner for one GPU: fit, then immediately score.
#
# NOTE: bash reads a script incrementally while executing it, so editing this file while a
# queue is running corrupts the running shell (this happened twice). To add an arm type,
# copy this to a new versioned filename rather than editing it in place.
#
# usage: queue_runner_v2.sh <gpu> <arm> [arm ...]
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
    M1_nopart)
      python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH --no_partition \
        --results_dir diagnostics/moonshot/runs/$A > $L 2>&1 ;;
    M3_multistart)
      python -u -m fitter_3d.optimise_multistart --mesh_dir $MESH \
        --results_dir diagnostics/moonshot/runs/$A > $L 2>&1 ;;
    M5_handoff)
      python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
        --yaml_src diagnostics/moonshot/cfg/M5_handoff.yaml \
        --init_from diagnostics/moonshot/runs/M1_hier/H2_joint.npz > $L 2>&1 ;;
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
