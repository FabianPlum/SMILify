#!/bin/bash
# Sequential experiment runner for one GPU. Runs each arm then immediately scores it,
# so evaluation never competes with a fit for GPU memory (an earlier eval died that way).
#
# usage: queue_runner.sh <gpu> <arm> [arm ...]
#   arm names starting with 'M1' are dispatched to the hierarchical runner,
#   everything else to the moonshot runner with cfg/<arm>.yaml
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify

GPU=$1; shift
MESH=diagnostics/moonshot/bench50
export CUDA_VISIBLE_DEVICES=$GPU

for A in "$@"; do
  echo "[$(date +%H:%M:%S)] === $A on gpu $GPU ==="
  case "$A" in
    M1_hier)
      python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH \
        --results_dir diagnostics/moonshot/runs/$A > diagnostics/moonshot/runs/$A.log 2>&1 ;;
    M3_multistart)
      python -u -m fitter_3d.optimise_multistart --mesh_dir $MESH \
        --results_dir diagnostics/moonshot/runs/$A > diagnostics/moonshot/runs/$A.log 2>&1 ;;
    M1_nopart)
      python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH --no_partition \
        --results_dir diagnostics/moonshot/runs/$A > diagnostics/moonshot/runs/$A.log 2>&1 ;;
    M5_handoff)
      python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
        --yaml_src diagnostics/moonshot/cfg/M5_handoff.yaml \
        --init_from diagnostics/moonshot/runs/M1_hier/H2_joint.npz \
        > diagnostics/moonshot/runs/$A.log 2>&1 ;;
    *)
      python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
        --yaml_src diagnostics/moonshot/cfg/$A.yaml > diagnostics/moonshot/runs/$A.log 2>&1 ;;
  esac
  echo "[$(date +%H:%M:%S)] fit done: $A -> scoring"
  python -u diagnostics/moonshot/eval_run.py --run_dir diagnostics/moonshot/runs/$A \
     --mesh_dir $MESH --all_stages >> diagnostics/moonshot/runs/$A.log 2>&1
  echo "[$(date +%H:%M:%S)] scored: $A"
done
echo "[$(date +%H:%M:%S)] QUEUE COMPLETE on gpu $GPU"
