#!/bin/bash
# Seed repeats for the marginal claims.
#
# Every headline number in REPORT.md is single-seed. The large effects are safe --
# deform_mag -51% at 50/50 wins cannot be a seed artifact -- but the small ones are exactly
# where single-seed results go wrong: Khaoula's keypoint arm died on this, with run-to-run
# std ~12% against a 15% bar and one single-seed "improvement" inverting to a significant
# degradation under 3-seed replication.
#
# The marginal claim is A4_nofreeze's fscore@0.01 +0.7% (35/50, p=0.0066). This runs both
# C0_control and A4_nofreeze at two further seeds so the effect can be checked across three.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export CUDA_VISIBLE_DEVICES=${1:-1}
MESH=diagnostics/moonshot/bench50

for SEED in 1 2; do
  for A in C0_control A4_nofreeze; do
    R=diagnostics/moonshot/runs/${A}_s${SEED}
    echo "[$(date +%H:%M:%S)] $A seed $SEED"
    python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
      --yaml_src diagnostics/moonshot/cfg/$A.yaml --results_dir $R --seed $SEED \
      > ${R}.log 2>&1
    python -u diagnostics/moonshot/eval_run.py --run_dir $R --mesh_dir $MESH \
      --all_stages >> ${R}.log 2>&1
  done
done
echo "[$(date +%H:%M:%S)] SEEDS COMPLETE"
