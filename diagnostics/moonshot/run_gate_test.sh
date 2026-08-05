#!/bin/bash
# HELD-OUT TEST OF THE FITTABILITY GATE. DO NOT EDIT WHILE RUNNING.
#
# The radial_med <-> fit-quality correlation (rho = +0.81 vs chamfer, +0.80 vs deform_mag)
# was DISCOVERED on the 50 bench specimens. Re-scoring those would prove nothing. These two
# sets are drawn from the other 707 workers and are disjoint from bench50:
#     heldout_easy  50 lowest radial_med   (median 0.1391 -- below the reference corpus 0.1521)
#     heldout_rand  50 uniformly sampled   (median 0.2439 -- the corpus median is 0.2367)
#
# Identical recipe, identical seed, run in parallel on the two GPUs. If the gate is real,
# 'easy' beats 'rand' on the TARGET-INDEPENDENT metrics (edge_logratio, deform_mag) as well
# as the surface ones -- section 3 rules out judging this on surface proximity alone.
#
# Pre-registered reading, fixed before the run:
#   * gate WORKS  if easy beats rand on deform_mag AND edge_logratio, both p<0.01 unpaired.
#   * gate is a SIZE ARTEFACT if it wins only on surface proximity -- radial_med is computed
#     in a frame normalised by max|coord|, so it could merely be tracking how far the
#     extremities protrude rather than anything about fittability.
#   * gate FAILS if neither, in which case per-specimen fittability is not predictable from
#     this geometry and the top-50% strategy needs a different score.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
R=diagnostics/moonshot/runs

arm () { # arm <gpu> <tag>
  local g=$1 t=$2
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical \
    --mesh_dir diagnostics/moonshot/heldout_$t --midline 2.0 \
    --results_dir $R/GATE_$t >$R/GATE_$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot \
    --mesh_dir diagnostics/moonshot/heldout_$t \
    --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
    --init_from $R/GATE_$t/H2_joint.npz \
    --results_dir $R/GATE_${t}_M7 >>$R/GATE_$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/GATE_${t}_M7 --mesh_dir diagnostics/moonshot/heldout_$t --all_stages >>$R/GATE_$t.log 2>&1
  echo "[$(date +%H:%M:%S)] GATE_$t done"
}

arm 0 easy &
arm 1 rand &
wait
echo "[$(date +%H:%M:%S)] gate test complete"
