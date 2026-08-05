#!/bin/bash
# SOFT PARTITION — the intervention §6.4's root-cause finding points at.
# DO NOT EDIT WHILE RUNNING.
#
# §6.4: the registrations carry no learnable shape structure. A leave-one-out shape model
# built from them reconstructs a held-out specimen no better than the population mean
# (gen/spread 0.99-1.00), while the same metric reaches 0.42 at 10 modes and 0.00 at 20 on
# synthetic data with exact correspondence. So correspondence is not consistent ACROSS
# specimens, which is invisible to every surface metric.
#
# The hard partition gives each target point entirely to its argmin group, so a point
# equidistant between two legs commits fully to one and the fit is pulled toward that
# commitment. Softening (E-step of EM-style robust shape-model fitting) lets it contribute to
# several groups in proportion to how well each explains it.
#
# Two temperatures, because the right length scale is not known a priori:
#   0.03  ~ tight; only genuinely ambiguous points spread mass
#   0.08  ~ loose; a broad neighbourhood of groups shares each point
# Control is the hard partition (soft_partition 0) on the same specimens, same seed.
#
# PRE-REGISTERED READING, fixed before the run:
#   * The claim is about CORRESPONDENCE, so the primary outcome is probe_19 generalisation
#     (gen@10 / population spread). A win means it drops meaningfully below 0.99.
#   * Surface metrics are secondary and may legitimately worsen -- §3.
#   * If gen/spread stays at 0.99 for both temperatures, softening the assignment is NOT
#     sufficient to establish correspondence, and the problem is upstream of the partition.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs

arm () { # arm <gpu> <tag> <temp>
  local g=$1 t=$2 tmp=$3
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical \
    --mesh_dir $MESH --midline 2.0 --soft_partition $tmp \
    --results_dir $R/$t >$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
    --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
    --init_from $R/$t/H2_joint.npz --results_dir $R/${t}_M7 >>$R/$t.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/${t}_M7 --mesh_dir $MESH --all_stages >>$R/$t.log 2>&1
  echo "[$(date +%H:%M:%S)] $t done"
}

arm 0 SOFT_t03 0.03 &
arm 1 SOFT_t08 0.08 &
wait
echo "[$(date +%H:%M:%S)] soft-partition arms complete"
