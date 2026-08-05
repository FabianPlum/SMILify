#!/bin/bash
# G5: does the frozen, target-derived part field beat the fit-derived partition downstream?
# DO NOT EDIT WHILE RUNNING.
#
# The control is M8a_split_only, already run, which is byte-identical to M9a except for
# --part_field:
#   M8a  --split_distal --midline 2.0
#   M9a  --split_distal --midline 2.0 --part_field <net>
# so any difference at the H2_joint / H3_deform checkpoints is attributable to the partition
# and nothing else. That controlled pair is the actual G5 test, and it runs first.
#
# The other two arms test capabilities the fit-derived partition cannot have even in
# principle, because it does not exist until a fit does:
#   M9b  + --pf_body   restrict the H0 body stage to the target's own body points; until now
#                      H0 has always been fitted against a global chamfer including all six
#                      legs, and has necessarily been dragged by them.
#   M9c  + --pf_init   correspondence-free part-anchor pose initialisation before H0. This
#                      is the one aimed at the trapped-pose finding of section 5: it matches
#                      per-part centroid and second moment, an objective with no
#                      nearest-neighbour choice in it and therefore no multimodality.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify

MESH=diagnostics/moonshot/bench50
R=diagnostics/moonshot/runs
NET=${NET:-diagnostics/moonshot/partfield/net_B_mirror.pt}
G=${1:-1}

hier () { # hier <name> <extra args...>
  local n=$1
  shift
  CUDA_VISIBLE_DEVICES=$G python -u -m fitter_3d.optimise_hierarchical \
    --mesh_dir $MESH --split_distal --midline 2.0 --part_field $NET "$@" \
    --results_dir $R/$n >$R/$n.log 2>&1
  CUDA_VISIBLE_DEVICES=$G python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/$n --mesh_dir $MESH --all_stages >>$R/$n.log 2>&1
  echo "[$(date +%H:%M:%S)] $n done"
}

case "${2:-all}" in
a) hier M9a_field ;;
b) hier M9b_fieldbody --pf_body ;;
c) hier M9c_anchorinit --pf_body --pf_init 400 ;;
*)
  hier M9a_field
  hier M9c_anchorinit --pf_body --pf_init 400
  hier M9b_fieldbody --pf_body
  ;;
esac
echo "[$(date +%H:%M:%S)] M9 batch complete"
