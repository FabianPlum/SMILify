#!/bin/bash
# E7 -- DOES A CONVEXITY-DERIVED PART HIERARCHY INFORM HIERARCHICAL MATCHING?
# DO NOT EDIT WHILE RUNNING.
#
# E6 (SYNTHETIC_ROUNDTRIP.md) established, directly and for the first time, that the pipeline
# does not establish vertex-level correspondence even on its own noise-free geometry:
# 4.81% strict correctness, ~3.8 vertex-spacings of error, barely changed by noise. Five
# pre-registered interventions on HOW THE DATA TERM IS SHAPED have returned null (soft
# partition, frozen part field, rotation limits, anterior split, per-part robust scaling).
#
# This arm changes WHAT THE DATA TERM IS SHAPED BY. `fitter_3d/hull_decomposition.py` breaks
# the target into near-convex chunks by approximate convex decomposition and builds a merge
# tree whose height is the concavity a merge manufactures -- "where the hulls separate".
# `fitter_3d/hull_partition.py` then uses those chunks as VOTING UNITS: each chunk goes to
# one anatomical group as a whole, by majority vote of its own points, recomputed at every
# reassignment.
#
# WHY THIS IS NOT THE PART FIELD AGAIN. The part field froze the NAMING (which group each
# point belongs to) and lost decisively, because a frozen naming cannot recover from its own
# errors. Here only the CHUNKING is frozen -- a fact about the scan's geometry, not a belief
# about anatomy -- and the naming stays fit-derived and fully revisable.
#
# ARMS (all on synth_clean, where ground-truth correspondence is exact and known):
#   HULL_k13    k=13 fixed, criterion=volume
#   HULL_sched  granularity schedule 5,13,20,30 across H0/H1/H2/H3 -- uses the hierarchy as a
#               hierarchy rather than as one segmentation
#   HULL_amb    k=13 with min_majority 0.6: chunks with no clear winner abstain and their
#               points keep per-point labels, so an ambiguous chunk reduces influence rather
#               than committing (the lesson from the soft-partition/EM formulation)
#
# CONTROL: runs/SYN_clean, already on disk from E6. Byte-identical recipe apart from the
# partition, so the single variable really is the partition.
#
# PRE-REGISTERED READING, fixed before the run:
#   PRIMARY   `correct_frac` on SYN_clean from score_synth_roundtrip.py, against E6's 4.81%.
#   KILL      if no arm exceeds 4.81% by more than the between-seed spread, the hierarchy does
#             not inform matching and this is DROPPED rather than re-weighted -- the same
#             discipline that dropped E1 (joint limits) and E3 (anterior split).
#   SECONDARY median per-vertex error to the correct GT vertex (E6: 3.48% of extent, 3.8x the
#             local vertex spacing), and the per-part breakdown -- E6's worst part is the
#             gaster at 10.41%, a large smooth ellipsoid where chamfer has no positional
#             information. A convexity prior should help the gaster LEAST and the limbs MOST;
#             if it helps the gaster most, the mechanism is not what is claimed.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
D=diagnostics/moonshot/synth_clean

arm () {
  local g=$1 t=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $D \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 "$@" \
    --results_dir $R/${t}_hier >$R/$t.log 2>&1 || { echo "[$t] HIER FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $D \
    --yaml_src diagnostics/moonshot/cfg/M7_SYN_clean.yaml --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1 || { echo "[$t] MOONSHOT FAILED"; return 1; }
  echo "[$(date +%H:%M:%S)] $t done"
}

arm 0 HULL_k13   --hull_partition 13 --hull_criterion volume &
arm 1 HULL_sched --hull_partition 13 --hull_criterion volume --hull_k_schedule 5,13,20,30 &
wait
arm 0 HULL_amb   --hull_partition 13 --hull_criterion volume --hull_min_majority 0.6 &
wait
echo "[$(date +%H:%M:%S)] E7 fits complete"

python -u diagnostics/moonshot/score_synth_roundtrip.py \
  --pairs SYN_clean:synth_clean HULL_k13:synth_clean HULL_sched:synth_clean HULL_amb:synth_clean \
  --render_n 2 2>&1 | tee diagnostics/moonshot/hull/out/e7_roundtrip.txt
