#!/bin/bash
# THE A/B THAT TESTS THE PREPROCESSING HYPOTHESIS. DO NOT EDIT WHILE RUNNING.
#
# Control : M8a_split_only, already run on the RAW bench50.
# Treatment: identical config on bench50_clean -- same 50 specimens, same poses, same shape
#            space, same seed, same schedule. The ONLY variable is mesh quality.
#
# The cleanup demonstrably closed the gap to the reference corpus (medians over the 50):
#   vertex components 44.5 -> 4.0   (clean corpus 3.0)
#   boundary edges    0.0554 -> 0.0302 (0.0388)
#   non-manifold      0.0365 -> 0.0053 (0.0709)
#   triangle quality  0.8336 -> 0.8603 (0.8635)
#   median dihedral   10.70 -> 8.27 deg (8.20)
#   rough faces       0.0704 -> 0.0532 (0.0504)
#
# SCORED TWICE, deliberately:
#   ..._vs_clean : against the cleaned meshes -- the better estimate of the actual animal
#                  surface, since debris has been removed and cannot be rewarded.
#   ..._vs_raw   : against the original meshes, identical to how M8a_split_only was scored,
#                  so the control comparison is like-for-like.
# If cleaning helps, it should help on both. If it only helps against the cleaned target,
# the "gain" is just an easier scoring target, not a better registration.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
R=diagnostics/moonshot/runs
G=${1:-1}

CUDA_VISIBLE_DEVICES=$G python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir diagnostics/moonshot/bench50_clean --split_distal --midline 2.0 \
  --results_dir $R/P1_prepclean >$R/P1_prepclean.log 2>&1
CUDA_VISIBLE_DEVICES=$G python -u diagnostics/moonshot/eval_run.py \
  --run_dir $R/P1_prepclean --mesh_dir diagnostics/moonshot/bench50_clean --all_stages \
  --out $R/P1_prepclean/metrics_vs_clean.csv >>$R/P1_prepclean.log 2>&1
CUDA_VISIBLE_DEVICES=$G python -u diagnostics/moonshot/eval_run.py \
  --run_dir $R/P1_prepclean --mesh_dir diagnostics/moonshot/bench50 --all_stages \
  --out $R/P1_prepclean/metrics_vs_raw.csv >>$R/P1_prepclean.log 2>&1
# and score the RAW-fitted control against the CLEANED target, so both directions exist
CUDA_VISIBLE_DEVICES=$G python -u diagnostics/moonshot/eval_run.py \
  --run_dir $R/M8a_split_only --mesh_dir diagnostics/moonshot/bench50_clean --all_stages \
  --out $R/M8a_split_only/metrics_vs_clean.csv >>$R/P1_prepclean.log 2>&1
echo "[$(date +%H:%M:%S)] preprocessing A/B done"
