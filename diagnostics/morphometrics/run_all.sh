#!/bin/bash
# Regenerate every morphometric result and figure, in dependency order.
#
# The analysis modules auto-discover whatever MORPH_W* runs are on disk, so this is also the
# command to re-run as more fits land -- it always reports on the full corpus currently fitted.
# It does NOT refit: run run_m1_fit_all.sh for that (~2.5 h on 2x 4090).
#
# Order matters in one place only: calib_features.py writes out/feature_reliability.json, which
# is the ground-truth reliability the CORE_BLOCKS selection in measure.py is justified by. It
# runs on the SYNTHETIC corpus, so it does not change as worker chunks land, but it is regenerated
# first so the JSON on disk always matches the measurement code that produced the rest.
set -e
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
D=diagnostics/morphometrics
LOG=$D/out/run_all.log
mkdir -p $D/out
: >$LOG

step () {  # step <label> <command...>
  echo "=================================================================" | tee -a $LOG
  echo "[$(date +%H:%M:%S)] $1" | tee -a $LOG
  shift
  "$@" 2>&1 | tee -a $LOG
}

step "M0b  per-feature reliability vs ground truth (synthetic)" \
  python -u $D/calib_features.py

step "M3   signal, confounds, ICC, cross-corpus  ->  morphometrics.csv" \
  python -u $D/analyse.py

step "M5   genus index tables + cross-corpus agreement" \
  python -u $D/genus_table.py --min_n 5

step "M6   PCA / t-SNE / UMAP + HDBSCAN (best 50% by registration quality)" \
  python -u $D/embed.py --quality_top 50

step "M7   registration-quality filter sweep (deform/edge/normal) + random controls" \
  python -u $D/filter_quality.py --metric deform

step "fig  what the pipeline measures (best 10% of registrations)" \
  python -u $D/render_measurements.py --pick cephalotes odontomachus dorylus --quality_top 10

step "fig  specimens along each PC, posed and pose-reset (best 50%)" \
  python -u $D/render_pc_axis.py --quality_top 50

echo "=================================================================" | tee -a $LOG
echo "[$(date +%H:%M:%S)] complete. outputs:" | tee -a $LOG
ls -1 $D/out/*.png $D/out/*.csv $D/out/*.json 2>/dev/null | tee -a $LOG
