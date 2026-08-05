#!/bin/bash
# Everything queued for the next session, in dependency order.
#
# BLOCKED ON: /media/fabi/Data being mounted. The bench50 symlinks and the 757-worker corpus
# both live there; sda1 (label "Data", 10.9T) is present but unmounted, so every fitting run
# below will fail immediately with FileNotFoundError until it is back.
# Check with:  ls /media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker | wc -l
#
# usage: run_next_batch.sh [gpu0] [gpu1]
# NOTE: no `set -u` -- conda's activate.d scripts reference unbound vars and would abort.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify

MESH=diagnostics/moonshot/bench50
WORKER=/media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker
G0=${1:-0}; G1=${2:-1}

if [ ! -e "$MESH/$(ls $MESH | head -1)" ]; then
  echo "ABORT: $MESH symlinks are dangling -- /media/fabi/Data is not mounted."
  exit 1
fi

# ---------------------------------------------------------------- A. new arms (bench50)
# M8: M7 + the two probe-13 fixes. Tests whether per-segment weighting and a per-part robust
# scale close M7's one remaining regression (part_leg_distal +15.5% vs stock).
# Pre-registered success criterion: part_leg_distal regression vs baseline shrinks below the
# 5% per-part gate, WITHOUT losing M7's wins (deform_mag and edge_logratio must stay >=25%
# better than baseline, fscore@0.02 must not drop below baseline).
run_m8 () {
  export CUDA_VISIBLE_DEVICES=$G0
  python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH --split_distal \
    --part_robust 4.0 --midline 2.0 \
    --results_dir diagnostics/moonshot/runs/M8_sd_pr > diagnostics/moonshot/runs/M8_sd_pr.log 2>&1
  python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
    --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
    --init_from diagnostics/moonshot/runs/M8_sd_pr/H2_joint.npz \
    --results_dir diagnostics/moonshot/runs/M8_handoff >> diagnostics/moonshot/runs/M8_sd_pr.log 2>&1
  python -u diagnostics/moonshot/eval_run.py --run_dir diagnostics/moonshot/runs/M8_handoff \
    --mesh_dir $MESH --all_stages >> diagnostics/moonshot/runs/M8_sd_pr.log 2>&1
}

# ablations, so the two fixes are separable rather than shipped as a bundle
run_m8_ablations () {
  export CUDA_VISIBLE_DEVICES=$G0
  for A in "--split_distal:M8a_split_only" "--part_robust 4.0:M8b_robust_only"; do
    FLAG="${A%%:*}"; NAME="${A##*:}"
    python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH $FLAG --midline 2.0 \
      --results_dir diagnostics/moonshot/runs/$NAME > diagnostics/moonshot/runs/$NAME.log 2>&1
    python -u diagnostics/moonshot/eval_run.py --run_dir diagnostics/moonshot/runs/$NAME \
      --mesh_dir $MESH --all_stages >> diagnostics/moonshot/runs/$NAME.log 2>&1
  done
}

# ---------------------------------------------------------------- B. seed-repeat M7 (item 5)
# M7's large effects (deform_mag -52%, edge -32%, midline -47%, all 47-50/50) are far outside
# plausible seed noise, but its smaller ones (fscore@0.01 +7.1% at 35/50) are not. Same
# protocol as A4: 3 seeds, sign must be consistent, mean must exceed between-seed sd.
run_m7_seeds () {
  export CUDA_VISIBLE_DEVICES=$G1
  for SEED in 1 2; do
    python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH --midline 2.0 --seed $SEED \
      --results_dir diagnostics/moonshot/runs/M1_sym_s$SEED > diagnostics/moonshot/runs/M1_sym_s$SEED.log 2>&1
    R=diagnostics/moonshot/runs/M7_handoff_midline_s$SEED
    python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
      --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml --seed $SEED \
      --init_from diagnostics/moonshot/runs/M1_sym_s$SEED/H2_joint.npz \
      --results_dir $R > ${R}.log 2>&1
    python -u diagnostics/moonshot/eval_run.py --run_dir $R --mesh_dir $MESH --all_stages >> ${R}.log 2>&1
  done
}

# ---------------------------------------------------------------- C. co-registration (item 4)
# Round 1 used 25 train specimens and gave real generalizing gains in part placement
# (held-out distal legs -26%, p=0.015) but did not close the surface gap, and produced
# FOLDED geometry (2.79% of adjacent faces >90 deg vs baseline 0.54%).
# Scaling to all 757 workers is the stated next step -- BUT the folding must be fixed first,
# or more directions just means more folding. See run_coregistration.py for the constraint.
run_corereg () {
  export CUDA_VISIBLE_DEVICES=$G1
  python -u diagnostics/moonshot/run_coregistration.py --worker_dir $WORKER --rounds 3 \
    > diagnostics/moonshot/runs/coreg.log 2>&1
}

case "${3:-all}" in
  m8)      run_m8 ;;
  ablate)  run_m8_ablations ;;
  seeds)   run_m7_seeds ;;
  coreg)   run_corereg ;;
  *)       run_m8 & run_m7_seeds & wait; run_m8_ablations ;;
esac
echo "[$(date +%H:%M:%S)] batch complete"
