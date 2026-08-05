#!/bin/bash
# THE DELIVERABLE: register the top 50% of the worker corpus. DO NOT EDIT WHILE RUNNING.
#
# 379 of 757 workers have radial_med <= 0.2367 (the corpus median), which is the gate
# validated on held-out specimens in report section 6.2 -- deform_mag +38.9% (p=4.5e-13) and
# edge_logratio +13.7% (p=5.6e-07) for gated versus uniformly-sampled specimens, i.e. it wins
# on the target-INDEPENDENT metrics and not merely on surface proximity.
#
# Chunked at 64 because a 50-mesh batch already peaks near 4.5 GB and SMAL3DFitter allocates
# batch_size = len(targets); 379 at once would not fit. Chunks alternate between the two GPUs.
#
# Recipe is the validated best arm (report section 2.1): hierarchical with the midline term,
# then the moonshot stage initialised from its H2_joint.
#
# Usage:  run_top50.sh [pkl]
#   pkl   model file; defaults to the ORIGINAL SMIL_OmniAnt.pkl. Pass the gated model only if
#         the shape-space A/B (S0_oldmodel vs S1_gated) showed it helps on the
#         target-independent metrics -- section 4 predicts it will not, since shape capacity
#         is not the binding constraint.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
R=diagnostics/moonshot/runs
PKL=${1:-3D_model_prep/SMIL_OmniAnt.pkl}
CHUNK=${CHUNK:-64}

python - "$CHUNK" <<'PY'
import pandas as pd, os, glob, sys
HERE = "diagnostics/moonshot"
W = "/media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker"
chunk = int(sys.argv[1])
r = pd.read_csv(f"{HERE}/out/worker_fittability_rank.csv")
top = r[r.radial_med <= r.radial_med.quantile(0.5)].sort_values("radial_med").reset_index(drop=True)
n = (len(top) + chunk - 1) // chunk
print(f"[top50] {len(top)} workers (radial_med <= {r.radial_med.quantile(0.5):.4f}) in {n} chunks of <= {chunk}")
for i in range(n):
    d = f"{HERE}/top50_c{i}"
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/*.obj"):
        os.unlink(f)
    for nm in top.name[i * chunk:(i + 1) * chunk]:
        os.symlink(os.path.join(W, nm), os.path.join(d, nm))
    print(f"[top50]   chunk {i}: {len(os.listdir(d))} meshes")
open(f"{HERE}/out/top50_nchunks.txt", "w").write(str(n))
PY

N=$(cat diagnostics/moonshot/out/top50_nchunks.txt)
echo "[top50] model: $PKL"

fit () { # fit <gpu> <chunk-index>
  local g=$1 i=$2
  local M=diagnostics/moonshot/top50_c$i
  SMILIFY_SMAL_FILE=$PKL CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical \
    --mesh_dir $M --midline 2.0 --results_dir $R/TOP50_c${i}_hier >$R/TOP50_c$i.log 2>&1
  SMILIFY_SMAL_FILE=$PKL CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot \
    --mesh_dir $M --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
    --init_from $R/TOP50_c${i}_hier/H2_joint.npz --results_dir $R/TOP50_c${i}_M7 >>$R/TOP50_c$i.log 2>&1
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/TOP50_c${i}_M7 --mesh_dir $M --all_stages >>$R/TOP50_c$i.log 2>&1
  echo "[$(date +%H:%M:%S)] TOP50 chunk $i done"
}

# even chunks on GPU0, odd on GPU1, each GPU sequential so memory stays bounded
( for ((i=0;i<N;i+=2)); do fit 0 $i; done ) &
( for ((i=1;i<N;i+=2)); do fit 1 $i; done ) &
wait
echo "[$(date +%H:%M:%S)] TOP50 complete: $N chunks"
