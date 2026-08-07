#!/bin/bash
# M1 — refit BOTH corpora under one consistent recipe, so morphometrics is never comparing
# specimens that were fitted differently. DO NOT EDIT WHILE RUNNING.
#
# WHY REFIT AT ALL. The fits already on disk are a patchwork: `CLEAN_M7` (81 ALL_ANTS_CLEAN)
# used the M7 recipe, `D1_low_MERGED` (128 gated workers) used D1. The headline morphometric
# question -- do the same genera give the same proportions in the two corpora? -- is only
# answerable if the recipe is held constant, otherwise a corpus difference and a recipe
# difference are perfectly confounded.
#
# RECIPE. D1, i.e. handoff `w_offset` 5.0/2.0. E8b measured this as the best available setting
# on the direct ground-truth round trip (6.69% correct vs the control's 4.81%, median vertex
# error 3.21% vs 3.48%), and REPORT_E8.md §4 adopts it as the default.
#
# NOTE ON `--offset`: it feeds the H3_deform stage, whose npz the handoff does not read
# (REPORT_E8.md §3.1), so it is INERT here. Passed anyway at the D1 value purely so the command
# line matches E8's provenance; it is not doing anything and must not be swept.
#
# SCOPE. ALL 757 workers, not the 379 that pass the fittability gate. Two reasons: taxonomic
# coverage of the shared genera is what limits the cross-corpus test, and keeping the ungated
# half lets the gate itself be evaluated as a morphometric quality filter post hoc rather than
# assumed. Gating is a column in the output table, not a precondition of the run.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
STAGE=diagnostics/morphometrics/stage
W=/media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker
CLEAN=/media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN
CHUNK=${CHUNK:-64}

python - "$W" "$CLEAN" "$STAGE" "$CHUNK" <<'PY'
import os, sys, glob, math
W, CLEAN, STAGE, chunk = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
work = sorted(f for f in os.listdir(W) if f.endswith(".obj"))
n = math.ceil(len(work) / chunk)
for i in range(n):
    d = f"{STAGE}/w{i}"
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/*.obj"):
        os.unlink(f)
    for nm in work[i * chunk:(i + 1) * chunk]:
        os.symlink(os.path.join(W, nm), os.path.join(d, nm))
print(f"[m1] {len(work)} workers in {n} chunks of <= {chunk}")
d = f"{STAGE}/clean"
os.makedirs(d, exist_ok=True)
for f in glob.glob(f"{d}/*.obj"):
    os.unlink(f)
cl = sorted(f for f in os.listdir(CLEAN) if f.endswith(".obj"))
for nm in cl:
    os.symlink(os.path.join(CLEAN, nm), os.path.join(d, nm))
print(f"[m1] {len(cl)} ALL_ANTS_CLEAN meshes in 1 chunk")
open(f"{STAGE}/nchunks.txt", "w").write(str(n))
PY

N=$(cat $STAGE/nchunks.txt)

fit () { # fit <gpu> <mesh_dir> <tag>
  local g=$1 M=$2 t=$3
  [ -f "$R/$t/Stage_3_deform_fine.npz" ] && { echo "[$t] already done, skipping"; return 0; }
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
    --results_dir $R/${t}_hier >$R/$t.log 2>&1 || { echo "[$t] HIER FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
    --yaml_src diagnostics/moonshot/cfg/D1_low.yaml --init_from $R/${t}_hier/H2_joint.npz \
    --results_dir $R/$t >>$R/$t.log 2>&1 || { echo "[$t] MOONSHOT FAILED"; return 1; }
  echo "[$(date +%H:%M:%S)] $t done"
}

# GPU0 takes even worker chunks plus the clean corpus, GPU1 the odd ones
( fit 0 $STAGE/clean MORPH_CLEAN
  for ((i=0;i<N;i+=2)); do fit 0 $STAGE/w$i MORPH_W$i; done ) &
( for ((i=1;i<N;i+=2)); do fit 1 $STAGE/w$i MORPH_W$i; done ) &
wait
echo "[$(date +%H:%M:%S)] M1 fits complete"
