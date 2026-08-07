#!/bin/bash
# E8 — DOES TURNING OFF FREE-FORM DEFORMATION BUY CORRESPONDENCE? DO NOT EDIT WHILE RUNNING.
#
# WHY. Three measured facts converge on this experiment:
#   §6.6  free-form `deform_verts` carries 100.0% of the rest-space shape variance across
#         specimens, and that component generalises at 0.98 -- i.e. it is per-specimen noise.
#   §6.9.3 constraining pose pushes compensation into `log_beta_scales` and `betas_trans`,
#         the two parameters that decide WHERE A JOINT SITS. The fit is placing the skeleton
#         wrong and hiding it.
#   E7 §5  83.3% of correspondence error never crosses a part boundary, so no data-term or
#         partition change can reach it. The remaining lever is what the model may represent.
#
# A fit with zero free-form offsets has correspondence-consistency BY CONSTRUCTION: rest-space
# geometry is then a pure function of `betas`, so template vertex v means the same thing in
# every specimen. The question is what that costs in surface accuracy, and whether the
# resulting registrations finally carry learnable shape structure.
#
# ARMS. Identical in every respect -- same specimens, same schedule, same iteration budget,
# same model, same seed -- except the L2 penalty on free-form offsets. Pinning deform with a
# large penalty rather than deleting the stage keeps the compute matched, so the comparison
# has one variable.
#
#   D0_control   --offset   3.0   handoff w_offset 0.2 / 0.08     (the current best recipe)
#   D1_low       --offset  30.0   handoff w_offset 5.0 / 2.0      (10x / 25x)
#   D2_frozen    --offset 300.0   handoff w_offset 200 / 200      (deform pinned at ~0)
#
# SPECIMENS. The gated top 50% by `radial_med` (§6.2), which is the validated operating point,
# taking the 128 best-ranked as two chunks of 64. n=128 is well above the n=50 probe-19 has
# been run at, and a paired 3-way comparison on identical specimens is what makes the small
# expected surface differences readable.
#
# PRE-REGISTERED READING, fixed before the run:
#   PRIMARY   probe-19 gen@10/spread. E6 calibrated the scale: ~5% correspondence correctness
#             reads 0.89, every worker arm so far reads 0.92-0.96, ALL_ANTS_CLEAN reads 0.5383,
#             exact correspondence reads 0.42. D2 must move DECISIVELY toward 0.5383 -- it is
#             structurally guaranteed to have consistent correspondence, so if it does NOT move
#             the metric, probe-19 is not measuring what it is believed to measure and that
#             finding supersedes the experiment.
#   SECONDARY surface cost: fscore@0.01/0.02 and chamfer_l2 against D0. §2.3 measured a
#             deform-free arm at fscore@0.02 = 0.7313 versus the baseline's ~0.92, so a large
#             surface regression is EXPECTED and is not by itself a failure -- the question is
#             whether the correspondence gained is worth it.
#   TERTIARY  betas sd. With free-form suppressed, shape must be carried by the shape space or
#             by `log_beta_scales`. If it moves into `log_beta_scales` again (§6.9.3) rather
#             than into `betas`, the registrations still will not transfer and that is the
#             thing to fix next.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
R=diagnostics/moonshot/runs
N_SPEC=${N_SPEC:-128}
CHUNK=${CHUNK:-64}

python - "$N_SPEC" "$CHUNK" <<'PY'
import os, glob, sys, csv
HERE = "diagnostics/moonshot"
W = "/media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker"
n_spec, chunk = int(sys.argv[1]), int(sys.argv[2])
rows = [r for r in csv.DictReader(open(f"{HERE}/out/fittability_gate.csv"))
        if r["keep_top50"].lower() == "true"]
rows.sort(key=lambda r: float(r["radial_med"]))
sel = rows[:n_spec]
n = (len(sel) + chunk - 1) // chunk
print(f"[e8] {len(sel)} gated workers (radial_med {float(sel[0]['radial_med']):.4f}"
      f"..{float(sel[-1]['radial_med']):.4f}) in {n} chunks")
for i in range(n):
    d = f"{HERE}/e8_c{i}"
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/*.obj"):
        os.unlink(f)
    for r in sel[i * chunk:(i + 1) * chunk]:
        src = os.path.join(W, r["name"])
        if os.path.exists(src):
            os.symlink(src, os.path.join(d, r["name"]))
    print(f"[e8]   chunk {i}: {len(os.listdir(d))} meshes")
open(f"{HERE}/out/e8_nchunks.txt", "w").write(str(n))
PY

N=$(cat diagnostics/moonshot/out/e8_nchunks.txt)

fit () { # fit <gpu> <chunk> <arm> <hier_offset>
  local g=$1 i=$2 arm=$3 off=$4
  local M=diagnostics/moonshot/e8_c$i
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
    --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset $off \
    --results_dir $R/${arm}_c${i}_hier >$R/${arm}_c$i.log 2>&1 || { echo "[$arm c$i] HIER FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$g python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
    --yaml_src diagnostics/moonshot/cfg/${arm}.yaml --init_from $R/${arm}_c${i}_hier/H2_joint.npz \
    --results_dir $R/${arm}_c${i} >>$R/${arm}_c$i.log 2>&1 || { echo "[$arm c$i] MOONSHOT FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$g python -u diagnostics/moonshot/eval_run.py \
    --run_dir $R/${arm}_c${i} --mesh_dir $M --all_stages >>$R/${arm}_c$i.log 2>&1
  echo "[$(date +%H:%M:%S)] $arm chunk $i done"
}

# GPU0 takes even chunks, GPU1 odd; each GPU runs its arms sequentially so memory stays bounded
( for arm_off in "D0_control 3.0" "D1_low 30.0" "D2_frozen 300.0"; do
    set -- $arm_off
    for ((i=0;i<N;i+=2)); do fit 0 $i "$1" "$2"; done
  done ) &
( for arm_off in "D0_control 3.0" "D1_low 30.0" "D2_frozen 300.0"; do
    set -- $arm_off
    for ((i=1;i<N;i+=2)); do fit 1 $i "$1" "$2"; done
  done ) &
wait
echo "[$(date +%H:%M:%S)] E8 fits complete"
