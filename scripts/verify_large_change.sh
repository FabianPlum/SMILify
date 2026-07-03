#!/usr/bin/env bash
# ==============================================================================
# Post-large-change verification for SMILify.
#
# Run this after ANY substantial change (new feature, refactor, shared-code edit)
# BEFORE committing / pushing / opening a PR. It runs, in order:
#
#   1. ruff  : lint (`ruff check .`) + format check (`ruff format --check .`)
#   2. pytest: the FULL suite, INCLUDING slow tests (`pytest`)
#   3. GETTING_STARTED short runs on the example stick assets:
#        - multi-view benchmark   (--max_batches 2)
#        - multi-view inference    (--max_frames 5)
#        - multi-view training     (capped at 60s)
#        - single-view training    (capped at 60s, camera-centric)
#      Skipped with a notice if the example assets are not present.
#
# ---- Conventions that MUST be followed (learned the hard way) ----------------
#   * Activate the conda env directly (source conda.sh + `conda activate`).
#     Do NOT wrap long GPU runs in `conda run` — it buffers/captures stdout, so a
#     timeout-killed run leaves an EMPTY log and you learn nothing.
#   * Use `python -u` (unbuffered) and redirect to a file, so output is written
#     live and survives a hard kill.
#   * Cap every training run at 60s (`timeout 60`). A minute is enough to confirm
#     it starts and the loss moves; running longer wastes time. Exit code 124
#     (timeout) counts as PASS for training.
#   * Training writes to a THROWAWAY temp dir (never the real output dirs), so a
#     verification run can never clobber a real checkpoint.
#
# Usage:
#   bash scripts/verify_large_change.sh                 # from the repo root
#   ENV_NAME=pytorch3d bash scripts/verify_large_change.sh
# ==============================================================================
set -o pipefail

cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
ENV_NAME="${ENV_NAME:-pytorch3d}"
TRAIN_CAP="${TRAIN_CAP:-60}"   # seconds to let each training run before killing

# --- Activate the conda env in THIS shell (NOT `conda run`) -------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/mambaforge")"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}" || { echo "ERROR: could not activate conda env '${ENV_NAME}'"; exit 1; }
fi
echo "python: $(command -v python)"
export PYTHONUNBUFFERED=1

LOGDIR="$(mktemp -d)"
fail=0
trap 'rm -rf "${LOGDIR}"' EXIT

step() { echo; echo "=============================================================="; echo "== $*"; echo "=============================================================="; }

# run_capped <label> <seconds> <cmd...> : run unbuffered, log live, hard-cap the
# runtime. rc 0 (finished) or 124 (hit the cap) => PASS; anything else => FAIL.
run_capped() {
    local label="$1"; local secs="$2"; shift 2
    local log="${LOGDIR}/${label}.log"
    echo "-- ${label} (<=${secs}s) --"
    timeout "${secs}" "$@" >"${log}" 2>&1
    local rc=$?
    if [ "${rc}" -eq 0 ] || [ "${rc}" -eq 124 ]; then
        tail -n 2 "${log}"
        echo "   [${label}] OK (rc=${rc})"
    else
        echo "   [${label}] FAILED (rc=${rc}); last lines:"
        tail -n 25 "${log}"
        fail=1
    fi
}

# Copy an example config to a temp file with its output dirs redirected to $1/out
# so a verification training run never touches real checkpoint/plot/viz dirs.
temp_config() {
    local src="$1"; local dst="$2"
    python - "$src" "$dst" "${LOGDIR}/out" <<'PY'
import json, sys
src, dst, outdir = sys.argv[1:4]
c = json.load(open(src))
c.setdefault("output", {})
c["output"]["checkpoint_dir"] = outdir + "/ckpt"
c["output"]["plots_dir"] = outdir + "/plots"
c["output"]["visualizations_dir"] = outdir + "/viz"
c["output"]["train_visualizations_dir"] = outdir + "/viz_train"
json.dump(c, open(dst, "w"), indent=2)
PY
}

# ---- 1. ruff -----------------------------------------------------------------
step "1/3  ruff (lint + format check)"
ruff check . || fail=1
ruff format --check . || fail=1

# ---- 2. pytest (incl. slow) --------------------------------------------------
step "2/3  pytest (full suite, incl. slow)"
pytest -q || fail=1

# ---- 3. GETTING_STARTED short runs ------------------------------------------
step "3/3  GETTING_STARTED short runs"
DATA="SMILySTICKS_centred_reprojected_FIXED.h5"
CKPT="SMILySTICKS_ViT_model.pth"
SMAL="3D_model_prep/SMILy_STICK.pkl"
MV_CFG="smal_fitter/neuralSMIL/configs/examples/getting_started.json"
SV_CFG="smal_fitter/neuralSMIL/configs/examples/getting_started_singleview.json"

if [ -f "${DATA}" ] && [ -f "${CKPT}" ] && [ -f "${SMAL}" ]; then
    BENCH_DIR="benchmark_multiview_$(basename "${CKPT}" .pth)_on_$(basename "${DATA}" .h5)"

    run_capped "benchmark_mv" 120 python -u -m smal_fitter.neuralSMIL.benchmark_model \
        --checkpoint "${CKPT}" --dataset_path "${DATA}" --smal-file "${SMAL}" \
        --orig_width 1530 --orig_height 1530 --max_batches 2

    run_capped "inference_mv" 120 python -u -m smal_fitter.neuralSMIL.run_multiview_inference \
        --dataset "${DATA}" --checkpoint "${CKPT}" --smal_file "${SMAL}" --max_frames 5

    MV_TMP="${LOGDIR}/mv_cfg.json"; temp_config "${MV_CFG}" "${MV_TMP}"
    run_capped "train_mv" "${TRAIN_CAP}" python -u -m smal_fitter.neuralSMIL.train_multiview_regressor \
        --config "${MV_TMP}" --dataset_path "${DATA}"

    SV_TMP="${LOGDIR}/sv_cfg.json"; temp_config "${SV_CFG}" "${SV_TMP}"
    run_capped "train_sv" "${TRAIN_CAP}" python -u -m smal_fitter.neuralSMIL.train_smil_regressor \
        --config "${SV_TMP}" --data_path "${DATA}"

    # Remove the regenerable benchmark/inference artifacts this script created.
    rm -rf "${BENCH_DIR}"
    rm -f "$(basename "${DATA}" .h5)"_*inference*.avi "$(basename "${DATA}" .h5)"_*inference*.mp4
else
    echo "  SKIP: GETTING_STARTED assets not found in the repo root"
    echo "        (need ${DATA} and ${CKPT}; see GETTING_STARTED.md section 2)."
    echo "        Ran ruff + pytest only."
fi

echo
if [ "${fail}" -eq 0 ]; then
    echo "VERIFY: PASS"
else
    echo "VERIFY: FAIL"
fi
exit "${fail}"
