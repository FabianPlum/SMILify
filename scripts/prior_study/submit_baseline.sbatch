#!/usr/bin/env bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=prior_study_results/slurm_%j.log
#SBATCH --error=prior_study_results/slurm_%j.log
#
# Background baseline study on a SLURM cluster (developed on JURECA-DC).
# Edit --account/--partition above for your site, then submit FROM THE REPO
# ROOT (the #SBATCH --output path is resolved at submit time, so the results
# dir must exist before sbatch):
#     mkdir -p prior_study_results
#     sbatch scripts/prior_study/submit_baseline.sbatch
# Monitor:
#     squeue --me
#     tail -f prior_study_results/slurm_<jobid>.log
# Cancel:
#     scancel <jobid>
#
# Override paths/label at submit time, e.g.:
#     LABEL=unconstrained sbatch --export=ALL,LABEL scripts/prior_study/submit_baseline.sbatch
#
# For multi-GPU inference also bump the GPU count in BOTH places:
#     edit  #SBATCH --gres=gpu:4  above, then submit with
#     NUM_GPUS=4 BATCH_SIZE=16 NUM_WORKERS=8 \
#       sbatch --export=ALL,NUM_GPUS,BATCH_SIZE,NUM_WORKERS scripts/prior_study/submit_baseline.sbatch
# (baseline is single-clip; multi-GPU rarely moves wall-clock much — profile first.)
#
# NOTE: intentionally no set -u. Conda activate hooks (libblas_mkl_activate.sh)
# reference unset vars like MKL_INTERFACE_LAYER and abort under set -u.
set -eo pipefail

# ---- locations --------------------------------------------------------------
# Defaults are generic: repo = submit directory, conda derived from `conda`
# on PATH. Override per site, e.g.:
#     REPO_DIR=/p/scratch/.../Holotype CONDA_SH=/path/to/conda.sh \
#       sbatch --export=ALL,REPO_DIR,CONDA_SH scripts/prior_study/submit_baseline.sbatch
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
REPO_DIR="${REPO_DIR:-.}"
CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-pytorch3d}"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
mkdir -p "$REPO_DIR/prior_study_results"

# ---- activate env -----------------------------------------------------------
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"

# ---- run the study (wrapper does pre-flight + benchmark + inference + analysis)
bash scripts/prior_study/run_baseline_study.sh

echo "Done — $(date)"
