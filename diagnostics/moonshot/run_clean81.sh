#!/bin/bash
# BEST-CASE CONTROL, never run before. DO NOT EDIT WHILE RUNNING.
#
# ALL_ANTS_CLEAN is the 81-mesh corpus that SMIL_OmniAnt.pkl's own shape space was built
# from -- clean scAnt scans and artist models, not ethanol-preserved CT. If the pipeline
# cannot fit THESE well, the limitation is the pipeline, not the worker data, and no amount
# of quality-gating the workers will help. If it CAN, gating is the right move.
# Measured differences from the worker bench: PCA-1 yaw median 0.5 deg (vs 1.0) and max
# 9.1 deg (vs 62.2), median 3 connected components, 4x scale spread (load_meshes normalises).
#
# Same recipe as the validated best arm M7_handoff_midline: hierarchical with the midline
# term, then the moonshot stage initialised from its H2_joint.
source /home/fabi/mambaforge/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/fabi/dev/SMILify
MESH=diagnostics/moonshot/clean81
R=diagnostics/moonshot/runs
G=${1:-0}

CUDA_VISIBLE_DEVICES=$G python -u -m fitter_3d.optimise_hierarchical \
  --mesh_dir $MESH --midline 2.0 --results_dir $R/CLEAN_hier >$R/CLEAN_hier.log 2>&1
CUDA_VISIBLE_DEVICES=$G python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH \
  --yaml_src diagnostics/moonshot/cfg/M7_handoff_midline.yaml \
  --init_from $R/CLEAN_hier/H2_joint.npz \
  --results_dir $R/CLEAN_M7 >$R/CLEAN_M7.log 2>&1
CUDA_VISIBLE_DEVICES=$G python -u diagnostics/moonshot/eval_run.py \
  --run_dir $R/CLEAN_M7 --mesh_dir $MESH --all_stages >>$R/CLEAN_M7.log 2>&1
CUDA_VISIBLE_DEVICES=$G python -u diagnostics/moonshot/eval_run.py \
  --run_dir $R/CLEAN_hier --mesh_dir $MESH --all_stages >>$R/CLEAN_hier.log 2>&1
echo "[$(date +%H:%M:%S)] clean81 done"
