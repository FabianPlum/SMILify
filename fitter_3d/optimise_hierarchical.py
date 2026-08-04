"""Runner for hierarchical part-anchored registration (M1).

Schedule, and why it is in this order:

  H0_body      Fit ONLY the body chain (thorax, gaster, head, mandibles, antennae) with a
               GLOBAL chamfer. Legs are frozen. The body is three blobs on an axis and is
               essentially unimodal, so this places the template reliably and, critically,
               places the six coxae -- which is what makes the leg partition meaningful.
  H1_legs      Freeze the body, partition the target by anatomical territory, and fit each
               leg chain against ONLY its own assigned target points. This is the step that
               removes the six-identical-legs multimodality.
  H2_joint     Unfreeze everything, keep the partition. Global refinement that can still
               not reward a leg for matching another leg's surface.
  H3_deform    Optional small free-form pass with an offset penalty, to absorb genuine
               species-specific detail the parametric model cannot express -- NOT to hide
               pose error. Offsets are expensive here by construction.
"""

import argparse
import gc
import glob
import os
import pickle
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagnostics", "moonshot")))

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_hierarchical import HierarchicalStage, vertex_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", required=True)
    ap.add_argument("--results_dir", default="diagnostics/moonshot/runs/M1_hier")
    ap.add_argument("--max_meshes", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_sample", type=int, default=8000)
    ap.add_argument("--body_its", type=int, default=900)
    ap.add_argument("--leg_its", type=int, default=1200)
    ap.add_argument("--joint_its", type=int, default=900)
    ap.add_argument("--deform_its", type=int, default=600)
    ap.add_argument("--midline", type=float, default=0.0,
                    help="weight for the midsagittal-planarity penalty. The M5 arm improved 8 of "
                         "11 metrics over the stock pipeline but regressed bilateral midline "
                         "deviation 43.8%; per-leg independent fitting plus out-of-plane body "
                         "rotation is the cause, and symmetry_penalty (joint scales) cannot see it.")
    ap.add_argument("--no_partition", action="store_true",
                    help="ablation: run the same schedule with a GLOBAL data term, so the "
                         "benefit of partitioning is separable from the benefit of staging")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    if args.max_meshes > 0:
        files = files[: args.max_meshes]
    names = [os.path.basename(f) for f in files]
    print(f"[hier] {len(files)} meshes", flush=True)
    _, targets = load_meshes(mesh_files=files, device=device)

    with open(config.SMAL_FILE, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    jnames = list(dd["J_names"])
    vg, gnames = vertex_groups(np.asarray(dd["weights"]), jnames)
    counts = {g: int((vg == i).sum()) for i, g in enumerate(gnames)}
    print(f"[hier] anatomical groups (template vertices): {counts}", flush=True)

    smal = SMAL3DFitter(batch_size=len(targets), device=device, shape_family=-1)

    common = dict(
        smal=smal,
        target_meshes=targets,
        vertex_group=vg,
        group_names=gnames,
        joint_names=jnames,
        device=device,
        out_dir=args.results_dir,
        n_sample=args.n_sample,
    )
    partitioned = not args.no_partition
    leg_groups = [g for g in gnames if g != "body"]

    stages = [
        # 1) body only, global data term -- places the coxae
        HierarchicalStage(
            "H0_body", args.body_its, active_groups=["body"], partitioned=False,
            lr=0.02, joint_lr=0.01, robust_kernel="gm", robust_scale=0.25,
            loss_weights={"w_edge": 0.05, "w_beta_prior": 0.002, "w_sym": 0.5,
                          "w_midline": args.midline}, **common,
        ),
        # 2) legs only, partitioned -- the multimodality fix
        HierarchicalStage(
            "H1_legs", args.leg_its, active_groups=leg_groups, partitioned=partitioned,
            lr=0.008, joint_lr=0.015, reassign_every=50,
            loss_weights={"w_edge": 0.05, "w_beta_prior": 0.002, "w_sym": 0.5,
                          "w_midline": args.midline}, **common,
        ),
        # 3) everything, still partitioned
        HierarchicalStage(
            "H2_joint", args.joint_its, active_groups=None, partitioned=partitioned,
            lr=0.005, joint_lr=0.005, reassign_every=50,
            loss_weights={"w_edge": 0.05, "w_beta_prior": 0.002, "w_sym": 0.5,
                          "w_midline": args.midline}, **common,
        ),
        # 4) small, expensive free-form pass
        HierarchicalStage(
            "H3_deform", args.deform_its, active_groups=None, partitioned=partitioned,
            lr=0.002, joint_lr=0.001, reassign_every=100, optimise_deform=True,
            loss_weights={"w_edge": 0.05, "w_beta_prior": 0.002, "w_sym": 0.5, "w_offset": 3.0,
                          "w_midline": args.midline},
            **common,
        ),
    ]

    for st in stages:
        st.run()
        st.save_npz(labels=names)
        if st.churn_history:
            ch = np.array(st.churn_history)
            print(f"    [{st.name}] partition churn: first {ch[0]:.3f} last {ch[-1]:.3f} "
                  f"mean {ch.mean():.3f}   (falling churn => the data term is settling)", flush=True)

    del smal, targets
    torch.cuda.empty_cache()
    gc.collect()
    print("[hier] done", flush=True)


if __name__ == "__main__":
    main()
