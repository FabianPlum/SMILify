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
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagnostics", "moonshot")))

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_hierarchical import HierarchicalStage, vertex_groups  # noqa: E402
from fitter_3d.partfield import PartFieldPartition  # noqa: E402
from pytorch3d.ops import sample_points_from_meshes  # noqa: E402


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
    ap.add_argument(
        "--midline",
        type=float,
        default=0.0,
        help="weight for the midsagittal-planarity penalty. The M5 arm improved 8 of "
        "11 metrics over the stock pipeline but regressed bilateral midline "
        "deviation 43.8%; per-leg independent fitting plus out-of-plane body "
        "rotation is the cause, and symmetry_penalty (joint scales) cannot see it.",
    )
    ap.add_argument(
        "--split_distal",
        action="store_true",
        help="give each leg a separate DISTAL group (tibia/tarsus/pretarsus). "
        "probe 13: coxa+trochanter+femur hold 94%% of a leg chain's area and "
        "tarsus+pretarsus 2.4%%, so a single per-chain data term leaves the tip "
        "unconstrained -- the cause of M7's remaining part_leg_distal regression.",
    )
    ap.add_argument(
        "--part_robust",
        type=float,
        default=0.0,
        help="if >0, use a PER-PART robust kernel whose scale is this multiple of "
        "each part's own template thickness. Measured body/distal-leg thickness "
        "ratio is 14.2x, so one global scale cannot serve both (finding 4).",
    )
    ap.add_argument(
        "--part_field",
        default="",
        help="path to a trained part-field checkpoint. Replaces the fit-derived "
        "partition with a frozen, TARGET-derived one and drops points the "
        "field calls debris from every data term. Requires --split_distal.",
    )
    ap.add_argument(
        "--pf_init",
        type=int,
        default=0,
        help="iterations of correspondence-free part-anchor pose init before H0. Matches "
        "each part's centroid and second moment to the target points the field assigned "
        "it, so there is no nearest-neighbour choice to make and no multimodality. "
        "Attacks the trapped-pose finding at the initialisation rather than the objective.",
    )
    ap.add_argument("--pf_init_mu", type=float, default=1.0, help="weight on the second-moment term of --pf_init")
    ap.add_argument(
        "--pf_body",
        action="store_true",
        help="also restrict the H0 body stage to the part field's body points. Only "
        "possible with a target-derived partition, since the fit-derived one does not "
        "exist until a fit does.",
    )
    ap.add_argument(
        "--pf_keep_debris",
        action="store_true",
        help="reassign points the field calls debris to their nearest non-debris class "
        "instead of dropping them. Probe 16: 90.8%% of predicted debris is INTERIOR to the "
        "animal and sits on the leg-body junction, taught by the debris-bridge augmentation. "
        "Dropping it blinds the fit at the coxae.",
    )
    ap.add_argument(
        "--pf_min_conf",
        type=float,
        default=0.0,
        help="part-field points below this max-softmax abstain (assigned no "
        "group) instead of committing. A frozen partition cannot correct "
        "itself, so abstention is how its uncertainty is made harmless.",
    )
    ap.add_argument(
        "--pf_ref_pts",
        type=int,
        default=30000,
        help="reference cloud size the part field is evaluated on; resampled "
        "target points are labelled by nearest neighbour into it",
    )
    ap.add_argument(
        "--beta_prior",
        type=float,
        default=0.0,
        help="shape-prior weight. DEFAULT 0, i.e. OFF, which is correct: betas_prec is legacy "
        "SMAL machinery the SMIL workflow does not use, and the stock ants_cfg.yaml sets no "
        "shape prior at all. Activating it was a mistake -- at 0.002 the prior gradient "
        "overtakes the chamfer gradient at |betas|~0.02 and is 4.9x larger at 0.1 (measured), "
        "which closed the shape space in every arm built on this pipeline. Kept as an argument "
        "only so the defect can be reproduced. See REPORT.md §6.8.",
    )
    ap.add_argument(
        "--joint_lr_mult",
        type=float,
        default=1.0,
        help="multiplier on every stage's joint_rot learning rate. The stock baseline slows "
        "pose 10x relative to shape (ants_cfg.yaml custom_lrs: joint_rot 0.002 against lr "
        "0.02), which is what keeps its betas alive; the hierarchical stages give pose an "
        "equal or HIGHER rate than shape, and pose (162 dof + 165 free joint scales) then "
        "out-competes 13 betas. See REPORT.md §6.6.",
    )
    ap.add_argument(
        "--jresid",
        type=float,
        default=0.0,
        help="penalty on the FREE per-joint scale/translation residual. Only meaningful with "
        "SMILIFY_COUPLE_JOINT_BLENDSHAPES=1, where it forces the betas to carry the variation "
        "instead of the unregularised per-specimen parameters (REPORT.md §6.6).",
    )
    ap.add_argument(
        "--split_anterior",
        action="store_true",
        help="give the head, mandibles and antennae their own partition groups instead of "
        "folding them into 'body'. REPORT §6.10: the fold means covering the head with "
        "mandible vertices costs the chamfer nothing, and every pose-mobile arm shows the "
        "head deforming 24-40%% LESS than the thorax while antennae deform 44-82%% MORE. The "
        "'too small to carry a data term' argument is false -- the head is 19.5%% of surface "
        "area, 3x any leg group.",
    )
    ap.add_argument(
        "--scale_cap",
        type=float,
        default=0.0,
        help="weight on a barrier penalising per-joint scale beyond a 2x free band "
        "(quadratic past |ln s| = ln 2). Nothing else in the pipeline bounds log_beta_scales: "
        "the stock baseline spans 138x on the head joint and 889x on an antenna, which is the "
        "head collapsing into the thorax while appendages explode to cover it. REPORT §6.10.",
    )
    ap.add_argument(
        "--trans_cap",
        type=float,
        default=0.0,
        help="weight on a barrier penalising per-joint TRANSLATION beyond an absolute band "
        "(default 0.02 model units, ~1.1%% of specimen extent). Separate from --scale_cap so "
        "parts can be allowed to shift without being allowed to resize. The free betas_trans "
        "is otherwise unregularised and reaches 0.97, over half the specimen extent.",
    )
    ap.add_argument(
        "--limit",
        type=float,
        default=0.0,
        help="weight on the authored per-joint rotation-limit hinge. Requires a model that "
        "carries 'joint_limits' (e.g. 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl via "
        "SMILIFY_SMAL_FILE). Constrains the distal leg joints that §4 shows the area-sampled "
        "data term cannot reach: M7/BPX put 37 of 96 constrained axes out of range on 100% of "
        "specimens, median 19.4 deg. Judge it with probe_19, not with surface metrics.",
    )
    ap.add_argument(
        "--soft_partition",
        type=float,
        default=0.0,
        help="responsibility temperature (a LENGTH in target units) for a SOFT partition. "
        "0 = the hard argmin assignment every previous arm used. Softening lets an ambiguous "
        "target point contribute to several groups in proportion to how well each explains "
        "it, so uncertainty reduces influence instead of becoming a wrong commitment -- the "
        "E-step of EM-style robust shape-model fitting. Judge it with probe_19, not with "
        "surface metrics.",
    )
    ap.add_argument(
        "--no_partition",
        action="store_true",
        help="ablation: run the same schedule with a GLOBAL data term, so the "
        "benefit of partitioning is separable from the benefit of staging",
    )
    # ------------------------------------------------------------------ hull hierarchy
    ap.add_argument(
        "--offset",
        type=float,
        default=3.0,
        help="w_offset in the H3_deform stage: the L2 penalty on free-form deform_verts. "
        "REPORT_HULL §5 measured that 83% of correspondence error is WITHIN a part, and "
        "§6.6 that free-form offsets carry 100% of the rest-space shape variance while "
        "generalising at 0.98 -- i.e. they are per-specimen noise that papers over pose "
        "error. Raising this trades surface proximity for correspondence consistency. "
        "Very large values (>=100) pin deform at ~0 while leaving every other parameter "
        "and the iteration budget untouched, which is the clean way to ablate free-form.",
    )
    ap.add_argument(
        "--hull_partition",
        type=int,
        default=0,
        help="cut size k for a CONVEXITY-derived partition prior (fitter_3d/hull_partition.py). "
        "0 = off. The target is broken into k near-convex chunks by an approximate convex "
        "decomposition, and each chunk is assigned to an anatomical group by MAJORITY VOTE of "
        "its own points, recomputed at every reassignment. So the chunking is target-derived "
        "and frozen while the naming stays fit-derived and revisable -- unlike --part_field, "
        "which freezes the naming too and lost decisively for that reason (REPORT §5.3).",
    )
    ap.add_argument(
        "--hull_criterion",
        default="volume",
        choices=["volume", "concavity", "hybrid", "visibility"],
        help="which notion of 'the hulls separate here' drives the merge tree",
    )
    ap.add_argument(
        "--hull_k_schedule",
        default="",
        help="comma-separated per-stage cut sizes, e.g. '5,13,20,30' for H0/H1/H2/H3. Empty "
        "keeps --hull_partition for every stage. This is the whole point of a hierarchy: "
        "coarse chunks while the body is placed, fine chunks while distal segments are fitted.",
    )
    ap.add_argument(
        "--hull_min_majority",
        type=float,
        default=0.0,
        help="a chunk whose winning group holds less than this fraction of its votes is treated "
        "as AMBIGUOUS and its points keep their own per-point labels. 0 = always vote.",
    )
    ap.add_argument("--hull_threshold", type=float, default=0.03, help="CoACD concavity threshold")
    ap.add_argument("--hull_points", type=int, default=20000, help="surface samples for the decomposition")
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
    vg, gnames = vertex_groups(
        np.asarray(dd["weights"]), jnames, split_distal=args.split_distal, split_anterior=args.split_anterior
    )
    counts = {g: int((vg == i).sum()) for i, g in enumerate(gnames)}
    print(f"[hier] anatomical groups (template vertices): {counts}", flush=True)

    smal = SMAL3DFitter(batch_size=len(targets), device=device, shape_family=-1)

    part_scale = None
    if args.part_robust > 0:
        from fitter_3d.trainer_hierarchical import part_thickness

        v_t = torch.tensor(np.asarray(dd["v_template"]), dtype=torch.float32, device=device)
        part_scale = part_thickness(v_t, torch.tensor(vg, device=device), len(gnames))
        print(
            f"[hier] per-part robust scale = {args.part_robust} x thickness; "
            f"thickness range {float(part_scale.min()):.5f}..{float(part_scale.max()):.5f} "
            f"({float(part_scale.max() / part_scale.min()):.1f}x)",
            flush=True,
        )

    # ------------------------------------------------------------------ learned part field
    # The default partition is derived from the CURRENT FIT and so can only ever confirm what
    # the fit already believes. This one is derived from the TARGET ALONE, computed once here
    # and never updated, so it is an opinion about the scan that the fit cannot edit.
    partition = None
    if args.part_field:
        from fitter_3d.partfield import PartFieldNet, predict_field, part_names

        pf_names = part_names(jnames)
        assert pf_names[:-1] == list(gnames), (
            "part-field classes must match the fitter's groups; the field is trained on the "
            "split_distal grouping, so --part_field requires --split_distal.\n"
            f"  field: {pf_names[:-1]}\n  fitter: {list(gnames)}"
        )
        ck = torch.load(args.part_field, map_location=device)
        net = PartFieldNet(n_classes=len(ck["names"]), width=ck.get("width", 1.0)).to(device)
        net.load_state_dict(ck["state"])
        net.eval()
        debris_id = pf_names.index("debris")

        ref_p, ref_l, ref_c, dbg = [], [], [], []
        for b in range(len(targets)):
            # predict_field normalises internally; the reference cloud is kept in the
            # fitter's own target frame, because that is the frame the resampled target
            # points are looked up in
            p = sample_points_from_meshes(targets[b], args.pf_ref_pts)[0]
            prob = predict_field(net, p, seed=0)
            conf, lab = prob.max(-1)
            ref_p.append(p)
            ref_l.append(lab)
            ref_c.append(conf)
            dbg.append(float((lab == debris_id).float().mean()))
        partition = PartFieldPartition(
            torch.stack(ref_p),
            torch.stack(ref_l),
            len(gnames),
            device,
            debris_id=debris_id,
            ref_conf=torch.stack(ref_c),
            min_conf=args.pf_min_conf,
            keep_debris=args.pf_keep_debris,
        )
        n_ref = len(targets) * args.pf_ref_pts
        print(
            f"[hier] part field {os.path.basename(args.part_field)} "
            f"(trained mIoU {ck.get('miou', float('nan')):.3f}) applied to "
            f"{len(targets)} targets at {args.pf_ref_pts} pts; "
            f"debris {100 * np.mean(dbg):.1f}% mean "
            f"({100 * np.min(dbg):.1f}-{100 * np.max(dbg):.1f}%); "
            f"low-confidence abstentions {100 * partition.n_lowconf / n_ref:.1f}% "
            f"(min_conf {args.pf_min_conf}); "
            + (
                f"debris REASSIGNED to nearest non-debris class ({100 * partition.n_reassigned / n_ref:.1f}% "
                f"of points), nothing excluded"
                if args.pf_keep_debris
                else "both are EXCLUDED from every data term"
            ),
            flush=True,
        )

    # ------------------------------------------------------------------ hull hierarchy
    if args.hull_partition > 0:
        if partition is not None:
            raise SystemExit("--hull_partition and --part_field are alternative partitions; pick one")
        from fitter_3d.hull_partition import build_hull_partition

        t_hull = time.time()
        partition = build_hull_partition(
            targets,
            vg,
            len(gnames),
            device,
            k=args.hull_partition,
            n_points=args.hull_points,
            coacd_threshold=args.hull_threshold,
            criterion=args.hull_criterion,
            seed=args.seed,
            min_majority=args.hull_min_majority,
        )
        print(
            f"[hier] hull partition: criterion={args.hull_criterion} k={args.hull_partition} "
            f"threshold={args.hull_threshold} min_majority={args.hull_min_majority} "
            f"({time.time() - t_hull:.0f}s for {len(targets)} targets)",
            flush=True,
        )

    common = dict(
        smal=smal,
        target_meshes=targets,
        partition=partition,
        vertex_group=vg,
        group_names=gnames,
        joint_names=jnames,
        device=device,
        out_dir=args.results_dir,
        n_sample=args.n_sample,
        split_distal=args.split_distal,
        split_anterior=args.split_anterior,
        part_scale=part_scale,
        soft_partition=args.soft_partition,
        robust_mult=args.part_robust if args.part_robust > 0 else 3.0,
    )
    partitioned = not args.no_partition
    ANTERIOR = {"head", "mandible", "antenna"}
    leg_groups = [g for g in gnames if g != "body" and g not in ANTERIOR]
    # H0 places the body; with --split_anterior the head is no longer part of 'body', so its
    # joints must be named explicitly or they would be frozen for the whole body stage.
    body_groups = ["body"] + [g for g in gnames if g in ANTERIOR]
    print(f"[hier] {len(gnames)} groups: {gnames}", flush=True)
    print(f"[hier] body stage moves {body_groups}, leg stage moves {len(leg_groups)} leg groups", flush=True)

    stages = [
        # 1) body only -- places the coxae.
        # The data term here is GLOBAL by default, because the fit-derived partition does not
        # exist yet at this point: it is computed from the fitted mesh, and at H0 the fitted
        # mesh is still the template. So the body is fitted against every target point,
        # including all six legs, and is necessarily dragged by them.
        # A TARGET-derived field has no such ordering constraint -- it is known before the
        # first iteration -- so --pf_body restricts H0 to the points the field calls body.
        # This is a capability the fit-derived partition cannot have even in principle, and
        # is therefore the more interesting of the two part-field arms.
        HierarchicalStage(
            "H0_body",
            args.body_its,
            active_groups=body_groups,
            partitioned=bool(args.pf_body and partition is not None),
            chamfer_groups=["body"] if (args.pf_body and partition is not None) else None,
            lr=0.02,
            joint_lr=0.01 * args.joint_lr_mult,
            robust_kernel="gm",
            robust_scale=0.25,
            loss_weights={
                "w_edge": 0.05,
                "w_beta_prior": args.beta_prior,
                "w_sym": 0.5,
                "w_midline": args.midline,
                "w_jresid": args.jresid,
                "w_limit": args.limit,
                "w_scale": args.scale_cap,
                "w_trans": args.trans_cap,
            },
            **common,
        ),
        # 2) legs only, partitioned -- the multimodality fix
        HierarchicalStage(
            "H1_legs",
            args.leg_its,
            active_groups=leg_groups,
            partitioned=partitioned,
            lr=0.008,
            joint_lr=0.015 * args.joint_lr_mult,
            reassign_every=50,
            loss_weights={
                "w_edge": 0.05,
                "w_beta_prior": args.beta_prior,
                "w_sym": 0.5,
                "w_midline": args.midline,
                "w_jresid": args.jresid,
                "w_limit": args.limit,
                "w_scale": args.scale_cap,
                "w_trans": args.trans_cap,
            },
            **common,
        ),
        # 3) everything, still partitioned
        HierarchicalStage(
            "H2_joint",
            args.joint_its,
            active_groups=None,
            partitioned=partitioned,
            lr=0.005,
            joint_lr=0.005 * args.joint_lr_mult,
            reassign_every=50,
            loss_weights={
                "w_edge": 0.05,
                "w_beta_prior": args.beta_prior,
                "w_sym": 0.5,
                "w_midline": args.midline,
                "w_jresid": args.jresid,
                "w_limit": args.limit,
                "w_scale": args.scale_cap,
                "w_trans": args.trans_cap,
            },
            **common,
        ),
        # 4) small, expensive free-form pass
        HierarchicalStage(
            "H3_deform",
            args.deform_its,
            active_groups=None,
            partitioned=partitioned,
            lr=0.002,
            joint_lr=0.001 * args.joint_lr_mult,
            reassign_every=100,
            optimise_deform=True,
            loss_weights={
                "w_edge": 0.05,
                "w_beta_prior": args.beta_prior,
                "w_sym": 0.5,
                "w_offset": args.offset,
                "w_midline": args.midline,
            },
            **common,
        ),
    ]

    # ---------------------------------------------------------------- part-anchor init
    # Runs BEFORE H0. Report section 5 found pose to be trapped rather than under-optimised
    # (raising the pose lr made the fit worse than its own init), and every fix so far has
    # attacked the objective rather than the starting point -- because until the part field
    # there was nothing specimen-specific to start from. This puts pose in the right basin
    # using an objective with no correspondence in it at all. See fitter_3d/part_anchor_init.
    if args.pf_init > 0:
        if partition is None:
            raise SystemExit(
                "--pf_init requires --part_field: with a fit-derived partition "
                "the anchors would be computed from the fit's own current "
                "opinion and the objective would already be satisfied."
            )
        from fitter_3d.part_anchor_init import PartAnchorInit

        tgt0 = sample_points_from_meshes(targets, args.n_sample).detach()
        partition.update(None, tgt0)
        init = PartAnchorInit(
            smal,
            partition,
            vg,
            gnames,
            device,
            n_it=args.pf_init,
            mu=args.pf_init_mu,
            faces=torch.tensor(np.asarray(dd["f"], dtype=np.int64), device=device),
        )
        init.run(tgt0)
        del tgt0
        torch.cuda.empty_cache()

    # The granularity knob: re-cut the same tree per stage. Nothing geometric is recomputed,
    # so a coarse body stage and a fine deform stage cost the same as one fixed k.
    hull_ks = [int(x) for x in args.hull_k_schedule.split(",") if x.strip()] if args.hull_k_schedule else []
    if hull_ks and args.hull_partition <= 0:
        raise SystemExit("--hull_k_schedule requires --hull_partition")
    if hull_ks and len(hull_ks) != len(stages):
        raise SystemExit(f"--hull_k_schedule needs {len(stages)} values (H0,H1,H2,H3), got {len(hull_ks)}")

    for si, st in enumerate(stages):
        if hull_ks:
            partition.set_k(hull_ks[si])
            print(f"    [{st.name}] hull partition re-cut to k={hull_ks[si]}", flush=True)
        st.run()
        st.save_npz(labels=names)
        if st.churn_history:
            ch = np.array(st.churn_history)
            print(
                f"    [{st.name}] partition churn: first {ch[0]:.3f} last {ch[-1]:.3f} "
                f"mean {ch.mean():.3f}   (falling churn => the data term is settling)",
                flush=True,
            )

    del smal, targets
    torch.cuda.empty_cache()
    gc.collect()
    print("[hier] done", flush=True)


if __name__ == "__main__":
    main()
