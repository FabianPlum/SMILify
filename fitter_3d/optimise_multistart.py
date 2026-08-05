"""M3 — multi-start pose search.

RATIONALE
Arm E1 showed the pose problem is multimodal, not step-size limited: raising the joint_rot
learning rate 10x made the fit worse than its own initialisation. The textbook response to a
multimodal objective that a local optimizer cannot escape is not a better local optimizer --
it is to start from several places and keep the best basin. This is basin hopping / random
restarts, and it is exactly what SMPLify-style pipelines do when they try multiple body
orientations before committing.
  [Bogo et al., ECCV 2016 "Keep it SMPL" -- multiple initialisations for the torso;
   Wales & Doye, J.Phys.Chem.A 1997 -- basin hopping;
   the rotation-seeding idea also appears in Go-ICP / FGR as a cheap alternative to
   branch-and-bound over SO(3): Zhou et al., ECCV 2016 "Fast Global Registration"]

WHAT VARIES PER START
  * global roll about the anterior-posterior axis (X). probe_04 measured that the
    chamfer-optimal roll is non-zero for 68% of specimens EVEN THOUGH the scans are
    canonically oriented -- i.e. plain chamfer prefers an anatomically wrong roll. Seeding
    several rolls and letting the full objective (not just chamfer) choose is the honest
    way to handle that.
  * a global leg-elevation offset applied to the coxa joints. probe_06 measured the
    template's legs sit at -20.6 deg elevation vs the -30..-60 deg of a standing ant, so
    the correct rest pose is somewhere in a range, not at a point.

Each start is run for a short pose-only budget, scored by the SAME objective, and the best
per specimen is kept. Because the batch dimension is per-specimen, different specimens may
select different starts -- which is the point.

Cost note: K starts cost K x the short budget, then one long refinement. With K=6 and a
300-iteration probe that is ~1800 extra iterations, comparable to one baseline run.
"""

import argparse
import gc
import glob
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagnostics", "moonshot")))

import config  # noqa: E402
from pytorch3d.ops import sample_points_from_meshes, knn_points  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter, get_meshes  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402


def axis_angle_x(deg, B, device):
    """Axis-angle vector for a rotation of `deg` about world X, batched."""
    t = np.radians(deg)
    v = torch.zeros(B, 3, device=device)
    v[:, 0] = t
    return v


def score_fit(smal, targets, n=6000):
    """Per-specimen symmetric chamfer -- the selection criterion. Returns (B,)."""
    with torch.no_grad():
        verts = smal()
        mesh = get_meshes(verts, smal.faces, device=verts.device)
        sp = sample_points_from_meshes(mesh, n)
        tp = sample_points_from_meshes(targets, n)
        d1 = knn_points(sp, tp, K=1).dists[..., 0].mean(dim=1)
        d2 = knn_points(tp, sp, K=1).dists[..., 0].mean(dim=1)
        return (d1 + d2).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", required=True)
    ap.add_argument("--results_dir", default="diagnostics/moonshot/runs/M3_multistart")
    ap.add_argument("--max_meshes", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe_its", type=int, default=250)
    ap.add_argument("--refine_its", type=int, default=1200)
    ap.add_argument("--deform_its", type=int, default=800)
    ap.add_argument("--rolls", type=str, default="-30,-15,0,15,30")
    ap.add_argument("--leg_elev", type=str, default="0.0,-0.35")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    if args.max_meshes > 0:
        files = files[: args.max_meshes]
    names = [os.path.basename(f) for f in files]
    _, targets = load_meshes(mesh_files=files, device=device)
    B = len(targets)
    print(f"[multistart] {B} meshes", flush=True)

    with open(config.SMAL_FILE, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    jnames = list(dd["J_names"])
    # coxa joints, in joint_rot indexing (which excludes the root)
    coxa_rows = [i - 1 for i, n in enumerate(jnames) if "_co_" in n and i > 0]
    print(f"[multistart] coxa joint rows: {coxa_rows}", flush=True)

    rolls = [float(x) for x in args.rolls.split(",")]
    elevs = [float(x) for x in args.leg_elev.split(",")]
    starts = [(r, e) for r in rolls for e in elevs]
    print(f"[multistart] {len(starts)} starts = {len(rolls)} rolls x {len(elevs)} leg elevations", flush=True)

    best_score = torch.full((B,), float("inf"), device=device)
    best_state = None

    common = dict(
        target_meshes=targets,
        out_dir=args.results_dir,
        device=device,
        mesh_names=names,
        n_sample=6000,
        edge_mode="rest",
        lr_decay=True,
    )

    for si, (roll, elev) in enumerate(starts):
        smal = SMAL3DFitter(batch_size=B, device=device, shape_family=-1)
        with torch.no_grad():
            smal.global_rot.data = axis_angle_x(roll, B, device)
            if elev != 0.0:
                # rotate every coxa about its own mediolateral axis to lower the legs.
                # Sign convention checked against the template: negative y-rotation at the
                # coxa moves the distal leg downward in -Z for both sides, because the
                # template's leg chains are mirrored in position but share axis convention.
                smal.joint_rot.data[:, coxa_rows, 1] = elev
        st = MoonshotStage(
            nits=args.probe_its,
            scheme="default",
            smal_3d_fitter=smal,
            name=f"probe_{si}",
            lr=0.02,
            custom_lrs={"joint_rot": 0.006},
            loss_weights={
                "w_chamfer": 1.0,
                "w_edge": 0.05,
                "w_normal": 0.01,
                "w_laplacian": 0.005,
                "w_beta_prior": 0.002,
                "w_sym": 0.5,
            },
            log_every=0,
            **common,
        )
        st.run()
        s = score_fit(smal, targets)
        improved = s < best_score
        n_imp = int(improved.sum())
        if best_state is None:
            best_state = {k: v.detach().clone() for k, v in smal.state_dict().items()}
            best_score = s.clone()
        else:
            for k, v in smal.state_dict().items():
                best_state[k][improved] = v.detach()[improved]
            best_score = torch.where(improved, s, best_score)
        print(
            f"  start {si:2d} roll={roll:+6.1f} elev={elev:+5.2f}  "
            f"mean score {float(s.mean()):.6f}  best-for {n_imp:2d}/{B} specimens",
            flush=True,
        )
        del smal, st
        torch.cuda.empty_cache()

    # ---- refine from the per-specimen best start ----
    print(f"[multistart] refining from per-specimen best (mean {float(best_score.mean()):.6f})", flush=True)
    smal = SMAL3DFitter(batch_size=B, device=device, shape_family=-1)
    smal.load_state_dict(best_state)

    lw = {"w_chamfer": 1.0, "w_edge": 0.05, "w_normal": 0.01, "w_laplacian": 0.005, "w_beta_prior": 0.002, "w_sym": 0.5}
    for nm, its, sch, lr, jlr, extra in [
        ("MS1_refine", args.refine_its, "default", 0.01, 0.004, {}),
        ("MS2_deform", args.deform_its, "all", 0.002, 0.001, {"w_offset": 3.0}),
    ]:
        w = dict(lw)
        w.update(extra)
        st = MoonshotStage(
            nits=its,
            scheme=sch,
            smal_3d_fitter=smal,
            name=nm,
            lr=lr,
            custom_lrs={"joint_rot": jlr},
            loss_weights=w,
            log_every=200,
            **common,
        )
        st.run()
        st.save_npz(labels=names)

    # record which start each specimen chose -- useful for the report
    np.savez(
        os.path.join(args.results_dir, "start_selection.npz"),
        best_score=best_score.cpu().numpy(),
        starts=np.array(starts),
        labels=np.array(names),
    )
    del smal
    torch.cuda.empty_cache()
    gc.collect()
    print("[multistart] done", flush=True)


if __name__ == "__main__":
    main()
