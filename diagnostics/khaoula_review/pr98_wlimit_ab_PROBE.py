"""PR #98, question 1: does `w_limit: 100` genuinely stop the mandibles being fitted?

The static analysis said: against OmniAnt_25PCs_joint_limited.pkl, 10 of 81 already-accepted
ALL_ANTS_CLEAN registrations violate the authored limits, ALL of them on `ma_r`/`ma_l` axis 1,
whose authored bound (+-10 deg) is TIGHTER than the range those fits actually use
(ma_r axis 1 spans [-17.4, +16.8] deg). Clamping to the bound without re-optimising moves the
mandible surface by up to 1.08% of specimen extent.

That is an upper bound, not an answer. In a real run the fit gets to compensate: the other two
mandible axes are wide open (+-30 / +-70 deg), and Stages 2-3 are `deform` stages where
`deform_verts` moves every vertex freely with NO limit term at all (the PR only puts w_limit in
Stage_1_default). So the question is empirical.

DESIGN
  arm A   w_limit = 0    (master behaviour)
  arm B   w_limit = 100  (what ants_cfg.yaml ships in this PR)
  identical in every other respect: same 4-stage ants_cfg schedule, same seed, same batch,
  same model (the new 25-PC joint-limited one).

  specimens: the 10 known violators + 10 non-violating controls. The controls matter -- if the
  prior degrades THEM too, the cost isn't about the mandibles at all.

READING, fixed before the run:
  * primary   mandible-region surface residual (mandible verts -> target surface). If the prior
              genuinely prevents mandible fitting, this rises for the violators in arm B.
  * secondary global residual. Guards against "mandible got better, everything else got worse".
  * mechanism final ma_* axis-1 angle and total hinge. Confirms the prior actually bound.
  * control   the 10 non-violators should be unchanged on all of the above.

Chamfer is deliberately NOT the headline: it is minimised by a collapsed model as easily as by
a correct one, which is exactly why the PR's own integration test asserting on it is unsound.
"""

import os
import pickle
import sys

import numpy as np
import torch

OUT = os.path.dirname(os.path.abspath(__file__))
MAIN_REPO = os.path.abspath(os.path.join(OUT, "..", ".."))
# `w_limit` only exists on the PR branch, so point PR98_REPO at a worktree checked out at the
# PR head. Defaults to this repo (which works once the PR is merged).
REPO = os.environ.get("PR98_REPO", MAIN_REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

from smal_fitter.neuralSMIL.configs import apply_smal_file_override  # noqa: E402

# untracked, so always resolved against the main checkout rather than the PR worktree
MODEL = os.path.join(MAIN_REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl")
apply_smal_file_override(MODEL, shape_family=-1)

import config  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter, Stage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402

MESH_DIR = "/media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN"
FIT = "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
N_CONTROL = 10
SEED = 0

# ants_cfg.yaml transcribed verbatim, with w_limit injected into Stage_1_default only
# (exactly where the PR puts it).
SCHEDULE = [
    dict(name="Stage_0_init", scheme="init_rot_lock", nits=100, lr=0.05, loss_weights={}, custom_lrs=None),
    dict(
        name="Stage_1_default",
        scheme="default",
        nits=300,
        lr=0.02,
        loss_weights={"w_chamfer": 1.0, "w_edge": 0.8, "w_normal": 0.02, "w_laplacian": 0.01},
        custom_lrs={"joint_rot": 0.002},
    ),
    dict(
        name="Stage_2_deform_coarse",
        scheme="deform",
        nits=1000,
        lr=0.002,
        loss_weights={"w_chamfer": 1.0, "w_edge": 0.8, "w_normal": 0.005, "w_laplacian": 0.01},
        custom_lrs=None,
    ),
    dict(
        name="Stage_3_deform_fine",
        scheme="deform",
        nits=1000,
        lr=0.0005,
        loss_weights={"w_chamfer": 0.5, "w_edge": 0.2, "w_normal": 0.002, "w_laplacian": 0.001},
        custom_lrs=None,
    ),
]


def load_pkl(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def pick_specimens():
    """The 10 violators (from the accepted fit) + N_CONTROL seeded non-violating controls."""
    dd = load_pkl(MODEL)
    jl = np.asarray(dd["joint_limits"], float)
    mn, mx = jl[1:, :, 0], jl[1:, :, 1]
    d = np.load(FIT)
    labels = [str(x) for x in d["labels"]]
    jr = d["joint_rot"].astype(np.float64)
    h = np.maximum(jr - mx, 0) + np.maximum(mn - jr, 0)
    bad = h.max(axis=(1, 2)) > 0

    avail = {f[:-4] for f in os.listdir(MESH_DIR) if f.endswith(".obj")}
    viol = [x for x in np.array(labels)[bad] if x in avail]
    ok = [x for x in np.array(labels)[~bad] if x in avail]
    rng = np.random.default_rng(SEED)
    ctrl = list(rng.choice(sorted(ok), size=min(N_CONTROL, len(ok)), replace=False))
    return viol, ctrl, (mn, mx)


def surface_residual(verts, tgt_pts):
    """Per-vertex distance from fitted verts to a dense sampling of the target surface."""
    return knn_points(verts, tgt_pts, K=1).dists.sqrt()[..., 0]  # (B, V)


def run_arm(w_limit, names, target_meshes, tgt_pts, mand_mask, device, ma_idx):
    jn_ma_r, jn_ma_l = ma_idx
    torch.manual_seed(SEED)
    fitter = SMAL3DFitter(batch_size=len(names), device=device, shape_family=-1)
    snap = {}
    for st in SCHEDULE:
        lw = dict(st["loss_weights"])
        if st["name"] == "Stage_1_default":
            lw["w_limit"] = w_limit
        stage = Stage(
            nits=st["nits"],
            scheme=st["scheme"],
            smal_3d_fitter=fitter,
            target_meshes=target_meshes,
            mesh_names=names,
            name=st["name"],
            loss_weights=lw,
            lr=st["lr"],
            custom_lrs=st["custom_lrs"],
            device=device,
            out_dir=os.path.join(OUT, "ab_tmp"),
        )
        stage.run()
        with torch.no_grad():
            v = fitter()
            r = surface_residual(v, tgt_pts)
            # SHAPE CHANNELS. If the mandible rotation was standing in for mandible SHAPE
            # (the violators are trap-jaw / unusually long or short mandibled taxa), then
            # constraining the pose channel should push that variation into deform_verts --
            # larger mandible-region deformation for the violators, unchanged for controls.
            dvm = fitter.deform_verts.detach().norm(dim=-1)  # (B, V)
            snap[st["name"]] = dict(
                resid_all=r.mean(1).cpu().numpy(),
                resid_mand=r[:, mand_mask].mean(1).cpu().numpy(),
                resid_mand_p90=torch.quantile(r[:, mand_mask], 0.9, dim=1).cpu().numpy(),
                joint_rot=fitter.joint_rot.detach().cpu().numpy(),
                deform_mand=dvm[:, mand_mask].mean(1).cpu().numpy(),
                deform_all=dvm.mean(1).cpu().numpy(),
                betas=fitter.betas.detach().cpu().numpy(),
                logscale_mand=fitter.log_beta_scales.detach()[:, [jn_ma_r, jn_ma_l]].abs().mean((1, 2)).cpu().numpy(),
            )
        print(
            f"  [{'w=0 ' if w_limit == 0 else 'w=100'}] {st['name']:<22} "
            f"resid_all {snap[st['name']]['resid_all'].mean():.5f}  "
            f"resid_mand {snap[st['name']]['resid_mand'].mean():.5f}",
            flush=True,
        )
    return snap


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    viol, ctrl, (mn, mx) = pick_specimens()
    names = viol + ctrl
    print(f"[ab] model {MODEL}  N_POSE={config.N_POSE}  N_BETAS={config.N_BETAS}")
    print(f"[ab] {len(viol)} violators + {len(ctrl)} controls = {len(names)} specimens")
    print(f"[ab] violators: {viol}")
    print(f"[ab] controls:  {ctrl}\n")

    files = [os.path.join(MESH_DIR, f"{n}.obj") for n in names]
    mesh_names, target_meshes = load_meshes(mesh_files=files, device=device)
    torch.manual_seed(SEED)
    tgt_pts = sample_points_from_meshes(target_meshes, 100_000)

    dd = load_pkl(MODEL)
    jn = list(dd["J_names"])
    own = np.asarray(dd["weights"]).argmax(1)
    ma_idx = (jn.index("ma_r"), jn.index("ma_l"))
    mand_mask = torch.tensor(np.isin(own, list(ma_idx)), device=device)
    print(f"[ab] mandible vertices: {int(mand_mask.sum())}\n")

    A = run_arm(0.0, names, target_meshes, tgt_pts, mand_mask, device, ma_idx)
    print()
    B = run_arm(100.0, names, target_meshes, tgt_pts, mand_mask, device, ma_idx)

    mn_t, mx_t = torch.tensor(mn), torch.tensor(mx)
    nv = len(viol)
    print("\n" + "=" * 92)
    print("RESULT -- arm A (w_limit=0, master) vs arm B (w_limit=100, what this PR ships)")
    print("=" * 92)
    for stage in ["Stage_1_default", "Stage_3_deform_fine"]:
        a, b = A[stage], B[stage]
        print(f"\n{stage}")
        for grp, sl in [("VIOLATORS", slice(0, nv)), ("controls ", slice(nv, None))]:
            for key, lab in [
                ("resid_mand", "mandible residual"),
                ("resid_mand_p90", "mandible p90    "),
                ("resid_all", "global residual "),
            ]:
                va, vb = a[key][sl].mean(), b[key][sl].mean()
                d = 100 * (vb - va) / va
                print(f"  {grp}  {lab}  A {va:.5f}   B {vb:.5f}   {d:+6.2f}%")
            ja = torch.tensor(a["joint_rot"][sl])
            jb = torch.tensor(b["joint_rot"][sl])
            ha = (torch.clamp(ja - mx_t, min=0) + torch.clamp(mn_t - ja, min=0)).sum((1, 2))
            hb = (torch.clamp(jb - mx_t, min=0) + torch.clamp(mn_t - jb, min=0)).sum((1, 2))
            print(
                f"  {grp}  total hinge (rad)  A {ha.mean():.5f}   B {hb.mean():.5f}   "
                f"specimens still violating: A {int((ha > 1e-6).sum())}  B {int((hb > 1e-6).sum())}"
            )
            for key, lab in [
                ("deform_mand", "deform |dv| mandible"),
                ("deform_all", "deform |dv| global  "),
                ("logscale_mand", "|log_beta_scale| ma "),
                ("betas", "|betas| mean        "),
            ]:
                va = np.abs(a[key][sl]).mean()
                vb = np.abs(b[key][sl]).mean()
                print(f"  {grp}  {lab}  A {va:.5f}   B {vb:.5f}   {100 * (vb - va) / max(va, 1e-12):+6.2f}%")

    np.savez(
        os.path.join(OUT, "pr98_wlimit_ab_PROBE.npz"),
        names=np.array(names),
        n_viol=nv,
        **{f"A_{k}_{m}": v[m] for k, v in A.items() for m in v},
        **{f"B_{k}_{m}": v[m] for k, v in B.items() for m in v},
    )
    print(f"\nwrote {os.path.join(OUT, 'pr98_wlimit_ab_PROBE.npz')}")


if __name__ == "__main__":
    main()
