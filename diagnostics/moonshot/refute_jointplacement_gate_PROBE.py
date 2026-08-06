"""REFUTE-A -- independent re-derivation of the forward-pass gate and of the model flags
that decide what "the joint" even is.

Checks, from scratch (no reuse of the prior agent's conclusions):
  1. static_joint_locs / sym_verts / J_regressor storage type / shapedirs shape per model
  2. vertex reconstruction error with align on and off, per corpus
  3. whether the JOINT definition used downstream (J_regressor . v_shaped, then
     batch_global_rigid) is the one the fitter would have used
  4. whether J_regressor rows sum to 1 and how localised they are (a J_regressor whose
     support is a thin surface patch necessarily places the joint ON the surface)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_placement_common import (  # noqa: E402
    align_template,
    global_rigid,
    joint_regress,
    load_model,
    rodrigues,
)

ROOT = "/home/fabi/dev/SMILify"
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

MODELS = [
    ("OmniAnt_25PCs_joint_limited.pkl", f"{ROOT}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"),
    ("SMIL_OmniAnt.pkl", f"{ROOT}/3D_model_prep/SMIL_OmniAnt.pkl"),
    ("SMPL_fit.pkl", f"{ROOT}/3D_model_prep/SMPL_fit.pkl"),
]

RUNS = [
    (
        "LIM_0",
        f"{ROOT}/diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        f"{ROOT}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
    ),
    (
        "baseline",
        f"{ROOT}/diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        f"{ROOT}/3D_model_prep/SMIL_OmniAnt.pkl",
    ),
    ("CLEAN", CLEAN_NPZ, f"{ROOT}/3D_model_prep/SMPL_fit.pkl"),
]


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def forward(npz, mdl, align, static):
    d = np.load(npz, allow_pickle=True)
    dd = load_model(mdl)
    n = d["betas"].shape[0]
    nJ = np.asarray(dd["J_regressor"]).shape[0]
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    if align:
        v_template = align_template(v_template, dd["sym_verts"])
    betas = d["betas"].astype(np.float64)
    nb = betas.shape[1]
    sd = np.asarray(dd["shapedirs"], dtype=np.float64)
    v_shaped = v_template[None] + np.einsum("nk,vck->nvc", betas, sd[:, :, :nb])
    Jr = np.asarray(dd["J_regressor"], dtype=np.float64)
    if static:
        J = np.repeat(np.asarray(dd["J"], dtype=np.float64)[None], n, axis=0)
    else:
        J = joint_regress(v_shaped, Jr)
    theta = np.concatenate([d["global_rot"][:, None, :], d["joint_rot"]], axis=1).astype(np.float64)
    Rs = rodrigues(theta.reshape(-1, 3)).reshape(n, nJ, 3, 3)
    lbs = d["log_beta_scales"].astype(np.float64)
    bt = d["betas_trans"].astype(np.float64) if "betas_trans" in d.files else None
    parents = np.asarray(dd["kintree_table"][0]).astype(int)
    Jt, A = global_rigid(Rs, J, parents, lbs, bt)
    W = np.asarray(dd["weights"], dtype=np.float64)
    T = np.einsum("vj,njab->nvab", W, A)
    vh = np.concatenate([v_shaped, np.ones((n, v_shaped.shape[1], 1))], axis=2)
    verts = np.einsum("nvab,nvb->nva", T, vh)[:, :, :3] + d["trans"][:, None, :]
    verts_full = verts + d["deform_verts"].astype(np.float64)
    ref = d["verts"].astype(np.float64)
    err = np.linalg.norm(verts_full - ref, axis=2)
    diag = np.linalg.norm(ref.max(1) - ref.min(1), axis=1).mean()
    # joints under the two competing definitions
    J_static = Jt + d["trans"][:, None, :]
    J_regress_posed = joint_regress(verts, Jr)  # smal_torch else-branch (no deform)
    J_regress_deformed = joint_regress(ref, Jr)  # regressed off the FINAL stored mesh
    return dict(
        err=err, diag=diag, J_static=J_static, J_rp=J_regress_posed, J_rd=J_regress_deformed, d=d, dd=dd, n=n, Jr=Jr
    )


def main():
    outp = f"{ROOT}/diagnostics/moonshot/refute_jointplacement_gate_out.txt"
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)

        print("=" * 92, file=fh)
        print("1. MODEL FLAGS (these decide what J is)", file=fh)
        for nm, p in MODELS:
            dd = load_model(p)
            Jr = dd["J_regressor"]
            sd = np.asarray(dd["shapedirs"])
            stat = dd.get("static_joint_locs", "<absent>")
            has_sym = "sym_verts" in dd
            v = np.asarray(dd["v_template"], dtype=np.float64)
            print(f"  {nm}", file=fh)
            print(
                f"     v_template {v.shape}  shapedirs {sd.shape}  J_regressor "
                f"{type(Jr).__name__} {np.asarray(Jr).shape}",
                file=fh,
            )
            print(
                f"     static_joint_locs={stat}   sym_verts present={has_sym}   "
                f"scaledirs={'yes' if dd.get('scaledirs') is not None else 'no'}  "
                f"transdirs={'yes' if dd.get('transdirs') is not None else 'no'}",
                file=fh,
            )
            Jrd = np.asarray(Jr, dtype=np.float64)
            rs = Jrd.sum(1)
            nnz = (np.abs(Jrd) > 1e-8).sum(1)
            print(
                f"     J_regressor row sums: min={rs.min():.6f} max={rs.max():.6f}   "
                f"nnz per row: median={np.median(nnz):.0f} min={nnz.min()} max={nnz.max()}",
                file=fh,
            )
            # scalar mean used by align_template
            print(
                f"     mean(v_template) [scalar] = {v.mean():.6e}   "
                f"mean y of sym verts = "
                f"{v[np.asarray(dd['sym_verts']).astype(int), 1].mean() if has_sym else float('nan'):.6e}",
                file=fh,
            )

        print("\n" + "=" * 92, file=fh)
        print("2. FORWARD-PASS GATE  (vert reconstruction error / bbox diag)", file=fh)
        keep = {}
        for tag, npz, mdl in RUNS:
            dd = load_model(mdl)
            for align in (False, True):
                for static in [False, True] if "J" in dd else [False]:
                    try:
                        r = forward(npz, mdl, align, static)
                    except Exception as e:
                        print(f"  {tag:<10} align={align} static={static}: FAILED {type(e).__name__}: {e}", file=fh)
                        continue
                    rel = r["err"].mean() / r["diag"]
                    print(
                        f"  {tag:<10} align={str(align):<5} static={str(static):<5} "
                        f"rel_err mean={rel:.3e}  max={r['err'].max() / r['diag']:.3e}",
                        file=fh,
                    )
                    keep[(tag, align, static)] = r
            fh.flush()

        print("\n" + "=" * 92, file=fh)
        print("3. HOW DIFFERENT ARE THE COMPETING JOINT DEFINITIONS?", file=fh)
        print("   J_static  = batch_global_rigid output + trans   (what the probe used)", file=fh)
        print("   J_rp      = J_regressor . posed verts           (smal_torch STATIC=False branch)", file=fh)
        print("   J_rd      = J_regressor . FINAL stored verts    (includes deform_verts)", file=fh)
        for tag, align in (("LIM_0", False), ("baseline", False), ("CLEAN", True)):
            r = keep.get((tag, align, False))
            if r is None:
                continue
            diag = r["diag"]
            a = np.linalg.norm(r["J_static"] - r["J_rp"], axis=2) / diag * 100
            b = np.linalg.norm(r["J_static"] - r["J_rd"], axis=2) / diag * 100
            c = np.linalg.norm(r["J_rp"] - r["J_rd"], axis=2) / diag * 100
            print(
                f"  {tag:<10} |static-regressed|      median={np.median(a):6.3f}%diag  "
                f"p90={np.percentile(a, 90):6.3f}%",
                file=fh,
            )
            print(
                f"  {tag:<10} |static-regressed_def|  median={np.median(b):6.3f}%diag  "
                f"p90={np.percentile(b, 90):6.3f}%",
                file=fh,
            )
            print(
                f"  {tag:<10} |regressed-reg_def|     median={np.median(c):6.3f}%diag  "
                f"p90={np.percentile(c, 90):6.3f}%   <- pure deform_verts effect",
                file=fh,
            )
        print(f"\nwrote {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
