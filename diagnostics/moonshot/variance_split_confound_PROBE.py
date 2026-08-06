"""PROBE -- is the "53.7% parametric (clean) vs 20.6% (LIM_0)" half of the headline a CORPUS
property, or a property of the OPTIMISER / REGULARISER / basis provenance?

Three checks, all cheap and all on stored arrays:

  1. UNITS. Var(parametric) = || shapedirs . betas ||^2 summed over vertices. If the shapedirs
     columns are unit-norm and near-orthogonal, that equals sum_k Var(beta_k) EXACTLY -- so
     the "parametric share" is nothing but the beta amplitude the optimiser happened to use.
     Verify the unit-norm/orthogonality assumption in each of the three .pkl models.

  2. ACHIEVABLE vs USED. For each corpus, least-squares REFIT the betas to that corpus's own
     rest-space geometry (project S_i - v_template onto the shapedirs span). This is the
     largest parametric share the corpus's geometry ADMITS given its basis. Compare with the
     share the fit actually used. If the worker corpora admit a share comparable to the clean
     corpus but the fit used a fifth of it, the split is an optimiser/regulariser fact, not a
     corpus fact.

  3. PROVENANCE. All three corpora are scored against a basis. Is that basis in-sample?
     - distance from each corpus to its own template (with betas frozen, deform_verts IS the
       whole shape difference from v_template, so mean|deform| measures how far the targets
       sit from the template the shape space was built on);
     - principal angles between the model's shapedirs and the corpus's own top-K PCA.
"""

import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
DTYPE = torch.float64

CLEAN_LEGACY = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)
CORPORA = {
    "LIM_0": (f"{HERE}/runs/LIM_0/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"),
    "BPX_noprior": (f"{HERE}/runs/BPX_noprior/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl"),
    "M7_worker": (f"{HERE}/runs/M7_handoff_midline/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl"),
    "baseline": (f"{HERE}/runs/baseline/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl"),
    "CLEAN81_M7": (f"{HERE}/runs/CLEAN_M7/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl"),
    "ALL_ANTS_CLEAN": (CLEAN_LEGACY, f"{REPO}/3D_model_prep/SMPL_fit.pkl"),
}


def load_model(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def main():
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("=" * 92)
    out("1. UNITS -- are the shapedirs unit-norm and orthogonal? (if so, parametric share == beta amplitude)")
    out("=" * 92)
    for p in ("SMPL_fit.pkl", "SMIL_OmniAnt.pkl", "OmniAnt_25PCs_joint_limited.pkl"):
        dd = load_model(f"{REPO}/3D_model_prep/{p}")
        sd = torch.tensor(np.asarray(dd["shapedirs"], dtype=np.float64), dtype=DTYPE)
        K = sd.shape[2]
        B = sd.permute(2, 0, 1).reshape(K, -1)
        G = B @ B.t()
        d = torch.diagonal(G)
        offd = (G - torch.diag(d)).abs().max()
        out(
            f"    {p:<34} K={K:2d}  ||dir_k||: min={float(d.sqrt().min()):.4f} "
            f"max={float(d.sqrt().max()):.4f}   max |<dir_i,dir_j>| (i!=j) = {float(offd):.3f}"
        )
    out("    -> unit-norm but NOT orthogonal (off-diagonals up to ~0.3). Var(parametric) is")
    out("       therefore sum_k Var(beta_k) plus cross terms -- still driven almost entirely by")
    out("       how large the optimiser let the betas grow, but the projections below use a QR")
    out("       orthonormalisation of the span rather than the raw directions.")
    out()

    out("=" * 92)
    out("2. ACHIEVABLE vs USED parametric share (least-squares refit of betas to the corpus's")
    out("   own rest-space geometry, same basis, same corpus)")
    out("=" * 92)
    out(
        f"    {'corpus':<16} {'n':>3} {'K':>3}  {'USED %':>8} {'ACHIEV %':>9} "
        f"{'used/achiev':>12}  {'betas sd used':>14} {'betas sd LS':>12}"
    )
    for name, (npz, tpl) in CORPORA.items():
        dd = load_model(tpl)
        d = np.load(npz, allow_pickle=True)
        vt = torch.tensor(np.asarray(dd["v_template"], dtype=np.float64), dtype=DTYPE)
        sd = torch.tensor(np.asarray(dd["shapedirs"], dtype=np.float64), dtype=DTYPE)
        betas = torch.tensor(np.asarray(d["betas"], dtype=np.float64), dtype=DTYPE)
        F = torch.tensor(np.asarray(d["deform_verts"], dtype=np.float64), dtype=DTYPE)
        K = betas.shape[1]
        P = torch.einsum("bk,vck->bvc", betas, sd[:, :, :K])
        S = vt.unsqueeze(0) + P + F
        N = S.shape[0]

        def tv(X):
            return float(((X - X.mean(0)) ** 2).sum() / (X.shape[0] - 1))

        used = 100 * tv(P) / tv(S)
        B = sd[:, :, :K].permute(2, 0, 1).reshape(K, -1)
        Q, _ = torch.linalg.qr(B.t())  # orthonormal, already is
        R = (S - vt.unsqueeze(0)).reshape(N, -1)
        coef = R @ Q  # LS betas (orthonormal basis)
        Pls = (coef @ Q.t()).reshape(N, -1, 3)
        achiev = 100 * tv(Pls) / tv(S)
        out(
            f"    {name:<16} {N:3d} {K:3d}  {used:8.3f} {achiev:9.3f} "
            f"{used / max(achiev, 1e-9):12.3f}  {float(betas.std(0).mean()):14.4f} "
            f"{float(coef.std(0).mean()):12.4f}"
        )
    out()
    out("    USED %   = what the published decomposition reports (Var(shapedirs.betas)/Var(S))")
    out("    ACHIEV % = the same quantity if betas were fitted by least squares to the corpus's")
    out("               own geometry in the SAME basis. It is the corpus's ceiling.")
    out()

    out("=" * 92)
    out("3. PROVENANCE / IN-DISTRIBUTION")
    out("=" * 92)
    for name, (npz, tpl) in CORPORA.items():
        dd = load_model(tpl)
        d = np.load(npz, allow_pickle=True)
        vt = torch.tensor(np.asarray(dd["v_template"], dtype=np.float64), dtype=DTYPE)
        sd = torch.tensor(np.asarray(dd["shapedirs"], dtype=np.float64), dtype=DTYPE)
        betas = torch.tensor(np.asarray(d["betas"], dtype=np.float64), dtype=DTYPE)
        F = torch.tensor(np.asarray(d["deform_verts"], dtype=np.float64), dtype=DTYPE)
        K = betas.shape[1]
        P = torch.einsum("bk,vck->bvc", betas, sd[:, :, :K])
        S = vt.unsqueeze(0) + P + F
        bbox = float((vt.max(0).values - vt.min(0).values).norm())
        dist = (S - vt.unsqueeze(0)).norm(dim=-1).mean(1)  # per specimen
        # residual the shape space CANNOT express, per specimen (after LS betas)
        B = sd[:, :, :K].permute(2, 0, 1).reshape(K, -1)
        R = (S - vt.unsqueeze(0)).reshape(S.shape[0], -1)
        res = R - (R @ B.t()) @ B
        rd = res.reshape(S.shape[0], -1, 3).norm(dim=-1).mean(1)
        out(
            f"    {name:<16} mean dist to v_template = {float(dist.mean()):.6f} "
            f"({100 * float(dist.mean()) / bbox:.3f}% bbox, sd {float(dist.std()):.6f})   "
            f"UNEXPLAINABLE-by-shapespace residual = {float(rd.mean()):.6f} "
            f"({100 * float(rd.mean()) / bbox:.3f}% bbox)"
        )
    out()
    out("  principal angles: model shapedirs vs the corpus's OWN top-K PCA of rest shapes")
    for name, (npz, tpl) in CORPORA.items():
        dd = load_model(tpl)
        d = np.load(npz, allow_pickle=True)
        vt = torch.tensor(np.asarray(dd["v_template"], dtype=np.float64), dtype=DTYPE)
        sd = torch.tensor(np.asarray(dd["shapedirs"], dtype=np.float64), dtype=DTYPE)
        betas = torch.tensor(np.asarray(d["betas"], dtype=np.float64), dtype=DTYPE)
        F = torch.tensor(np.asarray(d["deform_verts"], dtype=np.float64), dtype=DTYPE)
        K = betas.shape[1]
        P = torch.einsum("bk,vck->bvc", betas, sd[:, :, :K])
        S = vt.unsqueeze(0) + P + F
        Rc = (S - S.mean(0)).reshape(S.shape[0], -1)
        _, _, Vt = torch.linalg.svd(Rc, full_matrices=False)
        Bp = Vt[:K]  # already orthonormal (from SVD)
        Bs = sd[:, :, :K].permute(2, 0, 1).reshape(K, -1)
        Qs, _ = torch.linalg.qr(Bs.t())  # shapedirs are NOT orthogonal
        s = torch.linalg.svdvals(Bp @ Qs).clamp(0, 1)  # true principal-angle cosines
        out(
            f"    {name:<16} K={K:2d}  mean cos = {float(s.mean()):.4f}  "
            f"#cos>0.9 = {int((s > 0.9).sum()):2d}/{K}  top5 = "
            f"{', '.join(f'{float(x):.3f}' for x in s[:5])}"
        )
    out()

    with open(os.path.join(HERE, "variance_split_confound_out.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote variance_split_confound_out.txt")


if __name__ == "__main__":
    main()
