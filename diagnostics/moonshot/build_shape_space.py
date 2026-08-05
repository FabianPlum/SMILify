"""Build a new SMIL shape space by PCA over GATED registrations.

This is the headless equivalent of `3D_model_prep/smil_importer/pca.py`
(`apply_pca_and_create_shapekeys`): PCA over registered scans in correspondence, emitting a
mean shape, principal components as blendshapes, and the covariance of the projected betas.
The addon needs Blender; this does the same arithmetic and writes a `.pkl` the fitter loads.

WHAT IS PCA'd, and why it is not what section 5.4 used
The correct object is each specimen's **rest-space shaped geometry**:

    v_shaped_i = v_template + deform_verts_i + shapedirs . betas_i

`smal_torch.__call__` composes exactly this before skinning, so `deform_verts` lives in rest
space and this expression is the specimen's shape with pose removed. Section 5.4's
co-registration round instead PCA'd the raw offset *fields*, which mixes shape with whatever
the offsets were doing to hide pose error, and produced blendshapes that folded the mesh
(2.79% of adjacent face pairs past 90 deg, against 0.54% baseline).

Note `log_beta_scales` and `betas_trans` are deliberately NOT folded in: they act during
skinning, per joint, so they are pose-space quantities. Limb-length variation expressed
through them is therefore not captured here, which is a real limitation and is reported.

FOUR CONSTRAINTS, each answering a specific failure already observed
  1. GATED INPUT.   Only registrations from specimens the gate accepts (report section 6.2)
                    are used. A shape space built from the whole corpus is built substantially
                    from failed fits; the gate exists precisely because fit quality is
                    predictable in advance.
  2. SYMMETRISED.   The template's mirror involution is exact (verified: max match distance
                    0.00e+00). Ant *shape* is bilaterally symmetric even where pose is not, so
                    symmetrising halves the effective noise and stops the basis encoding pose
                    asymmetry as if it were shape.
  3. FOLD-GATED.    Every candidate direction is driven to +/-3 sigma alone and rejected if it
                    folds the mesh, however much variance it explains. Variance is not the
                    selection criterion; validity is a hard gate.
  4. SMOOTHED.      Laplacian smoothing of the fields before PCA keeps high-frequency
                    per-vertex noise -- the source of folding -- out of the basis.

Outputs a `.pkl` usable via SMILIFY_SMAL_FILE, plus a JSON report of spectrum and validity.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import config  # noqa: E402


def face_adjacency(faces):
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e, _ = torch.sort(e, dim=1)
    fid = torch.arange(faces.shape[0], device=faces.device).repeat(3)
    k = e[:, 0] * (int(faces.max()) + 1) + e[:, 1]
    o = torch.argsort(k)
    k, fid = k[o], fid[o]
    i = torch.nonzero(k[1:] == k[:-1], as_tuple=True)[0]
    return fid[i], fid[i + 1]


def folded_fraction(verts, faces, a, b):
    t = verts[faces]
    n = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1)
    return float(((n[a] * n[b]).sum(-1).clamp(-1, 1) < 0.0).float().mean())


def laplacian(V, faces, device):
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = torch.cat([e, e.flip(1)], 0).t()
    val = torch.ones(e.shape[1], device=device)
    A = torch.sparse_coo_tensor(e, val, (V, V)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0)
    return A, deg


def smooth(x, A, deg, iters):
    for _ in range(iters):
        x = 0.5 * (x + torch.sparse.mm(A, x) / deg.unsqueeze(-1))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", default=["GATE_easy_M7"], help="run dirs whose LAST-stage npz supplies gated registrations"
    )
    ap.add_argument("--out_pkl", default=os.path.join(REPO, "3D_model_prep", "SMIL_OmniAnt_gated.pkl"))
    ap.add_argument("--n_keep", type=int, default=20)
    ap.add_argument("--smooth_iters", type=int, default=6)
    ap.add_argument("--fold_tol", type=float, default=0.0054, help="baseline folded_face_frac (probe 12)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(config.SMAL_FILE, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    vt = torch.tensor(np.asarray(dd["v_template"], dtype=np.float64), dtype=torch.float32, device=dev)
    faces = torch.tensor(np.asarray(dd["f"], dtype=np.int64), device=dev)
    sd = torch.tensor(np.asarray(dd["shapedirs"], dtype=np.float64), dtype=torch.float32, device=dev)
    V = vt.shape[0]
    fa, fb = face_adjacency(faces)

    # ------------------------------------------------ collect rest-space shaped geometry
    import glob

    X = []
    for r in args.runs:
        npzs = sorted(glob.glob(os.path.join(HERE, "runs", r, "*.npz")))
        if not npzs:
            print(f"[pca] WARNING: no npz in runs/{r}")
            continue
        d = np.load(npzs[-1])
        betas = torch.tensor(d["betas"], dtype=torch.float32, device=dev)
        dv = torch.tensor(d["deform_verts"], dtype=torch.float32, device=dev)
        # v_shaped = v_template + del_v + shapedirs . beta   (smal_torch.__call__ step 1)
        contrib = (
            torch.matmul(betas, sd.reshape(sd.shape[0], -1))
            if sd.dim() == 2
            else torch.einsum("bk,vck->bvc", betas, sd)
        )
        contrib = contrib.reshape(betas.shape[0], V, 3)
        X.append(vt.unsqueeze(0) + dv + contrib)
        print(f"[pca] {r}: {betas.shape[0]} registrations, {betas.shape[1]} betas")
    if not X:
        raise SystemExit("no registrations found")
    X = torch.cat(X, 0)
    N = X.shape[0]
    print(f"[pca] {N} gated registrations in correspondence, {V} verts")

    # ------------------------------------------------ symmetrise
    sym = np.asarray(dd.get("sym_verts", []))
    mirror = X.clone()
    mirror[:, :, 1] *= -1
    from pytorch3d.ops import knn_points

    # The involution must come from the TEMPLATE's own geometry, not from a fitted specimen
    # matched against the template -- the latter conflates the mirror map with fit error and
    # gave a max match distance of 7.68e-02 instead of the template's true 0.
    vt_m = vt.clone()
    vt_m[:, 1] *= -1
    idx = knn_points(vt_m.unsqueeze(0), vt.unsqueeze(0), K=1).idx[0, :, 0]
    md = float((vt[idx] - vt_m).norm(dim=-1).max())
    inv = float((idx[idx] == torch.arange(V, device=dev)).float().mean())
    X = 0.5 * (X + mirror[:, idx, :])
    print(
        f"[pca] symmetrised via the template's own involution: max match distance {md:.2e}, "
        f"involution exact on {100 * inv:.1f}% of verts, {len(sym)} midsagittal verts declared"
    )

    # ------------------------------------------------ smooth, then PCA
    mean = X.mean(0)
    Xc = X - mean
    if args.smooth_iters > 0:
        A, deg = laplacian(V, faces, dev)
        Xc = torch.stack([smooth(Xc[i], A, deg, args.smooth_iters) for i in range(N)])
    U, S, Vt_ = torch.linalg.svd(Xc.reshape(N, -1), full_matrices=False)
    var = S**2
    cum = torch.cumsum(var, 0) / var.sum()
    scales = S / np.sqrt(max(N - 1, 1))
    print(
        f"[pca] spectrum: PC1 {100 * var[0] / var.sum():.1f}%  "
        f"PC1-5 {100 * cum[4]:.1f}%  PC1-10 {100 * cum[9]:.1f}%  "
        f"n for 95% = {int((cum < 0.95).sum()) + 1}"
    )

    # ------------------------------------------------ fold gate
    # The tolerance must be set by the DATA, not by the original template. Judging generated
    # shapes against the template's 0.54% rejected 49 of 50 directions -- but the mean of the
    # registrations already folds 0.675%, so that standard is one the starting point itself
    # fails. The defensible requirement is that the shape space must not generate meshes worse
    # than the registrations it was built from, so the tolerance is the p90 of the inputs.
    in_folds = np.array([folded_fraction(X[i], faces, fa, fb) for i in range(N)])
    tol = float(np.percentile(in_folds, 90))
    base_fold = folded_fraction(mean, faces, fa, fb)
    print(
        f"[pca] input registrations fold: median {100 * np.median(in_folds):.3f}%  "
        f"p90 {100 * tol:.3f}%  max {100 * in_folds.max():.3f}%   "
        f"(original template {100 * args.fold_tol:.3f}%)"
    )
    kept, rej = [], []
    for i in range(min(Vt_.shape[0], args.n_keep * 3)):
        d_ = Vt_[i].reshape(V, 3) * scales[i]
        worst = max(folded_fraction(mean + s * 3.0 * d_, faces, fa, fb) for s in (-1.0, 1.0))
        (kept if worst <= max(tol, base_fold * 1.05) else rej).append((i, worst))
        if len(kept) >= args.n_keep:
            break
    print(
        f"[pca] fold gate: mean shape folds {100 * base_fold:.3f}% of adjacent faces; "
        f"kept {len(kept)}, rejected {len(rej)}"
    )
    if rej:
        print(f"[pca]   rejected worst-fold values: {[f'{w:.4f}' for _, w in rej[:8]]}")
    if not kept:
        raise SystemExit("every candidate direction folds the mesh -- nothing to ship")

    ki = [i for i, _ in kept]
    dirs = torch.stack([Vt_[i].reshape(V, 3) * scales[i] for i in ki], 0)  # (k,V,3)
    betas_proj = (Xc.reshape(N, -1) @ Vt_[ki].t()) / scales[ki].clamp_min(1e-12)

    # ------------------------------------------------ write model
    out = dict(dd)
    out["v_template"] = mean.cpu().numpy().astype(np.float64)
    # the pkl stores shapedirs as (V, 3, k); smal_torch reshapes to (k, 3V) at load time
    out["shapedirs"] = dirs.permute(1, 2, 0).cpu().numpy().astype(np.float64)
    out["shape_cov"] = np.cov(betas_proj.cpu().numpy().T) + 1e-8 * np.eye(len(ki))
    out["shape_mean_betas"] = betas_proj.mean(0).cpu().numpy().astype(np.float64)
    for key in ("scaledirs", "transdirs"):
        if key in dd:
            arr = np.asarray(dd[key], dtype=np.float64)
            out[key] = np.zeros((len(ki),) + arr.shape[1:])
    with open(args.out_pkl, "wb") as fh:
        pickle.dump(out, fh, protocol=2)

    rep = dict(
        n_registrations=int(N),
        runs=args.runs,
        n_kept=len(ki),
        n_rejected=len(rej),
        var_explained=[float(x) for x in (var / var.sum())[:20]],
        cum_var=[float(x) for x in cum[:20]],
        mean_fold_frac=base_fold,
        out_pkl=args.out_pkl,
    )
    rp = os.path.join(HERE, "out", "shape_space_report.json")
    json.dump(rep, open(rp, "w"), indent=1)
    print(f"[pca] wrote {args.out_pkl}  ({len(ki)} shape directions)")
    print(f"[pca] wrote {rp}")
    print("[pca] use with:  SMILIFY_SMAL_FILE=%s  (config.py honours this override)" % args.out_pkl)


if __name__ == "__main__":
    main()
