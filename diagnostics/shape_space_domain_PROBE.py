"""shape_space_domain_PROBE.py -- why the M6 co-registration round produced folded geometry.

Two hypotheses, both testable on artefacts already on disk:

H1 (FRAME MISMATCH).  `deform_verts` is added AFTER skinning
    (`fitter_3d/trainer.py:240`:  verts = verts + _deform_verts, applied to posed+
    translated vertices), whereas `shapedirs` are applied BEFORE skinning
    (`smal_model/smal_torch.py:236-247`:  v_shaped = v_template + beta @ shapedirs).
    M6 PCA'd the posed-frame offsets and appended them as rest-frame shapedirs.
    Prediction: un-posing the offsets (rotating each vertex offset back through the
    inverse of its LBS transform) should concentrate the PCA spectrum and reduce
    folding, because per-specimen limb orientation stops leaking into "shape".

H2 (WRONG DOMAIN).  PCA on Euclidean vertex displacements is not constrained to
    produce valid geometry.  Registration literature (SCAPE, Anguelov et al. 2005;
    Coregistration, Hirshberg et al. ECCV 2012) instead models per-triangle
    deformation gradients and recovers vertices by a least-squares "stitch".
    Prediction: PCA in the deformation-gradient domain + Poisson stitch folds less
    at equal reconstruction error.

H3 (NO SMOOTHNESS TERM).  Hirshberg's E_D penalises ||D_i - D_j||^2 / h_ij^2 over
    adjacent faces.  M6 had no such term.  Prediction: Laplacian-smoothing the
    learned directions reduces folding.

Metric matches diagnostics/moonshot/metrics.py exactly: fraction of adjacent face
pairs whose normals differ by more than 90 deg, plus the p99 dihedral.

No GPU needed.  Runtime ~1 min.
"""

import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

RUN = os.path.join(REPO, "diagnostics/moonshot/runs/A4_nofreeze/Stage_3_deform_fine.npz")
PKL = os.path.join(REPO, "3D_model_prep/SMIL_OmniAnt.pkl")
K = 24  # same number of new directions M6 appended


# --------------------------------------------------------------------------- fold metric
def face_adjacency(f):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = np.sort(e, axis=1)
    fid = np.tile(np.arange(f.shape[0]), 3)
    k = e[:, 0] * (f.max() + 1) + e[:, 1]
    o = np.argsort(k, kind="stable")
    k, fid = k[o], fid[o]
    idx = np.nonzero(k[1:] == k[:-1])[0]
    return fid[idx], fid[idx + 1]


def fold_stats(v, f, adj):
    t = v[f]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    a, b = adj
    ang = np.degrees(np.arccos(np.clip((n[a] * n[b]).sum(-1), -1, 1)))
    return float((ang > 90).mean()), float(np.percentile(ang, 99))


# --------------------------------------------------------------------------- deformation gradients
def face_frames(v, f):
    """Per-face 3x3 frame [e1, e2, n] (edge vectors + unit normal), as in SCAPE's
    'unstitched triangle' representation."""
    t = v[f]
    e1 = t[:, 1] - t[:, 0]
    e2 = t[:, 2] - t[:, 0]
    n = np.cross(e1, e2)
    n /= np.sqrt(np.linalg.norm(n, axis=1, keepdims=True)) + 1e-12  # sqrt: standard SCAPE trick
    return np.stack([e1, e2, n], axis=2)  # (F,3,3)


def build_stitch(f, nv, V0):
    """Least-squares stitch operator: given target edge vectors, solve for vertices.
    A is (2F, V); returns a factorised solver for A^T A (+ tiny ridge for the
    translation nullspace) and A^T itself."""
    F = f.shape[0]
    rows = np.concatenate([np.arange(F), np.arange(F), np.arange(F, 2 * F), np.arange(F, 2 * F)])
    cols = np.concatenate([f[:, 1], f[:, 0], f[:, 2], f[:, 0]])
    vals = np.concatenate([np.ones(F), -np.ones(F), np.ones(F), -np.ones(F)])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * F, nv))
    AtA = (A.T @ A).tocsc() + 1e-8 * sp.eye(nv, format="csc")
    return spl.factorized(AtA), A, V0


def stitch(solver, A, V0, D, f, Vt):
    """D: (F,3,3) predicted deformation gradients.  Target edges = D @ template edges."""
    Bt = face_frames(Vt, f)  # (F,3,3) template frames
    e1 = np.einsum("fij,fj->fi", D, Bt[:, :, 0])
    e2 = np.einsum("fij,fj->fi", D, Bt[:, :, 1])
    b = np.concatenate([e1, e2], axis=0)  # (2F,3)
    rhs = A.T @ b
    V = np.stack([solver(rhs[:, i]) for i in range(3)], axis=1)
    # stitch is translation-free: re-centre on the reference
    return V - V.mean(0) + V0.mean(0)


# --------------------------------------------------------------------------- PCA helper
def pca_fit(X, k):
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    return mu, Vt[:k], S, np.cumsum(S**2) / (S**2).sum()


def pca_project(X, mu, B):
    return (X - mu) @ B.T @ B + mu


def main():
    d = np.load(RUN, allow_pickle=True)
    dv = d["deform_verts"].astype(np.float64)
    B, V, _ = dv.shape
    with open(PKL, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    vt = np.asarray(dd["v_template"], dtype=np.float64)
    f = np.asarray(dd["f"], dtype=np.int64)
    adj = face_adjacency(f)
    extent = float(np.linalg.norm(vt.max(0) - vt.min(0)))
    print(f"template {vt.shape[0]} verts / {f.shape[0]} faces, bbox diag {extent:.4f}")
    print(
        f"offsets  {dv.shape}, mean |dv| = {np.linalg.norm(dv, axis=2).mean():.5f} "
        f"({100 * np.linalg.norm(dv, axis=2).mean() / extent:.2f}% of bbox diag)"
    )

    base_fold, base_p99 = fold_stats(vt, f, adj)
    print("\nWHERE THE FOLDS ACTUALLY ARE (measured on the fits themselves)")
    print(f"  template rest mesh                 folded {100 * base_fold:6.3f}%   p99 dihedral {base_p99:6.1f} deg")
    fitv = d["verts"].astype(np.float64)
    fr = [fold_stats(fitv[i], f, adj) for i in range(B)]
    print(
        f"  full fit  (model + free offsets)   folded {100 * np.mean([x[0] for x in fr]):6.3f}%   "
        f"p99 dihedral {np.mean([x[1] for x in fr]):6.1f} deg"
    )
    fr = [fold_stats(fitv[i] - dv[i], f, adj) for i in range(B)]
    print(
        f"  model-only (pose+shape, no offset) folded {100 * np.mean([x[0] for x in fr]):6.3f}%   "
        f"p99 dihedral {np.mean([x[1] for x in fr]):6.1f} deg"
    )

    # ---------------------------------------------------------------- H1: un-pose the offsets
    import torch

    import config  # noqa: F401  (SMAL module reads globals from it)
    from smal_model.batch_lbs import batch_global_rigid_transformation, batch_rodrigues

    weights = np.asarray(dd["weights"], dtype=np.float64)  # (V, J)
    J_reg = np.asarray(
        dd["J_regressor"].todense() if sp.issparse(dd["J_regressor"]) else dd["J_regressor"], dtype=np.float64
    )
    kin = np.asarray(dd["kintree_table"])
    parents = kin[0].astype(np.int64).copy()
    parents[0] = -1
    NJ = weights.shape[1]
    shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)  # (V,3,13)

    theta = np.concatenate([d["global_rot"][:, None, :], d["joint_rot"]], axis=1).astype(np.float64)
    betas = d["betas"].astype(np.float64)
    lbs_scale = torch.tensor(d["log_beta_scales"].astype(np.float32))
    lbs_trans = torch.tensor(d["betas_trans"].astype(np.float32))

    v_shaped = vt[None] + np.einsum("bk,vdk->bvd", betas, shapedirs)  # (B,V,3)
    if J_reg.shape[0] == V:  # (V,J)
        Jt = np.einsum("bvd,vj->bjd", v_shaped, J_reg)
    else:  # (J,V)
        Jt = np.einsum("bvd,jv->bjd", v_shaped, J_reg)

    Rs = batch_rodrigues(torch.tensor(theta.reshape(-1, 3).astype(np.float32))).reshape(B, NJ, 3, 3)
    _, Aglob = batch_global_rigid_transformation(
        Rs, torch.tensor(Jt.astype(np.float32)), parents, betas_logscale=lbs_scale, betas_trans=lbs_trans, num_joints=NJ
    )
    Aglob = Aglob.detach().numpy().astype(np.float64)  # (B,J,4,4)
    T = np.einsum("vj,bjmn->bvmn", weights, Aglob)  # (B,V,4,4)
    M = T[:, :, :3, :3]  # linear part per vertex

    # orthonormalise (the LBS linear part carries per-joint scaling) via polar decomposition
    Uu, _, Vvt = np.linalg.svd(M)
    R = Uu @ Vvt
    dv_rest = np.einsum("bvji,bvj->bvi", R, dv)  # R^T dv  -> rest frame

    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=2, axis2=3) - 1) / 2, -1, 1)))
    print(
        f"\n  per-vertex LBS rotation away from rest: mean {ang.mean():.1f} deg, "
        f"p90 {np.percentile(ang, 90):.1f} deg, max {ang.max():.1f} deg"
    )

    allD = np.stack([defgrad(vt, vt + dv_rest[i], f) for i in range(B)]).reshape(B, -1)
    for name, X in (
        ("posed-frame euclid offsets (what M6 used)", dv.reshape(B, -1)),
        ("rest-frame euclid offsets", dv_rest.reshape(B, -1)),
        ("rest-frame deformation gradients", allD),
    ):
        _, _, S, cum = pca_fit(X, K)
        print(f"\nPCA spectrum over all {B} specimens, {name}:")
        print("   " + "  ".join(f"PC{i + 1}:{100 * (cum[i] - (cum[i - 1] if i else 0)):.1f}%" for i in range(5)))
        print(
            f"   cumulative @24 = {100 * cum[min(23, len(cum) - 1)]:.1f}%   "
            f"components for 90% = {int(np.searchsorted(cum, 0.90)) + 1}"
        )

    # ---------------------------------------------------------------- held-out comparison
    idx = np.arange(B)
    tr, te = idx[idx % 2 == 0], idx[idx % 2 == 1]
    L = mesh_laplacian(f, V)
    solver, A, _ = build_stitch(f, V, vt)

    print(f"\n{'arm':<44}{'folded%':>9}{'p99deg':>9}{'recon RMSE':>13}{'% bbox':>9}")
    print("-" * 84)

    def report(name, recon_offsets):
        fr, p9, rm = [], [], []
        for j, i in enumerate(te):
            vrec = vt + recon_offsets[j]
            a, b = fold_stats(vrec, f, adj)
            fr.append(a)
            p9.append(b)
            rm.append(np.sqrt(((recon_offsets[j] - target[j]) ** 2).sum(1).mean()))
        print(
            f"{name:<44}{100 * np.mean(fr):>8.3f}%{np.mean(p9):>9.1f}"
            f"{np.mean(rm):>13.6f}{100 * np.mean(rm) / extent:>8.2f}%"
        )
        return np.mean(fr)

    # ---- arm A: exactly M6 (Euclidean PCA on posed-frame offsets)
    target = dv[te]
    mu, Bd, _, _ = pca_fit(dv[tr].reshape(len(tr), -1), K)
    report(
        "A  M6 as-built: euclid PCA, posed frame", pca_project(dv[te].reshape(len(te), -1), mu, Bd).reshape(-1, V, 3)
    )

    # ---- arm B: same, but smoothed directions (Hirshberg E_D analogue)
    Bs = np.stack([smooth(L, b.reshape(V, 3), 8).ravel() for b in Bd])
    Bs /= np.linalg.norm(Bs, axis=1, keepdims=True)
    report("B  + Laplacian-smoothed directions", pca_project(dv[te].reshape(len(te), -1), mu, Bs).reshape(-1, V, 3))

    # ---- arm C: un-posed (rest-frame) offsets, Euclidean PCA
    target = dv_rest[te]
    mu2, B2, _, _ = pca_fit(dv_rest[tr].reshape(len(tr), -1), K)
    report(
        "C  euclid PCA, REST frame (H1 fix)", pca_project(dv_rest[te].reshape(len(te), -1), mu2, B2).reshape(-1, V, 3)
    )

    # ---- arm D: deformation-gradient PCA + stitch, rest frame (H2 fix)
    Dtr = np.stack([defgrad(vt, vt + dv_rest[i], f) for i in tr])  # (n,F,3,3)
    Dte = np.stack([defgrad(vt, vt + dv_rest[i], f) for i in te])
    mu3, B3, _, cum3 = pca_fit(Dtr.reshape(len(tr), -1), K)
    Drec = pca_project(Dte.reshape(len(te), -1), mu3, B3).reshape(-1, f.shape[0], 3, 3)
    rec = np.stack([stitch(solver, A, vt, Drec[j], f, vt) - vt for j in range(len(te))])
    report("D  defgrad PCA + stitch, rest (H1+H2)", rec)

    # ---- arm E: defgrad PCA + stitch + smoothed directions (H2+H3)
    B3s = np.stack([smooth_faces(f, V, b.reshape(-1, 9), 8).ravel() for b in B3])
    B3s /= np.linalg.norm(B3s, axis=1, keepdims=True)
    Drec = pca_project(Dte.reshape(len(te), -1), mu3, B3s).reshape(-1, f.shape[0], 3, 3)
    rec = np.stack([stitch(solver, A, vt, Drec[j], f, vt) - vt for j in range(len(te))])
    report("E  defgrad PCA + stitch + smoothing (H2+H3)", rec)

    # ---- reference: the fits themselves (upper bound on achievable fidelity)
    target = dv_rest[te]
    report("--- reference: raw fitted offsets, rest", dv_rest[te])
    target = dv[te]
    report("--- reference: raw fitted offsets, posed", dv[te])

    # ------------------------------------------------------------------ is the offset SHAPE or POSE?
    # If the free-form offsets are mostly repairing pose-dependent skinning artefacts they
    # belong in `posedirs` (a function of pose), not in `shapedirs` (a function of identity).
    print("\nWHAT SIGNAL DO THE OFFSETS CARRY?  leave-one-out R^2 of a linear predictor")
    # NB: centring must happen INSIDE the fold.  Centring on all B rows makes the held-out
    # row an exact linear combination of the training rows, and a near-interpolating ridge
    # then scores R^2 ~ 1 even on permuted labels.  Verified: permutation control below.
    pose_feat = (Rs.reshape(B, NJ, 9).numpy()[:, 1:, :] - np.eye(3).ravel()[None, None, :]).reshape(B, -1)
    # Y has rank <= B, so solving in its own orthonormal row basis is lossless for R^2
    # and ~600x cheaper than carrying 30687 RHS columns through every ridge solve.
    _u, _s, _vt = np.linalg.svd(dv_rest.reshape(B, -1), full_matrices=False)
    Y0 = _u * _s
    rng = np.random.default_rng(0)

    def loo_r2(Xf, Yf):
        best = -np.inf
        for lam in (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5):
            err = tot = 0.0
            for i in range(B):
                m = np.ones(B, bool)
                m[i] = False
                Xm = Xf[m] - Xf[m].mean(0)
                Ym = Yf[m] - Yf[m].mean(0)
                sc = Xm.std() + 1e-12
                Xm = Xm / sc
                xi = (Xf[i] - Xf[m].mean(0)) / sc
                yi = Yf[i] - Yf[m].mean(0)
                W = np.linalg.solve(Xm.T @ Xm + lam * np.eye(Xm.shape[1]), Xm.T @ Ym)
                err += ((yi - xi @ W) ** 2).sum()
                tot += (yi**2).sum()
            best = max(best, 1 - err / tot)
        return best

    for nm, Xf in (
        ("pose  (Rs - I, SMAL's own pose_feature)", pose_feat),
        ("shape (the 13 fitted betas)", betas),
        ("joint scales (log_beta_scales)", d["log_beta_scales"].reshape(B, -1).astype(np.float64)),
    ):
        r = loo_r2(Xf, Y0)
        rp = max(loo_r2(Xf[rng.permutation(B)], Y0) for _ in range(2))
        print(f"   {nm:<44} LOO R^2 = {r:6.3f}   (permuted control {rp:6.3f})")
    iso = 24 / (B - 1)
    print(f"   [isotropic-noise baseline for 'cumulative @24' with {B} samples: {100 * iso:.1f}%]")

    # ------------------------------------------------------------------ where do the folds sit?
    part = np.asarray(dd["weights"], dtype=np.float64).argmax(1)
    fc = part[f].min(1)
    a_, b_ = adj
    t = (fitv[0] - dv[0])[f]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ang = np.degrees(np.arccos(np.clip((n[a_] * n[b_]).sum(-1), -1, 1)))
    cross_joint = fc[a_] != fc[b_]
    print("\nWHERE THE MODEL-ONLY FOLDS SIT (specimen 0, LBS-argmax part labels)")
    print(f"   adjacent-face pairs spanning two different joints: {100 * cross_joint.mean():.1f}% of all pairs")
    print(f"   folded pairs that span two joints:                 {100 * cross_joint[ang > 90].mean():.1f}%")
    print(
        f"   fold rate within a joint {100 * (ang[~cross_joint] > 90).mean():.3f}%   "
        f"across joints {100 * (ang[cross_joint] > 90).mean():.3f}%"
    )


def mesh_laplacian(f, nv):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    W = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
    W.data[:] = 1.0
    deg = np.asarray(W.sum(1)).ravel()
    return sp.diags(1.0 / np.maximum(deg, 1)) @ W


def smooth(L, X, n):
    for _ in range(n):
        X = 0.5 * X + 0.5 * (L @ X)
    return X


_FADJ = {}


def smooth_faces(f, nv, X, n):
    """Hirshberg's E_D analogue: diffuse a per-face quantity over face adjacency."""
    key = f.shape[0]
    if key not in _FADJ:
        a, b = face_adjacency(f)
        e = np.concatenate([np.stack([a, b], 1), np.stack([b, a], 1)], 0)
        W = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(f.shape[0], f.shape[0]))
        deg = np.asarray(W.sum(1)).ravel()
        _FADJ[key] = sp.diags(1.0 / np.maximum(deg, 1)) @ W
    Lf = _FADJ[key]
    for _ in range(n):
        X = 0.5 * X + 0.5 * (Lf @ X)
    return X


def defgrad(V0, V1, f):
    B0 = face_frames(V0, f)
    B1 = face_frames(V1, f)
    return B1 @ np.linalg.inv(B0)


if __name__ == "__main__":
    main()
