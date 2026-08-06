"""refute-23: independent re-derivation of probe-23 / probe-23b.

Probe-23 concluded that JOINT PLACEMENT is where the fitter spends its freedom, on three legs:
  (i)   workers displace joints 7.31% of body diagonal vs ALL_ANTS_CLEAN 2.40%  -> "3x more"
  (ii)  limits ON (LIM_1x) raises |log_beta_scales| +22.4% and |betas_trans| +10.4%,
        per-joint rho(violation-rate drop, d|lbs|) = +0.85  -> "compensation went to placement"
  (iii) at no cost in surface fit (chamfer ratio 1.045, p = 0.089)

This file recomputes all of it from the raw .npz with an INDEPENDENT implementation, and adds
the controls probe-23 did not run:

 R1  FORWARD-MODEL VALIDATION. Every arm's stored `verts` is reconstructed from its stored
     parameters through a faithful numpy port of smal_torch.__call__ + batch_lbs. If the
     assumed template / kinematics / propagate_scaling is wrong for an arm, its whole column
     is invalid. probe-23 never checked this and the CLEAN arm is a legacy pipeline.
 R2  re-derivation of the displacement table with independent code.
 R3  ROOT-RELATIVE decomposition. With propagate_scaling=False the accumulated linear map is
     M_i = S_0^-1 S_i (diagonals commute), so ONLY log_beta_scales RELATIVE TO THE ROOT has any
     geometric effect. A corpus with a large common-mode scale has a large |lbs| that does
     nothing. |lbs| is therefore not comparable across corpora unless the common mode matches.
 R4  TEMPLATE CROSSOVER. The same lbs/bt arrays evaluated on the OTHER corpus' rest skeleton,
     to separate "the parameters are bigger" from "the template's bones are shorter".
 R5  DEPTH CONTROL on the number probe-23b did NOT depth-control: rho(violation-rate drop,
     d|lbs|) = +0.848. probe-23b showed depth kills the analogous +0.898. Also tests whether
     d|lbs| is simply proportional to baseline |lbs|.
 R6  DOSE-RESPONSE. LIM_3x (w_limit 3x) exists on disk and probe-23 ignored it. If limits push
     compensation into placement, 3x must push further than 1x.
 R7  robustness of the "local bone distortion" ratio to its small-bone denominator.
 R8  chamfer re-derivation + the hierarchical-stage data term the limit prior actually paid.

OUTPUT: refute_23_placement_recheck_out.txt
"""

import os
import pickle
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

RUNS = os.path.join(HERE, "runs")
BENCH = os.path.join(HERE, "bench50")
PREP = os.path.join(REPO, "3D_model_prep")
LIMIT_MODEL = os.path.join(PREP, "OmniAnt_25PCs_joint_limited.pkl")
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

ARMS = [
    ("LIM_0", os.path.join(RUNS, "LIM_0", "Stage_3_deform_fine.npz"), LIMIT_MODEL),
    ("LIM_1x", os.path.join(RUNS, "LIM_1x", "Stage_3_deform_fine.npz"), LIMIT_MODEL),
    ("LIM_3x", os.path.join(RUNS, "LIM_3x", "Stage_3_deform_fine.npz"), LIMIT_MODEL),
    (
        "BPX_noprior",
        os.path.join(RUNS, "BPX_noprior", "Stage_3_deform_fine.npz"),
        os.path.join(PREP, "SMIL_OmniAnt.pkl"),
    ),
    ("ALL_ANTS_CLEAN", CLEAN_NPZ, os.path.join(PREP, "SMPL_fit.pkl")),
]


def load_pkl(p):
    with open(p, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


# --------------------------------------------------------------------------- forward model
def rodrigues(theta):
    """Port of smal_model.batch_lbs.batch_rodrigues, including its `theta + 1e-8` guard."""
    theta = np.asarray(theta, dtype=np.float64)
    ang = np.linalg.norm(theta + 1e-8, axis=-1, keepdims=True)
    r = theta / ang
    ang = ang[..., None]
    cos, sin = np.cos(ang), np.sin(ang)
    outer = r[..., :, None] * r[..., None, :]
    z = np.zeros(r.shape[:-1])
    H = np.stack(
        [
            np.stack([z, -r[..., 2], r[..., 1]], -1),
            np.stack([r[..., 2], z, -r[..., 0]], -1),
            np.stack([-r[..., 1], r[..., 0], z], -1),
        ],
        -2,
    )
    return cos * np.eye(3) + (1 - cos) * outer + sin * H


def fk(J, parents, Rs=None, log_scales=None, trans=None, propagate_scaling=False):
    """Port of batch_global_rigid_transformation. Returns (t, M) with t=(B,J,3) absolute joint
    positions and M=(B,J,3,3) the accumulated linear part (the 3x3 block of `results`)."""
    B, NJ, _ = J.shape
    S = np.ones((B, NJ, 3)) if log_scales is None else np.exp(log_scales)
    T = np.zeros((B, NJ, 3)) if trans is None else np.asarray(trans) * np.array([1.0, -1.0, 1.0])
    R = np.broadcast_to(np.eye(3), (B, NJ, 3, 3)) if Rs is None else Rs
    M = np.zeros((B, NJ, 3, 3))
    t = np.zeros((B, NJ, 3))
    M[:, 0] = R[:, 0]
    t[:, 0] = J[:, 0]
    idx = np.arange(3)
    for i in range(1, NJ):
        p = parents[i]
        j_here = J[:, i] - J[:, p] + T[:, i]
        s_par_inv = np.zeros((B, 3, 3))
        if propagate_scaling:
            s_par_inv[:, idx, idx] = 1.0
        else:
            s_par_inv[:, idx, idx] = 1.0 / S[:, p]
        s_i = np.zeros((B, 3, 3))
        s_i[:, idx, idx] = S[:, i]
        rot_new = s_par_inv @ R[:, i] @ s_i
        t[:, i] = t[:, p] + np.einsum("bij,bj->bi", M[:, p], j_here)
        M[:, i] = M[:, p] @ rot_new
    return t, M


def rest_joints(dd, betas):
    vt = np.asarray(dd["v_template"], dtype=np.float64)
    sd = np.asarray(dd["shapedirs"], dtype=np.float64)
    K = betas.shape[1]
    assert sd.shape[2] >= K, f"shapedirs has {sd.shape[2]} comps but betas has {K}"
    v_shaped = vt[None] + np.einsum("bk,vck->bvc", betas[:, :K], sd[:, :, :K])
    Jr = np.asarray(dd["J_regressor"], dtype=np.float64)
    assert Jr.shape == (55, vt.shape[0]), f"J_regressor {Jr.shape}"
    return np.einsum("jv,bvc->bjc", Jr, v_shaped), v_shaped


def skin(dd, v_shaped, J, t, M):
    """verts = sum_j w_j (M_j v + t_j - M_j J_j).  Port of the A-matrix + blend in smal_torch."""
    W = np.asarray(dd["weights"], dtype=np.float64)
    t_rel = t - np.einsum("bjik,bjk->bji", M, J)
    out = np.empty_like(v_shaped)
    B = v_shaped.shape[0]
    for b in range(B):  # keep memory flat
        Mv = np.einsum("vj,jik->vik", W, M[b])
        tv = W @ t_rel[b]
        out[b] = np.einsum("vik,vk->vi", Mv, v_shaped[b]) + tv
    return out


def full_forward(dd, d, propagate_scaling=False):
    betas = np.asarray(d["betas"], dtype=np.float64)
    lbs = np.asarray(d["log_beta_scales"], dtype=np.float64)
    bt = np.asarray(d["betas_trans"], dtype=np.float64) if "betas_trans" in d.files else None
    theta = np.concatenate(
        [np.asarray(d["global_rot"], dtype=np.float64)[:, None, :], np.asarray(d["joint_rot"], dtype=np.float64)],
        axis=1,
    )
    parents = np.asarray(dd["kintree_table"])[0].astype(int)
    J, v_shaped = rest_joints(dd, betas)
    Rs = rodrigues(theta.reshape(-1, 3)).reshape(theta.shape[0], 55, 3, 3)
    t, M = fk(J, parents, Rs=Rs, log_scales=lbs, trans=bt, propagate_scaling=propagate_scaling)
    v = skin(dd, v_shaped, J, t, M)
    v = v + np.asarray(d["trans"], dtype=np.float64)[:, None, :]
    v = v + np.asarray(d["deform_verts"], dtype=np.float64)
    return v, J, v_shaped


def limit_arrays(dd_lim):
    jl = np.asarray(dd_lim["joint_limits"], dtype=np.float64)
    lo, hi = jl[:, :, 0], jl[:, :, 1]
    free = np.isclose(np.abs(lo), np.pi) & np.isclose(np.abs(hi), np.pi)
    return lo, hi, free


def partial_rho(x, y, z):
    """Spearman partial correlation of x,y controlling z (rank residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    dd_lim = load_pkl(LIMIT_MODEL)
    _names = list(dd_lim["J_names"])
    parents = np.asarray(dd_lim["kintree_table"])[0].astype(int)
    lo, hi, free = limit_arrays(dd_lim)
    constrained_ax = ~free
    constrained_ax[0] = False
    constrained = constrained_ax.any(1)
    depth = np.zeros(55, int)
    for j in range(1, 55):
        depth[j] = depth[parents[j]] + 1

    # ----------------------------------------------------------------- R0 template audit
    print("=" * 100)
    print("R0 -- template audit (are the three corpora comparable at all?)")
    print("=" * 100)
    tmpl = {}
    for nm in ("OmniAnt_25PCs_joint_limited.pkl", "SMIL_OmniAnt.pkl", "SMPL_fit.pkl"):
        tmpl[nm] = load_pkl(os.path.join(PREP, nm))
    ref = tmpl["OmniAnt_25PCs_joint_limited.pkl"]
    for nm, dd in tmpl.items():
        v = np.asarray(dd["v_template"], float)
        Jr = np.asarray(dd["J_regressor"], float)
        Jrest = Jr @ v
        Jref = np.asarray(ref["J_regressor"], float) @ np.asarray(ref["v_template"], float)
        body = np.linalg.norm(v.max(0) - v.min(0))
        same_names = list(dd["J_names"]) == list(ref["J_names"])
        same_kin = np.array_equal(np.asarray(dd["kintree_table"]), np.asarray(ref["kintree_table"]))
        bl = np.linalg.norm(Jrest[1:] - Jrest[parents[1:]], axis=1)
        blr = np.linalg.norm(Jref[1:] - Jref[parents[1:]], axis=1)
        print(
            f"  {nm:<38} V={v.shape[0]:>6} K={np.asarray(dd['shapedirs']).shape[2]:>3} "
            f"body={body:.4f}  J_names==ref {same_names}  kintree==ref {same_kin}"
        )
        print(
            f"      rest-J max |diff| vs ref = {np.abs(Jrest - Jref).max():.4f} "
            f"({100 * np.abs(Jrest - Jref).max() / body:.2f}% of body)   "
            f"mean bone length {bl.mean():.4f} vs ref {blr.mean():.4f} "
            f"(ratio {bl.mean() / blr.mean():.3f})"
        )
        print(
            f"      rest bone-direction angle vs ref: mean "
            f"{np.degrees(np.nanmean(np.arccos(np.clip(np.einsum('ij,ij->i', (Jrest[1:] - Jrest[parents[1:]]) / np.maximum(bl, 1e-9)[:, None], (Jref[1:] - Jref[parents[1:]]) / np.maximum(blr, 1e-9)[:, None]), -1, 1)))):.2f} deg"
            f"   max {np.degrees(np.nanmax(np.arccos(np.clip(np.einsum('ij,ij->i', (Jrest[1:] - Jrest[parents[1:]]) / np.maximum(bl, 1e-9)[:, None], (Jref[1:] - Jref[parents[1:]]) / np.maximum(blr, 1e-9)[:, None]), -1, 1)))):.2f} deg"
        )
        print(
            f"      scaledirs present: {'scaledirs' in dd}   transdirs: {'transdirs' in dd}   "
            f"joint_limits: {'joint_limits' in dd}"
        )

    # ----------------------------------------------------------------- load arms
    S = {}
    for arm, npz, mp in ARMS:
        if not os.path.exists(npz):
            print(f"\n  [{arm}] MISSING {npz}")
            continue
        d = np.load(npz, allow_pickle=True)
        dd = load_pkl(mp)
        S[arm] = dict(d=d, dd=dd, path=npz, model=os.path.basename(mp))

    # ----------------------------------------------------------------- R1 forward check
    print("\n" + "=" * 100)
    print("R1 -- FORWARD-MODEL VALIDATION: does the assumed kinematics reproduce the stored verts?")
    print("=" * 100)
    print("  reconstruct stored `verts` from stored params. If an arm does not reconstruct, every")
    print("  derived number for that arm (displacement, mesh-level effect) is measured with the")
    print("  wrong forward model.  err = mean per-vertex ||recon - stored|| / body bbox diagonal.\n")
    print(f"  {'arm':<16} {'template':<34} {'prop=False':>12} {'prop=True':>12}")
    print("  " + "-" * 78)
    for arm in S:
        d, dd = S[arm]["d"], S[arm]["dd"]
        stored = np.asarray(d["verts"], dtype=np.float64)
        res = {}
        for ps in (False, True):
            v, J, v_shaped = full_forward(dd, d, propagate_scaling=ps)
            body = np.linalg.norm(v_shaped.max(1) - v_shaped.min(1), axis=1)
            res[ps] = 100 * (np.linalg.norm(v - stored, axis=2).mean(1) / body).mean()
            if not ps:
                S[arm].update(J=J, v_shaped=v_shaped, body=body)
        print(f"  {arm:<16} {S[arm]['model']:<34} {res[False]:>11.4f}% {res[True]:>11.4f}%")

    # ----------------------------------------------------------------- R2 displacement
    print("\n" + "=" * 100)
    print("R2 -- re-derivation of the joint-displacement table (independent implementation)")
    print("=" * 100)
    print(
        f"  {'arm':<16} {'n':>4} {'|lbs|':>9} {'|bt|':>9} {'cumdisp%':>10} {'lbs-only%':>10} "
        f"{'bt-only%':>9} {'bonedist%mean':>14} {'bonedist%med':>13}"
    )
    print("  " + "-" * 100)
    for arm in S:
        d, dd = S[arm]["d"], S[arm]["dd"]
        lbs = np.asarray(d["log_beta_scales"], dtype=np.float64)
        bt = np.asarray(d["betas_trans"], dtype=np.float64) if "betas_trans" in d.files else np.zeros_like(lbs)
        J, body = S[arm]["J"], S[arm]["body"]
        Jb, _ = fk(J, parents)
        assert np.abs(Jb - J).max() < 1e-9
        Jts, _ = fk(J, parents, log_scales=lbs, trans=bt)
        Js, _ = fk(J, parents, log_scales=lbs)
        Jt, _ = fk(J, parents, trans=bt)
        cum = 100 * np.linalg.norm(Jts - Jb, axis=2) / body[:, None]
        cums = 100 * np.linalg.norm(Js - Jb, axis=2) / body[:, None]
        cumt = 100 * np.linalg.norm(Jt - Jb, axis=2) / body[:, None]
        bl = np.linalg.norm(J - J[:, parents.clip(0)], axis=2)
        bl[:, 0] = np.nan
        bl[bl < 1e-3 * body[:, None]] = np.nan
        vec_new = Jts - Jts[:, parents.clip(0)]
        vec_old = Jb - Jb[:, parents.clip(0)]
        rel = np.linalg.norm(vec_new - vec_old, axis=2) / bl
        S[arm].update(lbs=lbs, bt=bt, cum=cum, rel=rel)
        print(
            f"  {arm:<16} {lbs.shape[0]:>4} {np.linalg.norm(lbs, axis=2).mean():>9.5f} "
            f"{np.linalg.norm(bt, axis=2).mean():>9.5f} {cum.mean():>10.3f} {cums.mean():>10.3f} "
            f"{cumt.mean():>9.3f} {100 * np.nanmean(rel[:, 1:]):>14.2f} "
            f"{100 * np.nanmedian(rel[:, 1:]):>13.2f}"
        )

    # ----------------------------------------------------------------- R3 root-relative lbs
    print("\n" + "=" * 100)
    print("R3 -- ROOT-RELATIVE decomposition of log_beta_scales")
    print("=" * 100)
    print("  With propagate_scaling=False the accumulated map is M_i = S_root^-1 . S_i, so only")
    print("  (lbs_i - lbs_root) has ANY geometric effect. A common-mode scale inflates |lbs| for free.")
    print(
        f"\n  {'arm':<16} {'|lbs| raw':>11} {'|lbs - lbs_root|':>18} {'|lbs_root|':>12} "
        f"{'|mean over joints|':>19} {'sd across joints':>18}"
    )
    print("  " + "-" * 100)
    for arm in S:
        lbs = S[arm]["lbs"]
        rel = lbs - lbs[:, 0:1, :]
        print(
            f"  {arm:<16} {np.linalg.norm(lbs, axis=2).mean():>11.5f} "
            f"{np.linalg.norm(rel, axis=2).mean():>18.5f} "
            f"{np.linalg.norm(lbs[:, 0], axis=1).mean():>12.5f} "
            f"{np.abs(lbs.mean(1)).mean():>19.5f} {lbs.std(1).mean():>18.5f}"
        )

    # ----------------------------------------------------------------- R4 template crossover
    print("\n" + "=" * 100)
    print("R4 -- TEMPLATE CROSSOVER: same parameters, other corpus' rest skeleton")
    print("=" * 100)
    print("  cumulative displacement is a function of (rest J, lbs, bt). Swapping only the rest J")
    print("  isolates 'the parameters are bigger' from 'the template's bones are arranged differently'.\n")
    skels = {}
    for nm in ("OmniAnt_25PCs_joint_limited.pkl", "SMIL_OmniAnt.pkl", "SMPL_fit.pkl"):
        dd = tmpl[nm]
        v = np.asarray(dd["v_template"], float)
        skels[nm] = (np.asarray(dd["J_regressor"], float) @ v, np.linalg.norm(v.max(0) - v.min(0)))
    print(f"  {'params from':<16} {'rest skeleton':<38} {'cumdisp%':>10}")
    print("  " + "-" * 68)
    for arm in ("LIM_0", "LIM_1x", "ALL_ANTS_CLEAN"):
        if arm not in S:
            continue
        lbs, bt = S[arm]["lbs"], S[arm]["bt"]
        for nm, (Jm, bodym) in skels.items():
            Jrep = np.repeat(Jm[None], lbs.shape[0], 0)
            Jb, _ = fk(Jrep, parents)
            Jts, _ = fk(Jrep, parents, log_scales=lbs, trans=bt)
            c = 100 * np.linalg.norm(Jts - Jb, axis=2).mean() / bodym
            print(f"  {arm:<16} {nm:<38} {c:>10.3f}")

    # ----------------------------------------------------------------- R5 depth control
    print("\n" + "=" * 100)
    print("R5 -- DEPTH CONTROL on rho(violation-rate DROP, d|log_beta_scales|) = +0.848")
    print("=" * 100)

    def viol(arm):
        d = S[arm]["d"]
        full = np.concatenate([np.asarray(d["global_rot"], float)[:, None, :], np.asarray(d["joint_rot"], float)], 1)
        over = np.maximum(full - hi[None], 0) + np.maximum(lo[None] - full, 0)
        over[:, ~constrained_ax] = 0.0
        return (over > 0).any(2), np.degrees(over).sum(2)

    cj = np.where(constrained)[0]
    v0, _ = viol("LIM_0")
    for other in ("LIM_1x", "LIM_3x"):
        if other not in S:
            continue
        v1, _ = viol(other)
        drop = v0.mean(0) - v1.mean(0)
        m0 = np.linalg.norm(S["LIM_0"]["lbs"], axis=2).mean(0)
        m1 = np.linalg.norm(S[other]["lbs"], axis=2).mean(0)
        b0 = np.linalg.norm(S["LIM_0"]["bt"], axis=2).mean(0)
        b1 = np.linalg.norm(S[other]["bt"], axis=2).mean(0)
        dl, db = m1 - m0, b1 - b0
        print(f"\n  ### LIM_0 -> {other}")
        print(
            f"    rho(drop, d|lbs|)                        = {spearmanr(drop[cj], dl[cj])[0]:+.3f}"
            f"   <- probe-23's surviving evidence"
        )
        print(f"    rho(drop, depth)                         = {spearmanr(drop[cj], depth[cj])[0]:+.3f}")
        print(f"    rho(d|lbs|, depth)                       = {spearmanr(dl[cj], depth[cj])[0]:+.3f}")
        print(f"    PARTIAL rho(drop, d|lbs| | depth)        = {partial_rho(drop[cj], dl[cj], depth[cj]):+.3f}")
        print(f"    rho(drop, d|bt|)                         = {spearmanr(drop[cj], db[cj])[0]:+.3f}")
        print(f"    PARTIAL rho(drop, d|bt| | depth)         = {partial_rho(drop[cj], db[cj], depth[cj]):+.3f}")
        print(
            f"    rho(d|lbs|, BASELINE |lbs| in LIM_0)     = {spearmanr(dl[cj], m0[cj])[0]:+.3f}"
            f"   (is the growth just multiplicative?)"
        )
        print(f"    PARTIAL rho(drop, d|lbs| | baseline|lbs|)= {partial_rho(drop[cj], dl[cj], m0[cj]):+.3f}")
        # non-constrained joints: they cannot violate, so any growth there is NOT compensation
        nj = np.where(~constrained)[0]
        nj = nj[nj != 0]
        print(
            f"    |lbs| growth on CONSTRAINED joints  n={len(cj):>2}: "
            f"{m0[cj].mean():.4f} -> {m1[cj].mean():.4f} ({100 * (m1[cj].mean() / m0[cj].mean() - 1):+.1f}%)"
        )
        print(
            f"    |lbs| growth on UNCONSTRAINABLE     n={len(nj):>2}: "
            f"{m0[nj].mean():.4f} -> {m1[nj].mean():.4f} ({100 * (m1[nj].mean() / m0[nj].mean() - 1):+.1f}%)"
            f"   <- these joints have NO limit to escape"
        )

    # ----------------------------------------------------------------- R6 dose-response
    print("\n" + "=" * 100)
    print("R6 -- DOSE-RESPONSE across w_limit (hier 0 / 0.273 / 0.819)")
    print("=" * 100)
    print(
        f"  {'arm':<10} {'viol ax':>9} {'|lbs|':>9} {'|bt|':>9} {'|betas|':>9} {'|dv|':>9} "
        f"{'cumdisp%':>10} {'bonedist%':>11} {'meshdisp lbs+bt%':>17}"
    )
    print("  " + "-" * 100)
    for arm in ("LIM_0", "LIM_1x", "LIM_3x"):
        if arm not in S:
            continue
        d, dd = S[arm]["d"], S[arm]["dd"]
        vj, _ = viol(arm)
        lbs, bt = S[arm]["lbs"], S[arm]["bt"]
        J, v_shaped, body = S[arm]["J"], S[arm]["v_shaped"], S[arm]["body"]
        t0, M0 = fk(J, parents)
        t1, M1 = fk(J, parents, log_scales=lbs, trans=bt)
        v0 = skin(dd, v_shaped, J, t0, M0)
        v1 = skin(dd, v_shaped, J, t1, M1)
        mesh = 100 * np.linalg.norm(v1 - v0, axis=2).mean() / body.mean()
        over_ax = np.zeros(lbs.shape[0])
        full = np.concatenate([np.asarray(d["global_rot"], float)[:, None, :], np.asarray(d["joint_rot"], float)], 1)
        ov = np.maximum(full - hi[None], 0) + np.maximum(lo[None] - full, 0)
        ov[:, ~constrained_ax] = 0.0
        over_ax = (ov[:, constrained_ax] > 0).sum(1)
        print(
            f"  {arm:<10} {over_ax.mean():>9.2f} {np.linalg.norm(lbs, axis=2).mean():>9.5f} "
            f"{np.linalg.norm(bt, axis=2).mean():>9.5f} "
            f"{np.abs(np.asarray(d['betas'], float)).mean():>9.5f} "
            f"{np.linalg.norm(np.asarray(d['deform_verts'], float), axis=2).mean():>9.5f} "
            f"{S[arm]['cum'].mean():>10.3f} {100 * np.nanmean(S[arm]['rel'][:, 1:]):>11.2f} "
            f"{mesh:>17.2f}"
        )
        S[arm]["mesh"] = mesh

    # paired tests LIM_0 vs LIM_1x vs LIM_3x
    print("\n  paired wilcoxon vs LIM_0 (same 50 specimens, same order):")
    lab0 = np.asarray(S["LIM_0"]["d"]["labels"])
    for other in ("LIM_1x", "LIM_3x"):
        if other not in S:
            continue
        assert np.array_equal(lab0, np.asarray(S[other]["d"]["labels"]))
        for key, f in (
            ("|lbs|", lambda a: np.linalg.norm(S[a]["lbs"], axis=2).mean(1)),
            ("|bt|", lambda a: np.linalg.norm(S[a]["bt"], axis=2).mean(1)),
            ("cumdisp%", lambda a: S[a]["cum"].mean(1)),
            ("bonedist", lambda a: np.nanmean(S[a]["rel"][:, 1:], axis=1)),
        ):
            a, b = f("LIM_0"), f(other)
            print(
                f"    {other:<7} {key:<10} {a.mean():>10.5f} -> {b.mean():>10.5f} "
                f"({100 * (b.mean() / a.mean() - 1):+6.1f}%)  p={wilcoxon(a, b).pvalue:.3g}"
            )

    # ----------------------------------------------------------------- R7 bone ratio
    print("\n" + "=" * 100)
    print("R7 -- is 'local bone distortion' driven by short bones in the denominator?")
    print("=" * 100)
    for arm in ("LIM_0", "BPX_noprior", "ALL_ANTS_CLEAN"):
        if arm not in S:
            continue
        J, body = S[arm]["J"], S[arm]["body"]
        bl = np.linalg.norm(J - J[:, parents.clip(0)], axis=2)[:, 1:] / body[:, None]
        r = S[arm]["rel"][:, 1:]
        ok = np.isfinite(r)
        print(
            f"  {arm:<16} bone len (% body): mean {100 * bl.mean():.2f} p5 {100 * np.percentile(bl, 5):.2f} "
            f"min {100 * bl.min():.3f}   n_bones_used {int(ok.sum() / r.shape[0])}/{r.shape[1]}"
        )
        # unnormalised distortion, in % of BODY rather than % of bone
        vec_new = None
        Jts, _ = fk(J, parents, log_scales=S[arm]["lbs"], trans=S[arm]["bt"])
        Jb, _ = fk(J, parents)
        loc = np.linalg.norm((Jts - Jts[:, parents.clip(0)]) - (Jb - Jb[:, parents.clip(0)]), axis=2)[:, 1:]
        print(
            f"                   distortion: %bone mean {100 * np.nanmean(r):.2f} median "
            f"{100 * np.nanmedian(r):.2f}    %BODY mean {100 * (loc / body[:, None]).mean():.3f}"
        )

    # ----------------------------------------------------------------- R8 chamfer
    print("\n" + "=" * 100)
    print("R8 -- surface-fit cost re-derivation")
    print("=" * 100)
    from scipy.spatial import cKDTree

    def read_obj(path):
        vs = []
        with open(path) as f:
            for line in f:
                if line.startswith("v "):
                    vs.append([float(x) for x in line.split()[1:4]])
        v = np.asarray(vs, float)
        v = v - v.mean(0)
        return v / np.abs(v).max(0).max()

    res = {}
    for arm in ("LIM_0", "LIM_1x", "LIM_3x"):
        if arm not in S:
            continue
        X = S[arm]["d"]
        vals = []
        for i, lab in enumerate(np.asarray(X["labels"])):
            tp = os.path.join(BENCH, str(lab))
            if not os.path.exists(tp):
                continue
            tv = read_obj(tp)
            fv = np.asarray(X["verts"], float)[i]
            d1 = cKDTree(tv).query(fv)[0]
            d2 = cKDTree(fv).query(tv)[0]
            vals.append(0.5 * ((d1**2).mean() + (d2**2).mean()))
        res[arm] = np.asarray(vals)
        print(f"  {arm:<8} n={len(vals):>3} mean sq-chamfer {res[arm].mean():.6e} median {np.median(res[arm]):.6e}")
    for other in ("LIM_1x", "LIM_3x"):
        if other in res:
            a, b = res["LIM_0"], res[other]
            print(
                f"  {other}/LIM_0 = {b.mean() / a.mean():.4f}  paired wilcoxon p={wilcoxon(a, b).pvalue:.3g}  "
                f"{int((b < a).sum())}/{len(a)} specimens fit better under limits"
            )
    print("\n  NOTE the deform stages are preceded by a hierarchical stage where the limit term is")
    print("  47x heavier. From the run logs, final H1_legs chamfer: LIM_0 0.00581, LIM_1x 0.00626")
    print("  (+7.7%). The deform stages then bury that with free per-vertex offsets.")


if __name__ == "__main__":
    main()
