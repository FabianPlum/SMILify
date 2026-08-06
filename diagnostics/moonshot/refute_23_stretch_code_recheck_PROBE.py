"""REFUTE PROBE for probe-23's headline:

  "Worker limb segments in the POSED skeleton are stretched ~2x more than clean ones
   (median per-specimen mean|log(fitted/template)| 0.30-0.37 vs 0.185) and that stretch
   tracks rotation violations."

This probe re-derives every number from the raw .npz + .pkl, then attacks the measurement
itself.  Everything below is CPU numpy on stored arrays.

CHECKS
  R0  TEMPLATE FACTS.  Vertex counts, shapedirs widths, J_regressor density, J_names /
      kintree identity, and -- the one that matters -- the per-bone REST LENGTH L0 of each
      of the three templates.  probe-23's whole metric is L/L0 with a DIFFERENT L0 for the
      worker corpora (OmniAnt) and the clean corpus (SMPL_fit).
  R1  REPRODUCTION.  Recompute probe-23's D_pose / D_rest medians to confirm I am measuring
      the same quantity before attacking it.
  R2  FORWARD-CONVENTION TEST FOR THE CLEAN CORPUS.  probe-23b's own output says the clean
      rebuild is a MISMATCH (rel 6.9e-04, plus a 3.9e-03 rigid offset) while all five worker
      rebuilds are EXACT.  The clean column of the headline therefore rests on an
      UNVALIDATED forward model.  Try the 2x2 of conventions
      (deform_verts posed-space vs rest-space) x (propagate_scaling False vs True)
      and report which one actually reproduces the stored clean verts.  If the winner is
      not the one probe-23 assumed, the clean L1/L4 are wrong.
  R3  betas_trans CONTROL, done properly.  probe-23 concedes betas_trans is absent from the
      clean corpus but claims "excluding betas_trans the conclusion survives".  That claim
      was made from STEPWISE ratios (L3/L2, L4/L3) which do not compose.  Recompute the
      posed skeleton with betas_trans = 0 and compare D_pose worker vs clean directly.
  R4  ROOT-ANISOTROPY CONTROL.  With propagate_scaling=False the accumulated map is
      M_p = R_0 . s_0^-1 . R_1..R_p . s_p, so EVERY bone is multiplied by the same
      anisotropic diagonal s_0^-1.  That is a GLOBAL AFFINE transform of the whole animal,
      not a joint-placement error.  probe-23's normaliser divides by a SCALAR (median body
      bone ratio) and cannot remove it.  Measure the anisotropy of s_0 and recompute D_pose
      with s_0 removed, and with the best per-specimen diagonal removed.
  R5  TEMPLATE-FREE DISPERSION.  "Stretched implausibly" should mean the bone is unusual for
      the population, not that it differs from an arbitrary template.  Recompute every
      statistic against the CORPUS's own median bone length instead of L0, and split
      probe-23's D_pose into a SYSTEMATIC part (per-bone corpus median, i.e. template
      mismatch, identical for every specimen and therefore incapable of explaining
      per-specimen correspondence failure) and an IDIOSYNCRATIC part.
  R6  SHORT-BONE ARTIFACT.  betas_trans is an ABSOLUTE offset added to j_here.  For a bone
      whose template length is comparable to |betas_trans|, the RATIO explodes.  Report
      L0 per bone against |betas_trans|, and re-run the headline with an absolute
      (corpus-scale-normalised) length error instead of a log ratio.
"""

import itertools
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

LIMITED = os.path.join(REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl")
OMNI = os.path.join(REPO, "3D_model_prep", "SMIL_OmniAnt.pkl")
SMPLFIT = os.path.join(REPO, "3D_model_prep", "SMPL_fit.pkl")
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)
RUNS = os.path.join(HERE, "runs")

CORPORA = [
    ("LIM_0", os.path.join(RUNS, "LIM_0", "Stage_3_deform_fine.npz"), LIMITED, "worker"),
    ("LIM_1x", os.path.join(RUNS, "LIM_1x", "Stage_3_deform_fine.npz"), LIMITED, "worker"),
    ("BPX_noprior", os.path.join(RUNS, "BPX_noprior", "Stage_3_deform_fine.npz"), OMNI, "worker"),
    ("M7_handoff_midline", os.path.join(RUNS, "M7_handoff_midline", "Stage_3_deform_fine.npz"), OMNI, "worker"),
    ("baseline", os.path.join(RUNS, "baseline", "Stage_3_deform_fine.npz"), OMNI, "worker"),
    ("ALL_ANTS_CLEAN", CLEAN_NPZ, SMPLFIT, "clean"),
]
GROUP_ORDER = ["body", "mandible", "antenna", "leg_prox", "leg_dist", "wing"]


def load_pkl(p):
    with open(p, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def group_of(name):
    if name.startswith("b_"):
        return "body"
    if name.startswith("ma_"):
        return "mandible"
    if name.startswith("an_"):
        return "antenna"
    if name.startswith("w_"):
        return "wing"
    if name.startswith("l_"):
        return "leg_prox" if name.split("_")[2] in ("co", "tr", "fe") else "leg_dist"
    return "other"


def rodrigues(rv):
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    r = rv / np.maximum(theta, 1e-12)
    ct, st = np.cos(theta)[..., None], np.sin(theta)[..., None]
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    K = np.zeros((rv.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -z, y
    K[:, 1, 0], K[:, 1, 2] = z, -x
    K[:, 2, 0], K[:, 2, 1] = -y, x
    outer = r[:, :, None] * r[:, None, :]
    return ct * np.eye(3)[None] + st * K + (1 - ct) * outer


def fk(J, parents, Rs, logscale, trans_off, propagate=False):
    """batch_global_rigid_transformation port. Returns (newJ, res 4x4)."""
    n, Nj = J.shape[0], J.shape[1]
    S = np.exp(logscale)
    toff = trans_off * np.array([1.0, -1.0, 1.0])
    res = np.zeros((n, Nj, 4, 4))
    res[:, :, 3, 3] = 1.0
    res[:, 0, :3, :3] = Rs[:, 0]
    res[:, 0, :3, 3] = J[:, 0]
    for i in range(1, Nj):
        p = parents[i]
        A = np.zeros((n, 4, 4))
        A[:, 3, 3] = 1.0
        if propagate:
            A[:, :3, :3] = Rs[:, i] * S[:, i][:, None, :]
        else:
            A[:, :3, :3] = (1.0 / S[:, p])[:, :, None] * Rs[:, i] * S[:, i][:, None, :]
        A[:, :3, 3] = (J[:, i] - J[:, p]) + toff[:, i]
        res[:, i] = np.einsum("nab,nbc->nac", res[:, p], A)
    return res[:, :, :3, 3], res


def skin(dd, v_shaped, J, parents, Rs, lbs, bt, trans, dv_posed, dv_rest_used, propagate):
    """Full LBS.  dv_rest_used means deform_verts was already folded into v_shaped/J."""
    W = np.asarray(dd["weights"], dtype=np.float64)
    n, Nj = J.shape[0], J.shape[1]
    _, res = fk(J, parents, Rs, lbs, bt, propagate=propagate)
    Jw0 = np.concatenate([J, np.zeros((n, Nj, 1))], -1)
    init_bone = np.einsum("njab,njb->nja", res, Jw0)
    Arel = res.copy()
    Arel[:, :, :, 3] -= init_bone
    T = np.einsum("vj,njab->nvab", W, Arel)
    vh = np.concatenate([v_shaped, np.ones((n, v_shaped.shape[1], 1))], -1)
    out = np.einsum("nvab,nvb->nva", T, vh)[:, :, :3] + trans[:, None, :]
    if not dv_rest_used:
        out = out + dv_posed
    return out


def pct(a, q):
    return float(np.percentile(a, q))


def spearman(x, y):
    from math import lgamma

    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), n

    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(n, float)
        r[o] = np.arange(1, n + 1)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(cnt.size)
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0, n
    t = rho * np.sqrt((n - 2) / (1 - rho**2))
    df = n - 2

    def betacf(a, b, xx):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * xx / qap
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        h = d
        for m_ in range(1, 300):
            m2 = 2 * m_
            aa = m_ * (b - m_) * xx / ((qam + m2) * (a + m2))
            d = 1.0 / max(abs(1.0 + aa * d), 1e-30) * np.sign(1.0 + aa * d)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            h *= d * c
            aa = -(a + m_) * (qab + m_) * xx / ((a + m2) * (qap + m2))
            d = 1.0 / max(abs(1.0 + aa * d), 1e-30) * np.sign(1.0 + aa * d)
            c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
            de = d * c
            h *= de
            if abs(de - 1.0) < 3e-16:
                break
        return h

    def betai(a, b, xx):
        if xx <= 0:
            return 0.0
        if xx >= 1:
            return 1.0
        bt_ = np.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * np.log(xx) + b * np.log(1 - xx))
        if xx < (a + 1) / (a + b + 2):
            return bt_ * betacf(a, b, xx) / a
        return 1.0 - bt_ * betacf(b, a, 1 - xx) / b

    return rho, float(betai(0.5 * df, 0.5, df / (df + t * t))), n


# =====================================================================================
def main():
    lim = load_pkl(LIMITED)
    names = [str(x) for x in lim["J_names"]]
    parents = np.asarray(lim["kintree_table"])[0].astype(int)
    JL = np.asarray(lim["joint_limits"], dtype=np.float64)
    free = np.isclose(np.abs(JL), np.pi, atol=1e-3).all(-1)
    Nj = len(names)
    bones = [i for i in range(1, Nj) if names[i] != "b_h"]
    bgrp = np.array([group_of(names[i]) for i in bones])
    leg_mask = np.isin(bgrp, ["leg_prox", "leg_dist"])
    bnames = [names[i] for i in bones]

    print("=" * 104)
    print("REFUTE-23 -- independent re-derivation of the 'worker limb segments are stretched 2x' claim")
    print("=" * 104)

    # ------------------------------------------------------------------ R0 template facts
    print("\n" + "=" * 104)
    print("R0  TEMPLATE FACTS -- the two corpora are measured against DIFFERENT L0")
    print("=" * 104)
    tmpls = {"LIMITED(OmniAnt25)": LIMITED, "OMNI(SMIL_OmniAnt)": OMNI, "SMPLFIT(clean)": SMPLFIT}
    dds = {k: load_pkl(v) for k, v in tmpls.items()}
    for k, dd in dds.items():
        Jr = dd["J_regressor"]
        print(
            f"  {k:22s} V={dd['v_template'].shape[0]:6d}  shapedirs K={dd['shapedirs'].shape[-1]:3d}  "
            f"J_regressor type={type(Jr).__name__} shape={np.asarray(Jr).shape} "
            f"nnz/row={np.count_nonzero(np.asarray(Jr, float)) / 55:.1f}  "
            f"posedirs={np.asarray(dd['posedirs']).shape}  scaledirs={'scaledirs' in dd}"
        )
    for a, b in itertools.combinations(tmpls, 2):
        na = [str(x) for x in dds[a]["J_names"]]
        nb = [str(x) for x in dds[b]["J_names"]]
        print(
            f"  {a:22s} vs {b:22s}  J_names identical={na == nb}  "
            f"kintree identical={np.array_equal(np.asarray(dds[a]['kintree_table']), np.asarray(dds[b]['kintree_table']))}"
        )

    L0s = {}
    for k, dd in dds.items():
        Jt = np.asarray(dd["J_regressor"], float) @ np.asarray(dd["v_template"], float)
        L0s[k] = np.linalg.norm(Jt[bones] - Jt[parents[bones]], axis=-1)
    print("\n  per-bone TEMPLATE rest length L0, and the cross-template ratio that probe-23's")
    print("  worker-vs-clean comparison silently divides by:")
    print(
        f"  {'bone':<14}{'group':<10}{'L0 OmniAnt25':>14}{'L0 SMIL_Omni':>14}{'L0 SMPL_fit':>13}{'Omni25/SMPLfit':>16}"
    )
    order = np.argsort(L0s["SMPLFIT(clean)"])
    for idx in order:
        print(
            f"  {bnames[idx]:<14}{bgrp[idx]:<10}{L0s['LIMITED(OmniAnt25)'][idx]:14.5f}"
            f"{L0s['OMNI(SMIL_OmniAnt)'][idx]:14.5f}{L0s['SMPLFIT(clean)'][idx]:13.5f}"
            f"{L0s['LIMITED(OmniAnt25)'][idx] / L0s['SMPLFIT(clean)'][idx]:16.3f}"
        )
    for g in GROUP_ORDER:
        m = bgrp == g
        r = L0s["LIMITED(OmniAnt25)"][m] / L0s["SMPLFIT(clean)"][m]
        r2 = L0s["OMNI(SMIL_OmniAnt)"][m] / L0s["SMPLFIT(clean)"][m]
        print(
            f"  GROUP {g:<10} Omni25/SMPLfit  p05={pct(r, 5):.3f} p50={pct(r, 50):.3f} p95={pct(r, 95):.3f} "
            f"mean|log|={np.abs(np.log(r)).mean():.4f}   |  SMILOmni/SMPLfit mean|log|={np.abs(np.log(r2)).mean():.4f}"
        )

    # ------------------------------------------------------------------ load corpora
    print("\n" + "=" * 104)
    print("R2  FORWARD-CONVENTION TEST -- which forward model actually reproduces stored `verts`?")
    print("    variants: dv=posed  (verts = skin(vt+sd.b) + trans + dv)   [what probe-23 assumed]")
    print("              dv=rest   (verts = skin(vt+sd.b+dv) + trans)")
    print("              x propagate_scaling False / True")
    print("    residual is reported AFTER removing a per-specimen rigid offset (lengths are")
    print("    translation invariant); 'scale' is the mean bbox diagonal.")
    print("=" * 104)
    print(
        f"  {'run':<20}{'dv=posed,prop=F':>18}{'dv=rest,prop=F':>18}{'dv=posed,prop=T':>18}{'dv=rest,prop=T':>18}   winner"
    )

    ST = {}
    for run, npz_path, tmpl_path, kind in CORPORA:
        if not os.path.exists(npz_path):
            print(f"  !! MISSING {run}")
            continue
        dd = load_pkl(tmpl_path)
        d = np.load(npz_path, allow_pickle=True)
        vt = np.asarray(dd["v_template"], float)
        sd = np.asarray(dd["shapedirs"], float)
        Jr = np.asarray(dd["J_regressor"], float)
        betas = np.asarray(d["betas"], float)
        n, K = betas.shape
        assert K == sd.shape[-1], f"{run}: betas K={K} but shapedirs K={sd.shape[-1]}"
        dv = np.asarray(d["deform_verts"], float)
        lbs = np.asarray(d["log_beta_scales"], float)
        bt = np.asarray(d["betas_trans"], float) if "betas_trans" in d.files else np.zeros_like(lbs)
        trans = np.asarray(d["trans"], float)
        pose = np.concatenate([np.asarray(d["global_rot"], float)[:, None, :], np.asarray(d["joint_rot"], float)], 1)
        Rs = rodrigues(pose.reshape(-1, 3)).reshape(n, Nj, 3, 3)
        vs = np.asarray(d["verts"], float)
        scale = float(np.linalg.norm(vs.max(1) - vs.min(1), axis=-1).mean())

        v_noD = vt[None] + np.einsum("bk,vck->bvc", betas, sd)
        v_wD = v_noD + dv
        J_noD = np.einsum("jv,bvc->bjc", Jr, v_noD)
        J_wD = np.einsum("jv,bvc->bjc", Jr, v_wD)

        errs = {}
        for tag, (vsh, Jj, dvrest) in {
            "dv=posed": (v_noD, J_noD, False),
            "dv=rest": (v_wD, J_wD, True),
        }.items():
            for prop in (False, True):
                vr = skin(dd, vsh, Jj, parents, Rs, lbs, bt, trans, dv, dvrest, prop)
                resid = vr - vs
                resid = resid - resid.mean(1, keepdims=True)
                errs[(tag, prop)] = float(np.abs(resid).max() / scale)
        best = min(errs, key=errs.get)
        print(
            f"  {run:<20}{errs[('dv=posed', False)]:18.2e}{errs[('dv=rest', False)]:18.2e}"
            f"{errs[('dv=posed', True)]:18.2e}{errs[('dv=rest', True)]:18.2e}   "
            f"{best[0]},prop={best[1]}"
        )

        newJ, _ = fk(J_noD, parents, Rs, lbs, bt, propagate=False)
        newJ_nobt, _ = fk(J_noD, parents, Rs, lbs, np.zeros_like(bt), propagate=False)
        Jt = Jr @ vt
        ST[run] = dict(
            kind=kind,
            n=n,
            K=K,
            dd=dd,
            d=d,
            lbs=lbs,
            bt=bt,
            Rs=Rs,
            trans=trans,
            dv=dv,
            J_t=Jt,
            J_f=J_noD,
            J_fD=J_wD,
            newJ=newJ,
            newJ_nobt=newJ_nobt,
            v_noD=v_noD,
            v_wD=v_wD,
            has_bt=("betas_trans" in d.files),
            errs=errs,
            scale=scale,
        )

    # ------------------------------------------------------------------ length definitions
    for run, st in ST.items():
        Jt, Jf, nJ, nJb = st["J_t"], st["J_f"], st["newJ"], st["newJ_nobt"]
        st["L0"] = np.linalg.norm(Jt[bones] - Jt[parents[bones]], axis=-1)
        st["L1"] = np.linalg.norm(Jf[:, bones] - Jf[:, parents[bones]], axis=-1)
        st["L4"] = np.linalg.norm(nJ[:, bones] - nJ[:, parents[bones]], axis=-1)
        st["L4nb"] = np.linalg.norm(nJb[:, bones] - nJb[:, parents[bones]], axis=-1)
        # L3: parent scale + betas_trans, root anisotropy REMOVED (rotations are length preserving)
        S = np.exp(st["lbs"])
        off = Jf[:, bones] - Jf[:, parents[bones]]
        toff = st["bt"] * np.array([1.0, -1.0, 1.0])
        st["L3"] = np.linalg.norm((off + toff[:, bones]) * S[:, parents[bones]], axis=-1)
        st["L3nb"] = np.linalg.norm(off * S[:, parents[bones]], axis=-1)

    body = bgrp == "body"

    def norm_ratio(L, L0):
        R = L / L0[None]
        return R / np.median(R[:, body], axis=1)[:, None]

    for run, st in ST.items():
        for k in ["L1", "L3", "L3nb", "L4", "L4nb"]:
            st["R" + k[1:]] = norm_ratio(st[k], st["L0"])

    # ------------------------------------------------------------------ R1 reproduction
    print("\n" + "=" * 104)
    print("R1  REPRODUCTION of probe-23's headline (median over specimens of mean|log ratio| over LEG bones)")
    print("=" * 104)
    print(f"  {'run':<20}{'kind':<7}{'D_rest(L1)':>12}{'D_pose(L4)':>12}{'D_dist(L4)':>12}   probe-23 published D_pose")
    pub = {"LIM_0": 0.303, "LIM_1x": 0.313, "BPX_noprior": 0.369, "M7_handoff_midline": 0.338, "ALL_ANTS_CLEAN": 0.185}
    md = bgrp == "leg_dist"
    for run, st in ST.items():
        st["D_rest"] = np.abs(np.log(st["R1"][:, leg_mask])).mean(1)
        st["D_pose"] = np.abs(np.log(st["R4"][:, leg_mask])).mean(1)
        st["D_dist"] = np.abs(np.log(st["R4"][:, md])).mean(1)
        print(
            f"  {run:<20}{st['kind']:<7}{np.median(st['D_rest']):12.4f}{np.median(st['D_pose']):12.4f}"
            f"{np.median(st['D_dist']):12.4f}   {pub.get(run, float('nan')):.3f}"
        )

    # ------------------------------------------------------------------ R3/R4 controls
    print("\n" + "=" * 104)
    print("R3/R4  CONTROLS -- strip the two mechanisms that CANNOT be joint-placement error")
    print("  D_pose      L4 : everything (parent scale + betas_trans + global root anisotropy)")
    print("  D_nobt      L4 with betas_trans = 0  (removes a parameter the clean corpus does not have)")
    print("  D_noroot    L3 : root anisotropy s_0^-1 removed (rotations preserve length, so this is exact)")
    print("  D_bare      L3 with betas_trans = 0 : parent scale + rest shape only == the ONLY")
    print("              mechanism both corpora can express")
    print("=" * 104)
    print(f"  {'run':<20}{'kind':<7}{'D_pose':>10}{'D_nobt':>10}{'D_noroot':>10}{'D_bare':>10}{'|log s0| aniso':>16}")
    for run, st in ST.items():
        st["D_nobt"] = np.abs(np.log(st["R4nb"][:, leg_mask])).mean(1)
        st["D_noroot"] = np.abs(np.log(st["R3"][:, leg_mask])).mean(1)
        st["D_bare"] = np.abs(np.log(st["R3nb"][:, leg_mask])).mean(1)
        s0 = st["lbs"][:, 0, :]
        aniso = s0.max(1) - s0.min(1)  # log-range across the 3 root axes
        st["aniso"] = aniso
        print(
            f"  {run:<20}{st['kind']:<7}{np.median(st['D_pose']):10.4f}{np.median(st['D_nobt']):10.4f}"
            f"{np.median(st['D_noroot']):10.4f}{np.median(st['D_bare']):10.4f}{np.median(aniso):16.4f}"
        )
    print("\n  worker/clean RATIO of each statistic (headline claims ~2x):")
    cl = ST.get("ALL_ANTS_CLEAN")
    if cl is not None:
        for run, st in ST.items():
            if st["kind"] != "worker":
                continue
            r = [np.median(st[k]) / np.median(cl[k]) for k in ["D_pose", "D_nobt", "D_noroot", "D_bare"]]
            print(
                f"  {run:<20} D_pose {r[0]:5.2f}x   D_nobt {r[1]:5.2f}x   D_noroot {r[2]:5.2f}x   D_bare {r[3]:5.2f}x"
            )

    print("\n  same four statistics for the DISTAL leg only (the headline's strongest claim):")
    print(f"  {'run':<20}{'D_pose':>10}{'D_nobt':>10}{'D_noroot':>10}{'D_bare':>10}")
    for run, st in ST.items():
        vals = [np.median(np.abs(np.log(st[k][:, md])).mean(1)) for k in ["R4", "R4nb", "R3", "R3nb"]]
        st["Ddist_bare"] = np.abs(np.log(st["R3nb"][:, md])).mean(1)
        print(f"  {run:<20}" + "".join(f"{v:10.4f}" for v in vals))

    print("\n  pretarsus bone (ta->pt), median ratio and mean|log ratio| under each control:")
    ptm = np.array([bn.endswith("_pt_l") or bn.endswith("_pt_r") or "_pt_" in bn for bn in bnames])
    print(
        f"  {'run':<20}{'med L4':>10}{'med L4nb':>10}{'med L3':>10}{'med L3nb':>10}{'|log| L4':>10}{'|log| L3nb':>12}"
    )
    for run, st in ST.items():
        row = [np.median(st[k][:, ptm]) for k in ["R4", "R4nb", "R3", "R3nb"]]
        row += [np.abs(np.log(st["R4"][:, ptm])).mean(), np.abs(np.log(st["R3nb"][:, ptm])).mean()]
        print(f"  {run:<20}" + "".join(f"{v:10.4f}" for v in row[:4]) + f"{row[4]:10.4f}{row[5]:12.4f}")

    # ------------------------------------------------------------------ R5 template-free
    print("\n" + "=" * 104)
    print("R5  TEMPLATE-FREE.  Replace the template L0 by the CORPUS's own per-bone median length.")
    print("    SYS  = mean over leg bones of |median_specimens log(L/L0)|  -> systematic template")
    print("           mismatch, IDENTICAL for every specimen, cannot explain per-specimen failure")
    print("    IDIO = median over specimens of mean|log(L / corpus_median_L)| -> real per-specimen")
    print("           dispersion, i.e. what 'this segment is stretched unnaturally' should mean")
    print("=" * 104)
    print(
        f"  {'run':<20}{'kind':<7}{'POSED SYS':>11}{'POSED IDIO':>12}{'BARE SYS':>10}{'BARE IDIO':>11}{'REST IDIO':>11}"
    )
    for run, st in ST.items():
        out = []
        for key in ["R4", "R3nb", "R1"]:
            R = st[key][:, leg_mask]
            lg = np.log(R)
            sysm = float(np.abs(np.median(lg, axis=0)).mean())
            idio = float(np.median(np.abs(lg - np.median(lg, axis=0, keepdims=True)).mean(1)))
            out += [sysm, idio]
        st["IDIO_pose"] = np.abs(
            np.log(st["R4"][:, leg_mask]) - np.median(np.log(st["R4"][:, leg_mask]), 0, keepdims=True)
        ).mean(1)
        st["IDIO_bare"] = np.abs(
            np.log(st["R3nb"][:, leg_mask]) - np.median(np.log(st["R3nb"][:, leg_mask]), 0, keepdims=True)
        ).mean(1)
        print(f"  {run:<20}{st['kind']:<7}{out[0]:11.4f}{out[1]:12.4f}{out[2]:10.4f}{out[3]:11.4f}{out[5]:11.4f}")

    # ------------------------------------------------------------------ R6 short-bone artifact
    print("\n" + "=" * 104)
    print("R6  SHORT-BONE ARTIFACT.  betas_trans is an ABSOLUTE offset added to j_here, so the")
    print("    log RATIO it produces scales like |betas_trans| / L0.  Rank bones by L0.")
    print("=" * 104)
    print(
        f"  {'bone':<14}{'L0(Omni25)':>12}{'L0(SMPLfit)':>12}{'med|bt| LIM_0':>15}{'|bt|/L0':>10}{'|log L4| LIM_0':>16}{'|log L4| CLEAN':>16}"
    )
    l0o, l0s = L0s["LIMITED(OmniAnt25)"], L0s["SMPLFIT(clean)"]
    o = np.argsort(l0o)
    lim0, clean = ST.get("LIM_0"), ST.get("ALL_ANTS_CLEAN")
    for idx in o:
        j = bones[idx]
        mbt = float(np.median(np.linalg.norm(lim0["bt"][:, j], axis=-1))) if lim0 is not None else np.nan
        a = float(np.abs(np.log(lim0["R4"][:, idx])).mean()) if lim0 is not None else np.nan
        b = float(np.abs(np.log(clean["R4"][:, idx])).mean()) if clean is not None else np.nan
        print(f"  {bnames[idx]:<14}{l0o[idx]:12.5f}{l0s[idx]:12.5f}{mbt:15.5f}{mbt / l0o[idx]:10.3f}{a:16.4f}{b:16.4f}")
    if lim0 is not None:
        x = np.array([np.median(np.linalg.norm(lim0["bt"][:, bones[i]], axis=-1)) / l0o[i] for i in range(len(bones))])
        y = np.array([np.abs(np.log(lim0["R4"][:, i])).mean() for i in range(len(bones))])
        r, p, nn = spearman(x, y)
        print(
            f"\n  ACROSS BONES (LIM_0): Spearman( median|betas_trans|/L0 , mean|log posed ratio| ) = {r:.3f} p={p:.4g} n={nn}"
        )
        r2, p2, _ = spearman(l0o, y)
        print(
            f"  ACROSS BONES (LIM_0): Spearman( template L0 , mean|log posed ratio| )             = {r2:.3f} p={p2:.4g}"
        )
        yc = np.array([np.abs(np.log(clean["R4"][:, i])).mean() for i in range(len(bones))])
        r3, p3, _ = spearman(l0s, yc)
        print(
            f"  ACROSS BONES (CLEAN): Spearman( template L0 , mean|log posed ratio| )             = {r3:.3f} p={p3:.4g}"
        )

    print("\n  ABSOLUTE length error instead of log ratio (mm-equivalent, divided by the specimen's")
    print("  own bbox diagonal so the two corpora are on the same scale):")
    print(f"  {'run':<20}{'kind':<7}{'legs |dL|/bbox':>16}{'distal |dL|/bbox':>18}{'bare legs |dL|/bbox':>21}")
    for run, st in ST.items():
        vs = np.asarray(st["d"]["verts"], float)
        bb = np.linalg.norm(vs.max(1) - vs.min(1), axis=-1)
        size4 = np.median(st["L4"][:, body] / st["L0"][body], axis=1)
        size3 = np.median(st["L3nb"][:, body] / st["L0"][body], axis=1)
        e4 = np.abs(st["L4"] / size4[:, None] - st["L0"][None]) / bb[:, None]
        e3 = np.abs(st["L3nb"] / size3[:, None] - st["L0"][None]) / bb[:, None]
        print(
            f"  {run:<20}{st['kind']:<7}{np.median(e4[:, leg_mask].mean(1)):16.5f}"
            f"{np.median(e4[:, md].mean(1)):18.5f}{np.median(e3[:, leg_mask].mean(1)):21.5f}"
        )

    # ------------------------------------------------------------------ violations
    print("\n" + "=" * 104)
    print("R7  Does the stretch still track rotation violations once the non-placement mechanisms")
    print("    are removed?  Spearman per specimen vs n_violations.")
    print("=" * 104)
    lo, hi = JL[1:, :, 0], JL[1:, :, 1]
    con = ~free[1:]
    print(f"  {'run':<20}{'D_pose':>16}{'D_nobt':>16}{'D_bare':>16}{'IDIO_bare':>16}")
    for run, st in ST.items():
        jr = np.asarray(st["d"]["joint_rot"], float)
        va = np.where(con[None], np.maximum(np.clip(lo[None] - jr, 0, None), np.clip(jr - hi[None], 0, None)), 0.0)
        nv = (va > 1e-6).sum(axis=(1, 2)).astype(float)
        st["n_viol"] = nv
        cells = []
        for k in ["D_pose", "D_nobt", "D_bare", "IDIO_bare"]:
            r, p, _ = spearman(st[k], nv)
            cells.append(f"{r:7.3f}(p{p:6.4f})")
        print(f"  {run:<20}" + "".join(f"{c:>16}" for c in cells))

    print("\ndone.")


if __name__ == "__main__":
    main()
