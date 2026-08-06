"""REFUTE-23b -- isolate betas_trans as the whole of probe-23's worker/clean posed gap.

refute_23_stretch_code_recheck_PROBE.py already showed:
    D_pose (probe-23's statistic)      worker 0.303-0.369  vs clean 0.185   -> 1.64-1.99x
    D_nobt (same, betas_trans = 0)     worker 0.181-0.202  vs clean 0.185   -> 0.98-1.09x

This probe finishes the job:
  B1  probe-23's CAVEAT claims "excluding betas_trans the conclusion survives but shrinks",
      citing the STEPWISE ratios L3/L2 = 0.172-0.180 worker vs 0.129 clean and
      L4/L3 = 0.150-0.159 vs 0.069.  Those steps are not comparable: for a worker the
      parent-scale step acts on a vector ALREADY displaced by betas_trans, and the root step
      likewise.  Recompute each mechanism's step on the SAME starting vector for both corpora.
  B2  Tail statistics that the headline quotes (p95 4.37, "% bones > 1.25x", pretarsus 1.97-2.01)
      recomputed with betas_trans = 0.
  B3  All three normalisers probe-23c used (body-normalised / all-bone / unnormalised),
      with and without betas_trans, so the "normaliser robustness" caveat is answered.
  B4  Template provenance: is each corpus's own template centred on that corpus?
  B5  SMIL_OmniAnt template left/right asymmetry -- the pretarsus headline is measured
      against it.
  B6  D_pose vs probe-19 LOO error, raw, with and without betas_trans.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from refute_23_stretch_code_recheck_PROBE import (  # noqa: E402
    CORPORA,
    fk,
    group_of,
    load_pkl,
    LIMITED,
    OMNI,
    rodrigues,
    spearman,
)


def loo_generalisation(X, k):
    N = X.shape[0]
    Xf = X.reshape(N, -1).astype(np.float64)
    err = np.empty(N)
    for i in range(N):
        keep = np.ones(N, bool)
        keep[i] = False
        Y = Xf[keep]
        mu = Y.mean(0)
        _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        B = Vt[:k]
        err[i] = np.linalg.norm((r - (r @ B.T) @ B).reshape(-1, 3), axis=-1).mean()
    return err


def main():
    lim = load_pkl(LIMITED)
    names = [str(x) for x in lim["J_names"]]
    parents = np.asarray(lim["kintree_table"])[0].astype(int)
    Nj = len(names)
    bones = [i for i in range(1, Nj) if names[i] != "b_h"]
    bgrp = np.array([group_of(names[i]) for i in bones])
    bnames = [names[i] for i in bones]
    leg = np.isin(bgrp, ["leg_prox", "leg_dist"])
    dist = bgrp == "leg_dist"
    body = bgrp == "body"
    ptm = np.array(["_pt_" in b for b in bnames])

    print("=" * 100)
    print("REFUTE-23b -- betas_trans isolation")
    print("=" * 100)

    ST = {}
    for run, npz, tmpl, kind in CORPORA:
        if not os.path.exists(npz):
            continue
        dd = load_pkl(tmpl)
        d = np.load(npz, allow_pickle=True)
        vt = np.asarray(dd["v_template"], float)
        sd = np.asarray(dd["shapedirs"], float)
        Jr = np.asarray(dd["J_regressor"], float)
        betas = np.asarray(d["betas"], float)
        n = betas.shape[0]
        lbs = np.asarray(d["log_beta_scales"], float)
        bt = np.asarray(d["betas_trans"], float) if "betas_trans" in d.files else np.zeros_like(lbs)
        pose = np.concatenate([np.asarray(d["global_rot"], float)[:, None, :], np.asarray(d["joint_rot"], float)], 1)
        Rs = rodrigues(pose.reshape(-1, 3)).reshape(n, Nj, 3, 3)
        v_skel = vt[None] + np.einsum("bk,vck->bvc", betas, sd)
        J = np.einsum("jv,bvc->bjc", Jr, v_skel)
        Jt = Jr @ vt
        S = np.exp(lbs)
        toff = bt * np.array([1.0, -1.0, 1.0])
        off = J[:, bones] - J[:, parents[bones]]
        Z = np.zeros_like(bt)

        L0 = np.linalg.norm(Jt[bones] - Jt[parents[bones]], axis=-1)
        L1 = np.linalg.norm(off, axis=-1)
        L2 = np.linalg.norm(off + toff[:, bones], axis=-1)
        L3 = np.linalg.norm((off + toff[:, bones]) * S[:, parents[bones]], axis=-1)
        L3nb = np.linalg.norm(off * S[:, parents[bones]], axis=-1)
        nJ, _ = fk(J, parents, Rs, lbs, bt, propagate=False)
        nJb, _ = fk(J, parents, Rs, lbs, Z, propagate=False)
        L4 = np.linalg.norm(nJ[:, bones] - nJ[:, parents[bones]], axis=-1)
        L4nb = np.linalg.norm(nJb[:, bones] - nJb[:, parents[bones]], axis=-1)
        ST[run] = dict(
            kind=kind,
            n=n,
            d=d,
            dd=dd,
            v_skel=v_skel,
            dv=np.asarray(d["deform_verts"], float),
            L0=L0,
            L1=L1,
            L2=L2,
            L3=L3,
            L3nb=L3nb,
            L4=L4,
            L4nb=L4nb,
            bt=bt,
            lbs=lbs,
            betas=betas,
        )

    # ---------------------------------------------------------------- B1
    print("\n" + "=" * 100)
    print("B1  MECHANISM STEPS, computed on the SAME starting vector for both corpora (leg bones)")
    print("    probe-23 compared L3/L2 and L4/L3, which for a worker start from a betas_trans-")
    print("    displaced vector.  Here every step starts from L1 (the rest bone).")
    print("=" * 100)
    print(
        f"  {'run':<20}{'kind':<7}{'bt step':>10}{'scale step':>12}{'root step':>11}{'TOTAL(L4/L1)':>14}{'TOTAL no bt':>13}"
    )
    for run, st in ST.items():
        f = lambda a, b: float(np.abs(np.log(a[:, leg] / b[:, leg])).mean())  # noqa: E731
        print(
            f"  {run:<20}{st['kind']:<7}{f(st['L2'], st['L1']):10.4f}{f(st['L3nb'], st['L1']):12.4f}"
            f"{f(st['L4nb'], st['L3nb']):11.4f}{f(st['L4'], st['L1']):14.4f}{f(st['L4nb'], st['L1']):13.4f}"
        )
    print("\n  probe-23's caveat quoted L3/L2=0.172-0.180 (worker) vs 0.129 (clean) as the")
    print("  'parent-scale step surviving'.  The scale-step column above is the like-for-like one.")

    # ---------------------------------------------------------------- B2
    print("\n" + "=" * 100)
    print("B2  TAIL STATISTICS the headline quotes, with and without betas_trans")
    print("    (ratios size-normalised by the median body-bone ratio, exactly as probe-23)")
    print("=" * 100)

    def R(st, key):
        r = st[key] / st["L0"][None]
        return r / np.median(r[:, body], axis=1)[:, None]

    print(
        f"  {'run':<20}{'set':<12}{'p50 WITH':>10}{'p95 WITH':>10}{'>1.25 WITH':>12}{'p50 NO-bt':>11}{'p95 NO-bt':>11}{'>1.25 NO-bt':>13}"
    )
    for run, st in ST.items():
        for tag, m in [("distal leg", dist), ("all legs", leg), ("pretarsus", ptm)]:
            a = R(st, "L4")[:, m].ravel()
            b = R(st, "L4nb")[:, m].ravel()
            print(
                f"  {run:<20}{tag:<12}{np.median(a):10.3f}{np.percentile(a, 95):10.3f}"
                f"{100 * (a > 1.25).mean():11.1f}%{np.median(b):11.3f}{np.percentile(b, 95):11.3f}"
                f"{100 * (b > 1.25).mean():12.1f}%"
            )
        print("  " + "-" * 96)

    # ---------------------------------------------------------------- B3
    print("\n" + "=" * 100)
    print("B3  NORMALISER ROBUSTNESS x betas_trans  (median over specimens of mean|log| over legs)")
    print("=" * 100)
    print(
        f"  {'run':<20}{'body-norm':>11}{'all-bone':>10}{'unnorm':>9}{'| body NObt':>13}{'all NObt':>10}{'unnorm NObt':>13}"
    )
    for run, st in ST.items():
        out = []
        for key in ["L4", "L4nb"]:
            r = st[key] / st["L0"][None]
            nb = r / np.median(r[:, body], axis=1)[:, None]
            na = r / np.median(r, axis=1)[:, None]
            out += [
                float(np.median(np.abs(np.log(nb[:, leg])).mean(1))),
                float(np.median(np.abs(np.log(na[:, leg])).mean(1))),
                float(np.median(np.abs(np.log(r[:, leg])).mean(1))),
            ]
        print(f"  {run:<20}{out[0]:11.4f}{out[1]:10.4f}{out[2]:9.4f}{out[3]:13.4f}{out[4]:10.4f}{out[5]:13.4f}")
    cl = ST["ALL_ANTS_CLEAN"]

    def med(st, key, mode):
        r = st[key] / st["L0"][None]
        if mode == "body":
            r = r / np.median(r[:, body], axis=1)[:, None]
        elif mode == "all":
            r = r / np.median(r, axis=1)[:, None]
        return float(np.median(np.abs(np.log(r[:, leg])).mean(1)))

    print("\n  worker/clean GAP under each normaliser:")
    print(
        f"  {'run':<20}{'body WITH':>11}{'all WITH':>10}{'unn WITH':>10}{'| body NObt':>13}{'all NObt':>10}{'unn NObt':>12}"
    )
    for run, st in ST.items():
        if st["kind"] != "worker":
            continue
        g = [med(st, k, m) / med(cl, k, m) for k in ["L4", "L4nb"] for m in ["body", "all", "unnorm"]]
        print(f"  {run:<20}" + "".join(f"{v:10.2f}x" for v in g[:3]) + "  " + "".join(f"{v:10.2f}x" for v in g[3:]))

    # ---------------------------------------------------------------- B4
    print("\n" + "=" * 100)
    print("B4  TEMPLATE PROVENANCE -- is each corpus centred on its own template?")
    print("    off-centre = |mean_specimens(v_shaped) - v_template| / population spread")
    print("=" * 100)
    print(f"  {'run':<20}{'kind':<7}{'|mean betas|':>13}{'off-centre(rest)':>18}{'off-centre(+dv)':>17}")
    for run, st in ST.items():
        vt = np.asarray(st["dd"]["v_template"], float)
        X = st["v_skel"]
        sp = float(np.linalg.norm(X - X.mean(0, keepdims=True), axis=-1).mean())
        oc = float(np.linalg.norm(X.mean(0) - vt, axis=-1).mean() / max(sp, 1e-12))
        X2 = st["v_skel"] + st["dv"]
        sp2 = float(np.linalg.norm(X2 - X2.mean(0, keepdims=True), axis=-1).mean())
        oc2 = float(np.linalg.norm(X2.mean(0) - vt, axis=-1).mean() / max(sp2, 1e-12))
        print(f"  {run:<20}{st['kind']:<7}{np.abs(st['betas'].mean(0)).mean():13.4f}{oc:18.3f}{oc2:17.3f}")

    # ---------------------------------------------------------------- B5
    print("\n" + "=" * 100)
    print("B5  TEMPLATE LEFT/RIGHT ASYMMETRY of the pretarsus bone (the headline's vivid number)")
    print("=" * 100)
    for tag, path in [
        ("OmniAnt25 (LIM runs)", LIMITED),
        ("SMIL_OmniAnt (BPX/M7)", OMNI),
        ("SMPL_fit (clean)", CORPORA[-1][2]),
    ]:
        dd = load_pkl(path)
        Jt = np.asarray(dd["J_regressor"], float) @ np.asarray(dd["v_template"], float)
        L0 = np.linalg.norm(Jt[bones] - Jt[parents[bones]], axis=-1)
        row = {bnames[i]: L0[i] for i in range(len(bones))}
        s = "  ".join(
            f"{k}={row[k]:.4f}" for k in ["l_1_pt_l", "l_1_pt_r", "l_2_pt_l", "l_2_pt_r", "l_3_pt_l", "l_3_pt_r"]
        )
        asym = max(
            row["l_1_pt_l"] / row["l_1_pt_r"],
            row["l_1_pt_r"] / row["l_1_pt_l"],
            row["l_2_pt_l"] / row["l_2_pt_r"],
            row["l_2_pt_r"] / row["l_2_pt_l"],
            row["l_3_pt_l"] / row["l_3_pt_r"],
            row["l_3_pt_r"] / row["l_3_pt_l"],
        )
        print(f"  {tag:<24}{s}   max L/R asymmetry = {asym:.2f}x")

    # ---------------------------------------------------------------- B6
    print("\n" + "=" * 100)
    print("B6  D_pose vs probe-19 LOO error, RAW, with and without betas_trans")
    print("=" * 100)
    print(f"  {'run':<20}{'rho WITH bt':>14}{'p':>9}{'rho NO bt':>12}{'p':>9}")
    for run, st in ST.items():
        X = st["v_skel"] + st["dv"]
        g19 = loo_generalisation(X, min(10, X.shape[0] - 2))
        out = []
        for key in ["L4", "L4nb"]:
            r = st[key] / st["L0"][None]
            r = r / np.median(r[:, body], axis=1)[:, None]
            D = np.abs(np.log(r[:, leg])).mean(1)
            out.append(spearman(D, g19)[:2])
        print(f"  {run:<20}{out[0][0]:14.3f}{out[0][1]:9.4f}{out[1][0]:12.3f}{out[1][1]:9.4f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
