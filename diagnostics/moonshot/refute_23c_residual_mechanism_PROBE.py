"""REFUTE-23c -- what is left of the worker/clean posed-stretch gap after betas_trans, and
what mechanism produces it?

refute_23 / refute_23b established:
  * probe-23's published statistic D_pose reproduces exactly (0.303/0.313/0.369/0.338 vs 0.185)
  * setting betas_trans = 0 collapses the worker/clean gap to 0.98-1.09x under the published
    (body-bone) normaliser, but only to 1.23-1.36x (all-bone) / 1.45-1.57x (unnormalised)

C1  FULL 2x2x3 GRID: {betas_trans on/off} x {root scale s_0 on/off} x {3 normalisers}, so the
    reader can see the gap under every combination rather than a hand-picked one.
C2  IS THE ROOT TERM ANISOTROPIC (a shear-like distortion) OR ISOTROPIC (a plain global size
    factor that any size-normaliser is supposed to remove)?  Split exp(-lbs[0]) into its
    geometric-mean (isotropic) part and its residual (anisotropic) part and report both.
C3  Is |betas_trans| itself predictive of the two failures it is being blamed for
    (rotation violations, probe-19 error)?  LIM_1x has the LARGEST betas_trans of the four
    non-degenerate worker arms and ~0.8 violations/specimen, which is already a hint.
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
    rodrigues,
    spearman,
)
from refute_23b_betastrans_isolation_PROBE import loo_generalisation  # noqa: E402


def main():
    lim = load_pkl(LIMITED)
    names = [str(x) for x in lim["J_names"]]
    parents = np.asarray(lim["kintree_table"])[0].astype(int)
    JL = np.asarray(lim["joint_limits"], float)
    free = np.isclose(np.abs(JL), np.pi, atol=1e-3).all(-1)
    Nj = len(names)
    bones = [i for i in range(1, Nj) if names[i] != "b_h"]
    bgrp = np.array([group_of(names[i]) for i in bones])
    leg = np.isin(bgrp, ["leg_prox", "leg_dist"])
    body = bgrp == "body"

    print("=" * 100)
    print("REFUTE-23c -- residual mechanism after betas_trans")
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
        L0 = np.linalg.norm(Jt[bones] - Jt[parents[bones]], axis=-1)
        Z = np.zeros_like(bt)
        lbs0 = lbs.copy()
        lbs0[:, 0, :] = 0.0  # kill the root term only
        L = {}
        for bt_tag, BT in [("bt", bt), ("nobt", Z)]:
            for rt_tag, LB in [("root", lbs), ("noroot", lbs0)]:
                nJ, _ = fk(J, parents, Rs, LB, BT, propagate=False)
                L[(bt_tag, rt_tag)] = np.linalg.norm(nJ[:, bones] - nJ[:, parents[bones]], axis=-1)
        jr = np.asarray(d["joint_rot"], float)
        lo, hi = JL[1:, :, 0], JL[1:, :, 1]
        con = ~free[1:]
        va = np.where(con[None], np.maximum(np.clip(lo[None] - jr, 0, None), np.clip(jr - hi[None], 0, None)), 0.0)
        ST[run] = dict(
            kind=kind,
            L0=L0,
            L=L,
            lbs=lbs,
            bt=bt,
            n=n,
            n_viol=(va > 1e-6).sum(axis=(1, 2)).astype(float),
            X=v_skel + np.asarray(d["deform_verts"], float),
        )

    def D(st, key, mode):
        r = st["L"][key] / st["L0"][None]
        if mode == "body":
            r = r / np.median(r[:, body], axis=1)[:, None]
        elif mode == "all":
            r = r / np.median(r, axis=1)[:, None]
        return np.abs(np.log(r[:, leg])).mean(1)

    print("\n" + "=" * 100)
    print("C1  FULL GRID -- median over specimens of mean|log(L/L0)| over the 36 leg bones")
    print("=" * 100)
    combos = [("bt", "root"), ("nobt", "root"), ("bt", "noroot"), ("nobt", "noroot")]
    hdr = f"  {'run':<20}{'kind':<7}"
    for c in combos:
        hdr += f"{c[0] + '/' + c[1]:>15}"
    for mode in ["body", "all", "unnorm"]:
        print(f"\n  normaliser = {mode}   (published probe-23 statistic is body / bt / root)")
        print(hdr)
        for run, st in ST.items():
            row = f"  {run:<20}{st['kind']:<7}"
            for c in combos:
                row += f"{np.median(D(st, c, mode)):15.4f}"
            print(row)
        cl = ST["ALL_ANTS_CLEAN"]
        print(f"  {'--> worker/clean gap':<27}")
        for run, st in ST.items():
            if st["kind"] != "worker":
                continue
            row = f"  {run:<27}"
            for c in combos:
                row += f"{np.median(D(st, c, mode)) / np.median(D(cl, c, mode)):14.2f}x"
            print(row)

    print("\n" + "=" * 100)
    print("C2  ROOT TERM: isotropic global size factor vs anisotropic distortion")
    print("    s0inv = exp(-log_beta_scales[:,0,:]).  iso = geometric mean over the 3 axes")
    print("    (a plain whole-animal size factor); aniso = max/min of s0inv/iso.")
    print("=" * 100)
    print(f"  {'run':<20}{'kind':<7}{'iso p05':>10}{'iso p50':>10}{'iso p95':>10}{'aniso p50':>12}{'aniso p95':>12}")
    for run, st in ST.items():
        s0 = np.exp(-st["lbs"][:, 0, :])
        iso = np.exp(np.log(s0).mean(1))
        an = s0 / iso[:, None]
        rng = an.max(1) / an.min(1)
        print(
            f"  {run:<20}{st['kind']:<7}{np.percentile(iso, 5):10.3f}{np.median(iso):10.3f}"
            f"{np.percentile(iso, 95):10.3f}{np.median(rng):12.3f}{np.percentile(rng, 95):12.3f}"
        )
    print("\n  |log iso| (the pure global size term, in the same units as every D above):")
    for run, st in ST.items():
        iso = np.exp(np.log(np.exp(-st["lbs"][:, 0, :])).mean(1))
        print(f"    {run:<20} median |log iso| = {np.median(np.abs(np.log(iso))):.4f}")

    print("\n" + "=" * 100)
    print("C3  Is betas_trans magnitude itself predictive of the failures it is blamed for?")
    print("=" * 100)
    print(f"  {'run':<20}{'med |bt| leg joints':>21}{'rho vs n_viol':>15}{'p':>9}{'rho vs probe19':>16}{'p':>9}")
    legj = [i for i in range(1, Nj) if group_of(names[i]) in ("leg_prox", "leg_dist")]
    for run, st in ST.items():
        b = np.linalg.norm(st["bt"][:, legj], axis=-1).mean(1)
        if b.std() == 0:
            print(f"  {run:<20}{b.mean():21.5f}{'n/a (all zero)':>15}")
            continue
        g19 = loo_generalisation(st["X"], min(10, st["n"] - 2))
        r1, p1, _ = spearman(b, st["n_viol"])
        r2, p2, _ = spearman(b, g19)
        print(f"  {run:<20}{np.median(b):21.5f}{r1:15.3f}{p1:9.4f}{r2:16.3f}{p2:9.4f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
