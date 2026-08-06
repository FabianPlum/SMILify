"""refute-23b: follow-ups the first pass opened.

R9   AXIS DECOMPOSITION. R3 found ALL_ANTS_CLEAN has a LARGER root-relative |log_beta_scales|
     (0.530) than the workers (0.453) yet 3x LESS joint displacement. With diagonal scales the
     joint only moves if the scaled axis has a component ALONG the bone. So the worker/CLEAN gap
     may be about WHICH AXIS is scaled (length vs girth), not about how hard joints are moved.
R10  DOSE-RESPONSE POWER. LIM_3x showed no further |lbs| growth over LIM_1x. Is that because
     there was no residual constraint left to compensate for? Measure the overshoot actually
     remaining at 1x.
R11  CLEAN reconstruction residual (0.1676% in R1): is it a systematic model mismatch that
     could bias the CLEAN column?
R12  Is the "+31.8% on constrained joints vs +4.4% on unconstrainable" split depth-confounded?
R13  betas dose-response p-values, and the mesh-level geometric effect per arm.

OUTPUT: refute_23b_mechanism_recheck_out.txt
"""

import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from refute_23_placement_recheck_PROBE import (  # noqa: E402
    ARMS,
    PREP,
    fk,
    full_forward,
    limit_arrays,
    load_pkl,
    rest_joints,
    skin,
)

LIMIT_MODEL = os.path.join(PREP, "OmniAnt_25PCs_joint_limited.pkl")


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
    children = {j: [k for k in range(1, 55) if parents[k] == j] for j in range(55)}

    S = {}
    for arm, npz, mp in ARMS:
        if not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        dd = load_pkl(mp)
        betas = np.asarray(d["betas"], float)
        J, v_shaped = rest_joints(dd, betas)
        S[arm] = dict(
            d=d,
            dd=dd,
            J=J,
            v_shaped=v_shaped,
            body=np.linalg.norm(v_shaped.max(1) - v_shaped.min(1), axis=1),
            lbs=np.asarray(d["log_beta_scales"], float),
            bt=np.asarray(d["betas_trans"], float) if "betas_trans" in d.files else None,
        )
        if S[arm]["bt"] is None:
            S[arm]["bt"] = np.zeros_like(S[arm]["lbs"])

    print("=" * 100)
    print("R9 -- which AXIS does each corpus scale? along-bone (changes segment length, moves the")
    print("      child joint) vs perpendicular (changes girth, moves no joint)")
    print("=" * 100)
    print("  For every joint j with a child c: unit bone direction u = (J_c - J_j)/|.| in rest space.")
    print("  s = exp(lbs_j - lbs_root) is the DIAGONAL that actually acts (M_j = S_root^-1 S_j).")
    print("  along     = |(s-1) component projected onto u|   -> length change of the child bone")
    print("  perp      = |(s-1) component orthogonal to u|    -> girth change only")
    print("  Averaged over all (specimen, parent-of-a-child joint).\n")
    print(f"  {'arm':<16} {'|s-1| total':>12} {'along-bone':>12} {'perp':>10} {'along frac':>11}")
    print("  " + "-" * 66)
    for arm in S:
        J, lbs = S[arm]["J"], S[arm]["lbs"]
        rel = lbs - lbs[:, 0:1, :]
        s = np.exp(rel) - 1.0  # (B,55,3) diagonal deviation from identity
        al, pe, tot = [], [], []
        for j in range(1, 55):
            cs = children[j]
            if not cs:
                continue
            u = J[:, cs[0]] - J[:, j]
            n = np.linalg.norm(u, axis=1, keepdims=True)
            ok = n[:, 0] > 1e-6
            if ok.sum() == 0:
                continue
            u = u / np.maximum(n, 1e-12)
            v = s[:, j]  # (B,3)  the vector (s-1) acts componentwise on the bone
            comp = (v * u).sum(1)  # scalar: how much the bone lengthens per unit length
            perp = np.linalg.norm(v * u - comp[:, None] * u, axis=1)
            al.append(np.abs(comp)[ok])
            pe.append(perp[ok])
            tot.append(np.linalg.norm(v, axis=1)[ok])
        al, pe, tot = np.concatenate(al), np.concatenate(pe), np.concatenate(tot)
        print(
            f"  {arm:<16} {tot.mean():>12.4f} {al.mean():>12.4f} {pe.mean():>10.4f} "
            f"{al.mean() / (al.mean() + pe.mean()):>11.3f}"
        )

    print("\n  -> if the along-bone column tracks the joint-displacement column, the 3x is about")
    print("     WHERE the scale points, not only about how large it is.")

    print("\n" + "=" * 100)
    print("R10 -- dose-response power: how much constraint violation was LEFT at 1x?")
    print("=" * 100)
    print(f"  {'arm':<10} {'viol axes':>10} {'total overshoot deg/spec':>26} {'max overshoot deg':>19}")
    print("  " + "-" * 70)
    for arm in ("LIM_0", "LIM_1x", "LIM_3x"):
        if arm not in S:
            continue
        d = S[arm]["d"]
        full = np.concatenate([np.asarray(d["global_rot"], float)[:, None, :], np.asarray(d["joint_rot"], float)], 1)
        ov = np.maximum(full - hi[None], 0) + np.maximum(lo[None] - full, 0)
        ov[:, ~constrained_ax] = 0.0
        print(
            f"  {arm:<10} {(ov[:, constrained_ax] > 0).sum(1).mean():>10.2f} "
            f"{np.degrees(ov).sum((1, 2)).mean():>26.3f} {np.degrees(ov).max():>19.2f}"
        )
    print("\n  -> if 1x already removed essentially all overshoot, LIM_3x adds little extra pressure")
    print("     and the flat |lbs| between 1x and 3x is weak (not zero) evidence.")

    print("\n" + "=" * 100)
    print("R11 -- ALL_ANTS_CLEAN forward-reconstruction residual: systematic or noise?")
    print("=" * 100)
    arm = "ALL_ANTS_CLEAN"
    if arm in S:
        d, dd = S[arm]["d"], S[arm]["dd"]
        v, J, v_shaped = full_forward(dd, d, propagate_scaling=False)
        stored = np.asarray(d["verts"], float)
        err = np.linalg.norm(v - stored, axis=2)
        body = S[arm]["body"]
        print(
            f"  per-vertex error / body: mean {100 * (err.mean(1) / body).mean():.4f}%  "
            f"p95 {100 * (np.percentile(err, 95, axis=1) / body).mean():.4f}%  "
            f"max {100 * (err.max(1) / body).mean():.4f}%"
        )
        # is it concentrated on the vertices whose skinning weights do not sum to 1?
        W = np.asarray(dd["weights"], float)
        bad = np.abs(W.sum(1) - 1) > 1e-6
        print(f"  vertices with weight rows != 1: {int(bad.sum())} / {len(bad)}")
        print(f"    mean err on those      : {100 * (err[:, bad].mean() / body.mean()):.4f}%")
        print(f"    mean err on the rest   : {100 * (err[:, ~bad].mean() / body.mean()):.4f}%")
        # same diagnostic on a worker arm for contrast
        for w in ("LIM_0",):
            dw, ddw = S[w]["d"], S[w]["dd"]
            vw, _, _ = full_forward(ddw, dw, propagate_scaling=False)
            ew = np.linalg.norm(vw - np.asarray(dw["verts"], float), axis=2)
            print(f"  contrast {w}: mean per-vertex err / body = {100 * (ew.mean(1) / S[w]['body']).mean():.6f}%")

    print("\n" + "=" * 100)
    print("R12 -- is 'constrained joints grew 32%, unconstrainable grew 4%' a depth artifact?")
    print("=" * 100)
    cj = np.where(constrained)[0]
    nj = np.array([j for j in np.where(~constrained)[0] if j != 0])
    print(
        f"  depth of CONSTRAINED joints    (n={len(cj)}): mean {depth[cj].mean():.2f}  "
        f"range {depth[cj].min()}-{depth[cj].max()}"
    )
    print(
        f"  depth of UNCONSTRAINABLE joints(n={len(nj)}): mean {depth[nj].mean():.2f}  "
        f"range {depth[nj].min()}-{depth[nj].max()}"
    )
    print(f"  mannwhitney p on depth = {mannwhitneyu(depth[cj], depth[nj]).pvalue:.3g}")
    m0 = np.linalg.norm(S["LIM_0"]["lbs"], axis=2).mean(0)
    for other in ("LIM_1x", "LIM_3x"):
        if other not in S:
            continue
        m1 = np.linalg.norm(S[other]["lbs"], axis=2).mean(0)
        print(f"\n  {other}: |lbs| growth by depth level (all 55 joints, constrained vs not)")
        print(f"    {'depth':>5} {'n_constr':>9} {'growth_constr':>14} {'n_uncon':>8} {'growth_uncon':>13}")
        for dl in sorted(set(depth[1:])):
            a = [j for j in cj if depth[j] == dl]
            b = [j for j in nj if depth[j] == dl]
            ga = f"{100 * (m1[a].mean() / m0[a].mean() - 1):+.1f}%" if a else "-"
            gb = f"{100 * (m1[b].mean() / m0[b].mean() - 1):+.1f}%" if b else "-"
            print(f"    {dl:>5} {len(a):>9} {ga:>14} {len(b):>8} {gb:>13}")

    print("\n" + "=" * 100)
    print("R13 -- what actually moves monotonically with the limit dose?")
    print("=" * 100)
    _lab0 = np.asarray(S["LIM_0"]["d"]["labels"])
    rows = []
    for arm in ("LIM_0", "LIM_1x", "LIM_3x"):
        if arm not in S:
            continue
        d, dd = S[arm]["d"], S[arm]["dd"]
        J, v_shaped, body = S[arm]["J"], S[arm]["v_shaped"], S[arm]["body"]
        t0, M0 = fk(J, parents)
        t1, M1 = fk(J, parents, log_scales=S[arm]["lbs"], trans=S[arm]["bt"])
        v0 = skin(dd, v_shaped, J, t0, M0)
        v1 = skin(dd, v_shaped, J, t1, M1)
        mesh = 100 * (np.linalg.norm(v1 - v0, axis=2).mean(1) / body)
        rows.append(
            (arm, np.abs(np.asarray(d["betas"], float)).mean(1), np.linalg.norm(S[arm]["lbs"], axis=2).mean(1), mesh)
        )
    print(f"  {'arm':<10} {'|betas|':>10} {'|lbs|':>10} {'mesh disp by lbs+bt (% body)':>32}")
    for arm, bmean, lmean, mesh in rows:
        print(f"  {arm:<10} {bmean.mean():>10.5f} {lmean.mean():>10.5f} {mesh.mean():>32.4f}")
    base = rows[0]
    for arm, bmean, lmean, mesh in rows[1:]:
        print(
            f"  vs LIM_0  {arm:<8} |betas| p={wilcoxon(base[1], bmean).pvalue:.3g} "
            f"({100 * (bmean.mean() / base[1].mean() - 1):+.1f}%)   "
            f"mesh-disp p={wilcoxon(base[3], mesh).pvalue:.3g} "
            f"({100 * (mesh.mean() / base[3].mean() - 1):+.1f}%)"
        )
    print("\n  -> |lbs| jumps 22% at 1x then STOPS. The geometric quantity it is supposed to be a")
    print("     proxy for (mesh displacement caused by lbs+bt) does not move at all.")


if __name__ == "__main__":
    main()
