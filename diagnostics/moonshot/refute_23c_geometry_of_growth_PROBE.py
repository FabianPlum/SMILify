"""refute-23c: the +22.4% |log_beta_scales| is measured in LOG PARAMETER space. What does it do
in GEOMETRY space, restricted to the joints that actually carry a rotation limit?

The quantity that acts on geometry is s = exp(lbs_j - lbs_root); the bone j->child is mapped by
S_root^-1 S_j, so (s - 1) is the fractional deformation and its component along the bone is the
segment-length change (the thing that moves the child joint).

R14  paired LIM_0 vs LIM_1x / LIM_3x on CONSTRAINED joints only:
       |lbs|            log-parameter norm      (probe-23's +22.4%)
       |s - 1|          geometric magnitude
       signed along-bone fractional length change
       |along-bone|     magnitude of segment-length change
       child-bone rest-space length change in % of body
R15  where does the log-space growth go, positive (lengthen) or negative (shorten)?

OUTPUT: refute_23c_geometry_of_growth_out.txt
"""

import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from refute_23_placement_recheck_PROBE import (  # noqa: E402
    ARMS,
    PREP,
    limit_arrays,
    load_pkl,
    rest_joints,
)

LIMIT_MODEL = os.path.join(PREP, "OmniAnt_25PCs_joint_limited.pkl")


def main():
    dd_lim = load_pkl(LIMIT_MODEL)
    names = list(dd_lim["J_names"])
    parents = np.asarray(dd_lim["kintree_table"])[0].astype(int)
    lo, hi, free = limit_arrays(dd_lim)
    constrained_ax = ~free
    constrained_ax[0] = False
    constrained = constrained_ax.any(1)
    children = {j: [k for k in range(1, 55) if parents[k] == j] for j in range(55)}

    S = {}
    for arm, npz, mp in ARMS:
        if not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        dd = load_pkl(mp)
        betas = np.asarray(d["betas"], float)
        J, v_shaped = rest_joints(dd, betas)
        lbs = np.asarray(d["log_beta_scales"], float)
        bt = np.asarray(d["betas_trans"], float) if "betas_trans" in d.files else np.zeros_like(lbs)
        S[arm] = dict(J=J, body=np.linalg.norm(v_shaped.max(1) - v_shaped.min(1), axis=1), lbs=lbs, bt=bt)

    # joints that have a child AND are constrained
    jj = [j for j in range(1, 55) if constrained[j] and children[j]]
    print("=" * 100)
    print("R14 -- LOG-PARAMETER growth vs GEOMETRIC growth, on the constrained joints only")
    print("=" * 100)
    print(f"  n constrained joints with a child = {len(jj)}   n specimens = {S['LIM_0']['lbs'].shape[0]}")
    print("  s = exp(lbs_j - lbs_root); the bone j->child is mapped by that diagonal.\n")

    def metrics(arm):
        J, lbs, body = S[arm]["J"], S[arm]["lbs"], S[arm]["body"]
        rel = lbs - lbs[:, 0:1, :]
        s = np.exp(rel)
        out = {}
        out["|lbs|"] = np.linalg.norm(lbs[:, jj], axis=2).mean(1)
        out["|s-1|"] = np.linalg.norm(s[:, jj] - 1.0, axis=2).mean(1)
        along_s, along_a, dlen = [], [], []
        for j in jj:
            c = children[j][0]
            u = J[:, c] - J[:, j]
            n = np.linalg.norm(u, axis=1, keepdims=True)
            _un = u / np.maximum(n, 1e-12)
            newvec = s[:, j] * u  # diagonal acting on the bone vector
            frac = (np.linalg.norm(newvec, axis=1) - n[:, 0]) / np.maximum(n[:, 0], 1e-12)
            along_s.append(frac)
            along_a.append(np.abs(frac))
            dlen.append(np.abs(np.linalg.norm(newvec, axis=1) - n[:, 0]) / body)
        out["signed dLen/L"] = np.mean(np.stack(along_s, 1), 1)
        out["|dLen|/L"] = np.mean(np.stack(along_a, 1), 1)
        out["|dLen| %body"] = 100 * np.mean(np.stack(dlen, 1), 1)
        return out

    m = {a: metrics(a) for a in ("LIM_0", "LIM_1x", "LIM_3x") if a in S}
    keys = list(m["LIM_0"].keys())
    print(f"  {'quantity':<16} {'LIM_0':>10} {'LIM_1x':>10} {'chg':>8} {'p':>10} {'LIM_3x':>10} {'chg':>8} {'p':>10}")
    print("  " + "-" * 88)
    for k in keys:
        a = m["LIM_0"][k]
        row = f"  {k:<16} {a.mean():>10.4f}"
        for other in ("LIM_1x", "LIM_3x"):
            if other not in m:
                continue
            b = m[other][k]
            chg = 100 * (b.mean() / a.mean() - 1) if a.mean() != 0 else np.nan
            row += f" {b.mean():>10.4f} {chg:>+7.1f}% {wilcoxon(a, b).pvalue:>10.3g}"
        print(row)

    print("\n  corpus reference on the SAME joint set (unpaired, different corpus/template):")
    for arm in ("BPX_noprior", "ALL_ANTS_CLEAN"):
        if arm not in S:
            continue
        mm = metrics(arm)
        print(f"    {arm:<16} " + "  ".join(f"{k} {mm[k].mean():.4f}" for k in keys))

    print("\n" + "=" * 100)
    print("R15 -- sign of the log-space growth (does the fit LENGTHEN or SHORTEN the constrained")
    print("       segments when rotation is forbidden?)")
    print("=" * 100)
    print(f"  {'arm':<10} {'mean lbs (constr, root-rel)':>28} {'frac of axes with lbs<0':>25}")
    for arm in ("LIM_0", "LIM_1x", "LIM_3x"):
        if arm not in S:
            continue
        rel = S[arm]["lbs"] - S[arm]["lbs"][:, 0:1, :]
        r = rel[:, jj]
        print(f"  {arm:<10} {r.mean():>28.5f} {float((r < 0).mean()):>25.3f}")

    print("\n  per-joint detail, top movers by d|lbs| (LIM_1x - LIM_0):")
    d0 = np.linalg.norm(S["LIM_0"]["lbs"], axis=2).mean(0)
    d1 = np.linalg.norm(S["LIM_1x"]["lbs"], axis=2).mean(0)
    order = sorted(jj, key=lambda j: -(d1[j] - d0[j]))[:10]
    print(
        f"  {'joint':<12} {'|lbs|0':>8} {'|lbs|1':>8} {'d|lbs|%':>9} {'|s-1|0':>8} {'|s-1|1':>8} "
        f"{'d|s-1|%':>9} {'|dLen|/L 0':>11} {'|dLen|/L 1':>11} {'chg':>8}"
    )
    for j in order:
        row = [names[j]]
        vals = []
        for arm in ("LIM_0", "LIM_1x"):
            J, lbs = S[arm]["J"], S[arm]["lbs"]
            rel = lbs - lbs[:, 0:1, :]
            s = np.exp(rel)
            c = children[j][0]
            u = J[:, c] - J[:, j]
            n = np.linalg.norm(u, axis=1)
            frac = np.abs((np.linalg.norm(s[:, j] * u, axis=1) - n) / np.maximum(n, 1e-12))
            vals.append(
                (np.linalg.norm(lbs[:, j], axis=1).mean(), np.linalg.norm(s[:, j] - 1, axis=1).mean(), frac.mean())
            )
        (l0, g0, f0), (l1, g1, f1) = vals
        print(
            f"  {row[0]:<12} {l0:>8.4f} {l1:>8.4f} {100 * (l1 / l0 - 1):>+8.1f}% "
            f"{g0:>8.4f} {g1:>8.4f} {100 * (g1 / g0 - 1):>+8.1f}% "
            f"{f0:>11.4f} {f1:>11.4f} {100 * (f1 / f0 - 1):>+7.1f}%"
        )


if __name__ == "__main__":
    main()
