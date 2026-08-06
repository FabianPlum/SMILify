"""PROBE 25 -- decompose the 61.5 vs 30.1 gap, and remove the ceiling artefact from probe 23.

Probe 23's "excess over own floor" (worker 19.1pp vs clean 20.3pp, p=0.66) can be criticised:
the worker floor is already 42.4%, so only 57.6pp of headroom remains, against 90.2pp for
clean. A compressed excess could be a ceiling artefact rather than equality.

The ceiling-free statistic is CONDITIONAL:
    among the joints where the surface-consistent reference J_ref = J_reg . posed_verts is
    INSIDE the scan -- i.e. where "inside" is demonstrably reachable for that joint, that
    specimen, that scan -- what fraction of the FITTED joints are outside?
That is a like-for-like per-joint failure rate with the corpus-specific floor conditioned out.

Also reported:
  * the reverse conditional (J_ref outside, J_fit inside) -- how often the fit beats its floor
  * the additive decomposition of the 31.4pp corpus gap
  * the same conditional on the genus-matched subset
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from joint_placement_common import joint_group

ROOT = "/home/fabi/dev/SMILify"
store = np.load(os.path.join(ROOT, "diagnostics/moonshot/joint_placement_altcause.npy"), allow_pickle=True).item()


def genus(label):
    b = os.path.splitext(str(label))[0]
    b = b.split("_CASENT")[0].split("_OKENT")[0]
    return b.replace("_", "-").split("-")[0].split(" ")[0].lower()


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_decomp_out.txt")
    fout = open(outp, "w")
    fh = Tee(sys.stdout, fout)
    print(__doc__, file=fh)

    W = store["LIM_0 (worker)"]["rows"]
    B = store["baseline (worker, stock)"]["rows"]
    C = store["ALL_ANTS_CLEAN (81)"]["rows"]
    names = store["LIM_0 (worker)"]["names"]
    groups = np.array([joint_group(n) for n in names])

    def cond(rows):
        """per-specimen: P(fit outside | ref inside), P(fit inside | ref outside)."""
        a, b, na, nb = [], [], [], []
        for r in rows:
            fi = r["out_fit"]
            re = r["out_ref"]
            m = ~re
            a.append(fi[m].mean() if m.any() else np.nan)
            na.append(m.sum())
            m2 = re
            b.append((~fi[m2]).mean() if m2.any() else np.nan)
            nb.append(m2.sum())
        return np.array(a), np.array(b), np.array(na), np.array(nb)

    print("--- CEILING-FREE CONDITIONAL  P(fitted joint OUTSIDE | reference joint INSIDE)", file=fh)
    res = {}
    for lbl, rows in (("worker LIM_0", W), ("worker baseline", B), ("clean", C)):
        a, b, na, nb = cond(rows)
        res[lbl] = (a, b)
        print(
            f"   {lbl:<16} P(out|ref in) = {100 * np.nanmean(a):5.1f}%  "
            f"(median {100 * np.nanmedian(a):5.1f}%, n joints/spec={np.mean(na):.1f})    "
            f"P(in|ref out) = {100 * np.nanmean(b):5.1f}%  (n={np.mean(nb):.1f})",
            file=fh,
        )
    aw, ac = res["worker LIM_0"][0], res["clean"][0]
    u, p = stats.mannwhitneyu(aw[~np.isnan(aw)], ac[~np.isnan(ac)])
    print(
        f"   worker vs clean: {100 * np.nanmean(aw):.1f}% vs {100 * np.nanmean(ac):.1f}%  "
        f"ratio={np.nanmean(aw) / np.nanmean(ac):.2f}  U={u:.0f}  p={p:.3g}",
        file=fh,
    )
    ab = res["worker baseline"][0]
    u2, p2 = stats.mannwhitneyu(ab[~np.isnan(ab)], ac[~np.isnan(ac)])
    print(
        f"   baseline vs clean: {100 * np.nanmean(ab):.1f}% vs {100 * np.nanmean(ac):.1f}%  "
        f"ratio={np.nanmean(ab) / np.nanmean(ac):.2f}  p={p2:.3g}",
        file=fh,
    )

    print("\n--- per group, P(fit outside | ref inside)", file=fh)
    print(f"   {'group':<14}{'worker':>9}{'clean':>9}{'ratio':>8}", file=fh)
    for g in sorted(set(groups)):
        m = groups == g

        def cg(rows):
            v = []
            for r in rows:
                sel = m & (~r["out_ref"])
                if sel.any():
                    v.append(r["out_fit"][sel].mean())
            return np.mean(v) if v else np.nan

        a, b = cg(W), cg(C)
        print(f"   {g:<14}{100 * a:9.1f}{100 * b:9.1f}{a / b:8.2f}", file=fh)

    print("\n--- ADDITIVE DECOMPOSITION of the corpus gap in %joints-outside-scan", file=fh)
    fw = np.mean([r["out_fit"].mean() for r in W]) * 100
    fc = np.mean([r["out_fit"].mean() for r in C]) * 100
    rw = np.mean([r["out_ref"].mean() for r in W]) * 100
    rc = np.mean([r["out_ref"].mean() for r in C]) * 100
    print(f"   observed gap                                  {fw:5.1f} - {fc:5.1f} = {fw - fc:5.1f} pp", file=fh)
    print(
        f"   carried by the surface-consistent FLOOR       {rw:5.1f} - {rc:5.1f} = {rw - rc:5.1f} pp"
        f"   ({100 * (rw - rc) / (fw - fc):.0f}% of the gap)",
        file=fh,
    )
    print(
        f"   left for joint PLACEMENT (excess over floor)  "
        f"{fw - rw:5.1f} - {fc - rc:5.1f} = {(fw - rw) - (fc - rc):5.1f} pp"
        f"   ({100 * ((fw - rw) - (fc - rc)) / (fw - fc):.0f}% of the gap)",
        file=fh,
    )
    print("   of the FLOOR component, probe 24 attributes:", file=fh)
    print("     articulation alone, zero placement error       19.0 - 11.1 =   7.9 pp", file=fh)
    print("     J_ref vs its OWN fitted mesh                   15.1 -  9.4 =   5.7 pp", file=fh)
    print(
        "     fitted surface vs scan, locally at the joints  42.4 - 15.1 vs 9.8 - 9.4"
        f" = {(42.4 - 15.1) - (9.8 - 9.4):5.1f} pp",
        file=fh,
    )

    print(
        "\n--- gap of the REFERENCE skeleton to the scan (how far the fitted SURFACE sits from the scan at the joints)",
        file=fh,
    )
    for lbl, rows in (("worker LIM_0", W), ("worker baseline", B), ("clean", C)):
        gr = np.concatenate([r["gap_ref"] for r in rows])
        gf = np.concatenate([r["gap_fit"] for r in rows])
        ch = np.array([r["chamf"] for r in rows])
        print(
            f"   {lbl:<16} median gap_ref={100 * np.median(gr):.3f}%diag  "
            f"median gap_fit={100 * np.median(gf):.3f}%diag  "
            f"global chamfer={100 * np.median(ch):.3f}%   "
            f"gap_ref/chamfer={np.median(gr) / np.median(ch):.2f}",
            file=fh,
        )

    print("\n--- genus-matched, ceiling-free conditional", file=fh)
    gw, gc = {}, {}
    for r in W:
        gw.setdefault(genus(r["label"]), []).append(r)
    for r in C:
        gc.setdefault(genus(r["label"]), []).append(r)
    shared = sorted(set(gw) & set(gc))
    aa, bb = [], []
    for g in shared:
        a = np.nanmean(cond(gw[g])[0])
        b = np.nanmean(cond(gc[g])[0])
        aa.append(a)
        bb.append(b)
        print(f"   {g:<18} worker={100 * a:5.1f}%  clean={100 * b:5.1f}%", file=fh)
    w, pw = stats.wilcoxon(aa, bb)
    print(
        f"   paired by genus (n={len(aa)}): worker {100 * np.mean(aa):.1f}% vs "
        f"clean {100 * np.mean(bb):.1f}%  wilcoxon p={pw:.3g}",
        file=fh,
    )

    fh.flush()
    fout.close()
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
