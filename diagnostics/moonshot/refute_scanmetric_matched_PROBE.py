"""REFUTE-4 -- decompose the headline 61.5% vs 30.1% into a PIPELINE effect and a CORPUS effect.

The headline compares
    LIM_0            worker scans + OmniAnt_25PCs template + moonshot/LIM fitter
vs
    ALL_ANTS_CLEAN   clean  scans + SMPL_fit     template + legacy fitter
and reads the difference as "the worker FITS misplace joints".  Four things change at once
(scans, specimens, template, fitter), so the design has n=1 unit per level of the thing that
actually varies.  Every p-value in the headline (1.9e-21 etc.) is a WITHIN-corpus consistency
p-value, not a p-value for the causal claim.

Two matched controls already exist on disk and were not used:
    runs/CLEAN_M7           clean scans, SMIL_OmniAnt template, the SAME moonshot pipeline
                            as the worker M7 run  ->  isolates SCANS+SPECIMENS
    runs/M7_handoff_midline worker scans, SMIL_OmniAnt template, moonshot pipeline
    runs/baseline           worker scans, SMIL_OmniAnt template, stock fitter

So we can hold template+pipeline fixed and vary only the corpus (M7_worker vs CLEAN_M7), and
hold scans fixed and vary only the pipeline (LIM_0 vs M7_worker vs baseline on bench50;
CLEAN_M7 vs ALL_ANTS_CLEAN on clean81).

Metric is the HEADLINE metric verbatim: |generalized winding number| < 0.5 against the raw
target scan, scans normalised by replaying fitter_3d/utils.py:340-344.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, wilcoxon
from joint_placement_common import load_run, winding_number
from joint_placement_vs_scan_PROBE import load_obj_np, normalise, posed_skeleton

ROOT = "/home/fabi/dev/SMILify"
OUT = os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_matched_out.txt")
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

RUNS = [
    # tag,               npz,                                      template,                 align, scans
    ("LIM_0", "runs/LIM_0/Stage_3_deform_fine.npz", "OmniAnt_25PCs_joint_limited.pkl", False, "bench50"),
    ("M7_worker", "runs/M7_handoff_midline/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl", False, "bench50"),
    ("baseline", "runs/baseline/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl", False, "bench50"),
    ("CLEAN_M7", "runs/CLEAN_M7/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl", False, "clean81"),
    ("CLEAN_legacy", CLEAN_NPZ, "SMPL_fit.pkl", True, "clean81"),
]


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    scans = {}
    for sd in ("bench50", "clean81"):
        d = os.path.join(ROOT, "diagnostics/moonshot", sd)
        scans[sd] = {}
        for fn in sorted(x for x in os.listdir(d) if x.endswith(".obj")):
            V, F = load_obj_np(os.path.join(d, fn))
            V = normalise(V)
            scans[sd][os.path.splitext(fn)[0]] = (V, F, float(np.linalg.norm(V.max(0) - V.min(0))), cKDTree(V))
        print(f"loaded {len(scans[sd])} scans from {sd}", file=sys.stderr)

    per = {}
    for tag, npz, mdl, align, sd in RUNS:
        npz = npz if npz.startswith("/") else os.path.join(ROOT, "diagnostics/moonshot", npz)
        R = load_run(npz, os.path.join(ROOT, "3D_model_prep", mdl), align=align)
        Jp = posed_skeleton(R)
        labels = [os.path.splitext(str(x))[0] for x in R["d"]["labels"]]
        rows = {}
        for i, k in enumerate(labels):
            if k not in scans[sd]:
                continue
            V, F, diag, tree = scans[sd][k]
            j = Jp[i]
            w = winding_number(j, V, F)
            gap, _ = tree.query(j)
            dfv, _ = tree.query(R["d"]["verts"][i].astype(np.float64))
            rows[k] = (float((np.abs(w) < 0.5).mean()), float(np.median(gap) / diag), float(dfv.mean() / diag))
        per[tag] = rows
        print(f"  {tag}: {len(rows)} specimens", file=sys.stderr, flush=True)

    with open(OUT, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        print(
            f"\n{'run':<15}{'scans':<10}{'template':<34}{'n':>4}{'%out(GWN)':>11}{'med gap%':>10}{'chamfer%':>10}",
            file=fh,
        )
        for tag, npz, mdl, align, sd in RUNS:
            a = np.array(list(per[tag].values()))
            print(
                f"{tag:<15}{sd:<10}{mdl:<34}{a.shape[0]:4d}"
                f"{100 * np.median(a[:, 0]):11.2f}{100 * np.median(a[:, 1]):10.3f}"
                f"{100 * np.median(a[:, 2]):10.3f}",
                file=fh,
            )

        print("\n" + "=" * 92, file=fh)
        print("A) PIPELINE VARIED, SCANS HELD FIXED (paired, same specimen)", file=fh)
        for sd, pairs in (
            ("bench50", [("LIM_0", "M7_worker"), ("LIM_0", "baseline"), ("M7_worker", "baseline")]),
            ("clean81", [("CLEAN_M7", "CLEAN_legacy")]),
        ):
            for x, y in pairs:
                common = sorted(set(per[x]) & set(per[y]))
                a = np.array([per[x][k][0] for k in common]) * 100
                b = np.array([per[y][k][0] for k in common]) * 100
                s, p = wilcoxon(a, b)
                print(
                    f"   [{sd}] {x:<12} {np.median(a):6.2f}   vs  {y:<14} {np.median(b):6.2f}"
                    f"   median paired diff {np.median(a - b):+6.2f}  n={len(common)}  p={p:.2e}",
                    file=fh,
                )

        print("\nB) CORPUS VARIED, TEMPLATE AND PIPELINE HELD FIXED  (the matched control)", file=fh)
        a = np.array([v[0] for v in per["M7_worker"].values()]) * 100
        b = np.array([v[0] for v in per["CLEAN_M7"].values()]) * 100
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        print(
            f"   M7 pipeline + SMIL_OmniAnt:  worker scans {np.median(a):6.2f}  vs  "
            f"clean scans {np.median(b):6.2f}   ratio {np.median(a) / np.median(b):.2f}  p={p:.2e}",
            file=fh,
        )

        print("\nC) THE HEADLINE CONTRAST, AND WHAT THE CONTROLS ATTRIBUTE IT TO", file=fh)
        h_w = np.median([v[0] for v in per["LIM_0"].values()]) * 100
        h_c = np.median([v[0] for v in per["CLEAN_legacy"].values()]) * 100
        m_w = np.median(a)
        m_c = np.median(b)
        print(
            f"   headline               LIM_0 {h_w:.2f}  -  CLEAN_legacy {h_c:.2f}  = {h_w - h_c:+.2f} pts"
            f"   (ratio {h_w / h_c:.2f})",
            file=fh,
        )
        print(
            f"   matched corpus effect  M7_worker {m_w:.2f}  -  CLEAN_M7 {m_c:.2f}  = {m_w - m_c:+.2f} pts"
            f"   (ratio {m_w / m_c:.2f})",
            file=fh,
        )
        cm = np.median([v[0] for v in per["CLEAN_M7"].values()]) * 100
        cl = np.median([v[0] for v in per["CLEAN_legacy"].values()]) * 100
        print(
            f"   pipeline effect ON THE SAME CLEAN SCANS  CLEAN_M7 {cm:.2f}  -  CLEAN_legacy "
            f"{cl:.2f}  = {cm - cl:+.2f} pts   (ratio {cm / cl:.2f})",
            file=fh,
        )
        lw = np.median([v[0] for v in per["LIM_0"].values()]) * 100
        print(
            f"   pipeline effect ON THE SAME WORKER SCANS LIM_0 {lw:.2f}  -  M7_worker {m_w:.2f}"
            f"  = {lw - m_w:+.2f} pts",
            file=fh,
        )
        print(
            "\n   => share of the headline gap explained WITHOUT any corpus difference "
            f"(pipeline alone, clean scans): {100 * (cm - cl) / (h_w - h_c):.1f}%",
            file=fh,
        )
        print("=" * 92, file=fh)
        np.savez(
            os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_matched.npz"),
            **{f"{t}_val": np.array(list(per[t].values())) for t in per},
            **{f"{t}_key": np.array(list(per[t].keys())) for t in per},
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
