"""REFUTE-6 -- are the worker SCANS missing the body parts whose joints are scored "outside"?

The metric asks "is the fitted joint inside the target scan".  If the scan does not CONTAIN the
body part (broken tarsus, snapped antenna, unresolved thin structure in the CT isosurface), the
joint there is outside no matter how well it was placed.  That is a property of the specimen and
its scan, not of the registration.

The prior work gated on the MEAN fitted->scan nearest-neighbour distance (0.317% worker vs
0.222% clean) and called the frames matched.  A mean cannot see a missing limb: 3% of the
vertices sitting 3% of the diagonal away from anything moves the mean by 0.09%.  This probe
measures the TAIL instead, and localises it to the joints.

Per specimen:
  orphan_v    fraction of fitted-mesh vertices further than T from any scan point
              (model geometry standing in empty space => the scan lacks that part)
  orphan_j    fraction of the 55 joints whose skinning-dominant body region is orphaned:
              measured as the fraction of joints whose 200 nearest FITTED-MESH vertices are
              on average further than T from the scan
  extra_s     fraction of scan points further than T from the fitted mesh (debris / medium)
  outfrac     the headline metric, for the joint-level association
and then, WITHIN each corpus, the per-joint association between "this joint's neighbourhood is
missing from the scan" and "this joint is scored outside".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr
from joint_placement_common import load_run, winding_number
from joint_placement_vs_scan_PROBE import load_obj_np, normalise, posed_skeleton

ROOT = "/home/fabi/dev/SMILify"
OUT = os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_completeness_out.txt")
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

RUNS = [
    ("LIM_0", "runs/LIM_0/Stage_3_deform_fine.npz", "OmniAnt_25PCs_joint_limited.pkl", False, "bench50"),
    ("M7_worker", "runs/M7_handoff_midline/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl", False, "bench50"),
    ("CLEAN_M7", "runs/CLEAN_M7/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl", False, "clean81"),
    ("CLEAN_legacy", CLEAN_NPZ, "SMPL_fit.pkl", True, "clean81"),
]
T = 0.01  # 1% of the scan bbox diagonal
KNN = 200


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
        print(f"loaded {sd}", file=sys.stderr, flush=True)

    per = {}
    perjoint = {}
    for tag, npz, mdl, align, sd in RUNS:
        npz = npz if npz.startswith("/") else os.path.join(ROOT, "diagnostics/moonshot", npz)
        R = load_run(npz, os.path.join(ROOT, "3D_model_prep", mdl), align=align)
        Jp = posed_skeleton(R)
        labels = [os.path.splitext(str(x))[0] for x in R["d"]["labels"]]
        rows, pj_orph, pj_out = [], [], []
        for i, k in enumerate(labels):
            if k not in scans[sd]:
                continue
            V, F, diag, tree = scans[sd][k]
            fv = R["d"]["verts"][i].astype(np.float64)
            dv, _ = tree.query(fv)
            dv = dv / diag
            orphan_v = float((dv > T).mean())
            ftree = cKDTree(fv)
            ds, _ = ftree.query(V)
            extra_s = float((ds / diag > T).mean())
            # joint-local orphan-ness: mean scan-distance of the joint's nearest model verts
            _, nb = ftree.query(Jp[i], k=KNN)
            jorph = dv[nb].mean(1)
            w = winding_number(Jp[i], V, F)
            out = np.abs(w) < 0.5
            rows.append((orphan_v, extra_s, float((jorph > T).mean()), float(out.mean()), float(np.median(jorph))))
            pj_orph.append(jorph)
            pj_out.append(out)
        per[tag] = np.array(rows)
        perjoint[tag] = (np.array(pj_orph), np.array(pj_out))
        print(f"  {tag} done", file=sys.stderr, flush=True)

    with open(OUT, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        print(f"threshold T = {100 * T:.1f}% of the scan bbox diagonal;  knn={KNN}\n", file=fh)
        print(
            f"{'run':<14}{'n':>4}{'orphan model verts %':>22}{'extra scan pts %':>19}"
            f"{'joints in orphan region %':>27}{'%out':>8}",
            file=fh,
        )
        for tag, *_ in RUNS:
            a = per[tag]
            print(
                f"{tag:<14}{a.shape[0]:4d}{100 * np.median(a[:, 0]):22.3f}{100 * np.median(a[:, 1]):19.3f}"
                f"{100 * np.median(a[:, 2]):27.3f}{100 * np.median(a[:, 3]):8.2f}",
                file=fh,
            )

        print("\n" + "=" * 96, file=fh)
        print("A) IS THE SCAN MISSING GEOMETRY THE MODEL CLAIMS EXISTS?  (worker vs clean,", file=fh)
        print("   template and pipeline held fixed: M7_worker vs CLEAN_M7)", file=fh)
        for k, nm in (
            (0, "orphan model verts"),
            (1, "extra scan points"),
            (2, "joints in an orphan region"),
            (4, "median joint-local scan gap"),
        ):
            x, y = per["M7_worker"][:, k], per["CLEAN_M7"][:, k]
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            r = np.median(x) / np.median(y) if np.median(y) > 0 else np.inf
            print(
                f"   {nm:<32} worker {100 * np.median(x):8.3f}   clean {100 * np.median(y):8.3f}"
                f"   ratio {r:6.2f}   p={p:.2e}",
                file=fh,
            )

        print("\nB) DOES ORPHAN-NESS EXPLAIN 'OUTSIDE' AT THE JOINT LEVEL, WITHIN A CORPUS?", file=fh)
        for tag in ("M7_worker", "CLEAN_M7", "LIM_0", "CLEAN_legacy"):
            o, ou = perjoint[tag]
            rho, p = spearmanr(o.ravel(), ou.ravel().astype(float))
            in_o = ou.ravel()[o.ravel() > T]
            in_n = ou.ravel()[o.ravel() <= T]
            print(
                f"   {tag:<14} rho(joint-local scan gap, outside) = {rho:+.3f} (p={p:.1e})   "
                f"P(outside | orphan)={100 * in_o.mean() if in_o.size else float('nan'):5.1f}%  "
                f"P(outside | not orphan)={100 * in_n.mean():5.1f}%  "
                f"n_orphan={in_o.size}/{o.size}",
                file=fh,
            )

        print("\nC) RESTRICTED TO JOINTS WHOSE NEIGHBOURHOOD IS PRESENT IN BOTH CORPORA", file=fh)
        print("   (drop every joint sitting in an orphan region; recompute the headline)", file=fh)
        for a_tag, b_tag in (("M7_worker", "CLEAN_M7"), ("LIM_0", "CLEAN_legacy")):
            oa, ua = perjoint[a_tag]
            ob, ub = perjoint[b_tag]
            xa = np.array([ua[i][oa[i] <= T].mean() if (oa[i] <= T).any() else np.nan for i in range(ua.shape[0])])
            xb = np.array([ub[i][ob[i] <= T].mean() if (ob[i] <= T).any() else np.nan for i in range(ub.shape[0])])
            xa, xb = xa[~np.isnan(xa)], xb[~np.isnan(xb)]
            u, p = mannwhitneyu(xa, xb, alternative="two-sided")
            raw_a = np.median(per[a_tag][:, 3]) * 100
            raw_b = np.median(per[b_tag][:, 3]) * 100
            print(
                f"   {a_tag:<12} vs {b_tag:<14} raw {raw_a:5.2f}/{raw_b:5.2f} ratio "
                f"{raw_a / raw_b:.2f}  ->  present-only {100 * np.median(xa):5.2f}/"
                f"{100 * np.median(xb):5.2f} ratio {np.median(xa) / np.median(xb):.2f}  p={p:.2e}",
                file=fh,
            )
        print("=" * 96, file=fh)
        np.savez(
            os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_completeness.npz"),
            **{f"{t}": per[t] for t in per},
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
