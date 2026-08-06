"""PROBE 24 -- remove the SCAN-DENSITY confound from the centrality statistic.

Probe 21 used the k=128 nearest scan points. Worker scans carry ~49k vertices and clean
scans ~12k, so a fixed k covers 0.26% of a worker scan's vertices but 1.06% of a clean
one -- a physically SMALLER patch on workers, which is locally flatter and therefore biases
centrality UP for workers, in the same direction as the reported effect.

This probe recomputes centrality over a neighbourhood of fixed PHYSICAL radius
(fractions of the scan bbox diagonal), which is density independent. It also reports the
measured scan resolution so the size of the original bias is on record.

Interpretation is unchanged: centrality 0 = the scan surface wraps around the joint,
1 = every nearby surface point lies on one side of it (joint is beside the limb, in air).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from joint_placement_common import load_run, joint_group
from joint_placement_vs_scan_PROBE import load_obj_np, normalise, posed_skeleton

ROOT = "/home/fabi/dev/SMILify"
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)
RUNS = [
    (
        "LIM_0 (worker)",
        "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
        "diagnostics/moonshot/bench50",
    ),
    (
        "baseline (worker)",
        "diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
        False,
        "diagnostics/moonshot/bench50",
    ),
    ("ALL_ANTS_CLEAN", CLEAN_NPZ, "3D_model_prep/SMPL_fit.pkl", True, "diagnostics/moonshot/clean81"),
]
RADII = [0.01, 0.02, 0.04]  # fraction of the scan bbox diagonal


def centrality_radius(J, V, tree, r, min_pts=12):
    """|| mean(p in ball(J,r)) - J || / mean(||p - J||).  nan if the ball is too sparse."""
    out = np.full(J.shape[0], np.nan)
    idx = tree.query_ball_point(J, r)
    for i, ii in enumerate(idx):
        if len(ii) < min_pts:
            continue
        p = V[ii]
        d = np.linalg.norm(p - J[i], axis=1)
        m = d > 0
        if m.sum() < min_pts:
            continue
        out[i] = np.linalg.norm(p[m].mean(0) - J[i]) / d[m].mean()
    return out


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_density_out.txt")
    keep = {}
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        for tag, npz, mdl, align, scan_dir in RUNS:
            npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
            R = load_run(npz, os.path.join(ROOT, mdl), align=align)
            scan_dir = os.path.join(ROOT, scan_dir)
            avail = {
                os.path.splitext(x)[0]: os.path.join(scan_dir, x) for x in os.listdir(scan_dir) if x.endswith(".obj")
            }
            Jp = posed_skeleton(R)
            labels = [os.path.splitext(str(x))[0] for x in R["d"]["labels"]]
            nv, spacing = [], []
            per_r = {r: [] for r in RADII}
            groups = np.array([joint_group(nm) for nm in R["J_names"]])
            for i in range(R["n"]):
                if labels[i] not in avail:
                    continue
                V, _ = load_obj_np(avail[labels[i]])
                V = normalise(V)
                diag = np.linalg.norm(V.max(0) - V.min(0))
                tree = cKDTree(V)
                nv.append(V.shape[0])
                dd, _ = tree.query(V, k=2)
                spacing.append(np.median(dd[:, 1]) / diag)
                for r in RADII:
                    per_r[r].append(centrality_radius(Jp[i], V, tree, r * diag))
            print(f"\n### {tag}   n_specimens={len(nv)}", file=fh)
            print(
                f"   scan vertices: median={int(np.median(nv))}   "
                f"median nearest-neighbour spacing = {100 * np.median(spacing):.4f}% of diag",
                file=fh,
            )
            print(
                f"   {'radius(%diag)':<15}{'n valid':>9}{'median':>9}{'mean':>9}"
                f"{'p90':>9}{'%>0.9':>9}   leg_middle  leg_distal  body",
                file=fh,
            )
            for r in RADII:
                a = np.array(per_r[r])
                v = a[~np.isnan(a)]
                gm = [
                    np.nanmedian(a[:, groups == g]) if np.isfinite(a[:, groups == g]).any() else np.nan
                    for g in ("leg_middle", "leg_distal", "body")
                ]
                print(
                    f"   {100 * r:<15.1f}{v.size:9d}{np.median(v):9.3f}{v.mean():9.3f}"
                    f"{np.percentile(v, 90):9.3f}{100 * (v > 0.9).mean():9.1f}   "
                    + "  ".join(f"{x:10.3f}" for x in gm),
                    file=fh,
                )
            keep[tag] = {r: np.array(per_r[r]) for r in RADII}
            fh.flush()

        print("\n" + "=" * 92, file=fh)
        print("PER-SPECIMEN worker vs clean at matched PHYSICAL radius (Mann-Whitney U)", file=fh)
        print(f"{'radius(%diag)':<15}{'worker med':>12}{'clean med':>12}{'ratio':>9}{'p':>12}", file=fh)
        for r in RADII:
            a = np.nanmedian(keep["LIM_0 (worker)"][r], axis=1)
            b = np.nanmedian(keep["ALL_ANTS_CLEAN"][r], axis=1)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            print(
                f"   {100 * r:<12.1f}{np.median(a):12.3f}{np.median(b):12.3f}"
                f"{np.median(a) / np.median(b):9.2f}{p:12.2e}",
                file=fh,
            )
        print("=" * 92, file=fh)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
