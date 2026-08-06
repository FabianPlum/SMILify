"""PROBE 23 -- is the scan-based joint metric just a shadow of a worse SURFACE fit?

Probe 21 measured that worker joints sit outside their own scan 61.5% of the time vs 30.1%
for the clean corpus. But the worker SURFACE fit is also worse (fitted->scan mean NN
0.317% vs 0.222% of the scan diagonal). If joint-outside-ness is simply proportional to
surface-fit error, the joint metric carries no independent information.

This probe computes, per specimen:
    chamfer   fitted-mesh -> scan mean nearest-neighbour distance / scan diagonal
    outfrac   fraction of the 55 joints outside the scan (|winding number| < 0.5)
    gap       median joint->scan distance / scan diagonal
    centr     median surface centrality of the joints wrt the scan
and reports (a) the within-corpus Spearman correlation between chamfer and each joint
metric, and (b) a Mann-Whitney U across specimens (worker vs clean), which is the correct
unit of independence -- unlike the pairwise pooling used elsewhere.

Also dumps per-specimen arrays to joint_placement_confound.npz for later reuse.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, mannwhitneyu
from joint_placement_common import load_run, winding_number
from joint_placement_vs_scan_PROBE import load_obj_np, normalise, posed_skeleton, centrality

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
    ("ALL_ANTS_CLEAN", CLEAN_NPZ, "3D_model_prep/SMPL_fit.pkl", True, "diagnostics/moonshot/clean81"),
]


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_confound_out.txt")
    store = {}
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
            rows = []
            for i in range(R["n"]):
                if labels[i] not in avail:
                    continue
                V, F = load_obj_np(avail[labels[i]])
                V = normalise(V)
                diag = np.linalg.norm(V.max(0) - V.min(0))
                tree = cKDTree(V)
                dfv, _ = tree.query(R["d"]["verts"][i].astype(np.float64))
                w = winding_number(Jp[i], V, F)
                gap, _ = tree.query(Jp[i])
                cen = centrality(Jp[i], V, tree)
                rows.append(
                    (
                        dfv.mean() / diag,
                        float((np.abs(w) < 0.5).mean()),
                        float(np.median(gap) / diag),
                        float(np.median(cen)),
                    )
                )
            a = np.array(rows)
            store[tag] = a
            print(f"\n### {tag}   n={a.shape[0]}", file=fh)
            print(
                f"   {'metric':<12}{'median':>10}{'mean':>10}{'sd':>10}{'spearman r vs chamfer':>24}{'p':>10}", file=fh
            )
            for k, nm in enumerate(["chamfer", "outfrac", "gap", "centrality"]):
                if k == 0:
                    print(
                        f"   {nm:<12}{np.median(a[:, k]):10.4f}{a[:, k].mean():10.4f}"
                        f"{a[:, k].std():10.4f}{'--':>24}{'--':>10}",
                        file=fh,
                    )
                else:
                    r, p = spearmanr(a[:, 0], a[:, k])
                    print(
                        f"   {nm:<12}{np.median(a[:, k]):10.4f}{a[:, k].mean():10.4f}"
                        f"{a[:, k].std():10.4f}{r:24.3f}{p:10.2e}",
                        file=fh,
                    )

        A, B = store["LIM_0 (worker)"], store["ALL_ANTS_CLEAN"]
        print("\n" + "=" * 92, file=fh)
        print("PER-SPECIMEN worker vs clean (Mann-Whitney U, two-sided)", file=fh)
        print(f"{'metric':<12}{'worker med':>12}{'clean med':>12}{'ratio':>9}{'U':>10}{'p':>12}", file=fh)
        for k, nm in enumerate(["chamfer", "outfrac", "gap", "centrality"]):
            u, p = mannwhitneyu(A[:, k], B[:, k], alternative="two-sided")
            mw, mc = np.median(A[:, k]), np.median(B[:, k])
            print(f"{nm:<12}{mw:12.4f}{mc:12.4f}{mw / mc:9.2f}{u:10.0f}{p:12.2e}", file=fh)

        # partial: does the joint metric survive controlling for chamfer?
        print("\nControlling for surface fit: regress each joint metric on chamfer WITHIN each", file=fh)
        print("corpus, then compare the residuals across corpora.", file=fh)
        for k, nm in enumerate(["outfrac", "gap", "centrality"], start=1):
            allx = np.concatenate([A[:, 0], B[:, 0]])
            ally = np.concatenate([A[:, k], B[:, k]])
            sl, ic = np.polyfit(allx, ally, 1)
            ra = A[:, k] - (sl * A[:, 0] + ic)
            rb = B[:, k] - (sl * B[:, 0] + ic)
            u, p = mannwhitneyu(ra, rb, alternative="two-sided")
            print(
                f"   {nm:<12} pooled slope={sl:8.3f}  resid median worker={np.median(ra):+.4f}"
                f"  clean={np.median(rb):+.4f}  U={u:.0f}  p={p:.2e}",
                file=fh,
            )
        print("=" * 92, file=fh)
    np.savez(
        os.path.join(ROOT, "diagnostics/moonshot/joint_placement_confound.npz"),
        **{k.split()[0]: v for k, v in store.items()},
    )
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
