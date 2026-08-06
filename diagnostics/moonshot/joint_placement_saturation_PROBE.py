"""PROBE 28 -- what is the DYNAMIC RANGE of "% joints outside the scan" on each corpus?

Probe 27 measured that deleting BOTH joint-placement channels from the worker fits
(log_beta_scales and betas_trans, the two tensors the claim indicts) moves every joint by a
median 4.33% of the body diagonal -- roughly nine times the median joint-to-scan gap -- and
leaves %-outside at 61.6% against 61.5% for the unmodified fit (p=0.87). A statistic that does
not move when you delete its supposed cause is saturated.

This probe maps the saturation directly and model-free: displace every fitted joint by an
isotropic random vector of magnitude eps*diag and sweep eps. The resulting curve is the
metric's response function on that corpus.

  * If the worker curve is flat near 61% from eps=0 upward, the reading carries no information
    about placement on that corpus, and 61.5 vs 30.1 compares a saturated meter to a live one.
  * The eps at which each corpus reaches its OBSERVED value is the placement error the metric
    implies -- if those differ by much more than the observed ratio, the metric is nonlinear
    between corpora and the ratio 2.12 is not interpretable as "2.12x worse placement".

Also reported: the same sweep started from the SURFACE-CONSISTENT reference skeleton
J_reg . posed_verts (zero placement error by construction), which gives the response function
free of any fit error at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from joint_placement_common import load_run, global_rigid, winding_number

ROOT = "/home/fabi/dev/SMILify"
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)
EPS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # % of scan bbox diagonal
NDIR = 4

RUNS = [
    (
        "LIM_0 (worker)",
        "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
        "diagnostics/moonshot/bench50",
    ),
    ("ALL_ANTS_CLEAN (81)", CLEAN_NPZ, "3D_model_prep/SMPL_fit.pkl", True, "diagnostics/moonshot/clean81"),
]


def load_obj_np(path):
    V, F = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=int)


def normalise(V):
    V = V - V.mean(0)
    return V / np.abs(V).max(0).max()


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_saturation_out.txt")
    fout = open(outp, "w")
    fh = Tee(sys.stdout, fout)
    print(__doc__, file=fh)
    rng = np.random.default_rng(11)
    curves = {}

    for tag, npz, mdl, align, scan_dir in RUNS:
        npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
        R = load_run(npz, os.path.join(ROOT, mdl), align=align)
        Jp, _ = global_rigid(R["Rs"], R["J"], R["parents"], R["logscale"], R["betas_trans"])
        Jp = Jp + R["trans"][:, None, :]
        Jreg = np.asarray(R["dd"]["J_regressor"], dtype=np.float64)
        labels = [str(x) for x in R["d"]["labels"]]
        sd = os.path.join(ROOT, scan_dir)
        avail = {os.path.splitext(x)[0]: os.path.join(sd, x) for x in os.listdir(sd) if x.endswith(".obj")}
        acc = {("fit", e): [] for e in EPS}
        acc.update({("ref", e): [] for e in EPS})
        print(f"\n{'=' * 100}\n### {tag}  n={R['n']}", file=fh)
        for i in range(R["n"]):
            key = os.path.splitext(labels[i])[0]
            if key not in avail:
                continue
            V, F = load_obj_np(avail[key])
            V = normalise(V)
            diag = float(np.linalg.norm(V.max(0) - V.min(0)))
            fv = R["d"]["verts"][i].astype(np.float64)
            for name, J in (("fit", Jp[i]), ("ref", Jreg @ fv)):
                for e in EPS:
                    if e == 0.0:
                        acc[(name, e)].append((np.abs(winding_number(J, V, F)) < 0.5).mean())
                        continue
                    vals = []
                    for _ in range(NDIR):
                        d = rng.normal(size=J.shape)
                        d /= np.linalg.norm(d, axis=1, keepdims=True)
                        q = J + (e / 100.0) * diag * d
                        vals.append((np.abs(winding_number(q, V, F)) < 0.5).mean())
                    acc[(name, e)].append(np.mean(vals))
            if (i + 1) % 10 == 0:
                print(f"   ... {i + 1}", file=fh)
                fh.flush()
        curves[tag] = acc
        print(f"   {'eps %diag':<12}{'from FITTED skeleton':>22}{'from SURFACE-CONSISTENT ref':>30}", file=fh)
        for e in EPS:
            a = 100 * np.mean(acc[("fit", e)])
            b = 100 * np.mean(acc[("ref", e)])
            print(f"   {e:<12.2f}{a:22.1f}{b:30.1f}", file=fh)
        fh.flush()

    print(f"\n{'=' * 100}\nRESPONSE FUNCTION SUMMARY", file=fh)
    w = curves["LIM_0 (worker)"]
    c = curves["ALL_ANTS_CLEAN (81)"]
    print(f"   {'eps %diag':<12}{'worker':>10}{'clean':>10}{'worker-eps0':>14}{'clean-eps0':>13}", file=fh)
    w0 = 100 * np.mean(w[("fit", 0.0)])
    c0 = 100 * np.mean(c[("fit", 0.0)])
    for e in EPS:
        a = 100 * np.mean(w[("fit", e)])
        b = 100 * np.mean(c[("fit", e)])
        print(f"   {e:<12.2f}{a:10.1f}{b:10.1f}{a - w0:14.1f}{b - c0:13.1f}", file=fh)
    print(
        f"\n   worker sensitivity  d(%out)/d(eps) over 0->1%diag = "
        f"{(100 * np.mean(w[('fit', 1.0)]) - w0) / 1.0:+.1f} pp per %diag",
        file=fh,
    )
    print(
        f"   clean  sensitivity  d(%out)/d(eps) over 0->1%diag = "
        f"{(100 * np.mean(c[('fit', 1.0)]) - c0) / 1.0:+.1f} pp per %diag",
        file=fh,
    )
    print(
        f"   worker saturation ceiling (eps=8%) = {100 * np.mean(w[('fit', 8.0)]):.1f}%   "
        f"clean = {100 * np.mean(c[('fit', 8.0)]):.1f}%",
        file=fh,
    )
    print(
        f"   observed worker 61.5% sits at {100 * (w0 - 100 * np.mean(w[('ref', 0.0)])) / max(1e-9, 100 * np.mean(w[('fit', 8.0)]) - 100 * np.mean(w[('ref', 0.0)])):.0f}% "
        f"of its own dynamic range (ref-floor -> ceiling)",
        file=fh,
    )
    print(
        f"   observed clean  30.1% sits at {100 * (c0 - 100 * np.mean(c[('ref', 0.0)])) / max(1e-9, 100 * np.mean(c[('fit', 8.0)]) - 100 * np.mean(c[('ref', 0.0)])):.0f}% "
        f"of its own dynamic range",
        file=fh,
    )
    a = np.array(w[("fit", 0.0)])
    b = np.array(w[("fit", 4.0)])
    t, p = stats.wilcoxon(a, b)
    print(f"   worker: fitted vs fitted+4%diag random displacement, paired p={p:.3g}", file=fh)
    a = np.array(c[("fit", 0.0)])
    b = np.array(c[("fit", 4.0)])
    t, p = stats.wilcoxon(a, b)
    print(f"   clean : fitted vs fitted+4%diag random displacement, paired p={p:.3g}", file=fh)

    fh.flush()
    fout.close()
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
