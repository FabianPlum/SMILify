"""PROBE 27 -- direct ablation of the two channels the claim indicts.

The claim attributes "61.5% of worker joints outside the scan" to JOINT PLACEMENT, i.e. to
log_beta_scales (per-joint segment scaling) and betas_trans (per-joint translation offsets),
the free, essentially unregularised per-specimen parameters. That is a causal statement about
two named tensors, and it is directly testable: switch them off and re-measure.

  full        as fitted
  bt=0        betas_trans zeroed        (the channel that moves joints outright)
  lbs=0       log_beta_scales zeroed    (the channel that changes segment length)
  both=0      both zeroed -> joints come from J_regressor . v_shaped and pose alone

Everything else (betas, pose, global translation, the scan) is held fixed. If the free
placement channels are what puts the joints outside the animal, switching them off must move
%-outside DOWN, towards the shape-space-only placement. If %-outside goes UP, the channels are
correcting placement, not corrupting it, and the metric is reporting something else.

The same ablation is run on the clean corpus for log_beta_scales (its npz has no betas_trans).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
from joint_placement_common import load_run, global_rigid, joint_group, winding_number

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
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_ablation_out.txt")
    fout = open(outp, "w")
    fh = Tee(sys.stdout, fout)
    print(__doc__, file=fh)

    for tag, npz, mdl, align, scan_dir in RUNS:
        npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
        R = load_run(npz, os.path.join(ROOT, mdl), align=align)
        names = R["J_names"]
        groups = np.array([joint_group(n) for n in names])
        z = np.zeros_like(R["logscale"])
        variants = {"full": (R["logscale"], R["betas_trans"]), "lbs=0": (z, R["betas_trans"]), "both=0": (z, None)}
        if R["betas_trans"] is not None:
            variants["bt=0"] = (R["logscale"], None)
        J = {}
        for k, (ls, bt) in variants.items():
            j, _ = global_rigid(R["Rs"], R["J"], R["parents"], ls, bt)
            J[k] = j + R["trans"][:, None, :]

        labels = [str(x) for x in R["d"]["labels"]]
        sd = os.path.join(ROOT, scan_dir)
        avail = {os.path.splitext(x)[0]: os.path.join(sd, x) for x in os.listdir(sd) if x.endswith(".obj")}
        acc = {k: [] for k in variants}
        gapacc = {k: [] for k in variants}
        shift = {k: [] for k in variants if k != "full"}
        print(f"\n{'=' * 100}\n### {tag}  n={R['n']}", file=fh)
        for i in range(R["n"]):
            key = os.path.splitext(labels[i])[0]
            if key not in avail:
                continue
            V, F = load_obj_np(avail[key])
            V = normalise(V)
            diag = float(np.linalg.norm(V.max(0) - V.min(0)))
            tree = cKDTree(V)
            for k in variants:
                w = winding_number(J[k][i], V, F)
                acc[k].append(np.abs(w) < 0.5)
                gapacc[k].append(tree.query(J[k][i])[0] / diag)
                if k != "full":
                    shift[k].append(np.linalg.norm(J[k][i] - J["full"][i], axis=1) / diag)
            if (i + 1) % 20 == 0:
                print(f"   ... {i + 1}", file=fh)
                fh.flush()
        for k in variants:
            a = np.array(acc[k])
            g = np.array(gapacc[k])
            extra = ""
            if k != "full":
                s = np.array(shift[k])
                u, p = stats.wilcoxon(np.array(acc["full"]).mean(1), a.mean(1))
                extra = f"   median joint move={100 * np.median(s):.3f}%diag   wilcoxon vs full p={p:.3g}"
            print(
                f"   {k:<8} %outside={100 * a.mean():5.1f}   median gap={100 * np.median(g):.3f}%diag{extra}", file=fh
            )
        print(f"   {'group':<14}" + "".join(f"{k:>10}" for k in variants), file=fh)
        for gname in sorted(set(groups)):
            m = groups == gname
            print(
                f"   {gname:<14}" + "".join(f"{100 * np.array(acc[k])[:, m].mean():10.1f}" for k in variants), file=fh
            )
        fh.flush()

    fh.flush()
    fout.close()
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
