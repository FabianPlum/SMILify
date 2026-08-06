"""REFUTE-8 -- the placement ablation, run on the CLEAN corpus, to close the argument.

REFUTE-5 measured on the 50 worker specimens that swapping the fitted rest-space skeleton for
a specimen-INDEPENDENT one leaves the headline metric statistically unchanged:
    LIM_0 fitted 61.82  ->  J_regressor.v_template 61.82 (p=0.58)  ->  authored dd['J'] 61.82 (p=0.75)
i.e. the metric cannot see specimen-specific joint placement at all on that corpus.

If the same holds on the clean corpus, then the headline's 61.5-vs-30.1 gap is produced
entirely by things that are NOT joint placement (pose, translation, per-joint scale, and the
scans themselves), and the statistic cannot be used as evidence about joint placement.

Variants (fitted pose / trans always kept):
    fitted                  the fit as stored
    template_J_regressor    rest joints = J_regressor . v_template  (drops the betas channel)
    no_scale                drop log_beta_scales
    no_scale_no_trans       drop log_beta_scales and betas_trans -> the whole placement budget
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import wilcoxon
from joint_placement_common import load_run, global_rigid, winding_number, joint_regress
from joint_placement_vs_scan_PROBE import load_obj_np, normalise

ROOT = "/home/fabi/dev/SMILify"
OUT = os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_ablate_clean_out.txt")
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

RUNS = [
    ("CLEAN_legacy", CLEAN_NPZ, "SMPL_fit.pkl", True),
    ("CLEAN_M7", "runs/CLEAN_M7/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl", False),
]


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def skel(R, Jrest=None, use_scale=True, use_trans=True):
    J = R["J"] if Jrest is None else Jrest
    JJ, _ = global_rigid(
        R["Rs"], J, R["parents"], R["logscale"] if use_scale else None, R["betas_trans"] if use_trans else None
    )
    return JJ + R["trans"][:, None, :]


def main():
    d = os.path.join(ROOT, "diagnostics/moonshot/clean81")
    scans = {}
    for fn in sorted(x for x in os.listdir(d) if x.endswith(".obj")):
        V, F = load_obj_np(os.path.join(d, fn))
        scans[os.path.splitext(fn)[0]] = (normalise(V), F)
    print(f"loaded {len(scans)} clean scans", file=sys.stderr, flush=True)

    res = {}
    for tag, npz, mdl, align in RUNS:
        npz = npz if npz.startswith("/") else os.path.join(ROOT, "diagnostics/moonshot", npz)
        R = load_run(npz, os.path.join(ROOT, "3D_model_prep", mdl), align=align)
        n, dd = R["n"], R["dd"]
        Jtpl = np.repeat(
            joint_regress(R["v_template"][None], np.asarray(dd["J_regressor"], dtype=np.float64)), n, axis=0
        )
        variants = {
            "fitted": skel(R),
            "template_J_regressor": skel(R, Jrest=Jtpl),
            "no_scale": skel(R, use_scale=False),
            "no_scale_no_trans": skel(R, use_scale=False, use_trans=False),
        }
        labels = [os.path.splitext(str(x))[0] for x in R["d"]["labels"]]
        out = {v: {} for v in variants}
        for i, k in enumerate(labels):
            if k not in scans:
                continue
            V, F = scans[k]
            for v, J in variants.items():
                w = winding_number(J[i], V, F)
                out[v][k] = float((np.abs(w) < 0.5).mean())
        res[tag] = out
        print(f"  {tag} done", file=sys.stderr, flush=True)

    with open(OUT, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        vs = ["fitted", "template_J_regressor", "no_scale", "no_scale_no_trans"]
        print(f"\n{'run':<15}" + "".join(f"{v:>24}" for v in vs), file=fh)
        for tag, *_ in RUNS:
            print(f"{tag:<15}" + "".join(f"{100 * np.median(list(res[tag][v].values())):24.2f}" for v in vs), file=fh)
        print("\n(median %% of the 55 joints outside the raw scan)", file=fh)
        print("\n" + "=" * 96, file=fh)
        for tag, *_ in RUNS:
            ks = list(res[tag]["fitted"])
            base = np.array([res[tag]["fitted"][k] for k in ks]) * 100
            for v in vs[1:]:
                x = np.array([res[tag][v][k] for k in ks]) * 100
                s, p = wilcoxon(base, x)
                print(
                    f"   {tag:<14} {v:<24} {np.median(x):6.2f}  (fitted {np.median(base):6.2f}, "
                    f"median paired change {np.median(x - base):+6.2f} pts, mean {np.mean(x - base):+6.2f}, "
                    f"p={p:.2e})",
                    file=fh,
                )
        print("=" * 96, file=fh)
        np.savez(
            os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_ablate_clean.npz"),
            **{f"{t}_{v}": np.array(list(res[t][v].values())) for t in res for v in vs},
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
