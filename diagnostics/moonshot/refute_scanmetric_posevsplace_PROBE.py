"""REFUTE-5 -- does "% joints outside the scan" measure JOINT PLACEMENT, or POSE?

The posed joint the metric scores is
    J_posed = f(betas, log_beta_scales, betas_trans, joint_rot, global_rot, trans)
so a joint can land outside the animal because the REST skeleton is misplaced (the hypothesis)
or because the POSE is wrong (a different failure).  The headline does not separate them.

A discriminating pair already exists on disk, measured in probe 23:
    LIM_0   rotation-limit violations 37.42 axes/specimen, median overshoot 20.5 deg
    LIM_1x  rotation-limit violations  0.76 axes/specimen, median overshoot  0.0 deg
    but their REST-SPACE joint displacement is nearly identical:
            cumulative joint displacement 7.309% (LIM_0) vs 6.873% (LIM_1x) of body,
            trans-only 3.888% vs 4.059%, scale-only 4.506% vs 4.417%.
So LIM_0 -> LIM_1x is an almost pure POSE intervention at fixed placement, on the same 50
specimens and the same 50 scans.  Whatever the metric does across that pair is what it is
measuring.

Also scored: the ANATOMICAL-PLACEMENT NULL.  Replace the fitted rest joints J by the AUTHORED
template joints dd['J'] (and by J_regressor . v_template), keeping the fitted pose, scale,
translation.  That is a skeleton with the template's placement and the specimen's pose.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from joint_placement_common import load_run, global_rigid, winding_number, joint_regress
from joint_placement_vs_scan_PROBE import load_obj_np, normalise

ROOT = "/home/fabi/dev/SMILify"
OUT = os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_posevsplace_out.txt")

RUNS = [
    ("LIM_0", "runs/LIM_0/Stage_3_deform_fine.npz", "OmniAnt_25PCs_joint_limited.pkl"),
    ("LIM_1x", "runs/LIM_1x/Stage_3_deform_fine.npz", "OmniAnt_25PCs_joint_limited.pkl"),
    ("LIM_3x", "runs/LIM_3x/Stage_3_deform_fine.npz", "OmniAnt_25PCs_joint_limited.pkl"),
    ("M7_worker", "runs/M7_handoff_midline/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl"),
    ("BPX_noprior", "runs/BPX_noprior/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl"),
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
    d = os.path.join(ROOT, "diagnostics/moonshot/bench50")
    scans = {}
    for fn in sorted(x for x in os.listdir(d) if x.endswith(".obj")):
        V, F = load_obj_np(os.path.join(d, fn))
        V = normalise(V)
        scans[os.path.splitext(fn)[0]] = (V, F, float(np.linalg.norm(V.max(0) - V.min(0))), cKDTree(V))
    print(f"loaded {len(scans)} bench50 scans", file=sys.stderr)

    res = {}
    for tag, npz, mdl in RUNS:
        R = load_run(
            os.path.join(ROOT, "diagnostics/moonshot", npz), os.path.join(ROOT, "3D_model_prep", mdl), align=False
        )
        labels = [os.path.splitext(str(x))[0] for x in R["d"]["labels"]]
        n = R["n"]
        dd = R["dd"]
        Jauth = np.repeat(np.asarray(dd["J"], dtype=np.float64)[None], n, axis=0)
        Jtpl = np.repeat(
            joint_regress(R["v_template"][None], np.asarray(dd["J_regressor"], dtype=np.float64)), n, axis=0
        )
        variants = {
            "fitted": skel(R),
            "no_betas_trans": skel(R, use_trans=False),
            "no_scale_no_trans": skel(R, use_scale=False, use_trans=False),
            "template_J_regressor": skel(R, Jrest=Jtpl),
            "authored_J": skel(R, Jrest=Jauth),
        }
        out = {v: {} for v in variants}
        for i, k in enumerate(labels):
            if k not in scans:
                continue
            V, F, diag, tree = scans[k]
            for v, J in variants.items():
                w = winding_number(J[i], V, F)
                out[v][k] = float((np.abs(w) < 0.5).mean())
        res[tag] = out
        print(f"  {tag} done", file=sys.stderr, flush=True)

    with open(OUT, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        vs = ["fitted", "no_betas_trans", "no_scale_no_trans", "template_J_regressor", "authored_J"]
        print(f"\n{'run':<14}" + "".join(f"{v:>22}" for v in vs), file=fh)
        print("(median %% of the 55 joints outside the raw scan, n=50 worker specimens)", file=fh)
        for tag, _, _ in RUNS:
            row = "".join(f"{100 * np.median(list(res[tag][v].values())):22.2f}" for v in vs)
            print(f"{tag:<14}{row}", file=fh)

        print("\n" + "=" * 96, file=fh)
        print("A) NEARLY-PURE POSE INTERVENTION at fixed placement: LIM_0 -> LIM_1x", file=fh)
        print("   (probe 23: violations 37.42 -> 0.76 axes/specimen; rest-space joint", file=fh)
        print("    displacement 7.309% -> 6.873% of body, i.e. essentially unchanged)", file=fh)
        common = sorted(set(res["LIM_0"]["fitted"]) & set(res["LIM_1x"]["fitted"]))
        a = np.array([res["LIM_0"]["fitted"][k] for k in common]) * 100
        b = np.array([res["LIM_1x"]["fitted"][k] for k in common]) * 100
        s, p = wilcoxon(a, b)
        print(
            f"   %outside  LIM_0 {np.median(a):.2f}  ->  LIM_1x {np.median(b):.2f}   "
            f"median paired change {np.median(a - b):+.2f} pts   n={len(common)}  p={p:.2e}",
            file=fh,
        )
        c = np.array([res["LIM_3x"]["fitted"][k] for k in common]) * 100
        s, p = wilcoxon(a, c)
        print(
            f"   %outside  LIM_0 {np.median(a):.2f}  ->  LIM_3x {np.median(c):.2f}   "
            f"median paired change {np.median(a - c):+.2f} pts   n={len(common)}  p={p:.2e}",
            file=fh,
        )

        print("\nB) ABLATING THE PLACEMENT CHANNELS (same pose, same scan)", file=fh)
        for tag in ("LIM_0", "M7_worker"):
            base = np.array([res[tag]["fitted"][k] for k in res[tag]["fitted"]]) * 100
            for v in vs[1:]:
                x = np.array([res[tag][v][k] for k in res[tag]["fitted"]]) * 100
                s, p = wilcoxon(base, x)
                print(
                    f"   {tag:<12} {v:<24} {np.median(x):6.2f}   (fitted {np.median(base):6.2f}, "
                    f"change {np.median(x - base):+6.2f} pts, p={p:.2e})",
                    file=fh,
                )
        print("=" * 96, file=fh)
        np.savez(
            os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_posevsplace.npz"),
            **{f"{t}_{v}": np.array(list(res[t][v].values())) for t in res for v in vs},
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
