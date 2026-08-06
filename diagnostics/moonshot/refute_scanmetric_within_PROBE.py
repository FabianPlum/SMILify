"""REFUTE-7 -- WITHIN each corpus, does "% joints outside the scan" track the actual amount of
rest-space joint displacement the fit applied?

If %outside is a measure of joint PLACEMENT error, then within a corpus the specimens whose
fits shove their joints furthest from the J_regressor positions (via log_beta_scales and
betas_trans -- the only two channels that can move a joint in rest space, per
joint_placement_common.py's reading of trainer.py:224/240) should be the ones whose joints end
up outside the animal.  A near-zero within-corpus correlation means the statistic carries no
per-specimen information about placement, and the whole worker-vs-clean signal is a single
corpus-level offset with n=1 unit per group.

Rest-space joint displacement is computed exactly as the forward pass does, with identity
rotations:  J_disp = || global_rigid(I, J, parents, logscale, betas_trans) - J || / body_diag.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import spearmanr
from joint_placement_common import load_run, global_rigid

ROOT = "/home/fabi/dev/SMILify"
OUT = os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_within_out.txt")
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

RUNS = [
    ("LIM_0", "runs/LIM_0/Stage_3_deform_fine.npz", "OmniAnt_25PCs_joint_limited.pkl", False),
    ("M7_worker", "runs/M7_handoff_midline/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl", False),
    ("CLEAN_M7", "runs/CLEAN_M7/Stage_3_deform_fine.npz", "SMIL_OmniAnt.pkl", False),
    ("CLEAN_legacy", CLEAN_NPZ, "SMPL_fit.pkl", True),
]


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    M = np.load(os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_matched.npz"), allow_pickle=True)
    with open(OUT, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        print(
            f"\n{'run':<14}{'n':>4}{'rest joint disp %body':>24}{'|betas| L2':>13}"
            f"{'rho(disp, %out)':>18}{'p':>10}{'rho(|betas|,%out)':>20}{'p':>10}",
            file=fh,
        )
        for tag, npz, mdl, align in RUNS:
            npz = npz if npz.startswith("/") else os.path.join(ROOT, "diagnostics/moonshot", npz)
            R = load_run(npz, os.path.join(ROOT, "3D_model_prep", mdl), align=align)
            n = R["n"]
            eye = np.tile(np.eye(3), (n, R["nJ"], 1, 1))
            J1, _ = global_rigid(eye, R["J"], R["parents"], R["logscale"], R["betas_trans"])
            diag = np.linalg.norm(R["v_shaped"].max(1) - R["v_shaped"].min(1), axis=1)
            disp = np.linalg.norm(J1 - R["J"], axis=2).mean(1) / diag
            bn = np.linalg.norm(R["d"]["betas"].astype(np.float64), axis=1)
            keys = [os.path.splitext(str(x))[0] for x in R["d"]["labels"]]
            mk = list(M[f"{tag}_key"])
            mv = M[f"{tag}_val"][:, 0]
            idx = {k: i for i, k in enumerate(mk)}
            sel = [i for i, k in enumerate(keys) if k in idx]
            out = np.array([mv[idx[keys[i]]] for i in sel])
            d_ = disp[sel]
            b_ = bn[sel]
            r1, p1 = spearmanr(d_, out)
            r2, p2 = spearmanr(b_, out)
            print(
                f"{tag:<14}{len(sel):4d}{100 * np.median(d_):24.3f}{np.median(b_):13.3f}"
                f"{r1:18.3f}{p1:10.2e}{r2:20.3f}{p2:10.2e}",
                file=fh,
            )
        print("\nInterpretation: a placement metric should have rho(disp, %out) > 0 within a", file=fh)
        print("corpus. Values near zero mean the statistic separates the two corpora by a", file=fh)
        print("constant offset and orders specimens within a corpus at chance.", file=fh)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
