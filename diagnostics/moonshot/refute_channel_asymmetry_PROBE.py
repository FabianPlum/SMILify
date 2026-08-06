"""REFUTE-F -- the worker pipeline has 165 joint-placement parameters the CLEAN one does not.

The CLEAN npz has NO `betas_trans` key at all (verified: 'betas_trans' not in d.files); the
worker runs do. betas_trans is a free, per-specimen (55,3) offset added to every bone's
local offset in batch_lbs.py:158. So the worker fitter can move joints directly and the
clean fitter cannot. Any metric that scores "how well are the joints placed" will therefore
separate the two corpora even if the SCANS were identical.

This probe measures how far betas_trans actually moves the worker joints, and how far
log_beta_scales does, both as % of the body diagonal, so the size of that structural
advantage/handicap can be compared with the claimed effect (median joint->scan gap
0.495% (worker) vs 0.280% (clean) of the diagonal).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_placement_common import (  # noqa: E402
    align_template,
    global_rigid,
    joint_regress,
    load_model,
    rodrigues,
)

ROOT = "/home/fabi/dev/SMILify"
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)
RUNS = [
    (
        "LIM_0",
        f"{ROOT}/diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        f"{ROOT}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
    ),
    (
        "LIM_1x",
        f"{ROOT}/diagnostics/moonshot/runs/LIM_1x/Stage_3_deform_fine.npz",
        f"{ROOT}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
    ),
    (
        "baseline",
        f"{ROOT}/diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        f"{ROOT}/3D_model_prep/SMIL_OmniAnt.pkl",
        False,
    ),
    ("CLEAN", CLEAN_NPZ, f"{ROOT}/3D_model_prep/SMPL_fit.pkl", True),
]


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = f"{ROOT}/diagnostics/moonshot/refute_channel_asymmetry_out.txt"
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        print(
            f"{'run':<10}{'has bt':>8}{'|dJ| betas_trans':>19}{'|dJ| logscale':>16}"
            f"{'sd(logscale)':>15}{'sd(betas)':>12}{'|J_st-J_reg|':>14}",
            file=fh,
        )
        for tag, npz, mdl, align in RUNS:
            d = np.load(npz, allow_pickle=True)
            dd = load_model(mdl)
            n = d["betas"].shape[0]
            nJ = np.asarray(dd["J_regressor"]).shape[0]
            vt = np.asarray(dd["v_template"], float)
            if align:
                vt = align_template(vt, dd["sym_verts"])
            betas = d["betas"].astype(np.float64)
            sd = np.asarray(dd["shapedirs"], float)
            v_shaped = vt[None] + np.einsum("nk,vck->nvc", betas, sd[:, :, : betas.shape[1]])
            Jr = np.asarray(dd["J_regressor"], float)
            J = joint_regress(v_shaped, Jr)
            theta = np.concatenate([d["global_rot"][:, None, :], d["joint_rot"]], 1).astype(float)
            Rs = rodrigues(theta.reshape(-1, 3)).reshape(n, nJ, 3, 3)
            par = np.asarray(dd["kintree_table"][0]).astype(int)
            lbs = d["log_beta_scales"].astype(np.float64)
            bt = d["betas_trans"].astype(np.float64) if "betas_trans" in d.files else None
            ref = d["verts"].astype(np.float64)
            diag = np.linalg.norm(ref.max(1) - ref.min(1), axis=1).mean()

            Jfull, A = global_rigid(Rs, J, par, lbs, bt)
            Jnobt, _ = global_rigid(Rs, J, par, lbs, None)
            Jnosc, _ = global_rigid(Rs, J, par, None, bt)
            dbt = np.linalg.norm(Jfull - Jnobt, axis=2) / diag * 100
            dsc = np.linalg.norm(Jfull - Jnosc, axis=2) / diag * 100
            W = np.asarray(dd["weights"], float)
            T = np.einsum("vj,njab->nvab", W, A)
            vh = np.concatenate([v_shaped, np.ones((n, v_shaped.shape[1], 1))], 2)
            v_posed = np.einsum("nvab,nvb->nva", T, vh)[:, :, :3] + d["trans"][:, None, :]
            J_static = Jfull + d["trans"][:, None, :]
            J_rp = joint_regress(v_posed, Jr)
            dd_def = np.linalg.norm(J_static - J_rp, axis=2) / diag * 100
            print(
                f"{tag:<10}{str(bt is not None):>8}{np.median(dbt):19.3f}{np.median(dsc):16.3f}"
                f"{lbs.std(0).mean():15.4f}{betas.std(0).mean():12.4f}"
                f"{np.median(dd_def):14.3f}",
                file=fh,
            )
            fh.flush()
        print("\nunits: |dJ| columns are the median over all joints/specimens of the joint", file=fh)
        print("displacement caused by switching that channel off, in % of the body diagonal.", file=fh)
        print("Compare with the claimed effect size: median joint->scan gap 0.495% (worker)", file=fh)
        print("vs 0.280% (clean) of the diagonal, i.e. a 0.215 %diag difference.", file=fh)
    print(f"wrote {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
