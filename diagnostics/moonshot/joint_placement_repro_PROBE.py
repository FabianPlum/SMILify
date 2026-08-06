"""PROBE 20a -- verify the numpy re-implementation of the forward pass.

Reconstructs `verts` from the stored parameters and compares against the `verts` array
saved in the npz. If this does not match to numerical precision, every joint position
computed downstream is meaningless, so this gate runs first.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from joint_placement_common import load_run, global_rigid

RUNS = [
    (
        "LIM_0 (worker, w_limit=0)",
        "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
    ),
    (
        "LIM_1x (worker, w_limit=0.273)",
        "diagnostics/moonshot/runs/LIM_1x/Stage_3_deform_fine.npz",
        "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
    ),
    (
        "M7_handoff_midline (worker)",
        "diagnostics/moonshot/runs/M7_handoff_midline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
    ),
    (
        "ALL_ANTS_CLEAN (81)",
        "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz",
        "3D_model_prep/SMPL_fit.pkl",
    ),
]

ROOT = "/home/fabi/dev/SMILify"


def main():
    for name, npz, mdl in RUNS:
        npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
        mdl = mdl if mdl.startswith("/") else os.path.join(ROOT, mdl)
        try:
            R = load_run(npz, mdl)
        except Exception as e:
            print(f"{name}: LOAD FAILED {type(e).__name__}: {e}")
            continue

        n, nJ = R["n"], R["nJ"]
        _, A = global_rigid(R["Rs"], R["J"], R["parents"], R["logscale"], R["betas_trans"])
        W = R["weights"]  # (V,55)
        v_posed = R["v_shaped"]  # posedirs are empty -> no pose blendshape
        T = np.einsum("vj,njab->nvab", W, A)  # (n,V,4,4)
        vh = np.concatenate([v_posed, np.ones((n, v_posed.shape[1], 1))], axis=2)
        verts = np.einsum("nvab,nvb->nva", T, vh)[:, :, :3]
        verts = verts + R["trans"][:, None, :] + R["d"]["deform_verts"].astype(np.float64)

        ref = R["d"]["verts"].astype(np.float64)
        err = np.linalg.norm(verts - ref, axis=2)
        scale = np.linalg.norm(ref.max(1) - ref.min(1), axis=1).mean()  # mean bbox diagonal
        print(f"{name}")
        print(
            f"   n={n} nJ={nJ} nverts={v_posed.shape[1]} nbetas={R['d']['betas'].shape[1]} "
            f"betas_trans_present={'betas_trans' in R['d'].files}"
        )
        print(f"   mean bbox diag           = {scale:.6f}")
        print(
            f"   vert reconstruction err  mean={err.mean():.3e}  max={err.max():.3e}  "
            f"(rel to bbox diag: mean={err.mean() / scale:.2e} max={err.max() / scale:.2e})"
        )
        print()


if __name__ == "__main__":
    main()
