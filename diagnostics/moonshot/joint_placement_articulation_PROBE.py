"""PROBE 24 -- is "% joints outside the surface" a measure of PLACEMENT, or of ARTICULATION?

Probe 23 measured that the clean corpus is fitted with mean |joint rotation| 0.088 rad (5 deg)
and the worker corpus with 0.872 rad (50 deg) -- a factor 9.9. Every joint-vs-surface metric
is evaluated on a POSED mesh, so before any of it can be read as placement quality we have to
know what articulation on its own does to the number.

THE CONTROLLED EXPERIMENT (zero placement error by construction)
Take the AUTHORED template mesh and the AUTHORED skeleton. Skin the mesh with the model's own
LBS using a pose theta, and take the skeleton straight out of the same
batch_global_rigid_transformation. Surface and skeleton are then EXACTLY consistent: there is
no shape error, no betas_trans, no log_beta_scales, no deform_verts, no fitting at all. The
only thing that varies is theta.

  arm A   theta = 0                          -> the published "template floor"
  arm B   theta = each CLEAN specimen's pose  (81 poses)
  arm C   theta = each WORKER specimen's pose (50 poses, LIM_0)
  arm D   theta = c * worker pose, c in [0,1] -> dose-response

If arm C scores far above arm A/B with placement error identically zero, then the 61.5 vs 30.1
split is (at least partly) a pose artefact and cannot be attributed to joint placement.

SECOND CHECK: is probe 23's surface-consistent floor J_reg . posed_verts itself valid?
Measure it against its OWN fitted mesh (not the scan). If it is inside its own mesh, the floor
is real (the fitted surface locally disagrees with the scan). If it is outside its own mesh,
the linear regressor breaks under bending and the floor is itself an articulation artefact --
which refutes the placement reading either way, but by a different mechanism.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from joint_placement_common import (
    load_run,
    load_model,
    joint_regress,
    global_rigid,
    rodrigues,
    joint_group,
    winding_number,
)

ROOT = "/home/fabi/dev/SMILify"
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)


def lbs(v_rest, A, W):
    """v_rest (V,3), A (J,4,4), W (V,J) -> posed verts (V,3). Matches smal_torch skinning."""
    T = np.einsum("vj,jab->vab", W, A)
    vh = np.concatenate([v_rest, np.ones((v_rest.shape[0], 1))], axis=1)
    return np.einsum("vab,vb->va", T, vh)[:, :3]


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def outside_frac(J, V, F):
    return np.abs(winding_number(J, V, F)) < 0.5


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_articulation_out.txt")
    fout = open(outp, "w")
    fh = Tee(sys.stdout, fout)
    print(__doc__, file=fh)

    mdl = os.path.join(ROOT, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
    dd = load_model(mdl)
    v_t = np.asarray(dd["v_template"], dtype=np.float64)
    F = np.asarray(dd["f"]).astype(int)
    W = np.asarray(dd["weights"], dtype=np.float64)
    Jreg = np.asarray(dd["J_regressor"], dtype=np.float64)
    parents = np.asarray(dd["kintree_table"][0]).astype(int)
    names = list(dd["J_names"])
    groups = np.array([joint_group(n) for n in names])
    J0 = joint_regress(v_t[None], Jreg)
    nJ = J0.shape[1]
    print(f"template: V={v_t.shape} F={F.shape} joints={nJ}", file=fh)

    # poses from the two corpora
    dW = np.load(os.path.join(ROOT, "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz"), allow_pickle=True)
    dC = np.load(CLEAN_NPZ, allow_pickle=True)
    thW = np.concatenate([np.zeros_like(dW["global_rot"])[:, None, :], dW["joint_rot"]], axis=1).astype(np.float64)
    thC = np.concatenate([np.zeros_like(dC["global_rot"])[:, None, :], dC["joint_rot"]], axis=1).astype(np.float64)
    print(
        f"pose magnitude (mean |theta_j| over 54 joints, rad): "
        f"worker={np.linalg.norm(thW[:, 1:], axis=2).mean():.4f}  "
        f"clean={np.linalg.norm(thC[:, 1:], axis=2).mean():.4f}",
        file=fh,
    )

    def run_arm(theta_set, tag):
        """theta_set (N,J,3). Returns per-pose outside fraction, groupwise means."""
        outs, mags = [], []
        gm = {g: [] for g in sorted(set(groups))}
        for k in range(theta_set.shape[0]):
            th = theta_set[k][None]
            Rs = rodrigues(th.reshape(-1, 3)).reshape(1, nJ, 3, 3)
            Jp, A = global_rigid(Rs, J0, parents)
            V = lbs(v_t, A[0], W)
            o = outside_frac(Jp[0], V, F)
            outs.append(o.mean())
            mags.append(np.linalg.norm(theta_set[k][1:], axis=1).mean())
            for g in gm:
                gm[g].append(o[groups == g].mean())
        outs = np.array(outs)
        print(
            f"   {tag:<34} %joints outside OWN posed mesh: mean={100 * outs.mean():5.1f}  "
            f"median={100 * np.median(outs):5.1f}  p90={100 * np.percentile(outs, 90):5.1f}   "
            f"(mean|theta|={np.mean(mags):.3f} rad)",
            file=fh,
        )
        return outs, np.array(mags), {g: np.mean(v) for g, v in gm.items()}

    print("\n--- ARM A: theta = 0 (rest pose) --------------------------------------", file=fh)
    a_out, _, a_g = run_arm(np.zeros((1, nJ, 3)), "rest pose (the published floor)")

    print("\n--- ARM B: theta = CLEAN corpus poses ---------------------------------", file=fh)
    b_out, b_mag, b_g = run_arm(thC, "clean poses on template")

    print("\n--- ARM C: theta = WORKER corpus poses --------------------------------", file=fh)
    c_out, c_mag, c_g = run_arm(thW, "worker poses on template")

    u, p = stats.mannwhitneyu(c_out, b_out)
    print(
        f"\n   ARM C vs ARM B: {100 * c_out.mean():.1f}% vs {100 * b_out.mean():.1f}%  "
        f"U={u:.0f}  p={p:.3g}   PLACEMENT ERROR IS ZERO IN BOTH ARMS",
        file=fh,
    )
    print(f"   {'group':<14}{'A rest':>9}{'B clean':>9}{'C worker':>10}{'C-B pp':>9}", file=fh)
    for g in sorted(set(groups)):
        print(
            f"   {g:<14}{100 * a_g[g]:9.1f}{100 * b_g[g]:9.1f}{100 * c_g[g]:10.1f}{100 * (c_g[g] - b_g[g]):9.1f}",
            file=fh,
        )

    print("\n--- ARM D: dose-response, theta = c * worker pose ---------------------", file=fh)
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        o, m, _ = run_arm(c * thW, f"c={c:.2f}")
    s, ps = stats.spearmanr(np.concatenate([b_mag, c_mag]), np.concatenate([b_out, c_out]))
    print(f"\n   spearman(mean|theta|, %outside) pooled over B+C = {s:+.3f}  p={ps:.3g}", file=fh)
    s2, ps2 = stats.spearmanr(c_mag, c_out)
    print(f"   spearman within worker poses only = {s2:+.3f}  p={ps2:.3g}", file=fh)

    # ---------------- second check: is the probe-23 floor valid? -------------------
    print("\n" + "=" * 100, file=fh)
    print("SECOND CHECK: J_reg . posed_verts against its OWN fitted mesh", file=fh)
    for tag, npz, m, align in (
        (
            "LIM_0 (worker)",
            os.path.join(ROOT, "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz"),
            os.path.join(ROOT, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"),
            False,
        ),
        ("ALL_ANTS_CLEAN", CLEAN_NPZ, os.path.join(ROOT, "3D_model_prep/SMPL_fit.pkl"), True),
    ):
        R = load_run(npz, m, align=align)
        Jr = np.asarray(R["dd"]["J_regressor"], dtype=np.float64)
        Ff = np.asarray(R["dd"]["f"]).astype(int)
        Jf, _ = global_rigid(R["Rs"], R["J"], R["parents"], R["logscale"], R["betas_trans"])
        Jf = Jf + R["trans"][:, None, :]
        oref, ofit, dist = [], [], []
        for i in range(R["n"]):
            fv = R["d"]["verts"][i].astype(np.float64)
            jr = Jr @ fv
            diag = np.linalg.norm(fv.max(0) - fv.min(0))
            oref.append(np.abs(winding_number(jr, fv, Ff)) < 0.5)
            ofit.append(np.abs(winding_number(Jf[i], fv, Ff)) < 0.5)
            dist.append(np.linalg.norm(jr - Jf[i], axis=1) / diag)
        oref, ofit, dist = np.array(oref), np.array(ofit), np.array(dist)
        print(
            f"   {tag:<16} J_reg.posed_verts outside OWN mesh = {100 * oref.mean():5.1f}%   "
            f"fitted skeleton outside OWN mesh = {100 * ofit.mean():5.1f}%   "
            f"median |J_ref - J_fit| = {100 * np.median(dist):.3f}%diag",
            file=fh,
        )

    fh.flush()
    fout.close()
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
