"""PR #98 review probe: does `w_limit: 100.0` in the shipped ants_cfg.yaml actually bind?

PR #98 adds a joint-rotation hinge prior to `fitter_3d` and, in the same change, sets
`w_limit: 100.0` in `fitter_3d/ants_cfg.yaml` Stage_1_default -- ACTIVE, not commented.
The PR body states the feature is "off by default (0.0)" and that the production model is
unaffected because `SMIL_OmniAnt.pkl` carries no `joint_limits`. Both statements were true
when the PR was written. Neither is true against `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`,
which authors limits on 33 of 55 joints (99 of 165 axes, median half-width 20 deg).

So the question this probe answers is empirical, not stylistic:

  Taking 81 registrations that were ALREADY ACCEPTED as good (the ALL_ANTS_CLEAN run that
  the OmniAnt shape space itself was built from), how much of their pose would the new
  limits penalise, and how does `w_limit * loss_limit` compare with the surface terms it
  would be competing against?

If the answer is "a lot", then merging the yaml line as-is silently changes what the
registration pipeline optimises the moment the new model is pointed at -- which is a
different decision from merging the mechanism.

Reference scale: Stage_1_default in ants_cfg.yaml uses w_chamfer=1.0 over a chamfer that
runs 0.005-0.045 in these units (measured from diagnostics/moonshot/runs/*.log).

Reads only; writes nothing outside diagnostics/.
"""

import os
import pickle

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.join(REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl")
FIT_DIR = "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING"
CHAMFER_REF = (0.005, 0.045)  # w_chamfer=1.0, so this IS the chamfer term's contribution
W_LIMIT = 100.0  # the value the PR puts in ants_cfg.yaml Stage_1_default


def load_pkl(path):
    with open(path, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def hinge(joint_rot, min_l, max_l):
    """Exactly Stage.forward()'s term: mean over EVERY (N_POSE, 3) entry, violated or not."""
    return np.maximum(joint_rot - max_l, 0.0) + np.maximum(min_l - joint_rot, 0.0)


def main():
    dd = load_pkl(MODEL)
    jn = list(dd["J_names"])
    jl = np.asarray(dd["joint_limits"], dtype=np.float64)

    # Mirror _joint_limit_tensors_from_dd: root's 3 axes come first and are sliced off.
    min_l = jl[1:, :, 0]  # (N_POSE, 3)
    max_l = jl[1:, :, 1]
    n_pose = min_l.shape[0]
    print(f"[probe] model {os.path.basename(MODEL)}: J={len(jn)}, N_POSE={n_pose}")
    wide = np.isclose(np.abs(min_l), np.pi) & np.isclose(np.abs(max_l), np.pi)
    print(
        f"[probe] constrained axes {int((~wide).sum())}/{wide.size}, "
        f"median half-width {np.degrees(np.median((max_l - min_l)[~wide]) / 2):.1f} deg"
    )

    for stage in ["Stage_1_default", "Stage_3_deform_fine"]:
        p = os.path.join(FIT_DIR, f"{stage}.npz")
        if not os.path.isfile(p):
            print(f"[probe] SKIP {stage}: not found")
            continue
        jr = np.load(p)["joint_rot"].astype(np.float64)  # (n, N_POSE, 3)
        n = jr.shape[0]
        assert jr.shape[1:] == (n_pose, 3), f"{jr.shape} vs {(n_pose, 3)}"

        h = hinge(jr, min_l, max_l)  # (n, N_POSE, 3)
        loss = h.reshape(n, -1).mean(1)  # what Stage.forward() computes, per specimen
        viol = h > 0

        print()
        print("=" * 78)
        print(f"{stage}  --  {n} registrations already accepted as good")
        print("=" * 78)
        print(
            f"  axes in violation           {viol.sum(axis=(1, 2)).mean():.1f} / {n_pose * 3} "
            f"per specimen (min {viol.sum(axis=(1, 2)).min()}, max {viol.sum(axis=(1, 2)).max()})"
        )
        print(f"  specimens with >=1 violation {int((viol.any(axis=(1, 2))).sum())} / {n}")
        v = h[viol]
        if v.size:
            print(
                f"  violation magnitude          median {np.degrees(np.median(v)):.1f} deg, "
                f"p90 {np.degrees(np.percentile(v, 90)):.1f} deg, max {np.degrees(v.max()):.1f} deg"
            )
        print(f"  loss_limit (mean over all)   median {np.median(loss):.5f}, max {loss.max():.5f}")
        print(
            f"  w_limit * loss_limit @ {W_LIMIT:g}   median {W_LIMIT * np.median(loss):.3f}, "
            f"max {W_LIMIT * loss.max():.3f}"
        )
        print(f"  chamfer term for comparison  {CHAMFER_REF[0]:.3f} - {CHAMFER_REF[1]:.3f}")
        ratio = W_LIMIT * np.median(loss) / CHAMFER_REF[1]
        print(f"  -> limit term is {ratio:.0f}x the LARGEST chamfer contribution")

        # which joints carry it
        per_joint = viol.sum(axis=(0, 2))
        order = np.argsort(-per_joint)
        top = [(jn[i + 1], int(per_joint[i])) for i in order[:8] if per_joint[i] > 0]
        print(f"  worst joints (violating specimen-axes): {top}")

        # a fair counter-check: is this just a few outlier specimens?
        print(
            f"  loss_limit percentiles       p10 {np.percentile(loss, 10):.5f}  "
            f"p50 {np.median(loss):.5f}  p90 {np.percentile(loss, 90):.5f}"
        )


if __name__ == "__main__":
    main()
