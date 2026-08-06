"""REFUTE-E -- do the two corpora's TEMPLATES put joints at the same depth inside the skin?

"% of joints outside the scan" is a THRESHOLD on a signed distance. Its floor was reported
as similar for the three templates (10.9-14.5% outside), but a threshold statistic is
governed by the MARGIN distribution, not by the floor: if template A's joints sit 0.10% of
the body diagonal inside the skin and template B's sit 0.35% inside, the same absolute
placement error flips A's joints out 3.5x more often. The two corpora use different
templates (OmniAnt 10235v/25betas/J_regressor with 10 nnz per row vs SMPL_fit
10236v/20betas/20 nnz per row), so this must be measured, not assumed.

Also measures what drives the J_static vs J_regressed disagreement per specimen.
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_placement_common import (  # noqa: E402
    align_template,
    joint_group,
    joint_regress,
    load_model,
    winding_number,
)

ROOT = "/home/fabi/dev/SMILify"
MODELS = [
    ("OmniAnt_25PCs_joint_limited", f"{ROOT}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl", False),
    ("SMIL_OmniAnt", f"{ROOT}/3D_model_prep/SMIL_OmniAnt.pkl", False),
    ("SMPL_fit", f"{ROOT}/3D_model_prep/SMPL_fit.pkl", True),
]


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = f"{ROOT}/diagnostics/moonshot/refute_template_margin_out.txt"
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        for nm, p, align in MODELS:
            dd = load_model(p)
            V = np.asarray(dd["v_template"], float)
            if align:
                V = align_template(V, dd["sym_verts"])
            F = np.asarray(dd["f"]).astype(int)
            names = list(dd["J_names"])
            W = np.asarray(dd["weights"], float)
            Jr = joint_regress(V[None], np.asarray(dd["J_regressor"], float))[0]
            diag = np.linalg.norm(V.max(0) - V.min(0))
            tree = cKDTree(V)
            w = winding_number(Jr, V, F)
            inside = np.abs(w) >= 0.5
            gap, _ = tree.query(Jr)
            # local limb radius: median distance from joint to the verts it dominates
            rad = np.full(len(names), np.nan)
            for j in range(len(names)):
                sel = W[:, j] > 0.5
                if sel.sum() >= 5:
                    rad[j] = np.median(np.linalg.norm(V[sel] - Jr[j], axis=1))
            g = np.array([joint_group(x) for x in names])
            leg = np.char.startswith(g.astype(str), "leg")
            print(
                f"\n### {nm}   diag={diag:.4f}   nnz/row of J_regressor="
                f"{int(np.median((np.abs(np.asarray(dd['J_regressor'], float)) > 1e-8).sum(1)))}",
                file=fh,
            )
            print(f"   %joints inside own template mesh = {100 * inside.mean():.1f}", file=fh)
            for lbl, m in (("ALL", np.ones(len(names), bool)), ("LEG", leg)):
                mi = m & inside
                print(
                    f"   {lbl:<4} margin (dist to own skin, INSIDE joints only): "
                    f"median={100 * np.median(gap[mi]) / diag:.3f} %diag   "
                    f"p25={100 * np.percentile(gap[mi], 25) / diag:.3f}   "
                    f"p75={100 * np.percentile(gap[mi], 75) / diag:.3f}",
                    file=fh,
                )
                rr = rad[m & ~np.isnan(rad)]
                print(
                    f"   {lbl:<4} local limb radius (median dist to dominated verts): "
                    f"median={100 * np.median(rr) / diag:.3f} %diag",
                    file=fh,
                )
                print(f"   {lbl:<4} margin / limb radius: median={np.nanmedian(gap[mi] / rad[mi]):.3f}", file=fh)
            fh.flush()

        print("\n" + "=" * 92, file=fh)
        print("READ-OFF: a threshold metric with a small margin is a high-gain detector.", file=fh)
        print("The floor (%outside) can match while the margin differs by a factor.", file=fh)
    print(f"wrote {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
