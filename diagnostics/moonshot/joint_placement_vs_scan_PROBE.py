"""PROBE 21 -- score fitted joint placement against the RAW TARGET SCAN.

This is the only proxy here that involves no template, no self-consistency and no biological
assumption beyond "a skeletal joint lies inside the animal".

Target scans are put into the fitted frame by replaying fitter_3d/utils.py:340-344 exactly:
    centre = verts.mean(0); verts -= centre; scale = max(|verts|.max(0)); verts /= scale
A gate measures fitted-mesh -> target-scan chamfer; if the frames did not match the gate
would blow up and every number below would be meaningless.

Metrics per joint, against the target scan only:
  gwn   generalized winding number wrt the scan (validated on the scan itself first)
  gap   distance to the nearest scan point, as % of the scan bbox diagonal
  cen   surface centrality  || mean_k(p) - j || / mean_k(||p - j||)  over the k nearest scan
        points. 0 = the scan wraps around the joint (joint is inside the limb);
        1 = every nearby scan point is on the same side (joint is beside the limb, in air).
        Scale free and orientation free, so it is comparable between corpora with different
        scan densities -- unlike gwn, which needs a closed surface.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
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
    (
        "baseline (worker, stock)",
        "diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
        False,
        "diagnostics/moonshot/bench50",
    ),
    ("ALL_ANTS_CLEAN (81)", CLEAN_NPZ, "3D_model_prep/SMPL_fit.pkl", True, "diagnostics/moonshot/clean81"),
]

K = 128  # neighbours for the centrality statistic
N_SPEC = int(os.environ.get("N_SPEC", "0")) or None  # 0 -> all


def load_obj_np(path):
    V, F = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
                F.append(idx)
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=int)


def normalise(V):
    """Replays fitter_3d/utils.py:340-344."""
    V = V - V.mean(0)
    return V / np.abs(V).max(0).max()


def posed_skeleton(R):
    J, _ = global_rigid(R["Rs"], R["J"], R["parents"], R["logscale"], R["betas_trans"])
    return J + R["trans"][:, None, :]


def gwn_selfcheck(V, F, rng, n=200):
    """Validate the winding number on this (possibly non-watertight) scan.

    Offsets face centroids along +/- the face normal by 1e-3 of the bbox diagonal and checks
    that GWN separates them. Returns (median_inside, median_outside, separation_ok).
    """
    idx = rng.choice(F.shape[0], size=min(n, F.shape[0]), replace=False)
    tri = V[F[idx]]
    cen = tri.mean(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nl = np.linalg.norm(nrm, axis=1, keepdims=True)
    keep = nl[:, 0] > 0
    cen, nrm, nl = cen[keep], nrm[keep], nl[keep]
    nrm = nrm / nl
    eps = 1e-3 * np.linalg.norm(V.max(0) - V.min(0))
    w_out = winding_number(cen + eps * nrm, V, F)
    w_in = winding_number(cen - eps * nrm, V, F)
    mi, mo = float(np.median(np.abs(w_in))), float(np.median(np.abs(w_out)))
    if mi < mo:  # normals point inward -> swap
        mi, mo = mo, mi
    return mi, mo, (mi > 0.8 and mo < 0.2)


def centrality(J, V, tree, k=K):
    d, nb = tree.query(J, k=k)
    return np.linalg.norm(V[nb].mean(1) - J, axis=1) / d.mean(1)


def template_floor(mdl, align, fh):
    """The floor of every joint-vs-surface metric: the AUTHORED skeleton in the AUTHORED mesh.

    If the template's own joints score X, no fit can be expected to beat X, and a metric
    whose floor is already high is not a usable discriminator.
    """
    from joint_placement_common import load_model, align_template, joint_regress

    dd = load_model(mdl)
    V = np.asarray(dd["v_template"], dtype=np.float64)
    if align:
        V = align_template(V, dd["sym_verts"])
    F = np.asarray(dd["f"]).astype(int)
    names = list(dd["J_names"])
    Jr = joint_regress(V[None], np.asarray(dd["J_regressor"], dtype=np.float64))[0]
    Ja = np.asarray(dd["J"], dtype=np.float64)
    diag = np.linalg.norm(V.max(0) - V.min(0))
    tree = cKDTree(V)
    groups = np.array([joint_group(nm) for nm in names])
    print(f"\n--- TEMPLATE FLOOR  {os.path.basename(mdl)} (align={align})  bbox diag={diag:.4f}", file=fh)
    for lbl, J in (("J_regressor.v_template", Jr), ("authored dd['J']", Ja)):
        w = winding_number(J, V, F)
        gap, _ = tree.query(J)
        cen = centrality(J, V, tree)
        out = np.abs(w) < 0.5
        print(
            f"    {lbl:<24} %outside={100 * out.mean():5.1f}  med gap={100 * np.median(gap) / diag:6.3f}%diag"
            f"  med centrality={np.median(cen):.3f}  p90={np.percentile(cen, 90):.3f}",
            file=fh,
        )
        if lbl.startswith("J_reg"):
            for g in sorted(set(groups)):
                m = groups == g
                print(
                    f"        {g:<14} %outside={100 * out[m].mean():5.1f}  "
                    f"med gap={100 * np.median(gap[m]) / diag:6.3f}%diag  "
                    f"med centrality={np.median(cen[m]):.3f}",
                    file=fh,
                )


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_vs_scan_out.txt")
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        rng = np.random.default_rng(0)
        summary = []
        for mdl, align in [
            ("3D_model_prep/OmniAnt_25PCs_joint_limited.pkl", False),
            ("3D_model_prep/SMIL_OmniAnt.pkl", False),
            ("3D_model_prep/SMPL_fit.pkl", True),
        ]:
            template_floor(os.path.join(ROOT, mdl), align, fh)
        fh.flush()

        for tag, npz, mdl, align, scan_dir in RUNS:
            npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
            mdl = os.path.join(ROOT, mdl)
            scan_dir = os.path.join(ROOT, scan_dir)
            R = load_run(npz, mdl, align=align)
            names = R["J_names"]
            Jp = posed_skeleton(R)
            labels = [str(x) for x in R["d"]["labels"]]
            n = R["n"] if N_SPEC is None else min(R["n"], N_SPEC)
            print(f"\n### {tag}   n={R['n']} (using {n})   scans={scan_dir}", file=fh)

            avail = {
                os.path.splitext(x)[0]: os.path.join(scan_dir, x) for x in os.listdir(scan_dir) if x.endswith(".obj")
            }
            gwn_all, gap_all, cen_all, chamf, gwn_ok = [], [], [], [], []
            cen_fit = []  # same centrality statistic, but against the FITTED mesh
            missing = 0
            for i in range(n):
                key = os.path.splitext(labels[i])[0]
                if key not in avail:
                    missing += 1
                    continue
                V, F = load_obj_np(avail[key])
                V = normalise(V)
                diag = np.linalg.norm(V.max(0) - V.min(0))
                tree = cKDTree(V)
                # --- frame gate: fitted mesh -> scan one-sided chamfer
                fv = R["d"]["verts"][i].astype(np.float64)
                dfv, _ = tree.query(fv)
                chamf.append(dfv.mean() / diag)
                # --- gwn validity on this scan
                if i < 5:
                    mi, mo, ok = gwn_selfcheck(V, F, rng)
                    gwn_ok.append((mi, mo, ok))
                # --- per joint
                j = Jp[i]
                w = winding_number(j, V, F)
                d, nb = tree.query(j, k=K)
                cen = np.linalg.norm(V[nb].mean(1) - j, axis=1) / d.mean(1)
                gap, _ = tree.query(j)
                gwn_all.append(w)
                gap_all.append(gap / diag)
                cen_all.append(cen)
                ftree = cKDTree(fv)
                cen_fit.append(centrality(j, fv, ftree))

            if missing:
                print(f"   WARNING: {missing} labels had no matching .obj in {scan_dir}", file=fh)
            if not gwn_all:
                print("   nothing measured", file=fh)
                continue
            gwn_all = np.array(gwn_all)
            gap_all = np.array(gap_all)
            cen_all = np.array(cen_all)
            chamf = np.array(chamf)

            print(
                f"   FRAME GATE fitted->scan one-sided chamfer (mean NN dist / scan diag): "
                f"median={100 * np.median(chamf):.3f}%  max={100 * chamf.max():.3f}%   "
                f"{'OK' if np.median(chamf) < 0.02 else '*** FRAME MISMATCH ***'}",
                file=fh,
            )
            oks = [o for _, _, o in gwn_ok]
            print(
                f"   GWN VALIDITY on scans (first {len(gwn_ok)}): median |w| just inside="
                f"{np.mean([a for a, _, _ in gwn_ok]):.3f}  just outside="
                f"{np.mean([b for _, b, _ in gwn_ok]):.3f}  usable={sum(oks)}/{len(oks)}",
                file=fh,
            )

            outside = np.abs(gwn_all) < 0.5
            groups = np.array([joint_group(nm) for nm in names])
            print(
                f"   joints OUTSIDE the scan: {100 * outside.mean():.1f}%   "
                f"per specimen mean={outside.sum(1).mean():.1f} of {len(names)}",
                file=fh,
            )
            print(
                f"   {'group':<14}{'n':>4}{'%out(gwn)':>11}{'med gap%':>10}{'p90 gap%':>10}"
                f"{'med centr':>11}{'p90 centr':>11}{'%centr>0.9':>12}",
                file=fh,
            )
            rows = {}
            for g in sorted(set(groups)):
                m = groups == g
                rows[g] = (
                    100 * outside[:, m].mean(),
                    100 * np.median(gap_all[:, m]),
                    100 * np.percentile(gap_all[:, m], 90),
                    np.median(cen_all[:, m]),
                    np.percentile(cen_all[:, m], 90),
                    100 * (cen_all[:, m] > 0.9).mean(),
                )
                print(
                    f"   {g:<14}{m.sum():4d}{rows[g][0]:11.1f}{rows[g][1]:10.3f}{rows[g][2]:10.3f}"
                    f"{rows[g][3]:11.3f}{rows[g][4]:11.3f}{rows[g][5]:12.1f}",
                    file=fh,
                )
            print(
                f"   {'ALL':<14}{len(names):4d}{100 * outside.mean():11.1f}"
                f"{100 * np.median(gap_all):10.3f}{100 * np.percentile(gap_all, 90):10.3f}"
                f"{np.median(cen_all):11.3f}{np.percentile(cen_all, 90):11.3f}"
                f"{100 * (cen_all > 0.9).mean():12.1f}",
                file=fh,
            )
            cen_fit = np.array(cen_fit)
            print(
                f"   centrality vs FITTED mesh (self-consistency variant): "
                f"median={np.median(cen_fit):.3f}  p90={np.percentile(cen_fit, 90):.3f}  "
                f"%>0.9={100 * (cen_fit > 0.9).mean():.1f}",
                file=fh,
            )
            per_joint = outside.mean(0)
            order = np.argsort(-per_joint)[:10]
            print(
                "   worst joints (%% outside scan): "
                + ", ".join(f"{names[j]}={100 * per_joint[j]:.0f}" for j in order),
                file=fh,
            )
            summary.append(
                (tag, 100 * outside.mean(), 100 * np.median(gap_all), np.median(cen_all), 100 * (cen_all > 0.9).mean())
            )
            fh.flush()

        print("\n" + "=" * 92, file=fh)
        print("SUMMARY (vs raw target scan)", file=fh)
        print(
            f"{'run':<28}{'%joints outside':>17}{'med gap %diag':>15}{'med centrality':>16}{'% centrality>0.9':>18}",
            file=fh,
        )
        for r in summary:
            print(f"{r[0]:<28}{r[1]:17.1f}{r[2]:15.3f}{r[3]:16.3f}{r[4]:18.1f}", file=fh)
        print("=" * 92, file=fh)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
