"""PROBE 23 -- ALTERNATIVE-CAUSE test of the claim "61.5% vs 30.1% joints-outside-scan
proves worker joint PLACEMENT is worse".

The claim (probe 21) compares two corpora that differ in far more than fit quality:

  * bench50   = 50 raw CT worker scans from antscan_proofread_castes/worker
                (~50k verts each, real tarsi/spines/hairs)
  * clean81   = ALL_ANTS_CLEAN, the corpus OmniAnt's own shape space was built from
                (preprocess_meshes.py says so explicitly), ~5k verts each

"Fraction of joints OUTSIDE the surface" is not a scale-free quantity. A joint is outside iff
its placement error exceeds the LOCAL HALF-THICKNESS of the body part it sits in. So the same
metric will report a higher number on a corpus with thinner limbs / finer scans EVEN IF THE
PLACEMENT ERROR IS IDENTICAL. Probe 21's floor (10.9-14.5%) was measured on the smooth
authored TEMPLATE mesh, never on the actual worker scans, so it cannot control for this.

WHAT THIS PROBE MEASURES

 F1  CORPUS-SPECIFIC FLOOR.  J_regressor rows are non-negative and sum to 1 (verified), so
     J_ref = J_regressor . (posed fitted verts, deform included) is a skeleton that is
     PERFECTLY CONSISTENT WITH THE FITTED SURFACE by construction -- it is a convex
     combination of the very surface vertices that match the scan. Its %-outside against the
     raw scan is the floor for THAT scan, with placement error removed by construction.

 F2  LOCAL LIMB HALF-THICKNESS.  Shrinking-ball medial radius T at the scan point nearest each
     joint (Ma et al.); T/diag is the error budget available at that joint. Then
       rho = |j - c_medial| / T   (rho<1 == inside the maximal inscribed ball)
     is the scale-free placement error, and P(T < eps) is the outside-fraction a UNIFORM
     placement error of eps would produce on that corpus.

 F3  EQUIVALENT-ERROR INVERSION.  Solve P(T < eps) = observed outfrac in each corpus. If the
     two corpora need the SAME eps, thinness alone explains the 61.5 vs 30.1 split.

 F4  ARTICULATION / CONTRACTION, and whether it predicts outfrac within a corpus.

 F5  GENUS-MATCHED subset (both corpora contain Acanthostichus, Acromyrmex, Cephalotes, ...).

 F6  deform_verts magnitude relative to T -- deform_verts is added AFTER skinning, so if it is
     of order the limb radius the surface leaves the bones behind and "joint outside scan"
     measures the surface offset, not the joint.
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
    (
        "baseline (worker, stock)",
        "diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
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


def posed_skeleton(R):
    J, _ = global_rigid(R["Rs"], R["J"], R["parents"], R["logscale"], R["betas_trans"])
    return J + R["trans"][:, None, :]


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    nl = np.linalg.norm(vn, axis=1, keepdims=True)
    nl[nl == 0] = 1.0
    return vn / nl


def medial_radius(P, N, V, cap):
    """Shrinking-ball medial radius at surface points P with INWARD unit normals N.

    r = min over surface samples q of |q-p|^2 / (2 (q-p).n)  restricted to (q-p).n > 0.
    Returns (r, centre).  cap bounds r for points where the ball never closes.
    """
    r = np.full(P.shape[0], cap)
    for i in range(P.shape[0]):
        d = V - P[i]
        num = np.einsum("ij,ij->i", d, d)
        den = 2.0 * (d @ N[i])
        m = (den > 1e-12) & (num > 1e-18)
        if m.any():
            r[i] = min(cap, float(np.min(num[m] / den[m])))
    return r, P + r[:, None] * N


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def genus(label):
    b = os.path.splitext(str(label))[0]
    b = b.split("_CASENT")[0].split("_OKENT")[0]
    b = b.replace("_", "-").split("-")[0].split(" ")[0]
    return b.lower()


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_altcause_out.txt")
    fout = open(outp, "w")
    fh = Tee(sys.stdout, fout)
    print(__doc__, file=fh)

    store = {}
    for tag, npz, mdl, align, scan_dir in RUNS:
        npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
        R = load_run(npz, os.path.join(ROOT, mdl), align=align)
        names = R["J_names"]
        Jp = posed_skeleton(R)
        J_reg = np.asarray(R["dd"]["J_regressor"], dtype=np.float64)
        labels = [str(x) for x in R["d"]["labels"]]
        sd = os.path.join(ROOT, scan_dir)
        avail = {os.path.splitext(x)[0]: os.path.join(sd, x) for x in os.listdir(sd) if x.endswith(".obj")}
        print(f"\n{'=' * 100}\n### {tag}   n={R['n']}   scans={scan_dir}", file=fh)

        rec = dict(tag=tag, names=names, rows=[])
        for i in range(R["n"]):
            key = os.path.splitext(labels[i])[0]
            if key not in avail:
                continue
            V, F = load_obj_np(avail[key])
            V = normalise(V)
            diag = float(np.linalg.norm(V.max(0) - V.min(0)))
            tree = cKDTree(V)

            j_fit = Jp[i]  # fitted skeleton
            fv = R["d"]["verts"][i].astype(np.float64)  # fitted mesh (deform incl)
            j_ref = J_reg @ fv  # surface-consistent floor

            w_fit = winding_number(j_fit, V, F)
            w_ref = winding_number(j_ref, V, F)
            gap_fit, nn_fit = tree.query(j_fit)
            gap_ref, _ = tree.query(j_ref)

            # ---- local medial radius at the scan point nearest each fitted joint
            VN = vertex_normals(V, F)
            P = V[nn_fit]
            n0 = VN[nn_fit]
            eps = 1e-3 * diag
            w_probe = winding_number(P - eps * n0, V, F)
            # q = P - eps*n0 inside  =>  the INWARD direction is -n0
            inward = np.where(np.abs(w_probe) > 0.5, -1.0, 1.0)[:, None]
            Nin = n0 * inward
            T, C = medial_radius(P, Nin, V, cap=0.25 * diag)
            rho = np.linalg.norm(j_fit - C, axis=1) / np.maximum(T, 1e-9)

            # ---- deform_verts magnitude near each joint (support of the regressor row)
            dvn = np.linalg.norm(R["d"]["deform_verts"][i].astype(np.float64), axis=1)
            dv_at_joint = J_reg @ dvn

            # ---- articulation / contraction
            spread = float(np.linalg.norm(V - V.mean(0), axis=1).mean() / diag)
            legrot = float(np.linalg.norm(R["theta"][i, 1:], axis=1).mean())

            fchamf = float(tree.query(fv)[0].mean() / diag)
            rec["rows"].append(
                dict(
                    label=labels[i],
                    diag=diag,
                    nv=len(V),
                    nf=len(F),
                    out_fit=np.abs(w_fit) < 0.5,
                    out_ref=np.abs(w_ref) < 0.5,
                    gap_fit=gap_fit / diag,
                    gap_ref=gap_ref / diag,
                    T=T / diag,
                    rho=rho,
                    dv=dv_at_joint / diag,
                    spread=spread,
                    legrot=legrot,
                    chamf=fchamf,
                    nnsp=float(np.median(tree.query(V, k=2)[0][:, 1]) / diag),
                )
            )
            if (i + 1) % 10 == 0:
                print(f"    ... {i + 1}", file=fh)
                fh.flush()
        store[tag] = rec

        rows = rec["rows"]
        of_fit = np.array([r["out_fit"].mean() for r in rows])
        of_ref = np.array([r["out_ref"].mean() for r in rows])
        Tall = np.concatenate([r["T"] for r in rows])
        rho_all = np.concatenate([r["rho"] for r in rows])
        print(
            f"  scans: nv median={np.median([r['nv'] for r in rows]):.0f}  "
            f"nf median={np.median([r['nf'] for r in rows]):.0f}  "
            f"NN spacing median={100 * np.median([r['nnsp'] for r in rows]):.3f}%diag",
            file=fh,
        )
        print(
            f"  F1 %joints outside scan: FITTED skeleton={100 * of_fit.mean():.1f}   "
            f"SURFACE-CONSISTENT floor J_reg.posed_verts={100 * of_ref.mean():.1f}   "
            f"excess={100 * (of_fit.mean() - of_ref.mean()):.1f}pp   "
            f"ratio={of_fit.mean() / max(of_ref.mean(), 1e-9):.2f}",
            file=fh,
        )
        print(
            f"  F2 local medial radius T/diag: median={100 * np.median(Tall):.3f}%  "
            f"q25={100 * np.percentile(Tall, 25):.3f}%  q75={100 * np.percentile(Tall, 75):.3f}%",
            file=fh,
        )
        print(
            f"  F2 rho=|j-c_medial|/T : median={np.median(rho_all):.3f}  "
            f"p75={np.percentile(rho_all, 75):.3f}  %rho>1={100 * (rho_all > 1).mean():.1f}",
            file=fh,
        )
        print(
            f"  F6 |deform_verts| at joint / diag: median={100 * np.median(np.concatenate([r['dv'] for r in rows])):.3f}%"
            f"   ratio to T: median={np.median(np.concatenate([r['dv'] for r in rows]) / np.maximum(np.concatenate([r['T'] for r in rows]), 1e-9)):.3f}",
            file=fh,
        )
        print(
            f"  F4 scan spread (mean|v-c|/diag) median={np.median([r['spread'] for r in rows]):.4f}   "
            f"mean |joint rot| rad median={np.median([r['legrot'] for r in rows]):.4f}",
            file=fh,
        )
        groups = np.array([joint_group(n) for n in names])
        print(
            f"  {'group':<14}{'%out fit':>10}{'%out FLOOR':>12}{'excess pp':>11}{'T/diag %':>10}{'med rho':>9}", file=fh
        )
        for g in sorted(set(groups)):
            m = groups == g
            a = np.mean([r["out_fit"][m].mean() for r in rows]) * 100
            b = np.mean([r["out_ref"][m].mean() for r in rows]) * 100
            t = np.median(np.concatenate([r["T"][m] for r in rows])) * 100
            rr = np.median(np.concatenate([r["rho"][m] for r in rows]))
            print(f"  {g:<14}{a:10.1f}{b:12.1f}{a - b:11.1f}{t:10.3f}{rr:9.3f}", file=fh)
        fh.flush()

    np.save(os.path.join(ROOT, "diagnostics/moonshot/joint_placement_altcause.npy"), store, allow_pickle=True)

    # ================= cross-corpus =================
    W = store["LIM_0 (worker)"]["rows"]
    C = store["ALL_ANTS_CLEAN (81)"]["rows"]
    B = store["baseline (worker, stock)"]["rows"]
    print(f"\n{'=' * 100}\nCROSS-CORPUS", file=fh)

    def col(rows, k, agg=np.mean):
        return np.array([agg(r[k]) for r in rows])

    print("\n-- F1  does the SURFACE-CONSISTENT floor already reproduce the gap?", file=fh)
    for lbl, rows in (("worker LIM_0", W), ("worker baseline", B), ("clean", C)):
        f = col(rows, "out_fit") * 100
        r = col(rows, "out_ref") * 100
        print(
            f"   {lbl:<16} fitted={f.mean():5.1f}%  floor={r.mean():5.1f}%  excess={f.mean() - r.mean():5.1f}pp",
            file=fh,
        )
    u, p = stats.mannwhitneyu(col(W, "out_fit") - col(W, "out_ref"), col(C, "out_fit") - col(C, "out_ref"))
    print(
        f"   EXCESS over own floor, worker vs clean: {100 * np.mean(col(W, 'out_fit') - col(W, 'out_ref')):.1f}pp "
        f"vs {100 * np.mean(col(C, 'out_fit') - col(C, 'out_ref')):.1f}pp   U={u:.0f} p={p:.3g}",
        file=fh,
    )
    u, p = stats.mannwhitneyu(col(W, "out_ref"), col(C, "out_ref"))
    print(
        f"   FLOOR alone, worker vs clean: {100 * col(W, 'out_ref').mean():.1f}% vs "
        f"{100 * col(C, 'out_ref').mean():.1f}%  U={u:.0f} p={p:.3g}",
        file=fh,
    )

    print("\n-- F2/F3  equivalent-error inversion   P(T < eps) = observed outfrac", file=fh)
    Tw = np.concatenate([r["T"] for r in W])
    Tc = np.concatenate([r["T"] for r in C])
    Tb = np.concatenate([r["T"] for r in B])
    ofw, ofc = col(W, "out_fit").mean(), col(C, "out_fit").mean()
    ew = np.percentile(Tw, 100 * ofw)
    ec = np.percentile(Tc, 100 * ofc)
    print(f"   worker: outfrac={100 * ofw:.1f}%  -> eps={100 * ew:.3f}%diag", file=fh)
    print(f"   clean : outfrac={100 * ofc:.1f}%  -> eps={100 * ec:.3f}%diag", file=fh)
    print(f"   ratio eps_worker/eps_clean = {ew / ec:.2f}   (vs the claimed outfrac ratio {ofw / ofc:.2f})", file=fh)
    print(
        f"   median T/diag: worker={100 * np.median(Tw):.3f}%  clean={100 * np.median(Tc):.3f}%  "
        f"baseline-scans={100 * np.median(Tb):.3f}%   thinness ratio={np.median(Tc) / np.median(Tw):.2f}",
        file=fh,
    )
    print(
        "   COUNTERFACTUAL: outfrac the WORKER scans would show at the CLEAN corpus' eps: "
        f"{100 * (Tw < ec).mean():.1f}%   (observed worker {100 * ofw:.1f}%)",
        file=fh,
    )
    print(
        "   COUNTERFACTUAL: outfrac the CLEAN scans would show at the WORKER corpus' eps: "
        f"{100 * (Tc < ew).mean():.1f}%   (observed clean {100 * ofc:.1f}%)",
        file=fh,
    )

    print("\n-- F2  scale-free placement error rho = |j - medial centre| / T", file=fh)
    rw = np.array([np.median(r["rho"]) for r in W])
    rc = np.array([np.median(r["rho"]) for r in C])
    u, p = stats.mannwhitneyu(rw, rc)
    print(
        f"   per-specimen median rho: worker={np.median(rw):.3f}  clean={np.median(rc):.3f}  "
        f"ratio={np.median(rw) / np.median(rc):.2f}  U={u:.0f} p={p:.3g}",
        file=fh,
    )

    print("\n-- F4  contraction / articulation", file=fh)
    for k in ("spread", "legrot", "chamf", "nnsp"):
        a = np.array([r[k] for r in W])
        b = np.array([r[k] for r in C])
        u, p = stats.mannwhitneyu(a, b)
        print(
            f"   {k:<8} worker={np.median(a):.4f}  clean={np.median(b):.4f}  "
            f"ratio={np.median(a) / np.median(b):.2f}  p={p:.3g}",
            file=fh,
        )
    # within-corpus predictive power of thinness
    for lbl, rows in (("worker", W), ("clean", C)):
        of = col(rows, "out_fit")
        med_t = np.array([np.median(r["T"]) for r in rows])
        s, ps = stats.spearmanr(of, med_t)
        s2, ps2 = stats.spearmanr(of, [r["spread"] for r in rows])
        print(
            f"   within {lbl}: spearman(outfrac, median T) = {s:+.3f} (p={ps:.3g}); "
            f"spearman(outfrac, scan spread) = {s2:+.3f} (p={ps2:.3g})",
            file=fh,
        )

    print("\n-- F5  genus-matched subset", file=fh)
    gw = {}
    for r in W:
        gw.setdefault(genus(r["label"]), []).append(r)
    gc = {}
    for r in C:
        gc.setdefault(genus(r["label"]), []).append(r)
    shared = sorted(set(gw) & set(gc))
    print(f"   shared genera ({len(shared)}): {', '.join(shared)}", file=fh)
    aw, ac, tw2, tc2 = [], [], [], []
    for g in shared:
        a = np.mean([r["out_fit"].mean() for r in gw[g]])
        b = np.mean([r["out_fit"].mean() for r in gc[g]])
        ta = np.median(np.concatenate([r["T"] for r in gw[g]]))
        tb = np.median(np.concatenate([r["T"] for r in gc[g]]))
        aw.append(a)
        ac.append(b)
        tw2.append(ta)
        tc2.append(tb)
        print(
            f"     {g:<18} worker %out={100 * a:5.1f} (n={len(gw[g])})  "
            f"clean %out={100 * b:5.1f} (n={len(gc[g])})   T/diag {100 * ta:.3f}% vs {100 * tb:.3f}%",
            file=fh,
        )
    if aw:
        w2, p2 = stats.wilcoxon(aw, ac)
        print(
            f"   paired-by-genus: worker {100 * np.mean(aw):.1f}% vs clean {100 * np.mean(ac):.1f}%  "
            f"wilcoxon p={p2:.3g}; median T {100 * np.median(tw2):.3f}% vs {100 * np.median(tc2):.3f}%",
            file=fh,
        )

    fh.flush()
    fout.close()
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
