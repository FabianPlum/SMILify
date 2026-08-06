"""PROBE 26 -- is the inside/outside INSTRUMENT itself biased between the two corpora?

The whole claim rests on a binary produced by a generalized winding number thresholded at 0.5.
GWN equals +/-1 inside a CLOSED, consistently oriented surface. The worker scans are raw CT
isosurfaces; preprocess_meshes.py already measured, on 40 vs 40 meshes:

      vertex components (median)      clean 3       worker 34.5
      boundary-edge fraction          clean 0.0388  worker 0.0471

Holes and stray components make the solid angle leak, which pushes |w| DOWN and therefore
mislabels genuinely interior points as "outside". Probe 21's validity gate offsets face
centroids by 1e-3 of the diagonal along +/- the normal -- a purely LOCAL test, one thousandth
of the body, which a holey mesh passes trivially. It does not test the instrument at the depth
a joint actually sits at.

TEST 1  FALSE-OUTSIDE RATE ON GUARANTEED-INTERIOR POINTS
  Sample surface points, orient the normal inward (locally, by GWN), run the shrinking-ball
  medial estimate to get radius r, and probe at p + alpha*r*n for alpha in {0.5, 1.0}. Those
  points are interior by construction for any closed surface. Any of them that GWN reports as
  "outside" is an instrument error. Also report median |w| there: a watertight mesh gives 1.0.

TEST 2  MESH INTEGRITY per scan actually used (components, boundary edges).

TEST 3  ARTICULATION CONDITIONAL. Probe 25's ceiling-free statistic is P(fit outside | ref
  inside). Compute the same conditional in probe 24's zero-placement-error articulation arms:
  P(joint outside its own posed mesh | that joint was inside at rest). That is the part of the
  conditional that articulation alone contributes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy import stats
from joint_placement_common import load_model, joint_regress, global_rigid, rodrigues, joint_group, winding_number

ROOT = "/home/fabi/dev/SMILify"
NSAMP = 150


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


def integrity(V, F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    nc, _ = connected_components(A.maximum(A.T), directed=False)
    es = np.sort(e, axis=1)
    _, cnt = np.unique(es, axis=0, return_counts=True)
    return nc, float((cnt == 1).mean())


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    nl = np.linalg.norm(vn, axis=1, keepdims=True)
    nl[nl == 0] = 1.0
    return vn / nl


def medial(P, N, V, cap):
    r = np.full(P.shape[0], cap)
    for i in range(P.shape[0]):
        d = V - P[i]
        num = np.einsum("ij,ij->i", d, d)
        den = 2.0 * (d @ N[i])
        m = (den > 1e-12) & (num > 1e-18)
        if m.any():
            r[i] = min(cap, float(np.min(num[m] / den[m])))
    return r


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_instrument_out.txt")
    fout = open(outp, "w")
    fh = Tee(sys.stdout, fout)
    print(__doc__, file=fh)
    rng = np.random.default_rng(7)

    print("=" * 100, file=fh)
    print("TEST 1+2  instrument bias and mesh integrity  (all 50 worker / 81 clean scans)", file=fh)
    summ = {}
    for corpus, d in (("worker bench50", "diagnostics/moonshot/bench50"), ("clean81", "diagnostics/moonshot/clean81")):
        sd = os.path.join(ROOT, d)
        files = sorted(x for x in os.listdir(sd) if x.endswith(".obj"))
        fo5, fo10, wmed, ncs, bef, far = [], [], [], [], [], []
        for fn in files:
            V, F = load_obj_np(os.path.join(sd, fn))
            V = normalise(V)
            diag = float(np.linalg.norm(V.max(0) - V.min(0)))
            nc, be = integrity(V, F)
            ncs.append(nc)
            bef.append(be)
            VN = vertex_normals(V, F)
            idx = rng.choice(len(V), size=min(NSAMP, len(V)), replace=False)
            P, n0 = V[idx], VN[idx]
            eps = 1e-3 * diag
            wp = winding_number(P - eps * n0, V, F)
            # q = P - eps*n0 inside  =>  the INWARD direction is -n0
            inward = np.where(np.abs(wp) > 0.5, -1.0, 1.0)[:, None]
            N = n0 * inward
            r = medial(P, N, V, cap=0.25 * diag)
            ok = r > 3 * eps
            for alpha, acc in ((0.5, fo5), (1.0, fo10)):
                q = P[ok] + alpha * r[ok][:, None] * N[ok]
                w = winding_number(q, V, F)
                acc.append(float((np.abs(w) < 0.5).mean()))
                if alpha == 1.0:
                    wmed.append(float(np.median(np.abs(w))))
            far.append(float(np.abs(winding_number((V.max(0) + 5 * diag)[None], V, F))[0]))
        summ[corpus] = dict(
            fo5=np.array(fo5),
            fo10=np.array(fo10),
            w=np.array(wmed),
            nc=np.array(ncs),
            be=np.array(bef),
            far=np.array(far),
        )
        s = summ[corpus]
        print(f"\n  {corpus}   n={len(files)}", file=fh)
        print(
            f"    connected components   median={np.median(s['nc']):.0f}   "
            f"p90={np.percentile(s['nc'], 90):.0f}   max={s['nc'].max():.0f}",
            file=fh,
        )
        print(f"    boundary-edge fraction median={np.median(s['be']):.4f}", file=fh)
        print(f"    GWN far outside (should be 0) median={np.median(s['far']):.4f}  max={s['far'].max():.4f}", file=fh)
        print(
            f"    median |w| at the MEDIAL CENTRE (watertight -> 1.000) = "
            f"{np.median(s['w']):.3f}   worst specimen={s['w'].min():.3f}",
            file=fh,
        )
        print(
            f"    FALSE-OUTSIDE rate on guaranteed-interior probes: "
            f"alpha=0.5 -> {100 * np.mean(s['fo5']):.1f}%   alpha=1.0 -> {100 * np.mean(s['fo10']):.1f}%",
            file=fh,
        )
        fh.flush()
    a, b = summ["worker bench50"], summ["clean81"]
    for k, lbl in (
        ("fo10", "false-outside @ medial centre"),
        ("fo5", "false-outside @ half depth"),
        ("w", "median |w| interior"),
        ("nc", "components"),
        ("be", "boundary frac"),
    ):
        u, p = stats.mannwhitneyu(a[k], b[k])
        print(f"\n  {lbl:<30} worker={np.median(a[k]):.4f}  clean={np.median(b[k]):.4f}  p={p:.3g}", file=fh)

    print("\n" + "=" * 100, file=fh)
    print("TEST 3  the articulation-only contribution to the ceiling-free conditional", file=fh)
    dd = load_model(os.path.join(ROOT, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"))
    v_t = np.asarray(dd["v_template"], dtype=np.float64)
    F = np.asarray(dd["f"]).astype(int)
    Wt = np.asarray(dd["weights"], dtype=np.float64)
    Jreg = np.asarray(dd["J_regressor"], dtype=np.float64)
    parents = np.asarray(dd["kintree_table"][0]).astype(int)
    names = list(dd["J_names"])
    groups = np.array([joint_group(n) for n in names])
    J0 = joint_regress(v_t[None], Jreg)
    nJ = J0.shape[1]
    Rs0 = rodrigues(np.zeros((nJ, 3))).reshape(1, nJ, 3, 3)
    Jr0, A0 = global_rigid(Rs0, J0, parents)
    T0 = np.einsum("vj,jab->vab", Wt, A0[0])
    V0 = np.einsum("vab,vb->va", T0, np.concatenate([v_t, np.ones((len(v_t), 1))], 1))[:, :3]
    rest_in = ~(np.abs(winding_number(Jr0[0], V0, F)) < 0.5)
    print(f"  joints inside at rest: {rest_in.sum()}/{nJ}", file=fh)

    dW = np.load(os.path.join(ROOT, "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz"), allow_pickle=True)
    dC = np.load(
        "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/"
        "fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz",
        allow_pickle=True,
    )
    for tag, d in (("worker poses", dW), ("clean poses", dC)):
        th = np.concatenate([np.zeros_like(d["global_rot"])[:, None, :], d["joint_rot"]], axis=1).astype(np.float64)
        cond, per_g = [], {g: [] for g in sorted(set(groups))}
        for k in range(th.shape[0]):
            Rs = rodrigues(th[k]).reshape(1, nJ, 3, 3)
            Jp, A = global_rigid(Rs, J0, parents)
            Tm = np.einsum("vj,jab->vab", Wt, A[0])
            V = np.einsum("vab,vb->va", Tm, np.concatenate([v_t, np.ones((len(v_t), 1))], 1))[:, :3]
            o = np.abs(winding_number(Jp[0], V, F)) < 0.5
            cond.append(o[rest_in].mean())
            for g in per_g:
                m = rest_in & (groups == g)
                if m.any():
                    per_g[g].append(o[m].mean())
        cond = np.array(cond)
        print(
            f"  {tag:<14} P(outside | inside at rest) = {100 * cond.mean():.1f}%   "
            f"median={100 * np.median(cond):.1f}%   max={100 * cond.max():.1f}%",
            file=fh,
        )
        print("      " + "  ".join(f"{g}={100 * np.mean(v):.1f}" for g, v in per_g.items() if v), file=fh)

    fh.flush()
    fout.close()
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
