"""PROBE: can a geodesic/Reeb decomposition of the raw ant scan isolate the six legs?

Motivation (see diagnostics/moonshot/REPORT.md sec.5): pose is trapped in a wrong basin
because chamfer on a hexapod is multimodal -- six near-identical, mutually substitutable
legs. Fixing that needs the *assignment* solved, not the optimiser tuned. This probe tests
the cheapest discriminative structure available: the scan's own branching topology.

Method (no mesh connectivity used -- CT meshes have hundreds of components):
  1. area-sample N points from the mesh surface
  2. build a radius/kNN graph over the points (bridges the mesh's topological holes)
  3. geodesic (graph) distance from a body seed
  4. sweep level sets of that distance; each connected component that survives past the
     body radius is a limb branch
  5. label each branch by its own canonical coordinates: sign(y) -> left/right,
     order of attachment x -> leg 1/2/3, so the assignment is read off, not searched.

Outputs a per-specimen table. Nothing here is written into the fitting pipeline.
"""

import glob
import sys
import time

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

BENCH = "/home/fabi/dev/SMILify/diagnostics/moonshot/bench50/*.obj"
NPTS = 20000
K = 10


def build_graph(P, k=K):
    """kNN graph with symmetric edges, weights = euclidean length."""
    tree = cKDTree(P)
    d, idx = tree.query(P, k=k + 1)
    rows = np.repeat(np.arange(len(P)), k)
    cols = idx[:, 1:].ravel()
    w = d[:, 1:].ravel()
    A = coo_matrix((w, (rows, cols)), shape=(len(P), len(P))).tocsr()
    A = A.maximum(A.T)  # symmetrise
    return A, d[:, 1:].mean()


def bridge(A, P, k, max_extra_frac=0.02):
    """Connect graph components by their mutually-closest points until 1 component."""
    n_extra = 0
    for _ in range(200):
        ncomp, lab = connected_components(A, directed=False)
        if ncomp == 1:
            break
        sizes = np.bincount(lab)
        main = np.argmax(sizes)
        main_idx = np.where(lab == main)[0]
        other_idx = np.where(lab != main)[0]
        t = cKDTree(P[main_idx])
        d, j = t.query(P[other_idx], k=1)
        # link every non-main point cluster-wise: cheapest link per component
        rows, cols, w = [], [], []
        for c in np.unique(lab[other_idx]):
            sel = np.where(lab[other_idx] == c)[0]
            b = sel[np.argmin(d[sel])]
            rows.append(other_idx[b])
            cols.append(main_idx[j[b]])
            w.append(d[b])
        n_extra += len(rows)
        E = coo_matrix((np.array(w), (np.array(rows), np.array(cols))), shape=A.shape).tocsr()
        A = A.maximum(E).maximum(E.T)
    return A, n_extra


def analyse(path, verbose=False):
    m = trimesh.load(path, process=False)
    t0 = time.time()
    P, _ = trimesh.sample.sample_surface(m, NPTS)
    P = np.asarray(P, dtype=np.float64)
    # canonical frame per REPORT sec.7: body along x (gaster -x, head +x), y lateral, z up
    ext = P.max(0) - P.min(0)
    axis = int(np.argmax(ext))
    scale = ext[axis]
    A, mean_knn = build_graph(P)
    ncomp0, _ = connected_components(A, directed=False)
    A, n_extra = bridge(A, P, K)

    # seed = medial-ish body point: the point with the largest distance to the
    # surface bounding-box centre is a bad seed, so use the densest region instead
    # (thorax/gaster are volumetric, legs are thin) -> point with most neighbours in r
    tree = cKDTree(P)
    r = 0.03 * scale
    dens = np.array(tree.query_ball_point(P, r, return_length=True))
    seed = int(np.argmax(dens))

    g = dijkstra(A, directed=False, indices=seed)
    g = g / scale  # geodesic distance in units of body length

    # Reeb sweep: how many components does the level set {g > t} have?
    res = {}
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        sel = np.where(g > t)[0]
        if len(sel) < 50:
            res[t] = (0, 0)
            continue
        sub = A[sel][:, sel]
        nc, lab = connected_components(sub, directed=False)
        sizes = np.bincount(lab)
        big = int((sizes > 0.002 * NPTS).sum())  # ignore specks
        res[t] = (nc, big)

    # pick the sweep level with the most stable branch count, report branch labels
    t_best = 0.30
    sel = np.where(g > t_best)[0]
    out = []
    if len(sel) > 50:
        sub = A[sel][:, sel]
        nc, lab = connected_components(sub, directed=False)
        sizes = np.bincount(lab)
        keep = np.where(sizes > 0.002 * NPTS)[0]
        for c in keep:
            pts = P[sel[lab == c]]
            out.append((len(pts), pts.mean(0), pts))
    return dict(
        name=path.split("/")[-1][:38],
        nV=len(m.vertices),
        ncomp_mesh=len(m.split(only_watertight=False)) if len(m.faces) < 300000 else -1,
        ncomp_graph=ncomp0,
        n_bridge=n_extra,
        scale=scale,
        axis=axis,
        sweep=res,
        branches=out,
        secs=time.time() - t0,
    )


if __name__ == "__main__":
    files = sorted(glob.glob(BENCH))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"{NPTS} area samples, k={K} kNN graph, geodesic from densest point\n")
    hdr = f"{'specimen':40s} {'grcomp':>7s} {'bridge':>6s} | branches at geodesic t (fraction of body length)"
    print(hdr)
    print(f"{'':40s} {'':7s} {'':6s} | " + "  ".join(f"{t:.2f}" for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]))
    for f in files[:n]:
        r = analyse(f)
        row = "  ".join(f"{r['sweep'][t][1]:4d}" for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
        print(f"{r['name']:40s} {r['ncomp_graph']:7d} {r['n_bridge']:6d} | {row}   ({r['secs']:.1f}s)")
