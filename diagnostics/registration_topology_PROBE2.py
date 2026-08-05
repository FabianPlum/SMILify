"""PROBE 2: geodesic/Reeb branch decomposition on the ANT TEMPLATE (target drive unmounted).

Same method as registration_topology_PROBE.py. Target scans live on /media/fabi/Data which
is not mounted in this session, so this runs on SMIL_OmniAnt's own rest-pose mesh as a
stand-in. What this DOES test: whether a geodesic sweep on a kNN point graph over an
ant-shaped surface separates the six legs + two antennae, how many branches it finds, and
how fast. What it does NOT test: robustness to CT debris, holes and hundreds of components.
"""

import pickle
import time

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

PKL = "/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl"
NPTS = 20000
K = 10


def build_graph(P, k=K):
    tree = cKDTree(P)
    d, idx = tree.query(P, k=k + 1)
    rows = np.repeat(np.arange(len(P)), k)
    cols = idx[:, 1:].ravel()
    A = coo_matrix((d[:, 1:].ravel(), (rows, cols)), shape=(len(P), len(P))).tocsr()
    return A.maximum(A.T)


def bridge(A, P):
    extra = 0
    for _ in range(200):
        nc, lab = connected_components(A, directed=False)
        if nc == 1:
            return A, extra
        sizes = np.bincount(lab)
        main = int(np.argmax(sizes))
        mi = np.where(lab == main)[0]
        oi = np.where(lab != main)[0]
        t = cKDTree(P[mi])
        d, j = t.query(P[oi], k=1)
        rows, cols, w = [], [], []
        for c in np.unique(lab[oi]):
            s = np.where(lab[oi] == c)[0]
            b = s[np.argmin(d[s])]
            rows.append(oi[b])
            cols.append(mi[j[b]])
            w.append(d[b])
        extra += len(rows)
        E = coo_matrix((np.array(w), (np.array(rows), np.array(cols))), shape=A.shape).tocsr()
        A = A.maximum(E).maximum(E.T)
    return A, extra


def run(mesh, tag):
    t0 = time.time()
    P, _ = trimesh.sample.sample_surface(mesh, NPTS)
    P = np.asarray(P, float)
    ext = P.max(0) - P.min(0)
    scale = ext.max()
    A = build_graph(P)
    nc0, _ = connected_components(A, directed=False)
    A, extra = bridge(A, P)
    tree = cKDTree(P)
    dens = np.array(tree.query_ball_point(P, 0.03 * scale, return_length=True))
    seed = int(np.argmax(dens))
    g = dijkstra(A, directed=False, indices=seed) / scale
    ts = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    counts = []
    for t in ts:
        sel = np.where(g > t)[0]
        if len(sel) < 50:
            counts.append(0)
            continue
        nc, lab = connected_components(A[sel][:, sel], directed=False)
        sizes = np.bincount(lab)
        counts.append(int((sizes > 0.002 * NPTS).sum()))
    print(
        f"{tag:28s} graph-comps {nc0:3d} bridged {extra:3d} | "
        + "  ".join(f"{c:3d}" for c in counts)
        + f"   ({time.time() - t0:.1f}s)"
    )

    # detail at the most informative level, with canonical labels
    best_t = ts[int(np.argmax(counts))]
    sel = np.where(g > best_t)[0]
    nc, lab = connected_components(A[sel][:, sel], directed=False)
    sizes = np.bincount(lab)
    keep = np.argsort(-sizes)[: min(14, (sizes > 0.002 * NPTS).sum())]
    print(f"    branches at t={best_t}:")
    rows = []
    for c in keep:
        pts = P[sel[lab == c]]
        cen = pts.mean(0)
        # attachment = branch point closest (geodesically) to the seed
        gi = g[sel[lab == c]]
        att = pts[np.argmin(gi)]
        rows.append((len(pts), cen, att))
    rows.sort(key=lambda r: (r[2][1] > 0, r[2][0]))
    for n, cen, att in rows:
        side = "L" if att[1] > 0 else "R"
        print(
            f"      n={n:5d}  attach x={att[0]:+.3f} y={att[1]:+.3f} z={att[2]:+.3f}"
            f"  side={side}  tip_centroid=({cen[0]:+.3f},{cen[1]:+.3f},{cen[2]:+.3f})"
        )


if __name__ == "__main__":
    d = pickle.load(open(PKL, "rb"), encoding="latin1")
    v = np.array(d["v_template"], float)
    f = np.array(d["f"], int)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    print(f"template: {len(v)} verts, {len(f)} faces, {NPTS} samples, k={K}")
    print(f"{'':28s} {'':16s} | branch count at t = 0.10 0.15 0.20 0.25 0.30 0.35 0.40")
    run(mesh, "SMIL_OmniAnt rest pose")
