"""PROBE (feedforward-angle): how badly does a Euclidean-kNN point backbone smear
anatomically distinct parts together on REAL ant scans?

ArtEq (Feng et al., ICCV 2023) names this as its primary failure mode: "self-contact may
lead to incorrect feature aggregation, where features that belong to different body parts
are convolved together due to their proximity in Euclidean space." Every PointNet++ /
PointTransformer / sparse-conv backbone builds its receptive field from Euclidean
neighbourhoods (ball query / kNN / voxel), so if two different legs are Euclidean-near but
geodesic-far, the backbone physically cannot keep them apart.

Metric: sample S source vertices per scan. For each, take its k Euclidean nearest
neighbours within radius r (fraction of body length). Compute exact graph (geodesic-proxy)
distance along mesh edges. A neighbour is CONTAMINATING if geodesic/Euclidean > RATIO
(i.e. it is touching in space but far away along the surface -> a different appendage).

Report per-scan and pooled contamination rate at several radii. Interpretation:
  low  (<2%)  -> plain Euclidean backbone is fine, use PointNet++/PTv3 off the shelf
  high (>10%) -> Euclidean receptive fields are structurally wrong here; need small
                 kernels (ArtEq's fix), geodesic/graph features, or per-part heads
"""

import glob
import os
import sys

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

MESH_DIR = "/home/fabi/dev/SMILify/diagnostics/moonshot/bench50"
N_SCANS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_SRC = 400  # source vertices per scan
K = 16  # neighbours per source (typical ball-query / kNN budget)
RATIOS = 5.0  # geodesic/euclidean ratio above which a neighbour is "contaminating"
RADII = [0.01, 0.02, 0.04]  # ball-query radius as fraction of body length


def load(path):
    m = trimesh.load(path, process=False, force="mesh")
    V = np.asarray(m.vertices, float)
    F = np.asarray(m.faces, np.int64)
    return V, F


def edge_graph(V, F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    n = len(V)
    return coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()


def probe(path, rng):
    V, F = load(path)
    body = V[:, 0].max() - V[:, 0].min()
    body = max(body, np.ptp(V, axis=0).max())
    G = edge_graph(V, F)
    tree = cKDTree(V)
    src = rng.choice(len(V), size=min(N_SRC, len(V)), replace=False)

    out = {}
    for rad_frac in RADII:
        r = rad_frac * body
        # cap dijkstra at the largest geodesic we care about
        D = dijkstra(G, directed=False, indices=src, limit=RATIOS * r)
        contam = 0
        total = 0
        for i, s in enumerate(src):
            nb = tree.query_ball_point(V[s], r)
            nb = [j for j in nb if j != s]
            if not nb:
                continue
            nb = np.asarray(nb)
            if len(nb) > K:
                d_e = np.linalg.norm(V[nb] - V[s], axis=1)
                nb = nb[np.argsort(d_e)[:K]]
            d_e = np.linalg.norm(V[nb] - V[s], axis=1)
            d_g = D[i, nb]
            bad = (~np.isfinite(d_g)) | (d_g > RATIOS * np.maximum(d_e, 1e-12))
            contam += int(bad.sum())
            total += len(nb)
        out[rad_frac] = (contam, total)
    return out, len(V), len(F)


def main():
    files = sorted(glob.glob(os.path.join(MESH_DIR, "*.obj")))
    if not files:
        print(f"no meshes in {MESH_DIR}")
        return
    rng = np.random.default_rng(0)
    files = files[:N_SCANS]
    print(f"{'scan':<52s} {'nV':>7s} " + " ".join(f"r={r:<5.2f}" for r in RADII))
    pooled = {r: [0, 0] for r in RADII}
    for f in files:
        try:
            res, nv, nf = probe(f, rng)
        except Exception as exc:  # noqa: BLE001
            print(f"{os.path.basename(f)[:50]:<52s} FAILED {exc}")
            continue
        row = []
        for r in RADII:
            c, t = res[r]
            pooled[r][0] += c
            pooled[r][1] += t
            row.append(f"{100.0 * c / max(t, 1):6.1f}%")
        print(f"{os.path.basename(f)[:50]:<52s} {nv:7d} " + " ".join(row))
    print()
    print("POOLED contamination rate (Euclidean-near but geodesic-far neighbours):")
    for r in RADII:
        c, t = pooled[r]
        print(f"  ball radius {r * 100:.0f}% of body length: {100.0 * c / max(t, 1):.2f}%  ({c}/{t})")


if __name__ == "__main__":
    main()
