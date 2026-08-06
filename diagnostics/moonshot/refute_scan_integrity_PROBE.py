"""REFUTE-B -- are the two scan corpora the same KIND of object?

The inside/outside test (generalized winding number) is only defined on a closed,
consistently oriented surface. The prior probe validated GWN by offsetting RANDOM face
centroids, which is dominated by the huge body surface; the legs (where the claim lives)
are a small fraction of the surface area.

Measures per scan: verts, faces, dropped polygon corners (the OBJ parser keeps only the
first 3 of each face line), boundary edges, non-manifold edges, connected components,
duplicate-vertex fraction, mean NN spacing, and the bbox diagonal.
"""

import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

ROOT = "/home/fabi/dev/SMILify"
DIRS = [
    ("worker (bench50)", f"{ROOT}/diagnostics/moonshot/bench50"),
    ("clean  (clean81)", f"{ROOT}/diagnostics/moonshot/clean81"),
]


def load_obj_full(path):
    V, F, nquad, ncorner_dropped = [], [], 0, 0
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                toks = line.split()[1:]
                if len(toks) > 3:
                    nquad += 1
                    ncorner_dropped += len(toks) - 3
                idx = [int(t.split("/")[0]) - 1 for t in toks[:3]]
                F.append(idx)
    return np.asarray(V, float), np.asarray(F, int), nquad, ncorner_dropped


def edge_stats(F):
    cnt = defaultdict(int)
    for tri in F:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            cnt[(min(a, b), max(a, b))] += 1
    vals = np.fromiter(cnt.values(), int)
    return int((vals == 1).sum()), int((vals > 2).sum()), len(vals)


def components(nv, F):
    parent = np.arange(nv)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in F:
        a = find(tri[0])
        for o in tri[1:]:
            b = find(o)
            if a != b:
                parent[b] = a
    roots = np.array([find(i) for i in range(nv)])
    used = np.zeros(nv, bool)
    used[F.ravel()] = True
    return len(np.unique(roots[used]))


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = f"{ROOT}/diagnostics/moonshot/refute_scan_integrity_out.txt"
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        for tag, d in DIRS:
            files = sorted(x for x in os.listdir(d) if x.endswith(".obj"))
            rows = []
            for x in files:
                V, F, nq, ndrop = load_obj_full(os.path.join(d, x))
                nb, nnm, ne = edge_stats(F)
                nc = components(V.shape[0], F)
                Vn = V - V.mean(0)
                Vn = Vn / np.abs(Vn).max()
                diag = np.linalg.norm(Vn.max(0) - Vn.min(0))
                dd, _ = cKDTree(Vn).query(Vn, k=2)
                rows.append((x, V.shape[0], F.shape[0], nq, ndrop, nb, nnm, ne, nc, 100 * dd[:, 1].mean() / diag, diag))
            arr = rows
            print(f"\n### {tag}   n={len(arr)}", file=fh)
            print(f"   {'stat':<26}{'median':>12}{'min':>12}{'max':>12}", file=fh)
            for j, nm in (
                (1, "n_verts"),
                (2, "n_tris_parsed"),
                (3, "n_poly>3"),
                (4, "corners_dropped"),
                (5, "boundary_edges"),
                (6, "nonmanifold_edges"),
                (8, "connected_components"),
                (9, "NN spacing %diag"),
            ):
                v = np.array([r[j] for r in arr], float)
                print(f"   {nm:<26}{np.median(v):12.3f}{v.min():12.3f}{v.max():12.3f}", file=fh)
            nbad = sum(1 for r in arr if r[5] > 0 or r[6] > 0)
            print(f"   scans with boundary or non-manifold edges: {nbad}/{len(arr)}", file=fh)
            nmulti = sum(1 for r in arr if r[8] > 1)
            print(f"   scans with >1 connected component:        {nmulti}/{len(arr)}", file=fh)
            worst = sorted(arr, key=lambda r: -(r[5] + r[6]))[:5]
            for w in worst:
                print(f"     {w[0][:48]:<50} bnd={w[5]:6d} nonman={w[6]:5d} comp={w[8]:3d} quads={w[3]:5d}", file=fh)
            fh.flush()
    print(f"wrote {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
