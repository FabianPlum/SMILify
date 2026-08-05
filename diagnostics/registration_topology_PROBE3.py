"""PROBE 3: name the geodesic branches using the template's own LBS weights.

Takes the branches found by the geodesic sweep (PROBE2) and, for each, reports the joint
that the template's skinning weights assign to the nearest template vertices. If the sweep
is discriminative, each branch should map to exactly one leg chain / antenna / gaster,
which is the "which leg is which" question answered combinatorially rather than by
gradient descent.
"""

import pickle
from collections import Counter

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

PKL = "/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl"
NPTS = 20000
K = 10
T = 0.35

d = pickle.load(open(PKL, "rb"), encoding="latin1")
v = np.array(d["v_template"], float)
f = np.array(d["f"], int)
W = np.array(d["weights"], float)
names = [str(n) for n in d["J_names"]]
vert_joint = np.array([names[i] for i in W.argmax(1)])
J = np.array(d["J"], float)

mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
P, _ = trimesh.sample.sample_surface(mesh, NPTS)
P = np.asarray(P, float)
scale = (P.max(0) - P.min(0)).max()

tree = cKDTree(P)
dd, idx = tree.query(P, k=K + 1)
rows = np.repeat(np.arange(len(P)), K)
A = coo_matrix((dd[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(len(P), len(P))).tocsr()
A = A.maximum(A.T)
# bridge disconnected components to the main one
for _ in range(50):
    nc, lab = connected_components(A, directed=False)
    if nc == 1:
        break
    main = int(np.argmax(np.bincount(lab)))
    mi, oi = np.where(lab == main)[0], np.where(lab != main)[0]
    t2 = cKDTree(P[mi])
    dist, j = t2.query(P[oi], k=1)
    r_, c_, w_ = [], [], []
    for c in np.unique(lab[oi]):
        s = np.where(lab[oi] == c)[0]
        b = s[np.argmin(dist[s])]
        r_.append(oi[b])
        c_.append(mi[j[b]])
        w_.append(dist[b])
    E = coo_matrix((np.array(w_), (np.array(r_), np.array(c_))), shape=A.shape).tocsr()
    A = A.maximum(E).maximum(E.T)

dens = np.array(tree.query_ball_point(P, 0.03 * scale, return_length=True))
seed = int(np.argmax(dens))
g = dijkstra(A, directed=False, indices=seed) / scale

vt = cKDTree(v)
sel = np.where(g > T)[0]
nc, lab = connected_components(A[sel][:, sel], directed=False)
sizes = np.bincount(lab)
keep = [c for c in np.argsort(-sizes) if sizes[c] > 0.002 * NPTS]

print(f"geodesic sweep at t={T} body-lengths from the densest interior point")
print(f"{len(keep)} branches from {NPTS} samples\n")
print(f"{'n':>6s} {'attach x':>9s} {'y':>7s} {'side':>5s}  dominant skinning joints (share of branch)")
rows_out = []
for c in keep:
    pts = P[sel[lab == c]]
    gi = g[sel[lab == c]]
    att = pts[np.argmin(gi)]
    _, vi = vt.query(pts, k=1)
    cnt = Counter(vert_joint[vi])
    tot = sum(cnt.values())
    # collapse leg chains to their chain id
    chain = Counter()
    for k_, n_ in cnt.items():
        key = (
            "_".join(k_.split("_")[:2])
            if k_.startswith("l_")
            else ("antenna_" + k_[-1] if k_.startswith("an_") else k_)
        )
        chain[key] += n_
    top = ", ".join(f"{k_}:{n_ / tot:.0%}" for k_, n_ in chain.most_common(3))
    rows_out.append((att[1] > 0, att[0], len(pts), att, top))
rows_out.sort()
for _, _, n, att, top in rows_out:
    print(f"{n:6d} {att[0]:+9.3f} {att[1]:+7.3f} {'L' if att[1] > 0 else 'R':>5s}  {top}")
