"""PROBE 3: does a discrete rotation search fix the initialisation?

Measures, per scan, the symmetric (area-weighted) chamfer between template and target:
  (a) IDENTITY  -- exactly what fitter_3d/utils.py load_meshes() gives Stage_0 today
  (b) best of the 24 proper axis-permutation/sign rotations of the PCA frame
  (c) best of (b) followed by 10 iterations of rigid ICP (scale-free)
This is the empirical test of the "orientation/canonicalisation is suspect" hypothesis.
"""

import glob
import os
import pickle
import numpy as np
import trimesh
from scipy.spatial import cKDTree

TEMPLATE = "/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl"
SCAN_DIR = "/media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN"
OUT = "/home/fabi/dev/SMILify/diagnostics/registration_rotation_search_probe_out.txt"
NPTS = 4000
rng = np.random.default_rng(0)
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def area_sample(v, f, n):
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    p, _ = trimesh.sample.sample_surface(m, n)
    return np.asarray(p)


def norm_aw(p):
    """area-weighted-sample centroid + RMS radius: density independent."""
    p = p - p.mean(0)
    return p / np.sqrt((p**2).sum(1).mean())


def pca_frame(p):
    x = p - p.mean(0)
    w, V = np.linalg.eigh(x.T @ x / len(x))
    return V[:, np.argsort(w)[::-1]].T


def sym_chamfer(a, b):
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return float((da**2).mean() + (db**2).mean())


def proper_rots():
    """the 24 rotations that permute+flip coordinate axes, det=+1."""
    import itertools

    R = []
    for perm in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, j in enumerate(perm):
                M[i, j] = s[i]
            if abs(np.linalg.det(M) - 1) < 1e-9:
                R.append(M)
    return R


ROTS = proper_rots()


def icp(src, dst, iters=10):
    s = src.copy()
    tree = cKDTree(dst)
    for _ in range(iters):
        _, idx = tree.query(s)
        A = s - s.mean(0)
        B = dst[idx] - dst[idx].mean(0)
        U, _, Vt = np.linalg.svd(A.T @ B)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        s = (s - s.mean(0)) @ R.T + dst[idx].mean(0)
    return s


d = pickle.load(open(TEMPLATE, "rb"), encoding="latin1")
T = norm_aw(area_sample(np.asarray(d["v_template"], float), np.asarray(d["f"], np.int64), NPTS))
Tf = pca_frame(T)

log(f"{'scan':<38} {'identity':>10} {'rot24':>10} {'rot24+icp':>10} {'gain%':>7}")
rows = []
for fp in sorted(glob.glob(os.path.join(SCAN_DIR, "*.obj"))):
    m = trimesh.load(fp, process=False, force="mesh")
    S = norm_aw(area_sample(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), NPTS))
    e_id = sym_chamfer(T, S)
    Sf = pca_frame(S)
    best, bestS = np.inf, None
    for R in ROTS:
        # express template in target's PCA frame with axis permutation/flip R
        Tr = T @ Tf.T @ R.T @ Sf
        e = sym_chamfer(Tr, S)
        if e < best:
            best, bestS = e, Tr
    e_icp = sym_chamfer(icp(bestS, S), S)
    rows.append((e_id, best, e_icp))
    log(f"{os.path.basename(fp):<38} {e_id:>10.4f} {best:>10.4f} {e_icp:>10.4f} {100 * (1 - e_icp / e_id):>6.1f}%")

r = np.array(rows)
log("")
log(f"n = {len(r)}")
for j, nm in enumerate(["identity (CURRENT)", "24-way rot search ", "rot24 + rigid ICP "]):
    log(f"  {nm}  median {np.median(r[:, j]):.4f}   mean {r[:, j].mean():.4f}   p90 {np.percentile(r[:, j], 90):.4f}")
log(
    f"  median symmetric-chamfer reduction, current -> rot24+ICP: "
    f"{100 * (1 - np.median(r[:, 2]) / np.median(r[:, 0])):.1f}%"
)
log(f"  scans where identity is already the best of the 24: {(r[:, 0] <= r[:, 1] + 1e-9).sum()} / {len(r)}")
log(f"  scans where rot24+ICP improves by >25%: {(1 - r[:, 2] / r[:, 0] > 0.25).sum()} / {len(r)}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("written ->", OUT)
