"""PROBE 2: sign-disambiguated orientation check, template vs targets.

PCA eigenvectors have arbitrary sign, so the ~180 deg results in probe 1 are not
proof of a flip. Disambiguate each axis sign by the third moment (skewness) of the
projected coordinate -- an ant is strongly asymmetric head-vs-gaster along its long
axis -- then re-measure the template->target rotation.
"""

import glob
import os
import pickle
import numpy as np
import trimesh

TEMPLATE = "/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl"
SCAN_DIR = "/media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN"
OUT = "/home/fabi/dev/SMILify/diagnostics/registration_orientation_probe2_out.txt"
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def signed_pca(pts):
    x = pts - pts.mean(0)
    w, V = np.linalg.eigh(x.T @ x / len(x))
    o = np.argsort(w)[::-1]
    V = V[:, o].T  # rows = axes, descending variance
    for k in range(3):  # fix sign by skewness of projection
        p = x @ V[k]
        if (p**3).mean() < 0:
            V[k] = -V[k]
    if np.linalg.det(V) < 0:  # keep a right-handed frame
        V[2] = -V[2]
    return V


d = pickle.load(open(TEMPLATE, "rb"), encoding="latin1")
tv = np.asarray(d["v_template"], dtype=np.float64)
T = signed_pca(tv)
log("template signed-PCA axes (rows):")
log(np.round(T, 3))
log("")
log(f"{'scan':<28} {'rot(T->S) deg':>14} {'axis1 dot':>10} {'axis2 dot':>10} {'axis3 dot':>10}")
angs = []
for fp in sorted(glob.glob(os.path.join(SCAN_DIR, "*.obj"))):
    v = np.asarray(trimesh.load(fp, process=False, force="mesh").vertices, dtype=np.float64)
    S = signed_pca(v)
    R = S.T @ T
    a = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    angs.append(a)
    dots = [float(S[k] @ T[k]) for k in range(3)]
    log(f"{os.path.basename(fp):<28} {a:>14.1f} {dots[0]:>10.3f} {dots[1]:>10.3f} {dots[2]:>10.3f}")
angs = np.array(angs)
log("")
log(f"n = {len(angs)}   median {np.median(angs):.1f} deg   p90 {np.percentile(angs, 90):.1f}   max {angs.max():.1f}")
log(f"scans with residual rotation > 30 deg: {(angs > 30).sum()} / {len(angs)}")
log(f"scans with residual rotation > 90 deg: {(angs > 90).sum()} / {len(angs)}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("written ->", OUT)
