"""PROBE: quantify the two claims made in the non-rigid registration literature review.

1. ORIENTATION / CANONICALISATION: is the target's principal frame consistent with the
   template's after `load_meshes()` normalisation? (bbox extents + PCA eigenvalues/axes)
2. CHAMFER DENSITY BIAS: how non-uniform is vertex density on the template and on the
   targets? Vertex-mean centroid vs area-weighted centroid quantifies the bias that
   `verts.mean(0)` normalisation and `chamfer(target_samples, src.verts_padded())` inherit.

Run:  /home/fabi/mambaforge/envs/pytorch3d/bin/python diagnostics/registration_lit_probe.py
Writes: diagnostics/registration_lit_probe_out.txt
"""

import glob
import os
import pickle

import numpy as np
import trimesh

TEMPLATE = "/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl"
SCAN_DIR = "/media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN"
N_SCANS = 8
OUT = "/home/fabi/dev/SMILify/diagnostics/registration_lit_probe_out.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def tri_areas(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def area_centroid(v, f):
    ar = tri_areas(v, f)
    cen = v[f].mean(1)
    return (cen * ar[:, None]).sum(0) / ar.sum()


def pca_report(pts, name):
    """PCA of a point set; returns eigenvalues (desc) and axes as rows."""
    x = pts - pts.mean(0)
    cov = x.T @ x / len(x)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    sd = np.sqrt(w)
    log(f"  {name}: pca_sd = {np.round(sd, 4)}   sd ratios = {np.round(sd / sd[0], 3)}")
    return sd, V.T


def norm_current(v):
    """Exactly what fitter_3d/utils.py load_meshes() does."""
    v = v - v.mean(0)
    return v / np.abs(v).max()


def norm_areaweighted(v, f):
    """Proposed: area-weighted centroid + RMS-radius scale (density independent)."""
    ar = tri_areas(v, f)
    c = area_centroid(v, f)
    v = v - c
    tc = v[f].mean(1)
    rms = np.sqrt((ar * (tc**2).sum(1)).sum() / ar.sum())
    return v / rms


# ---------------------------------------------------------------- template
d = pickle.load(open(TEMPLATE, "rb"), encoding="latin1")
tv = np.asarray(d["v_template"], dtype=np.float64)
tf = np.asarray(d["f"], dtype=np.int64)

log("=" * 78)
log("TEMPLATE  SMIL_OmniAnt.pkl")
log("=" * 78)
log(f"  verts {tv.shape}  faces {tf.shape}")
tvn = norm_current(tv)
log(f"  bbox extent (current norm) = {np.round(tvn.max(0) - tvn.min(0), 4)}")
t_sd, t_axes = pca_report(tvn, "template")
log(f"  pca axes (rows)=\n{np.round(t_axes, 3)}")

# density bias on the template
cv, ca = tv.mean(0), area_centroid(tv, tf)
diag = np.linalg.norm(tv.max(0) - tv.min(0))
log(f"  vertex-mean vs area-weighted centroid offset = {np.linalg.norm(cv - ca) / diag * 100:.2f}% of bbox diag")

# per-vertex "area share" -> how uneven is vertex density over surface area
ar = tri_areas(tv, tf)
vert_area = np.zeros(len(tv))
np.add.at(vert_area, tf.ravel(), np.repeat(ar / 3.0, 3))
va = vert_area / vert_area.mean()
log(
    f"  per-vertex area share: min {va.min():.3f}  p5 {np.percentile(va, 5):.3f}  "
    f"median {np.median(va):.3f}  p95 {np.percentile(va, 95):.3f}  max {va.max():.1f}"
)
log(
    f"  -> vertex-based chamfer over-weights the densest 10% of verts by "
    f"{np.median(va) / np.percentile(va, 10):.1f}x relative to area-uniform sampling"
)

# ---------------------------------------------------------------- targets
log()
log("=" * 78)
log(f"TARGETS  (first {N_SCANS} of {SCAN_DIR})")
log("=" * 78)
files = sorted(glob.glob(os.path.join(SCAN_DIR, "*.obj")))[:N_SCANS]
log(f"  total scans in dir: {len(sorted(glob.glob(os.path.join(SCAN_DIR, '*.obj'))))}")

all_axes = []
for fp in files:
    m = trimesh.load(fp, process=False, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    log()
    log(f"  {os.path.basename(fp)}  V={len(v)} F={len(f)}")
    log(f"    raw extent = {np.round(v.max(0) - v.min(0), 1)}")
    vn = norm_current(v)
    log(f"    bbox extent (current norm) = {np.round(vn.max(0) - vn.min(0), 4)}")
    sd, axes = pca_report(vn, "target  ")
    all_axes.append(axes)
    # how far is the target principal frame from the template's?
    R = axes.T @ t_axes  # maps template frame -> target frame (up to sign/permutation)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    log(f"    rotation template-PCA -> target-PCA: {ang:.1f} deg (sign/permutation ambiguous)")
    cv2_, ca2 = v.mean(0), area_centroid(v, f)
    dg = np.linalg.norm(v.max(0) - v.min(0))
    log(f"    vertex-mean vs area-weighted centroid offset = {np.linalg.norm(cv2_ - ca2) / dg * 100:.2f}% of bbox diag")
    vn2 = norm_areaweighted(v, f)
    log(f"    bbox extent (area-weighted norm) = {np.round(vn2.max(0) - vn2.min(0), 4)}")

# consistency of target orientation across scans
log()
log("  cross-target principal-axis consistency (angle of axis 1 between scan pairs, deg):")
a1 = np.array([a[0] for a in all_axes])
M = np.degrees(np.arccos(np.clip(np.abs(a1 @ a1.T), -1, 1)))
log(np.round(M, 1))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nwritten -> {OUT}")
