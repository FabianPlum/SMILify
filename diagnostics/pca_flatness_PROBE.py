"""PROBE 3: split the 530 scans into 'merely ROTATED' vs 'genuinely NOT FLAT'.

PCA-frame extents are rotation-invariant, so they separate the two causes of the
axis-aligned bbox mismatch reported against the template:
  - low PCA flatness + wrong axis-aligned thin axis  -> pure rotation, fixable by canonicalisation
  - high PCA flatness                                -> genuinely 3D-splayed specimen, rotation cannot fix
Area-weighted sampling is used so the statistic is not biased by scanner tessellation density.
"""

import glob
import pickle
import numpy as np
import trimesh

SCANS = "/media/fabi/Data/SMILify_DATASETS_BACKUP/custom_processing/antscan_proofread_castes/half_workers/*.obj"


def pca_stats(V, F, n=50_000, seed=0):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), size=n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    fl = (u + v) > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    P = a[idx] + u * (b[idx] - a[idx]) + v * (c[idx] - a[idx])
    P = P - P.mean(0)
    cov = (P.T @ P) / len(P)
    w, R = np.linalg.eigh(cov)
    P = P @ R[:, ::-1]
    ext = P.max(0) - P.min(0)
    return ext / ext[0], (w[::-1] / w.sum())


with open("/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl", "rb") as fh:
    dd = pickle.load(fh, encoding="latin1")
t_ext, t_spec = pca_stats(np.asarray(dd["v_template"], float), np.asarray(dd["f"], np.int64))
print(f"TEMPLATE  PCA-frame ext ratio = {np.round(t_ext, 3)}   spectrum = {np.round(t_spec, 3)}")
print(f"TEMPLATE  PCA flatness (e3/e1) = {t_ext[2]:.3f}\n")

rows = []
for p in sorted(glob.glob(SCANS)):
    m = trimesh.load(p, process=False, force="mesh")
    e, s = pca_stats(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))
    V = np.asarray(m.vertices, float)
    V = V - V.mean(0)
    V /= np.abs(V).max()
    ae = V.max(0) - V.min(0)
    rows.append((e[2], ae.argmin(), ae.min() / ae.max()))
f_pca = np.array([r[0] for r in rows])
thin = np.array([r[1] for r in rows])
f_aa = np.array([r[2] for r in rows])

print(f"n = {len(rows)}")
print(
    f"PCA flatness e3/e1:  mean={f_pca.mean():.3f} median={np.median(f_pca):.3f} "
    f"p10={np.percentile(f_pca, 10):.3f} p90={np.percentile(f_pca, 90):.3f}   (template {t_ext[2]:.3f})"
)
TOL = 1.6  # allow scans up to 1.6x less flat than the template
lim = t_ext[2] * TOL
rot_only = f_pca <= lim
print(
    f"\nscans as flat as the template (e3/e1 <= {lim:.3f}): {rot_only.sum()}/{len(rows)} "
    f"= {100 * rot_only.mean():.1f}%   -> their bbox mismatch is PURE ROTATION"
)
print(
    f"genuinely non-flat specimens                      : {(~rot_only).sum()}/{len(rows)} "
    f"= {100 * (~rot_only).mean():.1f}%   -> rotation cannot fix these"
)
print(f"\nof the {rot_only.sum()} flat (rotation-only) scans, thinnest axis-aligned axis:")
for a in range(3):
    n = ((thin == a) & rot_only).sum()
    print(
        f"   {'XYZ'[a]} : {n:4d}  ({100 * n / max(rot_only.sum(), 1):5.1f}%)"
        + ("   <- template-consistent" if a == 2 else "   <- MIS-ORIENTED")
    )
np.save("/home/fabi/dev/SMILify/diagnostics/pca_flatness.npy", np.stack([f_pca, thin.astype(float), f_aa]))
