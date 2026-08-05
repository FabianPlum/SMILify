"""PROBE: is the template-vs-target extent mismatch a ROTATION problem or a genuine SHAPE problem?

Method: compare rotation-INVARIANT descriptors (area-weighted PCA eigenvalue spectrum of the
surface point distribution). If the spectra match but the axis-aligned bboxes don't, the gap is
pure rotation -> fixable by canonicalisation. If the spectra also differ, the target genuinely is
more isotropic than the template (splayed legs / raised antennae) and rotation alone won't fix it.
"""

import glob
import pickle
import numpy as np

TEMPLATE = "/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl"
SCANS = "/media/fabi/Data/SMILify_DATASETS_BACKUP/custom_processing/antscan_proofread_castes/half_workers/*.obj"


def load_obj(path):
    V, F = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def area_samples(V, F, n=200_000, seed=0):
    """Area-weighted surface samples -> unbiased by tessellation density."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), size=n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[idx] + u * (b[idx] - a[idx]) + v * (c[idx] - a[idx])


def descriptor(V, F, name):
    P = area_samples(V, F)
    # --- what the CURRENT pipeline does: vertex-mean centroid + global max-abs scale ---
    cur = V - V.mean(0)
    cur = cur / np.abs(cur).max()
    cur_ext = cur.max(0) - cur.min(0)
    # --- area-weighted centroid + RMS radius scale (outlier-insensitive) ---
    Pc = P - P.mean(0)
    rms = np.sqrt((Pc**2).sum(1).mean())
    # rotation-invariant: PCA eigenvalue spectrum, normalised to trace 1
    cov = (Pc.T @ Pc) / len(Pc)
    w = np.linalg.eigvalsh(cov)[::-1]
    spec = w / w.sum()
    # extent in the PCA frame (i.e. after optimal rigid canonicalisation)
    R = np.linalg.eigh(cov)[1][:, ::-1]
    Pr = Pc @ R
    pca_ext = (Pr.max(0) - Pr.min(0)) / rms
    print(
        f"{name:<34s} axis-aligned ext(cur norm)=[{cur_ext[0]:.2f} {cur_ext[1]:.2f} {cur_ext[2]:.2f}]"
        f"  PCA-frame ext/rms=[{pca_ext[0]:.2f} {pca_ext[1]:.2f} {pca_ext[2]:.2f}]"
        f"  eig spectrum=[{spec[0]:.3f} {spec[1]:.3f} {spec[2]:.3f}]"
        f"  maxabs/rms={np.abs(V - V.mean(0)).max() / rms:.2f}"
    )
    return spec, pca_ext


with open(TEMPLATE, "rb") as fh:
    dd = pickle.load(fh, encoding="latin1")
tv = np.asarray(dd["v_template"], np.float64)
tf = np.asarray(dd["f"], np.int64)
print(f"template verts {tv.shape} faces {tf.shape}\n")
descriptor(tv, tf, "TEMPLATE SMIL_OmniAnt")
print()
files = sorted(glob.glob(SCANS))
print(f"{len(files)} scans found; probing 8\n")
specs = []
for p in files[:: max(1, len(files) // 8)][:8]:
    s, _ = descriptor(*load_obj(p), p.split("/")[-1])
    specs.append(s)
specs = np.array(specs)
print(f"\ntarget spectrum mean={specs.mean(0)} std={specs.std(0)}")
