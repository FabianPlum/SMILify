"""PROBE 2: how many of the 530 half_workers scans are rotated w.r.t. the template?

The template SMIL_OmniAnt is FLAT IN Z (axis-aligned extents [1.91, 1.53, 0.52] after the
pipeline's own normalisation). A correctly-oriented ant scan must also be flat in Z.
We census which axis is thinnest for every scan.
"""

import glob
import pickle
import numpy as np

SCANS = "/media/fabi/Data/SMILify_DATASETS_BACKUP/custom_processing/antscan_proofread_castes/half_workers/*.obj"


def verts_of(path):
    out = []
    with open(path) as fh:
        for ln in fh:
            if ln[:2] == "v ":
                out.append(ln.split()[1:4])
    return np.asarray(out, np.float64)


with open("/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl", "rb") as fh:
    tv = np.asarray(pickle.load(fh, encoding="latin1")["v_template"], np.float64)
tv = tv - tv.mean(0)
tv /= np.abs(tv).max()
text = tv.max(0) - tv.min(0)
print(
    f"TEMPLATE extents = {np.round(text, 3)}  -> thinnest axis = {'XYZ'[text.argmin()]}"
    f"  flatness = {text.min() / text.max():.3f}\n"
)

files = sorted(glob.glob(SCANS))
thin, flat, aniso = [], [], []
for p in files:
    V = verts_of(p)
    V = V - V.mean(0)
    V /= np.abs(V).max()
    e = V.max(0) - V.min(0)
    thin.append(e.argmin())
    flat.append(e.min() / e.max())
    aniso.append(e)
thin, flat, aniso = np.array(thin), np.array(flat), np.array(aniso)
print(f"n = {len(files)} scans")
for a in range(3):
    n = (thin == a).sum()
    print(f"  thinnest axis = {'XYZ'[a]} : {n:4d}  ({100 * n / len(files):5.1f} %)")
print(f"\ntemplate-consistent (thinnest = Z): {(thin == 2).sum()}/{len(files)} = {100 * (thin == 2).mean():.1f} %")
print(
    f"\nflatness min/max ratio: mean={flat.mean():.3f} median={np.median(flat):.3f}"
    f" p10={np.percentile(flat, 10):.3f} p90={np.percentile(flat, 90):.3f}"
    f"  (template {text.min() / text.max():.3f})"
)
n_round = (flat > 0.6).sum()
print(
    f"scans that are NOT flat at all (min/max > 0.6, no rotation can make them match a flat "
    f"template): {n_round} ({100 * n_round / len(files):.1f} %)"
)
print(f"\nmean axis-aligned extents over all scans = {np.round(aniso.mean(0), 3)}")
