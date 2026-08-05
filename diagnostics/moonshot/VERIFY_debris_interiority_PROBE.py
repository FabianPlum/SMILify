import numpy as np
from scipy.spatial import cKDTree, Delaunay

d = np.load("/home/fabi/dev/SMILify/diagnostics/moonshot/partfield/synth_train.npz")
pts, lab = d["pts"], d["lab"]
rng = np.random.default_rng(0)
ins_all, near_all = [], []
for i in range(60):
    p, l = pts[i], lab[i]
    m = l == 13
    if m.sum() < 20:
        continue
    clean = p[~m]
    sub = clean[rng.choice(len(clean), min(3000, len(clean)), replace=False)]
    try:
        hull = Delaunay(sub)
    except Exception:
        continue
    inside = hull.find_simplex(p[m]) >= 0
    dd, _ = cKDTree(clean).query(p[m], k=1)
    ins_all.append(inside)
    near_all.append(dd)
ins = np.concatenate(ins_all)
near = np.concatenate(near_all)
print(
    "synthetic TRAINING debris: interior-to-hull fraction %.1f%%  (probe16 measured 90.8%% on PREDICTED debris)"
    % (100 * ins.mean())
)
print("  of the near-surface (<0.035) debris, interior fraction: %.1f%%" % (100 * ins[near < 0.035].mean()))
print("  of the far (>=0.035) debris,        interior fraction: %.1f%%" % (100 * ins[near >= 0.035].mean()))
