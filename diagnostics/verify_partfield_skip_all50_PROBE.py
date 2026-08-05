"""VERIFY part 2: run the field over all 50 bench specimens, count n_t<10 skips per
specimen, and cross-tabulate against the ACTUAL per-specimen G5 regression on disk
(M9a_field vs its control M8a_split_only at H3_deform). CPU only.
"""

import csv
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # noqa: E402
from fitter_3d.partfield import PartFieldNet, predict_field, part_names  # noqa: E402
from fitter_3d.trainer_hierarchical import vertex_groups  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"
N_SAMPLE = 8000
CACHE = "diagnostics/verify_partfield_labels_PROBE.npz"

with open(config.SMAL_FILE, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
jnames = list(dd["J_names"])
vg, gnames = vertex_groups(np.asarray(dd["weights"]), jnames, split_distal=True)
pf_names = part_names(jnames)
debris_id = pf_names.index("debris")
DISTAL = [i for i, g in enumerate(gnames) if "d_" in g]

d = np.load("diagnostics/moonshot/partfield/real_weak.npz", allow_pickle=True)
pts_all, files = d["pts"], d["files"]

if os.path.exists(CACHE):
    c = np.load(CACHE)
    LAB, CONF = c["lab"], c["conf"]
else:
    ck = torch.load("diagnostics/moonshot/partfield/net_B_mirror.pt", map_location=DEV)
    net = PartFieldNet(n_classes=len(ck["names"]), width=ck.get("width", 1.0)).to(DEV)
    net.load_state_dict(ck["state"])
    net.eval()
    L, C = [], []
    for i in range(len(pts_all)):
        prob = predict_field(net, torch.tensor(pts_all[i], dtype=torch.float32), device=DEV, seed=0)
        conf, lab = prob.max(-1)
        L.append(lab.numpy())
        C.append(conf.numpy())
        print(f"  field {i + 1}/{len(pts_all)}", flush=True)
    LAB, CONF = np.stack(L), np.stack(C)
    np.savez_compressed(CACHE, lab=LAB, conf=CONF)

# ---- skip counts per specimen, averaged over 5 target redraws (the fitter redraws too)
rng = np.random.default_rng(7)
n_skip = np.zeros(len(LAB))
skip_detail = {}
for i in range(len(LAB)):
    lab = LAB[i].copy()
    lab[lab == debris_id] = -1
    per_g = np.zeros(len(gnames))
    for draw in range(5):
        sel = rng.choice(len(lab), size=N_SAMPLE, replace=False)
        ls = lab[sel]
        for g in range(len(gnames)):
            if int((ls == g).sum()) < 10:
                per_g[g] += 1
    n_skip[i] = (per_g >= 3).sum()  # skipped in a majority of redraws
    skip_detail[str(files[i])] = [(gnames[g], int((lab == g).sum())) for g in range(len(gnames)) if per_g[g] >= 3]


def load_metrics(run, stage="H3_deform"):
    out = {}
    with open(f"diagnostics/moonshot/runs/{run}/metrics.csv") as f:
        for r in csv.DictReader(f):
            if r["stage"] == stage:
                out[r["mesh"]] = {k: float(v) for k, v in r.items() if k not in ("mesh", "stage")}
    return out


m9, m8 = load_metrics("M9a_field"), load_metrics("M8a_split_only")
key = "part_leg_distal_dist_mean"

print(f"\ntotal skipped (specimen,group) terms over 50 specimens: {int(n_skip.sum())} / {50 * len(gnames)}")
print(f"specimens with >=1 skipped distal group: {int((n_skip > 0).sum())} / 50")
print(f"distal groups skipped: {int(n_skip.sum())} / {50 * len(DISTAL)} distal terms")

rows = []
for i, fn in enumerate(files):
    fn = str(fn)
    if fn not in m9 or fn not in m8:
        continue
    rows.append((n_skip[i], m9[fn][key], m8[fn][key], m9[fn]["chamfer_l2"], m8[fn]["chamfer_l2"], fn))

a = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows])
grp0 = a[a[:, 0] == 0]
grp1 = a[a[:, 0] > 0]
print(f"\nmatched {len(a)} specimens.  mean part_leg_distal  M9a {a[:, 1].mean():.5f}  M8a {a[:, 2].mean():.5f}")
print(
    f"  specimens with 0 skipped groups (n={len(grp0)}): "
    f"M9a {grp0[:, 1].mean():.5f}  M8a {grp0[:, 2].mean():.5f}  delta {grp0[:, 1].mean() - grp0[:, 2].mean():+.5f} "
    f"({100 * (grp0[:, 1].mean() / grp0[:, 2].mean() - 1):+.1f}%)"
)
print(
    f"  specimens with >0 skipped groups (n={len(grp1)}): "
    f"M9a {grp1[:, 1].mean():.5f}  M8a {grp1[:, 2].mean():.5f}  delta {grp1[:, 1].mean() - grp1[:, 2].mean():+.5f} "
    f"({100 * (grp1[:, 1].mean() / grp1[:, 2].mean() - 1):+.1f}%)"
)
dd_ = a[:, 1] - a[:, 2]
print(f"\ncorr(n_skip, delta part_leg_distal) = {np.corrcoef(a[:, 0], dd_)[0, 1]:+.3f}")
print(f"corr(n_skip, delta chamfer_l2)      = {np.corrcoef(a[:, 0], a[:, 3] - a[:, 4])[0, 1]:+.3f}")

print("\nper-specimen (sorted by n_skip):")
for r in sorted(rows, key=lambda x: -x[0]):
    print(
        f"  skip={int(r[0])}  distal M9a {r[1]:.5f} vs M8a {r[2]:.5f}  ({100 * (r[1] / r[2] - 1):+6.1f}%)  {r[5][:46]}"
    )

print("\nNesomyrmex detail:")
for k, v in skip_detail.items():
    if "Nesomyrmex" in k:
        print(f"  {k}: skipped {v}")
