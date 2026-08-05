"""VERIFY: does the frozen PartFieldPartition starve distal groups below the n_t<10 skip?

CPU-only. Uses the shipped checkpoint and the same 30k reference clouds the fitter would
build (real_weak.npz holds normalise()d 30k area-uniform samples of the same bench50 scans,
plus the fit-derived labels for cross-tabulation).
"""

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
N_SPEC = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_SAMPLE = 8000

with open(config.SMAL_FILE, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
jnames = list(dd["J_names"])
vg, gnames = vertex_groups(np.asarray(dd["weights"]), jnames, split_distal=True)
pf_names = part_names(jnames)
assert pf_names[:-1] == list(gnames)
debris_id = pf_names.index("debris")
print(f"groups ({len(gnames)}): {list(gnames)}   debris_id={debris_id}")

DISTAL = [i for i, g in enumerate(gnames) if g.startswith("l") and "d_" in g]
PROX = [i for i, g in enumerate(gnames) if g.startswith("l") and "p_" in g]
print(f"distal ids {[gnames[i] for i in DISTAL]}")
print(f"prox   ids {[gnames[i] for i in PROX]}")

ck = torch.load("diagnostics/moonshot/partfield/net_B_mirror.pt", map_location=DEV)
net = PartFieldNet(n_classes=len(ck["names"]), width=ck.get("width", 1.0)).to(DEV)
net.load_state_dict(ck["state"])
net.eval()
assert list(ck["names"]) == pf_names, (list(ck["names"]), pf_names)

d = np.load("diagnostics/moonshot/partfield/real_weak.npz", allow_pickle=True)
pts_all, lab_all, files = d["pts"], d["lab"], d["files"]

rows = []
share_f_d = share_f_p = share_w_d = share_w_p = 0.0
conf_d, conf_p = [], []
for i in range(N_SPEC):
    p = torch.tensor(pts_all[i], dtype=torch.float32)
    prob = predict_field(net, p, device=DEV, seed=0)
    conf, lab = prob.max(-1)
    lab = lab.numpy()
    conf = conf.numpy()
    weak = lab_all[i]

    fd = np.isin(lab, DISTAL).mean()
    fp = np.isin(lab, PROX).mean()
    wd = np.isin(weak, DISTAL).mean()
    wp = np.isin(weak, PROX).mean()
    share_f_d += fd
    share_f_p += fp
    share_w_d += wd
    share_w_p += wp
    if np.isin(lab, DISTAL).any():
        conf_d.append(conf[np.isin(lab, DISTAL)].mean())
    if np.isin(lab, PROX).any():
        conf_p.append(conf[np.isin(lab, PROX)].mean())

    rows.append(dict(name=str(files[i]), lab=lab, conf=conf, weak=weak, fd=fd, wd=wd))
    print(f"[{i:2d}] {str(files[i])[:44]:44s} field distal {100 * fd:5.2f}%  weak distal {100 * wd:5.2f}%")

n = len(rows)
print(
    f"\nMEAN over {n}: distal share field {100 * share_f_d / n:.2f}% vs weak {100 * share_w_d / n:.2f}% "
    f"(ratio {share_f_d / max(share_w_d, 1e-9):.2f})"
)
print(
    f"MEAN over {n}: prox   share field {100 * share_f_p / n:.2f}% vs weak {100 * share_w_p / n:.2f}% "
    f"(ratio {share_f_p / max(share_w_p, 1e-9):.2f})"
)
print(f"mean max-softmax on distal {np.mean(conf_d):.3f}  on proximal {np.mean(conf_p):.3f}")


def count_skips(min_conf, n_draws=3):
    """Simulate the fitter: draw N_SAMPLE area-uniform target points and count per-group."""
    out = []
    for draw in range(n_draws):
        rng = np.random.default_rng(1000 + draw)
        skips, tot, skipped_names = 0, 0, []
        for r in rows:
            lab = r["lab"].copy()
            lab[lab == debris_id] = -1
            if min_conf > 0:
                lab[r["conf"] < min_conf] = -1
            sel = rng.choice(len(lab), size=N_SAMPLE, replace=False)
            ls = lab[sel]
            for g in range(len(gnames)):
                n_t = int((ls == g).sum())
                tot += 1
                if n_t < 10:
                    skips += 1
                    skipped_names.append((r["name"][:30], gnames[g], n_t))
        out.append((skips, tot, skipped_names))
    return out


for mc in (0.0, 0.5, 0.7):
    res = count_skips(mc)
    counts = [f"{s}/{t}" for s, t, _ in res]
    print(f"\n--- pf_min_conf={mc}: skipped (specimen,group) data terms over 3 draws: {counts}")
    s, t, nm = res[0]
    dist_only = all(g.startswith("l") and "d_" in g for _, g, _ in nm)
    print(f"    all skipped groups distal? {dist_only}")
    for a, b, c in nm[:25]:
        print(f"      {a:32s} {b:8s} n_t={c}")

# also: fit-derived (weak) labels for comparison -- how many skips would THEY cause?
for r in rows:
    r["lab_backup"] = r["lab"]
    r["lab"] = r["weak"]
    r["conf"] = np.ones_like(r["conf"])
res = count_skips(0.0)
print(f"\n--- FIT-DERIVED (weak) labels, same threshold: {[f'{s}/{t}' for s, t, _ in res]}")
s, t, nm = res[0]
for a, b, c in nm[:25]:
    print(f"      {a:32s} {b:8s} n_t={c}")
