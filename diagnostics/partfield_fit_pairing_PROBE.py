"""CPU-only check of the make_partfield_data.py fit<->mesh pairing claim."""

import os
import sys
import pickle
import numpy as np
import torch

REPO = "/home/fabi/dev/SMILify"
HERE = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
import config  # noqa
from fitter_3d.utils import load_meshes  # noqa
from fitter_3d.partfield import part_names, template_vertex_labels, normalise  # noqa
from pytorch3d.ops import knn_points, sample_points_from_meshes

dev = torch.device("cpu")
DEBRIS_THRESH = 0.035
MESH_DIR = os.path.join(HERE, "bench50")
NPZ = os.path.join(HERE, "runs", "M7_handoff_midline", "Stage_3_deform_fine.npz")

with open(config.SMAL_FILE, "rb") as fh:
    u = pickle._Unpickler(fh)
    u.encoding = "latin1"
    dd = u.load()
jn = [n.decode() if isinstance(n, bytes) else n for n in dd["J_names"]]
names = part_names(jn)
debris_id = names.index("debris")
vlab = torch.tensor(template_vertex_labels(dd["weights"], jn), device=dev)

fit = np.load(NPZ, allow_pickle=True)
n_spec = fit["betas"].shape[0]
files = sorted(f for f in os.listdir(MESH_DIR) if f.endswith(".obj"))
lab_npz = [str(x) for x in fit["labels"]]

print("assert len(files)==n_spec :", len(files) == n_spec)
print("npz labels == sorted(listdir) today :", lab_npz == files)

# --- simulate the claimed scenario: a same-length npz whose rows are in a different order
rs = np.random.default_rng(0)
perm = rs.permutation(n_spec)
print("\nSIMULATED RERUN (rows permuted, labels permuted with them)")
print("  len check still passes :", len(files) == n_spec)
print("  positional identity ok :", [lab_npz[p] for p in perm] == files)
mismatch = sum(lab_npz[perm[i]] != files[i] for i in range(n_spec))
print(f"  rows now naming a different specimen: {mismatch}/{n_spec}")

# --- quantify the real-weak damage on a subset
SUB = list(range(6))
fitv = torch.tensor(fit["verts"], dtype=torch.float32, device=dev)
_, meshes = load_meshes(mesh_dir=MESH_DIR, sorting=sorted, device="cpu")
torch.manual_seed(0)


def debris_frac(scan_i, fit_row):
    tgt = sample_points_from_meshes(meshes[scan_i], 30000)[0]
    both = torch.cat([tgt, fitv[fit_row]], 0)
    _, c, s = normalise(both)
    tn, fn = (tgt - c) / s, (fitv[fit_row] - c) / s
    kn = knn_points(tn.unsqueeze(0), fn.unsqueeze(0), K=1)
    lab = vlab[kn.idx[0, :, 0]].clone()
    d = kn.dists[0, :, 0].sqrt()
    frac = float((d > DEBRIS_THRESH).float().mean())
    lab[d > DEBRIS_THRESH] = debris_id
    return frac, lab


print("\n  specimen                                     paired%   mispaired%   label-agree%")
for i in SUB:
    j = int(perm[i])
    f_ok, l_ok = debris_frac(i, i)
    f_bad, l_bad = debris_frac(i, j)
    agree = float((l_ok == l_bad).float().mean())
    print(f"  {files[i][:42]:44s} {100 * f_ok:6.1f}  {100 * f_bad:10.1f}  {100 * agree:12.1f}")
