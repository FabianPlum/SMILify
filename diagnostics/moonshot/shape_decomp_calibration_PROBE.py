"""Calibration of the decomposition probe against the historically reported probe-19 numbers.

Reproduces, with probe_19_correspondence_quality.py's own code path (float32, same rest_shapes
formula), the published figures:
    M7_handoff_midline  spread 0.01996  gen@10 0.01970  ratio 0.9870
    ALL_ANTS_CLEAN      gen/spread @10 0.5383  @40 0.4586
and prints the float32-vs-float64 difference, so the decomposition numbers can be read against
them without ambiguity about the estimator.
"""

import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

CASES = [
    (
        "M7_handoff_midline",
        os.path.join(HERE, "runs/M7_handoff_midline/Stage_3_deform_fine.npz"),
        os.path.join(REPO, "3D_model_prep/SMIL_OmniAnt.pkl"),
    ),
    (
        "ALL_ANTS_CLEAN",
        "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz",
        os.path.join(REPO, "3D_model_prep/SMPL_fit.pkl"),
    ),
]


def rest(npz, tpl, dtype):
    with open(tpl, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    d = np.load(npz, allow_pickle=True)
    vt = torch.tensor(np.asarray(dd["v_template"], dtype=np.float64), dtype=dtype)
    sd = torch.tensor(np.asarray(dd["shapedirs"], dtype=np.float64), dtype=dtype)
    betas = torch.tensor(np.asarray(d["betas"]), dtype=dtype)
    dv = torch.tensor(np.asarray(d["deform_verts"]), dtype=dtype)
    k = betas.shape[1]
    return vt.unsqueeze(0) + dv + torch.einsum("bk,vck->bvc", betas, sd[:, :, :k])


def gen(X, ks):
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    kmax = max(ks)
    errs = np.full((N, kmax), np.nan)
    for i in range(N):
        keep = torch.ones(N, dtype=torch.bool)
        keep[i] = False
        Y = Xf[keep]
        mu = Y.mean(0)
        _, _, Vt = torch.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        for k in range(1, kmax + 1):
            B = Vt[:k]
            rec = (r @ B.t()) @ B
            errs[i, k - 1] = float((r - rec).reshape(-1, 3).norm(dim=-1).mean())
    m = np.nanmean(errs, 0)
    return {k: float(m[k - 1]) for k in ks}


for name, npz, tpl in CASES:
    for dt, lab in ((torch.float32, "float32"), (torch.float64, "float64")):
        X = rest(npz, tpl, dt)
        n = X.shape[0]
        ks = [k for k in (1, 5, 10, 20, 40) if k <= n - 2]
        g = gen(X, ks)
        sp_pop = float((X - X.mean(0)).norm(dim=-1).mean())
        row = "  ".join(f"@{k} {g[k]:.5f} ({g[k] / sp_pop:.4f})" for k in ks)
        print(f"{name:<16} {lab}  n={n}  pop-spread {sp_pop:.6f}   gen (gen/spread) {row}", flush=True)
