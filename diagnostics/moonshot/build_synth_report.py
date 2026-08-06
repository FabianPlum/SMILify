"""Extra figures for SYNTHETIC_ROUNDTRIP.md: the corpus gallery and the metric calibration.

Two things `score_synth_roundtrip.py` does not cover:

  CORPUS GALLERY   what the synthetic targets actually look like, clean beside noisy. Without
                   this the reader has to take on trust that the targets are ant-shaped and
                   that the noise level is comparable to real scan roughness.
  CALIBRATION      probe-19 gen/spread against DIRECT correspondence correctness, on the same
                   fits. Five experiments in this report were ranked by probe-19; this is the
                   first opportunity to check whether that ranking meant anything, because it
                   is the first time both a direct and an indirect measure exist together.
"""

import os
import pickle
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from pytorch3d.io import load_obj  # noqa: E402
from render_3d import DEV, make_renderer, render  # noqa: E402
from score_synth_roundtrip import score  # noqa: E402

PAIRS = [("SYN_clean", "synth_clean"), ("SYN_noisy", "synth_noisy")]


def load_pkl(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def gen_over_spread(npz_path, dd, k=10):
    """probe-19's measure on a given fit: LOO reconstruction error / population spread."""
    d = np.load(npz_path)
    vt = np.asarray(dd["v_template"], float)
    sd = np.asarray(dd["shapedirs"], float)
    b = d["betas"].astype(float)
    dv = d["deform_verts"].astype(float)
    K = b.shape[1]
    X = (vt[None] + dv + np.einsum("bk,vck->bvc", b, sd[:, :, :K])).reshape(len(b), -1)
    n = len(X)
    e0, ek = [], []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        Y = X[m]
        mu = Y.mean(0)
        _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        r = X[i] - mu
        e0.append(np.sqrt((r**2).mean()))
        rec = (r @ Vt[:k].T) @ Vt[:k]
        ek.append(np.sqrt(((r - rec) ** 2).mean()))
    return float(np.mean(ek) / np.mean(e0))


def corpus_gallery(n_show=4):
    renderer = make_renderer(image_size=380, elev=24.0, azim=140.0)
    fig, axes = plt.subplots(len(PAIRS), n_show, figsize=(3.4 * n_show, 3.4 * len(PAIRS)))
    axes = np.atleast_2d(axes)
    for r, (_run, corpus) in enumerate(PAIRS):
        d = os.path.join(HERE, corpus)
        files = sorted(f for f in os.listdir(d) if f.endswith(".obj"))[:n_show]
        for c, f in enumerate(files):
            v, fc, _ = load_obj(os.path.join(d, f), load_textures=False)
            v = v.to(DEV)
            v = (v - v.mean(0)) / v.abs().max()
            img = render(v, fc.verts_idx.to(DEV), torch.full((v.shape[0], 3), 0.65, device=DEV), renderer)
            axes[r, c].imshow(img)
            axes[r, c].set_title(f"{corpus} — {f[:-4]}", fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle("The synthetic corpus — targets generated from the model, so correspondence is known", fontsize=12)
    plt.tight_layout()
    p = os.path.join(HERE, "out", "synth_corpus.png")
    fig.savefig(p, dpi=115, bbox_inches="tight")
    print(f"wrote {p}")


def calibration(dd):
    """Direct correctness vs probe-19, the indirect metric five experiments were ranked by."""
    rows = []
    for run, corpus in PAIRS:
        p = os.path.join(HERE, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            continue
        r = score(run, corpus)
        if r is None:
            continue
        rows.append((run, 100 * np.concatenate([x["correct"] for x in r]).mean(), gen_over_spread(p, dd)))
    if not rows:
        print("[calib] no fits yet")
        return rows

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for run, corr, gen in rows:
        ax.scatter(gen, corr, s=90, label=f"{run}  ({corr:.1f}% correct, gen {gen:.3f})")
    for ref, lab, c in [
        (0.42, "synthetic exact-correspondence reference", "#2f855a"),
        (0.5383, "ALL_ANTS_CLEAN registrations", "#2b6cb0"),
        (0.9221, "best worker arm (LIM_0)", "#c53030"),
    ]:
        ax.axvline(ref, ls="--", lw=1.2, c=c, label=lab)
    ax.set_xlabel("probe-19 gen@10 / spread  (INDIRECT, lower = better)")
    ax.set_ylabel("% vertices with CORRECT correspondence  (DIRECT)")
    ax.set_title("Does the indirect metric track the direct one?", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    p = os.path.join(HERE, "out", "synth_calibration.png")
    fig.savefig(p, dpi=120)
    print(f"wrote {p}")
    for run, corr, gen in rows:
        print(f"  {run:<12} direct correctness {corr:6.2f}%   probe-19 gen/spread {gen:.4f}")
    return rows


def main():
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    dd = load_pkl(os.path.join(REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl"))
    corpus_gallery()
    calibration(dd)


if __name__ == "__main__":
    main()
