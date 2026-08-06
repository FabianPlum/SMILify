"""How many PCs should the new SMIL model keep?

Analyses the user's clean ALL_ANTS_CLEAN fit
(`Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING`, 81 specimens, template
`3D_model_prep/SMPL_fit.pkl` at 10,236 verts / 20 betas) and reports the component count for
90% explained variance -- plus the number that actually GENERALISES, which is the one that
should drive the decision.

WHY BOTH NUMBERS
Explained variance is measured IN-SAMPLE. With 81 specimens in ~30,000 dimensions, PCA can
always explain a large fraction of the training set; that says nothing about whether those
directions describe real shape or per-specimen fitting noise. Measured on the worker
registrations earlier in this work, 20 modes explained 63-68% in-sample while improving
held-out reconstruction by ~2% -- the modes were noise. So the recommendation here is
cross-checked with leave-one-out reconstruction, which is the same diagnostic
(Davies TMI 2002, Styner IPMI 2003) used in probe 19.

WHAT IS PCA'd
Three candidate inputs, because the answer differs sharply between them:
  posed      the `verts` array straight from the fit. WRONG for a shape space -- every
             specimen is in a different pose, so the leading components would encode pose.
             Included only to show the size of that error.
  unposed    v_template + deform_verts + shapedirs . betas, i.e. exactly what
             smal_torch.__call__ composes before skinning. This is the shape with pose
             removed and is the correct object.
  entangled  what `apply_entangled_pca_and_create_shapekeys` actually builds:
             concatenate [vertex_features (v*3), scale_data (j), translation_features (j*3)]
             with NO normalisation (the addon skips it when the magnitude ratio is < 10).
             This is what the Blender export will run, so it is the number to act on.

Note the addon expects `scale_data` of shape (n, j) -- one isotropic scale per joint -- while
the fit stores `log_beta_scales` as (n, j, 3), anisotropic. The mean over the three axes is
used here; if the addon is fed something else the entangled spectrum will shift slightly.
"""

import os
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
FIT = "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
TPL = os.path.join(REPO, "3D_model_prep", "SMPL_fit.pkl")


def spectrum(X):
    """Cumulative explained variance of PCA over rows of X (n, d)."""
    Xc = X - X.mean(0)
    S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    v = S**2
    return np.cumsum(v) / v.sum()


def loo_generalisation(X, kmax):
    """Leave-one-out reconstruction error with k modes, per-sample RMS, in input units."""
    n = X.shape[0]
    errs = np.full((n, kmax + 1), np.nan)
    for i in range(n):
        keep = np.ones(n, bool)
        keep[i] = False
        Y = X[keep]
        mu = Y.mean(0)
        _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        r = X[i] - mu
        errs[i, 0] = np.sqrt((r**2).mean())  # k=0: predict the mean
        for k in range(1, kmax + 1):
            B = Vt[:k]
            rec = (r @ B.T) @ B
            errs[i, k] = np.sqrt(((r - rec) ** 2).mean())
    return np.nanmean(errs, 0)


def main():
    d = np.load(FIT)
    with open(TPL, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        tpl = u.load()
    vt = np.asarray(tpl["v_template"], dtype=np.float64)
    sd = np.asarray(tpl["shapedirs"], dtype=np.float64)  # (V, 3, K)
    n = d["betas"].shape[0]
    V = vt.shape[0]
    print(f"[pc] {n} specimens, template {TPL.split('/')[-1]} ({V} verts, {sd.shape[-1]} betas)")

    betas = d["betas"].astype(np.float64)
    dv = d["deform_verts"].astype(np.float64)
    K = betas.shape[1]
    unposed = vt[None] + dv + np.einsum("bk,vck->bvc", betas, sd[:, :, :K])
    posed = d["verts"].astype(np.float64)

    scale = np.exp(d["log_beta_scales"].astype(np.float64)).mean(-1)  # (n, J) isotropic
    trans = d.get("betas_trans")
    trans = np.zeros((n, scale.shape[1], 3)) if trans is None else trans.astype(np.float64)
    ent = np.concatenate([unposed.reshape(n, -1), scale, trans.reshape(n, -1)], axis=1)

    sets = [
        ("posed verts (WRONG - encodes pose)", posed.reshape(n, -1)),
        ("unposed verts (shape only)", unposed.reshape(n, -1)),
        ("ENTANGLED verts+scale+trans (what the addon builds)", ent),
    ]

    kmax = min(40, n - 2)
    print()
    print(f"{'input':<52}{'PCs for 80%':>12}{'90%':>6}{'95%':>6}{'99%':>6}")
    res = {}
    for name, X in sets:
        c = spectrum(X)
        f = lambda t: int(np.searchsorted(c, t) + 1)  # noqa: E731
        res[name] = (c, loo_generalisation(X, kmax))
        print(f"{name:<52}{f(0.80):>12}{f(0.90):>6}{f(0.95):>6}{f(0.99):>6}")

    print()
    print("LEAVE-ONE-OUT reconstruction error (k=0 is 'predict the mean'). The useful number of")
    print("PCs is where this STOPS falling -- beyond that the modes fit per-specimen noise.")
    print(f"{'input':<52}{'k=0':>9}{'k=5':>9}{'k=10':>9}{'k=20':>9}{'k=40':>9}{'best k':>8}")
    for name, (c, g) in res.items():
        ks = [0, 5, 10, 20, min(40, len(g) - 1)]
        best = int(np.argmin(g))
        print(f"{name:<52}" + "".join(f"{g[k]:>9.5f}" for k in ks) + f"{best:>8}")

    # the recommendation, from the entangled input the addon will actually use
    c, g = res["ENTANGLED verts+scale+trans (what the addon builds)"]
    n90 = int(np.searchsorted(c, 0.90) + 1)
    best = int(np.argmin(g))
    knee = next((k for k in range(1, len(g)) if g[k] > 0.995 * g[k - 1]), len(g) - 1)
    print()
    print("=" * 84)
    print(f"  90% of explained variance needs           {n90} PCs")
    print(f"  leave-one-out error is minimised at       {best} PCs")
    print(f"  leave-one-out stops improving (<0.5%/PC)  {knee} PCs")
    print(f"  hard ceiling (n-1 with {n} specimens)      {n - 1} PCs")
    print("=" * 84)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for name, (c, g) in res.items():
        lab = name.split(" (")[0]
        ax[0].plot(np.arange(1, min(60, len(c)) + 1), 100 * c[: min(60, len(c))], marker="o", ms=2.5, label=lab)
        ax[1].plot(np.arange(len(g)), g / g[0], marker="o", ms=2.5, label=lab)
    ax[0].axhline(90, c="#c53030", ls="--", lw=1.5, label="90%")
    ax[0].set_xlabel("PCs")
    ax[0].set_ylabel("cumulative explained variance (%)")
    ax[0].set_title("In-sample variance (what '90%' usually means)")
    ax[1].axhline(1.0, c="#666", ls=":", lw=1.2, label="no better than the mean")
    ax[1].set_xlabel("PCs")
    ax[1].set_ylabel("LOO error / error at k=0")
    ax[1].set_title("Held-out reconstruction (what actually generalises)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    plt.tight_layout()
    p = os.path.join(HERE, "out", "pc_recommendation.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=115)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
