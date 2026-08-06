"""REFUTE PROBE — is the clean-vs-worker gen/spread gap a CORRESPONDENCE fact, or a
SPECTRAL-CONCENTRATION + IN-DISTRIBUTION fact?

The claim under test (prior agent, "SUPPORTS"):
    "the clean corpus's free-form deform_verts field is itself STRUCTURED (gen/spread 0.7932)
     while the workers' is PURE PER-SPECIMEN NOISE (0.9998, i.e. 10 PCA modes explain
     literally nothing out-of-sample) ... a gap that survives a matched control holding
     pipeline, losses, template, beta-freeze and corpus size constant (clean 0.7994 vs
     worker 0.9870 at n=50)."

ALTERNATIVE CAUSE being tested here:
  (i)  gen@k / spread is, by construction, a function of the SAMPLE COVARIANCE EIGEN-SPECTRUM
       and n.  Under an RMS-per-vertex residual it is EXACTLY invariant to any orthogonal
       transform of the 3V-dimensional ambient space -- including transforms that destroy
       vertex correspondence entirely.  So a number near 1.0 does not imply "noise"; it
       implies "variance not concentrated in <=k stably-estimable modes at this n".
  (ii) The clean corpus is in-distribution by construction (OmniAnt's shape space and
       template were built from it; REPORT.md 6.1/6.4.1), and the workers are contracted /
       folded and out of distribution.  That predicts exactly what we see: clean fits are
       far tighter, their residual field is low-rank; worker fits need a large, specimen-
       specific residual, which is high-rank.

SECTIONS
  1  CALIBRATION      reproduce the prior agent's headline numbers with their own estimator
  2  SPECTRUM         effective dimensionality (participation ratio), decay exponent
  3  ORTHOGONAL SCRAMBLE  destroy correspondence with an ambient orthogonal map; re-measure
  4  EXACT-CORRESPONDENCE SYNTHETIC  corpora with correspondence perfect BY CONSTRUCTION and
                      the measured spectra; plus a power-law sweep locating each real corpus
  5  IN-DISTRIBUTION  template distance, fit quality, shapedirs-span occupancy
"""

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, REPO)

LEGACY = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

CORPORA = [
    (
        "LIM_0",
        f"{HERE}/runs/LIM_0/Stage_3_deform_fine.npz",
        f"{REPO}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        "worker",
    ),
    (
        "BPX_noprior",
        f"{HERE}/runs/BPX_noprior/Stage_3_deform_fine.npz",
        f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl",
        "worker",
    ),
    (
        "M7_handoff_midline",
        f"{HERE}/runs/M7_handoff_midline/Stage_3_deform_fine.npz",
        f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl",
        "worker",
    ),
    ("baseline", f"{HERE}/runs/baseline/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl", "worker"),
    ("CLEAN81_M7", f"{HERE}/runs/CLEAN_M7/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl", "clean"),
    ("ALL_ANTS_CLEAN", LEGACY, f"{REPO}/3D_model_prep/SMPL_fit.pkl", "clean"),
]
KS = [1, 5, 10, 20, 40]


def load_model(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def parts(npz, tpl):
    d = np.load(npz, allow_pickle=True)
    dd = load_model(tpl)
    vt = np.asarray(dd["v_template"], dtype=np.float64)
    sd = np.asarray(dd["shapedirs"], dtype=np.float64)
    b = np.asarray(d["betas"], dtype=np.float64)
    F = np.asarray(d["deform_verts"], dtype=np.float64)
    P = np.einsum("bk,vck->bvc", b, sd[:, :, : b.shape[1]])
    return vt, P, F, d, dd


# ---------------------------------------------------------------- the estimator, verbatim
def loo_gen(X, ks):
    """probe-19 generalisation (prior agent's loo_gen, numpy port). X: (N,V,3).
    error = MEAN PER-VERTEX EUCLIDEAN distance of the LOO-PCA residual."""
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    kmax = max([k for k in ks if k <= N - 2] + [1])
    errs = np.full((N, kmax + 1), np.nan)
    for i in range(N):
        keep = np.ones(N, bool)
        keep[i] = False
        Y = Xf[keep]
        mu = Y.mean(0)
        _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        errs[i, 0] = np.linalg.norm(r.reshape(-1, 3), axis=-1).mean()
        for k in range(1, kmax + 1):
            B = Vt[:k]
            rec = (r @ B.T) @ B
            errs[i, k] = np.linalg.norm((r - rec).reshape(-1, 3), axis=-1).mean()
    m = np.nanmean(errs, 0)
    return {k: (float(m[k]) if k <= kmax else np.nan) for k in ks}


def pop_spread(X):
    return float(np.linalg.norm(X - X.mean(0), axis=-1).mean())


def loo_gen_rms(X, ks):
    """The same thing with an RMS (Frobenius) residual instead of mean-per-vertex-Euclidean.
    This variant is EXACTLY invariant to any orthogonal map of the 3V ambient space."""
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    kmax = max([k for k in ks if k <= N - 2] + [1])
    errs = np.full((N, kmax + 1), np.nan)
    for i in range(N):
        keep = np.ones(N, bool)
        keep[i] = False
        Y = Xf[keep]
        mu = Y.mean(0)
        _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        errs[i, 0] = np.linalg.norm(r)
        for k in range(1, kmax + 1):
            B = Vt[:k]
            errs[i, k] = np.linalg.norm(r - (r @ B.T) @ B)
    m = np.nanmean(errs, 0)
    return {k: (float(m[k]) if k <= kmax else np.nan) for k in ks}


def pop_spread_rms(X):
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    return float(np.linalg.norm(Xf - Xf.mean(0), axis=1).mean())


def spectrum(X):
    N = X.shape[0]
    Xc = (X - X.mean(0)).reshape(N, -1)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    ev = ev[ev > 0]
    f = ev / ev.sum()
    pr = float(1.0 / (f**2).sum())  # participation ratio = effective dimension
    j = np.arange(1, min(20, len(f)) + 1)
    alpha = float(-np.polyfit(np.log(j), np.log(f[: len(j)]), 1)[0])  # power-law decay
    return f, pr, alpha


def scramble(X, rng):
    """Random ORTHOGONAL map of the 3V ambient space: coordinate permutation + sign flips.
    Applied identically to every specimen, so it is a change of basis, not per-specimen noise.
    It destroys vertex correspondence semantics completely (coordinate v of the 'vertex'
    now comes from three unrelated places on the mesh)."""
    N, V, _ = X.shape
    Xf = X.reshape(N, -1).copy()
    p = rng.permutation(Xf.shape[1])
    s = rng.choice([-1.0, 1.0], Xf.shape[1])
    return (Xf[:, p] * s).reshape(N, V, 3)


def synth_from_spectrum(f, N, V, rng, scale=1.0):
    """Corpus with EXACT correspondence by construction: a Gaussian in a random orthonormal
    basis of the mesh's coordinate space with the supplied population spectrum.  There is no
    registration step anywhere, so correspondence cannot be at fault."""
    m = min(len(f), N - 1)
    lam = np.sqrt(f[:m])
    Z = rng.standard_normal((N, m)) * lam
    # random orthonormal directions in R^{3V}
    G = rng.standard_normal((m, 3 * V))
    Q, _ = np.linalg.qr(G.T)
    X = (Z @ Q.T) * scale
    return X.reshape(N, V, 3)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {}

    print("=" * 96)
    print("SECTION 1+2 — calibration of the prior agent's estimator, and the SPECTRUM of each corpus")
    print("=" * 96)
    print(
        f"{'corpus':<20}{'kind':<7}{'n':>4}{'effdim':>8}{'alpha':>7}{'cum@10':>8}"
        f"{'gen10/sp':>10}{'gen40/sp':>10}{'RMSgen10/sp':>13}"
    )
    store = {}
    for name, npz, tpl, kind in CORPORA:
        if not os.path.exists(npz):
            print(f"[skip] {name}")
            continue
        vt, P, F, d, dd = parts(npz, tpl)
        S = vt[None] + P + F
        N, _V = S.shape[0], S.shape[1]
        store[name] = dict(S=S, F=F, P=P, vt=vt, dd=dd, d=d, kind=kind)
        for lab, X in (("FREE(deform_verts)", F), ("SUM(rest shape)", S)):
            f, pr, alpha = spectrum(X)
            g = loo_gen(X, KS)
            sp = pop_spread(X)
            gr = loo_gen_rms(X, KS)
            spr = pop_spread_rms(X)
            cum10 = float(f[:10].sum())
            print(
                f"{name + ' ' + lab:<27}{kind:<7}{N:>4}{pr:>8.2f}{alpha:>7.2f}{cum10:>8.3f}"
                f"{g[10] / sp:>10.4f}{g[40] / sp:>10.4f}{gr[10] / spr:>13.4f}"
            )
            res.setdefault(name, {})[lab] = dict(
                n=N,
                kind=kind,
                effdim=pr,
                alpha=alpha,
                cum10=cum10,
                ratio={str(k): (g[k] / sp if np.isfinite(g[k]) else None) for k in KS},
                ratio_rms={str(k): (gr[k] / spr if np.isfinite(gr[k]) else None) for k in KS},
                spread=sp,
            )
        print()

    print("=" * 96)
    print("SECTION 3 — DESTROY CORRESPONDENCE, keep the spectrum: random orthogonal map of the")
    print("            3V ambient space (coordinate permutation + sign flip, same map for every")
    print("            specimen).  After this, 'vertex i' is a meaningless label.")
    print("=" * 96)
    print(
        f"{'corpus (free-form field)':<28}{'gen10/sp real':>15}{'scrambled':>12}{'RMS real':>11}{'RMS scrambled':>15}"
    )
    for name in store:
        F = store[name]["F"]
        sp, g = pop_spread(F), loo_gen(F, [10])
        spr, gr = pop_spread_rms(F), loo_gen_rms(F, [10])
        Fs = scramble(F, rng)
        sp2, g2 = pop_spread(Fs), loo_gen(Fs, [10])
        spr2, gr2 = pop_spread_rms(Fs), loo_gen_rms(Fs, [10])
        print(f"{name:<28}{g[10] / sp:>15.4f}{g2[10] / sp2:>12.4f}{gr[10] / spr:>11.4f}{gr2[10] / spr2:>15.4f}")
        res[name]["scramble"] = dict(
            real=g[10] / sp, scrambled=g2[10] / sp2, rms_real=gr[10] / spr, rms_scrambled=gr2[10] / spr2
        )

    print("\n" + "=" * 96)
    print("SECTION 4 — EXACT-CORRESPONDENCE SYNTHETIC CORPORA.  Gaussian samples in a random")
    print("            orthonormal basis carrying each corpus's own measured spectrum.  No")
    print("            registration exists, so correspondence is perfect by construction.")
    print("=" * 96)
    print(
        f"{'spectrum taken from':<28}{'n':>4}{'real gen10/sp':>15}{'synthetic gen10/sp':>21}"
        f"{'real@40':>10}{'synth@40':>11}"
    )
    for name in store:
        F = store[name]["F"]
        N, _V = F.shape[0], F.shape[1]
        f, pr, alpha = spectrum(F)
        sp, g = pop_spread(F), loo_gen(F, [10, 40])
        r10, r40 = [], []
        for t in range(3):
            Xs = synth_from_spectrum(f, N, 600, np.random.default_rng(100 + t))
            gs = loo_gen(Xs, [10, 40])
            sps = pop_spread(Xs)
            r10.append(gs[10] / sps)
            r40.append(gs[40] / sps if np.isfinite(gs[40]) else np.nan)
        print(
            f"{name:<28}{N:>4}{g[10] / sp:>15.4f}{np.mean(r10):>15.4f} +/-{np.std(r10):<5.4f}"
            f"{g[40] / sp:>10.4f}{np.nanmean(r40):>11.4f}"
        )
        res[name]["synthetic_exact_correspondence"] = dict(
            real10=g[10] / sp,
            synth10=float(np.mean(r10)),
            synth10_sd=float(np.std(r10)),
            real40=g[40] / sp,
            synth40=float(np.nanmean(r40)),
        )

    print("\n  power-law sweep (exact correspondence, n=50 and n=81, lambda_j ~ j^-alpha):")
    print(f"    {'alpha':>6}{'  n=50 gen10/sp':>17}{'  n=81 gen10/sp':>17}")
    sweep = {}
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0]:
        row = {}
        for N in (50, 81):
            f = np.arange(1, N) ** (-alpha)
            f = f / f.sum()
            Xs = synth_from_spectrum(f, N, 600, np.random.default_rng(7))
            row[N] = loo_gen(Xs, [10])[10] / pop_spread(Xs)
        print(f"    {alpha:>6.1f}{row[50]:>17.4f}{row[81]:>17.4f}")
        sweep[str(alpha)] = row
    res["powerlaw_sweep"] = {k: {str(n): v for n, v in r.items()} for k, r in sweep.items()}

    print("\n" + "=" * 96)
    print("SECTION 5 — IN-DISTRIBUTION-BY-CONSTRUCTION evidence")
    print("=" * 96)
    # template distance vs corpus spread, and shapedirs-span occupancy of the free-form field
    print(f"{'corpus':<20}{'|mean(F)| / spread':>20}{'F var inside shapedirs span':>30}{'top10 PCA inside span':>23}")
    for name in store:
        F, dd = store[name]["F"], store[name]["dd"]
        N, _V = F.shape[0], F.shape[1]
        sd = np.asarray(dd["shapedirs"], dtype=np.float64)
        K = sd.shape[2]
        B = sd.reshape(-1, K).T  # (K, 3V)
        Q, _ = np.linalg.qr(B.T)  # (3V, K) orthonormal basis of the span
        Ff = F.reshape(N, -1)
        Fc = Ff - Ff.mean(0)
        inside = float((np.linalg.norm(Fc @ Q, axis=1) ** 2).sum() / (np.linalg.norm(Fc, axis=1) ** 2).sum())
        # how much of the corpus's own leading 10 PCA modes lives in that span
        _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
        top = Vt[:10]
        inside_pc = float((np.linalg.norm(top @ Q, axis=1) ** 2).mean())
        meanF = float(np.linalg.norm(F.mean(0), axis=-1).mean()) / pop_spread(F)
        print(f"{name:<20}{meanF:>20.3f}{100 * inside:>29.1f}%{100 * inside_pc:>22.1f}%")
        res[name]["indist"] = dict(
            mean_field_over_spread=meanF, F_var_in_span=inside, top10_in_span=inside_pc, K=int(K)
        )

    json.dump(res, open(os.path.join(OUT, "refute25_spectral_confound.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/refute25_spectral_confound.json")


if __name__ == "__main__":
    main()
