"""REFUTATION PROBE for the "clean corpus has a structured free-form field" headline.

WHAT IS BEING CHECKED (independently re-derived from the raw .npz + .pkl, numpy float64,
no reuse of shape_variance_decomp_PROBE.py code):

  R0  audit          array shapes, template pairing, dtypes, einsum axis convention
                     (shapedirs . betas recomputed with an explicit loop, not einsum)
  R1  reproduction   variance split (parametric / free-form / 2cov) and probe-19 LOO
                     gen/spread for P, F, S -- do the published numbers replicate?
  R2  normaliser     the published ratio divides a LEAVE-ONE-OUT numerator by a
                     FULL-POPULATION denominator. gen/LOO-spread is also reported, since
                     the LOO deviation is inflated by N/(N-1), which differs between the
                     n=50 worker corpora (1.0204) and the n=81 clean corpus (1.0125).
  R3  TRIVIAL MODES  how much of each corpus's "structure" is a per-specimen similarity
                     transform (centroid offset, isotropic scale, rigid rotation) rather
                     than anatomy? gen/spread recomputed after Procrustes alignment.
  R4  DUPLICATES     nearest-neighbour distance / population spread. If the clean corpus
                     contains near-twins (it has 'ectatomma-tuberculatum' AND
                     'ectatomma-tuberculatum (1)'), leave-one-out PCA generalises for a
                     reason that has nothing to do with correspondence quality.
  R5  FIELD SIZE     mean |deform_verts| for the MATCHED pair (CLEAN81_M7 vs worker M7),
                     which the original probe did not report -- it compared LIM_0 against
                     the LEGACY clean fit instead.
  R6  IN-DISTRIB     fraction of each corpus's free-form field that lies inside the
                     model's own shapedirs span, plus the fit residual (chamfer) from
                     metrics.csv. The repo's own preprocess_meshes.py docstring states the
                     clean corpus is in-distribution by construction ("OmniAnt's own shape
                     space was built with the baseline workflow on that exact corpus").

Everything is CPU/numpy. Output -> shape_decomp_REFUTE_PROBE_out.txt
"""

import csv
import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")

CLEAN_LEGACY = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

CORPORA = [
    # name, npz, template, family
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
    ("ALL_ANTS_CLEAN", CLEAN_LEGACY, f"{REPO}/3D_model_prep/SMPL_fit.pkl", "clean"),
    ("CLEAN81_M7", f"{HERE}/runs/CLEAN_M7/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl", "clean"),
]
KS = [1, 5, 10, 20, 40]


def load_pkl(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def load(npz, tpl):
    d = np.load(npz, allow_pickle=True)
    dd = load_pkl(tpl)
    vt = np.asarray(dd["v_template"], dtype=np.float64)
    sd = np.asarray(dd["shapedirs"], dtype=np.float64)
    betas = np.asarray(d["betas"], dtype=np.float64)
    F = np.asarray(d["deform_verts"], dtype=np.float64)
    K = betas.shape[1]
    assert sd.shape[0] == F.shape[1] == vt.shape[0], (sd.shape, F.shape, vt.shape)
    assert sd.shape[1] == 3 and sd.shape[2] >= K, sd.shape
    # explicit matmul, NOT einsum: (N,K) @ (K, V*3) -> (N,V,3)
    P = (betas @ sd[:, :, :K].reshape(-1, K).T).reshape(betas.shape[0], sd.shape[0], 3)
    return dict(
        npz=npz,
        tpl=tpl,
        d=d,
        dd=dd,
        vt=vt,
        sd=sd,
        betas=betas,
        F=F,
        P=P,
        S=vt[None] + F + P,
        labels=[str(x) for x in d["labels"]],
        K=K,
    )


# ---------------------------------------------------------------- metrics
def tvar(X):
    return float(((X - X.mean(0)) ** 2).sum() / (X.shape[0] - 1))


def cov2(A, B):
    return float(2 * ((A - A.mean(0)) * (B - B.mean(0))).sum() / (A.shape[0] - 1))


def pop_spread(X):
    return float(np.linalg.norm(X - X.mean(0), axis=-1).mean())


def loo(X, ks=KS):
    """probe-19 generalisation. Returns (loo_spread, {k: gen}). float64, numpy SVD."""
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    kmax = min(max(ks), N - 2)
    e0 = np.zeros(N)
    ek = np.zeros((N, kmax + 1))
    for i in range(N):
        m = np.ones(N, bool)
        m[i] = False
        Y = Xf[m]
        mu = Y.mean(0)
        _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        e0[i] = np.linalg.norm(r.reshape(-1, 3), axis=-1).mean()
        for k in range(1, kmax + 1):
            B = Vt[:k]
            res = r - (r @ B.T) @ B
            ek[i, k] = np.linalg.norm(res.reshape(-1, 3), axis=-1).mean()
    return float(e0.mean()), {k: (float(ek[:, k].mean()) if k <= kmax else np.nan) for k in ks}


def report_loo(tag, X, out):
    ls, g = loo(X)
    ps = pop_spread(X)
    row = "  ".join(f"@{k}={g[k] / ps:.4f}" if np.isfinite(g[k]) else f"@{k}=--" for k in KS)
    row2 = "  ".join(f"@{k}={g[k] / ls:.4f}" if np.isfinite(g[k]) else f"@{k}=--" for k in KS)
    out(f"    {tag:<34} popspread={ps:.6f} loospread={ls:.6f} (ratio {ls / ps:.4f})")
    out(f"      gen/POPspread (published defn) {row}")
    out(f"      gen/LOOspread (self-consistent) {row2}")
    return dict(
        pop=ps,
        loo_spread=ls,
        gen={str(k): g[k] for k in KS},
        r_pop={str(k): g[k] / ps for k in KS},
        r_loo={str(k): g[k] / ls for k in KS},
    )


# ------------------------------------------------- similarity-transform removal
def procrustes_align(X, iters=3, scale=True, rot=True):
    """Remove per-specimen translation (+ optional isotropic scale, rigid rotation)
    by generalised Procrustes to the running mean. Returns aligned copy."""
    A = X - X.mean(1, keepdims=True)  # translation always removed
    for _ in range(iters):
        mu = A.mean(0)
        mu = mu - mu.mean(0)
        for i in range(A.shape[0]):
            Ai = A[i]
            if rot:
                U, S, Vt = np.linalg.svd(Ai.T @ mu)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
                Ai = Ai @ R
            if scale:
                num = float((Ai * mu).sum())
                den = float((Ai * Ai).sum())
                Ai = Ai * (num / max(den, 1e-300))
            A[i] = Ai
    return A


def main():
    os.makedirs(OUT, exist_ok=True)
    fh = open(os.path.join(HERE, "shape_decomp_REFUTE_PROBE_out.txt"), "w")

    def out(s=""):
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()

    res = {}
    store = {}

    out("=" * 100)
    out("R0  AUDIT — shapes, template pairing, einsum convention")
    out("=" * 100)
    for name, npz, tpl, fam in CORPORA:
        if not os.path.exists(npz):
            out(f"  [MISSING] {name}: {npz}")
            continue
        c = load(npz, tpl)
        store[name] = c
        d = c["d"]
        out(
            f"  {name:<20} n={c['S'].shape[0]:3d} V={c['S'].shape[1]:6d} K={c['K']:3d} "
            f"tpl={os.path.basename(tpl):<32} keys={sorted(d.files)}"
        )
        # independent check of the shapedirs contraction on one specimen, explicit loop
        i = 0
        ref = np.zeros((c["sd"].shape[0], 3))
        for k in range(c["K"]):
            ref += c["betas"][i, k] * c["sd"][:, :, k]
        err = np.abs(ref - c["P"][i]).max()
        out(f"      explicit-loop check of shapedirs.betas on specimen 0: max abs diff {err:.3e}")

    out()
    out("=" * 100)
    out("R1 + R2  variance split and LOO generalisation, re-derived (numpy float64)")
    out("=" * 100)
    for name, npz, tpl, fam in CORPORA:
        if name not in store:
            continue
        c = store[name]
        P, F, S, vt = c["P"], c["F"], c["S"], c["vt"]
        vP, vF, vS = tvar(P), tvar(F), tvar(S)
        c2 = cov2(P, F)
        bbox = float(np.linalg.norm(vt.max(0) - vt.min(0)))
        out(f"\n  {name}  [{fam}]  n={S.shape[0]} K={c['K']} bbox={bbox:.4f}")
        out(f"    betas sd (pooled)  {c['betas'].std(0, ddof=1).mean():.6f}")
        out(
            f"    |deform_verts|     {np.linalg.norm(F, axis=-1).mean():.6f}  "
            f"({100 * np.linalg.norm(F, axis=-1).mean() / bbox:.3f}% bbox)"
        )
        out(f"    Var parametric {vP:12.6g}  {100 * vP / vS:7.3f}%")
        out(f"    Var free-form  {vF:12.6g}  {100 * vF / vS:7.3f}%")
        out(f"    2*Cov          {c2:12.6g}  {100 * c2 / vS:7.3f}%   [sum check {vP + vF + c2:.6g} vs {vS:.6g}]")
        e = {}
        for tag, X in (("(a) parametric P", P), ("(b) free-form F", F), ("(c) sum S", S)):
            e[tag] = report_loo(tag, X, out)
        res[name] = dict(
            fam=fam,
            n=int(S.shape[0]),
            K=int(c["K"]),
            bbox=bbox,
            share_param=100 * vP / vS,
            share_free=100 * vF / vS,
            share_cov=100 * c2 / vS,
            deform_mag=float(np.linalg.norm(F, axis=-1).mean()),
            loo=e,
        )

    out()
    out("=" * 100)
    out("R3  TRIVIAL MODES — is the clean corpus's 'structure' a similarity transform?")
    out("    S_raw      : as published")
    out("    S_transl   : per-specimen centroid removed")
    out("    S_scale    : centroid + isotropic scale removed")
    out("    S_proc     : centroid + scale + rigid rotation removed (GPA, 3 iters)")
    out("=" * 100)
    for name in store:
        c = store[name]
        S = c["S"]
        out(f"\n  {name}  [n={S.shape[0]}]")
        # how big is the per-specimen centroid / scale variation, in spread units
        cen = S.mean(1)
        cen_sd = float(np.linalg.norm(cen - cen.mean(0), axis=-1).mean())
        rad = np.linalg.norm(S - cen[:, None], axis=-1).mean(1)
        out(f"    centroid scatter {cen_sd:.6f}  = {100 * cen_sd / pop_spread(S):6.2f}% of pop spread")
        out(f"    mean radius per specimen: mean {rad.mean():.5f} CV {rad.std(ddof=1) / rad.mean():.4f}")
        variants = {
            "S_raw": S,
            "S_transl": S - S.mean(1, keepdims=True),
            "S_scale": procrustes_align(S.copy(), scale=True, rot=False),
            "S_proc": procrustes_align(S.copy(), scale=True, rot=True),
        }
        res[name]["trivial"] = {}
        for tag, X in variants.items():
            res[name]["trivial"][tag] = report_loo(tag, X, out)
        # same for the free-form field alone
        F = c["F"]
        res[name]["trivial_F"] = {}
        for tag, X in (("F_raw", F), ("F_transl", F - F.mean(1, keepdims=True))):
            res[name]["trivial_F"][tag] = report_loo(tag, X, out)

    out()
    out("=" * 100)
    out("R4  NEAR-DUPLICATES — nearest-neighbour distance / population spread")
    out("=" * 100)
    for name in store:
        c = store[name]
        S = c["S"]
        N = S.shape[0]
        _Xf = (S - S.mean(0)).reshape(N, -1)
        # per-vertex mean euclidean distance between every pair
        D = np.zeros((N, N))
        for i in range(N):
            diff = (S[i][None] - S).reshape(N, -1, 3)
            D[i] = np.linalg.norm(diff, axis=-1).mean(1)
        np.fill_diagonal(D, np.inf)
        nn = D.min(1)
        nnidx = D.argmin(1)
        ps = pop_spread(S)
        out(
            f"\n  {name}: pop spread {ps:.6f}; NN dist mean {nn.mean():.6f} "
            f"({nn.mean() / ps:.4f} x spread), median {np.median(nn):.6f}, min {nn.min():.6f}"
        )
        ordr = np.argsort(nn)[:6]
        for i in ordr:
            out(
                f"      closest pair: {c['labels'][i]:<45} <-> {c['labels'][nnidx[i]]:<45} "
                f"d={nn[i]:.6f} ({nn[i] / ps:.3f}x)"
            )
        res[name]["nn_over_spread"] = float(nn.mean() / ps)

    out()
    out("=" * 100)
    out("R6  IN-DISTRIBUTION — free-form field size and how much of it lies in shapedirs span")
    out("=" * 100)
    for name in store:
        c = store[name]
        F, sd, K = c["F"], c["sd"], c["K"]
        N, _V = F.shape[0], F.shape[1]
        B = sd[:, :, :K].reshape(-1, K)  # (V*3, K)
        Q, _ = np.linalg.qr(B)
        Fc = (F - F.mean(0)).reshape(N, -1)
        proj = Fc @ Q
        frac = float((proj**2).sum() / max((Fc**2).sum(), 1e-300))
        out(
            f"  {name:<20} |F| {np.linalg.norm(F, axis=-1).mean():.6f}   "
            f"frac of centred free-form variance inside span(shapedirs[:{K}]) = {100 * frac:6.3f}%"
        )
        res[name]["F_in_shapedirs_span"] = frac
        # chamfer from metrics.csv where available
        mcsv = os.path.join(os.path.dirname(c["npz"]), "metrics.csv")
        if os.path.exists(mcsv):
            rows = list(csv.DictReader(open(mcsv)))
            st = [r for r in rows if r["stage"] == "Stage_3_deform_fine"] or rows
            ch = np.array([float(r["chamfer_l1"]) for r in st])
            dm = np.array([float(r["deform_mag_mean"]) for r in st])
            out(
                f"      metrics.csv stage={st[0]['stage']}  chamfer_l1 mean {ch.mean():.6f}   "
                f"deform_mag_mean {dm.mean():.6f}"
            )
            res[name]["chamfer_l1"] = float(ch.mean())

    json.dump(res, open(os.path.join(OUT, "shape_decomp_REFUTE.json"), "w"), indent=1)
    out(f"\nwrote {os.path.join(OUT, 'shape_decomp_REFUTE.json')}")
    fh.close()


if __name__ == "__main__":
    main()
