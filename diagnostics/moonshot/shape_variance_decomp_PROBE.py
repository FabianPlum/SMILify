"""PROBE — decompose the rest-space shape variance into PARAMETRIC (shapedirs.betas) and
FREE-FORM (deform_verts) parts, and run probe-19's leave-one-out generalisation metric on
each part separately, for the WORKER corpora and for the CLEAN corpus.

WHY
Worker registrations reach probe-19 gen/spread ~0.92-1.00 (no learnable structure); the
ALL_ANTS_CLEAN corpus reaches 0.5383@10 / 0.4586@40 with the SAME pipeline. A previous
measurement on workers found the free-form part carried 100.0% of the across-specimen
variance and generalised at 0.9794, while the parametric part generalised at 0.0008 (it is
a K-dimensional linear space, so that is near-tautological). The question this probe answers:

    does the CLEAN corpus also carry ~100% of its variance free-form?

If yes, then deform_verts is capturing consistent anatomy in one corpus and per-specimen
noise in the other, and the defect is localised to the free-form field, NOT to the
parametric/free-form split.

DEFINITIONS (stated exactly, because the headline is a ratio)
  rest-space shape   S_i = v_template + deform_verts_i + shapedirs . betas_i   (pose removed;
                     identical to smal_torch.py:263-267 and to probe_19.rest_shapes)
  parametric part    P_i = shapedirs . betas_i
  free-form part     F_i = deform_verts_i
  S = const + P + F, so Var(S) = Var(P) + Var(F) + 2 Cov(P,F), all three reported.
  total variance     TV(X) = sum over all V*3 entries of the across-specimen variance (ddof=1)
  gen@k              probe_19.generalisation: leave-one-out, PCA on the other N-1, k modes,
                     error = MEAN PER-VERTEX EUCLIDEAN DISTANCE of the residual.
  spread             the same quantity at k=0, i.e. mean-over-i of the mean per-vertex
                     distance from specimen i to the LOO mean of the others. This is the
                     natural normaliser: gen/spread = 1 means "k modes explain nothing".
                     A full-population (non-LOO) variant is printed alongside as a check.
"""

import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

OUT = os.path.join(HERE, "out")
DTYPE = torch.float64
DEV = torch.device("cpu")

CORPORA = [
    # name, npz, template pkl
    (
        "LIM_0",
        os.path.join(HERE, "runs/LIM_0/Stage_3_deform_fine.npz"),
        os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"),
    ),
    (
        "LIM_1x",
        os.path.join(HERE, "runs/LIM_1x/Stage_3_deform_fine.npz"),
        os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"),
    ),
    (
        "BPX_noprior",
        os.path.join(HERE, "runs/BPX_noprior/Stage_3_deform_fine.npz"),
        os.path.join(REPO, "3D_model_prep/SMIL_OmniAnt.pkl"),
    ),
    (
        "M7_handoff_midline",
        os.path.join(HERE, "runs/M7_handoff_midline/Stage_3_deform_fine.npz"),
        os.path.join(REPO, "3D_model_prep/SMIL_OmniAnt.pkl"),
    ),
    (
        "baseline",
        os.path.join(HERE, "runs/baseline/Stage_3_deform_fine.npz"),
        os.path.join(REPO, "3D_model_prep/SMIL_OmniAnt.pkl"),
    ),
    (
        "ALL_ANTS_CLEAN",
        "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz",
        os.path.join(REPO, "3D_model_prep/SMPL_fit.pkl"),
    ),
    # the same 81 CLEAN meshes re-registered by the CURRENT pipeline with the SAME recipe and
    # template as the worker arm M7_handoff_midline -- the apples-to-apples corpus contrast,
    # holding code, losses, template and beta-freeze constant and changing only the scans.
    (
        "CLEAN81_M7_currentpipe",
        os.path.join(HERE, "runs/CLEAN_M7/Stage_3_deform_fine.npz"),
        os.path.join(REPO, "3D_model_prep/SMIL_OmniAnt.pkl"),
    ),
    (
        "CLEAN81_hier_H3",
        os.path.join(HERE, "runs/CLEAN_hier/H3_deform.npz"),
        os.path.join(REPO, "3D_model_prep/SMIL_OmniAnt.pkl"),
    ),
]

KS = [1, 5, 10, 20, 40]


def load_model(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def parts(npz, dd):
    d = np.load(npz, allow_pickle=True)
    vt = torch.tensor(np.asarray(dd["v_template"], dtype=np.float64), dtype=DTYPE, device=DEV)
    sd = torch.tensor(np.asarray(dd["shapedirs"], dtype=np.float64), dtype=DTYPE, device=DEV)
    betas = torch.tensor(np.asarray(d["betas"], dtype=np.float64), dtype=DTYPE, device=DEV)
    F = torch.tensor(np.asarray(d["deform_verts"], dtype=np.float64), dtype=DTYPE, device=DEV)
    k = betas.shape[1]
    assert sd.shape[0] == F.shape[1], f"vert mismatch {sd.shape} vs {F.shape}"
    assert sd.shape[2] >= k, f"only {sd.shape[2]} shapedirs for {k} betas"
    P = torch.einsum("bk,vck->bvc", betas, sd[:, :, :k])
    return vt, P, F, betas, d


def total_var(X):
    """sum over all V*3 entries of the across-specimen variance (ddof=1)."""
    return float(((X - X.mean(0)) ** 2).sum() / (X.shape[0] - 1))


def cov_term(A, B):
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    return float((Ac * Bc).sum() / (A.shape[0] - 1))


def loo_gen(X, ks):
    """probe-19 generalisation + the k=0 spread, leave-one-out. Returns (spread, {k: gen})."""
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    kmax = max([k for k in ks if k <= N - 2] + [1])
    errs = np.full((N, kmax + 1), np.nan)  # column 0 = k=0 (spread)
    for i in range(N):
        keep = torch.ones(N, dtype=torch.bool)
        keep[i] = False
        Y = Xf[keep]
        mu = Y.mean(0)
        _, _, Vt = torch.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        errs[i, 0] = float(r.reshape(-1, 3).norm(dim=-1).mean())
        for k in range(1, kmax + 1):
            B = Vt[:k]
            rec = (r @ B.t()) @ B
            errs[i, k] = float((r - rec).reshape(-1, 3).norm(dim=-1).mean())
    m = np.nanmean(errs, 0)
    return float(m[0]), {k: (float(m[k]) if k <= kmax else float("nan")) for k in ks}


def pop_spread(X):
    """The HISTORICAL normaliser (calibrated: reproduces M7 0.01996 / 0.9870 exactly).
    Mean per-vertex distance to the full-population mean."""
    mu = X.mean(0)
    return float((X - mu).norm(dim=-1).mean())


def compactness(X, ks):
    N = X.shape[0]
    Xc = (X - X.mean(0)).reshape(N, -1)
    S = torch.linalg.svdvals(Xc)
    c = torch.cumsum((S**2) / (S**2).sum(), 0).cpu().numpy()
    return {k: (float(c[k - 1]) if k <= len(c) else float("nan")) for k in ks}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for name, npz, tpl in CORPORA:
        if not os.path.exists(npz):
            print(f"[skip] {name}: {npz} missing", flush=True)
            continue
        dd = load_model(tpl)
        vt, P, F, betas, d = parts(npz, dd)
        S = vt.unsqueeze(0) + P + F
        N, V = S.shape[0], S.shape[1]
        # mesh scale, for cross-corpus comparability of ABSOLUTE numbers
        bbox = float((vt.max(0).values - vt.min(0).values).norm())

        vP, vF, vS = total_var(P), total_var(F), total_var(S)
        cPF = cov_term(P, F)

        print(
            f"\n{'=' * 78}\n{name}   n={N}  V={V}  K={betas.shape[1]}  "
            f"template={os.path.basename(tpl)}  bbox_diag={bbox:.5f}",
            flush=True,
        )
        print(
            f"  betas sd (across specimens, pooled) = {float(betas.std(0).mean()):.6f}   "
            f"|beta| mean = {float(betas.abs().mean()):.6f}",
            flush=True,
        )
        print(
            f"  deform_verts mean magnitude         = {float(F.norm(dim=-1).mean()):.6f}  "
            f"({100 * float(F.norm(dim=-1).mean()) / bbox:.3f}% of bbox diag)",
            flush=True,
        )
        print("  --- across-specimen variance decomposition (sum over V*3 entries, ddof=1)")
        print(f"    Var(parametric  shapedirs.betas) = {vP:.6g}   share {100 * vP / vS:7.3f}%")
        print(f"    Var(free-form   deform_verts)    = {vF:.6g}   share {100 * vF / vS:7.3f}%")
        print(f"    2*Cov(param, free-form)          = {2 * cPF:.6g}   share {100 * 2 * cPF / vS:7.3f}%")
        print(f"    Var(sum S)                       = {vS:.6g}   [check {vP + vF + 2 * cPF:.6g}]")
        corr = cPF / max(np.sqrt(vP * vF), 1e-300)
        print(f"    corr(param, free-form)           = {corr:+.4f}", flush=True)

        entry = dict(
            n=N,
            V=V,
            K=int(betas.shape[1]),
            template=os.path.basename(tpl),
            bbox=bbox,
            betas_sd=float(betas.std(0).mean()),
            var_param=vP,
            var_free=vF,
            cov2=2 * cPF,
            var_sum=vS,
            share_param=100 * vP / vS,
            share_free=100 * vF / vS,
            share_cov=100 * 2 * cPF / vS,
            corr_param_free=corr,
            deform_mag=float(F.norm(dim=-1).mean()),
        )

        print(
            "  --- probe-19 LOO generalisation; ratio = gen / POPULATION spread "
            "(the historical normaliser, calibrated on M7)"
        )
        for lab, X in (("(a) parametric only", P), ("(b) free-form only", F), ("(c) sum (rest shape)", S)):
            sploo, gen = loo_gen(X, KS)
            ps = pop_spread(X)
            cmp_ = compactness(X, [5, 10])
            row = "  ".join(f"@{k}={gen[k]:.5f}({gen[k] / ps:.4f})" if np.isfinite(gen[k]) else f"@{k}=--" for k in KS)
            print(
                f"    {lab:<22} spread={ps:.6f} [LOO {sploo:.6f}]  "
                f"cmp@5={100 * cmp_[5]:5.1f}% cmp@10={100 * cmp_[10]:5.1f}%  gen(gen/spread) {row}",
                flush=True,
            )
            entry[lab.split(")")[0].strip("(")] = dict(
                spread=ps,
                loo_spread=sploo,
                compactness={str(k): cmp_[k] for k in (5, 10)},
                gen={str(k): gen[k] for k in KS},
                ratio={str(k): (gen[k] / ps if np.isfinite(gen[k]) else None) for k in KS},
            )
        res[name] = entry

    # ---- control: is the CLEAN advantage just n=81 vs the workers' n=50? subsample to 50.
    # Ratios here use the POPULATION spread of the SAME subsample, matching the main table.
    for cname, npz, tpl in [c for c in CORPORA if c[0].startswith(("ALL_ANTS_CLEAN", "CLEAN81"))]:
        if not os.path.exists(npz):
            continue
        dd = load_model(tpl)
        vt, P, F, betas, d = parts(npz, dd)
        S = vt.unsqueeze(0) + P + F
        print(
            f"\n{'=' * 78}\nCONTROL: {cname} subsampled to n=50 (5 draws) — is the clean advantage just corpus size?",
            flush=True,
        )
        sub = {"a": [], "b": [], "c": []}
        rng = np.random.default_rng(0)
        for _t in range(5):
            idx = torch.tensor(rng.choice(S.shape[0], 50, replace=False))
            for key, X in (("a", P[idx]), ("b", F[idx]), ("c", S[idx])):
                _sploo, gen = loo_gen(X, [10, 40])
                ps = pop_spread(X)
                sub[key].append((gen[10] / ps, gen[40] / ps))
        for key, lab in (("a", "parametric only"), ("b", "free-form only"), ("c", "sum")):
            arr = np.array(sub[key])
            print(
                f"    {lab:<22} gen/spread @10 = {arr[:, 0].mean():.4f} +/- {arr[:, 0].std():.4f}"
                f"   @40 = {arr[:, 1].mean():.4f} +/- {arr[:, 1].std():.4f}",
                flush=True,
            )
            res.setdefault(cname + "_sub50", {})[lab] = dict(
                r10=float(arr[:, 0].mean()),
                r10_sd=float(arr[:, 0].std()),
                r40=float(arr[:, 1].mean()),
                r40_sd=float(arr[:, 1].std()),
            )

    json.dump(res, open(os.path.join(OUT, "shape_variance_decomp.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(OUT, 'shape_variance_decomp.json')}")


if __name__ == "__main__":
    main()
