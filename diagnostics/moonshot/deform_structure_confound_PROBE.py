"""PROBE -- adversarial STATISTICAL re-examination of the claim

    "the clean corpus's free-form deform_verts field is STRUCTURED (gen/spread 0.7932) while
     the workers' is pure per-specimen noise (0.9998), a gap that survives a matched control
     holding pipeline, losses, template, beta-freeze and corpus size constant"

The prior report controlled for corpus SIZE. This probe asks whether the surviving gap is
larger than the remaining sources of variation and whether it survives four confounds that
were NOT controlled:

  A. CALIBRATION           reproduce the published ratios exactly (else nothing else counts).
  B. UNCERTAINTY           per-specimen distribution, outlier leverage, bootstrap CI over
                           specimens, and a REPEATED-SUBSAMPLE control with 40 draws instead
                           of 5 (the published +/-0.0112 came from 5 heavily-overlapping
                           draws of 50-from-81, which understates corpus-level variability).
  C. PHYLOGENETIC LEAKAGE  leave-one-out lets a congener (or an outright duplicate --
                           ALL_ANTS_CLEAN contains BOTH "ectatomma-tuberculatum.obj" and
                           "ectatomma-tuberculatum (1).obj", plus an unnamed 01..20 block that
                           is 25% of the corpus) sit in the training fold. Redo the metric
                           leave-one-GENUS-out, and with the 01..20 block treated as one unit.
  D. SNR / FIELD MAGNITUDE the DECISIVE matched control is not magnitude-matched: the clean
                           free-form field is 0.00888 mean magnitude, the worker's 0.02051 --
                           2.3x. gen/spread is a signal-to-TOTAL ratio, so it falls
                           mechanically as incoherent error is added at fixed structure.
                           Test: inject iid noise into the CLEAN field, sweep amplitude, and
                           ask what magnitude is needed to reproduce the worker's ratio.
  E. BASIS TRANSFER        M7 (worker) and CLEAN81_M7 share topology (10229 verts, same
                           template). Cross-apply each corpus's top-k PCA basis to the other.
  F. IN-DISTRIBUTION       preprocess_meshes.py states the shape space was built from the
                           clean corpus. Test it: is v_template the clean corpus mean? do the
                           shapedirs span the clean corpus's own PCA subspace?
  G. SPECIMEN SIZE         how much of each corpus's free-form structure is a global uniform
                           scale mode (the confound named in the task)?
  H. METRIC CHOICE         the published estimator is MEAN PER-VERTEX EUCLIDEAN DISTANCE, an
                           L1-flavoured statistic that rewards "few large smooth modes" and
                           punishes distributed variation. Recompute with the variance
                           (energy) estimator and see whether the gap has the same size.

Definitions are copied verbatim from shape_variance_decomp_PROBE.py so the numbers are
directly comparable.
"""

import os
import pickle
import re
import sys

import numpy as np
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

DTYPE = torch.float64
CLEAN_LEGACY = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

CORPORA = {
    "LIM_0": (f"{HERE}/runs/LIM_0/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"),
    "M7_worker": (f"{HERE}/runs/M7_handoff_midline/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl"),
    "BPX_noprior": (f"{HERE}/runs/BPX_noprior/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl"),
    "CLEAN81_M7": (f"{HERE}/runs/CLEAN_M7/Stage_3_deform_fine.npz", f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl"),
    "ALL_ANTS_CLEAN": (CLEAN_LEGACY, f"{REPO}/3D_model_prep/SMPL_fit.pkl"),
}


def load_model(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def load_corpus(name):
    npz, tpl = CORPORA[name]
    dd = load_model(tpl)
    d = np.load(npz, allow_pickle=True)
    vt = torch.tensor(np.asarray(dd["v_template"], dtype=np.float64), dtype=DTYPE)
    sd = torch.tensor(np.asarray(dd["shapedirs"], dtype=np.float64), dtype=DTYPE)
    betas = torch.tensor(np.asarray(d["betas"], dtype=np.float64), dtype=DTYPE)
    F = torch.tensor(np.asarray(d["deform_verts"], dtype=np.float64), dtype=DTYPE)
    P = torch.einsum("bk,vck->bvc", betas, sd[:, :, : betas.shape[1]])
    labels = [str(x) for x in d["labels"]]
    return dict(name=name, vt=vt, sd=sd, betas=betas, F=F, P=P, S=vt.unsqueeze(0) + P + F, labels=labels, dd=dd)


# ---------------------------------------------------------------- published estimator
def pop_spread(X):
    return float((X - X.mean(0)).norm(dim=-1).mean())


def loo_gen_blocked(X, ks, blocks=None):
    """Leave-one-out, or leave-one-BLOCK-out if `blocks` (array of group ids) is given.
    Returns {k: mean per-vertex residual distance}. Same estimator as the published probe."""
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    blocks = np.arange(N) if blocks is None else np.asarray(blocks)
    kmax = max(ks)
    errs = np.full((N, kmax + 1), np.nan)
    for i in range(N):
        keep = torch.tensor(blocks != blocks[i])
        if int(keep.sum()) < kmax + 2:
            continue
        Y = Xf[keep]
        mu = Y.mean(0)
        _, _, Vt = torch.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        errs[i, 0] = float(r.reshape(-1, 3).norm(dim=-1).mean())
        for k in ks:
            B = Vt[:k]
            rec = (r @ B.t()) @ B
            errs[i, k] = float((r - rec).reshape(-1, 3).norm(dim=-1).mean())
    return errs


def energy_loo(X, ks, blocks=None):
    """ENERGY estimator: 1 - explained variance fraction, out of sample. Uses squared error,
    which is what 'variance explained' actually means, unlike mean per-vertex distance."""
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    blocks = np.arange(N) if blocks is None else np.asarray(blocks)
    num = {k: [] for k in ks}
    den = []
    for i in range(N):
        keep = torch.tensor(blocks != blocks[i])
        if int(keep.sum()) < max(ks) + 2:
            continue
        Y = Xf[keep]
        mu = Y.mean(0)
        _, _, Vt = torch.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        den.append(float((r**2).sum()))
        for k in ks:
            B = Vt[:k]
            rec = (r @ B.t()) @ B
            num[k].append(float(((r - rec) ** 2).sum()))
    return {k: float(np.sum(num[k]) / np.sum(den)) for k in ks}


def genus_of(lab):
    s = lab.replace(".obj", "").strip()
    if re.fullmatch(r"\d+", s):
        return "NUMBERED_BLOCK"
    s = re.sub(r"\s*\(\d+\)$", "", s)
    if "_" in s:  # worker: Genus_species_CASENT....
        return s.split("_")[0].lower()
    return s.split("-")[0].lower()  # clean: genus-species


def report(tag, X, ks, blocks=None, out=print):
    ps = pop_spread(X)
    errs = loo_gen_blocked(X, ks, blocks)
    m = np.nanmean(errs, 0)
    n_used = int(np.isfinite(errs[:, ks[0]]).sum())
    row = "  ".join(f"@{k}={m[k] / ps:.4f}" for k in ks)
    out(f"    {tag:<34} n={n_used:3d} spread={ps:.6f}  gen/spread {row}")
    return errs, ps


def main():
    ks = [10, 20, 40]
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    C = {k: load_corpus(k) for k in CORPORA}

    # ================================================================== A. CALIBRATION
    out("=" * 90)
    out("A. CALIBRATION -- reproduce the published free-form gen/spread numbers")
    out("=" * 90)
    pub = {"LIM_0": 0.9998, "M7_worker": 0.9870, "BPX_noprior": 0.9988, "CLEAN81_M7": 0.7792, "ALL_ANTS_CLEAN": 0.7932}
    base = {}
    for k, c in C.items():
        errs, ps = report(f"{k} free-form", c["F"], [10, 20, 40], None, out)
        r10 = float(np.nanmean(errs[:, 10])) / ps
        base[k] = dict(errs=errs, ps=ps, r10=r10)
        out(
            f"        published @10 = {pub[k]:.4f}   here = {r10:.4f}   "
            f"delta = {r10 - pub[k]:+.4f}   mean|F| = {float(c['F'].norm(dim=-1).mean()):.6f}"
        )
    out()

    # ================================================================== B. UNCERTAINTY
    out("=" * 90)
    out("B. UNCERTAINTY -- is the gap bigger than the sampling noise?")
    out("=" * 90)
    out("  per-specimen ratio e_i(10)/spread : distribution, outlier leverage, bootstrap CI")
    for k in C:
        e = base[k]["errs"][:, 10] / base[k]["ps"]
        n = len(e)
        rng = np.random.default_rng(1)
        bs = np.array([e[rng.integers(0, n, n)].mean() for _ in range(4000)])
        # jackknife: largest single-specimen leverage on the mean
        jk = np.array([np.delete(e, i).mean() for i in range(n)])
        out(
            f"    {k:<16} mean={e.mean():.4f} median={np.median(e):.4f} sd={e.std(ddof=1):.4f} "
            f"min={e.min():.4f} max={e.max():.4f}  boot95=[{np.quantile(bs, 0.025):.4f},"
            f"{np.quantile(bs, 0.975):.4f}]  max|jackknife shift|={np.abs(jk - e.mean()).max():.4f}"
        )
    out()
    out("  REPEATED SUBSAMPLE of the clean corpora to n=50, 40 draws (published used 5):")
    for k in ("CLEAN81_M7", "ALL_ANTS_CLEAN"):
        F = C[k]["F"]
        rng = np.random.default_rng(7)
        vals = []
        for _ in range(40):
            idx = torch.tensor(rng.choice(F.shape[0], 50, replace=False))
            X = F[idx]
            vals.append(float(np.nanmean(loo_gen_blocked(X, [10])[:, 10])) / pop_spread(X))
        v = np.array(vals)
        out(
            f"    {k:<16} n=50 free-form gen/spread@10 = {v.mean():.4f} +/- {v.std(ddof=1):.4f} "
            f"(range {v.min():.4f}-{v.max():.4f}, 40 draws)"
        )
    out()

    # ========================================================== C. PHYLOGENETIC LEAKAGE
    out("=" * 90)
    out("C. PHYLOGENETIC / DUPLICATE LEAKAGE -- leave-one-GENUS-out instead of one-specimen")
    out("=" * 90)
    for k, c in C.items():
        g = [genus_of(x) for x in c["labels"]]
        u, cnt = np.unique(g, return_counts=True)
        multi = {a: int(b) for a, b in zip(u, cnt) if b > 1}
        out(f"    {k:<16} {len(u)} groups for {len(g)} specimens; multi-specimen groups: {multi}")
    out()
    for k, c in C.items():
        g = np.array([genus_of(x) for x in c["labels"]])
        report(f"{k} free-form LOGO", c["F"], ks, g, out)
    out()
    out("  clean-corpus variant: 01..20 numbered block treated as 20 INDEPENDENT specimens")
    for k in ("CLEAN81_M7", "ALL_ANTS_CLEAN"):
        c = C[k]
        g = np.array([genus_of(x) for x in c["labels"]], dtype=object)
        for i, lab in enumerate(c["labels"]):
            if g[i] == "NUMBERED_BLOCK":
                g[i] = f"NUM_{i}"
        report(f"{k} LOGO (block split)", c["F"], ks, g, out)
    out()
    out("  and: clean corpus with the 20 numbered meshes REMOVED entirely (n=61)")
    for k in ("CLEAN81_M7", "ALL_ANTS_CLEAN"):
        c = C[k]
        keep = np.array([genus_of(x) != "NUMBERED_BLOCK" for x in c["labels"]])
        Xs = c["F"][torch.tensor(keep)]
        gg = np.array([genus_of(x) for x in np.array(c["labels"])[keep]])
        report(f"{k} n=61 plain LOO", Xs, ks, None, out)
        report(f"{k} n=61 LOGO", Xs, ks, gg, out)
    out()
    out("  nearest-neighbour leakage: corr(LOO error@10, distance to nearest other specimen)")
    for k, c in C.items():
        F = c["F"].reshape(c["F"].shape[0], -1)
        D = torch.cdist(F, F)
        D.fill_diagonal_(float("inf"))
        nn = D.min(1).values.numpy()
        e = base[k]["errs"][:, 10]
        out(
            f"    {k:<16} corr = {np.corrcoef(nn, e)[0, 1]:+.3f}   "
            f"NN-dist mean={nn.mean():.4f} min={nn.min():.4f} "
            f"(min/mean = {nn.min() / nn.mean():.3f})"
        )
    out()

    # ==================================================== D. SNR / FIELD-MAGNITUDE NULL
    out("=" * 90)
    out("D. SNR CONFOUND -- inject iid noise into the CLEAN free-form field and sweep")
    out("=" * 90)
    out("  Question: gen/spread is signal/TOTAL. The decisive matched control compares a")
    out("  0.00888-magnitude clean field against a 0.02051-magnitude worker field (2.3x).")
    out("  How much incoherent error must be added to the CLEAN field to make it look like")
    out("  the worker field on this metric?")
    F = C["CLEAN81_M7"]["F"]
    rng = np.random.default_rng(3)
    out(
        f"    worker M7 reference: mean|F| = {float(C['M7_worker']['F'].norm(dim=-1).mean()):.6f}  "
        f"gen/spread@10 = {base['M7_worker']['r10']:.4f}"
    )
    for sig in [0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020]:
        Z = F + torch.tensor(rng.normal(0, sig, F.shape), dtype=DTYPE)
        m = float(Z.norm(dim=-1).mean())
        e = loo_gen_blocked(Z, [10, 40])
        ps = pop_spread(Z)
        out(
            f"    sigma={sig:.3f}  mean|F|={m:.6f}  gen/spread @10={np.nanmean(e[:, 10]) / ps:.4f}"
            f"  @40={np.nanmean(e[:, 40]) / ps:.4f}"
        )
    out()
    out("  same sweep on the worker field, for symmetry (does it start from a floor?):")
    Fw = C["M7_worker"]["F"]
    for sig in [0.0, 0.005, 0.010]:
        Z = Fw + torch.tensor(rng.normal(0, sig, Fw.shape), dtype=DTYPE)
        e = loo_gen_blocked(Z, [10])
        out(
            f"    sigma={sig:.3f}  mean|F|={float(Z.norm(dim=-1).mean()):.6f}  "
            f"gen/spread@10={np.nanmean(e[:, 10]) / pop_spread(Z):.4f}"
        )
    out()

    # ============================================================== E. BASIS TRANSFER
    out("=" * 90)
    out("E. BASIS TRANSFER -- M7_worker and CLEAN81_M7 share topology and template")
    out("=" * 90)
    A, B = C["M7_worker"]["F"], C["CLEAN81_M7"]["F"]
    assert A.shape[1] == B.shape[1]

    def basis(X, k):
        Xc = (X - X.mean(0)).reshape(X.shape[0], -1)
        _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
        return Vt[:k], X.mean(0).reshape(-1)

    for k in (10, 20):
        Ba, mua = basis(A, k)
        Bb, mub = basis(B, k)
        for nm, X, Bz, muz, src in (
            ("worker by CLEAN basis", A, Bb, mub, "clean"),
            ("worker by WORKER basis", A, Ba, mua, "worker"),
            ("clean  by WORKER basis", B, Ba, mua, "worker"),
            ("clean  by CLEAN basis", B, Bb, mub, "clean"),
        ):
            R = X.reshape(X.shape[0], -1) - muz
            rec = (R @ Bz.t()) @ Bz
            frac = float(((R - rec) ** 2).sum() / (R**2).sum())
            out(f"    k={k:2d}  {nm:<24} residual energy fraction = {frac:.4f} (explained {1 - frac:.4f})")
        # principal angles between the two bases
        s = torch.linalg.svdvals(Ba @ Bb.t()).clamp(-1, 1)
        out(
            f"    k={k:2d}  principal angles worker-vs-clean basis: "
            f"cos = {', '.join(f'{float(x):.2f}' for x in s[:5])} ...  mean={float(s.mean()):.3f}"
        )
    out()

    # ============================================================ F. IN-DISTRIBUTION
    out("=" * 90)
    out("F. IN-DISTRIBUTION CONFOUND -- was the shape space built from the CLEAN corpus?")
    out("=" * 90)
    out("  (preprocess_meshes.py: 'OmniAnt's own shape space was built with the baseline")
    out("   workflow on that exact corpus, so those scans are in-distribution by construction')")
    for k in ("ALL_ANTS_CLEAN", "CLEAN81_M7", "LIM_0", "M7_worker"):
        c = C[k]
        S = c["S"]
        vt = c["vt"]
        mu = S.mean(0)
        d_tpl = float((mu - vt).norm(dim=-1).mean())
        bbox = float((vt.max(0).values - vt.min(0).values).norm())
        # how much of the corpus's own rest-shape variation lies inside the shapedirs span?
        K = c["betas"].shape[1]
        Bsd = c["sd"][:, :, :K].permute(2, 0, 1).reshape(K, -1)
        Q, _ = torch.linalg.qr(Bsd.t())
        R = (S - mu).reshape(S.shape[0], -1)
        proj = R @ Q
        inside = float((proj**2).sum() / (R**2).sum())
        out(
            f"    {k:<16} |corpus mean rest-shape - v_template| = {d_tpl:.6f} "
            f"({100 * d_tpl / bbox:.3f}% bbox)   fraction of corpus rest-shape variance "
            f"INSIDE shapedirs span = {100 * inside:.2f}%"
        )
    out()
    out("  Subspace overlap: top-K PCA of ALL_ANTS_CLEAN rest shapes vs SMPL_fit shapedirs")
    c = C["ALL_ANTS_CLEAN"]
    S = c["S"]
    K = 20
    Rc = (S - S.mean(0)).reshape(S.shape[0], -1)
    _, _, Vt = torch.linalg.svd(Rc, full_matrices=False)
    Bpca = Vt[:K]
    Bsd = c["sd"][:, :, :K].permute(2, 0, 1).reshape(K, -1)
    Qs, _ = torch.linalg.qr(Bsd.t())
    s = torch.linalg.svdvals(Bpca @ Qs).clamp(-1, 1)
    out(f"    principal-angle cosines (K=20): {', '.join(f'{float(x):.3f}' for x in s)}")
    out(f"    mean cos = {float(s.mean()):.4f}   #cos>0.9 = {int((s > 0.9).sum())}/{K}")
    out()

    # ============================================================== G. SPECIMEN SIZE
    out("=" * 90)
    out("G. SPECIMEN-SIZE CONFOUND -- is the clean structure just a global scale mode?")
    out("=" * 90)
    for k, c in C.items():
        S, vt = c["S"], c["vt"]
        cen = vt - vt.mean(0)
        dirn = (cen / cen.norm()).reshape(-1)  # uniform-scale direction
        F = c["F"]
        Rf = (F - F.mean(0)).reshape(F.shape[0], -1)
        a = Rf @ dirn
        share_F = float((a**2).sum() / (Rf**2).sum())
        Rs = (S - S.mean(0)).reshape(S.shape[0], -1)
        b = Rs @ dirn
        share_S = float((b**2).sum() / (Rs**2).sum())
        # remove the scale mode from the free-form field and redo the metric
        Fp = (Rf - torch.outer(a, dirn)).reshape(F.shape) + F.mean(0)
        e = loo_gen_blocked(Fp, [10])
        out(
            f"    {k:<16} uniform-scale share of free-form var = {100 * share_F:6.2f}% "
            f"(of rest-shape var {100 * share_S:6.2f}%)   free-form gen/spread@10 after "
            f"removing it = {np.nanmean(e[:, 10]) / pop_spread(Fp):.4f} "
            f"(was {base[k]['r10']:.4f})"
        )
    out()

    # ============================================================== H. METRIC CHOICE
    out("=" * 90)
    out("H. METRIC CHOICE -- energy (variance-explained) estimator instead of mean distance")
    out("=" * 90)
    for k, c in C.items():
        en = energy_loo(c["F"], [10, 20, 40])
        out(
            f"    {k:<16} free-form OUT-OF-SAMPLE residual energy fraction  "
            + "  ".join(f"@{a}={b:.4f}" for a, b in en.items())
        )
    out()

    with open(os.path.join(HERE, "deform_structure_confound_out.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote deform_structure_confound_out.txt")


if __name__ == "__main__":
    main()
