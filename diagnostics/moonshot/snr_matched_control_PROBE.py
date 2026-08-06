"""PROBE -- the two quantitative questions left open by deform_structure_confound_PROBE.py.

Q1. "the workers' free-form field is PURE per-specimen noise (0.9998, i.e. 10 PCA modes
     explain literally nothing out-of-sample)".
    What does an ACTUAL pure-noise field score on this estimator at the same n and D? If iid
    noise scores 1.0000 and the workers score 0.9870 / residual-energy 0.9517, then the worker
    field is NOT indistinguishable from noise and "literally nothing" is false. Establish the
    null by simulation, with a CI.

Q2. Is the clean/worker gap in the DECISIVE matched control (CLEAN81_M7 0.7792 vs M7 0.9870)
    an independent fact about STRUCTURE, or a restatement of the 2.3x difference in field
    MAGNITUDE (0.00888 vs 0.02051)? gen/spread is signal-over-TOTAL, so it falls mechanically
    as incoherent error is added at fixed structure. Two tests:
      (a) sweep iid noise into the clean field and report BOTH the ratio and the ABSOLUTE
          out-of-sample explained energy (which is what "how much real structure is in there"
          actually means, and is invariant to the ratio's denominator);
      (b) a FULLY matched comparison: clean subsampled to n=50, genus-blocked, and noised to
          the worker's exact mean field magnitude -- versus worker M7 at n=50, genus-blocked.

Q3. Spatial bandwidth. A field that is smooth on the mesh is low-rank-friendly regardless of
    corpus. Measure mesh-Laplacian roughness per unit magnitude for each corpus's field: if
    the clean fields are markedly smoother, "structured" may be a smoothness/regularisation
    fact rather than an anatomy fact.

Uses a Gram-matrix LOO that is verified against the published estimator before anything else
is reported.
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


def load_F(name):
    npz, tpl = CORPORA[name]
    d = np.load(npz, allow_pickle=True)
    F = torch.tensor(np.asarray(d["deform_verts"], dtype=np.float64), dtype=DTYPE)
    faces = torch.tensor(np.asarray(d["faces"][0], dtype=np.int64))
    return F, [str(x) for x in d["labels"]], faces, tpl


def genus_of(lab):
    s = lab.replace(".obj", "").strip()
    if re.fullmatch(r"\d+", s):
        return "NUMBERED_BLOCK"
    s = re.sub(r"\s*\(\d+\)$", "", s)
    return (s.split("_")[0] if "_" in s else s.split("-")[0]).lower()


def loo_stats(X, ks, blocks=None):
    """Leave-one-out (or leave-one-block-out) PCA via the Gram matrix.
    Returns dict with
      ratio[k]  mean per-vertex residual distance / population spread  (published estimator)
      resE[k]   out-of-sample RESIDUAL energy fraction
      absE[k]   ABSOLUTE out-of-sample EXPLAINED energy, per vertex (mean over specimens of
                (|r|^2 - |r_resid|^2)/V) -- how much real structure was recovered, in units
                that do not depend on the total field size.
    """
    N, V = X.shape[0], X.shape[1]
    Xf = X.reshape(N, -1)
    blocks = np.arange(N) if blocks is None else np.asarray(blocks)
    kmax = max(ks)
    dist = {k: [] for k in ks}
    num = {k: [] for k in ks}
    absx = {k: [] for k in ks}
    den, d0 = [], []
    for i in range(N):
        keep = torch.tensor(blocks != blocks[i])
        if int(keep.sum()) < kmax + 2:
            continue
        Y = Xf[keep]
        mu = Y.mean(0)
        Yc = Y - mu
        G = Yc @ Yc.t()
        ev, U = torch.linalg.eigh(G)
        idx = torch.argsort(ev, descending=True)
        ev, U = ev[idx].clamp_min(0), U[:, idx]
        s = ev.sqrt()
        a = Xf[i] - mu
        g = a @ Yc.t()  # (m,)
        c = (g @ U) / s.clamp_min(1e-30)  # coefficients on the PCs
        d0.append(float(a.reshape(-1, 3).norm(dim=-1).mean()))
        den.append(float((a**2).sum()))
        for k in ks:
            w = U[:, :k] @ (c[:k] / s[:k].clamp_min(1e-30))
            rec = w @ Yc
            r = a - rec
            dist[k].append(float(r.reshape(-1, 3).norm(dim=-1).mean()))
            num[k].append(float((r**2).sum()))
            absx[k].append(float(((a**2).sum() - (r**2).sum()) / V))
    ps = float((X - X.mean(0)).norm(dim=-1).mean())
    return dict(
        spread=ps,
        loo0=float(np.mean(d0)),
        ratio={k: float(np.mean(dist[k])) / ps for k in ks},
        resE={k: float(np.sum(num[k]) / np.sum(den)) for k in ks},
        absE={k: float(np.mean(absx[k])) for k in ks},
        n=len(den),
    )


def main():
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    ks = [10, 40]
    Fs, labs, faces, tpls = {}, {}, {}, {}
    for k in CORPORA:
        Fs[k], labs[k], faces[k], tpls[k] = load_F(k)

    out("=" * 92)
    out("0. VERIFY the fast Gram LOO against the published numbers")
    out("=" * 92)
    pub = {"LIM_0": 0.9998, "M7_worker": 0.9870, "BPX_noprior": 0.9988, "CLEAN81_M7": 0.7792, "ALL_ANTS_CLEAN": 0.7932}
    base = {}
    for k in CORPORA:
        st = loo_stats(Fs[k], ks)
        base[k] = st
        out(
            f"    {k:<16} ratio@10 = {st['ratio'][10]:.4f}  (published {pub[k]:.4f}, "
            f"delta {st['ratio'][10] - pub[k]:+.5f})   mean|F| = "
            f"{float(Fs[k].norm(dim=-1).mean()):.6f}"
        )
    out()

    out("=" * 92)
    out("Q1. WHAT DOES ACTUAL PURE NOISE SCORE? -- the null for 'explains literally nothing'")
    out("=" * 92)
    out("    iid Gaussian field, same n and same dimension as each corpus, 8 realisations")
    for k in ("LIM_0", "M7_worker", "CLEAN81_M7"):
        N, V = Fs[k].shape[0], Fs[k].shape[1]
        rng = np.random.default_rng(11)
        r10, e10, e40 = [], [], []
        for _ in range(8):
            Z = torch.tensor(rng.normal(0, 1, (N, V, 3)), dtype=DTYPE)
            st = loo_stats(Z, ks)
            r10.append(st["ratio"][10])
            e10.append(st["resE"][10])
            e40.append(st["resE"][40])
        out(
            f"    NULL n={N:3d}  ratio@10 = {np.mean(r10):.4f} +/- {np.std(r10, ddof=1):.4f}   "
            f"residual-energy@10 = {np.mean(e10):.4f} +/- {np.std(e10, ddof=1):.4f}   "
            f"@40 = {np.mean(e40):.4f}"
        )
    out()
    out("    observed, same estimator:")
    for k in CORPORA:
        st = base[k]
        out(
            f"    {k:<16} ratio@10 = {st['ratio'][10]:.4f}   residual-energy@10 = "
            f"{st['resE'][10]:.4f}  (structure recovered out of sample = "
            f"{100 * (1 - st['resE'][10]):.2f}% of the field's energy)"
        )
    out()

    out("=" * 92)
    out("Q2a. SNR SWEEP -- iid noise into the CLEAN field, ratio AND absolute explained energy")
    out("=" * 92)
    wm = float(Fs["M7_worker"].norm(dim=-1).mean())
    wst = base["M7_worker"]
    out(
        f"    worker M7 target: mean|F| = {wm:.6f}  ratio@10 = {wst['ratio'][10]:.4f}  "
        f"resE@10 = {wst['resE'][10]:.4f}  ABSOLUTE explained energy@10 = {wst['absE'][10]:.3e}"
    )
    out()
    F = Fs["CLEAN81_M7"]
    rng = np.random.default_rng(5)
    out(f"    {'sigma':>7} {'mean|F|':>10} {'ratio@10':>9} {'resE@10':>9} {'absExpl@10':>12} {'absExpl@40':>12}")
    for sig in [0.0, 0.004, 0.008, 0.010, 0.011, 0.012, 0.014, 0.020]:
        Z = F + torch.tensor(rng.normal(0, sig, F.shape), dtype=DTYPE)
        st = loo_stats(Z, ks)
        out(
            f"    {sig:7.4f} {float(Z.norm(dim=-1).mean()):10.6f} {st['ratio'][10]:9.4f} "
            f"{st['resE'][10]:9.4f} {st['absE'][10]:12.3e} {st['absE'][40]:12.3e}"
        )
    out()
    out("    NOTE the absolute explained energy is the scale-free statement of 'how much real")
    out("    structure did 10 out-of-sample modes recover'. The ratio is that divided by the")
    out("    field's total energy, so it falls as noise is added even when the structure is")
    out("    untouched.")
    out()

    out("=" * 92)
    out("Q2b. FULLY MATCHED CONTROL: clean n=50 + genus-blocked + noised to the worker's")
    out("     exact field magnitude, versus worker M7 n=50 + genus-blocked")
    out("=" * 92)
    gw = np.array([genus_of(x) for x in labs["M7_worker"]])
    stw = loo_stats(Fs["M7_worker"], ks, gw)
    out(
        f"    WORKER M7   n=50 genus-blocked:  ratio@10 = {stw['ratio'][10]:.4f}  "
        f"resE@10 = {stw['resE'][10]:.4f}  absExpl@10 = {stw['absE'][10]:.3e}  "
        f"mean|F| = {wm:.6f}"
    )
    Fc = Fs["CLEAN81_M7"]
    gc_all = np.array([genus_of(x) for x in labs["CLEAN81_M7"]])
    rng = np.random.default_rng(9)
    rows = []
    for t in range(10):
        idx = torch.tensor(rng.choice(Fc.shape[0], 50, replace=False))
        X = Fc[idx]
        g = gc_all[idx.numpy()]
        # solve for sigma that matches the worker mean field magnitude
        lo, hi = 0.0, 0.05
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            Z = X + torch.tensor(rng.normal(0, mid, X.shape), dtype=DTYPE)
            if float(Z.norm(dim=-1).mean()) < wm:
                lo = mid
            else:
                hi = mid
        sig = 0.5 * (lo + hi)
        Z = X + torch.tensor(rng.normal(0, sig, X.shape), dtype=DTYPE)
        st = loo_stats(Z, ks, g)
        rows.append((sig, float(Z.norm(dim=-1).mean()), st["ratio"][10], st["resE"][10], st["absE"][10]))
    A = np.array(rows)
    out("    CLEAN81_M7  n=50 genus-blocked, magnitude-matched (10 draws):")
    out(f"        sigma        = {A[:, 0].mean():.5f} +/- {A[:, 0].std(ddof=1):.5f}")
    out(f"        mean|F|      = {A[:, 1].mean():.6f}  (target {wm:.6f})")
    out(f"        ratio@10     = {A[:, 2].mean():.4f} +/- {A[:, 2].std(ddof=1):.4f}   [worker {stw['ratio'][10]:.4f}]")
    out(f"        resE@10      = {A[:, 3].mean():.4f} +/- {A[:, 3].std(ddof=1):.4f}   [worker {stw['resE'][10]:.4f}]")
    out(f"        absExpl@10   = {A[:, 4].mean():.3e} +/- {A[:, 4].std(ddof=1):.1e}   [worker {stw['absE'][10]:.3e}]")
    out()
    out("    and the same clean draws WITHOUT the magnitude match, for reference:")
    rng = np.random.default_rng(9)
    rows = []
    for t in range(10):
        idx = torch.tensor(rng.choice(Fc.shape[0], 50, replace=False))
        st = loo_stats(Fc[idx], ks, gc_all[idx.numpy()])
        rows.append((st["ratio"][10], st["resE"][10], st["absE"][10]))
    A0 = np.array(rows)
    out(
        f"        ratio@10 = {A0[:, 0].mean():.4f} +/- {A0[:, 0].std(ddof=1):.4f}   "
        f"resE@10 = {A0[:, 1].mean():.4f}   absExpl@10 = {A0[:, 2].mean():.3e}"
    )
    out()

    out("=" * 92)
    out("Q3. SPATIAL BANDWIDTH of the free-form fields (mesh-Laplacian roughness per unit")
    out("    magnitude). A smoother field is low-rank-friendly independently of anatomy.")
    out("=" * 92)
    for k in CORPORA:
        F, f = Fs[k], faces[k]
        V = F.shape[1]
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
        e = torch.cat([e, e.flip(1)], 0)
        deg = torch.zeros(V, dtype=DTYPE).index_add_(0, e[:, 0], torch.ones(e.shape[0], dtype=DTYPE))
        Fc = F - F.mean(0)  # across-specimen variation only
        lap = torch.zeros_like(Fc)
        for i in range(Fc.shape[0]):
            acc = torch.zeros(V, 3, dtype=DTYPE)
            acc.index_add_(0, e[:, 0], Fc[i][e[:, 1]])
            lap[i] = Fc[i] - acc / deg.clamp_min(1).unsqueeze(-1)
        rough = float((lap**2).sum() / (Fc**2).sum())
        out(
            f"    {k:<16} Laplacian roughness = {rough:.4f}   "
            f"(1.0 = white on the mesh; 0 = perfectly smooth)   "
            f"mean|F| = {float(F.norm(dim=-1).mean()):.6f}"
        )
    out()

    with open(os.path.join(HERE, "snr_matched_control_out.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote snr_matched_control_out.txt")


if __name__ == "__main__":
    main()
