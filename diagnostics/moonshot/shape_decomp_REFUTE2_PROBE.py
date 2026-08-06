"""REFUTATION PROBE 2 — is the worker free-form field "pure per-specimen NOISE"?

The headline claims the worker deform_verts field is "pure per-specimen noise (0.9998, i.e.
10 PCA modes explain literally nothing out-of-sample)" whereas the clean corpus's is
"structured". gen/spread ~ 1 is consistent with noise, but it is EQUALLY consistent with a
reproducible, specimen-specific field that 49 training samples cannot span. Those two have
opposite implications, and gen/spread alone cannot tell them apart. Three tests here:

  T1 SEED REPRODUCIBILITY. runs/M7_handoff_midline{,_s1,_s2} are three independent optimiser
     seeds over the SAME 50 worker meshes (same handoff recipe, different upstream
     hierarchical init). If the field were optimiser noise, F_i would differ across seeds by
     as much as it differs across specimens. Measured: per-specimen cross-seed distance
     relative to the across-specimen spread, and the cross-seed correlation of F.

  T2 NOISE FLOOR vs UNEXPLAINED RESIDUAL. gen@10 in ABSOLUTE units against the cross-seed
     distance. If gen@10 >> the seed-to-seed disagreement, the unexplained part is
     reproducible signal, not noise.

  T3 SPATIAL FREQUENCY. Umbrella-smooth F over the template mesh (k passes) and split it into
     a low-frequency part and a high-frequency residual. If the worker field's failure to
     generalise is surface-roughness chasing (worker scans are CT isosurfaces: 28% rougher
     dihedral, 44% worse edge-CV per preprocess_meshes.py), the LOW-frequency part should
     generalise far better. If it is scrambled correspondence, smoothing will not help.

  T4 TEMPLATE CENTREDNESS. ||mean_i F_i|| and |F| per corpus. SMIL_OmniAnt's shape space was
     built from the ALL_ANTS_CLEAN corpus (preprocess_meshes.py docstring: "OmniAnt's own
     shape space was built with the baseline workflow on that exact corpus, so those scans
     are in-distribution by construction and a good fit there proves nothing about
     generalisation"). If so the template sits ON the clean corpus and OFF the worker one,
     which is not held constant by re-using the same template file.

Output -> shape_decomp_REFUTE2_PROBE_out.txt
"""

import json
import os
import pickle

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
KS = [1, 5, 10, 20, 40]

SEED_FAMILIES = {
    "M7_handoff_midline": ["M7_handoff_midline", "M7_handoff_midline_s1", "M7_handoff_midline_s2"],
    "C0_control": ["C0_control", "C0_control_s1", "C0_control_s2"],
    "A4_nofreeze": ["A4_nofreeze", "A4_nofreeze_s1", "A4_nofreeze_s2"],
    "M1_sym(H3)": ["M1_sym", "M1_sym_s1", "M1_sym_s2"],
}

FREQ_CASES = [
    (
        "M7_handoff_midline (worker)",
        f"{HERE}/runs/M7_handoff_midline/Stage_3_deform_fine.npz",
        f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl",
    ),
    (
        "CLEAN81_M7 (clean, matched)",
        f"{HERE}/runs/CLEAN_M7/Stage_3_deform_fine.npz",
        f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl",
    ),
    (
        "LIM_0 (worker)",
        f"{HERE}/runs/LIM_0/Stage_3_deform_fine.npz",
        f"{REPO}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
    ),
    (
        "ALL_ANTS_CLEAN (legacy)",
        "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/"
        "Stage_3_deform_fine.npz",
        f"{REPO}/3D_model_prep/SMPL_fit.pkl",
    ),
]


def load_pkl(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def pick(run):
    for f in ("Stage_3_deform_fine.npz", "H3_deform.npz", "Stage_2_deform_coarse.npz"):
        p = os.path.join(HERE, "runs", run, f)
        if os.path.exists(p):
            return p
    return None


def pop_spread(X):
    return float(np.linalg.norm(X - X.mean(0), axis=-1).mean())


def loo(X, ks=KS):
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
            ek[i, k] = np.linalg.norm((r - (r @ B.T) @ B).reshape(-1, 3), axis=-1).mean()
    return float(e0.mean()), {k: (float(ek[:, k].mean()) if k <= kmax else np.nan) for k in ks}


def smoother(faces, V):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(V, V)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    return sp.diags(1.0 / deg) @ A


def smooth(F, M, iters, lam=0.7):
    Y = F.copy()
    for _ in range(iters):
        for i in range(Y.shape[0]):
            Y[i] = (1 - lam) * Y[i] + lam * (M @ Y[i])
    return Y


def main():
    os.makedirs(OUT, exist_ok=True)
    fh = open(os.path.join(HERE, "shape_decomp_REFUTE2_PROBE_out.txt"), "w")

    def out(s=""):
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()

    res = {}

    out("=" * 100)
    out("T1/T2  SEED REPRODUCIBILITY of deform_verts on the SAME 50 worker meshes")
    out("       'per-specimen noise' predicts cross-seed distance ~ across-specimen spread.")
    out("=" * 100)
    for fam, runs in SEED_FAMILIES.items():
        ps = [pick(r) for r in runs]
        if any(p is None for p in ps):
            out(f"  [skip] {fam}: missing {[r for r, p in zip(runs, ps) if p is None]}")
            continue
        Fs, labs = [], None
        for p in ps:
            d = np.load(p, allow_pickle=True)
            Fs.append(np.asarray(d["deform_verts"], dtype=np.float64))
            lb = [str(x) for x in d["labels"]]
            if labs is None:
                labs = lb
            assert labs == lb, "label order differs between seeds"
        sp0 = pop_spread(Fs[0])
        out(f"\n  {fam}   n={Fs[0].shape[0]}  across-specimen spread of F = {sp0:.6f}")
        pairs = [(0, 1), (0, 2), (1, 2)]
        dd = []
        for a, b in pairs:
            per = np.linalg.norm(Fs[a] - Fs[b], axis=-1).mean(1)  # per specimen
            dd.append(per.mean())
            # correlation of the centred fields
            A = (Fs[a] - Fs[a].mean(0)).reshape(Fs[a].shape[0], -1)
            B = (Fs[b] - Fs[b].mean(0)).reshape(Fs[b].shape[0], -1)
            r = float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))
            # per-specimen correlation of the raw fields
            rr = [
                float((Fs[a][i] * Fs[b][i]).sum() / np.sqrt((Fs[a][i] ** 2).sum() * (Fs[b][i] ** 2).sum()))
                for i in range(Fs[a].shape[0])
            ]
            out(
                f"    seed {runs[a]} vs {runs[b]}: cross-seed dist {per.mean():.6f} "
                f"= {100 * per.mean() / sp0:5.1f}% of spread;  corr(centred F) {r:+.4f};  "
                f"median per-specimen corr(F) {np.median(rr):+.4f}"
            )
        noise = float(np.mean(dd))
        ls, g = loo(Fs[0], [10, 40])
        out(f"    -> mean cross-seed distance (NOISE FLOOR) {noise:.6f}  ({100 * noise / sp0:.1f}% of spread)")
        out(f"    -> LOO gen@10 {g[10]:.6f} ({g[10] / sp0:.4f} of spread), gen@40 {g[40]:.6f} ({g[40] / sp0:.4f})")
        out(f"    -> unexplained residual @10 is {g[10] / max(noise, 1e-12):.2f}x the seed-to-seed disagreement")
        res[fam] = dict(spread=sp0, noise=noise, gen10=g[10], gen40=g[40], gen10_over_noise=g[10] / max(noise, 1e-12))

    out()
    out("=" * 100)
    out("T3  SPATIAL FREQUENCY — does the unlearnable part live at high frequency?")
    out("    F = F_low (umbrella-smoothed, 40 passes lam=0.7) + F_high")
    out("=" * 100)
    for name, npz, tpl in FREQ_CASES:
        if not os.path.exists(npz):
            out(f"  [skip] {name}")
            continue
        d = np.load(npz, allow_pickle=True)
        dd = load_pkl(tpl)
        F = np.asarray(d["deform_verts"], dtype=np.float64)
        faces = np.asarray(dd["f"], dtype=np.int64)
        M = smoother(faces, F.shape[1])
        Flo = smooth(F, M, 40)
        Fhi = F - Flo
        out(f"\n  {name}  n={F.shape[0]} V={F.shape[1]}")
        out(
            f"    |F| {np.linalg.norm(F, axis=-1).mean():.6f}   "
            f"|F_low| {np.linalg.norm(Flo, axis=-1).mean():.6f}   "
            f"|F_high| {np.linalg.norm(Fhi, axis=-1).mean():.6f}   "
            f"(high frac of energy {100 * (Fhi**2).sum() / (F**2).sum():.1f}%)"
        )
        entry = {}
        for tag, X in (("F (raw)", F), ("F_low", Flo), ("F_high", Fhi)):
            ls, g = loo(X)
            p = pop_spread(X)
            row = "  ".join(f"@{k}={g[k] / p:.4f}" if np.isfinite(g[k]) else f"@{k}=--" for k in KS)
            out(f"    {tag:<10} spread {p:.6f}   gen/spread {row}")
            entry[tag] = dict(spread=p, ratio={str(k): (g[k] / p) for k in KS})
        res.setdefault("freq", {})[name] = entry

    out()
    out("=" * 100)
    out("T4  TEMPLATE CENTREDNESS — where does v_template sit relative to each corpus?")
    out("    (betas are frozen in M7/CLEAN81_M7, so mean_i F_i IS the template-to-corpus offset)")
    out("=" * 100)
    for name, npz, tpl in FREQ_CASES:
        if not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        F = np.asarray(d["deform_verts"], dtype=np.float64)
        mF = F.mean(0)
        out(
            f"  {name:<30} |mean_i F_i| = {np.linalg.norm(mF, axis=-1).mean():.6f}   "
            f"|F| = {np.linalg.norm(F, axis=-1).mean():.6f}   "
            f"spread(F) = {pop_spread(F):.6f}"
        )
        res.setdefault("centre", {})[name] = dict(
            mean_offset=float(np.linalg.norm(mF, axis=-1).mean()),
            mag=float(np.linalg.norm(F, axis=-1).mean()),
            spread=pop_spread(F),
        )

    json.dump(res, open(os.path.join(OUT, "shape_decomp_REFUTE2.json"), "w"), indent=1)
    out(f"\nwrote {os.path.join(OUT, 'shape_decomp_REFUTE2.json')}")
    fh.close()


if __name__ == "__main__":
    main()
