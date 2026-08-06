"""REFUTATION PROBE 4 — is the CLEAN corpus's "structured free-form field" mostly the model's
OWN shape basis, re-expressed as free-form offsets?

REFUTE1/R6 measured, with the SAME template file (SMIL_OmniAnt.pkl, K=13) and the same
pipeline:
    fraction of the CENTRED deform_verts variance lying inside span(shapedirs[:13])
        CLEAN81_M7  24.44%          worker M7_handoff_midline  2.66%
    (isotropic-noise expectation for a 13-dim subspace of 30687 dims: 0.042%)
SMIL_OmniAnt's shape space was built from the ALL_ANTS_CLEAN corpus — the repo's own
preprocess_meshes.py says so: "OmniAnt's own shape space was built with the baseline workflow
on that exact corpus, so those scans are in-distribution by construction and a good fit there
proves nothing about generalisation." Re-using the same template file therefore does NOT hold
the instrument constant between the two corpora: it is train data for one and test data for
the other.

TEST. Split F into the component inside span(shapedirs[:K]) and the component orthogonal to
it, and re-run the probe-19 LOO metric on each. If the clean corpus's advantage is carried by
the in-span component, the published "structured vs noise" contrast is measuring
in-distribution-ness of the shape basis, not correspondence quality.

Also reported: the same split using a RANDOM K-dim subspace (null control), so the in-span
number can be read against chance.

Output -> shape_decomp_REFUTE4_PROBE_out.txt
"""

import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
KS = [1, 5, 10, 20, 40]

CASES = [
    (
        "worker M7_handoff_midline",
        f"{HERE}/runs/M7_handoff_midline/Stage_3_deform_fine.npz",
        f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl",
    ),
    (
        "CLEAN81_M7 (matched control)",
        f"{HERE}/runs/CLEAN_M7/Stage_3_deform_fine.npz",
        f"{REPO}/3D_model_prep/SMIL_OmniAnt.pkl",
    ),
    (
        "worker LIM_0",
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


def pop_spread(X):
    return float(np.linalg.norm(X - X.mean(0), axis=-1).mean())


def loo(X, ks=KS):
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    kmax = min(max(ks), N - 2)
    ek = np.zeros((N, kmax + 1))
    for i in range(N):
        m = np.ones(N, bool)
        m[i] = False
        Y = Xf[m]
        mu = Y.mean(0)
        _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        r = Xf[i] - mu
        for k in range(1, kmax + 1):
            B = Vt[:k]
            ek[i, k] = np.linalg.norm((r - (r @ B.T) @ B).reshape(-1, 3), axis=-1).mean()
    return {k: (float(ek[:, k].mean()) if k <= kmax else np.nan) for k in ks}


def main():
    os.makedirs(OUT, exist_ok=True)
    fh = open(os.path.join(HERE, "shape_decomp_REFUTE4_PROBE_out.txt"), "w")

    def out(s=""):
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()

    res = {}
    for name, npz, tpl in CASES:
        if not os.path.exists(npz):
            out(f"[skip] {name}")
            continue
        d = np.load(npz, allow_pickle=True)
        dd = load_pkl(tpl)
        F = np.asarray(d["deform_verts"], dtype=np.float64)
        K = int(np.asarray(d["betas"]).shape[1])
        sd = np.asarray(dd["shapedirs"], dtype=np.float64)[:, :, :K]
        N, V = F.shape[0], F.shape[1]
        B = sd.reshape(-1, K)
        Q, _ = np.linalg.qr(B)
        Ff = F.reshape(N, -1)
        Fin = (Ff @ Q) @ Q.T
        Fout = Ff - Fin
        Fin = Fin.reshape(N, V, 3)
        Fout = Fout.reshape(N, V, 3)
        Fc = Ff - Ff.mean(0)
        frac = float(((Fc @ Q) ** 2).sum() / (Fc**2).sum())
        # null: random K-dim subspace
        rng = np.random.default_rng(0)
        R = rng.standard_normal((B.shape[0], K))
        Qr, _ = np.linalg.qr(R)
        fracr = float(((Fc @ Qr) ** 2).sum() / (Fc**2).sum())

        out(f"\n{'=' * 96}")
        out(f"{name}   n={N} V={V} K={K}  template={os.path.basename(tpl)}")
        out(
            f"  centred free-form variance inside span(shapedirs[:{K}]) = {100 * frac:.3f}%   "
            f"(random {K}-dim subspace null: {100 * fracr:.4f}%)  ->  "
            f"{frac / max(fracr, 1e-12):.0f}x chance"
        )
        for tag, X in (
            ("F (raw, as published)", F),
            ("F_in  (inside shapedirs span)", Fin),
            ("F_out (orthogonal to it)", Fout),
        ):
            sp = pop_spread(X)
            g = loo(X)
            row = "  ".join(f"@{k}={g[k] / sp:.4f}" if np.isfinite(g[k]) else f"@{k}=--" for k in KS)
            out(f"    {tag:<32} spread {sp:.6f}  gen/spread {row}")
            res.setdefault(name, {})[tag] = dict(
                spread=sp, ratio={str(k): (g[k] / sp if np.isfinite(g[k]) else None) for k in KS}
            )
        res[name]["frac_in_span"] = frac
        res[name]["frac_in_random"] = fracr

    json.dump(res, open(os.path.join(OUT, "shape_decomp_REFUTE4.json"), "w"), indent=1)
    out(f"\nwrote {os.path.join(OUT, 'shape_decomp_REFUTE4.json')}")
    fh.close()


if __name__ == "__main__":
    main()
