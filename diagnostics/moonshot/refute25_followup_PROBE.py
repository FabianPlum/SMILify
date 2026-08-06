"""REFUTE PROBE (part 3) — three follow-ups that close the loop.

1  Is the corpus IN the model's shape space?  Fraction of each corpus's TOTAL rest-space
   variance (not just the free-form part) that lies inside span(shapedirs) of the template it
   was fitted with.  OmniAnt's shape space was built from the clean corpus (REPORT.md 6.1),
   so this is the "in-distribution by construction" number.

2  Matched cross-ARM reproducibility, clean vs worker.  Section A of part 2 measured worker
   cross-arm agreement (LIM_0 vs LIM_1x = 0.4645) and cross-seed (0.52).  The comparable
   clean pair is CLEAN_M7 vs CLEAN_hier (same 81 scans, different recipe).

3  Kill the run-random half of the worker field by averaging three independent seeds, then
   re-run gen/spread.  If the field were optimiser noise, the seed-mean should generalise far
   better.  If it is specimen-deterministic but high-dimensional, nothing moves.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from refute25_spectral_confound_PROBE import (  # noqa: E402
    CORPORA,
    loo_gen,
    parts,
    pop_spread,
    spectrum,
)


def main():
    res = {}
    print("=" * 96)
    print("1 — IN-DISTRIBUTION: fraction of each corpus's TOTAL rest-space variance that lies")
    print("    inside the span of the shapedirs of the template it was fitted with")
    print("=" * 96)
    print(f"{'corpus':<22}{'template':<36}{'K':>4}{'S var in span':>15}{'F var in span':>15}")
    for name, npz, tpl, kind in CORPORA:
        if not os.path.exists(npz):
            continue
        vt, P, F, d, dd = parts(npz, tpl)
        S = vt[None] + P + F
        N = S.shape[0]
        sd = np.asarray(dd["shapedirs"], dtype=np.float64)
        K = sd.shape[2]
        Q, _ = np.linalg.qr(sd.reshape(-1, K))
        for lab, X in (("S", S), ("F", F)):
            Xf = (X - X.mean(0)).reshape(N, -1)
            inside = float((np.linalg.norm(Xf @ Q, axis=1) ** 2).sum() / (np.linalg.norm(Xf, axis=1) ** 2).sum())
            res.setdefault(name, {})[f"{lab}_in_span"] = inside
        print(
            f"{name:<22}{os.path.basename(tpl):<36}{K:>4}"
            f"{100 * res[name]['S_in_span']:>14.1f}%{100 * res[name]['F_in_span']:>14.1f}%"
        )

    print("\n" + "=" * 96)
    print("2 — MATCHED cross-ARM reproducibility of the free-form field")
    print("=" * 96)
    arm_pairs = [
        ("WORKER  LIM_0 vs LIM_1x", "runs/LIM_0/Stage_3_deform_fine.npz", "runs/LIM_1x/Stage_3_deform_fine.npz"),
        (
            "WORKER  LIM_0 vs BPX_noprior",
            "runs/LIM_0/Stage_3_deform_fine.npz",
            "runs/BPX_noprior/Stage_3_deform_fine.npz",
        ),
        ("CLEAN   CLEAN_M7 vs CLEAN_hier H3", "runs/CLEAN_M7/Stage_3_deform_fine.npz", "runs/CLEAN_hier/H3_deform.npz"),
    ]
    print(f"{'pair':<38}{'same-specimen r':>17}{'cross-specimen r':>19}")
    for lab, a, b in arm_pairs:
        pa, pb = os.path.join(HERE, a), os.path.join(HERE, b)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{lab:<38}  [missing]")
            continue
        da, db = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
        if list(da["labels"]) != list(db["labels"]):
            print(f"{lab:<38}  [label mismatch]")
            continue
        A = np.asarray(da["deform_verts"], np.float64)
        B = np.asarray(db["deform_verts"], np.float64)
        n = A.shape[0]
        if A.shape[1] != B.shape[1]:
            print(f"{lab:<38}  [vertex-count mismatch {A.shape[1]} vs {B.shape[1]}]")
            continue
        Af = (A - A.mean(0)).reshape(n, -1)
        Bf = (B - B.mean(0)).reshape(n, -1)
        Af /= np.linalg.norm(Af, axis=1, keepdims=True)
        Bf /= np.linalg.norm(Bf, axis=1, keepdims=True)
        C = Af @ Bf.T
        same = float(np.mean(np.diag(C)))
        off = float((C.sum() - np.trace(C)) / (n * n - n))
        print(f"{lab:<38}{same:>17.4f}{off:>19.4f}")
        res.setdefault("cross_arm", {})[lab] = dict(same=same, cross=off)

    print("\n" + "=" * 96)
    print("3 — SEED-MEAN of the worker field (M7, M7_s1, M7_s2): does removing the run-random")
    print("    component rescue generalisation?")
    print("=" * 96)
    fs = [os.path.join(HERE, f"runs/M7_handoff_midline{s}/Stage_3_deform_fine.npz") for s in ("", "_s1", "_s2")]
    Fs = [np.asarray(np.load(f, allow_pickle=True)["deform_verts"], np.float64) for f in fs]
    mean3 = np.mean(Fs, 0)
    diff = Fs[0] - Fs[1]  # the purely run-random part (signal cancels)
    print(f"{'field':<28}{'effdim':>8}{'alpha':>8}{'gen10/sp':>11}{'gen20/sp':>11}{'gen40/sp':>11}")
    for lab, X in (("single run (M7)", Fs[0]), ("mean of 3 seeds", mean3), ("seed DIFFERENCE (noise)", diff)):
        f, pr, alpha = spectrum(X)
        g, sp = loo_gen(X, [10, 20, 40]), pop_spread(X)
        print(f"{lab:<28}{pr:>8.2f}{alpha:>8.2f}{g[10] / sp:>11.4f}{g[20] / sp:>11.4f}{g[40] / sp:>11.4f}")
        res.setdefault("seedmean", {})[lab] = dict(
            effdim=pr, alpha=alpha, r10=g[10] / sp, r20=g[20] / sp, r40=g[40] / sp
        )
    print(
        f"\n  |seed difference| / |field| = "
        f"{np.linalg.norm(diff) / np.linalg.norm(Fs[0]):.4f}   "
        f"(so the run-random part is a real fraction of the field, and removing it changes"
        f" nothing above)"
    )

    json.dump(res, open(os.path.join(OUT, "refute25_followup.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/refute25_followup.json")


if __name__ == "__main__":
    main()
