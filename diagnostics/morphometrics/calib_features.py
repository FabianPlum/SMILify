"""M0b — per-FEATURE reliability against ground truth, for every measurement the pipeline emits.

WHY THIS SUPERSEDES calib_segment_lengths.py
That first calibration measured bone lengths only. But the features that turned out to carry the
taxonomic signal are the mesh-derived ones -- head_len, head_wid, gaster_*, leaflen_ma -- and
none of them had ever been checked against ground truth. Reporting a result driven by
uncalibrated features would be exactly the mistake this project keeps having to undo.

Ground truth and fit go through the SAME code path (`measure_verts`), so a discrepancy is a
property of the fit, not of two different implementations of "head length".

WHAT IS REPORTED, and why R is the headline rather than error
For clustering and comparative work a constant bias is harmless -- it shifts every specimen
identically and cancels out of every covariance and every ratio. What cannot be recovered from
is failing to rank specimens correctly. So:

  R      corr(fitted, true) ACROSS specimens, per feature. The fraction of between-specimen
         variation the pipeline recovers. R^2 is the usable share.
  SNR    between-specimen SD of the TRUE value / SD of the (fitted - true) residual. >1 means
         real variation exceeds measurement noise.
  bias   median relative error, reported but deliberately NOT used for selection.

WHAT IT IS FOR: reliability weights. A standardised PCA gives every column equal influence, so
45% of the feature set being legs means legs get 45% of the say regardless of whether they carry
anything. Selecting features by RELIABILITY is legitimate because reliability is measured here,
on synthetic ground truth, independently of the taxonomic outcome. Selecting them by taxonomic
signal would be circular and is not done anywhere in this analysis.

CAVEAT, stated up front: n=12 synthetic specimens. Per-feature R at n=12 has a wide confidence
interval (roughly +-0.3 near R=0.6), so these values are used to form COARSE reliability tiers,
never to rank one feature above another.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402

OUT = os.path.join(HERE, "out")


def block_of(c):
    # `shapedev_X` belongs to X's block; without this it falls through to "other" and is
    # silently dropped by any block-based selection.
    if c.startswith("shapedev_"):
        c = c[len("shapedev_") :]
    if c.startswith("leaflen_ma") or c.startswith("mandible") or c.startswith("ma") or c == "attach_ma":
        return "mandible"
    if c.startswith("head"):
        return "head"
    if c.startswith("mesosoma"):
        return "mesosoma"
    if c.startswith("gaster") or "b_a" in c:
        return "gaster"
    if "an_" in c:
        return "antenna"
    if any(s in c for s in ("_ti", "_ta", "_pt")):
        return "leg_distal"
    if any(s in c for s in ("_co", "_tr", "_fe")):
        return "leg_prox"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["SYN_D1", "SYN_clean", "SYN_D2"])
    ap.add_argument("--corpus", default="synth_clean")
    ap.add_argument("--primary", default="SYN_D1", help="the run whose R becomes the reliability weight")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)

    gt = np.load(os.path.join(MOON, args.corpus, "ground_truth.npz"))
    gtv_all, names = gt["verts"], [str(x) for x in gt["names"]]

    out = {}
    for run in args.runs:
        p = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            print(f"[skip] {run}")
            continue
        d = np.load(p)
        labels = [str(x) for x in d["labels"]]
        gv = []
        for lab in labels:
            stem = lab[:-4] if lab.endswith(".obj") else lab
            ov, _, _ = load_obj(os.path.join(MOON, args.corpus, f"{stem}.obj"), load_textures=False)
            ov = ov.numpy()
            c = ov.mean(0)
            gv.append((gtv_all[names.index(stem)] - c) / np.abs(ov - c).max())
        rf, cols, dev = ms.measure_verts(d["verts"].astype(np.float64), labels, "synth", M, bones, tpa)
        rg, _, _ = ms.measure_verts(np.stack(gv), labels, "synth", M, bones, tpa)
        sc, _ = ms.symmetrise(rf, cols)
        ms.symmetrise(rg, cols)
        sd, _ = ms.symmetrise(rf, dev)
        ms.symmetrise(rg, dev)
        allc = sc + sd
        F = np.array([[r[c] for c in allc] for r in rf])
        G = np.array([[r[c] for c in allc] for r in rg])

        R, snr, bias = {}, {}, {}
        for k, c in enumerate(allc):
            f, g = F[:, k], G[:, k]
            R[c] = float(np.corrcoef(f, g)[0, 1]) if f.std() > 1e-12 and g.std() > 1e-12 else np.nan
            snr[c] = float(g.std() / max((f - g).std(), 1e-12))
            bias[c] = float(np.median(np.abs((f - g) / np.maximum(np.abs(g), 1e-12))))
        out[run] = dict(R=R, snr=snr, bias=bias, cols=allc)

        print(f"\n=== {run}   n={F.shape[0]} specimens, {len(allc)} features")
        blocks = sorted(set(block_of(c) for c in allc))
        print(f"  {'block':<12}{'n':>3}{'median R':>10}{'median SNR':>12}{'median |bias|':>15}")
        for b in blocks:
            m = [c for c in allc if block_of(c) == b]
            print(
                f"  {b:<12}{len(m):>3}{np.nanmedian([R[c] for c in m]):>10.3f}"
                f"{np.nanmedian([snr[c] for c in m]):>12.2f}{100 * np.nanmedian([bias[c] for c in m]):>14.1f}%"
            )
        print(
            f"  {'ALL':<12}{len(allc):>3}{np.nanmedian(list(R.values())):>10.3f}"
            f"{np.nanmedian(list(snr.values())):>12.2f}{100 * np.nanmedian(list(bias.values())):>14.1f}%"
        )

    prim = out.get(args.primary) or (list(out.values())[0] if out else None)
    if prim:
        R = prim["R"]
        allc = prim["cols"]
        print(f"\n=== RELIABILITY TIERS from {args.primary} (coarse; n=12 cannot support finer)")
        tiers = {"good (R>=0.75)": [], "fair (0.5<=R<0.75)": [], "poor (R<0.5)": []}
        for c in allc:
            r = R.get(c, np.nan)
            t = (
                "poor (R<0.5)"
                if not np.isfinite(r) or r < 0.5
                else ("good (R>=0.75)" if r >= 0.75 else "fair (0.5<=R<0.75)")
            )
            tiers[t].append(c)
        for t, cs in tiers.items():
            print(f"  {t:<20}{len(cs):>3}: {', '.join(sorted(cs))}")
        json.dump(
            dict(R=R, snr=prim["snr"], bias=prim["bias"], cols=allc, tiers={t: sorted(cs) for t, cs in tiers.items()}),
            open(os.path.join(OUT, "feature_reliability.json"), "w"),
            indent=1,
        )
        print(f"\nwrote {OUT}/feature_reliability.json")

        # ---- figure
        fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
        order = sorted(allc, key=lambda c: (block_of(c), -(R.get(c) if np.isfinite(R.get(c, np.nan)) else -1)))
        vals = [R.get(c, np.nan) for c in order]
        cols_ = {
            "head": "#e53e3e",
            "mesosoma": "#2b6cb0",
            "gaster": "#805ad5",
            "mandible": "#dd6b20",
            "antenna": "#38a169",
            "leg_prox": "#319795",
            "leg_distal": "#a0aec0",
            "other": "#000",
        }
        ax[0].barh(range(len(order)), vals, color=[cols_[block_of(c)] for c in order])
        ax[0].set_yticks(range(len(order)))
        ax[0].set_yticklabels(order, fontsize=5)
        ax[0].axvline(0.75, ls="--", c="k", lw=0.8)
        ax[0].axvline(0.5, ls=":", c="k", lw=0.8)
        ax[0].set_xlabel("R with ground truth, across specimens")
        ax[0].set_title("Per-feature reliability (n=12 synthetic)", fontsize=10)
        blocks = sorted(set(block_of(c) for c in allc))
        for b in blocks:
            m = [c for c in allc if block_of(c) == b]
            ax[1].scatter([len(m)] * len(m), [R.get(c, np.nan) for c in m], s=18, color=cols_[b], label=b)
        ax[1].set_xlabel("features in block  (= its share of influence in a standardised PCA)")
        ax[1].set_ylabel("R with ground truth")
        ax[1].axhline(0.5, ls=":", c="k", lw=0.8)
        ax[1].set_title("The dilution problem: big blocks are not the reliable ones", fontsize=10)
        ax[1].legend(fontsize=7)
        for a in ax:
            a.grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(OUT, "fig_feature_reliability.png")
        fig.savefig(p, dpi=125)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
