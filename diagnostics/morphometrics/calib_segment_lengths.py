"""M0 — THE GATE. Do SEGMENT LENGTHS survive the fit, when dense correspondence does not?

WHY THIS IS A DIFFERENT QUESTION FROM E6/E8
The whole moonshot investigation measured DENSE correspondence: does template vertex i land on
target vertex i? The answer is ~5-7%, and E7 showed 83% of that error never even crosses a part
boundary. That is fatal for anything needing per-vertex identity.

Morphometrics does not need per-vertex identity. It needs ~50 SEGMENT LENGTHS -- the distances
between consecutive skeleton joints. A segment length is a J_regressor-weighted average over
hundreds of vertices, so the within-part scrambling that destroys dense correspondence largely
AVERAGES OUT. This probe measures whether that intuition survives contact with the data.

WHAT ACTUALLY MATTERS, and it is not absolute accuracy
For clustering, PCA and species comparison, a CONSTANT bias is harmless -- it shifts every
specimen identically and vanishes from the covariance. What matters is whether BETWEEN-SPECIMEN
variation is recovered. So the headline is not "mean error", it is:

  R across specimens   per segment, corr(fitted length, true length) over the corpus.
                       R = 1 means the fit ranks specimens perfectly on that segment even if
                       every value is off by a constant factor. R ~ 0 means the segment carries
                       no recoverable morphometric signal and must be dropped.
  SNR                  between-specimen SD of the TRUE length, divided by the SD of the
                       (fitted - true) residual. > 1 means signal exceeds measurement noise.

Both are reported per segment, because §4/§6.9.4 of the moonshot report establish that trunk
and distal leg behave completely differently and averaging them hides everything.

FRAME. Identical to score_synth_roundtrip.py: each target is centred and divided by max|coord|
by the loader, so the fit lives in that frame; ground truth is put through the transform derived
from the EXPORTED mesh, since that is what the fitter saw. Lengths are therefore in units of
specimen extent and directly comparable between fit and truth.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, MOON)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402

OUT = os.path.join(HERE, "out")


def load_model():
    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    Jr = np.asarray(dd["J_regressor"])
    if hasattr(Jr, "toarray"):
        Jr = Jr.toarray()
    return dd, Jr, [str(x) for x in dd["J_names"]], np.asarray(dd["kintree_table"])[0]


def segment_table(jnames, parents):
    """Every joint with a real parent defines one bone. Returns [(child_idx, parent_idx, name)]."""
    segs = []
    for j, p in enumerate(parents):
        if p < 0 or p >= len(jnames):
            continue
        segs.append((j, int(p), jnames[j]))
    return segs


def lengths(verts, Jr, segs):
    """verts (n,V,3) -> (n, n_seg) bone lengths, in the units of `verts`."""
    J = np.einsum("jv,nvc->njc", Jr, verts)
    return np.stack([np.linalg.norm(J[:, c] - J[:, p], axis=-1) for c, p, _ in segs], axis=1)


def group_of(name):
    if name.startswith("b_a_"):
        return "gaster"
    if name == "b_h":
        return "head"
    if name.startswith("ma"):
        return "mandible"
    if name.startswith("an_"):
        return "antenna"
    if name.startswith("w_"):
        return "wing stub"
    if name.startswith("l_"):
        seg = name.split("_")[2]
        return "leg prox" if seg in ("co", "tr", "fe") else "leg distal"
    return "thorax"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["SYN_clean", "SYN_D1", "SYN_D2"])
    ap.add_argument("--corpus", default="synth_clean")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    dd, Jr, jnames, parents = load_model()
    segs = segment_table(jnames, parents)
    print(f"{len(segs)} bone segments from {len(jnames)} joints")

    gt = np.load(os.path.join(MOON, args.corpus, "ground_truth.npz"))
    gtv_all = gt["verts"]
    names = [str(x) for x in gt["names"]]

    res = {}
    for run in args.runs:
        p = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            print(f"[skip] {run}")
            continue
        d = np.load(p)
        labels = [str(x) for x in d["labels"]]
        fit_v, gt_v = [], []
        for i, nm in enumerate(labels):
            stem = nm[:-4] if nm.endswith(".obj") else nm
            j = names.index(stem)
            ov, _, _ = load_obj(os.path.join(MOON, args.corpus, f"{stem}.obj"), load_textures=False)
            ov = ov.numpy()
            c = ov.mean(0)
            s = np.abs(ov - c).max()
            gt_v.append((gtv_all[j] - c) / s)
            fit_v.append(d["verts"][i])
        Lf = lengths(np.stack(fit_v), Jr, segs)
        Lg = lengths(np.stack(gt_v), Jr, segs)
        res[run] = (Lf, Lg)

        rel = (Lf - Lg) / np.maximum(Lg, 1e-9)
        R = np.array(
            [
                np.corrcoef(Lf[:, k], Lg[:, k])[0, 1] if Lg[:, k].std() > 1e-12 and Lf[:, k].std() > 1e-12 else np.nan
                for k in range(len(segs))
            ]
        )
        snr = Lg.std(0) / np.maximum((Lf - Lg).std(0), 1e-12)
        print(f"\n=== {run}   n={Lf.shape[0]} specimens, {len(segs)} segments")
        print(f"  median |relative length error| : {100 * np.median(np.abs(rel)):.2f}%")
        print(f"  median R across specimens      : {np.nanmedian(R):.3f}")
        print(f"  segments with R > 0.9          : {int(np.nansum(R > 0.9))} / {len(segs)}")
        print(f"  segments with SNR > 1          : {int((snr > 1).sum())} / {len(segs)}")
        groups = sorted(set(group_of(n) for _, _, n in segs))
        print(f"  {'group':<12}{'n':>4}{'med|relerr|':>13}{'med R':>9}{'med SNR':>10}")
        for g in groups:
            m = np.array([group_of(n) == g for _, _, n in segs])
            print(
                f"  {g:<12}{int(m.sum()):>4}{100 * np.median(np.abs(rel[:, m])):>12.2f}%"
                f"{np.nanmedian(R[m]):>9.3f}{np.median(snr[m]):>10.2f}"
            )
        res[run] = dict(R=R.tolist(), snr=snr.tolist(), rel_med=float(np.median(np.abs(rel))))

    # ---------------- figure
    runs = [r for r in args.runs if r in res]
    if runs:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        gl = [group_of(n) for _, _, n in segs]
        groups = sorted(set(gl))
        for run in runs:
            R = np.array(res[run]["R"])
            ax[0].plot(np.sort(R[~np.isnan(R)])[::-1], marker="o", ms=2.5, label=run)
            vals = [np.nanmedian(R[[g == q for g in gl]]) for q in groups]
            ax[1].plot(range(len(groups)), vals, marker="s", ms=5, label=run)
        ax[0].axhline(0.9, ls="--", c="k", lw=0.8)
        ax[0].set_xlabel("segment (sorted)")
        ax[0].set_ylabel("R across specimens")
        ax[0].set_title("Is between-specimen length variation recovered?", fontsize=10)
        ax[1].set_xticks(range(len(groups)))
        ax[1].set_xticklabels(groups, rotation=25, ha="right")
        ax[1].set_ylabel("median R")
        ax[1].axhline(0.9, ls="--", c="k", lw=0.8)
        ax[1].set_title("...and where", fontsize=10)
        best = runs[-1] if "SYN_D1" not in runs else "SYN_D1"
        Lf, Lg = None, None
        d = np.load(os.path.join(MOON, "runs", best, "Stage_3_deform_fine.npz"))
        snr = np.array(res[best]["snr"])
        ax[2].hist(np.clip(snr, 0, 6), bins=30, color="#2b6cb0", alpha=0.8)
        ax[2].axvline(1.0, ls="--", c="r")
        ax[2].set_xlabel("SNR = between-specimen SD / residual SD")
        ax[2].set_ylabel("segments")
        ax[2].set_title(f"Signal vs measurement noise ({best})", fontsize=10)
        for a in ax[:2]:
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        ax[2].grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(OUT, "calib_segment_lengths.png")
        fig.savefig(p, dpi=120)
        print(f"\nwrote {p}")

    json.dump(
        {"segments": [n for _, _, n in segs], "runs": res},
        open(os.path.join(OUT, "calib_segment_lengths.json"), "w"),
        indent=1,
    )
    print(f"wrote {os.path.join(OUT, 'calib_segment_lengths.json')}")


if __name__ == "__main__":
    main()
