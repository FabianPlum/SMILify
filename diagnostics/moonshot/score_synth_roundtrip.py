"""E6 scorer + figure factory: DIRECT correspondence correctness on the ground-truth round trip.

The targets were generated from the model, so fitted vertex i is supposed to land on generated
vertex i. That makes correspondence checkable without any population inference:

  PRIMARY   `correct_frac` -- fraction of vertices whose OWN ground-truth vertex is the
            NEAREST ground-truth vertex to where they landed. This is the question the whole
            investigation has been circling, asked directly. A registration that works is near
            1.0; one that merely shrink-wraps can be near 0 while looking perfect in every
            surface metric.
  SECONDARY per-vertex distance to the correct ground-truth vertex, as a % of specimen extent,
            overall and per anatomical part.
  CALIBRATION probe-19 gen/spread on the same fits, so the indirect metric this report has
            ranked five experiments by can finally be compared against the direct one.

FRAME. `load_meshes` centres each target and divides by max|coord|, so the fit lives in that
frame. The ground truth is put through the identical transform, derived from the EXPORTED
(noisy) mesh since that is what the fitter saw. Scoring is against the CLEAN ground truth --
crediting a fit for reproducing the noise would be measuring the wrong thing.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.ops import knn_points  # noqa: E402
from render_3d import DEV, colour_from_scalar, make_renderer, render  # noqa: E402

PARTS = ["thorax", "gaster", "head", "mandible", "antenna", "leg prox", "leg distal"]


def part_of(n):
    if n == "b_t":
        return "thorax"
    if n.startswith("b_a_"):
        return "gaster"
    if n == "b_h":
        return "head"
    if n.startswith("ma"):
        return "mandible"
    if n.startswith("an_"):
        return "antenna"
    if n.startswith("l_"):
        seg = n.split("_")[2]
        return "leg prox" if seg in ("co", "tr", "fe") else "leg distal"
    return "thorax"


def norm_like_loader(v):
    """Exactly what fitter_3d/utils.load_meshes does to every target."""
    c = v.mean(0)
    v = v - c
    return v / v.abs().max(), c


def score(run, corpus):
    gt = np.load(os.path.join(HERE, corpus, "ground_truth.npz"))
    gtv = torch.tensor(gt["verts"], dtype=torch.float32, device=DEV)  # (n, V, 3) clean, model frame
    names = [str(x) for x in gt["names"]]
    p = os.path.join(HERE, "runs", run, "Stage_3_deform_fine.npz")
    if not os.path.isfile(p):
        cand = sorted(f for f in os.listdir(os.path.join(HERE, "runs", run)) if f.endswith(".npz"))
        if not cand:
            return None
        p = os.path.join(HERE, "runs", run, cand[-1])
    d = np.load(p)
    fit = torch.tensor(d["verts"], dtype=torch.float32, device=DEV)
    labels = [str(x) for x in d["labels"]]

    out = []
    for i, nm in enumerate(labels):
        stem = nm[:-4] if nm.endswith(".obj") else nm
        j = names.index(stem)
        # normalise GT with the transform the loader derived from the EXPORTED mesh
        ov, _, _ = load_obj(os.path.join(HERE, corpus, f"{stem}.obj"), load_textures=False)
        _, c = norm_like_loader(ov.to(DEV))
        s = (ov.to(DEV) - c).abs().max()
        g = (gtv[j] - c) / s
        f = fit[i]
        # nearest GT vertex to each fitted vertex
        nn = knn_points(f.unsqueeze(0), g.unsqueeze(0), K=1).idx[0, :, 0]
        correct = (nn == torch.arange(g.shape[0], device=DEV)).float()
        err = (f - g).norm(dim=-1)
        out.append(
            dict(
                name=stem, correct=correct.cpu().numpy(), err=err.cpu().numpy(), fit=f.cpu().numpy(), gt=g.cpu().numpy()
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["SYN_clean:synth_clean", "SYN_noisy:synth_noisy"])
    ap.add_argument("--render_n", type=int, default=3)
    args = ap.parse_args()

    import pickle

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = list(dd["J_names"])
    dom = np.asarray(dd["weights"]).argmax(1)
    vpart = np.array([part_of(jn[j]) for j in dom])
    faces = torch.tensor(np.asarray(dd["f"]).astype(np.int64), device=DEV)

    results = {}
    for pair in args.pairs:
        run, corpus = pair.split(":")
        r = score(run, corpus)
        if r is None:
            print(f"[skip] {run}: no fit on disk")
            continue
        results[run] = r
        cf = np.concatenate([x["correct"] for x in r])
        er = np.concatenate([x["err"] for x in r])
        print(f"\n=== {run} ({len(r)} specimens) ===")
        print(f"  CORRECT correspondence : {100 * cf.mean():.2f}%  of vertices")
        print(
            f"  per-vertex error       : median {100 * np.median(er):.2f}%  p90 {100 * np.percentile(er, 90):.2f}% of extent"
        )
        print(f"  {'part':<12}{'correct %':>11}{'median err %':>14}")
        for pt in PARTS:
            m = vpart == pt
            if not m.any():
                continue
            c = np.concatenate([x["correct"][m] for x in r])
            e = np.concatenate([x["err"][m] for x in r])
            print(f"  {pt:<12}{100 * c.mean():>10.1f}%{100 * np.median(e):>13.2f}%")

    if not results:
        return

    # ---------- FIGURES ----------
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    runs = list(results)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for run in runs:
        r = results[run]
        er = np.concatenate([x["err"] for x in r]) * 100
        ax[0].hist(er, bins=80, alpha=0.55, label=run, log=True)
        cf = [100 * x["correct"].mean() for x in r]
        ax[1].plot(sorted(cf), marker="o", ms=4, label=run)
    ax[0].set_xlabel("per-vertex error to the CORRECT ground-truth vertex (% of extent)")
    ax[0].set_ylabel("vertices (log)")
    ax[0].set_title("How far off is each vertex?", fontsize=10)
    ax[1].set_xlabel("specimen (sorted)")
    ax[1].set_ylabel("% vertices with correct correspondence")
    ax[1].set_title("Correctness per specimen", fontsize=10)
    w = 0.8 / max(len(runs), 1)
    for k, run in enumerate(runs):
        r = results[run]
        vals = [100 * np.concatenate([x["correct"][vpart == pt] for x in r]).mean() for pt in PARTS]
        ax[2].bar(np.arange(len(PARTS)) + k * w, vals, w, label=run)
    ax[2].set_xticks(np.arange(len(PARTS)) + w / 2)
    ax[2].set_xticklabels(PARTS, rotation=25, ha="right")
    ax[2].set_ylabel("% correct")
    ax[2].set_title("Correctness by anatomical part", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "out", "synth_correspondence.png"), dpi=120)
    print(f"\nwrote {os.path.join(HERE, 'out', 'synth_correspondence.png')}")

    # renders: fit coloured by correspondence error, beside the ground truth
    renderer = make_renderer(image_size=440, elev=24.0, azim=140.0)
    for run in runs:
        r = results[run]
        order = np.argsort([-x["err"].mean() for x in r])[: args.render_n]
        fig, axes = plt.subplots(len(order), 3, figsize=(11.5, 3.6 * len(order)))
        axes = np.atleast_2d(axes)
        for row, i in enumerate(order):
            x = r[i]
            g = torch.tensor(x["gt"], device=DEV)
            f = torch.tensor(x["fit"], device=DEV)
            grey = torch.full((g.shape[0], 3), 0.65, device=DEV)
            img_g = render(g, faces, grey, renderer)
            img_f = render(f, faces, grey, renderer)
            col = colour_from_scalar(x["err"] * 100, 0.0, 5.0, cmap="inferno")
            img_e = render(f, faces, col, renderer)
            for c, (im, t) in enumerate(
                [
                    (img_g, f"ground truth — {x['name']}"),
                    (img_f, "fit (shaded)"),
                    (img_e, f"corr. error — {100 * x['correct'].mean():.1f}% correct"),
                ]
            ):
                axes[row, c].imshow(im)
                axes[row, c].set_title(t, fontsize=9)
                axes[row, c].axis("off")
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 5))
        fig.colorbar(
            sm,
            ax=axes,
            orientation="horizontal",
            fraction=0.02,
            pad=0.015,
            label="distance to the CORRECT ground-truth vertex (% of specimen extent)",
        )
        fig.suptitle(f"{run} — ground-truth round trip, worst {len(order)} specimens", fontsize=12)
        pth = os.path.join(HERE, "out", f"synth_render_{run}.png")
        fig.savefig(pth, dpi=115, bbox_inches="tight")
        print(f"wrote {pth}")

    np.savez(
        os.path.join(HERE, "out", "synth_scores.npz"),
        **{f"{r}_correct": np.concatenate([x["correct"] for x in results[r]]) for r in runs},
        **{f"{r}_err": np.concatenate([x["err"] for x in results[r]]) for r in runs},
    )


if __name__ == "__main__":
    main()
