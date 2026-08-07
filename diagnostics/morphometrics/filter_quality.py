"""M7 — does filtering on REGISTRATION quality change what the shape space looks like?

DIFFERENT QUESTION FROM §2.4. That tested the `radial_med` gate, which scores the SCAN before
anything is fitted, and found it makes no difference. A scan can be pristine and still be
registered badly, and it is the registration every measurement is read off.

WHAT NOT TO USE, AND WHY THIS FILE ORIGINALLY GOT IT WRONG. The first version scored quality by
chamfer distance to the target scan. That is the one metric this project has repeatedly shown
cannot be trusted: surface proximity is exactly what shrink-wrapping buys, and a fit can drive
chamfer down by spending free-form deformation -- so the fits that score best can be the ones
whose measurements are least trustworthy. Reaching for it here was an inconsistency, and it is
kept only as `--metric chamfer` for comparison.

WHAT TO USE. Quality judged from the FIT ITSELF, as a composite z-score of three quantities that
fail in different ways (see `mesh_quality`):
  deform   free-form displacement the model needed after pose and shape were exhausted -- its own
           admission that it could not explain this specimen. A mean, so it misses local spikes.
  edge     non-uniform stretching of the surface against the template's edge lengths.
  normal   roughness between adjacent face normals, above the template's own. Geometry pulled
           strongly outward at a point is a local normal reversal, which this detects sharpest.

MEASURED: the three correlate +0.82 to +0.94 with each other and the composite correlates +0.93
with chamfer, so on this corpus they largely agree and the choice matters less than the act of
filtering. That agreement is itself informative -- it says D1's offset penalty is stiff enough
that the shrink-wrap decoupling never opens up.

WHAT IS SWEPT. Keep the best 10/25/50/75/90/100% and re-measure lot-blind genus lift, silhouette
of the TRUE labels, k-means at the true k, and HDBSCAN.

THE TRAP THIS AVOIDS. Filtering removes specimens, which removes genera, which raises chance and
mechanically flatters silhouette and ARI. So every cutoff is also run on RANDOM subsets of the
same size. The dashed control line is what size alone buys; the gap is what quality buys.
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
from calib_features import block_of  # noqa: E402
from measure import mesh_quality, quality_composite  # noqa: E402,F401
from analyse import accession_lot, loo_1nn  # noqa: E402

WORKER = "/media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker"
OUT = os.path.join(HERE, "out")
CACHE = os.path.join(OUT, "registration_error.json")


def registration_error(rows, runs, n_sample=8000, skip=frozenset()):
    """Symmetric chamfer between each fitted mesh and its own target scan, in extent units."""
    import torch
    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.loss import chamfer_distance

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    err = {}
    for run in runs:
        p = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            continue
        d = np.load(p)
        labels = [str(x) for x in d["labels"]]
        fit = torch.tensor(d["verts"], dtype=torch.float32, device=dev)
        for i, lab in enumerate(labels):
            if lab in skip:
                continue
            tp = os.path.join(WORKER, lab)
            if not os.path.isfile(tp):
                continue
            v, f, _ = load_obj(tp, load_textures=False)
            v = v.to(dev)
            # the loader's normalisation, applied identically so fit and target share a frame
            v = v - v.mean(0)
            v = v / v.abs().max()
            m = Meshes(verts=[v], faces=[f.verts_idx.to(dev)])
            tgt = sample_points_from_meshes(m, n_sample)
            cd, _ = chamfer_distance(tgt, fit[i].unsqueeze(0))
            err[lab] = float(cd)
        del fit
        torch.cuda.empty_cache()
        n_new = sum(1 for lab in labels if lab in err)
        if n_new:
            print(f"  {run}: {n_new} scored", flush=True)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", nargs="+", type=float, default=[10, 25, 50, 75, 90, 100])
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument(
        "--metric",
        choices=["deform", "chamfer"],
        default="deform",
        help="deform = free-form vertex displacement the fit needed AFTER pose and shape were "
        "exhausted (the parametric model's own admission of failure); chamfer = surface distance "
        "to the target scan",
    )
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)
    runs = sorted(
        d
        for d in os.listdir(os.path.join(MOON, "runs"))
        if d.startswith("MORPH_W") and not d.endswith("_hier") and os.path.isdir(os.path.join(MOON, "runs", d))
    )
    rows, cols, dev = [], None, None
    for r in runs:
        rr, c, v = ms.measure_run(r, "worker", M, bones, tpa)
        if rr:
            rows += rr
            cols, dev = c, v
    print(f"{len(rows)} fitted workers over {len(runs)} chunks")

    # Incremental cache keyed by specimen label, so a chunk that lands later costs only itself.
    err = {} if args.recompute else (json.load(open(CACHE)) if os.path.isfile(CACHE) else {})
    missing = [r["label"] for r in rows if r["label"] not in err]
    if missing:
        print(
            f"scoring registration error for {len(missing)} specimens "
            f"({len(err)} cached) — symmetric chamfer to the target scan..."
        )
        err.update(registration_error(rows, runs, skip=set(err)))
        json.dump(err, open(CACHE, "w"))
    else:
        print(f"registration error cached for all {len(err)} specimens ({CACHE})")

    keep = np.array([r["label"] in err for r in rows])
    rows = [r for r, k in zip(rows, keep) if k]
    if args.metric == "chamfer":
        e = np.array([err[r["label"]] for r in rows])
    else:
        # DEFORM MAGNITUDE, the default. The model has a pose space and a shape space; whatever
        # it still cannot express it must buy with free-form per-vertex displacement. That
        # residual is the parametric model's own admission that it failed to explain this
        # specimen, and it is the more principled notion of registration quality here, because
        # every measurement in this report is read off the parametric structure.
        #
        # Chamfer was the first metric tried and is kept as an option. The two agree far more
        # than expected -- Spearman +0.924, and the kept sets overlap 88% at the 50% cut -- so
        # the shrink-wrap failure mode (low chamfer bought with large deformation) does not
        # occur at D1's offset penalty. They are not interchangeable at strict cuts, though:
        # 20% of the retained set differs when keeping only the best quarter.
        e, QA = quality_composite(rows, M)
    print(
        f"registration error [{args.metric}]: median {np.median(e):.5f}  "
        f"p10 {np.percentile(e, 10):.5f}  p90 {np.percentile(e, 90):.5f}"
    )
    if args.metric == "deform":
        from scipy.stats import spearmanr

        nm = ("deform", "edge distortion", "normal roughness")
        print(f"  {'component':<20}{'median':>10}{'p90':>10}   pairwise Spearman")
        for j, n_ in enumerate(nm):
            rs = "  ".join(f"{n_[:4]}~{nm[k][:4]} {spearmanr(QA[:, j], QA[:, k])[0]:+.2f}" for k in range(3) if k != j)
            print(f"  {n_:<20}{np.median(QA[:, j]):>10.4f}{np.percentile(QA[:, j], 90):>10.4f}   {rs}")
        ch = json.load(open(CACHE)) if os.path.isfile(CACHE) else {}
        if ch:
            cv = np.array([ch.get(r["label"], np.nan) for r in rows])
            ok = np.isfinite(cv)
            print(
                f"  composite vs CHAMFER: Spearman {spearmanr(e[ok], cv[ok])[0]:+.3f} "
                f"— chamfer is reported for comparison only and is not part of the filter"
            )

    sc_all, _ = ms.symmetrise(rows, cols)
    sd_all, _ = ms.symmetrise(rows, dev, key="asym_shapedev")
    sc = [c for c in sc_all if block_of(c) in ms.CORE_BLOCKS]
    sd = [c for c in sd_all if block_of(c) in ms.CORE_BLOCKS]
    X = np.array([[r[c] for c in sc] for r in rows])
    gen = np.array([r["genus"] or "?" for r in rows])
    lot = np.array([accession_lot(r["specimen"]) for r in rows])
    rng = np.random.default_rng(0)

    def evaluate(idx, n_perm=200):
        Z, _ = ms.log_shape_ratios(np.maximum(X[idx], 1e-9))
        D = np.array([[r[c] for c in sd] for r in [rows[i] for i in idx]])
        D = (D - D.mean(0)) / np.maximum(D.std(0), 1e-9)
        F = np.hstack([Z, np.nan_to_num(D)])
        F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-9)
        P = PCA(n_components=min(10, F.shape[1]), random_state=0).fit_transform(F)
        y, lo = gen[idx], lot[idx]
        a = loo_1nn(P, y, groups=lo)
        null = np.array([loo_1nn(P, rng.permutation(y), groups=lo) for _ in range(n_perm)])
        km = KMeans(n_clusters=len(set(y)), n_init=10, random_state=0).fit_predict(P)
        hl = HDBSCAN(min_cluster_size=5).fit_predict(P)
        return dict(
            n=len(idx),
            n_genus=len(set(y)),
            acc=a,
            null=float(null.mean()),
            lift=a / max(null.mean(), 1e-9),
            silhouette=float(silhouette_score(P, y)),
            kmeans_ari=float(adjusted_rand_score(y, km)),
            hdbscan_k=len(set(hl)) - (1 if -1 in hl else 0),
            hdbscan_ari=float(adjusted_rand_score(y, hl)),
        )

    def subset(mask):
        """Specimens passing `mask`, restricted to genera with n>=3 within that subset."""
        sub = np.where(mask)[0]
        g = gen[sub]
        cnt = {x: int((g == x).sum()) for x in set(g)}
        return sub[np.array([cnt[x] >= 3 and x != "?" for x in g])]

    res = []
    print(
        f"\n{'keep':>6}{'n':>6}{'genera':>8}{'lot-blind':>11}{'null':>8}{'LIFT':>7}"
        f"{'silhouette':>12}{'kmeans ARI':>12}{'HDBSCAN k':>11}{'HDB ARI':>9}   vs RANDOM subset of equal size"
    )
    for cut in sorted(args.cuts):
        idx = subset(e <= np.percentile(e, cut))
        if len(idx) < 30:
            print(f"{cut:>5.0f}%{len(idx):>6}   too few after filtering")
            continue
        r_ = evaluate(idx)
        # MATCHED RANDOM CONTROL. Filtering shrinks the corpus, and a smaller corpus has fewer
        # genera, which mechanically flatters silhouette and ARI -- both improve when there are
        # simply fewer classes to confuse. So each cutoff is compared against random subsets of
        # the SAME SIZE drawn without regard to quality. The difference is the effect of quality;
        # the control's own trend is the effect of size.
        ctrl = []
        for s_ in range(3):
            pick = np.zeros(len(e), bool)
            pick[
                np.random.default_rng(s_).choice(len(e), len(np.where(e <= np.percentile(e, cut))[0]), replace=False)
            ] = True
            ci = subset(pick)
            if len(ci) >= 30:
                ctrl.append(evaluate(ci, n_perm=100))
        cm = (
            {k: float(np.mean([c_[k] for c_ in ctrl])) for k in ("lift", "silhouette", "kmeans_ari", "n_genus")}
            if ctrl
            else {}
        )
        r_["control"] = cm
        r_["cut"] = cut
        res.append(r_)
        cs = (
            (
                f"   lift {cm['lift']:.1f}  sil {cm['silhouette']:+.3f}  kARI {cm['kmeans_ari']:.3f}"
                f"  ({cm['n_genus']:.0f} gen)"
            )
            if cm
            else ""
        )
        print(
            f"{cut:>5.0f}%{r_['n']:>6}{r_['n_genus']:>8}{100 * r_['acc']:>10.1f}%{100 * r_['null']:>7.1f}%"
            f"{r_['lift']:>7.1f}{r_['silhouette']:>12.3f}{r_['kmeans_ari']:>12.3f}"
            f"{r_['hdbscan_k']:>11}{r_['hdbscan_ari']:>9.3f}{cs}"
        )

    json.dump(res, open(os.path.join(OUT, f"filter_quality_{args.metric}.json"), "w"), indent=1)

    # ---------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    c = [r["cut"] for r in res]
    ax[0].plot(c, [r["lift"] for r in res], "o-", color="#2b6cb0")
    ax[0].set_ylabel("lot-blind genus lift (× chance)")
    ax[0].axhline(1, ls=":", c="k", lw=0.8)
    ax[0].set_title("Does filtering sharpen the taxonomic signal?", fontsize=10)
    for r in res:
        ax[0].annotate(
            f"n={r['n']}\n{r['n_genus']}gen",
            (r["cut"], r["lift"]),
            fontsize=6,
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
        )
    ax[1].plot(c, [r["silhouette"] for r in res], "s-", color="#805ad5", label="quality-filtered")
    # controls exist only where the matched random subset was itself large enough
    cc = [r for r in res if r.get("control")]
    if cc:
        xs = [r["cut"] for r in cc]
        ax[0].plot(xs, [r["control"]["lift"] for r in cc], "o--", color="#a0aec0", label="random, same size")
        ax[1].plot(xs, [r["control"]["silhouette"] for r in cc], "s--", color="#a0aec0", label="random, same size")
        ax[2].plot(xs, [r["control"]["kmeans_ari"] for r in cc], "^--", color="#a0aec0", label="random, same size")
        ax[0].legend(fontsize=8)
        ax[1].legend(fontsize=8)
    ax[1].axhline(0, ls=":", c="k", lw=0.8)
    ax[1].set_ylabel("silhouette of TRUE genus labels")
    ax[1].set_title("Are genera becoming separated?", fontsize=10)
    ax[2].plot(c, [r["kmeans_ari"] for r in res], "^-", color="#38a169", label="k-means @ true k")
    ax[2].plot(c, [r["hdbscan_ari"] for r in res], "v-", color="#dd6b20", label="HDBSCAN")
    ax[2].set_ylabel("ARI vs genus")
    ax[2].set_title("Does cluster structure appear?", fontsize=10)
    ax[2].legend(fontsize=8)
    for a_ in ax:
        a_.set_xlabel("keep best N% by registration error")
        a_.grid(alpha=0.3)
    fig.suptitle("Filtering on REGISTRATION quality (post-fit surface error), not scan quality", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, f"fig_filter_quality{'' if args.metric == 'deform' else '_chamfer'}.png")
    fig.savefig(p, dpi=125)
    print(f"\nwrote {p}")
    print(f"wrote {os.path.join(OUT, f'filter_quality_{args.metric}.json')}")


if __name__ == "__main__":
    main()
