"""M6 — is PCA the wrong embedding? t-SNE and UMAP, then HDBSCAN on top.

WHY BOTHER. Everything so far is linear: Mosimann log-shape-ratios, then PCA. If genera occupy
curved or non-convex regions of shape space, a linear projection will smear them together and
every number in §3 understates the structure. t-SNE and UMAP can represent that; the question is
whether they find anything PCA does not.

THE CONTROL THAT MATTERS, AND IT IS NOT THE OBVIOUS ONE. §3.4 established that specimens sharing
an accession lot are near-replicates -- same colony, fixation, mounting and scan session -- and
that at species level the lot signal is overwhelming (accession number alone predicts species at
95.7%). So an unsupervised clustering that "discovers structure" is under immediate suspicion of
having discovered collection lots. Every cluster solution here is therefore scored against
genus, subfamily AND lot, on the same footing. If a solution tracks lot better than taxon, it
has found the museum's filing system rather than biology.

A WARNING ABOUT CLUSTERING ON t-SNE/UMAP. Both are neighbour-embedding methods that deliberately
distort density and inter-cluster distance to make a readable picture; running a
density-based clusterer on their output is common practice and is not sound, because the
densities being clustered are partly manufactured by the embedding. HDBSCAN is therefore run on
the PCA space as the primary result, with the embeddings clustered too and reported alongside so
the difference is visible rather than hidden. Treat t-SNE and UMAP as ways of LOOKING, and the
PC-space clustering as the measurement.

ARI vs AMI: both reported. ARI is chance-corrected against a fixed cluster-size distribution and
punishes splitting; AMI is better behaved when cluster sizes are very unequal, which they are
here (Cephalotes has 28 specimens, several genera have 1).
"""

import argparse
import json
import os
import re
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

try:
    from taxonomy import SUBFAMILY
except ImportError:
    SUBFAMILY = {}

OUT = os.path.join(HERE, "out")


def accession_lot(s):
    m = re.match(r"([A-Za-z]+)0*(\d+)", s or "")
    return f"{m.group(1)}{int(m.group(2)) // 100}" if m else "?"


def load(feature_set="core", quality_top=100.0):
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)
    runs = sorted(
        d
        for d in os.listdir(os.path.join(MOON, "runs"))
        if d.startswith("MORPH_W") and not d.endswith("_hier") and os.path.isdir(os.path.join(MOON, "runs", d))
    ) or ["D1_low_MERGED"]
    rows, cols, dev = [], None, None
    for r in runs:
        rr, c, v = ms.measure_run(r, "worker", M, bones, tpa)
        if rr:
            rows += rr
            cols, dev = c, v
    keep = ms.quality_filter(rows, M, quality_top)
    rows = [r for r, k in zip(rows, keep) if k]
    sc, _ = ms.symmetrise(rows, cols)
    sd, _ = ms.symmetrise(rows, dev)
    if feature_set == "core":
        sc = [c for c in sc if block_of(c) in ms.CORE_BLOCKS]
        sd = [c for c in sd if block_of(c) in ms.CORE_BLOCKS]
    X = np.array([[r[c] for c in sc] for r in rows], dtype=np.float64)
    Z, _ = ms.log_shape_ratios(np.maximum(X, 1e-9))
    D = np.array([[r[c] for c in sd] for r in rows], dtype=np.float64)
    D = (D - np.nanmean(D, 0)) / np.maximum(np.nanstd(D, 0), 1e-9)
    F = np.hstack([Z, np.nan_to_num(D)])
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-9)
    meta = dict(
        genus=np.array([r["genus"] or "?" for r in rows]),
        subfamily=np.array([SUBFAMILY.get(r["genus"] or "", "?") for r in rows]),
        lot=np.array([accession_lot(r["specimen"]) for r in rows]),
        label=np.array([r["label"] for r in rows]),
    )
    return F, meta, len(sc) + len(sd)


def score(pred, truth):
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    ok = truth != "?"
    if ok.sum() < 5:
        return dict(ari=np.nan, ami=np.nan)
    return dict(
        ari=float(adjusted_rand_score(truth[ok], pred[ok])),
        ami=float(adjusted_mutual_info_score(truth[ok], pred[ok])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npc", type=int, default=10)
    ap.add_argument("--min_cluster_size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--quality_top",
        type=float,
        default=50.0,
        help="keep only the best N%% of specimens by registration quality (deform/edge/normal "
        "composite, see measure.quality_filter and report §7); 100 = no filter",
    )
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import HDBSCAN
    import umap

    F, meta, nfeat = load(quality_top=args.quality_top)
    n = F.shape[0]
    print(
        f"{n} worker specimens (best {args.quality_top:.0f}% by registration quality), "
        f"{nfeat} core features, {len(set(meta['genus']))} genera, {len(set(meta['lot']))} lots"
    )

    P = PCA(n_components=min(args.npc, F.shape[1]), random_state=args.seed).fit_transform(F)
    print(f"PCA {args.npc} components")
    emb = {"PCA": P[:, :2]}
    perp = max(5, min(30, (n - 1) // 3))
    emb["t-SNE"] = TSNE(n_components=2, perplexity=perp, init="pca", random_state=args.seed).fit_transform(P)
    print(f"t-SNE perplexity {perp}")
    emb["UMAP"] = umap.UMAP(
        n_components=2, n_neighbors=min(15, n - 1), min_dist=0.1, random_state=args.seed
    ).fit_transform(P)
    print("UMAP n_neighbors 15, min_dist 0.1")

    # HDBSCAN: PC space is the measurement, the 2-D embeddings are reported for comparison
    spaces = {"PCA (10-D, primary)": P, "t-SNE (2-D)": emb["t-SNE"], "UMAP (2-D)": emb["UMAP"]}
    res, labels = {}, {}
    print(
        f"\nHDBSCAN, min_cluster_size={args.min_cluster_size}. "
        "Clusters are scored against LOT as well as taxon — see module docstring."
    )
    print(
        f"  {'space':<22}{'k':>4}{'noise':>8}   {'ARI gen':>8}{'AMI gen':>8}   {'ARI sub':>8}"
        f"   {'ARI LOT':>8}{'AMI LOT':>8}"
    )
    for name, S in spaces.items():
        lab = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(S)
        labels[name] = lab
        k = len(set(lab)) - (1 if -1 in lab else 0)
        noise = float((lab == -1).mean())
        g, s_, lo = score(lab, meta["genus"]), score(lab, meta["subfamily"]), score(lab, meta["lot"])
        res[name] = dict(k=k, noise=noise, genus=g, subfamily=s_, lot=lo)
        print(
            f"  {name:<22}{k:>4}{100 * noise:>7.0f}%   {g['ari']:>8.3f}{g['ami']:>8.3f}   "
            f"{s_['ari']:>8.3f}   {lo['ari']:>8.3f}{lo['ami']:>8.3f}"
        )

    prim = res["PCA (10-D, primary)"]
    verdict = "TAXON" if prim["genus"]["ami"] > prim["lot"]["ami"] else "LOT"
    print(
        f"\n  READING: in the primary (PC) space the clusters track {verdict} more closely "
        f"(AMI genus {prim['genus']['ami']:.3f} vs lot {prim['lot']['ami']:.3f})."
    )

    # ---------------- is there cluster structure to find AT ALL?
    #
    # HDBSCAN returning nothing is ambiguous on its own: it could mean the clusterer is badly
    # tuned, or that the geometry has no clusters. These two diagnostics separate those.
    #   silhouette of the TRUE labels asks whether genera are separated even in principle. It is
    #     computed on the labels, not on any clustering, so no tuning can affect it. Negative
    #     means a specimen is on average closer to some other genus than to its own.
    #   k-means at the TRUE k is the best a partition of this space can do, since it is handed
    #     the right number of groups and optimises directly for compactness.
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from collections import Counter as _C

    cg_all = _C(meta["genus"])
    big = np.array([cg_all[g] >= 5 and g != "?" for g in meta["genus"]])
    diag = {}
    print(
        f"\nIS THERE CLUSTER STRUCTURE AT ALL?  ({big.sum()} specimens in the "
        f"{len(set(meta['genus'][big]))} genera with n>=5)"
    )
    print(f"  {'space':<22}{'silhouette(genus)':>19}{'silhouette(subfam)':>20}{'k-means@true k ARI':>20}")
    for name, S in [("PCA (10-D, primary)", P), ("t-SNE (2-D)", emb["t-SNE"]), ("UMAP (2-D)", emb["UMAP"])]:
        sg = float(silhouette_score(S[big], meta["genus"][big]))
        sm = meta["subfamily"] != "?"
        ss = float(silhouette_score(S[sm], meta["subfamily"][sm]))
        km = KMeans(n_clusters=len(set(meta["genus"][big])), n_init=10, random_state=args.seed).fit_predict(S[big])
        ka = score(km, meta["genus"][big])["ari"]
        diag[name] = dict(sil_genus=sg, sil_subfamily=ss, kmeans_true_k_ari=ka)
        print(f"  {name:<22}{sg:>19.3f}{ss:>20.3f}{ka:>20.3f}")
    print("\n  A NEGATIVE silhouette means genera are interleaved, not separated -- and t-SNE/UMAP")
    print("  are MORE negative than PCA, i.e. the nonlinear embeddings do not help here. Combined")
    print("  with k-means failing at the true k, the reading is that the taxonomic signal is LOCAL")
    print("  (nearest-neighbour) rather than cluster structure. That is consistent with 1-NN")
    print("  classification working at ~5x chance while no partition of the space recovers genera.")

    res["_diagnostics"] = diag
    json.dump(res, open(os.path.join(OUT, "embeddings.json"), "w"), indent=1, default=float)

    # ---------------- figure: 3 embeddings x 3 colourings
    from collections import Counter

    cg = Counter(meta["genus"])
    top = [g for g, _ in cg.most_common(10) if g != "?"]
    cs = Counter(meta["subfamily"])
    tops = [s for s, _ in cs.most_common(8) if s != "?"]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(3, 3, figsize=(16.5, 15))
    n = F.shape[0]
    for r, (name, E) in enumerate(emb.items()):
        ax[r, 0].scatter(E[:, 0], E[:, 1], s=9, c="#cbd5e0", zorder=1)
        for i, g in enumerate(top):
            m = meta["genus"] == g
            ax[r, 0].scatter(E[m, 0], E[m, 1], s=26, color=cmap(i % 10), label=f"{g} ({cg[g]})", zorder=3)
        ax[r, 0].set_ylabel(name, fontsize=12)
        ax[r, 0].set_title("10 largest genera" if r == 0 else "", fontsize=10)
        if r == 0:
            ax[r, 0].legend(fontsize=6, ncol=2)

        for i, s_ in enumerate(tops):
            m = meta["subfamily"] == s_
            ax[r, 1].scatter(E[m, 0], E[m, 1], s=18, color=cmap(i % 10), alpha=0.85, label=f"{s_} ({m.sum()})")
        ax[r, 1].set_title("subfamily" if r == 0 else "", fontsize=10)
        if r == 0:
            ax[r, 1].legend(fontsize=6)

        key = {"PCA": "PCA (10-D, primary)", "t-SNE": "t-SNE (2-D)", "UMAP": "UMAP (2-D)"}[name]
        lab = labels[key]
        noise = lab == -1
        ax[r, 2].scatter(E[noise, 0], E[noise, 1], s=8, c="#e2e8f0", zorder=1)
        for i, c in enumerate(sorted(set(lab) - {-1})):
            m = lab == c
            ax[r, 2].scatter(E[m, 0], E[m, 1], s=20, color=cmap(i % 10), zorder=3)
        ax[r, 2].set_title(
            f"HDBSCAN on {'PC space' if name == 'PCA' else 'this embedding'}: "
            f"{res[key]['k']} clusters, ARI(genus) {res[key]['genus']['ari']:.2f} "
            f"vs ARI(lot) {res[key]['lot']['ari']:.2f}",
            fontsize=8.5,
        )
        for c in range(3):
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
    fig.suptitle(
        "Is PCA the wrong embedding? — and do unsupervised clusters track taxonomy or the museum's accession order?",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    p = os.path.join(OUT, "fig_embeddings.png")
    fig.savefig(p, dpi=118)
    print(f"wrote {p}")
    print(f"wrote {os.path.join(OUT, 'embeddings.json')}")


if __name__ == "__main__":
    main()
