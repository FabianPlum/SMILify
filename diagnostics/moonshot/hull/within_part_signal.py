"""Is there ANY signal that could fix the 83% within-part error? — the follow-on probe.

E7 established that 83.3% of correspondence error never crosses a part boundary, so no
partition of any quality can reach it. The obvious question is whether anything can.

Chamfer's blindness inside a part is structural: on a smooth gaster, many correspondences give
the same surface error, so the data term is flat along the surface and carries no positional
information. The classical answer is an INTRINSIC descriptor -- a per-vertex quantity derived
from the surface's own Laplace-Beltrami operator, which varies along a smooth part even where
Euclidean proximity does not, and is invariant to pose because bending does not change the
intrinsic metric. That is the foundation of the functional-maps family (Ovsjanikov 2012 and
successors: ZoomOut, Smooth Shells, ULRSSM SIGGRAPH 2023, Hybrid Functional Maps CVPR 2024).

This probe asks the cheapest decisive version of the question, with no new dependencies and no
learning:

    Using the HEAT KERNEL SIGNATURE (Sun, Ovsjanikov & Guibas, SGP 2009) computed independently
    on the template and on the target, does nearest-neighbour matching IN DESCRIPTOR SPACE
    recover the correct vertex better than the fitted pipeline does?

Ground truth is exact: the synthetic targets are the model's own geometry, so template vertex i
must match target vertex i.

PRE-REGISTERED READING, fixed before running:
  * HKS matching well above the pipeline's 4.81%  -- intrinsic signal exists and is being
    ignored; the functional-maps direction is validated and worth real implementation effort.
  * HKS matching near the pipeline's 4.81%        -- the within-part error may be irreducible
    from the target geometry alone, and correspondence must come from the model/prior side
    rather than from a better data term. That would be a strong negative and would redirect
    the whole investigation.
  * HKS strong WITHIN part but weak globally      -- expected, and the useful case: it would
    say descriptors should be used to refine inside a part, with the existing machinery
    handling the between-part assignment.

CONTROLS, because REPORT §7.8's lesson is that the trivial rule keeps winning:
  * `spatial`  -- nearest target vertex to the TEMPLATE's own rest-pose vertex. A floor.
  * `fitted`   -- nearest target vertex to where the actual FIT put that vertex. This is the
                  pipeline's own 4.81%, recomputed here so all rows share one protocol.
  * `random`   -- a uniformly random target vertex inside the correct part. The chance level
                  that any "within-part" number must be read against.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from scipy.spatial import cKDTree  # noqa: E402


# ======================================================================================
# intrinsic geometry
# ======================================================================================
def cotangent_laplacian(v, f):
    """Cotangent-weighted Laplacian L and lumped (barycentric) mass matrix M.

    The standard discretisation (Pinkall & Polthier 1993; Meyer et al. 2003). L is symmetric
    positive semi-definite, and the generalised problem L phi = lambda M phi gives the
    Laplace-Beltrami eigenfunctions.
    """
    n = len(v)
    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    e0, e1, e2 = v[i2] - v[i1], v[i0] - v[i2], v[i1] - v[i0]
    # cot of the angle opposite each edge, via cross/dot
    cr = np.cross(e1, -e2)
    area2 = np.linalg.norm(cr, axis=1)
    area2 = np.maximum(area2, 1e-14)
    cot0 = -(e1 * e2).sum(1) / area2
    cot1 = -(e2 * e0).sum(1) / area2
    cot2 = -(e0 * e1).sum(1) / area2

    ii = np.concatenate([i1, i2, i2, i0, i0, i1])
    jj = np.concatenate([i2, i1, i0, i2, i1, i0])
    ww = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
    W = sp.coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W

    # lumped mass: a third of each incident triangle's area
    tri_area = 0.5 * area2
    m = np.zeros(n)
    np.add.at(m, i0, tri_area / 3.0)
    np.add.at(m, i1, tri_area / 3.0)
    np.add.at(m, i2, tri_area / 3.0)
    return L.tocsc(), sp.diags(np.maximum(m, 1e-12))


def hks(v, f, n_eig=120, n_times=24):
    """Heat Kernel Signature (Sun, Ovsjanikov & Guibas, SGP 2009), per vertex.

    HKS(x, t) = sum_k exp(-lambda_k t) phi_k(x)^2 -- how much heat remains at x after time t.
    Small t is curvature-like and local; large t is global and part-aware. Intrinsic, so it is
    invariant to the pose change between the template and a posed target, which is exactly the
    property chamfer lacks.

    Scale-normalised per vertex (each vertex's signature divided by its own sum) so that
    overall mesh scale and local sampling density drop out and the descriptor compares across
    two independently-tessellated meshes.
    """
    L, M = cotangent_laplacian(v, f)
    # shift-invert around a small negative sigma for the smallest eigenvalues
    vals, vecs = spla.eigsh(L, k=n_eig, M=M, sigma=-1e-6, which="LM")
    vals = np.maximum(vals, 1e-12)
    tmin, tmax = 4 * np.log(10) / vals[-1], 4 * np.log(10) / vals[1]
    ts = np.exp(np.linspace(np.log(tmin), np.log(tmax), n_times))
    sig = (np.exp(-vals[None, :] * ts[:, None]) @ (vecs**2).T).T  # (V, n_times)
    return sig / np.maximum(sig.sum(1, keepdims=True), 1e-12)


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--n_eig", type=int, default=120)
    ap.add_argument("--run", default="SYN_clean")
    ap.add_argument("--gt_groups", default="16", choices=["7", "13", "16"])
    args = ap.parse_args()

    import config
    from fitter_3d.trainer_hierarchical import anatomical_groups

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [str(x) for x in dd["J_names"]]
    sd, sa = args.gt_groups in ("13", "16"), args.gt_groups == "16"
    jg = anatomical_groups(jn, split_distal=sd, split_anterior=sa)
    gnames = sorted(set(jg.values()))
    n2i = {nm: i for i, nm in enumerate(gnames)}
    vpart = np.array([n2i[jg[int(d)]] for d in np.asarray(dd["weights"]).argmax(1)], dtype=np.int64)

    v_tpl = np.asarray(dd["v_template"], dtype=np.float64)
    faces = np.asarray(dd["f"]).astype(np.int64)
    V = len(v_tpl)

    print(f"template {V} verts, {len(faces)} faces, {len(gnames)} anatomical groups")
    print(f"computing HKS with {args.n_eig} eigenpairs...\n", flush=True)
    d_tpl = hks(v_tpl, faces, n_eig=args.n_eig)

    gt = np.load(os.path.join(MOON, "synth_clean", "ground_truth.npz"))
    gtv = gt["verts"]
    names = [str(x) for x in gt["names"]]

    fitp = os.path.join(MOON, "runs", args.run, "Stage_3_deform_fine.npz")
    fitd = np.load(fitp) if os.path.isfile(fitp) else None
    fit_labels = [str(x) for x in fitd["labels"]] if fitd is not None else []

    rng = np.random.default_rng(0)
    acc = {k: [] for k in ("hks_global", "hks_within", "spatial", "fitted", "random_within")}
    err = {k: [] for k in acc}
    ident = np.arange(V)

    for si in range(min(args.n, len(names))):
        tgt = gtv[si].astype(np.float64)
        d_tgt = hks(tgt, faces, n_eig=args.n_eig)

        # frame-match: both meshes normalised the way the loader does
        def norm(x):
            c = x.mean(0)
            return (x - c) / np.abs(x - c).max()

        tn = norm(tgt)
        extent = 1.0

        def score(match, key):
            acc[key].append(float((match == ident).mean()))
            err[key].append(np.linalg.norm(tn[match] - tn[ident], axis=1) / extent)

        # --- HKS nearest neighbour, globally
        score(cKDTree(d_tgt).query(d_tpl, k=1)[1], "hks_global")

        # --- HKS nearest neighbour, restricted to the CORRECT part (the oracle-part case:
        #     isolates the within-part question from the between-part one)
        m_within = np.empty(V, dtype=np.int64)
        m_rand = np.empty(V, dtype=np.int64)
        for g in range(len(gnames)):
            idx = np.nonzero(vpart == g)[0]
            if len(idx) == 0:
                continue
            m_within[idx] = idx[cKDTree(d_tgt[idx]).query(d_tpl[idx], k=1)[1]]
            m_rand[idx] = idx[rng.integers(0, len(idx), len(idx))]
        score(m_within, "hks_within")
        score(m_rand, "random_within")

        # --- spatial NN from the template's own rest pose (a floor)
        score(cKDTree(tn).query(norm(v_tpl), k=1)[1], "spatial")

        # --- the pipeline's own answer, same protocol
        if fitd is not None and names[si] in [x[:-4] if x.endswith(".obj") else x for x in fit_labels]:
            bi = [x[:-4] if x.endswith(".obj") else x for x in fit_labels].index(names[si])
            fv = fitd["verts"][bi].astype(np.float64)
            score(cKDTree(tn).query(fv, k=1)[1], "fitted")
        print(f"  {names[si]}  done", flush=True)

    print(f"\n{'method':<16}{'correct %':>11}{'median err':>13}{'p90 err':>10}   what it is")
    print("-" * 92)
    desc = {
        "fitted": "the pipeline's own result (E6: 4.81%)",
        "spatial": "nearest target vertex to the TEMPLATE rest pose",
        "random_within": "chance level inside the correct part",
        "hks_global": "intrinsic descriptor NN, whole mesh",
        "hks_within": "intrinsic descriptor NN, correct part given",
    }
    for k in ("fitted", "spatial", "random_within", "hks_global", "hks_within"):
        if not acc[k]:
            continue
        e = np.concatenate(err[k])
        print(
            f"{k:<16}{100 * float(np.mean(acc[k])):>10.2f}%{100 * float(np.median(e)):>12.2f}%"
            f"{100 * float(np.percentile(e, 90)):>9.2f}%   {desc[k]}"
        )

    print("\nREADING")
    if acc["hks_within"] and acc["random_within"]:
        hw, rw = float(np.mean(acc["hks_within"])), float(np.mean(acc["random_within"]))
        print(
            f"  within-part HKS  {100 * hw:.2f}%  vs chance {100 * rw:.2f}%  -> "
            f"{'signal present' if hw > 3 * max(rw, 1e-9) else 'NO signal above chance'}"
        )
        me = np.median(np.concatenate(err["hks_within"]))
        mf = np.median(np.concatenate(err["fitted"])) if acc["fitted"] else float("nan")
        print(f"  median within-part error: HKS {100 * me:.2f}% of extent vs pipeline {100 * mf:.2f}%")
        print(f"  {'HKS beats the pipeline' if me < mf else 'HKS does NOT beat the pipeline'} on placement error.")

    import json

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "within_part_signal.json"), "w") as fh:
        json.dump(
            {
                k: dict(
                    correct=float(np.mean(v)) if v else None,
                    median_err=float(np.median(np.concatenate(err[k]))) if v else None,
                )
                for k, v in acc.items()
            },
            fh,
            indent=1,
        )
    print(f"\nwrote {os.path.join(HERE, 'out', 'within_part_signal.json')}")


if __name__ == "__main__":
    main()
