"""Does a DIFFERENT notion of "where the shape separates" beat CoACD's atoms?

§3.3 of REPORT_HULL.md found that the binding constraint on the whole convexity family is not
the merge criterion but the OVER-DECOMPOSITION: CoACD's atoms already straddle 25-45% of the
surface before any merging, because CoACD optimises collision-geometry concavity and its
cutting planes have no reason to follow anatomy.

That claim deserves a test with a method that is not CoACD and not plane-cutting at all.

**Skeletonization via Local Separators** (Bærentzen & Rotenberg, ACM TOG 40(5) 2021; multi-scale
variant MSLS, 2023) finds, for every vertex, a small vertex set whose removal disconnects its
neighbourhood — a *local separator*. Those separators are packed and each becomes a skeleton
node, and the vertex→separator map is a segmentation. It is a genuinely different mechanism:
purely combinatorial on the surface graph, no convexity, no cutting planes, no volume.

It is also the method whose premise is closest to the user's phrasing: it literally searches
for where the shape separates.

WHAT IS MEASURED
The same atom-level purity and straddle used in §3.3, so the numbers are directly comparable:

    purity   fraction of points whose ground-truth part is their own atom's majority part
    straddle fraction of surface held by atoms spanning more than one anatomical part

against CoACD at its best measured settings (merge=False, preprocess_resolution=100,
threshold 0.01: 450 atoms, purity 0.914, straddle 0.249).

PRE-REGISTERED READING:
  * LS atoms materially purer at comparable count -- §3.3's constraint is CoACD-specific, the
    convexity family should be re-run on LS atoms, and the ceiling lifts.
  * LS atoms comparable or worse                  -- the constraint is general: no unsupervised
    decomposition of this shape class puts its boundaries where anatomy does, and §5's ceiling
    is the operative limit regardless.

Note this cannot change §5. A perfect decomposition is still capped at 16.7% correspondence
correctness, because 83% of the error never crosses a part boundary at all.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

from fitter_3d.hull_decomposition import assign_to_hulls, coacd_hulls, normalise, sample_surface  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def purity(labels, gt):
    labels = np.asarray(labels)
    t = np.zeros((labels.max() + 1, gt.max() + 1), dtype=np.int64)
    np.add.at(t, (labels, gt), 1)
    return float(t.max(1).sum() / len(labels))


def straddle(labels, gt, thresh=0.9):
    labels = np.asarray(labels)
    t = np.zeros((labels.max() + 1, gt.max() + 1), dtype=np.int64)
    np.add.at(t, (labels, gt), 1)
    tot = t.sum(1)
    frac = t.max(1) / np.maximum(tot, 1)
    return float(tot[frac < thresh].sum() / len(labels))


def gel_manifold(v, f):
    from pygel3d import hmesh

    m = hmesh.Manifold()
    for tri in f:
        m.add_face(v[tri])
    hmesh.stitch(m)
    return m


def ls_atoms(v, f, multi_scale=False):
    """Per-mesh-vertex atom id from the local-separator skeleton's vertex->node map."""
    from pygel3d import graph as gr

    m = gel_manifold(v, f)
    g = gr.from_mesh(m)
    fn = gr.MSLS_skeleton_and_map if multi_scale else gr.LS_skeleton_and_map
    skel, mapping = fn(g)
    # `mapping` is a Graph whose node i carries the skeleton node that mesh-graph node i maps
    # to; PyGEL exposes it as a per-node attribute list
    lab = np.asarray([mapping[i] for i in range(len(v))], dtype=np.int64) if not hasattr(mapping, "nodes") else None
    if lab is None:
        lab = np.asarray(list(mapping), dtype=np.int64)
    _, lab = np.unique(lab, return_inverse=True)
    return lab, len(skel.nodes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--n_points", type=int, default=8000)
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
    gn = sorted(set(jg.values()))
    n2i = {nm: i for i, nm in enumerate(gn)}
    vpart = np.array([n2i[jg[int(d)]] for d in np.asarray(dd["weights"]).argmax(1)], dtype=np.int64)
    gtv = np.load(os.path.join(MOON, "synth_clean", "ground_truth.npz"))["verts"]

    rows = {}
    for si in range(args.n):
        p = os.path.join(MOON, "synth_clean", f"synth_{si:03d}.obj")
        m = trimesh.load(p, process=False, force="mesh")
        v, _, _ = normalise(np.asarray(m.vertices))
        f = np.asarray(m.faces)
        nm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        pts, _ = sample_surface(nm, args.n_points, seed=0)

        gvn = gtv[si] - gtv[si].mean(0)
        gvn = gvn / np.abs(gvn).max()
        gt = vpart[cKDTree(gvn).query(pts, k=1)[1]]

        for tag, fn in (("LS", False), ("MSLS", True)):
            try:
                vlab, n_nodes = ls_atoms(v, f, multi_scale=fn)
            except Exception as e:
                print(f"  {tag} FAILED on synth_{si:03d}: {type(e).__name__} {e}", flush=True)
                continue
            lab = vlab[cKDTree(v).query(pts, k=1)[1]]
            _, lab = np.unique(lab, return_inverse=True)
            rows.setdefault(tag, []).append((int(lab.max() + 1), purity(lab, gt), straddle(lab, gt)))
            print(f"  synth_{si:03d} {tag:<5} atoms={lab.max() + 1:4d} skel_nodes={n_nodes:4d}", flush=True)

        for tag, thr in (("CoACD thr0.03", 0.03), ("CoACD thr0.01", 0.01)):
            hulls = coacd_hulls(v, f, threshold=thr, seed=0)
            atom, _ = assign_to_hulls(pts, hulls)
            _, atom = np.unique(atom, return_inverse=True)
            rows.setdefault(tag, []).append((int(atom.max() + 1), purity(atom, gt), straddle(atom, gt)))
        print(f"  synth_{si:03d} CoACD done", flush=True)

    print(f"\n{'over-decomposition':<20}{'atoms':>8}{'purity':>10}{'straddle':>11}")
    print("-" * 50)
    for tag, v_ in rows.items():
        a = np.array(v_)
        print(f"{tag:<20}{a[:, 0].mean():>8.0f}{a[:, 1].mean():>10.4f}{a[:, 2].mean():>11.4f}")

    print("\nREADING")
    if "LS" in rows and "CoACD thr0.01" in rows:
        ls = np.array(rows["LS"])
        co = np.array(rows["CoACD thr0.01"])
        d = ls[:, 1].mean() - co[:, 1].mean()
        print(
            f"  local separators vs CoACD(0.01): purity {d:+.4f} at {ls[:, 0].mean():.0f} vs {co[:, 0].mean():.0f} atoms"
        )
        print(
            "  -> "
            + (
                "LS atoms are materially purer; §3.3's constraint is CoACD-specific."
                if d > 0.03
                else "no material gain; the constraint is general to unsupervised decomposition of this shape class."
            )
        )
    import json

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "local_separators.json"), "w") as fh:
        json.dump(
            {
                k: dict(
                    atoms=float(np.mean([x[0] for x in v_])),
                    purity=float(np.mean([x[1] for x in v_])),
                    straddle=float(np.mean([x[2] for x in v_])),
                )
                for k, v_ in rows.items()
            },
            fh,
            indent=1,
        )
    print(f"\nwrote {os.path.join(HERE, 'out', 'local_separators.json')}")


if __name__ == "__main__":
    main()
