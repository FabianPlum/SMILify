"""Aggressive mesh cleanup for the worker scans, and the A/B that tests whether it matters.

THE QUESTION
Is worker-scan mesh quality what stops the pipeline learning anything reasonable, or is the
blocker elsewhere (pose, correspondence)? The tempting comparison -- fit ALL_ANTS_CLEAN and
see that it works -- is CONFOUNDED and must not be used to answer this: OmniAnt's own shape
space was built with the baseline workflow on that exact corpus, so those scans are
in-distribution by construction and a good fit there proves nothing about generalisation.

The unconfounded test is to clean the WORKER meshes and refit them. Same specimens, same
poses, same shape space, one variable changed.

MEASURED STARTING POINT (40 clean vs 40 worker, both normalised as load_meshes does):

  topology        clean     worker
    vertex components (median)      3        34.5
    area fraction off main       0.0000     0.0030
    boundary-edge fraction       0.0388     0.0471
    non-manifold edge fraction    0.0709     0.0271   <- clean is WORSE

  surface
    triangle quality (median)     0.8635     0.8322
    edge-length CV                0.3000     0.4311   <- worker 44% worse
    median dihedral               8.20 deg   10.54 deg <- worker 28% rougher
    faces with dihedral > 60 deg  0.0504     0.0636

So the worker meshes are modestly rougher and more irregular, carry ~10x more connected
components holding only 0.3% of the area, and are actually CLEANER than the reference corpus
on non-manifold edges. That is not an obviously fatal difference, which is exactly why it has
to be tested rather than assumed in either direction.

WHAT THE CLEANUP DOES, and why each step is here
  1. drop tiny components   -- removes detached debris and mounting-medium fragments. Keeps
                               anything above --min_area_frac of total area, so a genuinely
                               detached tarsus is retained but speckle is not.
  2. merge duplicate verts  -- CT isosurfaces frequently emit unshared vertices, which break
                               edge adjacency and therefore every topological operator.
  3. fill small holes       -- boundary-edge fraction is 4.7%; holes make the target surface
                               locally absent, and chamfer then pulls the template into the
                               gap.
  4. Taubin smoothing       -- lambda/mu smoothing, which unlike plain Laplacian does not
                               shrink the volume. Targets the 28% dihedral-roughness excess
                               without eroding thin structures, which plain smoothing does.
  5. (optional) remesh      -- isotropic edge-length equalisation, aimed at the 44% edge-CV
                               excess. Off by default because it resamples the surface and
                               so changes what the chamfer sees in a way that is not purely
                               a quality improvement.

Every step is measured before/after so the cleanup can be shown to have actually done what
it claims, rather than assumed.
"""

import argparse
import glob
import json
import os

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))


def quality(m):
    """The same metrics used to characterise the clean/worker gap, on one mesh."""
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    if len(F) < 10:
        return {}
    c = V.mean(0)
    s = max(np.abs(V - c).max(), 1e-12)
    Vn = (V - c) / s
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    nc, lab = connected_components(A.maximum(A.T), directed=False)
    fa = m.area_faces
    ca = np.bincount(lab[F[:, 0]], weights=fa, minlength=nc)
    es = np.sort(e, axis=1)
    _, cnt = np.unique(es, axis=0, return_counts=True)
    tri = Vn[F]
    e0, e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]
    a = 0.5 * np.linalg.norm(np.cross(e0, -e2), axis=1)
    L = np.stack([np.linalg.norm(x, axis=1) for x in (e0, e1, e2)], 1)
    q = 4 * np.sqrt(3) * a / np.maximum((L**2).sum(1), 1e-20)
    try:
        adj = m.face_adjacency
        n = m.face_normals
        dih = np.degrees(np.arccos(np.clip((n[adj[:, 0]] * n[adj[:, 1]]).sum(1), -1, 1)))
    except Exception:
        dih = np.array([0.0])
    return dict(
        n_verts=int(len(V)),
        vcomp=int(nc),
        offmain=float(1 - ca.max() / max(ca.sum(), 1e-12)),
        bnd=float((cnt == 1).mean()),
        nonman=float((cnt > 2).mean()),
        tri_q=float(np.median(q)),
        edge_cv=float(L.std() / max(L.mean(), 1e-12)),
        dih_med=float(np.median(dih)),
        rough=float((dih > 60).mean()),
    )


def clean(m, min_area_frac=0.001, hole_size=30, taubin_iters=8, remesh=False):
    """Return a cleaned copy. Steps are ordered so each operates on a valid input."""
    m = m.copy()
    # 2. merge duplicates first, so connectivity is meaningful for everything after
    m.merge_vertices()
    m.remove_degenerate_faces()
    m.remove_duplicate_faces()
    m.remove_unreferenced_vertices()

    # 1. drop tiny components by AREA (not vertex count -- a long thin tarsus has few verts)
    try:
        parts = m.split(only_watertight=False)
        if len(parts) > 1:
            tot = sum(float(p.area) for p in parts)
            keep = [p for p in parts if float(p.area) / max(tot, 1e-12) >= min_area_frac]
            if keep:
                m = trimesh.util.concatenate(keep)
    except Exception:
        pass

    # 3. fill small holes
    try:
        m.fill_holes()
    except Exception:
        pass

    # 4. Taubin smoothing: lambda>0 shrink then mu<0 inflate, so volume is preserved.
    #    Plain Laplacian would erode exactly the thin distal structures probe 13 showed are
    #    already under-constrained (tarsus+pretarsus = 2.4% of leg-chain area).
    if taubin_iters > 0:
        try:
            trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=int(taubin_iters))
        except Exception:
            pass

    if remesh:
        try:
            L = float(np.median(np.linalg.norm(m.vertices[m.edges[:, 0]] - m.vertices[m.edges[:, 1]], axis=1)))
            m = m.subdivide_to_size(max_edge=L * 1.5)
        except Exception:
            pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=os.path.join(HERE, "bench50"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "bench50_clean"))
    ap.add_argument("--min_area_frac", type=float, default=0.001)
    ap.add_argument("--taubin_iters", type=int, default=8)
    ap.add_argument("--remesh", action="store_true")
    ap.add_argument("--report", default=os.path.join(HERE, "out", "preprocess_report.json"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    fs = sorted(glob.glob(os.path.join(args.in_dir, "*.obj")))
    print(f"[prep] {len(fs)} meshes -> {args.out_dir}")
    rows = []
    for i, f in enumerate(fs):
        m = trimesh.load(f, process=False, force="mesh")
        before = quality(m)
        mc = clean(m, args.min_area_frac, taubin_iters=args.taubin_iters, remesh=args.remesh)
        after = quality(mc)
        mc.export(os.path.join(args.out_dir, os.path.basename(f)))
        rows.append(dict(name=os.path.basename(f), before=before, after=after))
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(fs)}", flush=True)

    print("\n[prep] did the cleanup actually do what it claims?")
    print(f"{'metric':<12}{'before':>12}{'after':>12}{'target (clean corpus)':>24}")
    ref = dict(
        vcomp=3.0, offmain=0.0, bnd=0.0388, nonman=0.0709, tri_q=0.8635, edge_cv=0.3000, dih_med=8.196, rough=0.0504
    )
    for k in ["vcomp", "offmain", "bnd", "nonman", "tri_q", "edge_cv", "dih_med", "rough"]:
        b = np.median([r["before"].get(k, np.nan) for r in rows])
        a = np.median([r["after"].get(k, np.nan) for r in rows])
        print(f"{k:<12}{b:>12.4f}{a:>12.4f}{ref[k]:>24.4f}")
    json.dump(rows, open(args.report, "w"), indent=1)
    print(f"\n[prep] wrote {args.report}")


if __name__ == "__main__":
    main()
