"""GEOMRISK — adversarial probes on the volumetric-ellipsoid-proxy proposal.

Written by the geometry-risks reviewer. Every number here is measured, not argued.
Five independent questions, none of which needs an optimiser to exist:

  R1  FRAME OF THE SCALE.  The proposal says the ellipsoid's three semi-axes are
      `log_beta_scales`. Read from smal_model/batch_lbs.py:184, the per-joint scale enters as
      `rot_new = s_par_inv @ rot @ diag(exp(logscale))`, i.e. a DIAGONAL matrix in the joint's
      REST-LOCAL frame, and SMAL rest-local frames are world-axis-aligned. So the per-specimen
      knob on an ellipsoid is a scale along GLOBAL X/Y/Z, not along the bone. Measure: the
      angle of every bone to the nearest global axis, and the representation ceiling with
      ellipsoid axes locked to global axes vs locked to the bone frame vs free (SVD).
      probe_ellipsoid_ceiling.py measured the FREE case (0.689). Free orientation is not in
      the proposed parameterisation.

  R2  TWIST NULL SPACE.  An ellipsoid with two equal semi-axes is a solid of revolution:
      rotation about the long axis changes NOTHING, so the volumetric objective has exactly
      zero gradient in that direction. Measure the cross-section anisotropy (b/c ratio) of
      (i) the fitted ellipsoid and (ii) the TRUE segment solid, per segment. Where the true
      solid is itself near-axisymmetric no primitive can fix it; where the true solid is
      anisotropic but the ellipsoid is not, the PRIMITIVE is throwing the signal away.

  R3  K>1 CEILING, DONE ON THE INTERIOR.  probe_primitive_count.py clusters SURFACE VERTICES
      and reports the ceiling FALLING with K (0.690 -> 0.193 at K=16), concluding "best_k=1".
      A cluster of points on a curved shell is locally planar, so SVD gives a near-zero third
      semi-axis and each sub-ellipsoid is a wafer with ~no volume; recall collapses 0.86 ->
      0.22 exactly as that artefact predicts. Redo it clustering INTERIOR points.

  R4  INSIDE/OUTSIDE ON REAL SCANS.  trimesh's contains_points (trimesh/ray/ray_util.py:42)
      casts ONE fixed direction [0.4395, 0.6176, 0.6522] for every query point. On a mesh with
      1020-5521 components and open boundaries the resulting error is not noise to be averaged
      away, it is a shadow cast by every hole along that one direction. Measure the
      disagreement across many directions, and how much of the volume it moves.

  R5  LBS vs RIGID.  Ellipsoids rigidly bound to one bone are not the model's deformation.
      Measure how many template vertices are genuinely blended, and how far a rigid-argmax
      approximation moves them under a plausible pose.
"""

import json
import os
import pickle
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
OUT = {}


def load_model():
    import config

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def ell_from_pts(pts, V, q=98.0):
    """Ellipsoid with GIVEN axis frame V (rows = axes); semi-axes = q-th pct of |projection|."""
    c = pts.mean(0)
    ax = np.maximum(np.percentile(np.abs((pts - c) @ V.T), q, axis=0), 1e-9)
    return c, V, ax


def svd_frame(pts):
    _, _, V = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    return V


def bone_frame(d):
    """Orthonormal frame whose first axis is the bone direction d."""
    d = d / max(np.linalg.norm(d), 1e-12)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e2 = np.cross(d, a)
    e2 /= np.linalg.norm(e2)
    return np.stack([d, e2, np.cross(d, e2)])


def inside_any(x, ells):
    memb = np.zeros((len(ells), len(x)), dtype=bool)
    for i, e in enumerate(ells):
        if e is None:
            continue
        c, V, ax = e
        memb[i] = (((x - c) @ V.T / ax) ** 2).sum(1) <= 1.0
    return memb.any(0), memb


def main():
    from fitter_3d.trainer_hierarchical import anatomical_groups

    dd = load_model()
    jn = [str(x) for x in dd["J_names"]]
    v = np.asarray(dd["v_template"], np.float64)
    f = np.asarray(dd["f"]).astype(np.int64)
    w = np.asarray(dd["weights"])
    J = np.asarray(dd["J"], np.float64)
    par = np.asarray(dd["kintree_table"])[0]
    dom = w.argmax(1)
    n_j = len(jn)

    c0 = v.mean(0)
    sc = np.abs(v - c0).max()
    v = (v - c0) / sc
    Jn = (J - c0) / sc
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    print(f"template {len(v)}v watertight={mesh.is_watertight} vol={mesh.volume:.5f}")

    rng = np.random.default_rng(0)
    lo, hi = v.min(0) - 0.02, v.max(0) + 0.02
    N_MC = 300000
    x = lo + rng.random((N_MC, 3)) * (hi - lo)
    ins = mesh.contains(x)
    print(f"MC {N_MC}: {ins.sum()} interior ({100 * ins.mean():.2f}%)\n")

    from scipy.spatial import cKDTree

    xi = x[ins]
    seg_of = dom[cKDTree(v).query(xi, k=1)[1]]

    # ---------------------------------------------------------------- R1 frame of the scale
    print("=" * 92)
    print("R1  WHAT FRAME DOES THE PER-SPECIMEN SCALE ACT IN?")
    print("=" * 92)
    # bone direction for joint j = direction from j to its children (or from parent to j if leaf)
    bdir = {}
    for j in range(n_j):
        ch = [k for k in range(n_j) if par[k] == j and k != j]
        if ch:
            d = np.mean([Jn[k] - Jn[j] for k in ch], axis=0)
        elif par[j] >= 0:
            d = Jn[j] - Jn[par[j]]
        else:
            d = np.array([1.0, 0, 0])
        if np.linalg.norm(d) < 1e-9:
            d = np.array([1.0, 0, 0])
        bdir[j] = d / np.linalg.norm(d)

    angs = []
    for j in range(n_j):
        if (dom == j).sum() < 4:
            continue
        a = np.degrees(np.arccos(np.clip(np.abs(bdir[j]).max(), 0, 1)))
        angs.append((jn[j], a))
    aa = np.array([a for _, a in angs])
    print(f"  angle of bone direction to its NEAREST global axis, over {len(aa)} populated segments:")
    print(f"    median {np.median(aa):.1f} deg   p90 {np.percentile(aa, 90):.1f}   max {aa.max():.1f}")
    print(f"    segments within 15 deg of an axis: {int((aa < 15).sum())}/{len(aa)}")
    print("  worst-aligned segments:")
    for nm, a in sorted(angs, key=lambda t: -t[1])[:8]:
        print(f"    {nm:<12} {a:6.1f} deg")

    # ceiling under three axis conventions
    print("\n  representation ceiling, ONE ellipsoid per segment, three axis conventions:")
    res_frames = {}
    for name in ["free_svd", "global_axes", "bone_frame"]:
        ells = []
        for j in range(n_j):
            p = v[dom == j]
            if len(p) < 4:
                ells.append(None)
                continue
            V = svd_frame(p) if name == "free_svd" else (np.eye(3) if name == "global_axes" else bone_frame(bdir[j]))
            ells.append(ell_from_pts(p, V))
        un, _ = inside_any(x, ells)
        inter = (un & ins).sum()
        iou = inter / max((un | ins).sum(), 1)
        res_frames[name] = dict(iou=float(iou), prec=float(inter / max(un.sum(), 1)), rec=float(inter / ins.sum()))
        print(
            f"    {name:<13} IoU {iou:.4f}   precision {res_frames[name]['prec']:.4f}   recall {res_frames[name]['rec']:.4f}"
        )
    OUT["R1_bone_axis_angle_deg"] = dict(
        median=float(np.median(aa)),
        p90=float(np.percentile(aa, 90)),
        max=float(aa.max()),
        n_within_15deg=int((aa < 15).sum()),
        n=len(aa),
    )
    OUT["R1_ceiling_by_frame"] = res_frames

    # ---------------------------------------------------------------- R2 twist null space
    print("\n" + "=" * 92)
    print("R2  TWIST: IS THERE ANY VOLUMETRIC SIGNAL ABOUT ROTATION ABOUT THE BONE AXIS?")
    print("=" * 92)
    jg = anatomical_groups(jn, split_distal=True, split_anterior=True)
    rows = []
    for j in range(n_j):
        p = v[dom == j]
        if len(p) < 30:
            continue
        # ellipsoid cross-section anisotropy in the BONE frame
        Vb = bone_frame(bdir[j])
        _, _, axb = ell_from_pts(p, Vb)
        ell_aniso = max(axb[1], axb[2]) / max(min(axb[1], axb[2]), 1e-12)
        # TRUE solid cross-section anisotropy: interior points of this segment, in the bone frame
        sel = xi[seg_of == j]
        true_aniso = np.nan
        if len(sel) >= 50:
            q = (sel - sel.mean(0)) @ Vb.T
            cov = np.cov(q[:, 1:].T)
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            true_aniso = float(np.sqrt(ev[0] / max(ev[1], 1e-18)))
        rows.append((jn[j], jg[j], ell_aniso, true_aniso, len(sel)))

    leg = [r for r in rows if r[1].startswith("l")]
    body = [r for r in rows if r[1] in ("body", "head")]
    for nm, grp in [("LEG segments", leg), ("BODY/HEAD segments", body)]:
        e = np.array([r[2] for r in grp])
        t = np.array([r[3] for r in grp if np.isfinite(r[3])])
        print(
            f"  {nm:<20} n={len(grp):3d}  ellipsoid b/c median {np.median(e):.2f}   "
            f"TRUE-solid cross-section anisotropy median {np.median(t) if len(t) else float('nan'):.2f}"
        )
    n_axisym = sum(1 for r in rows if r[2] < 1.20)
    print("\n  segments whose fitted ellipsoid has b/c < 1.20 (twist is a NULL DIRECTION of the")
    print(f"  volumetric objective to within 20%): {n_axisym}/{len(rows)}")
    print("  per-segment detail (ellipsoid b/c, true-solid b/c):")
    for nm, g, e, t, n in sorted(rows, key=lambda r: r[2])[:12]:
        print(f"    {nm:<12} {g:<10} ell {e:5.2f}   true {t:5.2f}   ({n} interior pts)")
    OUT["R2_twist"] = dict(
        n_segments=len(rows),
        n_ell_aniso_below_1p2=int(n_axisym),
        leg_ell_bc_median=float(np.median([r[2] for r in leg])),
        body_ell_bc_median=float(np.median([r[2] for r in body])),
    )

    # direct measurement: rotate a segment about its bone axis, recompute IoU
    print("\n  DIRECT: rotate ONE segment's ellipsoid about its own bone axis, measure dIoU")
    base_ells = []
    for j in range(n_j):
        p = v[dom == j]
        base_ells.append(ell_from_pts(p, svd_frame(p)) if len(p) >= 4 else None)
    un0, _ = inside_any(x, base_ells)
    iou0 = (un0 & ins).sum() / max((un0 | ins).sum(), 1)
    twist_rows = []
    targets = [j for j in range(n_j) if base_ells[j] is not None and (dom == j).sum() > 50]
    for j in targets:
        d = bdir[j]
        K = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
        best = 0.0
        for th in (30.0, 60.0, 90.0):
            R = np.eye(3) + np.sin(np.radians(th)) * K + (1 - np.cos(np.radians(th))) * K @ K
            c, V, ax = base_ells[j]
            e2 = list(base_ells)
            e2[j] = (c, V @ R.T, ax)
            un, _ = inside_any(x, e2)
            iou = (un & ins).sum() / max((un | ins).sum(), 1)
            best = max(best, abs(iou0 - iou))
        twist_rows.append((jn[j], jg[j], best))
    tw = np.array([t[2] for t in twist_rows])
    print(f"    baseline union IoU {iou0:.4f}")
    print(f"    max |dIoU| over twists of 30/60/90 deg, per segment: median {np.median(tw):.2e}  max {tw.max():.2e}")
    print("    smallest (least observable twists):")
    for nm, g, b in sorted(twist_rows, key=lambda r: r[2])[:10]:
        print(f"      {nm:<12} {g:<10} |dIoU| {b:.2e}")
    OUT["R2_twist_dIoU"] = dict(
        baseline_iou=float(iou0),
        median=float(np.median(tw)),
        max=float(tw.max()),
        per_segment={t[0]: float(t[2]) for t in twist_rows},
    )

    # ---------------------------------------------------------------- R3 K>1 on the interior
    print("\n" + "=" * 92)
    print("R3  K>1 CEILING, CLUSTERING INTERIOR POINTS INSTEAD OF SURFACE VERTICES")
    print("=" * 92)
    from sklearn.cluster import KMeans

    # dense interior sample per segment so k-means has volume to work with
    r3 = {}
    for K in [1, 2, 4, 8]:
        ells = []
        for j in range(n_j):
            sel = xi[seg_of == j]
            surf = v[dom == j]
            if len(sel) < 4 * K:  # too few interior pts: fall back to the surface fit
                if len(surf) >= 4:
                    ells.append(ell_from_pts(surf, svd_frame(surf)))
                continue
            lab = KMeans(n_clusters=K, n_init=3, random_state=0).fit_predict(sel) if K > 1 else np.zeros(len(sel), int)
            for i in range(K):
                q = sel[lab == i]
                if len(q) >= 4:
                    ells.append(ell_from_pts(q, svd_frame(q), q=100.0))
        un, _ = inside_any(x, ells)
        inter = (un & ins).sum()
        iou = inter / max((un | ins).sum(), 1)
        r3[K] = dict(
            n_prims=len(ells), iou=float(iou), prec=float(inter / max(un.sum(), 1)), rec=float(inter / ins.sum())
        )
        print(
            f"  K={K:<3} n_prims={len(ells):4d}  IoU {iou:.4f}  precision {r3[K]['prec']:.4f}  recall {r3[K]['rec']:.4f}"
        )
    OUT["R3_interior_kmeans"] = r3

    # ---------------------------------------------------------------- R5 LBS vs rigid
    print("\n" + "=" * 92)
    print("R5  HOW MUCH OF THE TEMPLATE IS GENUINELY BLENDED BETWEEN BONES?")
    print("=" * 92)
    wmax = w.max(1)
    for thr in [0.99, 0.95, 0.9, 0.75]:
        print(
            f"  vertices with dominant weight < {thr:.2f}: {100 * (wmax < thr).mean():6.2f}%  ({int((wmax < thr).sum())} of {len(v)})"
        )
    n_eff = 1.0 / (w**2).sum(1)
    print(
        f"  effective #bones per vertex (1/sum w^2): median {np.median(n_eff):.2f}  p90 {np.percentile(n_eff, 90):.2f}  max {n_eff.max():.2f}"
    )
    # per-group blended fraction
    print("  blended (<0.9) fraction by anatomical group:")
    gb = {}
    for j in range(n_j):
        m = dom == j
        if m.sum() == 0:
            continue
        g = jg[j]
        a, b = gb.get(g, (0, 0))
        gb[g] = (a + int((wmax[m] < 0.9).sum()), b + int(m.sum()))
    for g in sorted(gb, key=lambda k: -gb[k][0] / max(gb[k][1], 1)):
        print(f"    {g:<12} {100 * gb[g][0] / max(gb[g][1], 1):6.2f}%   ({gb[g][1]} verts)")
    OUT["R5_blend"] = dict(
        frac_below_0p9=float((wmax < 0.9).mean()),
        frac_below_0p99=float((wmax < 0.99).mean()),
        n_eff_median=float(np.median(n_eff)),
        n_eff_p90=float(np.percentile(n_eff, 90)),
        by_group={g: float(gb[g][0] / max(gb[g][1], 1)) for g in gb},
    )

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "geomrisk.json"), "w") as fh:
        json.dump(OUT, fh, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'out', 'geomrisk.json')}")


if __name__ == "__main__":
    main()
