"""SDF (Shape Diameter Function) prior probe — symmetry-and-priors family, deepening pass.

Question: does a CGAL-style ray-cone SDF field, computed with trimesh+embree ONLY
(no new dependencies), give a usable target-derived part prior on
  (a) the SMIL ant template, where skinning-weight argmax gives exact part labels, and
  (b) real bench50 ethanol scans, which have holes / debris / touching limbs?

Gates borrowed verbatim from hull/eval_decomposition.py so results are comparable:
  G1 REPRODUCIBILITY  two independent surface samplings agree >= 90%
  G4 COST             < 60 s / specimen on CPU

Extra measurement specific to this family:
  does SDF separate the GASTER (the 10.4%-median-error worst part) from the LEGS?

Artifacts: diagnostics/moonshot/sdf_prior_out.txt  (kept on disk, per CLAUDE.md)
"""

import sys
import time
import pickle
import numpy as np
import trimesh

sys.path.insert(0, "/home/fabi/dev/SMILify")

RNG = np.random.default_rng(0)


def cone_dirs(normal, n_rays, half_angle):
    """n_rays directions inside a cone of half-angle `half_angle` about -normal."""
    axis = -normal / (np.linalg.norm(normal) + 1e-12)
    # cosine-ish uniform sampling within the cone
    u = RNG.random(n_rays)
    cos_t = 1.0 - u * (1.0 - np.cos(half_angle))
    sin_t = np.sqrt(np.clip(1 - cos_t**2, 0, 1))
    phi = RNG.random(n_rays) * 2 * np.pi
    # orthonormal basis around axis
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, tmp)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(axis, e1)
    return (cos_t[:, None] * axis + sin_t[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))


def sdf_field(mesh, points, normals, n_rays=25, cone_deg=120.0, batch=20000):
    """CGAL-style SDF: inward ray cone, first hit, outlier-rejected mean.

    Returns raw (unsmoothed, unnormalised) SDF per point. NaN where no ray hit.
    """
    inter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    half = np.deg2rad(cone_deg) / 2.0
    n_pts = len(points)
    dirs = np.empty((n_pts, n_rays, 3), np.float64)
    for i in range(n_pts):
        dirs[i] = cone_dirs(normals[i], n_rays, half)
    eps = 1e-5 * float(mesh.scale)
    origins = np.repeat(points + eps * (-normals), n_rays, axis=0)
    d_flat = dirs.reshape(-1, 3)

    lengths = np.full(len(origins), np.nan)
    for s in range(0, len(origins), batch):
        e = min(s + batch, len(origins))
        loc, idx_ray, _ = inter.intersects_location(origins[s:e], d_flat[s:e], multiple_hits=False)
        if len(idx_ray):
            lengths[s + idx_ray] = np.linalg.norm(loc - origins[s:e][idx_ray], axis=1)
    L = lengths.reshape(n_pts, n_rays)
    # weight each ray by cos(angle to axis)  (CGAL weights by inverse angle; cos is the
    # cheap monotone stand-in) then reject rays > 1 std from the median, then mean.
    out = np.full(n_pts, np.nan)
    for i in range(n_pts):
        v = L[i][np.isfinite(L[i])]
        if v.size < 3:
            continue
        med = np.median(v)
        sd = v.std()
        keep = v[np.abs(v - med) <= sd] if sd > 0 else v
        out[i] = keep.mean() if keep.size else med
    return out


def part_labels_from_smil():
    d = pickle.load(open("/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl", "rb"), encoding="latin1")
    V = np.asarray(d["v_template"], np.float64)
    F = np.asarray(d["f"], np.int64)
    W = np.asarray(d["weights"], np.float64)
    names = [str(n) for n in d["J_names"]]
    return V, F, W.argmax(1), names


def main():
    log = []

    def P(s):
        print(s, flush=True)
        log.append(s)

    # ---------- (a) SMIL ant template, exact part labels ----------
    V, F, jlab, names = part_labels_from_smil()
    tm = trimesh.Trimesh(V, F, process=False)
    P(f"TEMPLATE watertight={tm.is_watertight} winding_consistent={tm.is_winding_consistent} "
      f"n_v={len(V)} n_f={len(F)} components={len(tm.split(only_watertight=False))}")

    t0 = time.time()
    sdf_v = sdf_field(tm, tm.vertices, tm.vertex_normals, n_rays=25, cone_deg=120.0)
    t_tpl = time.time() - t0
    P(f"TEMPLATE sdf: {t_tpl:.1f}s  nan_frac={np.mean(~np.isfinite(sdf_v)):.4f}")

    scale = float(tm.scale)
    grp = {}
    for i, n in enumerate(names):
        ln = n.lower()
        if "gaster" in ln or "abdomen" in ln or "petiol" in ln:
            g = "gaster"
        elif any(k in ln for k in ("coxa", "femur", "tibia", "tarsus", "leg")):
            g = "leg"
        elif "antenn" in ln or "scape" in ln or "funicul" in ln:
            g = "antenna"
        elif "head" in ln or "mandib" in ln:
            g = "head"
        else:
            g = "thorax"
        grp[i] = g
    vg = np.array([grp[j] for j in jlab])
    P("TEMPLATE SDF by anatomical group (relative to bbox diagonal):")
    for g in ["gaster", "thorax", "head", "leg", "antenna"]:
        m = (vg == g) & np.isfinite(sdf_v)
        if m.sum():
            q = np.percentile(sdf_v[m] / scale, [10, 50, 90])
            P(f"   {g:8s} n={m.sum():5d}  sdf/scale p10={q[0]:.4f} p50={q[1]:.4f} p90={q[2]:.4f}")

    ga = np.isfinite(sdf_v) & (vg == "gaster")
    lg = np.isfinite(sdf_v) & (vg == "leg")
    if ga.sum() and lg.sum():
        # single-threshold separability gaster vs leg
        ths = np.linspace(sdf_v[np.isfinite(sdf_v)].min(), sdf_v[np.isfinite(sdf_v)].max(), 400)
        best = max(((sdf_v[ga] > t).mean() * 0.5 + (sdf_v[lg] <= t).mean() * 0.5, t) for t in ths)
        P(f"TEMPLATE gaster-vs-leg best balanced accuracy by ONE SDF threshold = {best[0]*100:.2f}%")
        # AUC
        from itertools import product  # noqa
        a, b = sdf_v[ga], sdf_v[lg]
        auc = (a[:, None] > b[None, :]).mean()
        P(f"TEMPLATE gaster-vs-leg AUC = {auc:.4f}")

    # ---------- (b) real scans ----------
    import glob
    scans = sorted(glob.glob("/home/fabi/dev/SMILify/diagnostics/moonshot/bench50/*.obj"))[:6]
    P("")
    P("REAL SCANS (bench50) — G1 reproducibility across two independent surface samplings, G4 cost")
    for path in scans:
        m = trimesh.load(path, process=False, force="mesh")
        comps = m.split(only_watertight=False)
        t0 = time.time()
        p1, fi1 = trimesh.sample.sample_surface(m, 4000)
        n1 = m.face_normals[fi1]
        p2, fi2 = trimesh.sample.sample_surface(m, 4000)
        n2 = m.face_normals[fi2]
        s1 = sdf_field(m, p1, n1, n_rays=25, cone_deg=120.0)
        s2 = sdf_field(m, p2, n2, n_rays=25, cone_deg=120.0)
        dt = time.time() - t0
        # transfer sampling-2 values onto sampling-1 points by nearest neighbour and correlate
        from scipy.spatial import cKDTree
        ok1, ok2 = np.isfinite(s1), np.isfinite(s2)
        tree = cKDTree(p2[ok2])
        _, nn = tree.query(p1[ok1])
        a, b = s1[ok1], s2[ok2][nn]
        r = np.corrcoef(a, b)[0, 1]
        rel = np.median(np.abs(a - b) / (0.5 * (a + b) + 1e-12))
        # 2-way threshold agreement (thick vs thin), the coarse split that matters
        t1, t2 = np.median(a), np.median(b)
        agree = ((a > t1) == (b > t2)).mean()
        P(f"  {path.split('/')[-1][:42]:44s} wt={str(m.is_watertight)[0]} comp={len(comps):3d} "
          f"nan={np.mean(~ok1):.3f} r={r:.4f} med_rel_diff={rel:.4f} 2way_agree={agree*100:5.2f}% {dt:.1f}s")

    open("/home/fabi/dev/SMILify/diagnostics/moonshot/sdf_prior_out.txt", "w").write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
