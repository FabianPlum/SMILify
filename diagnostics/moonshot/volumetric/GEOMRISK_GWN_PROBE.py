"""GEOMRISK R4b — generalised winding number vs ray parity on the real worker scans, plus an
honest re-measurement of the scans' topology (the earlier pass loaded with process=False, so
every unwelded duplicate vertex counted as a boundary and a component).

libigl DOES install in this env:  pip install --target <dir> libigl  -> libigl 2.6.2, exposes
igl.fast_winding_number (Barill et al., SIGGRAPH 2018) and igl.winding_number (Jacobson et al.,
SIGGRAPH 2013). The wheel drags numpy 2.x with it, which is incompatible with this env; delete
the vendored numpy/scipy from the target dir and it imports cleanly against numpy 1.26.4.

Measured here:
  * welded topology: boundary-edge fraction, component count, area held by debris
  * GWN field: what fraction of query points fall in the ambiguous band 0.2 < w < 0.8
  * GWN(0.5) vs 15-direction majority parity: volume ratio and disagreement
  * GWN stability under rotation of the specimen (the property ray parity fails)
  * voxel resolution needed for a distal leg segment to be more than one voxel thick
"""

import glob
import json
import os
import sys

import numpy as np
import trimesh

sys.path.append("/tmp/claude-1000/-home-fabi-dev-SMILify/81ae7faa-646c-46d1-9dd4-0b1fc0d775d6/scratchpad/libigl_t")
import igl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MOON = os.path.abspath(os.path.join(HERE, ".."))
N = 60000
K = 15


def welded_stats(m):
    mm = m.copy()
    mm.merge_vertices()
    e = mm.edges_sorted
    nb = trimesh.grouping.group_rows(e, require_count=1).shape[0]
    comps = trimesh.graph.connected_components(mm.face_adjacency, min_len=1, nodes=np.arange(len(mm.faces)))
    areas = np.array([mm.area_faces[c].sum() for c in comps])
    order = np.argsort(areas)[::-1]
    tot = areas.sum()
    return dict(
        n_boundary_edges=int(nb),
        boundary_edge_frac=float(nb / max(len(e), 1)),
        n_components=int(len(comps)),
        largest_component_area_frac=float(areas[order[0]] / tot),
        debris_area_frac=float(1.0 - areas[order[0]] / tot),
        n_comp_above_1pct=int((areas / tot > 0.01).sum()),
    )


def parity_majority(m, x, rng, k=K):
    votes = np.zeros(len(x), np.int32)
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    for dd in d:
        h = m.ray.intersects_id(
            ray_origins=x, ray_directions=np.tile(dd, (len(x), 1)), return_locations=False, multiple_hits=True
        )
        votes += (np.bincount(h[1], minlength=len(x)) % 2) == 1
    return votes > (k // 2)


def main():
    files = sorted(glob.glob(os.path.join(MOON, "bench50", "*.obj")))[:6]
    rng = np.random.default_rng(1)
    rows = []
    for path in files:
        raw = trimesh.load(path, process=False)
        if isinstance(raw, trimesh.Scene):
            raw = trimesh.util.concatenate(list(raw.geometry.values()))
        v = np.asarray(raw.vertices, np.float64)
        c0 = v.mean(0)
        sc = np.abs(v - c0).max()
        m = trimesh.Trimesh((v - c0) / sc, np.asarray(raw.faces), process=False)
        st = welded_stats(m)

        lo, hi = m.bounds[0] - 0.01, m.bounds[1] + 0.01
        x = lo + rng.random((N, 3)) * (hi - lo)
        vbox = float(np.prod(hi - lo))

        V = np.ascontiguousarray(m.vertices, np.float64)
        F = np.ascontiguousarray(m.faces, np.int32)
        w = igl.fast_winding_number(V, F, np.ascontiguousarray(x, np.float64))
        gwn = w > 0.5
        amb = float(((w > 0.2) & (w < 0.8)).sum() * vbox / N)

        par = parity_majority(m, x, np.random.default_rng(2))
        vol_g, vol_p = float(gwn.mean() * vbox), float(par.mean() * vbox)
        dis = float((gwn != par).sum() / max(par.sum(), 1))

        # rotation stability of GWN: rotate the mesh AND the query points by the same R -> the
        # answer must be identical. Rotate only the mesh and re-query the same world points
        # through the inverse rotation: any difference is estimator-not-geometry.
        R = trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3]
        w2 = igl.fast_winding_number(
            np.ascontiguousarray(V @ R.T, np.float64), F, np.ascontiguousarray(x @ R.T, np.float64)
        )
        rot_dis = float(((w2 > 0.5) != gwn).sum() / max(gwn.sum(), 1))

        # same test for ray parity: rotate the specimen, re-run the FIXED-direction trimesh test
        mrot = trimesh.Trimesh(V @ R.T, F, process=False)
        p0 = m.contains(x)
        p1 = mrot.contains(x @ R.T)
        rot_dis_ray = float((p0 != p1).sum() / max(p0.sum(), 1))

        rows.append(
            dict(
                name=os.path.basename(path)[:38],
                **st,
                vol_gwn=vol_g,
                vol_parity=vol_p,
                gwn_vs_parity_disagree=dis,
                ambiguous_band_vol_frac_of_interior=amb / max(vol_g, 1e-12),
                gwn_rotation_instability=rot_dis,
                ray_rotation_instability=rot_dis_ray,
            )
        )
        print(f"{rows[-1]['name']:<40}")
        print(
            f"   WELDED: bfrac={st['boundary_edge_frac']:.4f}  comps={st['n_components']:<5} "
            f"largest={100 * st['largest_component_area_frac']:.2f}% of area  "
            f"debris={100 * st['debris_area_frac']:.2f}%  comps>1% area={st['n_comp_above_1pct']}"
        )
        print(
            f"   volume  GWN {vol_g:.5f}   15-dir majority parity {vol_p:.5f}   ratio {vol_g / max(vol_p, 1e-12):.3f}"
        )
        print(f"   GWN vs parity disagreement {100 * dis:.2f}% of interior")
        print(
            f"   GWN ambiguous band 0.2<w<0.8: {100 * rows[-1]['ambiguous_band_vol_frac_of_interior']:.2f}% of interior volume"
        )
        print(
            f"   ROTATION INSTABILITY (same geometry, rotated):  GWN {100 * rot_dis:.3f}%   "
            f"trimesh.contains {100 * rot_dis_ray:.2f}%"
        )

    print("\n" + "=" * 92)
    print("VOXEL RESOLUTION NEEDED FOR THE APPENDAGES")
    print("=" * 92)
    import pickle

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
    import config

    with open(os.path.join(os.path.abspath(os.path.join(HERE, "..", "..", "..")), config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    vt = np.asarray(dd["v_template"], np.float64)
    dom = np.asarray(dd["weights"]).argmax(1)
    jn = [str(s) for s in dd["J_names"]]
    diag = float(np.linalg.norm(vt.max(0) - vt.min(0)))
    print(f"  template bbox diagonal = {diag:.4f} (model units)")
    print(f"  {'segment':<14}{'minor diameter':>16}{'% of diag':>11}{'grid N for 2 voxels':>22}")
    thin = []
    for j in range(len(jn)):
        p = vt[dom == j]
        if len(p) < 20:
            continue
        d = p - p.mean(0)
        _, s, _ = np.linalg.svd(d, full_matrices=False)
        ax = np.percentile(np.abs(d @ np.linalg.svd(d, full_matrices=False)[2].T), 98, axis=0)
        minor = 2 * min(ax)
        thin.append((jn[j], minor, minor / diag, int(np.ceil(2 * diag / minor))))
    for nm, mi, fr, gn in sorted(thin, key=lambda t: t[1])[:10]:
        print(f"  {nm:<14}{mi:>16.5f}{100 * fr:>10.3f}%{gn:>22d}")
    with open(os.path.join(HERE, "out", "geomrisk_gwn.json"), "w") as fh:
        json.dump(
            dict(
                per_mesh=rows,
                thin_segments=[dict(name=t[0], minor_diam=t[1], frac_diag=t[2], grid_N=t[3]) for t in thin],
            ),
            fh,
            indent=1,
        )
    print("\nwrote out/geomrisk_gwn.json")


if __name__ == "__main__":
    main()
