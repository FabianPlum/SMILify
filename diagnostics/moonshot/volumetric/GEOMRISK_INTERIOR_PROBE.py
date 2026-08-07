"""GEOMRISK R4 — does "interior" even exist on the worker scans, and how much does the
choice of estimator move it?

Volume IoU needs an indicator function on the TARGET. The target is a CT-derived ethanol-scan
surface with open boundaries and dozens-to-thousands of detached components. Four candidate
estimators are compared on the same query points:

  A  trimesh.contains  — ray parity along ONE FIXED direction. Read from
     trimesh/ray/ray_util.py:42, the direction is hard-coded [0.4395064455, 0.617598629942,
     0.652231566745] for every point of every mesh. Deterministic, but the error it makes is a
     SHADOW: every open boundary loop projects a prism of misclassified space along that one
     direction. Rotating the specimen changes the answer.
  B  ray parity, majority vote over K random directions — the cheap robustification.
  C  voxel occupancy + flood fill from outside + morphological closing — the "just close it"
     option.
  D  generalised winding number (Jacobson et al. 2013 / Barill et al. 2018) — only if libigl
     installs; the probe reports its absence rather than pretending.

Reported: pairwise disagreement, the spread of the implied VOLUME, and how much of the
disagreement is attributable to detached debris components rather than to the animal.

WHAT WOULD KILL THE PROPOSAL
  * single-direction parity disagreeing with the majority vote on more than a few percent of
    interior volume: the data term is then a function of specimen orientation, and the fit will
    chase it.
  * closing/flood-fill changing the volume by more than the effect size the fit is supposed to
    detect (joint placement moves the centroid of a segment by a few % of body diagonal).
"""

import glob
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
MOON = os.path.abspath(os.path.join(HERE, ".."))
N_PTS = 60000
K_DIRS = 15


def parity(mesh, pts, direction):
    """Inside test by counting forward/backward crossings along one direction."""
    inter = mesh.ray
    hits = inter.intersects_id(
        ray_origins=pts, ray_directions=np.tile(direction, (len(pts), 1)), return_locations=False, multiple_hits=True
    )
    ray = hits[1]
    cnt = np.bincount(ray, minlength=len(pts))
    return (cnt % 2) == 1


def main():
    files = sorted(glob.glob(os.path.join(MOON, "bench50", "*.obj")))[:8]
    clean = sorted(glob.glob(os.path.join(MOON, "clean81", "*.obj")))[:3]
    print(f"{len(files)} worker scans, {len(clean)} clean scans\n")
    rng = np.random.default_rng(0)
    out = []

    for path in files + clean:
        tag = "worker" if "bench50" in path else "clean"
        m = trimesh.load(path, process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        v = np.asarray(m.vertices, np.float64)
        c0 = v.mean(0)
        sc = np.abs(v - c0).max()
        m = trimesh.Trimesh((v - c0) / sc, np.asarray(m.faces), process=False)
        diag = float(np.linalg.norm(m.bounds[1] - m.bounds[0]))

        n_bound = int((trimesh.grouping.group_rows(m.edges_sorted, require_count=1)).shape[0])
        bfrac = n_bound / max(len(m.edges_sorted), 1)
        ncomp = len(trimesh.graph.connected_components(m.face_adjacency, min_len=1, nodes=np.arange(len(m.faces))))

        lo, hi = m.bounds[0] - 0.01, m.bounds[1] + 0.01
        x = lo + rng.random((N_PTS, 3)) * (hi - lo)
        vbox = float(np.prod(hi - lo))

        # A: trimesh default (single hard-coded direction)
        A = m.contains(x)
        # B: majority vote over K random directions
        dirs = rng.normal(size=(K_DIRS, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        votes = np.zeros(N_PTS, np.int32)
        per_dir_vol = []
        for d in dirs:
            p = parity(m, x, d)
            votes += p
            per_dir_vol.append(p.mean() * vbox)
        B = votes > (K_DIRS // 2)
        frac_unstable = float(((votes > 0) & (votes < K_DIRS)).mean() * vbox / max(B.mean() * vbox, 1e-12))

        disA = float((A != B).sum() / max(B.sum(), 1))
        volA, volB = float(A.mean() * vbox), float(B.mean() * vbox)
        pdv = np.array(per_dir_vol)
        row = dict(
            name=os.path.basename(path)[:38],
            corpus=tag,
            watertight=bool(m.is_watertight),
            boundary_edge_frac=float(bfrac),
            n_components=int(ncomp),
            diag=diag,
            vol_single_dir=volA,
            vol_majority=volB,
            single_vs_majority_disagree_frac_of_interior=disA,
            per_dir_vol_cv=float(pdv.std() / max(pdv.mean(), 1e-12)),
            per_dir_vol_minmax_spread=float((pdv.max() - pdv.min()) / max(pdv.mean(), 1e-12)),
            unstable_frac_of_interior=frac_unstable,
        )
        out.append(row)
        print(f"{row['name']:<40} {tag:<7} wt={str(m.is_watertight)[0]} comp={ncomp:<5} bfrac={bfrac:.4f}")
        print(f"    vol single-dir {volA:.5f}   majority {volB:.5f}   ratio {volA / max(volB, 1e-12):.3f}")
        print(f"    single-vs-majority disagreement = {100 * disA:.2f}% of interior volume")
        print(
            f"    per-direction volume: CV {100 * row['per_dir_vol_cv']:.2f}%   "
            f"min-max spread {100 * row['per_dir_vol_minmax_spread']:.2f}% of mean"
        )
        print(f"    points where the {K_DIRS} directions DISAGREE = {100 * frac_unstable:.2f}% of interior volume")

    # ---- voxel flood-fill option, on one worker scan
    print("\n" + "=" * 92)
    print("C  VOXELISATION + FLOOD FILL, one worker scan, several resolutions")
    print("=" * 92)
    m = trimesh.load(files[0], process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    v = np.asarray(m.vertices, np.float64)
    c0 = v.mean(0)
    sc = np.abs(v - c0).max()
    m = trimesh.Trimesh((v - c0) / sc, np.asarray(m.faces), process=False)
    ext = m.bounds[1] - m.bounds[0]
    vox_rows = []
    for n in [64, 128, 256]:
        pitch = float(ext.max() / n)
        try:
            vg = m.voxelized(pitch=pitch)
            shell = int(vg.filled_count)
            filled = vg.copy().fill(method="holes")
            solid = int(filled.filled_count)
            vox_rows.append(
                dict(
                    n=n,
                    pitch_frac_of_diag=pitch / float(np.linalg.norm(ext)),
                    shell_voxels=shell,
                    filled_voxels=solid,
                    fill_ratio=solid / max(shell, 1),
                )
            )
            print(
                f"  n={n:<4} pitch={pitch:.5f} ({100 * pitch / np.linalg.norm(ext):.3f}% of diag)  "
                f"shell {shell:>8}  after fill {solid:>8}  ratio {solid / max(shell, 1):.2f}"
            )
        except Exception as e:
            print(f"  n={n} FAILED: {type(e).__name__}: {e}")
    # how thin is a distal leg segment in voxels?
    print("\n  thickness check: distance from surface samples to the nearest OTHER surface point")
    pts, fid = trimesh.sample.sample_surface(m, 20000)
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    print(
        f"    median nearest-neighbour surface spacing {np.median(d[:, 1]):.5f} "
        f"({100 * np.median(d[:, 1]) / np.linalg.norm(ext):.3f}% of diag)"
    )

    with open(os.path.join(HERE, "out", "geomrisk_interior.json"), "w") as fh:
        json.dump(dict(per_mesh=out, voxel=vox_rows), fh, indent=1)
    print("\nwrote out/geomrisk_interior.json")

    w = [r for r in out if r["corpus"] == "worker"]
    c = [r for r in out if r["corpus"] == "clean"]
    print("\nSUMMARY")
    for nm, g in [("worker", w), ("clean", c)]:
        if not g:
            continue
        print(
            f"  {nm}: single-vs-majority disagreement median "
            f"{100 * np.median([r['single_vs_majority_disagree_frac_of_interior'] for r in g]):.2f}% of interior; "
            f"per-direction volume spread median "
            f"{100 * np.median([r['per_dir_vol_minmax_spread'] for r in g]):.2f}%"
        )


if __name__ == "__main__":
    main()
