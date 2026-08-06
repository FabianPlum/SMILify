"""REFUTE-2 -- measure the FALSE-OUTSIDE FLOOR of the "joint is outside the scan" oracle,
separately for each corpus, using control points that are inside the animal BY CONSTRUCTION
and have nothing to do with any fit.

WHY.  The headline (worker 61.5% of joints outside vs clean 30.1%) uses a generalized winding
number on the raw scan.  REFUTE-1 measured that 0/50 worker scans are winding-consistent
(20/81 clean are), worker boundary-edge fraction is 5.5% vs 3.2%, and an exterior flood fill
leaks through the worker surface everywhere: the largest enclosed interior blob holds 23-27%
of interior voxels for workers vs 96-97% for clean.  A GWN oracle on such a mesh reports
"outside" for genuinely interior points.  If that false-outside rate differs by corpus, the
headline comparison is measuring mesh provenance, not joint placement.

METHOD (per scan, orientation-free, fit-free)
  1. normalise exactly as the fit did (fitter_3d/utils.py:340-344)
  2. voxelise the surface at pitch = diag/N  (subdivision rasteriser -- ignores orientation)
  3. morphological CLOSING with a ball of radius R voxels -> seals holes up to ~2R voxels
  4. flood fill the exterior with 6-connectivity; INTERIOR = not-closed-surface and not-reached
  5. keep the LARGEST interior component only, then ERODE it by R voxels
     -> DEEP INTERIOR: voxels at least R voxels inside a sealed surface.  These points are
        inside the animal under a criterion that uses no face orientation and tolerates holes.
  6. evaluate the HEADLINE ORACLE (|GWN| < 0.5) at random deep-interior points
     -> the fraction it wrongly calls OUTSIDE is the corpus's false-outside FLOOR.
  7. re-score the actual fitted JOINTS with the ROBUST oracle (step 4/5 occupancy) and with
     the headline GWN oracle, so the two can be compared on identical points.

Also reported: the maximum inscribed radius (Euclidean distance transform of the interior)
at each joint -- how much room a joint has to be "inside" at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import trimesh
from scipy import ndimage
from scipy.stats import mannwhitneyu
from joint_placement_common import load_run, winding_number
from joint_placement_vs_scan_PROBE import posed_skeleton

ROOT = "/home/fabi/dev/SMILify"
OUT = os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_oracle_out.txt")
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

RUNS = [
    (
        "LIM_0 (worker)",
        "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
        "diagnostics/moonshot/bench50",
    ),
    (
        "baseline (worker)",
        "diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
        False,
        "diagnostics/moonshot/bench50",
    ),
    ("ALL_ANTS_CLEAN", CLEAN_NPZ, "3D_model_prep/SMPL_fit.pkl", True, "diagnostics/moonshot/clean81"),
]

N_GRID = int(os.environ.get("N_GRID", "288"))
R_CLOSE = int(os.environ.get("R_CLOSE", "3"))
N_CTRL = int(os.environ.get("N_CTRL", "120"))
N_SPEC = int(os.environ.get("N_SPEC", "0")) or None


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def normalise(V):
    V = V - V.mean(0)
    return V / np.abs(V).max(0).max()


def ball(r):
    z, y, x = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (z * z + y * y + x * x) <= r * r


def build_occupancy(V, F, n_grid=N_GRID, r_close=R_CLOSE):
    """Returns (interior_bool_grid, origin, pitch, edt_in_units)."""
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    pitch = diag / n_grid
    mm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    vg = trimesh.voxel.creation.voxelize_subdivide(mm, pitch)
    occ = np.asarray(vg.matrix, dtype=bool)
    origin = np.asarray(vg.transform)[:3, 3]  # world coord of voxel (0,0,0) centre
    pad = r_close + 2
    occ = np.pad(occ, pad, constant_values=False)
    origin = origin - pad * pitch
    se = ball(r_close)
    closed = ndimage.binary_closing(occ, structure=se)
    free = ~closed
    lab, _ = ndimage.label(free, structure=ndimage.generate_binary_structure(3, 1))
    ext = lab == lab[0, 0, 0]
    interior = free & (~ext)
    li, nli = ndimage.label(interior)
    if nli:
        sz = np.bincount(li.ravel())
        sz[0] = 0
        big = int(sz.argmax())
        frac_big = sz[big] / max(1, interior.sum())
        interior_main = li == big
    else:
        frac_big = 0.0
        interior_main = interior
    # solid = interior + the closed shell, used as the robust inside/outside test
    solid = interior | closed
    edt = ndimage.distance_transform_edt(solid) * pitch
    deep = ndimage.binary_erosion(interior_main, structure=ball(r_close))
    return dict(
        solid=solid,
        interior=interior,
        interior_main=interior_main,
        deep=deep,
        origin=origin,
        pitch=pitch,
        diag=diag,
        edt=edt,
        frac_big=frac_big,
        n_int=int(interior.sum()),
        n_deep=int(deep.sum()),
        n_surf=int(occ.sum()),
    )


def sample_at(grid, origin, pitch, pts):
    idx = np.floor((pts - origin) / pitch + 0.5).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(grid.shape)), axis=1)
    out = np.zeros(pts.shape[0], dtype=bool)
    out[ok] = grid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out


def sample_edt(edt, origin, pitch, pts):
    idx = np.floor((pts - origin) / pitch + 0.5).astype(int)
    idx = np.clip(idx, 0, np.array(edt.shape) - 1)
    return edt[idx[:, 0], idx[:, 1], idx[:, 2]]


def load_obj_tm(path):
    m = trimesh.load(path, process=True, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=int)


def main():
    rng = np.random.default_rng(0)
    store = {}
    with open(OUT, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        print(
            f"grid N={N_GRID}  closing radius={R_CLOSE} voxels "
            f"(= {200.0 * R_CLOSE / N_GRID:.2f}% of the scan diagonal)  control pts/scan={N_CTRL}",
            file=fh,
        )
        for tag, npz, mdl, align, scan_dir in RUNS:
            npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
            R = load_run(npz, os.path.join(ROOT, mdl), align=align)
            scan_dir = os.path.join(ROOT, scan_dir)
            avail = {
                os.path.splitext(x)[0]: os.path.join(scan_dir, x) for x in os.listdir(scan_dir) if x.endswith(".obj")
            }
            Jp = posed_skeleton(R)
            labels = [os.path.splitext(str(x))[0] for x in R["d"]["labels"]]
            nmax = R["n"] if N_SPEC is None else min(R["n"], N_SPEC)
            rows = []
            for i in range(nmax):
                if labels[i] not in avail:
                    continue
                V, F = load_obj_tm(avail[labels[i]])
                V = normalise(V)
                G = build_occupancy(V, F)
                j = Jp[i]
                # ---- headline oracle on the joints
                w = winding_number(j, V, F)
                gwn_out = np.abs(w) < 0.5
                # ---- robust oracle on the same joints
                rob_out = ~sample_at(G["solid"], G["origin"], G["pitch"], j)
                # ---- room available: max inscribed radius at the joint
                room = sample_edt(G["edt"], G["origin"], G["pitch"], j) / G["diag"]
                # ---- FALSE-OUTSIDE FLOOR: control points inside by construction
                dz = np.argwhere(G["deep"])
                if dz.shape[0] >= 10:
                    sel = dz[rng.choice(dz.shape[0], size=min(N_CTRL, dz.shape[0]), replace=False)]
                    cp = G["origin"] + sel * G["pitch"]
                    wc = winding_number(cp, V, F)
                    floor = float((np.abs(wc) < 0.5).mean())
                    ncp = sel.shape[0]
                else:
                    floor, ncp = np.nan, 0
                rows.append(
                    (
                        float(gwn_out.mean()),
                        float(rob_out.mean()),
                        floor,
                        float(np.median(room)),
                        G["frac_big"],
                        G["n_deep"],
                        ncp,
                        G["n_int"],
                        G["n_surf"],
                    )
                )
                if len(rows) % 10 == 0:
                    print(f"   ... {tag} {len(rows)} specimens", file=sys.stderr)
            a = np.array(rows, dtype=np.float64)
            store[tag] = a
            print(f"\n### {tag}   n={a.shape[0]}", file=fh)
            names = [
                "GWN %joints outside (HEADLINE)",
                "ROBUST %joints outside",
                "FALSE-OUTSIDE FLOOR (deep interior pts called outside by GWN)",
                "median inscribed radius at joints (%diag)",
                "frac of interior in largest blob",
                "n deep-interior voxels",
                "n control pts",
                "n interior voxels",
                "n surface voxels",
            ]
            for k, nm in enumerate(names):
                col = a[:, k]
                col = col[~np.isnan(col)]
                scale = 100.0 if k in (0, 1, 2, 3) else 1.0
                print(
                    f"   {nm:<62}{scale * np.median(col):10.3f}  (mean {scale * col.mean():.3f}, "
                    f"sd {scale * col.std():.3f})",
                    file=fh,
                )
            fh.flush()

        A = store["LIM_0 (worker)"]
        B = store["ALL_ANTS_CLEAN"]
        Bl = store["baseline (worker)"]
        print("\n" + "=" * 96, file=fh)
        print("WORKER vs CLEAN, per specimen (Mann-Whitney two-sided)", file=fh)
        print(f"{'metric':<44}{'worker':>10}{'clean':>10}{'ratio':>9}{'p':>12}", file=fh)
        for k, nm in enumerate(
            ["GWN %outside (HEADLINE)", "ROBUST %outside", "FALSE-OUTSIDE FLOOR", "inscribed radius %diag"]
        ):
            x = A[:, k][~np.isnan(A[:, k])]
            y = B[:, k][~np.isnan(B[:, k])]
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            print(
                f"{nm:<44}{100 * np.median(x):10.3f}{100 * np.median(y):10.3f}"
                f"{np.median(x) / np.median(y):9.2f}{p:12.2e}",
                file=fh,
            )

        print("\nHEADLINE EXCESS ABOVE EACH CORPUS' OWN FALSE-OUTSIDE FLOOR", file=fh)
        for nm, M in (("worker LIM_0", A), ("worker baseline", Bl), ("clean", B)):
            ok = ~np.isnan(M[:, 2])
            exc = M[ok, 0] - M[ok, 2]
            print(
                f"   {nm:<18} GWN %outside={100 * np.median(M[:, 0]):6.2f}  floor="
                f"{100 * np.median(M[ok, 2]):6.2f}  excess median={100 * np.median(exc):+6.2f}"
                f"  (mean {100 * exc.mean():+.2f})",
                file=fh,
            )
        okA, okB = ~np.isnan(A[:, 2]), ~np.isnan(B[:, 2])
        u, p = mannwhitneyu(A[okA, 0] - A[okA, 2], B[okB, 0] - B[okB, 2], alternative="two-sided")
        print(f"   worker-vs-clean on the EXCESS: p={p:.2e}", file=fh)

        print("\nPAIRED, SAME 50 SCANS: LIM_0 vs stock baseline", file=fh)
        n = min(A.shape[0], Bl.shape[0])
        from scipy.stats import wilcoxon

        for k, nm in enumerate(["GWN %outside (HEADLINE)", "ROBUST %outside"]):
            d = A[:n, k] - Bl[:n, k]
            try:
                s, p = wilcoxon(A[:n, k], Bl[:n, k])
            except Exception:
                p = np.nan
            print(
                f"   {nm:<28} LIM_0={100 * np.median(A[:n, k]):6.2f}  baseline="
                f"{100 * np.median(Bl[:n, k]):6.2f}  median diff={100 * np.median(d):+6.2f}"
                f"  paired p={p:.2e}",
                file=fh,
            )
        print("=" * 96, file=fh)
        np.savez(
            os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_oracle.npz"),
            **{k.split()[0]: v for k, v in store.items()},
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
