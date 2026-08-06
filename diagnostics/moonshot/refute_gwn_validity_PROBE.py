"""REFUTE-D -- is the inside/outside verdict trustworthy on these scans at JOINT DEPTH?

The generalized winding number is only 1-inside/0-outside for a CLOSED, consistently
oriented surface. Measured in refute_scan_integrity_out.txt:

    worker scans: median 8075 boundary edges, 4095 non-manifold edges, 44.5 components
    clean  scans: median 1999 boundary edges, 5239 non-manifold edges,  2.0 components

The prior probe's validity check offsets random face centroids by 1e-3 * diag -- i.e. it
only ever probes GWN at 0.1% of the body diagonal from the surface. The joints it then
scores sit at a median gap of 0.28-0.50% of the diagonal, 3-5x deeper, and inside thin
appendages. This probe measures the FALSE-OUTSIDE RATE of GWN as a function of depth, on
points that are inside by construction (surface vertices pushed along -normal), split by
body region, for both corpora.

If the worker false-outside rate at joint depth is much higher than the clean one, the
61.5% vs 30.1% headline is (at least partly) a scan-topology artefact.
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_placement_common import joint_group, load_model, winding_number  # noqa: E402

ROOT = "/home/fabi/dev/SMILify"
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)
RUNS = [
    (
        "worker",
        f"{ROOT}/diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        f"{ROOT}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        f"{ROOT}/diagnostics/moonshot/bench50",
        12,
    ),
    ("clean", CLEAN_NPZ, f"{ROOT}/3D_model_prep/SMPL_fit.pkl", f"{ROOT}/diagnostics/moonshot/clean81", 20),
]
DEPTH = [0.001, 0.002, 0.004, 0.008]
NPTS = 100
CHUNK = 2000


def load_obj_np(path):
    V, F = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, int)


def normalise(V):
    V = V - V.mean(0)
    return V / np.abs(V).max(0).max()


def vertex_normals(V, F):
    n = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(ln > 0, n / np.where(ln > 0, ln, 1), 0.0), (ln[:, 0] > 0)


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = f"{ROOT}/diagnostics/moonshot/refute_gwn_validity_out.txt"
    rng = np.random.default_rng(11)
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        res = {}
        for tag, npz, mdl, sdir, nmax in RUNS:
            d = np.load(npz, allow_pickle=True)
            dd = load_model(mdl)
            names = list(dd["J_names"])
            W = np.asarray(dd["weights"], float)
            dom = np.argmax(W, axis=1)
            grp_of_vert = np.array([joint_group(names[j]) for j in dom])
            is_leg = np.char.startswith(grp_of_vert.astype(str), "leg")
            labels = [os.path.splitext(str(x))[0] for x in d["labels"]]
            avail = {os.path.splitext(x)[0]: os.path.join(sdir, x) for x in os.listdir(sdir) if x.endswith(".obj")}
            acc = {("all", z): [] for z in DEPTH}
            acc.update({("leg", z): [] for z in DEPTH})
            out_ctrl = []
            used = 0
            for i in range(len(labels)):
                if used >= nmax:
                    break
                if labels[i] not in avail:
                    continue
                V, F = load_obj_np(avail[labels[i]])
                V = normalise(V)
                diag = np.linalg.norm(V.max(0) - V.min(0))
                N, ok = vertex_normals(V, F)
                fv = d["verts"][i].astype(np.float64)
                # region label for each scan vertex = region of the nearest fitted vertex
                _, nb = cKDTree(fv).query(V)
                legmask = is_leg[nb]
                # orient normals outward: the sign that puts the offset point OUTSIDE
                cand = np.where(ok)[0]
                probe = rng.choice(cand, size=min(60, cand.size), replace=False)
                wpos = winding_number(V[probe] + 0.001 * diag * N[probe], V, F, CHUNK)
                wneg = winding_number(V[probe] - 0.001 * diag * N[probe], V, F, CHUNK)
                # `sign` must point INWARD: pick the offset direction whose |w| is larger.
                sign = -1.0 if np.median(np.abs(wneg)) > np.median(np.abs(wpos)) else 1.0
                pts, meta = [], []
                for key, mask in (("all", ok), ("leg", ok & legmask)):
                    c = np.where(mask)[0]
                    if c.size < 20:
                        continue
                    sel = rng.choice(c, size=min(NPTS, c.size), replace=False)
                    for z in DEPTH:
                        pts.append(V[sel] + sign * z * diag * N[sel])
                        meta.append((key, z, sel.size))
                # outward control at the shallowest depth
                sel = rng.choice(np.where(ok)[0], size=min(NPTS, ok.sum()), replace=False)
                pts.append(V[sel] - sign * DEPTH[0] * diag * N[sel])
                meta.append(("OUT", DEPTH[0], sel.size))
                P = np.concatenate(pts, 0)
                w = winding_number(P, V, F, CHUNK)
                outside = np.abs(w) < 0.5
                k = 0
                for key, z, m in meta:
                    seg = outside[k : k + m]
                    k += m
                    if key == "OUT":
                        out_ctrl.append(1.0 - seg.mean())  # should be ~0
                    else:
                        acc[(key, z)].append(seg.mean())
                used += 1
            res[tag] = acc
            print(f"\n### {tag}   n_scans={used}", file=fh)
            print(
                f"   sanity: fraction of OUTWARD-offset points wrongly called INSIDE at "
                f"{100 * DEPTH[0]:.1f}%diag = {100 * np.median(out_ctrl):.2f}%",
                file=fh,
            )
            print(f"   {'depth %diag':<14}{'FALSE-OUTSIDE rate, all verts':>32}{'leg verts only':>18}", file=fh)
            for z in DEPTH:
                a = np.array(acc[("all", z)])
                b = np.array(acc[("leg", z)])
                print(f"   {100 * z:<14.2f}{100 * np.median(a):32.1f}{100 * np.median(b):18.1f}", file=fh)
            fh.flush()
        print("\n" + "=" * 92, file=fh)
        print("FALSE-OUTSIDE RATE, worker vs clean (median over scans)", file=fh)
        print(
            f"{'depth %diag':<14}{'all: worker':>13}{'clean':>9}{'ratio':>8}"
            f"{'  |  leg: worker':>18}{'clean':>9}{'ratio':>8}",
            file=fh,
        )
        for z in DEPTH:
            wa = np.median(res["worker"][("all", z)])
            ca = np.median(res["clean"][("all", z)])
            wl = np.median(res["worker"][("leg", z)])
            cl = np.median(res["clean"][("leg", z)])
            print(
                f"{100 * z:<14.2f}{100 * wa:13.1f}{100 * ca:9.1f}{wa / max(ca, 1e-9):8.2f}"
                f"{100 * wl:18.1f}{100 * cl:9.1f}{wl / max(cl, 1e-9):8.2f}",
                file=fh,
            )
        print("=" * 92, file=fh)
    print(f"wrote {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
