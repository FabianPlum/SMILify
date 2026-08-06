"""REFUTE-1 -- is the "% joints outside the scan" statistic a property of the FIT, or of the SCAN?

The headline claim rests on a generalized winding number (GWN) test:  |w| < 0.5  =>  the joint
is outside the animal.  GWN is only a valid inside/outside oracle on a CLOSED, CONSISTENTLY
ORIENTED mesh (Jacobson et al. 2013 degrade gracefully with holes, but NOT with flipped faces:
a flipped patch subtracts solid angle and drives |w| toward 0, i.e. toward "outside").

The two corpora come from two entirely different mesh pipelines:
    worker  /media/fabi/Data/.../antscan_proofread_castes/worker/*_processed.obj  (raw isosurface)
    clean   /media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN/*.obj         (Blender 2.92 export)

so the SCAN is a corpus-level variable that is perfectly confounded with the FIT.  This probe
audits the topology of every scan actually used and asks whether the GWN oracle is equally
trustworthy in both corpora.
"""

import os
import sys
import glob

import numpy as np
import trimesh

ROOT = "/home/fabi/dev/SMILify"
OUT = os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_topology_out.txt")


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def audit(path):
    m = trimesh.load(path, process=False, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=int)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    # boundary edges = edges used by exactly one face
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    n_edges = cnt.shape[0]
    bnd = float((cnt == 1).mean())
    nonman = float((cnt > 2).mean())
    try:
        ncomp = int(m.body_count)
    except Exception:
        ncomp = -1
    return dict(
        nv=V.shape[0],
        nf=F.shape[0],
        ne=n_edges,
        diag=diag,
        watertight=bool(m.is_watertight),
        wind_ok=bool(m.is_winding_consistent),
        bnd_frac=bnd,
        nonman_frac=nonman,
        ncomp=ncomp,
        euler=int(V.shape[0] - n_edges + F.shape[0]),
        vol=float(abs(m.volume)) if m.is_volume else float("nan"),
    )


def main():
    with open(OUT, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        res = {}
        for d in ("bench50", "clean81"):
            rows = []
            for p in sorted(glob.glob(os.path.join(ROOT, "diagnostics/moonshot", d, "*.obj"))):
                rows.append(audit(p))
            res[d] = rows
            print(f"\n### {d}   n={len(rows)}", file=fh)
            keys = ["nv", "nf", "bnd_frac", "nonman_frac", "ncomp", "euler"]
            print(f"   {'field':<14}{'median':>12}{'mean':>12}{'min':>12}{'max':>12}", file=fh)
            for k in keys:
                a = np.array([r[k] for r in rows], dtype=np.float64)
                print(f"   {k:<14}{np.median(a):12.4f}{a.mean():12.4f}{a.min():12.4f}{a.max():12.4f}", file=fh)
            wt = np.array([r["watertight"] for r in rows])
            wo = np.array([r["wind_ok"] for r in rows])
            print(f"   watertight   : {wt.sum():3d} / {len(rows)}", file=fh)
            print(f"   winding-consistent (REQUIRED for the GWN oracle): {wo.sum():3d} / {len(rows)}", file=fh)
        print("\n" + "=" * 92, file=fh)
        print("VERDICT ON THE ORACLE", file=fh)
        for d in ("bench50", "clean81"):
            wo = np.array([r["wind_ok"] for r in res[d]])
            bf = np.array([r["bnd_frac"] for r in res[d]])
            nc = np.array([r["ncomp"] for r in res[d]])
            print(
                f"   {d:<10} winding-consistent {100 * wo.mean():5.1f}%   "
                f"median boundary-edge frac {np.median(bf):.4f}   median components {np.median(nc):.0f}",
                file=fh,
            )
        print("=" * 92, file=fh)
        np.save(
            os.path.join(ROOT, "diagnostics/moonshot/refute_topology.npy"),
            np.array([res], dtype=object),
            allow_pickle=True,
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
