"""PROBE 4: is the non-flatness of the scans caused by thin appendages (legs/antennae splayed
out of plane), i.e. an ARTICULATION/POSE gap between template rest pose and specimens?

Robust flatness = flatness computed after discarding the 10% of surface area furthest from the
mid-plane. If robust flatness collapses to ~template flatness, the bulk body IS flat and the
non-flatness comes from appendages -> pose gap, not a rotation problem and not a body-shape problem.
Also renders orthographic point projections for visual confirmation.
"""

import glob
import pickle
import numpy as np
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCANS = "/media/fabi/Data/SMILify_DATASETS_BACKUP/custom_processing/antscan_proofread_castes/half_workers/*.obj"


def sample(V, F, n=60_000, seed=0):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(F), n, p=ar / ar.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    fl = (u + v) > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    return a[i] + u * (b[i] - a[i]) + v * (c[i] - a[i])


def canon(P):
    P = P - P.mean(0)
    _, R = np.linalg.eigh((P.T @ P) / len(P))
    P = P @ R[:, ::-1]
    return P / (P[:, 0].max() - P[:, 0].min())


def flatness(P, q=100):
    keep = np.abs(P[:, 2]) <= np.percentile(np.abs(P[:, 2]), q)
    Q = P[keep]
    e = Q.max(0) - Q.min(0)
    return e[2] / e[0]


with open("/home/fabi/dev/SMILify/3D_model_prep/SMIL_OmniAnt.pkl", "rb") as fh:
    dd = pickle.load(fh, encoding="latin1")
Pt = canon(sample(np.asarray(dd["v_template"], float), np.asarray(dd["f"], np.int64)))
print(f"{'name':<52s} {'flat100':>8s} {'flat90':>7s} {'flat75':>7s}")
print(f"{'TEMPLATE':<52s} {flatness(Pt):8.3f} {flatness(Pt, 90):7.3f} {flatness(Pt, 75):7.3f}")

files = sorted(glob.glob(SCANS))
sel = files[:: len(files) // 5][:5]
sets = [("TEMPLATE", Pt)]
f100, f90, f75 = [], [], []
for p in files[:: max(1, len(files) // 40)][:40]:
    m = trimesh.load(p, process=False, force="mesh")
    P = canon(sample(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)))
    f100.append(flatness(P))
    f90.append(flatness(P, 90))
    f75.append(flatness(P, 75))
    if p in sel:
        sets.append((p.split("/")[-1][:28], P))
        print(f"{p.split('/')[-1][:52]:<52s} {f100[-1]:8.3f} {f90[-1]:7.3f} {f75[-1]:7.3f}")
print(
    f"\n40-scan median: flat100={np.median(f100):.3f}  flat90={np.median(f90):.3f}  "
    f"flat75={np.median(f75):.3f}   | TEMPLATE {flatness(Pt):.3f} / {flatness(Pt, 90):.3f} / {flatness(Pt, 75):.3f}"
)

fig, axes = plt.subplots(len(sets), 3, figsize=(11, 2.6 * len(sets)))
for r, (nm, P) in enumerate(sets):
    for c, (i, j, lab) in enumerate([(0, 1, "PC1-PC2 (top)"), (0, 2, "PC1-PC3 (side)"), (1, 2, "PC2-PC3 (front)")]):
        ax = axes[r, c]
        ax.scatter(P[::12, i], P[::12, j], s=0.12, c="k", lw=0)
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(lab, fontsize=9)
        if c == 0:
            ax.set_ylabel(nm, fontsize=7)
plt.tight_layout()
plt.savefig("/home/fabi/dev/SMILify/diagnostics/pose_gap_PROBE.png", dpi=130)
print("\nwrote /home/fabi/dev/SMILify/diagnostics/pose_gap_PROBE.png")
