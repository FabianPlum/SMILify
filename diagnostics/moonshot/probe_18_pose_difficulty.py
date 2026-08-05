"""PROBE 18 — what actually predicts whether a worker scan can be fitted?

WHY NOT MESH QUALITY
The preprocessing A/B (report section 6.1) settled that: cleaning the meshes to reference
quality on 6 of 7 axes changed nothing, and slightly worsened the target-independent
correspondence metrics. So a fittability gate built on mesh-quality features would be
selecting on something that demonstrably does not matter.

THE HYPOTHESIS THIS TESTS
The failure mode that keeps recurring is correspondence ambiguity between limbs, and the
physical cause is SELF-CONTACT: ethanol-preserved specimens contract, so legs fold against
the body and against each other. Contact is exactly the configuration that
  * collapsed the geodesic branch decomposition (probe 14: 3 specimens reduced to ONE branch),
  * makes two different limbs locally indistinguishable to a chamfer,
  * cannot be undone by preprocessing, because the surfaces really are touching.

SELF-CONTACT, measured from the scan alone
For each area-sampled surface point p with outward normal n_p, look at surface points q
within a small radius. A pair is counted as CONTACT when the two surfaces are close AND
facing each other (n_p . n_q < -0.5) AND q lies on the inward side of p (n_p . (q-p) < 0).
Two patches that merely belong to the same smooth sheet have nearly parallel normals and are
excluded; two patches pressed together have opposed normals and are counted. This needs no
geodesic computation, which matters because a 20k-point all-pairs geodesic is not affordable
over 757 specimens.

Also measured, as pose-difficulty proxies that need no template:
  hull_fill       mesh volume / convex-hull volume. A splayed specimen is mostly empty hull;
                  a specimen with limbs tucked under the body fills its hull.
  radial spread   how far surface area lies from the principal axis, and its dispersion --
                  a proxy for how extended the limbs are.
  contact_area    fraction of surface area participating in contact.

THE TEST
These features, plus the mesh-quality features, are correlated against per-specimen fit
quality on the 50 bench specimens, using the TARGET-INDEPENDENT metrics (edge_logratio,
deform_mag) as the primary outcome -- because section 3 established that surface-proximity
metrics are gamed by shrink-wrapping and would rank a torn mesh as a good fit.

If self-contact predicts fit quality and mesh quality does not, the gate is built on contact.
If neither predicts, then per-specimen fittability is not knowable in advance from geometry,
which is itself a result worth having before anyone builds a triage pipeline on the idea.
"""

import argparse
import glob
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
N_SAMP = 20000


def pose_features(path, rng, contact_r=0.02):
    try:
        m = trimesh.load(path, process=False, force="mesh")
    except Exception:
        return None
    V = np.asarray(m.vertices, dtype=np.float64)
    if len(V) < 100 or len(m.faces) < 100:
        return None
    c = V.mean(0)
    s = max(np.abs(V - c).max(), 1e-12)
    m = trimesh.Trimesh(vertices=(V - c) / s, faces=np.asarray(m.faces), process=False)

    d = dict(name=os.path.basename(path))
    try:
        pts, fid = trimesh.sample.sample_surface(m, N_SAMP, seed=int(rng.integers(1 << 30)))
        pts = np.asarray(pts)
        nrm = m.face_normals[fid]
    except Exception:
        return None

    # ---- self-contact: close AND facing each other
    tree = cKDTree(pts)
    pairs = tree.query_pairs(contact_r, output_type="ndarray")
    if len(pairs):
        pi, pj = pairs[:, 0], pairs[:, 1]
        dv = pts[pj] - pts[pi]
        dn = np.linalg.norm(dv, axis=1) + 1e-12
        facing = (nrm[pi] * nrm[pj]).sum(1) < -0.5
        inward = (nrm[pi] * (dv / dn[:, None])).sum(1) < 0.0
        hit = facing & inward
        touched = np.unique(np.concatenate([pi[hit], pj[hit]])) if hit.any() else np.array([], int)
        d["contact_frac"] = float(len(touched) / len(pts))
        d["contact_pairs_per_pt"] = float(hit.sum() / len(pts))
    else:
        d["contact_frac"] = 0.0
        d["contact_pairs_per_pt"] = 0.0

    # ---- hull fill: tucked-in specimens fill their convex hull
    try:
        hv = float(m.convex_hull.volume)
        d["hull_fill"] = float(abs(m.volume) / max(hv, 1e-12)) if m.is_volume else np.nan
        d["area_over_hullarea"] = float(m.area / max(float(m.convex_hull.area), 1e-12))
    except Exception:
        d["hull_fill"] = d["area_over_hullarea"] = np.nan

    # ---- radial spread about the principal axis
    q = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(q[rng.choice(len(q), min(6000, len(q)), replace=False)], full_matrices=False)
    ax = Vt[0] / np.linalg.norm(Vt[0])
    perp = q - (q @ ax)[:, None] * ax[None, :]
    r = np.linalg.norm(perp, axis=1)
    med = np.median(r)
    d["radial_med"] = float(med)
    d["radial_p95_over_med"] = float(np.percentile(r, 95) / max(med, 1e-9))
    d["limb_area_frac"] = float((r > 2.0 * med).mean())
    d["extent_along_axis"] = float((q @ ax).max() - (q @ ax).min())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=[
        os.path.join(HERE, "bench50"),
        "/media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN",
    ])
    ap.add_argument("--tags", nargs="+", default=["bench50", "clean"])
    ap.add_argument("--contact_r", type=float, default=0.02)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import pandas as pd

    for tag, d_ in zip(args.tags, args.dirs):
        fs = sorted(glob.glob(os.path.join(d_, "*.obj")))
        print(f"[probe18] {tag}: {len(fs)} meshes", flush=True)
        rng = np.random.default_rng(0)
        rows = [r for r in (pose_features(f, rng, args.contact_r) for f in fs) if r]
        df = pd.DataFrame(rows)
        p = os.path.join(OUT, f"probe18_pose_{tag}.csv")
        df.to_csv(p, index=False)
        print(f"[probe18] wrote {p}")
        for k in ["contact_frac", "contact_pairs_per_pt", "hull_fill", "limb_area_frac", "radial_p95_over_med"]:
            if k in df:
                v = df[k].values.astype(float)
                print(f"    {k:<22} median {np.nanmedian(v):>8.4f}  p90 {np.nanpercentile(v, 90):>8.4f}")


if __name__ == "__main__":
    main()
