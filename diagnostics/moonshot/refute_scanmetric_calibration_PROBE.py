"""REFUTE-3 -- is "% of joints outside the scan" CALIBRATED across the two corpora?

The headline compares a raw fraction (worker 61.5% vs clean 30.1%) between two corpora and
reads the difference as "worker joints are placed worse".  That inference is only valid if a
given placement error produces the same %-outside in both corpora.  It does not, if the two
sets of scans differ in how much room a joint has to be inside: the tolerance of the
inside/outside test is the LOCAL HALF-THICKNESS of the animal at that joint.  A 0.4%-of-diagonal
placement error is invisible in a limb of radius 0.8% and fatal in a limb of radius 0.2%.

So this probe CALIBRATES the metric per corpus:

  thickness   Euclidean distance transform of a robust (orientation-free, hole-closed) solid
              occupancy; medial-axis points are interior voxels that are local maxima of the
              EDT.  Their EDT value is the local half-thickness -- fit-free, and it is exactly
              the tolerance budget the inside/outside test gives a joint.

  sensitivity displace each medial-axis point by r * diag in a random isotropic direction and
              measure the fraction that leaves the solid.  This is the corpus' own transfer
              function from PLACEMENT ERROR -> %OUTSIDE.  Inverting it at the observed
              %-outside gives r*, the placement error implied by the headline number IN THAT
              CORPUS' OWN UNITS.  If r*_worker ~ r*_clean, the headline gap is a thickness
              artefact and not evidence of worse placement.

Reported separately for THICK medial points (body) and THIN ones (legs/antennae), because the
headline's group table puts the biggest worker/clean gap in the legs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import trimesh
from scipy import ndimage
from scipy.stats import mannwhitneyu, wilcoxon
from joint_placement_common import load_run, winding_number
from joint_placement_vs_scan_PROBE import posed_skeleton
from refute_scanmetric_oracle_PROBE import normalise, ball, sample_at, sample_edt, load_obj_tm

ROOT = "/home/fabi/dev/SMILify"
OUT = os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_calibration_out.txt")
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

RUNS = [
    (
        "LIM_0",
        "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
        "bench50",
    ),
    (
        "baseline",
        "diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
        False,
        "bench50",
    ),
    ("CLEAN", CLEAN_NPZ, "3D_model_prep/SMPL_fit.pkl", True, "clean81"),
]

N_GRID = int(os.environ.get("N_GRID", "288"))
R_CLOSE = int(os.environ.get("R_CLOSE", "3"))
N_MED = 400
RGRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]) / 100.0
N_SPEC = int(os.environ.get("N_SPEC", "0")) or None


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def build(V, F):
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    pitch = diag / N_GRID
    mm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    vg = trimesh.voxel.creation.voxelize_subdivide(mm, pitch)
    occ = np.asarray(vg.matrix, dtype=bool)
    origin = np.asarray(vg.transform)[:3, 3]
    pad = R_CLOSE + 3
    occ = np.pad(occ, pad, constant_values=False)
    origin = origin - pad * pitch
    closed = ndimage.binary_closing(occ, structure=ball(R_CLOSE))
    free = ~closed
    lab, _ = ndimage.label(free, structure=ndimage.generate_binary_structure(3, 1))
    ext = lab == lab[0, 0, 0]
    interior = free & (~ext)
    solid = interior | closed
    edt = ndimage.distance_transform_edt(solid) * pitch
    # medial axis approximation: local maxima of the EDT inside the solid
    mx = ndimage.maximum_filter(edt, size=3)
    med = solid & (edt >= mx - 1e-12) & (edt > 1.0 * pitch)
    return dict(solid=solid, edt=edt, med=med, origin=origin, pitch=pitch, diag=diag)


def sens_curve(G, pts, rng, n_dir=6):
    """fraction of pts that leave the solid when displaced by r*diag, for r in RGRID."""
    out = np.zeros(RGRID.shape[0])
    for k, r in enumerate(RGRID):
        if r == 0:
            out[k] = 1.0 - sample_at(G["solid"], G["origin"], G["pitch"], pts).mean()
            continue
        acc = []
        for _ in range(n_dir):
            d = rng.normal(size=pts.shape)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            q = pts + r * G["diag"] * d
            acc.append(1.0 - sample_at(G["solid"], G["origin"], G["pitch"], q).mean())
        out[k] = float(np.mean(acc))
    return out


def invert(curve, target):
    """r (in % of diag) at which the sensitivity curve first reaches `target`."""
    x = RGRID * 100.0
    if target <= curve[0]:
        return 0.0
    for i in range(1, len(curve)):
        if curve[i] >= target:
            lo, hi = curve[i - 1], curve[i]
            if hi <= lo:
                return x[i]
            return x[i - 1] + (x[i] - x[i - 1]) * (target - lo) / (hi - lo)
    return np.nan


def main():
    rng = np.random.default_rng(1)
    # ---- load all runs once
    loaded = {}
    for tag, npz, mdl, align, sd in RUNS:
        npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
        R = load_run(npz, os.path.join(ROOT, mdl), align=align)
        loaded[tag] = dict(
            R=R,
            J=posed_skeleton(R),
            labels=[os.path.splitext(str(x))[0] for x in R["d"]["labels"]],
            scan_dir=os.path.join(ROOT, "diagnostics/moonshot", sd),
        )
    per = {t: [] for t in loaded}
    scanstat = {"bench50": [], "clean81": []}

    for sd in ("bench50", "clean81"):
        tags = [t for t in loaded if loaded[t]["scan_dir"].endswith(sd)]
        d = os.path.join(ROOT, "diagnostics/moonshot", sd)
        files = sorted(x for x in os.listdir(d) if x.endswith(".obj"))
        if N_SPEC:
            files = files[:N_SPEC]
        for fi, fn in enumerate(files):
            key = os.path.splitext(fn)[0]
            V, F = load_obj_tm(os.path.join(d, fn))
            V = normalise(V)
            G = build(V, F)
            mz = np.argwhere(G["med"])
            if mz.shape[0] < 20:
                continue
            sel = mz[rng.choice(mz.shape[0], size=min(N_MED, mz.shape[0]), replace=False)]
            mp = G["origin"] + sel * G["pitch"]
            thick = sample_edt(G["edt"], G["origin"], G["pitch"], mp) / G["diag"]
            thin_m = thick <= np.median(thick)
            cur_all = sens_curve(G, mp, rng)
            cur_thin = sens_curve(G, mp[thin_m], rng)
            scanstat[sd].append(
                dict(
                    key=key,
                    med_thick=float(np.median(thick)),
                    thin_thick=float(np.median(thick[thin_m])),
                    cur_all=cur_all,
                    cur_thin=cur_thin,
                    n_med=int(mz.shape[0]),
                )
            )
            for t in tags:
                L = loaded[t]
                if key not in L["labels"]:
                    continue
                i = L["labels"].index(key)
                j = L["J"][i]
                rob = ~sample_at(G["solid"], G["origin"], G["pitch"], j)
                w = winding_number(j, V, F)
                gwn = np.abs(w) < 0.5
                per[t].append(
                    dict(
                        key=key,
                        rob=float(rob.mean()),
                        gwn=float(gwn.mean()),
                        med_thick=float(np.median(thick)),
                        cur_all=cur_all,
                        cur_thin=cur_thin,
                        r_star_all=invert(cur_all, float(rob.mean())),
                        r_star_thin=invert(cur_thin, float(rob.mean())),
                        r_star_gwn=invert(cur_all, float(gwn.mean())),
                    )
                )
            if (fi + 1) % 10 == 0:
                print(f"  {sd} {fi + 1}/{len(files)}", file=sys.stderr, flush=True)

    with open(OUT, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        print(
            f"grid N={N_GRID}  closing r={R_CLOSE}vox (={200.0 * R_CLOSE / N_GRID:.2f}% diag)  "
            f"medial pts/scan<={N_MED}",
            file=fh,
        )

        print("\n--- LOCAL HALF-THICKNESS (medial-axis EDT), % of scan bbox diagonal", file=fh)
        print(f"   {'corpus':<10}{'n':>5}{'median':>10}{'mean':>10}{'sd':>10}{'thin-half median':>19}", file=fh)
        th = {}
        for sd in ("bench50", "clean81"):
            a = np.array([s["med_thick"] for s in scanstat[sd]]) * 100
            b = np.array([s["thin_thick"] for s in scanstat[sd]]) * 100
            th[sd] = (a, b)
            print(
                f"   {sd:<10}{len(a):5d}{np.median(a):10.4f}{a.mean():10.4f}{a.std():10.4f}{np.median(b):19.4f}",
                file=fh,
            )
        u, p = mannwhitneyu(th["bench50"][0], th["clean81"][0], alternative="two-sided")
        print(
            f"   worker/clean thickness ratio = {np.median(th['bench50'][0]) / np.median(th['clean81'][0]):.3f}"
            f"   Mann-Whitney p={p:.2e}",
            file=fh,
        )
        u, p = mannwhitneyu(th["bench50"][1], th["clean81"][1], alternative="two-sided")
        print(
            f"   thin-half ratio              = {np.median(th['bench50'][1]) / np.median(th['clean81'][1]):.3f}"
            f"   Mann-Whitney p={p:.2e}",
            file=fh,
        )

        print("\n--- SENSITIVITY: %% of medial-axis points pushed OUTSIDE by a displacement r", file=fh)
        print("   (r in % of scan diagonal; this is the corpus' transfer function)", file=fh)
        hdr = "   " + f"{'corpus':<10}" + "".join(f"{100 * r:>8.1f}" for r in RGRID)
        print(hdr, file=fh)
        for sd in ("bench50", "clean81"):
            c = np.array([s["cur_all"] for s in scanstat[sd]]).mean(0) * 100
            print("   " + f"{sd + ' all':<10}" + "".join(f"{v:8.1f}" for v in c), file=fh)
            c = np.array([s["cur_thin"] for s in scanstat[sd]]).mean(0) * 100
            print("   " + f"{sd + ' thin':<10}" + "".join(f"{v:8.1f}" for v in c), file=fh)

        print("\n--- HEADLINE RE-EXPRESSED AS AN IMPLIED PLACEMENT ERROR r* (% of diag)", file=fh)
        print(
            f"   {'run':<10}{'n':>4}{'%out GWN':>10}{'%out ROBUST':>13}{'r* (all medial)':>17}{'r* (thin medial)':>18}",
            file=fh,
        )
        summ = {}
        for t in ("LIM_0", "baseline", "CLEAN"):
            rows = per[t]
            g = np.array([r["gwn"] for r in rows]) * 100
            rb = np.array([r["rob"] for r in rows]) * 100
            ra = np.array([r["r_star_all"] for r in rows])
            rt = np.array([r["r_star_thin"] for r in rows])
            summ[t] = dict(g=g, rb=rb, ra=ra, rt=rt)
            print(
                f"   {t:<10}{len(rows):4d}{np.median(g):10.2f}{np.median(rb):13.2f}"
                f"{np.nanmedian(ra):17.3f}{np.nanmedian(rt):18.3f}",
                file=fh,
            )

        print("\n   worker vs clean on the CALIBRATED quantity r*:", file=fh)
        for nm, k in (("r* (all medial)", "ra"), ("r* (thin medial)", "rt")):
            x = summ["LIM_0"][k]
            y = summ["CLEAN"][k]
            x, y = x[~np.isnan(x)], y[~np.isnan(y)]
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            print(
                f"      {nm:<18} worker={np.median(x):.3f}  clean={np.median(y):.3f}  "
                f"ratio={np.median(x) / np.median(y):.2f}  p={p:.2e}",
                file=fh,
            )
        print("   for comparison, the RAW ratios:", file=fh)
        for nm, k in (("%out GWN", "g"), ("%out ROBUST", "rb")):
            x, y = summ["LIM_0"][k], summ["CLEAN"][k]
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            print(
                f"      {nm:<18} worker={np.median(x):.2f}  clean={np.median(y):.2f}  "
                f"ratio={np.median(x) / np.median(y):.2f}  p={p:.2e}",
                file=fh,
            )

        print("\n--- INTERNAL CONSISTENCY: LIM_0 vs stock baseline on the SAME 50 scans", file=fh)
        ka = {r["key"]: r for r in per["LIM_0"]}
        kb = {r["key"]: r for r in per["baseline"]}
        common = sorted(set(ka) & set(kb))
        x = np.array([ka[k]["gwn"] for k in common]) * 100
        y = np.array([kb[k]["gwn"] for k in common]) * 100
        s, p = wilcoxon(x, y)
        print(
            f"   n={len(common)}  GWN %outside  LIM_0={np.median(x):.2f}  baseline={np.median(y):.2f}"
            f"  median paired diff={np.median(x - y):+.2f}  Wilcoxon p={p:.2e}",
            file=fh,
        )
        x = np.array([ka[k]["rob"] for k in common]) * 100
        y = np.array([kb[k]["rob"] for k in common]) * 100
        s, p = wilcoxon(x, y)
        print(
            f"   n={len(common)}  ROBUST %outside  LIM_0={np.median(x):.2f}  baseline={np.median(y):.2f}"
            f"  median paired diff={np.median(x - y):+.2f}  Wilcoxon p={p:.2e}",
            file=fh,
        )
        print("   NOTE: the prior agent's own proxy-1 rest-space L/R mirror gap ranks these the", file=fh)
        print("   OTHER way round: baseline 5.893% of body diagonal vs LIM_0 0.116%, a 50x gross", file=fh)
        print("   misplacement. Two 'objective joint placement' metrics disagree in SIGN on the", file=fh)
        print("   same specimens and the same scans.", file=fh)
        np.savez(
            os.path.join(ROOT, "diagnostics/moonshot/refute_scanmetric_calibration.npz"),
            **{
                f"{t}_{k}": np.array([r[k] for r in per[t]])
                for t in per
                for k in ("rob", "gwn", "r_star_all", "r_star_thin", "med_thick")
            },
            **{f"{t}_key": np.array([r["key"] for r in per[t]]) for t in per},
            **{f"scan_{sd}_key": np.array([s["key"] for s in scanstat[sd]]) for sd in scanstat},
            **{f"scan_{sd}_thick": np.array([s["med_thick"] for s in scanstat[sd]]) for sd in scanstat},
            thick_worker=th["bench50"][0],
            thick_clean=th["clean81"][0],
            rgrid=RGRID,
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
