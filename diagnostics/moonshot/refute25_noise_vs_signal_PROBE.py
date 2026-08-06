"""REFUTE PROBE (part 2) — is the worker free-form field NOISE, or reproducible
specimen-specific SIGNAL that simply is not low-rank across specimens?

The claim under test says the worker deform_verts field is "pure per-specimen noise".
"Noise" is a testable word: noise does not reproduce.  The bench50 worker corpus was fitted
three times from three independent initialisations (M7_handoff_midline, _s1, _s2, seeded off
M1_sym / M1_sym_s1 / M1_sym_s2) and by several different arms.  If specimen i's field is the
same field every time, it is a deterministic function of that specimen -- signal, not noise.

Also measured here:
  * pose spread of each corpus (joint_rot), i.e. the posed-vs-contracted domain difference
  * per-specimen dose-response: does a specimen's LOO residual track how far its SCAN is
    from the clean corpus's regime (scan-only descriptors, no fit involved)?
"""

import csv
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, REPO)

LEGACY = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)


def load_model(p):
    with open(p, "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        return u.load()


def field(npz):
    d = np.load(npz, allow_pickle=True)
    return np.asarray(d["deform_verts"], dtype=np.float64), d


def cosmat(A, B):
    """row-wise cosine similarity between two (N, D) stacks."""
    a = A / np.linalg.norm(A, axis=1, keepdims=True)
    b = B / np.linalg.norm(B, axis=1, keepdims=True)
    return a @ b.T


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}

    print("=" * 96)
    print("SECTION A — REPRODUCIBILITY of the worker free-form field across INDEPENDENT runs")
    print("            (same 50 scans, different initialisation / different arm).")
    print("            diag = same specimen, run A vs run B.  off-diag = different specimens,")
    print("            which is the null a genuinely per-specimen-random field would sit at.")
    print("=" * 96)
    pairs = [
        (
            "M7 vs M7_s1 (seed replicate)",
            "runs/M7_handoff_midline/Stage_3_deform_fine.npz",
            "runs/M7_handoff_midline_s1/Stage_3_deform_fine.npz",
        ),
        (
            "M7 vs M7_s2 (seed replicate)",
            "runs/M7_handoff_midline/Stage_3_deform_fine.npz",
            "runs/M7_handoff_midline_s2/Stage_3_deform_fine.npz",
        ),
        (
            "M7_s1 vs M7_s2 (seed replicate)",
            "runs/M7_handoff_midline_s1/Stage_3_deform_fine.npz",
            "runs/M7_handoff_midline_s2/Stage_3_deform_fine.npz",
        ),
        (
            "LIM_0 vs LIM_1x (different arm)",
            "runs/LIM_0/Stage_3_deform_fine.npz",
            "runs/LIM_1x/Stage_3_deform_fine.npz",
        ),
        (
            "C0_control_s1 vs _s2 (seed)",
            "runs/C0_control_s1/Stage_3_deform_fine.npz",
            "runs/C0_control_s2/Stage_3_deform_fine.npz",
        ),
    ]
    print(f"{'pair':<36}{'same-specimen r':>17}{'cross-specimen r':>19}{'gap':>8}")
    for lab, a, b in pairs:
        pa, pb = os.path.join(HERE, a), os.path.join(HERE, b)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{lab:<36}  [missing]")
            continue
        A, da = field(pa)
        B, db = field(pb)
        if list(da["labels"]) != list(db["labels"]) or A.shape != B.shape:
            print(f"{lab:<36}  [label/shape mismatch]")
            continue
        N = A.shape[0]
        Af = (A - A.mean(0)).reshape(N, -1)
        Bf = (B - B.mean(0)).reshape(N, -1)
        C = cosmat(Af, Bf)
        same = float(np.mean(np.diag(C)))
        off = float((C.sum() - np.trace(C)) / (N * N - N))
        print(f"{lab:<36}{same:>17.4f}{off:>19.4f}{same - off:>8.4f}")
        res.setdefault("reproducibility", {})[lab] = dict(same=same, cross=off)

    print("\n  Interpretation guide: a field that is per-specimen NOISE would give same-specimen r")
    print("  close to the cross-specimen null.  A field that is a deterministic function of the")
    print("  specimen gives same-specimen r near 1 regardless of how it PCAs across specimens.")

    print("\n" + "=" * 96)
    print("SECTION B — POSE spread: the posed/extended vs contracted/folded domain difference")
    print("=" * 96)
    runs = [
        ("LIM_0 (worker)", "runs/LIM_0/Stage_3_deform_fine.npz"),
        ("M7 (worker)", "runs/M7_handoff_midline/Stage_3_deform_fine.npz"),
        ("baseline (worker)", "runs/baseline/Stage_3_deform_fine.npz"),
        ("CLEAN81_M7 (clean)", "runs/CLEAN_M7/Stage_3_deform_fine.npz"),
        ("ALL_ANTS_CLEAN (clean)", LEGACY),
    ]
    print(
        f"{'corpus':<26}{'mean|joint angle| deg':>23}{'sd across specimens deg':>26}{'mean dist to mean pose deg':>28}"
    )
    for lab, p in runs:
        p = p if os.path.isabs(p) else os.path.join(HERE, p)
        if not os.path.exists(p):
            print(f"{lab:<26} [missing]")
            continue
        d = np.load(p, allow_pickle=True)
        jr = np.asarray(d["joint_rot"], dtype=np.float64)  # (N, 54, 3) axis-angle
        ang = np.linalg.norm(jr, axis=-1)  # per-joint rotation magnitude
        mu = jr.mean(0)
        dev = np.linalg.norm(jr - mu, axis=-1)  # crude but consistent
        print(
            f"{lab:<26}{np.degrees(ang.mean()):>23.2f}{np.degrees(jr.std(0)).mean():>26.2f}"
            f"{np.degrees(dev.mean()):>28.2f}"
        )
        res.setdefault("pose", {})[lab] = dict(
            mean_angle_deg=float(np.degrees(ang.mean())),
            sd_deg=float(np.degrees(jr.std(0)).mean()),
            dist_to_mean_deg=float(np.degrees(dev.mean())),
        )

    print("\n" + "=" * 96)
    print("SECTION C — SCAN-ONLY descriptors (no fit involved): how far apart are the two")
    print("            populations of INPUT meshes?  Columns from probe17 feature tables.")
    print("=" * 96)

    def readcsv(p):
        with open(p) as fh:
            return list(csv.DictReader(fh))

    fc = readcsv(os.path.join(OUT, "probe17_features_clean.csv"))
    fw = readcsv(os.path.join(OUT, "probe17_features_worker.csv"))
    bench = set(os.listdir(os.path.join(HERE, "bench50")))
    fw50 = [r for r in fw if r["name"] in bench]
    cols = [
        "limb_area_frac",
        "radial_p95_over_med",
        "elongation",
        "flatness",
        "area_over_vol23",
        "clump_frac",
        "n_components",
        "frac_boundary_edges",
    ]
    print(f"{'feature':<24}{'clean81 median':>16}{'worker50 median':>17}{'clean pct-ile in worker dist':>30}")
    for c in cols:
        try:
            vc = np.array([float(r[c]) for r in fc if r[c] not in ("", "nan")])
            vw = np.array([float(r[c]) for r in fw50 if r[c] not in ("", "nan")])
        except Exception:
            continue
        if len(vc) == 0 or len(vw) == 0:
            continue
        pct = 100.0 * (vw < np.median(vc)).mean()
        print(f"{c:<24}{np.median(vc):>16.4f}{np.median(vw):>17.4f}{pct:>29.1f}%")
        res.setdefault("scan_features", {})[c] = dict(
            clean_median=float(np.median(vc)),
            worker_median=float(np.median(vw)),
            clean_median_pctile_in_worker=float(pct),
        )
    print("  (n clean = %d, n worker50 = %d)" % (len(fc), len(fw50)))

    print("\n" + "=" * 96)
    print("SECTION D — DOSE-RESPONSE inside the worker corpus: does a specimen's LOO residual")
    print("            track a SCAN-ONLY descriptor of how far it is from the clean regime?")
    print("=" * 96)
    A, d = field(os.path.join(HERE, "runs/LIM_0/Stage_3_deform_fine.npz"))
    lab = list(d["labels"])
    N = A.shape[0]
    Af = (A - A.mean(0)).reshape(N, -1)
    r10 = np.zeros(N)
    sp = np.zeros(N)
    for i in range(N):
        keep = np.ones(N, bool)
        keep[i] = False
        Y = Af[keep]
        mu = Y.mean(0)
        _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        r = Af[i] - mu
        B = Vt[:10]
        r10[i] = np.linalg.norm(r - (r @ B.T) @ B)
        sp[i] = np.linalg.norm(r)
    ratio = r10 / sp
    feat = {r["name"]: r for r in fw}
    for c in ["limb_area_frac", "radial_p95_over_med", "clump_frac", "elongation"]:
        v, y, y2 = [], [], []
        for i, nm in enumerate(lab):
            if nm in feat and feat[nm][c] not in ("", "nan"):
                v.append(float(feat[nm][c]))
                y.append(ratio[i])
                y2.append(np.linalg.norm(A[i], axis=-1).mean())
        if len(v) < 10:
            continue
        v, y, y2 = np.array(v), np.array(y), np.array(y2)
        print(
            f"  {c:<24} corr(scan feature, per-specimen LOO ratio) = {np.corrcoef(v, y)[0, 1]:+.3f}"
            f"   corr(feature, |deform|) = {np.corrcoef(v, y2)[0, 1]:+.3f}   n={len(v)}"
        )
        res.setdefault("dose", {})[c] = dict(
            corr_loo=float(np.corrcoef(v, y)[0, 1]), corr_deform=float(np.corrcoef(v, y2)[0, 1]), n=len(v)
        )

    json.dump(res, open(os.path.join(OUT, "refute25_noise_vs_signal.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/refute25_noise_vs_signal.json")


if __name__ == "__main__":
    main()
