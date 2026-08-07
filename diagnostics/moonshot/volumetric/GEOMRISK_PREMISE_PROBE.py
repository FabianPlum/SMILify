"""GEOMRISK R6 — the proposal's PREMISE, tested directly.

The stated rationale is: "volume/occupancy pins joint PLACEMENT (extent + centroid) where
surface chamfer cannot, especially on smooth blobs". That is an empirical claim about two
objective functions, and it can be settled without an optimiser.

Take an anatomical part of the TEMPLATE (watertight, so both metrics are exact). Displace it
rigidly by delta. Measure how much each objective moves, in units of that objective's OWN
sampling noise -- because an objective that moves a lot but is also noisy carries no more
information than one that moves a little and is quiet. Detectability = |metric(delta) -
metric(0)| / sd(metric). The objective with the higher detectability at a given delta is the
one that actually constrains placement.

Three displacements, chosen to separate the claims:
  T_long   translation along the part's LONG axis   -- "slide along a smooth surface", the
           motion the proposal says chamfer cannot see
  T_short  translation along the part's SHORTEST axis
  R_long   rotation about the part's LONG axis      -- the twist both objectives may be blind to

If chamfer's detectability is >= volume IoU's for T_long, the premise is false as stated, and
the volumetric term is not buying placement information that chamfer lacks.
"""

import json
import os
import pickle
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

N_SURF = 4000  # surface samples per side, per repeat -- a realistic per-iteration budget
N_MC = 20000  # occupancy samples per repeat, over the part's own bbox
N_REP = 12  # repeats, to get each metric's sampling sd


def part_submesh(v, f, keep):
    fm = keep[f].all(1)
    idx = -np.ones(len(v), np.int64)
    sel = np.unique(f[fm])
    idx[sel] = np.arange(len(sel))
    return trimesh.Trimesh(v[sel], idx[f[fm]], process=False)


def chamfer(a, b, rng):
    from scipy.spatial import cKDTree

    pa = trimesh.sample.sample_surface(a, N_SURF, seed=int(rng.integers(1 << 30)))[0]
    pb = trimesh.sample.sample_surface(b, N_SURF, seed=int(rng.integers(1 << 30)))[0]
    return float(cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean())


def viou(a, b, rng, lo, hi):
    x = lo + rng.random((N_MC, 3)) * (hi - lo)
    ia, ib = a.contains(x), b.contains(x)
    return float((ia & ib).sum() / max((ia | ib).sum(), 1))


def main():

    with open(os.path.join(REPO, __import__("config").SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [str(s) for s in dd["J_names"]]
    v = np.asarray(dd["v_template"], np.float64)
    f = np.asarray(dd["f"]).astype(np.int64)
    dom = np.asarray(dd["weights"]).argmax(1)
    c0 = v.mean(0)
    sc = np.abs(v - c0).max()
    v = (v - c0) / sc
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))

    # nearest-neighbour vertex spacing sets the scale that "3.8x local vertex spacing" refers to
    from scipy.spatial import cKDTree

    spacing = float(np.median(cKDTree(v).query(v, k=2)[0][:, 1]))
    print(f"template diag {diag:.4f}; median vertex spacing {spacing:.5f} ({100 * spacing / diag:.3f}% of diag)\n")

    parts = {
        "gaster": [j for j in range(len(jn)) if jn[j].startswith("b_a_")],
        "head": [j for j in range(len(jn)) if jn[j] == "b_h"],
        "mesosoma": [j for j in range(len(jn)) if jn[j] == "b_t"],
        "femur_R1": [j for j in range(len(jn)) if jn[j] == "l_1_fe_r"],
    }
    results = {}
    for pname, js in parts.items():
        keep = np.isin(dom, js)
        if keep.sum() < 50:
            print(f"{pname}: too few vertices, skipped")
            continue
        sub = part_submesh(v, f, keep)
        if not sub.is_watertight:
            sub.fill_holes()
        pts = np.asarray(sub.vertices)
        c = pts.mean(0)
        _, S, V = np.linalg.svd(pts - c, full_matrices=False)
        L = 2 * np.percentile(np.abs((pts - c) @ V[0]), 98)  # long-axis extent
        print("=" * 92)
        print(
            f"{pname}:  {len(pts)} verts, watertight={sub.is_watertight}, "
            f"long extent {L:.4f} ({100 * L / diag:.1f}% of diag), axis sd ratio {S[0] / S[2]:.2f}"
        )

        lo = sub.bounds[0] - 0.15 * L
        hi = sub.bounds[1] + 0.15 * L
        # Baselines. NOTE: viou(sub, sub) is identically 1 with zero variance because both
        # sides share the query points, so it is NOT a usable noise floor. Each metric's noise
        # is therefore estimated AT THE OPERATING POINT (at each delta), across repeats.
        base_c, base_i = [], []
        for r in range(N_REP):
            rng = np.random.default_rng(1000 + r)
            base_c.append(chamfer(sub, sub, rng))
            base_i.append(viou(sub, sub, rng, lo, hi))
        m_c, m_i = float(np.mean(base_c)), float(np.mean(base_i))
        print(f"  zero-displacement value: chamfer {m_c:.5f} (pure resampling floor)   volumeIoU {m_i:.5f}")

        rows = []
        for mode in ["T_long", "T_short", "R_long"]:
            print(
                f"  {mode:<8} {'delta':>11}{'chamfer':>11}{'sd_C':>10}{'detC':>8}{'volIoU':>9}{'sd_V':>10}{'detV':>8}{'winner':>9}"
            )
            for k, mult in enumerate([1.0, 2.0, 4.0, 8.0]):
                if mode.startswith("T"):
                    d = spacing * mult
                    ax = V[0] if mode == "T_long" else V[2]
                    T = trimesh.transformations.translation_matrix(ax * d)
                    lab = f"{d:.5f}"
                else:
                    th = np.radians(2.0 * mult)
                    T = trimesh.transformations.rotation_matrix(th, V[0], c)
                    lab = f"{np.degrees(th):.1f}deg"
                mv = sub.copy()
                mv.apply_transform(T)
                cs, iv = [], []
                for r in range(N_REP):
                    rng = np.random.default_rng(2000 + r)
                    cs.append(chamfer(sub, mv, rng))
                    iv.append(viou(sub, mv, rng, lo, hi))
                sd_c = max(float(np.std(cs)), 1e-9)
                sd_v = max(float(np.std(iv)), 1e-9)
                dc = abs(float(np.mean(cs)) - m_c) / sd_c
                di = abs(float(np.mean(iv)) - m_i) / sd_v
                win = "VOLUME" if di > dc else "chamfer"
                print(
                    f"  {'':8} {lab:>11}{np.mean(cs):11.5f}{sd_c:10.5f}{dc:8.1f}"
                    f"{np.mean(iv):9.5f}{sd_v:10.5f}{di:8.1f}{win:>9}"
                )
                rows.append(
                    dict(
                        part=pname,
                        mode=mode,
                        delta=lab,
                        chamfer=float(np.mean(cs)),
                        detect_chamfer=float(dc),
                        viou=float(np.mean(iv)),
                        detect_viou=float(di),
                    )
                )

        # basin of attraction: the displacement at which volume IoU reaches exactly zero, i.e.
        # the point beyond which the volumetric objective has NO gradient at all
        print(f"  {'BASIN':<8} displacement along the SHORT axis at which volume IoU hits 0:")
        rng = np.random.default_rng(77)
        zero_at = None
        for mult in np.arange(0.5, 20.01, 0.5):
            mv = sub.copy()
            mv.apply_transform(trimesh.transformations.translation_matrix(V[2] * spacing * mult))
            if viou(sub, mv, rng, lo - 0.3 * L, hi + 0.3 * L) <= 0.0:
                zero_at = spacing * mult
                break
        if zero_at is None:
            print("           > 20 vertex spacings (not reached)")
        else:
            print(
                f"           {zero_at:.5f} = {zero_at / spacing:.1f} vertex spacings = "
                f"{100 * zero_at / diag:.3f}% of body diagonal"
            )
        results_basin = float(zero_at) if zero_at else None
        results[pname] = dict(
            rows=rows, long_extent=float(L), axis_ratio=float(S[0] / S[2]), viou_zero_at=results_basin
        )
        print()

    with open(os.path.join(HERE, "out", "geomrisk_premise.json"), "w") as fh:
        json.dump(dict(vertex_spacing=spacing, diag=diag, parts=results), fh, indent=1)
    print("wrote out/geomrisk_premise.json")


if __name__ == "__main__":
    main()
