"""PROBE 20 -- can fitted JOINT PLACEMENT be scored against something OBJECTIVE?

Three proxies, none of which needs an annotated joint ground truth:

  P1  BILATERAL ASYMMETRY of the fitted rest-space skeleton.
      The template symmetry plane is y = 0 (fitter_3d/trainer_moonshot.py:139) and the
      mirror map on joints is the _l/_r name pair. A correctly placed skeleton on a roughly
      symmetric specimen must be mirror-symmetric IN REST SPACE (pose is factored out by
      evaluating with identity rotations). Decomposed into the three channels that can move
      a joint: betas (via J_regressor . v_shaped), log_beta_scales, betas_trans.

  P2  SEGMENT LENGTHS inside a leg (coxa:trochanter:femur:tibia:tarsus).
      P2a left/right length asymmetry -- taxon-free, trustworthy.
      P2b cross-specimen spread of within-leg proportions -- CONFOUNDED by real taxonomic
          variation (both corpora span many ant genera); reported with that caveat.

  P3  JOINT-INSIDE-MESH. A skeletal joint must lie inside its own skin. Tested with the
      generalized winding number against the fitted, posed mesh stored in the npz. This is
      the only proxy that assumes nothing about biology at all.

      NOTE (read off fitter_3d/trainer.py:240): deform_verts is added AFTER skinning, so the
      surface can be dragged onto the scan while the skeleton stays where betas /
      log_beta_scales / betas_trans put it. Joints leaving the skin is therefore a
      mechanically possible failure of this pipeline, not an impossible one.

All lengths are reported as a percentage of that specimen's REST-SPACE BODY DIAGONAL
(bbox diagonal of v_shaped) so worker and clean corpora are comparable despite different
templates and betas counts.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from joint_placement_common import (
    load_run,
    global_rigid,
    lr_pairs,
    MIRROR,
    joint_group,
    leg_chain_indices,
    winding_number,
    joint_regress,
)

ROOT = "/home/fabi/dev/SMILify"
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

RUNS = [
    (
        "LIM_0        (worker, 50, w_limit=0)",
        "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
    ),
    (
        "LIM_1x       (worker, 50, limits on)",
        "diagnostics/moonshot/runs/LIM_1x/Stage_3_deform_fine.npz",
        "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
    ),
    (
        "BPX_noprior  (worker, 50)",
        "diagnostics/moonshot/runs/BPX_noprior/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
        False,
    ),
    (
        "M7_midline   (worker, 50)",
        "diagnostics/moonshot/runs/M7_handoff_midline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
        False,
    ),
    (
        "baseline     (worker, 50, pose frozen)",
        "diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
        False,
    ),
    ("ALL_ANTS_CLEAN (81)", CLEAN_NPZ, "3D_model_prep/SMPL_fit.pkl", True),
]

DO_P3 = os.environ.get("SKIP_P3", "0") != "1"


def pct(x):
    return 100.0 * x


def q(a, p):
    return float(np.percentile(np.asarray(a).ravel(), p))


def rest_skeleton(R, use_scale, use_trans):
    """Rest-space (identity rotation) shaped skeleton."""
    n, nJ = R["n"], R["nJ"]
    Rs = np.tile(np.eye(3), (n, nJ, 1, 1))
    ls = R["logscale"] if use_scale else None
    bt = R["betas_trans"] if (use_trans and R["betas_trans"] is not None) else None
    J, _ = global_rigid(Rs, R["J"], R["parents"], ls, bt)
    return J


def posed_skeleton(R):
    J, _ = global_rigid(R["Rs"], R["J"], R["parents"], R["logscale"], R["betas_trans"])
    return J + R["trans"][:, None, :]


def body_scale(R):
    """Rest-space bbox diagonal of the shaped mesh, per specimen. (n,)"""
    v = R["v_shaped"]
    return np.linalg.norm(v.max(1) - v.min(1), axis=1)


# ----------------------------------------------------------------------------------------
def p1_asymmetry(R, pairs, tag, fh):
    L = body_scale(R)  # (n,)
    names = R["J_names"]
    groups = np.array([joint_group(names[i]) for i in pairs[:, 0]])

    variants = [
        ("betas only        ", True, False, False),
        ("+ log_beta_scales ", True, True, False),
        ("+ betas_trans     ", True, False, True),
        ("FULL (as fitted)  ", True, True, True),
    ]
    res = {}
    for vname, _, us, ut in variants:
        if ut and R["betas_trans"] is None:
            res[vname] = None
            continue
        Jr = rest_skeleton(R, us, ut)
        mirrored = Jr[:, pairs[:, 1], :] * MIRROR
        dist = np.linalg.norm(Jr[:, pairs[:, 0], :] - mirrored, axis=2)  # (n,P)
        rel = pct(dist / L[:, None])
        res[vname] = rel

    print("  P1 bilateral asymmetry of the REST-SPACE skeleton  [% of rest body diagonal]", file=fh)
    print(f"     {'channel':<20}{'median':>9}{'mean':>9}{'p90':>9}{'max':>9}", file=fh)
    for vname, _, _, _ in variants:
        r = res[vname]
        if r is None:
            print(f"     {vname:<20}{'n/a (no betas_trans in this fit)':>36}", file=fh)
            continue
        print(f"     {vname:<20}{q(r, 50):9.3f}{r.mean():9.3f}{q(r, 90):9.3f}{r.max():9.3f}", file=fh)

    full = res["FULL (as fitted)  "] if res["FULL (as fitted)  "] is not None else res["+ log_beta_scales "]
    print("     by group (FULL):", file=fh)
    for g in sorted(set(groups)):
        m = groups == g
        print(
            f"       {g:<14} n_pairs={m.sum():3d}  median={q(full[:, m], 50):7.3f}  "
            f"mean={full[:, m].mean():7.3f}  p90={q(full[:, m], 90):7.3f}",
            file=fh,
        )
    return full, res


# ----------------------------------------------------------------------------------------
def p2_segments(R, tag, fh):
    names = R["J_names"]
    chains = leg_chain_indices(names)
    Jr = rest_skeleton(R, True, True)
    L = body_scale(R)
    _n = R["n"]

    # seg k = chain[k] -> chain[k+1], k = 0..4  (coxa, trochanter, femur, tibia, tarsus)
    seg_names = ["coxa", "trochanter", "femur", "tibia", "tarsus"]
    lens = {}
    for key, ch in chains.items():
        v = Jr[:, ch[1:], :] - Jr[:, ch[:-1], :]
        lens[key] = np.linalg.norm(v, axis=2)  # (n,5)

    legs = sorted({k[0] for k in chains})

    # --- P2a left/right asymmetry of segment length (taxon-free) ---
    print("  P2a left/right SEGMENT-LENGTH asymmetry  |L-R| / mean(L,R)  [%]", file=fh)
    print(f"     {'segment':<12}" + "".join(f"{'leg' + l:>10}" for l in legs) + f"{'all':>10}", file=fh)
    allrows = []
    for k, sn in enumerate(seg_names):
        row = []
        for lg in legs:
            if (lg, "l") in lens and (lg, "r") in lens:
                a, b = lens[(lg, "l")][:, k], lens[(lg, "r")][:, k]
                row.append(pct(np.abs(a - b) / (0.5 * (a + b))))
            else:
                row.append(None)
        allrows.append(row)
        cells = "".join(f"{np.median(r):10.2f}" if r is not None else f"{'-':>10}" for r in row)
        cat = np.concatenate([r for r in row if r is not None])
        print(f"     {sn:<12}{cells}{np.median(cat):10.2f}", file=fh)
    flat = np.concatenate([r for row in allrows for r in row if r is not None])
    print(
        f"     OVERALL  median={np.median(flat):.2f}%  mean={flat.mean():.2f}%  "
        f"p90={q(flat, 90):.2f}%  max={flat.max():.2f}%",
        file=fh,
    )

    # --- P2b within-leg proportions across specimens (TAXON-CONFOUNDED) ---
    print("  P2b within-leg PROPORTION spread across specimens  (seg / total leg length)", file=fh)
    print("      *** confounded: both corpora span many ant genera ***", file=fh)
    print(f"     {'segment':<12}{'mean prop':>11}{'sd':>9}{'CV %':>9}{'robCV %':>10}", file=fh)
    rob_all, cv_all = [], []
    for k, sn in enumerate(seg_names):
        vals = []
        for lg in legs:
            for side in ("l", "r"):
                if (lg, side) in lens:
                    tot = lens[(lg, side)].sum(axis=1)
                    vals.append(lens[(lg, side)][:, k] / tot)
        v = np.concatenate(vals)  # (n * n_legs*2,)
        cv = 100 * v.std() / v.mean()
        med = np.median(v)
        rcv = 100 * (q(v, 75) - q(v, 25)) / (1.349 * med) if med > 0 else np.nan
        cv_all.append(cv)
        rob_all.append(rcv)
        print(f"     {sn:<12}{v.mean():11.4f}{v.std():9.4f}{cv:9.2f}{rcv:10.2f}", file=fh)
    print(f"     MEAN over segments: CV={np.mean(cv_all):.2f}%   robustCV={np.nanmean(rob_all):.2f}%", file=fh)

    # --- degenerate segments: absurdly short / long relative to body ---
    allseg = np.stack([lens[k] for k in sorted(lens)], axis=1)  # (n, nlegs, 5)
    rel = allseg / L[:, None, None]
    print(
        f"     segment length as % of body diagonal: median={pct(np.median(rel)):.2f} "
        f"min={pct(rel.min()):.3f} max={pct(rel.max()):.2f}   "
        f"frac < 0.5%% of body = {pct((rel < 0.005).mean()):.2f}%",
        file=fh,
    )
    return flat


# ----------------------------------------------------------------------------------------
def p3_inside(R, tag, fh, max_specimens=None):
    d = R["d"]
    V = d["verts"].astype(np.float64)
    F = np.asarray(d["faces"][0]).astype(int)
    Jp = posed_skeleton(R)
    names = R["J_names"]
    n = R["n"] if max_specimens is None else min(R["n"], max_specimens)

    BL = np.linalg.norm(V.max(1) - V.min(1), axis=1)
    ws = np.zeros((n, R["nJ"]))
    dmin = np.zeros((n, R["nJ"]))
    for i in range(n):
        ws[i] = winding_number(Jp[i], V[i], F)
        dd_ = np.linalg.norm(V[i][None, :, :] - Jp[i][:, None, :], axis=2)
        dmin[i] = dd_.min(axis=1)

    outside = np.abs(ws) < 0.5
    groups = np.array([joint_group(nm) for nm in names])
    print("  P3 joints OUTSIDE the fitted mesh (generalized winding number, |w|<0.5)", file=fh)
    print(
        f"     overall: {pct(outside.mean()):.1f}% of {n}x{R['nJ']} joints "
        f"({outside.sum()} of {outside.size});  per specimen mean={outside.sum(1).mean():.1f} joints",
        file=fh,
    )
    print(f"     {'group':<14}{'n_joints':>9}{'% outside':>11}{'med gap %BL':>13}{'p90 gap %BL':>13}", file=fh)
    for g in sorted(set(groups)):
        m = groups == g
        o = outside[:, m]
        gap = (dmin[:, m] / BL[:n, None])[o]
        med = pct(np.median(gap)) if gap.size else 0.0
        p90 = pct(q(gap, 90)) if gap.size else 0.0
        print(f"     {g:<14}{m.sum():9d}{pct(o.mean()):11.1f}{med:13.3f}{p90:13.3f}", file=fh)

    # worst joints by name
    per_joint = outside.mean(axis=0)
    order = np.argsort(-per_joint)[:12]
    print("     worst joints: " + ", ".join(f"{names[j]}={pct(per_joint[j]):.0f}%" for j in order), file=fh)
    return outside, dmin / BL[:n, None]


# ----------------------------------------------------------------------------------------
def template_baseline(mdl_path, align, tag, fh):
    """Asymmetry floor of the TEMPLATE itself: the number a perfect fit would score."""
    from joint_placement_common import load_model, align_template

    dd = load_model(mdl_path)
    v = np.asarray(dd["v_template"], dtype=np.float64)
    if align:
        v = align_template(v, dd["sym_verts"])
    names = list(dd["J_names"])
    pairs = lr_pairs(names)
    J = joint_regress(v[None], np.asarray(dd["J_regressor"], dtype=np.float64))[0]
    L = np.linalg.norm(v.max(0) - v.min(0))
    dist = np.linalg.norm(J[pairs[:, 0]] - J[pairs[:, 1]] * MIRROR, axis=1)
    Jauth = np.asarray(dd["J"], dtype=np.float64)
    dist_auth = np.linalg.norm(Jauth[pairs[:, 0]] - Jauth[pairs[:, 1]] * MIRROR, axis=1)
    # mesh-level mirror asymmetry via nearest neighbour to the mirrored cloud
    from scipy.spatial import cKDTree

    tree = cKDTree(v * MIRROR)
    dmesh, _ = tree.query(v)
    print(f"  TEMPLATE FLOOR ({os.path.basename(mdl_path)}, align={align})  bbox diag={L:.4f}", file=fh)
    print(
        f"     J_regressor.v_template  L/R mirror gap: median={pct(np.median(dist) / L):.4f}%  "
        f"max={pct(dist.max() / L):.4f}%  of body diag",
        file=fh,
    )
    print(
        f"     authored dd['J']        L/R mirror gap: median={pct(np.median(dist_auth) / L):.4f}%  "
        f"max={pct(dist_auth.max() / L):.4f}%",
        file=fh,
    )
    print(
        f"     mesh surface mirror NN distance: median={pct(np.median(dmesh) / L):.4f}%  "
        f"p99={pct(q(dmesh, 99) / L):.4f}%",
        file=fh,
    )


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_objective_out.txt")
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        hdr = "=" * 92
        print(hdr, file=fh)
        for tmpl, align in [
            ("3D_model_prep/OmniAnt_25PCs_joint_limited.pkl", False),
            ("3D_model_prep/SMIL_OmniAnt.pkl", False),
            ("3D_model_prep/SMPL_fit.pkl", True),
        ]:
            template_baseline(os.path.join(ROOT, tmpl), align, tmpl, fh)
        print(hdr, file=fh)

        summary = []
        for tag, npz, mdl, align in RUNS:
            npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
            mdl = mdl if mdl.startswith("/") else os.path.join(ROOT, mdl)
            if not os.path.exists(npz):
                print(f"{tag}: MISSING {npz}", file=fh)
                continue
            R = load_run(npz, mdl, align=align)
            pairs = lr_pairs(R["J_names"])
            print(
                f"\n### {tag}    n={R['n']}  betas={R['d']['betas'].shape[1]}  "
                f"L/R joint pairs={len(pairs)}  betas_trans="
                f"{'yes' if R['betas_trans'] is not None else 'NO'}",
                file=fh,
            )
            full, res = p1_asymmetry(R, pairs, tag, fh)
            seg = p2_segments(R, tag, fh)
            out_frac = np.nan
            if DO_P3:
                outside, gap = p3_inside(R, tag, fh)
                out_frac = 100 * outside.mean()
            fh.flush()
            summary.append((tag, q(full, 50), full.mean(), np.median(seg), out_frac))

        print("\n" + hdr, file=fh)
        print("SUMMARY", file=fh)
        print(f"{'run':<40}{'P1 med %':>10}{'P1 mean %':>11}{'P2a med %':>11}{'P3 %out':>10}", file=fh)
        for tag, m, mu, s2, o3 in summary:
            print(f"{tag:<40}{m:10.3f}{mu:11.3f}{s2:11.2f}{o3:10.1f}", file=fh)
        print(hdr, file=fh)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
