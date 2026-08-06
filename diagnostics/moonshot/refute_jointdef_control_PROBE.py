"""REFUTE-C -- two attacks on "61.5% of worker joints lie outside their own scan vs 30.1%".

ATTACK 1 -- JOINT DEFINITION.
  `config.STATIC_JOINT_LOCATIONS` is driven by dd['static_joint_locs'], which is ABSENT in all
  three models, so the flag is False and smal_torch.py:380-382 returns
  joints = J_regressor . posed_verts, NOT the batch_global_rigid output the probe used.
  Measured in refute_jointplacement_gate_out.txt: the two definitions disagree by
  0.849 %diag (median) for LIM_0 but only 0.015 %diag for CLEAN -- a 57x asymmetry, and the
  worker disagreement alone is bigger than the entire claimed effect (median joint->scan gap
  0.50% vs 0.28% of diag). So the metric is re-run under every definition.

ATTACK 2 -- A ZERO-PLACEMENT-ERROR CONTROL POINT SET.
  C_w[j] = sum_v W[v,j] * verts[v] / sum_v W[v,j]   on the FINAL fitted verts.
  This is the skin-weight centroid of the fitted surface patch that joint j controls. It is
  a medial point of the FITTED mesh BY CONSTRUCTION and carries no joint-placement
  information at all. If C_w is also ~2x more often outside the scan for workers, then
  "% outside the scan" is measuring fitted-surface-vs-scan disagreement and/or scan
  pathology, not joint placement.

ATTACK 3 -- SENSITIVITY CALIBRATION.
  outfrac as a function of isotropic joint jitter eps*diag, per corpus. Gives the physical
  displacement that the 30.1 -> 61.5 gap corresponds to, and shows whether the metric has
  the same gain in both corpora (it does not if the corpora's limbs differ in thickness).
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_placement_common import (  # noqa: E402
    align_template,
    global_rigid,
    joint_group,
    joint_regress,
    load_model,
    rodrigues,
    winding_number,
)

ROOT = "/home/fabi/dev/SMILify"
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

RUNS = [
    (
        "LIM_0 (worker)",
        f"{ROOT}/diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        f"{ROOT}/3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
        f"{ROOT}/diagnostics/moonshot/bench50",
    ),
    ("CLEAN (81)", CLEAN_NPZ, f"{ROOT}/3D_model_prep/SMPL_fit.pkl", True, f"{ROOT}/diagnostics/moonshot/clean81"),
]

EPS = [0.0, 0.0025, 0.005, 0.010, 0.020]
NREP = 2
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


def build(npz, mdl, align):
    d = np.load(npz, allow_pickle=True)
    dd = load_model(mdl)
    n = d["betas"].shape[0]
    nJ = np.asarray(dd["J_regressor"]).shape[0]
    vt = np.asarray(dd["v_template"], float)
    if align:
        vt = align_template(vt, dd["sym_verts"])
    betas = d["betas"].astype(np.float64)
    nb = betas.shape[1]
    sd = np.asarray(dd["shapedirs"], float)
    v_shaped = vt[None] + np.einsum("nk,vck->nvc", betas, sd[:, :, :nb])
    Jr = np.asarray(dd["J_regressor"], float)
    J = joint_regress(v_shaped, Jr)
    theta = np.concatenate([d["global_rot"][:, None, :], d["joint_rot"]], 1).astype(np.float64)
    Rs = rodrigues(theta.reshape(-1, 3)).reshape(n, nJ, 3, 3)
    bt = d["betas_trans"].astype(np.float64) if "betas_trans" in d.files else None
    parents = np.asarray(dd["kintree_table"][0]).astype(int)
    Jt, A = global_rigid(Rs, J, parents, d["log_beta_scales"].astype(np.float64), bt)
    W = np.asarray(dd["weights"], float)
    T = np.einsum("vj,njab->nvab", W, A)
    vh = np.concatenate([v_shaped, np.ones((n, v_shaped.shape[1], 1))], 2)
    v_posed = np.einsum("nvab,nvb->nva", T, vh)[:, :, :3] + d["trans"][:, None, :]
    v_final = d["verts"].astype(np.float64)
    J_static = Jt + d["trans"][:, None, :]
    J_rp = joint_regress(v_posed, Jr)
    J_rd = joint_regress(v_final, Jr)
    ws = W.sum(0, keepdims=True)
    # w_1_r / w_1_l carry ZERO skinning weight in BOTH templates -- they deform nothing and
    # are unconstrained by the surface. Fall back to the joint itself so the arrays stay the
    # same shape; they are reported separately via the "no wing joints" subset.
    Wn = np.where(ws > 1e-9, W / np.where(ws > 1e-9, ws, 1.0), 0.0)
    dead = ws[0] <= 1e-9
    C_final = np.einsum("vj,nvc->njc", Wn, v_final)
    C_posed = np.einsum("vj,nvc->njc", Wn, v_posed)
    C_final[:, dead] = J_static[:, dead]
    C_posed[:, dead] = J_static[:, dead]
    return dict(
        d=d,
        dd=dd,
        n=n,
        names=list(dd["J_names"]),
        sets={
            "J_static (probe21)": J_static,
            "J_regressed (real cfg)": J_rp,
            "J_regressed_deformed": J_rd,
            "C_skinweight_final (CONTROL)": C_final,
            "C_skinweight_posed (CONTROL)": C_posed,
        },
        W=W,
        v_final=v_final,
        dead=dead,
    )


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = f"{ROOT}/diagnostics/moonshot/refute_jointdef_control_out.txt"
    rng = np.random.default_rng(7)
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        store = {}
        for tag, npz, mdl, align, sdir in RUNS:
            R = build(npz, mdl, align)
            names = R["names"]
            groups = np.array([joint_group(x) for x in names])
            labels = [os.path.splitext(str(x))[0] for x in R["d"]["labels"]]
            avail = {os.path.splitext(x)[0]: os.path.join(sdir, x) for x in os.listdir(sdir) if x.endswith(".obj")}
            setnames = list(R["sets"].keys())
            wing = np.array([x.startswith("w_") for x in names])
            per_spec = {s: [] for s in setnames}
            per_spec_nw = {s: [] for s in setnames}
            per_joint = {s: [] for s in setnames}
            per_spec_grp = {s: [] for s in setnames}
            jit = {e: [] for e in EPS}
            gap_static, thick = [], []
            miss = 0
            for i in range(R["n"]):
                if labels[i] not in avail:
                    miss += 1
                    continue
                V, F = load_obj_np(avail[labels[i]])
                V = normalise(V)
                diag = np.linalg.norm(V.max(0) - V.min(0))
                tree = cKDTree(V)
                pts, meta = [], []
                for s in setnames:
                    P = R["sets"][s][i]
                    pts.append(P)
                    meta.append((s, P.shape[0]))
                base = R["sets"]["J_static (probe21)"][i]
                for e in EPS:
                    for r in range(NREP if e > 0 else 1):
                        pts.append(base + rng.normal(0, e * diag, base.shape))
                        meta.append((f"jit{e}", base.shape[0]))
                P = np.concatenate(pts, 0)
                w = winding_number(P, V, F, chunk=CHUNK)
                out = np.abs(w) < 0.5
                k = 0
                jitacc = {e: [] for e in EPS}
                for nm, m in meta:
                    seg = out[k : k + m]
                    k += m
                    if nm.startswith("jit"):
                        jitacc[float(nm[3:])].append(seg.mean())
                    else:
                        per_spec[nm].append(seg.mean())
                        per_spec_nw[nm].append(seg[~wing].mean())
                        per_joint[nm].append(seg.astype(float))
                        per_spec_grp[nm].append([seg[groups == g].mean() for g in sorted(set(groups))])
                for e in EPS:
                    jit[e].append(np.mean(jitacc[e]))
                g, _ = tree.query(base)
                gap_static.append(np.median(g) / diag)
                # local limb thickness of the FITTED mesh at each joint: median distance from
                # the joint to the vertices it dominates (W>0.5)
                th = []
                for j in range(len(names)):
                    sel = R["W"][:, j] > 0.5
                    if sel.sum() >= 5:
                        th.append(np.median(np.linalg.norm(R["v_final"][i][sel] - base[j], axis=1)) / diag)
                thick.append(np.median(th))
            store[tag] = dict(
                per_spec=per_spec,
                per_spec_nw=per_spec_nw,
                jit=jit,
                gap=np.array(gap_static),
                thick=np.array(thick),
                per_joint={k: np.array(v) for k, v in per_joint.items()},
                names=names,
                grp=per_spec_grp,
                gnames=sorted(set(groups)),
            )
            print(f"\n### {tag}  n_used={len(gap_static)}  missing={miss}", file=fh)
            print(f"   {'point set':<32}{'%out all 55':>13}{'%out no wings (51)':>20}", file=fh)
            for s in setnames:
                v = np.array(per_spec[s])
                v2 = np.array(per_spec_nw[s])
                print(f"   {s:<32}{100 * np.median(v):13.1f}{100 * np.median(v2):20.1f}", file=fh)
            print(f"   median joint->scan gap (J_static) = {100 * np.median(gap_static):.3f} %diag", file=fh)
            print(
                f"   median local limb thickness at joints (fitted mesh, W>0.5) = {100 * np.median(thick):.3f} %diag",
                file=fh,
            )
            print(f"   {'jitter eps (%diag)':<24}{'% outside':>12}", file=fh)
            for e in EPS:
                print(f"   {100 * e:<24.2f}{100 * np.median(jit[e]):>12.1f}", file=fh)
            fh.flush()

        print("\n" + "=" * 92, file=fh)
        print("WORKER vs CLEAN, per point set (Mann-Whitney over specimens)", file=fh)
        a, b = store["LIM_0 (worker)"], store["CLEAN (81)"]
        print(
            f"{'point set':<32}{'worker':>10}{'clean':>10}{'ratio':>9}{'p':>12}   | no-wing worker/clean/ratio", file=fh
        )
        for s in a["per_spec"]:
            x, y = np.array(a["per_spec"][s]), np.array(b["per_spec"][s])
            x2, y2 = np.array(a["per_spec_nw"][s]), np.array(b["per_spec_nw"][s])
            u, p = mannwhitneyu(x, y)
            print(
                f"{s:<32}{100 * np.median(x):10.1f}{100 * np.median(y):10.1f}"
                f"{np.median(x) / max(np.median(y), 1e-9):9.2f}{p:12.2e}"
                f"   | {100 * np.median(x2):6.1f}{100 * np.median(y2):8.1f}"
                f"{np.median(x2) / max(np.median(y2), 1e-9):8.2f}",
                file=fh,
            )
        print("\nPER-JOINT %outside (J_static vs CONTROL), worker | clean:", file=fh)
        nm = a["names"]
        for j in range(len(nm)):
            print(
                f"  {nm[j]:<10} J_static {100 * a['per_joint']['J_static (probe21)'][:, j].mean():5.1f}"
                f" {100 * b['per_joint']['J_static (probe21)'][:, j].mean():5.1f}   CONTROL "
                f"{100 * a['per_joint']['C_skinweight_final (CONTROL)'][:, j].mean():5.1f}"
                f" {100 * b['per_joint']['C_skinweight_final (CONTROL)'][:, j].mean():5.1f}",
                file=fh,
            )
        print("\nPER-GROUP, control point set C_skinweight_final:", file=fh)
        gn = a["gnames"]
        for gi, g in enumerate(gn):
            x = np.array([r[gi] for r in a["grp"]["C_skinweight_final (CONTROL)"]])
            y = np.array([r[gi] for r in b["grp"]["C_skinweight_final (CONTROL)"]])
            xs = np.array([r[gi] for r in a["grp"]["J_static (probe21)"]])
            ys = np.array([r[gi] for r in b["grp"]["J_static (probe21)"]])
            print(
                f"  {g:<14} CONTROL worker={100 * np.median(x):5.1f} clean={100 * np.median(y):5.1f}"
                f"  ratio={np.median(x) / max(np.median(y), 1e-9):5.2f}   ||   "
                f"J_static worker={100 * np.median(xs):5.1f} clean={100 * np.median(ys):5.1f} "
                f"ratio={np.median(xs) / max(np.median(ys), 1e-9):5.2f}",
                file=fh,
            )
        print("\nMETRIC GAIN (how much jitter reproduces the other corpus):", file=fh)
        for tag in ("LIM_0 (worker)", "CLEAN (81)"):
            s = store[tag]
            print(
                f"  {tag:<16} " + "  ".join(f"eps={100 * e:.2f}%->{100 * np.median(s['jit'][e]):.1f}%" for e in EPS),
                file=fh,
            )
        print("=" * 92, file=fh)
    print(f"wrote {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
