"""REFUTE-G -- independent re-derivation of "% joints outside the OWN FITTED MESH".

This is the reference surface that is IDENTICAL in kind across the two corpora: the authored
template topology (10235 vs 10236 verts, 20466 vs 20468 faces), watertightness checked below,
no photogrammetry holes, no resolution difference, no scan-provenance difference. If the
worker/clean separation is a property of joint placement it must show up here too.

Also scores the same point sets as REFUTE-C so the definitional and control comparisons are
available on this surface as well.
"""

import os
import sys
from collections import defaultdict

import numpy as np
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
    ),
    (
        "baseline (worker)",
        f"{ROOT}/diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        f"{ROOT}/3D_model_prep/SMIL_OmniAnt.pkl",
        False,
    ),
    ("CLEAN (81)", CLEAN_NPZ, f"{ROOT}/3D_model_prep/SMPL_fit.pkl", True),
]


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def edge_stats(F):
    c = defaultdict(int)
    for t in F:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            c[(min(a, b), max(a, b))] += 1
    v = np.fromiter(c.values(), int)
    return int((v == 1).sum()), int((v > 2).sum()), len(v)


def main():
    outp = f"{ROOT}/diagnostics/moonshot/refute_fittedmesh_recheck_out.txt"
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        store = {}
        for tag, npz, mdl, align in RUNS:
            d = np.load(npz, allow_pickle=True)
            dd = load_model(mdl)
            n = d["betas"].shape[0]
            nJ = np.asarray(dd["J_regressor"]).shape[0]
            vt = np.asarray(dd["v_template"], float)
            if align:
                vt = align_template(vt, dd["sym_verts"])
            betas = d["betas"].astype(np.float64)
            sdirs = np.asarray(dd["shapedirs"], float)
            v_shaped = vt[None] + np.einsum("nk,vck->nvc", betas, sdirs[:, :, : betas.shape[1]])
            Jr = np.asarray(dd["J_regressor"], float)
            J = joint_regress(v_shaped, Jr)
            theta = np.concatenate([d["global_rot"][:, None, :], d["joint_rot"]], 1).astype(float)
            Rs = rodrigues(theta.reshape(-1, 3)).reshape(n, nJ, 3, 3)
            par = np.asarray(dd["kintree_table"][0]).astype(int)
            bt = d["betas_trans"].astype(np.float64) if "betas_trans" in d.files else None
            Jt, A = global_rigid(Rs, J, par, d["log_beta_scales"].astype(np.float64), bt)
            W = np.asarray(dd["weights"], float)
            T = np.einsum("vj,njab->nvab", W, A)
            vh = np.concatenate([v_shaped, np.ones((n, v_shaped.shape[1], 1))], 2)
            v_posed = np.einsum("nvab,nvb->nva", T, vh)[:, :, :3] + d["trans"][:, None, :]
            v_final = d["verts"].astype(np.float64)
            F = np.asarray(dd["f"]).astype(int)
            nb_, nm_, ne_ = edge_stats(F)
            names = list(dd["J_names"])
            groups = np.array([joint_group(x) for x in names])
            wing = np.array([x.startswith("w_") for x in names])
            ws = W.sum(0, keepdims=True)
            Wn = np.where(ws > 1e-9, W / np.where(ws > 1e-9, ws, 1.0), 0.0)
            dead = ws[0] <= 1e-9
            J_static = Jt + d["trans"][:, None, :]
            sets = {
                "J_static (probe20/21)": J_static,
                "J_regressed (real cfg)": joint_regress(v_posed, Jr),
                "C_skinweight (CONTROL)": np.einsum("vj,nvc->njc", Wn, v_final),
            }
            sets["C_skinweight (CONTROL)"][:, dead] = J_static[:, dead]
            print(f"\n### {tag}   n={n}  verts={v_final.shape[1]}  faces={F.shape[0]}", file=fh)
            print(
                f"   TEMPLATE MESH TOPOLOGY: boundary edges={nb_}  non-manifold edges={nm_} "
                f"  (of {ne_})   -> GWN {'valid' if nb_ == 0 and nm_ == 0 else 'DEGRADED'}",
                file=fh,
            )
            per = {k: [] for k in sets}
            for i in range(n):
                V = v_final[i]
                P = np.concatenate([sets[k][i] for k in sets], 0)
                w = winding_number(P, V, F, chunk=4000)
                out = np.abs(w) < 0.5
                for k, s in enumerate(sets):
                    per[s].append(out[k * nJ : (k + 1) * nJ])
            store[tag] = {k: np.array(v) for k, v in per.items()}
            print(f"   {'point set':<26}{'%out all 55':>13}{'%out no wings':>15}", file=fh)
            for s in sets:
                o = store[tag][s]
                print(f"   {s:<26}{100 * o.mean():13.1f}{100 * o[:, ~wing].mean():15.1f}", file=fh)
            print("   per-group %outside (J_static | CONTROL):", file=fh)
            for g in sorted(set(groups)):
                m = groups == g
                print(
                    f"     {g:<14}{100 * store[tag]['J_static (probe20/21)'][:, m].mean():8.1f}"
                    f"{100 * store[tag]['C_skinweight (CONTROL)'][:, m].mean():10.1f}",
                    file=fh,
                )
            fh.flush()

        print("\n" + "=" * 92, file=fh)
        print("WORKER vs CLEAN on the TEMPLATE-TOPOLOGY reference surface", file=fh)
        a, b = store["LIM_0 (worker)"], store["CLEAN (81)"]
        print(f"{'point set':<26}{'worker':>9}{'clean':>9}{'ratio':>8}{'p':>12}", file=fh)
        for s in a:
            x = a[s].mean(1)
            y = b[s].mean(1)
            _, p = mannwhitneyu(x, y)
            print(
                f"{s:<26}{100 * np.median(x):9.1f}{100 * np.median(y):9.1f}"
                f"{np.median(x) / max(np.median(y), 1e-9):8.2f}{p:12.2e}",
                file=fh,
            )
        print("\nCompare with the headline, which used the RAW SCAN as the reference surface:", file=fh)
        print("   worker 61.5  clean 30.1  ratio 2.12", file=fh)
        print("=" * 92, file=fh)
    print(f"wrote {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
