"""PROBE 22 -- de-confound proxy 2 by decomposing leg-proportion spread by TAXONOMIC LEVEL.

Proxy 2b (cross-specimen spread of within-leg segment proportions) is confounded: both
corpora span many ant genera, and coxa:femur:tibia:tarsus really does vary between genera.
This probe removes the confound by splitting the pairwise spread into

    REPEAT      byte-identical scan fitted twice   -> pure fitter noise, zero biology
    CONSPECIFIC same Genus species, two specimens  -> noise + individual variation
    CONGENERIC  same genus, different species      -> + species variation
    CROSS-GENUS everything else                    -> + genus variation (the number 2b reports)

If CROSS-GENUS is not much bigger than CONSPECIFIC, the fitted leg proportions carry almost
no taxonomic signal and the spread in 2b is measurement error, i.e. joint misplacement.
If CROSS-GENUS >> CONSPECIFIC, the spread is largely real biology and 2b is not evidence of
misplacement.

Taxon parsing:
  worker labels  'Genus_species_CASENT0744328_processed.obj' -> genus, species
  clean labels   'dolichoderus-attelaboides' -> genus 'dolichoderus'; bare numeric labels
                 ('01'..'20') have no taxon and are used only in the CROSS-GENUS pool.
  MEASURED: 'ectatomma-tuberculatum.obj' and 'ectatomma-tuberculatum (1).obj' are BYTE
  IDENTICAL (md5 d758d652435b080d0ad5a8bd1cd3cc4b), so that clean pair is a REPEAT.
"""

import sys
import os
import re
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from joint_placement_common import load_run, global_rigid, leg_chain_indices

ROOT = "/home/fabi/dev/SMILify"
CLEAN_NPZ = (
    "/media/fabi/Data/SMILify_LEGACY/Fitter_RESULTS/fit3d_results_ALL_ANTS_WITH_PART_SCALING/Stage_3_deform_fine.npz"
)

RUNS = [
    (
        "LIM_0 (worker)",
        "diagnostics/moonshot/runs/LIM_0/Stage_3_deform_fine.npz",
        "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        False,
        "worker",
    ),
    (
        "M7_midline (worker)",
        "diagnostics/moonshot/runs/M7_handoff_midline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
        False,
        "worker",
    ),
    (
        "baseline (worker, stock)",
        "diagnostics/moonshot/runs/baseline/Stage_3_deform_fine.npz",
        "3D_model_prep/SMIL_OmniAnt.pkl",
        False,
        "worker",
    ),
    ("ALL_ANTS_CLEAN (81)", CLEAN_NPZ, "3D_model_prep/SMPL_fit.pkl", True, "clean"),
]

SEG_NAMES = ["coxa", "trochanter", "femur", "tibia", "tarsus"]
REPEAT_CLEAN = {"ectatomma-tuberculatum", "ectatomma-tuberculatum (1)"}


def taxon(label, kind):
    lab = re.sub(r"\.obj$", "", str(label))
    if kind == "worker":
        t = lab.replace("_processed", "").split("_")
        return t[0].lower(), (t[0] + "_" + t[1]).lower() if len(t) > 1 else None
    lab_l = lab.lower()
    if re.fullmatch(r"\d+", lab_l):
        return None, None
    t = lab_l.split("-")
    if len(t) == 1:
        return t[0], None
    return t[0], t[0] + "-" + t[1]


def proportions(R):
    """(n, n_legs, 5) within-leg segment proportions, L/R averaged, rest space."""
    nJ = R["nJ"]
    Rs = np.tile(np.eye(3), (R["n"], nJ, 1, 1))
    Jr, _ = global_rigid(Rs, R["J"], R["parents"], R["logscale"], R["betas_trans"])
    chains = leg_chain_indices(R["J_names"])
    out = []
    for leg in ("1", "2", "3"):
        sides = [chains[(leg, s)] for s in ("l", "r") if (leg, s) in chains]
        lens = []
        for ch in sides:
            v = Jr[:, ch[1:], :] - Jr[:, ch[:-1], :]
            lens.append(np.linalg.norm(v, axis=2))
        L = np.mean(lens, axis=0)  # (n,5)  L/R averaged
        out.append(L / L.sum(axis=1, keepdims=True))
    return np.stack(out, axis=1)  # (n, 3, 5)


def pair_diff(P, i, j):
    """Median relative |difference| of proportions between two specimens, over legs x segs."""
    a, b = P[i], P[j]
    return float(np.median(np.abs(a - b) / (0.5 * (a + b))))


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        [t.write(x) for t in self.s]

    def flush(self):
        [t.flush() for t in self.s]


def main():
    outp = os.path.join(ROOT, "diagnostics/moonshot/joint_placement_taxon_out.txt")
    with open(outp, "w") as f:
        fh = Tee(sys.stdout, f)
        print(__doc__, file=fh)
        summary = []
        disp_store = {}
        for tag, npz, mdl, align, kind in RUNS:
            npz = npz if npz.startswith("/") else os.path.join(ROOT, npz)
            R = load_run(npz, os.path.join(ROOT, mdl), align=align)
            labels = [re.sub(r"\.obj$", "", str(x)) for x in R["d"]["labels"]]
            P = proportions(R)
            n = R["n"]
            tx = [taxon(l, kind) for l in labels]

            buckets = {"REPEAT": [], "CONSPECIFIC": [], "CONGENERIC": [], "CROSS-GENUS": []}
            for i, j in itertools.combinations(range(n), 2):
                gi, si = tx[i]
                gj, sj = tx[j]
                d = pair_diff(P, i, j)
                if kind == "clean" and labels[i] in REPEAT_CLEAN and labels[j] in REPEAT_CLEAN:
                    buckets["REPEAT"].append(d)
                elif gi is not None and gi == gj and si is not None and si == sj:
                    buckets["CONSPECIFIC"].append(d)
                elif gi is not None and gi == gj:
                    buckets["CONGENERIC"].append(d)
                else:
                    buckets["CROSS-GENUS"].append(d)

            print(f"\n### {tag}   n={n}", file=fh)
            print(f"   {'level':<14}{'n_pairs':>9}{'median':>10}{'mean':>10}{'p90':>10}", file=fh)
            med = {}
            for k in ("REPEAT", "CONSPECIFIC", "CONGENERIC", "CROSS-GENUS"):
                v = np.array(buckets[k])
                if v.size == 0:
                    print(f"   {k:<14}{0:>9}{'  -- none in this corpus':>30}", file=fh)
                    med[k] = np.nan
                    continue
                med[k] = float(np.median(v))
                print(
                    f"   {k:<14}{v.size:9d}{100 * np.median(v):10.2f}{100 * v.mean():10.2f}"
                    f"{100 * np.percentile(v, 90):10.2f}",
                    file=fh,
                )
            names = []
            for i, j in itertools.combinations(range(n), 2):
                gi, si = tx[i]
                gj, sj = tx[j]
                if (
                    gi is not None
                    and gi == gj
                    and si is not None
                    and si == sj
                    and not (kind == "clean" and labels[i] in REPEAT_CLEAN and labels[j] in REPEAT_CLEAN)
                ):
                    names.append(f"{labels[i][:34]} | {labels[j][:34]} -> {100 * pair_diff(P, i, j):.2f}%")
            if names:
                print("   conspecific pairs:", file=fh)
                for s in names:
                    print(f"      {s}", file=fh)
            ratio = med["CONSPECIFIC"] / med["CROSS-GENUS"] if med["CROSS-GENUS"] else np.nan
            print(
                f"   >>> CONSPECIFIC / CROSS-GENUS = {ratio:.2f}   "
                f"(1.0 = fitted leg proportions carry NO taxonomic signal)",
                file=fh,
            )
            # per-specimen dispersion: median proportion-difference to every other specimen.
            # This is the independent unit (a specimen), unlike the pairwise pool above.
            disp = np.array([np.median([pair_diff(P, i, j) for j in range(n) if j != i]) for i in range(n)])
            disp_store[tag] = disp
            print(
                f"   per-specimen dispersion (median diff to all others): "
                f"median={100 * np.median(disp):.2f}%  IQR=[{100 * np.percentile(disp, 25):.2f},"
                f"{100 * np.percentile(disp, 75):.2f}]%",
                file=fh,
            )
            summary.append((tag, med["REPEAT"], med["CONSPECIFIC"], med["CONGENERIC"], med["CROSS-GENUS"], ratio))
            fh.flush()

        print("\n" + "=" * 100, file=fh)
        print("SUMMARY -- median relative difference in within-leg segment proportions [%]", file=fh)
        print(f"{'run':<28}{'REPEAT':>9}{'CONSPEC':>9}{'CONGEN':>9}{'XGENUS':>9}{'CONSPEC/XGENUS':>16}", file=fh)
        for t, r, c, g, x, ra in summary:
            fmt = lambda v: f"{100 * v:9.2f}" if v == v else f"{'--':>9}"  # noqa: E731
            print(f"{t:<28}{fmt(r)}{fmt(c)}{fmt(g)}{fmt(x)}{ra:16.2f}", file=fh)
        print("=" * 100, file=fh)
        from scipy.stats import mannwhitneyu

        if "LIM_0 (worker)" in disp_store and "ALL_ANTS_CLEAN (81)" in disp_store:
            a, b = disp_store["LIM_0 (worker)"], disp_store["ALL_ANTS_CLEAN (81)"]
            u, pv = mannwhitneyu(a, b, alternative="two-sided")
            print(
                f"per-specimen dispersion, LIM_0 vs CLEAN: median {100 * np.median(a):.2f}% vs "
                f"{100 * np.median(b):.2f}%  ratio={np.median(a) / np.median(b):.2f}  "
                f"U={u:.0f}  p={pv:.2e}  (n={len(a)} vs {len(b)})",
                file=fh,
            )
        print("=" * 100, file=fh)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
