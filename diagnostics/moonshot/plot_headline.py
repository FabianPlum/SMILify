"""The report's centrepiece figure: what each approach actually buys, relative to the
stock pipeline.

Everything is expressed as "% better than the stock baseline", with the sign flipped for
lower-is-better metrics, so bars above zero are always improvements. Metrics are grouped
into the three things that matter and are constantly confused with each other:

  SURFACE      does the fit reach the target (what the optimizer maximises)
  MESH         is the result a valid mesh (what free-form deformation destroys)
  CORRESPOND.  do vertices land on matching anatomy (what registration is FOR)

The stock pipeline scores well on the first and badly on the other two; the whole point of
the report is that those are different questions.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RUNS = os.path.join(HERE, "runs")

GROUPS = [
    ("SURFACE", [("fscore@0.01", False), ("fscore@0.02", False), ("chamfer_l2", True), ("hausdorff_95", True)]),
    (
        "MESH VALIDITY",
        [
            ("edge_logratio_absmean", True),
            ("tri_quality_mean", False),
            ("deform_mag_mean", True),
            ("dihedral_p99", True),
        ],
    ),
    ("CORRESPONDENCE", [("nbr_keep", False), ("bilateral", True), ("midline_dev_mean", True)]),
]

ARMS = ["A4_nofreeze", "B1_best_off0p2", "M5_handoff", "M7_handoff_midline", "M4b_ceiling_free"]


def load_metrics():
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "metrics.csv"))):
        n = os.path.basename(os.path.dirname(p))
        d = pd.read_csv(p)
        last = list(dict.fromkeys(d["stage"]))[-1]
        out[n] = d[d["stage"] == last].mean(numeric_only=True).to_dict()
    # correspondence + tearing probes live in their own JSON files
    for fn, keys in [
        ("probe10_correspondence.json", ("nbr_keep", "bilateral")),
        ("probe12_tearing.json", ("dihedral_p99", "folded_face_frac")),
    ]:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        rows = j["rows"] if isinstance(j, dict) else j
        for r in rows:
            if r["arm"] in out:
                for k in keys:
                    if k in r:
                        out[r["arm"]][k] = r[k]
    return out


def main():
    m = load_metrics()
    if "baseline" not in m:
        raise SystemExit("no baseline")
    base = m["baseline"]
    arms = [a for a in ARMS if a in m]

    fig, axes = plt.subplots(1, len(GROUPS), figsize=(5.2 * len(GROUPS), 5.4), sharey=True)
    for ax, (gname, metrics) in zip(axes, GROUPS):
        metrics = [(k, lo) for k, lo in metrics if k in base]
        x = np.arange(len(metrics))
        w = 0.8 / max(len(arms), 1)
        for i, a in enumerate(arms):
            vals = []
            for k, lo in metrics:
                if k not in m[a] or k not in base:
                    vals.append(0.0)
                    continue
                b, v = base[k], m[a][k]
                rel = 100 * (v - b) / (abs(b) + 1e-12)
                vals.append(-rel if lo else rel)  # sign flipped so up = better
            ax.bar(x + i * w, vals, w, label=a)
        ax.axhline(0, c="k", lw=1)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(
            [k.replace("_absmean", "").replace("_mean", "") for k, _ in metrics], rotation=25, ha="right", fontsize=8
        )
        ax.set_title(gname, fontsize=11)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("% better than the stock pipeline  (up = better, all metrics)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(
        "What each approach buys, relative to the stock pipeline.  The stock fitter scores "
        "well on SURFACE and badly on the other two —\nand only the other two are what "
        "registration exists to produce.",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUT, "headline.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("wrote", p)

    # console version
    print(f"\n{'metric':<26}" + "".join(f"{a[:15]:>17}" for a in arms))
    for gname, metrics in GROUPS:
        print(f"-- {gname}")
        for k, lo in metrics:
            if k not in base:
                continue
            row = f"  {k:<24}"
            for a in arms:
                if k not in m[a]:
                    row += f"{'--':>17}"
                    continue
                rel = 100 * (m[a][k] - base[k]) / (abs(base[k]) + 1e-12)
                row += f"{(-rel if lo else rel):+16.1f}%"
            print(row)


if __name__ == "__main__":
    main()
