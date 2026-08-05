"""Summary plots across all experiment arms.

Four figures, each answering one question that the numbers alone answer badly:

  1. stage_trajectories  -- how each metric evolves ACROSS STAGES within a run. This is
     where the baseline's central pathology is visible in one glance: surface metrics
     improve steeply exactly where edge distortion and deform magnitude explode.

  2. tradeoff_scatter    -- surface accuracy against mesh integrity, one point per arm.
     The baseline sits at the far 'accurate but mangled' corner; the question for every
     intervention is whether it moves up-left (better on both) or just slides along the
     frontier (a trade, not a win).

  3. per_part            -- per-anatomical-part distance per arm. The aggregate hides that
     legs behave completely differently from the body, and every integrity intervention
     tested here paid for itself in leg accuracy.

  4. per_specimen        -- paired per-specimen deltas vs the control for the headline
     arms, so a win driven by 3 specimens is distinguishable from a broad one.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "out")


def load_all(runs_dir):
    d = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.csv"))):
        name = os.path.basename(os.path.dirname(p))
        try:
            d[name] = pd.read_csv(p)
        except Exception:
            pass
    return d


def fig_stage_trajectories(data, out):
    """Metric evolution across stages, for the baseline and the control."""
    keys = ["baseline", "C0_control", "A4_nofreeze", "M1_hier"]
    keys = [k for k in keys if k in data]
    metrics = [
        ("fscore@0.01", "F-score@0.01  (higher better)", False),
        ("edge_logratio_absmean", "edge distortion  (lower better)", True),
        ("deform_mag_mean", "|deform_verts| mean  (lower better)", True),
        ("midline_dev_mean", "midline deviation  (lower better)", True),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 3.9))
    for ax, (m, title, lower) in zip(axes, metrics):
        for k in keys:
            df = data[k]
            if m not in df.columns:
                continue
            g = df.groupby("stage", sort=False)[m].mean()
            ax.plot(range(len(g)), g.values, marker="o", ms=4, label=k)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("stage index")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    fig.suptitle(
        "Metric evolution across pipeline stages — the baseline buys surface accuracy with distortion", fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out, "summary_stage_trajectories.png")
    fig.savefig(p, dpi=115)
    plt.close(fig)
    print("wrote", p)


def fig_tradeoff(data, out):
    """Surface accuracy vs mesh integrity, final stage, one point per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, df in sorted(data.items()):
        last = list(dict.fromkeys(df["stage"]))[-1]
        f = df[df["stage"] == last]
        if "fscore@0.01" not in f.columns:
            continue
        x = f["edge_logratio_absmean"].mean()
        y = f["fscore@0.01"].mean()
        x2 = f["deform_mag_mean"].mean()
        hi = name in ("baseline", "C0_control")
        for ax, xv, xl in [(axes[0], x, "edge distortion (log-ratio)"), (axes[1], x2, "mean |deform_verts|")]:
            ax.scatter(
                xv,
                y,
                s=110 if hi else 65,
                alpha=0.85,
                c="#c53030" if hi else "#2b6cb0",
                zorder=3,
                edgecolors="k" if hi else "none",
                linewidths=1.2,
            )
            ax.annotate(name, (xv, y), fontsize=6.5, xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel(xl + "   →  worse mesh")
            ax.set_ylabel("F-score@0.01   →  better surface")
            ax.grid(alpha=0.25)
    axes[0].set_title("Surface accuracy vs edge distortion")
    axes[1].set_title("Surface accuracy vs free-form deformation")
    fig.suptitle("Up-and-left is strictly better. Red = stock pipeline / its harness control.", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, "summary_tradeoff.png")
    fig.savefig(p, dpi=115)
    plt.close(fig)
    print("wrote", p)


def fig_per_part(data, out):
    parts = ["body", "head", "antenna", "mandible", "leg", "leg_distal"]
    names = [n for n in sorted(data) if not n.startswith("hier_smoke")]
    fig, ax = plt.subplots(figsize=(max(9, 1.15 * len(names)), 5))
    w = 0.8 / len(parts)
    x = np.arange(len(names))
    for i, p_ in enumerate(parts):
        col = f"part_{p_}_dist_mean"
        vals = []
        for n in names:
            df = data[n]
            last = list(dict.fromkeys(df["stage"]))[-1]
            f = df[df["stage"] == last]
            vals.append(f[col].mean() if col in f.columns else np.nan)
        ax.bar(x + i * w, vals, w, label=p_)
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("mean distance to target surface (lower better)")
    ax.set_title(
        "Per-anatomical-part accuracy, final stage — aggregates hide that legs behave differently from the body"
    )
    ax.legend(fontsize=7, ncol=6)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    p = os.path.join(out, "summary_per_part.png")
    fig.savefig(p, dpi=115)
    plt.close(fig)
    print("wrote", p)


def fig_per_specimen(data, out, control="C0_control"):
    if control not in data:
        return
    arms = [a for a in ["A4_nofreeze", "A1_offset", "A3_priors", "A5_robust", "M1_hier"] if a in data]
    if not arms:
        return
    metrics = ["fscore@0.01", "deform_mag_mean", "edge_logratio_absmean"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2))
    cdf = data[control]
    clast = list(dict.fromkeys(cdf["stage"]))[-1]
    c = cdf[cdf["stage"] == clast].set_index("mesh").sort_index()
    for ax, m in zip(axes, metrics):
        for a in arms:
            adf = data[a]
            alast = list(dict.fromkeys(adf["stage"]))[-1]
            aa = adf[adf["stage"] == alast].set_index("mesh").sort_index()
            common = c.index.intersection(aa.index)
            if m not in c.columns or m not in aa.columns:
                continue
            delta = (
                100 * (aa.loc[common, m].values - c.loc[common, m].values) / (np.abs(c.loc[common, m].values) + 1e-12)
            )
            ax.plot(np.sort(delta), np.linspace(0, 100, len(delta)), lw=1.6, label=a)
        ax.axvline(0, c="k", lw=1, ls="--")
        ax.set_xlabel(f"per-specimen change in {m}  [%]")
        ax.set_ylabel("percentile of specimens")
        ax.set_title(m, fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5)
    fig.suptitle(
        f"Per-specimen paired change vs {control} — a curve entirely on one side of 0 "
        "means the effect is broad, not driven by a few specimens",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, "summary_per_specimen.png")
    fig.savefig(p, dpi=115)
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = load_all(args.runs_dir)
    print(f"loaded {len(data)} scored arms: {sorted(data)}")
    fig_stage_trajectories(data, OUT)
    fig_tradeoff(data, OUT)
    fig_per_part(data, OUT)
    fig_per_specimen(data, OUT)


if __name__ == "__main__":
    main()
