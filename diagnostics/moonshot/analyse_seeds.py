"""Seed-repeat analysis for the marginal claims.

The large effects in REPORT.md (deform_mag −51% at 50/50 wins) cannot be seed artifacts. The
small ones can: A4_nofreeze's fscore@0.01 gain is +0.7% at p=0.0066, which is exactly the
size of effect that a single seed gets wrong. Khaoula's keypoint arm died on this — run-to-run
std ~12% against a 15% bar, with one single-seed "improvement" inverting to a significant
degradation under 3-seed replication.

Reports, for each metric:
  * the per-seed A4-minus-C0 paired difference, so between-seed spread is visible
  * whether the sign is consistent across all three seeds (the thing that matters)
  * the pooled estimate with between-seed std
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

METRICS = [
    ("fscore@0.01", False),
    ("fscore@0.02", False),
    ("chamfer_l2", True),
    ("edge_logratio_absmean", True),
    ("deform_mag_mean", True),
    ("tri_quality_mean", False),
    ("part_leg_distal_dist_mean", True),
]


def load(run):
    p = os.path.join(RUNS, run, "metrics.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    last = list(dict.fromkeys(d["stage"]))[-1]
    return d[d["stage"] == last].set_index("mesh").sort_index()


def main():
    seeds = []
    for s, (c, a) in {
        0: ("C0_control", "A4_nofreeze"),
        1: ("C0_control_s1", "A4_nofreeze_s1"),
        2: ("C0_control_s2", "A4_nofreeze_s2"),
    }.items():
        dc, da = load(c), load(a)
        if dc is None or da is None:
            print(f"  seed {s}: missing ({c} / {a})")
            continue
        seeds.append((s, dc, da))

    if len(seeds) < 2:
        raise SystemExit("need at least two seeds")

    print("=" * 90)
    print(f"SEED REPEATS — A4_nofreeze vs C0_control, {len(seeds)} seeds, 50 paired specimens each")
    print("=" * 90)
    print("\n  Relative change per seed (sign flipped so + is always better):\n")
    hdr = f"  {'metric':<28}" + "".join(f"{'seed ' + str(s):>11}" for s, _, _ in seeds)
    print(hdr + f"{'mean':>10}{'sd':>8}  {'consistent?':>12}")
    print("  " + "-" * (len(hdr) + 30))

    for m, lower in METRICS:
        vals, wins = [], []
        for s, dc, da in seeds:
            common = dc.index.intersection(da.index)
            if m not in dc.columns:
                continue
            cv = dc.loc[common, m].values
            av = da.loc[common, m].values
            rel = 100 * (av.mean() - cv.mean()) / (abs(cv.mean()) + 1e-12)
            vals.append(-rel if lower else rel)
            wins.append(int((av < cv).sum()) if lower else int((av > cv).sum()))
        if not vals:
            continue
        v = np.array(vals)
        consistent = "YES" if (np.all(v > 0) or np.all(v < 0)) else "NO -- flips"
        row = f"  {m:<28}" + "".join(f"{x:+10.2f}%" for x in v)
        print(row + f"{v.mean():+9.2f}%{v.std():7.2f}  {consistent:>12}")

    print("\n  A claim is only safe if the sign is the same in every seed AND the mean is")
    print("  comfortably larger than the between-seed sd. Anything marked 'flips' should be")
    print("  reported as inconclusive, not as an effect.")

    # per-seed win counts for the marginal metric
    print("\n  Per-seed win counts for fscore@0.01 (the marginal claim):")
    for s, dc, da in seeds:
        common = dc.index.intersection(da.index)
        cv = dc.loc[common, "fscore@0.01"].values
        av = da.loc[common, "fscore@0.01"].values
        w = int((av > cv).sum())
        p = stats.binomtest(w, len(cv), 0.5).pvalue
        print(f"    seed {s}: {w}/{len(cv)} wins, sign-test p = {p:.4f}")


if __name__ == "__main__":
    main()
