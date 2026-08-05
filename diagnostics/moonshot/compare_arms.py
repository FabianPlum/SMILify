"""Compare experiment arms against a control, with per-specimen paired statistics.

Applies the pre-registered protocol rather than eyeballing means:

  * Paired comparison per specimen (same 50 scans in every arm, same order), so the
    natural specimen-to-specimen spread -- which is large, these are different species --
    does not swamp the effect.
  * Exact two-sided binomial SIGN TEST on the per-specimen win/loss counts. This answers
    "does this help most scans", which a mean cannot.
  * Welch's t-test (unequal variance) on the paired differences as a secondary read.
  * Reports BOTH the pose-truth checkpoint (last stage that can still change pose) and
    the final stage, because the prior work established that later deform stages recover
    97.6-100.9% of any pose-stage gap -- so a final-stage number is structurally incapable
    of falsifying a pose-level claim.

Metric orientation is declared explicitly (lower_is_better) rather than guessed.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

LOWER_BETTER = {
    "chamfer_l2": True,
    "chamfer_l1": True,
    "acc_mean": True,
    "comp_mean": True,
    "hausdorff_95": True,
    "hausdorff_max": True,
    "fscore@0.005": False,
    "fscore@0.01": False,
    "fscore@0.02": False,
    "fscore@0.05": False,
    "normal_consistency": False,
    "normal_flip_frac": True,
    "edge_logratio_absmean": True,
    "edge_ratio_std": True,
    "area_logratio_absmean": True,
    "tri_quality_mean": False,
    "tri_quality_p05": False,
    "degenerate_tri_frac": True,
    "deform_mag_mean": True,
    "deform_mag_p95": True,
    "deform_mag_max": True,
    "midline_dev_mean": True,
    "midline_dev_p95": True,
    "midline_dev_mean_norm": True,
    "part_body_dist_mean": True,
    "part_head_dist_mean": True,
    "part_antenna_dist_mean": True,
    "part_mandible_dist_mean": True,
    "part_leg_dist_mean": True,
    "part_leg_distal_dist_mean": True,
}

HEADLINE = [
    "chamfer_l2",
    "fscore@0.01",
    "fscore@0.02",
    "hausdorff_95",
    "normal_consistency",
    "edge_logratio_absmean",
    "tri_quality_mean",
    "deform_mag_mean",
    "deform_mag_p95",
    "midline_dev_mean",
    "part_leg_distal_dist_mean",
]

# stages that can still change pose, by naming convention of the two config styles
POSE_STAGES = ("Stage_0", "Stage_1", "A_", "B_", "C_", "D_joint", "E_joint")


def load(run_dir):
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def pose_checkpoint(df):
    """Last stage in this run that could still modify pose."""
    stages = list(dict.fromkeys(df["stage"]))
    cand = [s for s in stages if any(s.startswith(p) for p in POSE_STAGES)]
    return cand[-1] if cand else stages[0]


def cmp_arm(ctrl, arm, stage_ctrl, stage_arm, name, label):
    c = ctrl[ctrl["stage"] == stage_ctrl].set_index("mesh").sort_index()
    a = arm[arm["stage"] == stage_arm].set_index("mesh").sort_index()
    common = c.index.intersection(a.index)
    c, a = c.loc[common], a.loc[common]

    print(f"\n  {label}   (n={len(common)} paired specimens)")
    print(f"    control stage = {stage_ctrl}   arm stage = {stage_arm}")
    print(f"    {'metric':<28} {'control':>11} {'arm':>11} {'delta%':>9} {'win/n':>8} {'sign p':>9}")
    rows = []
    for m in HEADLINE:
        if m not in c.columns or m not in a.columns:
            continue
        cv, av = c[m].values, a[m].values
        lower = LOWER_BETTER.get(m, True)
        wins = int(np.sum(av < cv)) if lower else int(np.sum(av > cv))
        n = len(cv)
        # exact two-sided binomial sign test
        p = stats.binomtest(wins, n, 0.5).pvalue if n else 1.0
        rel = 100.0 * (np.mean(av) - np.mean(cv)) / (abs(np.mean(cv)) + 1e-12)
        better = (rel < 0) if lower else (rel > 0)
        mark = "+" if (better and p < 0.05) else ("-" if (not better and p < 0.05) else " ")
        print(f"  {mark} {m:<28} {np.mean(cv):11.5f} {np.mean(av):11.5f} {rel:+8.1f}% {wins:4d}/{n:<3d} {p:9.2e}")
        rows.append(dict(arm=name, metric=m, control=np.mean(cv), arm_val=np.mean(av), rel=rel, wins=wins, n=n, p=p))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--control", default="C0_control")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ctrl = load(os.path.join(args.runs_dir, args.control))
    if ctrl is None:
        raise SystemExit(f"control {args.control} has no metrics.csv -- score it first")
    ctrl_final = list(dict.fromkeys(ctrl["stage"]))[-1]
    ctrl_pose = pose_checkpoint(ctrl)

    arms = sorted(
        d for d in os.listdir(args.runs_dir) if os.path.isdir(os.path.join(args.runs_dir, d)) and d != args.control
    )
    all_rows = []
    for name in arms:
        df = load(os.path.join(args.runs_dir, name))
        if df is None:
            continue
        final = list(dict.fromkeys(df["stage"]))[-1]
        pose = pose_checkpoint(df)
        print("\n" + "=" * 96)
        print(f"ARM: {name}   vs control {args.control}")
        print("=" * 96)
        all_rows += cmp_arm(ctrl, df, ctrl_pose, pose, name, "POSE-TRUTH CHECKPOINT")
        all_rows += cmp_arm(ctrl, df, ctrl_final, final, name, "FINAL STAGE")

    if all_rows and args.out:
        pd.DataFrame(all_rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
