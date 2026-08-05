"""Independent check of the aggregate claims in Khaoula's updated penetration deliverable.

Source data: `1_penetration_weight_reduction_fix/penetration_regression_flags.csv`
(10 specimens x 3 seeds, total penetration burden with the gentle weight off vs on).

Claim under test, from that folder's README:
    "8 of 10 specimens show a real net improvement (3-seed average, consistent across
     seeds), 2 (GAGA-02-08, GAGA-04-06) show a real net regression, also consistent
     across seeds."

The reason this needs checking is that "3-seed average" is ambiguous, and the two natural
readings disagree on this data:

  (A) average the BURDENS across seeds, then take the percentage change
      -> this is what her table reports
  (B) count how many SEEDS regressed per specimen, and take the majority
      -> this is what the per-seed `flagged_regression` column implies

(A) is dominated by whichever seed happened to have the largest baseline burden, because
the burden is an unnormalised sum of depths. Where one seed's baseline is 10x another's,
that seed effectively decides the specimen's verdict. (B) is insensitive to that but throws
away magnitude.

Neither is wrong; they answer different questions. But they give different regression rates,
and the deliverable's headline uses only (A).
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "penetration_regression_flags.csv")


def short(s):
    return s.replace("antscan_processed_", "").replace("GAGA_5x_", "").replace("5x_", "").replace("_processed", "")


def main():
    df = pd.read_csv(CSV)
    df["spec"] = df["specimen"].map(short)

    print("=" * 100)
    print("INDEPENDENT CHECK — penetration regression flags, 10 specimens x 3 seeds")
    print("=" * 100)

    rows = []
    for spec, g in df.groupby("spec", sort=False):
        off = g["total_penetration_burden_off"].values
        on = g["total_penetration_burden_on"].values
        # (A) mean-of-burdens, then percentage  -- the deliverable's statistic
        a = 100 * (on.mean() - off.mean()) / off.mean()
        # (B) per-seed sign, majority vote
        seed_regress = int((on > off).sum())
        # (C) mean of the per-seed percentages -- a third reading, reported for context
        c = g["pct_change"].mean()
        rows.append(
            dict(
                spec=spec,
                A_mean_burden_pct=a,
                B_seeds_regressed=seed_regress,
                C_mean_of_pct=c,
                off_min=off.min(),
                off_max=off.max(),
                off_ratio=off.max() / off.min(),
            )
        )

    r = pd.DataFrame(rows)
    print(
        f"\n  {'specimen':<14} {'(A) burden-avg':>15} {'(B) seeds regr':>15} "
        f"{'(C) mean of %':>14} {'baseline spread':>17}"
    )
    print("  " + "-" * 80)
    for _, x in r.iterrows():
        mark = ""
        if x.A_mean_burden_pct > 0:
            mark += " A-REGRESS"
        if x.B_seeds_regressed >= 2:
            mark += " B-REGRESS"
        print(
            f"  {x.spec:<14} {x.A_mean_burden_pct:+14.1f}% {int(x.B_seeds_regressed):>10}/3     "
            f"{x.C_mean_of_pct:+13.1f}% {x.off_ratio:14.1f}x  {mark}"
        )

    a_reg = r[r.A_mean_burden_pct > 0]
    b_reg = r[r.B_seeds_regressed >= 2]
    c_reg = r[r.C_mean_of_pct > 0]

    print("\n  " + "=" * 80)
    print("  REGRESSION COUNTS UNDER EACH READING")
    print(f"    (A) burden-averaged  : {len(a_reg)}/10 regress  -> {sorted(a_reg.spec)}")
    print(f"    (B) majority of seeds: {len(b_reg)}/10 regress  -> {sorted(b_reg.spec)}")
    print(f"    (C) mean of per-seed%: {len(c_reg)}/10 regress  -> {sorted(c_reg.spec)}")

    print("\n  The deliverable reports (A): 2/10 regress, i.e. a ~20% regression rate.")
    print(
        "  Under (B) the rate is %d/10. The two disagree on: %s"
        % (len(b_reg), sorted(set(b_reg.spec) ^ set(a_reg.spec)))
    )

    print("\n  WHY THEY DISAGREE — baseline burden varies hugely between seeds on the")
    print("  specimens where the readings differ, so (A) is decided by one seed:")
    for spec in sorted(set(b_reg.spec) ^ set(a_reg.spec)):
        g = df[df.spec == spec]
        print(f"\n    {spec}:")
        for _, x in g.iterrows():
            flag = "REGRESS" if x["total_penetration_burden_on"] > x["total_penetration_burden_off"] else "improve"
            print(
                f"      seed {int(x['seed']):>2}: off {x['total_penetration_burden_off']:6.2f} -> "
                f"on {x['total_penetration_burden_on']:6.2f}  ({x['pct_change']:+7.1f}%)  {flag}"
            )
        off = g["total_penetration_burden_off"].values
        print(
            f"      -> baseline burden varies {off.min():.2f}..{off.max():.2f} "
            f"({off.max() / off.min():.1f}x) across seeds; the largest-baseline seed sets the (A) verdict"
        )

    # how noisy is the per-seed measurement overall?
    print("\n  " + "=" * 80)
    print("  PER-SEED NOISE ON THE COMBINED METRIC")
    sd = df.groupby("spec")["pct_change"].std()
    print(
        f"    per-specimen sd of the per-seed % change: median {sd.median():.0f} pp, "
        f"max {sd.max():.0f} pp ({sd.idxmax()})"
    )
    print("    A single-seed check on a new specimen is therefore not decisive -- which the")
    print("    deliverable also states. Worth carrying into the automatic flag's design:")
    print("    a 1-seed flag will mislabel specimens at this noise level.")


if __name__ == "__main__":
    main()
