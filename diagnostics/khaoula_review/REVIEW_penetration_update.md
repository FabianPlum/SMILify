# Review — Khaoula's updated penetration-loss deliverable (2026-08-05)

Read in full: `penetration_loss_deliverable/README.md`, `1_penetration_weight_reduction_fix/`
(README, `PIPELINE_INTEGRATION.md`, `penetration_regression_flags.csv`,
`gentle_shrink_aggregate.csv`, `gentle_shrink_per_specimen.csv`,
`stage3_tail_trend_summary.csv`), and the new `penetration_visual_evidence/` folder
(README + `comparison_table.csv`).

Numbers below are **recomputed from her raw CSVs**, not copied from her prose —
`verify_penetration_claims.py` in this directory reproduces all of it.

---

## 1. What changed since the previous version

This is a substantial and honest **downgrade** of the earlier claim, and the revision is
the good kind: she found a distinction that weakens her own result and led with it.

| | previous version | updated version |
|---|---|---|
| headline | "76% of collision benefit for 27.5% of the quality cost, sign test p=0.00195, 10/0" | severity and count separated; count does **not** reliably improve |
| recommendation | ship the gentle weight | ship as an **available config option** with a required per-specimen check |
| regressions | not reported | 2/10 specimens net regress, consistent across seeds |
| new evidence | — | 5 fresh specimens (seed 42), never used in a prior study |

**The core new finding is the severity/count split**, and it is the most valuable thing in
the update: *the loss reliably makes contact shallower, but not reliably less frequent.*
The old "76%" figure described the effect's magnitude shrinking between weight settings —
not the count dropping below baseline. She says so explicitly. That was a genuine
misreading in the earlier version and she caught it herself.

**I verified the split on her independent 5-specimen set** (which she does not do — she
reports only the combined burden there):

| specimen | penetrating vertices | mean depth among them |
|---|---|---|
| Allomerus | 62 → 225 (**+263%**) | +4% |
| Eurhopalothrix | 648 → 921 (+42%) | −1% |
| Gnamptogenys | 303 → 253 (−17%) | **−58%** |
| Stigmatomma | 337 → 376 (+12%) | −27% |
| Tetramorium | 429 → 286 (−33%) | −25% |

Count rises on 3/5, depth falls on 4/5. **The split replicates on data she didn't use to
find it** — that is a real, independently-confirmed result.

---

## 2. Where I get a different number than she does

Her headline is **"8 of 10 specimens net improve, 2 regress"** → a ~20% regression rate.
That depends entirely on how "3-seed average" is computed, and the natural readings
disagree:

| reading | regressions | which |
|---|---|---|
| **(A)** average the burdens, then take % change — *her statistic* | **2/10** | GAGA-02-08, GAGA-04-06 |
| **(B)** majority of seeds regressed | **4/10** | + 22-45, 24-41 |
| (C) mean of the per-seed percentages | 3/10 | + 24-41 |

The disagreement is on **22-45** and **24-41**, and the cause is that total penetration
burden is an *unnormalised sum of depths*, so a seed with a large baseline dominates the
average:

```
24-41:  seed  0: off  2.99 -> on  5.64  ( +89.0%)  REGRESS
        seed  1: off  1.98 -> on  5.87  (+197.2%)  REGRESS
        seed 42: off 27.53 -> on  2.29  ( -91.7%)  improve
        baseline burden varies 13.9x across seeds — seed 42 alone sets the verdict
```

**Two of three seeds regress on 24-41, and her table records it as a −57.5% improvement.**
She flags the inconsistency in her own table ("no — seeds range +197% to −92%"), so this is
not hidden; but the headline count of 2 is the optimistic reading of her own data, and the
20% regression rate should probably be quoted as **20–40% depending on aggregation**.

Per-seed noise on the combined metric is large: median sd **38 pp**, max **146 pp** (24-41).

### Why this matters for her recommendation

Her proposed gate is an **automatic per-specimen regression flag**, computed "as part of the
same run that produces the fit". That run is single-seed. At a per-seed sd of 38–146 pp, a
single-seed flag will mislabel specimens in both directions — it would have cleared 24-41 on
seed 42 and condemned it on seeds 0 and 1. **The automatic flag needs to be multi-seed (or
to use a normalised, less outlier-sensitive burden metric) to do the job she wants from it.**
This is the one thing I'd push back on before it ships.

---

## 3. Two smaller checks

**"Fit quality is unaffected — under 3% on every specimen."** Nearly right: the largest
move is Eurhopalothrix at **−3.3%** on fscore@0.01, marginally outside her stated bound.
Chamfer moves ≤3.0% everywhere. The claim holds in substance; the bound is slightly
overstated.

**Compute cost is not reported.** From her own `comparison_table.csv`, turning the loss on
costs **+26% wall time overall**, per specimen: +33%, +30%, +19%, −19%, **+115%**. That is
worth stating in a deliverable that recommends enabling something.

---

## 4. The most useful thing I can add: the mechanism is mis-attributed

Her explanation for the regressions is:

> "the resulting change in **body pose** can also bring previously-separate geometry into
> new contact elsewhere"

**Pose cannot change in the stages where this loss runs.** Her own `PIPELINE_INTEGRATION.md`
establishes the loss is applied at Stage 2 and Stage 3. In `fitter_3d/ants_cfg.yaml` both are
`scheme: 'deform'`, and `SMALParamGroup.param_map["deform"] == ["deform_verts"]`
([trainer.py:260](../../fitter_3d/trainer.py#L260)) — free-form vertex offsets only. Joint
rotations are frozen there. I verified this independently while diagnosing the same stages
for the registration work: measured per-stage change in `joint_rot` across Stages 2–3 is
exactly **0.00000**.

So what actually moves is not pose. **The penetration loss can only resolve a collision by
denting the mesh, never by rotating a limb away.**

That reframing explains her central finding rather than just restating it:

* **Denting reduces depth at the contact point** (severity improves — 9/10 specimens ✓)
* **Denting pushes the surrounding shell into new shallow contact** (count rises — the
  measured signature, on a majority of specimens ✓)
* **Allomerus's "+263% count at one leg"** is exactly what a local dent looks like: it is
  concentrated, not distributed, because a vertex-offset field is local while a joint
  rotation is not.

A real inter-part collision in an ant is anatomically resolved by rotating the limb away.
The current setup structurally cannot do that, so it does the only thing available.

### Concrete joint experiment I'd propose

Run the gentle penetration weight with the deform stages set to `scheme: 'all'` instead of
`'deform'`, so joint rotations can move while offsets are being penalised.

This is free and independently justified: in my registration work that one-word change
(`A4_nofreeze`) improved every metric at once at zero cost, replicated over 3 seeds
(deform magnitude −23.52% ± 0.05). Combined with the penetration loss it makes a falsifiable
prediction:

> **If the count-vs-severity split is caused by denting, then allowing joint rotation should
> shrink or invert it** — collisions get resolved by rotation rather than local deformation,
> so the "new shallow contact around the dent" mechanism disappears.
> If the split survives unchanged, denting is not the cause and the explanation is wrong.

I can run this on my 50-specimen benchmark as soon as `penetration_loss.py` is available —
**it is not in the repo** (only a reference copy in her Drive folder), so nothing on this
branch can currently exercise it.

---

## 5. Summary

**Good:** the severity/count separation is a real finding, honestly self-corrected, and it
replicates on data that wasn't used to find it. The revised "config option, not default"
recommendation is the right call. Refusing to ship it unconditionally on a 1-in-5 regression
rate is good judgement.

**Push back on:** the single-seed automatic flag (her own noise figures say it can't work),
the 2/10 regression count (2–4/10 depending on aggregation, and her burden metric is
outlier-sensitive), the unreported +26% compute cost, and the "body pose" attribution.

**Highest-value next step:** the `scheme: 'all'` experiment above. It costs one run, tests
her mechanism directly, and it is the same change that independently improved my
registration results.
