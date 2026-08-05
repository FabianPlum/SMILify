# Experimental defaults for this branch

**These are branch-local testing defaults. Nothing here is proposed for the live pipeline
yet** — `fitter_3d/ants_cfg.yaml` and `fitter_3d/trainer.py` remain untouched.

## `scheme: 'all'` instead of `scheme: 'deform'` in late stages — adopt for all new arms

Every new experiment on this branch should use `scheme: 'all'` in the stages that the stock
config runs as `scheme: 'deform'`. Rationale, measured:

* The stock late stages optimise `deform_verts` only, so 2000 of 2400 iterations cannot
  change a joint angle (measured per-stage `joint_rot` delta: exactly `0.00000`).
* Switching to `'all'` costs nothing — same iterations, same learning rates, same weights —
  and improved **every** metric simultaneously (`A4_nofreeze`).
* **Replicated over 3 seeds**, which matters because most of the effects are small:

  | metric (sign flipped so + is better) | mean ± sd over 3 seeds | verdict |
  |---|---|---|
  | deform_mag_mean | +23.52% ± 0.05 | solid |
  | part_leg_distal | +7.16% ± 1.19 | solid |
  | edge_logratio | +2.52% ± 0.15 | solid |
  | tri_quality | +1.36% ± 0.09 | solid |
  | fscore@0.01 | +0.91% ± 0.29 | solid |
  | fscore@0.02 | +0.46% ± 0.17 | solid |
  | chamfer_l2 | +2.84% ± 2.12 | **inconclusive — sign flips across seeds** |

Treating it as a testing default means later arms are not competing against a handicap that
is already known to be removable for free. The control arm (`C0_control`) deliberately keeps
`'deform'`, so the effect stays measurable rather than being silently absorbed.

**Why it is not shipped:** it has only been validated on the 50-specimen bench, and the
downstream consumers of these registrations (shape-space building, the Blender addon) have
not been re-checked against a fit whose pose keeps moving in the late stages. That is a
separate go/no-go.

## Hierarchical fitter flags now available

Both added after probe 13 measured *why* M7's distal legs still regressed:

* `--split_distal` — give each leg a separate distal group (tibia/tarsus/pretarsus).
  Proximal segments hold **94.0%** of a leg chain's surface area and the distal tip **2.4%**,
  so a single per-chain data term leaves the tip unconstrained: at `n_sample=8000` the
  pretarsus expects **~0.3 samples**, against the code's own `n_t < 10` skip threshold.
  Splitting gives a 2.4%-of-area structure a 50%-of-leg vote. 7 groups → 13.

* `--part_robust <mult>` — per-part robust kernel whose scale is `mult ×` that part's own
  template thickness, instead of one global scale. Measured thickness: body `0.117`,
  distal legs `0.008–0.012` — a **14.2×** ratio. This is the direct fix for finding 4
  (robust kernels read thin structures as outliers at 36× the body's rate).

**Note on the superseded idea:** the earlier next-step list proposed *coupling mirrored leg
chains*. Probe 13 rules that out — the template is left/right symmetric to **<0.3%** by area,
so a mirror-coupling term has almost nothing to correct, and coupling two equally
under-constrained tips constrains neither. The distal problem is intra-leg weighting, not
left/right.
