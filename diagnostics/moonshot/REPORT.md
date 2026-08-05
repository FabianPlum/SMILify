# Registration Moonshot — what is actually established

Branch `feature/registration_moonshot`. Template `3D_model_prep/SMIL_OmniAnt.pkl`
(10,229 verts, 55 joints, 13 shape directions). Bench: 50 specimens sampled with seed
20260804 from the 757 `antscan_proofread_castes/worker` scans. Hardware 2× RTX 4090.
Sources and papers: [SOURCES.md](SOURCES.md). Superseded first draft:
[REPORT_v1_superseded.md](REPORT_v1_superseded.md).

**This document carries only claims that survived checking.** Several confident findings from
earlier drafts were refuted — by seed replication, by ablation, by adversarial audit of my own
code, and by the user. They are not repeated here as findings; they are collected in §7 as
lessons, because the *pattern* in how they failed is the most transferable thing in this work.

---

## TL;DR

1. **The stock pipeline reaches good surface numbers by destroying correspondence.** It
   freezes pose for 83% of its iterations (measured per-stage `joint_rot` delta: exactly
   `0.00000`), then distorts the template's average edge by 74% and destroys 36% of local
   vertex neighbourhoods. Registration is *for* correspondence, so this is failing at the task
   while scoring well on the proxy.
2. **`M7_handoff_midline` is the best arm found**, replicated over 3 seeds: chamfer
   **+60.68 ± 0.34%**, free-form deformation **+52.22 ± 0.14%**, edge distortion
   **+32.16 ± 0.17%**, midline symmetry **+44.40 ± 2.47%**, fscore@0.02 **+4.56 ± 0.05%**.
   Every sign consistent, every mean far outside between-seed noise.
3. **A one-word free win:** `scheme: 'deform'` → `scheme: 'all'` in the late stages improves
   almost everything at zero cost, replicated over 3 seeds (deform_mag **+23.52 ± 0.05%**).
   `chamfer_l2` is **inconclusive** for this arm — it flips sign across seeds.
4. **Every per-part metric in this suite is source→target proximity, and shrink-wrapping games
   it.** This invalidates how per-part regressions were reported throughout earlier drafts.
   See §3 — it is the single most important methodological point here.
5. **The parametric model is more capable than the pipeline lets it be.** Pose+shape alone,
   with zero free-form offsets, reaches fscore@0.02 = 0.7313 with *exactly zero* mesh
   distortion. Enlarging the shape space 13 → 37 directions moves that only to **0.7356**, so
   **shape capacity is not the binding constraint** — correspondence is.
6. **Outlier rejection is contraindicated.** Robust kernels and ratio tests preferentially
   delete thin structures: distal legs are read as outliers at 36× the body's rate and retain
   5.8× fewer correspondences.
7. **Worker mesh quality is modestly worse than the reference corpus, not catastrophically
   so** — and it is fixable. Cleanup brings 6 of 7 quality metrics to reference levels (§6).

---

## 1. Diagnosis of the stock pipeline

**Pose is frozen for 83% of the optimisation.** `fitter_3d/ants_cfg.yaml` runs four stages;
stages 2 and 3 use `scheme: 'deform'`, which optimises `deform_verts` only. Measured
per-stage change in `joint_rot`:

| stage | iterations | Δ joint_rot | Δ betas | Δ deform |
|---|---|---|---|---|
| Stage_0_init | 200 | 0.31 | 0.00 | 0.00 |
| Stage_1_pose_shape | 200 | 1.42 | 0.83 | 0.00 |
| Stage_2_deform_coarse | 1000 | **0.00000** | **0.00000** | 4.87 |
| Stage_3_deform_fine | 1000 | **0.00000** | **0.00000** | 2.11 |

2000 of 2400 iterations cannot change a joint angle. Whatever pose error exists after
iteration 400 is permanent, and the remaining 83% of compute is spent hiding it with
per-vertex offsets.

**What that costs**, measured directly (probe 10, `fitter_3d/correspondence.py`): mean edge
distortion vs template **74%**, local vertex neighbourhoods destroyed **36%**, free-form
offset magnitude **2×** that of the best arm.

**Pose is trapped, not under-optimised.** Arm E1 raised the pose learning rate and budget; the
result was *worse than its own initialisation*. Increasing step size in a multimodal objective
— six substitutable legs — moves you between basins, not down. This remains the core
structural finding.

**Three code defects found by reading, all real:**
- `betas_prec` (Cholesky of the inverse shape covariance) is computed and never used — the
  shape prior was inactive.
- The edge regulariser compared against the *original template*, penalising legitimate pose
  change rather than distortion.
- Target points were sampled by vertex rather than by area, so densely-tessellated regions
  dominated the data term.

---

## 2. What works

### 2.1 `M7_handoff_midline` — the recommended arm, 3 seeds

Hierarchical part-anchored pose with a midsagittal-planarity constraint, handed to the
baseline's own deform budget with `scheme: 'all'` and a gentle offset penalty.

| metric (sign flipped so + is better) | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| chamfer_l2 | +61.03 | +60.78 | +60.23 | **+60.68 ± 0.34** |
| deform_mag_mean | +52.37 | +52.24 | +52.03 | **+52.22 ± 0.14** |
| edge_logratio | +32.33 | +32.22 | +31.93 | **+32.16 ± 0.17** |
| midline_dev | +46.47 | +45.80 | +40.93 | +44.40 ± 2.47 |
| fscore@0.01 | +7.06 | +6.63 | +6.71 | +6.80 ± 0.19 |
| fscore@0.02 | +4.62 | +4.52 | +4.53 | **+4.56 ± 0.05** |

**How the midline term was found.** `M5_handoff` (same arm without it) won 8 of 11 metrics but
regressed bilateral midline deviation by **+43.8%**. Diagnosis: `symmetry_penalty` ties
left/right joint *scales*, which cannot see an out-of-plane rotation of a body joint — that
moves midsagittal vertices off y=0 while leaving every scale pair identical. A four-line
planarity penalty on the 223 `sym_verts` (penalising the *variance* of their y, so a
legitimate global translation costs nothing) swung it from +43.8% worse to −46.5% better.

**The caveat, which matters (§3):** M7 is *worse* than baseline on every per-part
`within_tau` — legs 0.9541 vs 0.9843, body 0.9549 vs 0.9876, head 0.9554 vs 0.9897. That is
not evidence M7 is worse. Those metrics measure how close fitted vertices sit to the target
surface, which shrink-wrapping optimises directly, and the baseline deforms twice as much.

### 2.2 `A4_nofreeze` — free, replicated

`scheme: 'deform'` → `'all'` in the late stages. Same iterations, learning rates and weights.
Over 3 seeds: deform_mag **+23.52 ± 0.05**, part_leg_distal +7.16 ± 1.19, edge +2.52 ± 0.15,
tri_quality +1.36 ± 0.09, fscore@0.01 +0.91 ± 0.29, fscore@0.02 +0.46 ± 0.17. `chamfer_l2`
**flips sign across seeds (+2.84 ± 2.12) and is reported as inconclusive.**

Adopted as a branch-local testing default (`cfg/EXPERIMENTAL_DEFAULTS.md`), not shipped — the
downstream consumers of these registrations have not been re-checked against a fit whose pose
keeps moving late.

### 2.3 The model's own ceiling

Pose+shape only, zero offsets (`M4b_ceiling_free`): fscore@0.02 = **0.7313** with *exactly
zero* edge distortion and zero deformation. Visually this is the fit that looks like an ant.
The baseline reaches a higher surface score only by deforming.

---

## 3. What the metrics do and do not measure

**This section supersedes how per-part results were reported in every earlier draft.**

| kind | metrics | what it rewards |
|---|---|---|
| **surface proximity** | chamfer, fscore@τ, hausdorff, *all* `part_*_dist_mean`, *all* `part_*_within_tau` | getting vertices near the target surface — **by any means, including destroying the mesh** |
| **correspondence integrity** | edge_logratio, deform_mag, neighbourhood preservation, tri_quality, dihedral/folding | keeping the template's own structure while it moves |

Two demonstrations that the first kind is gameable:

* Artificially snapping every distal vertex onto its nearest target point improves
  `part_leg_distal_dist_mean` from 0.01151 to 0.00538 — a 53% "gain" — while `edge_logratio`
  explodes from 0.073 to **0.552** and the mesh is destroyed.
* `within_tau` is bounded and outlier-resistant, so it is the better of the two, but it is
  *still* one-way. The baseline beats M7 on every part's `within_tau` precisely because it
  shrink-wraps.

**Consequence.** A per-part proximity regression is not by itself evidence of a worse
registration, and earlier drafts treated it as such. Where an arm improves
correspondence-integrity metrics and loses per-part proximity, the honest reading is that it
stopped shrink-wrapping. Where an arm loses *both* — as the part-field arms do (§5.3) — the
result is unambiguous.

**Still missing:** a metric that measures correspondence *correctness* rather than
correspondence *stability*. Nothing here distinguishes "leg 2 is registered to leg 2" from
"leg 2 is registered to leg 3 without tearing". That gap is why §5.3's failure is hard to
localise, and closing it is the highest-value instrument work remaining.

---

## 4. Structural facts about the template and data

* **Leg-chain area budget** (probe 13): coxa+trochanter+femur hold **94.0%** of a leg chain's
  surface area; tibia+tarsus+pretarsus hold **2.4%**. At `n_sample=8000` the pretarsus expects
  **~0.3 samples** against the code's own `n_t < 10` skip threshold. Distal segments are
  effectively unconstrained by an area-sampled data term.
* **Bilateral symmetry** (probe 13): the template is left/right symmetric to **<0.3%** by area,
  and its mirror involution is exact (max match distance 0.00e+00). A mirror-coupling term has
  almost nothing to correct.
* **Thin-structure bias** (probe 09): at robust scale c=0.10, distal legs are read as outliers
  at **36×** the body's rate and retain **5.8× fewer** correspondences under mutual-NN + ratio.
* **Shape capacity is not binding** (probe 08, M6): 13 → 37 shape directions, both deform-free,
  moves fscore@0.02 only **0.7313 → 0.7356**. `posedirs` is literally shape `(0,)` — the
  pose-corrective blendshape slot is empty — and folding concentrates across joint boundaries
  (6.33%) versus within parts (1.48%).
* **Alignment**: worker scans are canonically aligned — PCA-1 yaw median **1.0°**, 90th pct
  9.4° — with exactly one 62° outlier (*Atopomyrmex mocquerysi*).

---

## 5. What was tried and did not work

### 5.1 Robust kernels and correspondence filters
Geman–McClure at a scale matched to the residual gave fscore@0.01 **+13.1%** at the pose
checkpoint (48/50, p=2e-12), but at too small a scale it starved the gradient (83% of points in
the kernel's zero-gradient region). Mutual-NN + Lowe ratio and Sinkhorn OT were implemented
(`fitter_3d/correspondence.py`) and all share the thin-structure bias above. **Per-part robust
scaling** (`--part_robust`, scale ∝ each part's own template thickness, measured 14.2× ratio
body-to-distal) was the principled fix and produced no distal gain.

### 5.2 Geodesic branch decomposition
A target-derived partition from the scan's own branching topology. On the template it separates
all six legs perfectly in 0.2 s. On the 50 real scans it reaches ≥6 branches on **58%** against
its own pre-registered 80% gate, and agrees with itself across two surface samplings only
**20%** of the time. Legs touching the body merge into one component and no threshold separates
them. **Rejected on its own gate.**

### 5.3 The learned part field
A PointNet++ segmentation network labelling raw scan points into 13 anatomical parts + debris,
trained on synthetic scans posed from the model itself (labels free from `weights.argmax`),
used as a **frozen, target-derived** replacement for the fit-derived partition.

*What it achieves:* held-out per-point accuracy **91.28%** against **24.83%** for an
anatomy-free baseline. On the four specimens where the geodesic sweep collapsed to ≤2 branches
(three to a *single* branch — limbs touching, no bottleneck to cut) it recovers **6/6 legs on
every one**. That capability difference from a level set is real.

*What it costs:* substituted into the fitter it loses decisively. Against the byte-identical
control `M8a_split_only` (differing only in `--part_field`), at `H3_deform`, n=50:
part_leg_distal within_tau **−18.5%** (8/50, p=1.2e-06), chamfer −47.6%, edge_logratio −22.9%,
fscore@0.02 −5.0%. It loses on *both* metric families, so §3's caveat does not rescue it.
Partition churn was 0.000 in every stage exactly as designed — the data term became perfectly
stationary around a worse answer.

*Three sub-hypotheses tested and refuted:*
- **Debris exclusion is the mechanism.** The class is genuinely mislearned — 90.8% of predicted
  debris lies *inside* the animal's convex hull, and its per-specimen fraction correlates
  **−0.879** with reproducibility, i.e. it is the network's "I don't know" channel, not a
  debris detector. But reassigning debris instead of dropping it (`M9d_keepdebris`) changes
  **nothing** (all p > 0.1) and still loses −19.8% to control.
- **Restricting the H0 body stage to the field's body points** (`--pf_body`, impossible with a
  fit-derived partition since it does not exist until a fit does): no effect.
- **Correspondence-free part-anchor pose initialisation** (`--pf_init`, matching per-part
  centroid and second moment, an objective with no nearest-neighbour choice in it): after
  fixing a serious bias in it (§7.9), it buys `edge_logratio` **+5.1%** (42/50, p=1.2e-06) and
  nothing else.

*Reading.* Circularity in the fit-derived partition is a real flaw **and also an
error-correcting mechanism**: a point assigned to the wrong leg gets reassigned once the fit
moves. A frozen assignment cannot recover, so its errors are permanent and its uncertainty
becomes deletion or wrong commitment rather than reduced weight. Target-derived is defensible;
*frozen* is the wrong way to use it.

### 5.4 Enlarging the shape space
One co-registration round over 25 specimens (append 24 PCA directions learned from offset
fields, refit pose+shape only) gave held-out gains in part placement, **did not close the
surface gap**, and produced badly folded geometry: **2.79%** of adjacent face pairs folded past
90° versus **0.54%** for the baseline. PCA directions fitted to displacement fields are
least-squares optimal and carry no guarantee the mesh they generate is valid. A fold-gated,
symmetrised, Laplacian-smoothed basis is implemented (`run_coregistration.py`) but the
multi-round run over all 757 workers has **not been executed**.

---

## 6. Mesh quality — measured, not assumed

40 random meshes from each corpus, both normalised as `load_meshes` does.

| | ALL_ANTS_CLEAN | worker |
|---|---|---|
| vertex-connected components (median) | 3 | **34.5** |
| area fraction off main component | 0.0000 | 0.0030 |
| boundary-edge fraction (holes) | 0.0388 | 0.0471 |
| non-manifold edge fraction | **0.0709** | 0.0271 |
| triangle quality (median) | 0.8635 | 0.8322 |
| edge-length CV | 0.3000 | **0.4311** |
| median dihedral (roughness) | 8.20° | **10.54°** |
| faces with dihedral > 60° | 0.0504 | 0.0636 |

Workers carry ~10× more connected components but those hold only **0.3% of the area** — small
debris, not fragmentation — and workers are *cleaner* than the reference corpus on non-manifold
edges. They are ~30–45% rougher and more irregular. **A real gap, but not orders of magnitude.**

`preprocess_meshes.py` closes most of it — merge duplicates, drop components below an area
threshold, fill holes, Taubin smooth (volume-preserving, so thin distal structures are not
eroded):

| metric | before | after | reference |
|---|---|---|---|
| vertex components | 44.5 | **4.0** | 3.0 |
| boundary edges | 0.0554 | **0.0302** | 0.0388 |
| non-manifold edges | 0.0365 | **0.0053** | 0.0709 |
| triangle quality | 0.8336 | **0.8603** | 0.8635 |
| median dihedral | 10.70° | **8.27°** | 8.20° |
| rough faces | 0.0704 | **0.0532** | 0.0504 |
| edge-length CV | 0.4709 | 0.4294 | 0.3000 |

### 6.1 And it does not help — the A/B result

`P1_prepclean` is `M8a_split_only` re-run on the cleaned meshes: same 50 specimens, same
poses, same shape space, same seed, same schedule, one variable. Scored **twice**, against the
cleaned target and against the raw target, so that an easier scoring target cannot be mistaken
for a better registration. That precaution was decisive — **the sign flips**:

| metric | vs CLEANED target | vs RAW target |
|---|---|---|
| fscore@0.02 | **+6.6%** (38/50, p=3.1e-04) | **−6.4%** (18/50, p=6.5e-02) |
| fscore@0.01 | +7.2% (35/50) | −9.1% (17/50) |
| chamfer_l2 | +6.2% (33/50) | −6.2% (13/50, p=9.4e-04) |
| part_leg_distal within_tau | **+11.4%** (35/50) | **−15.1%** (9/50, p=5.6e-06) |

The target-independent metrics — identical under both scorings, because they describe the fit
rather than its agreement with a target — are the ones that settle it:

| metric | Δ | wins | p |
|---|---|---|---|
| edge_logratio | **−2.9%** (more distortion) | 15/50 | 6.6e-03 |
| deform_mag | **−2.9%** (more deformation) | 12/50 | 3.1e-04 |
| tri_quality | +0.0% | 23/50 | ns |
| midline_dev | +5.2% | 27/50 | ns |

**Conclusion: mesh preprocessing is not what prevents these scans from being fitted.** Cleaning
makes the target easier to score against while producing a marginally *worse* registration by
every measure that does not depend on the target. Scored one way only, this would have been
reported as a 6.6% win.

That removes preprocessing from the candidate list. Combined with §8's pattern — six
correspondence-side interventions, all null — what remains is **pose**.

**A control that cannot answer whether this matters.** Fitting ALL_ANTS_CLEAN and observing
that it works is **confounded**: OmniAnt's own shape space was built with the baseline workflow
on that exact corpus, so those scans are in-distribution by construction. It bounds what the
pipeline can do and says nothing about generalisation. The irreducible difference is **pose** —
ethanol-preserved specimens are contracted and folded, the reference corpus is posed — and that
is not recoverable by preprocessing. Mesh quality is the part that *can* be targeted.

---

## 7. Claims of mine that were wrong

Kept because the pattern matters more than the individual errors: **every one produced
good-looking numbers, and every one was caught by building an adversary rather than by
inspecting results.**

1. **"The scans are not roll-canonical."** PCA on the old `half_workers` said so; the user said
   otherwise; a visual check confirmed the user. Surviving fragment: chamfer prefers an
   anatomically wrong roll for 68% of specimens.
2. **`w_beta_prior` mis-scaled by ~2 orders of magnitude**, silently freezing shape (|z| 0.01
   vs 0.37 unpriored). `betas_prec` is the Cholesky of the *inverse* covariance. Invalidated
   the first ceiling measurement (0.377 → corrected 0.731).
3. **The rest-edge term compared to the original template**, penalising legitimate pose change;
   it reached 3.53 against a chamfer of 0.005, a 35× imbalance.
4. **A broken churn diagnostic** compared freshly-resampled points and reported ~0.65 regardless
   of what the fit did. Fixed with a frozen probe set; real churn 0.062 → 0.005.
5. **"M5 fragments the mesh."** Read off renders. The tearing metric over 20 specimens said M5
   was *cleaner* than baseline.
6. **"Coupling mirrored leg chains will fix the distal regression."** My own recommendation,
   refuted by probe 13 before implementation: the template is L/R symmetric to <0.3% by area.
7. **"Per-part regressions show arm X is worse."** All per-part metrics are source→target
   proximity and are gamed by shrink-wrapping (§3).
8. **"The part field's gates validate it."** An anatomy-free rule — `label = x-tercile ×
   sign(y)`, which paints the gaster as legs — **beats** the trained field on every unsupervised
   gate (G1 98.65 vs 91.09, G2 100.00 vs 87.81, G3b 100.00 vs 99.43, G3c 91.94 vs 85.61) and
   fails only the supervised one. G3b is not even monotone in correctness: *true* labels score
   99.14%, below the trivial rule. I wrote in that probe's docstring that G3b "is a genuine test
   of whether the field means anything." It is not.
9. **"A perfect anchor initialisation is erased by the chamfer stages."** The initialisation was
   *biased*, not perfect: source moments over template **vertices** versus target moments over
   **area-sampled** points disagree even at a perfect fit (loss 5.49e-04, body centroid off
   7.93% of extent). The arm converged to 3.4e-05 — **16× below that floor** — i.e. it was
   deforming the model to chase a sampling artefact (chamfer 105× worse, mandibles rotated
   52–94°, joint scales 0.15…11.7). Fixed by barycentric area weighting (→ 1.48e-05).
   **§1's trapped-pose question is therefore still open** — that experiment did not test it.
10. **"Debris exclusion explains the part field's failure."** Refuted by its own ablation.
11. **"Workers have thousands of connected components versus the reference's three."** An
    artifact of loading with `process=False`: face-adjacency counts unshared vertices as
    disconnection. Both corpora show ~1500–3000 by that measure. The correct vertex-connected
    figures are 3 vs 34.5.
12. **Synthetic poses were interpolated by lerping axis-angle vectors**, which is not a rotation
    interpolation. Distal joints ended up bent to only **81–88%** of real angles, and 1.14% of
    pose pairs deviate by 63.6°. Partly self-inflicted distal weakness in the part field.

---

## 8. Where this leaves things

**Established and reusable:** the diagnosis (§1), M7 and A4 (§2), the structural facts (§4),
the mesh-quality baseline and cleanup (§6), and the metric critique (§3) — which invalidates a
class of conclusion, including several of my own.

**The recurring shape of every failure.** Partitioning, robust kernels, correspondence filters,
geodesic decomposition, the learned part field, shape-space enlargement — all attacked
*correspondence*, and none moved the surface result much. The two things that did work (M7, A4)
both work by letting **pose keep moving**. That is a consistent signal, and it points at pose,
not at the data term.

**Next, in order:**
1. **The preprocessing A/B** (§6) — cheapest remaining decisive experiment.
2. **A soft partition** — per-point class probabilities as chamfer *weights* rather than a hard
   argmax, so uncertainty reduces influence instead of causing deletion or wrong commitment.
   This is §5.3's conclusion, and it matches the EM formulation used for robust statistical
   shape model fitting (E-step: posterior that each point is explained by the model; M-step:
   weighted fit), which is self-correcting where the frozen partition was not.
3. **A correspondence-correctness metric** (§3), without which arms cannot be separated on the
   thing that matters.
4. **Quality-gated evaluation**: rank the 757 workers by intrinsic scan features and report on
   the fittable subset explicitly, rather than averaging over specimens no parametric fit could
   recover.
5. **`posedirs` is empty.** Pose-corrective blendshapes are the standard fix for exactly the
   joint-boundary folding measured in §4, and the slot has never been populated.

---

## 9. Reproducing

```bash
python diagnostics/moonshot/make_benchmark.py              # bench set (symlinks + manifest)
diagnostics/moonshot/queue_runner_v2.sh 0 M5_handoff       # one arm, then score it
python diagnostics/moonshot/compare_arms.py --control C0_control   # paired sign tests
python diagnostics/moonshot/preprocess_meshes.py           # mesh cleanup + before/after report
python diagnostics/moonshot/plot_summary.py                # figures
```

Fit outputs (~2.9 GB of `.npz`) are gitignored; the scored `metrics.csv` per arm is kept, so
every number above is checkable without re-running. The live training path
(`fitter_3d/trainer.py`, `utils.py`, `optimise.py`) is **unmodified**; all experiments live in
new files. The single change to `config.py` is an opt-in `SMILIFY_SMAL_FILE` override, needed
because `N_BETAS` is derived at import.

**Note on tracking:** `probe_*.py` is gitignored repo-wide by a pre-existing rule, so most probe
scripts referenced above are on disk but not in git. Their JSON/PNG outputs *are* tracked, so
the evidence is versioned even where the code is not.
