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
4. **`scaledirs` — the model's per-joint scale blendshapes — is never used by the fitter.**
   Instead 165 free per-specimen `log_beta_scales` carry 70–400× more inter-specimen variation
   than the shape space does, so shape is stored in a pose-space parameter that cannot
   transfer. One defect explaining four separate observations, and concretely fixable (§6.6).
5. **The registrations contain no learnable shape structure.** A leave-one-out shape model
   built from them reconstructs a held-out specimen no better than the population mean does
   (gen/spread 0.99–1.00), while the same metric reaches 0.42 at 10 modes and 0.00 at 20 on
   synthetic data with exact correspondence. This pipeline produces *surface fits*, not
   *registrations*, and no amount of additional corpus fixes it (§6.4).
5. **Every per-part metric in this suite is source→target proximity, and shrink-wrapping games
   it.** This invalidates how per-part regressions were reported throughout earlier drafts.
   See §3 — it is the single most important methodological point here.
5. **The parametric model is more capable than the pipeline lets it be.** Pose+shape alone,
   with zero free-form offsets, reaches fscore@0.02 = 0.7313 with *exactly zero* mesh
   distortion. Enlarging the shape space 13 → 37 directions moves that only to **0.7356**, so
   **shape capacity is not the binding constraint** — correspondence is.
6. **Outlier rejection is contraindicated.** Robust kernels and ratio tests preferentially
   delete thin structures: distal legs are read as outliers at 36× the body's rate and retain
   5.8× fewer correspondences.
7. **Mesh preprocessing is not the blocker.** Cleaning worker meshes to reference quality on
   6 of 7 axes makes the *scoring target* easier while making the registration marginally
   worse on every target-independent metric (§6.1).
8. **Fittability is predictable from the scan alone.** `radial_med` (surface distance from the
   principal axis) predicts fit quality at rho 0.80–0.81, validated on held-out specimens
   (deform_mag **+38.9%**, p=4.5e-13). The reference corpus sits at the **6.6th percentile**
   of the worker distribution — the workers are more compact than anything the template was
   built to represent. Restricting to the top 50% gives fscore@0.01 **+14.7%**, deform_mag
   **−19.3%**, chamfer **−35.8%** (§6.2).

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

## 6.2 A scan-only fittability gate that works

Since preprocessing is not the lever, the next question is whether *which specimen* is
predictable in advance. It is.

**The score.** `radial_med` — the median distance of area-sampled surface points from the
specimen's principal axis. No template, no fit, no labels.

**Discovery** (bench50, n=50, Spearman): rho = **+0.81** vs chamfer_l2, **+0.80** vs
deform_mag, **+0.66** vs edge_logratio. Consistently signed across *both* metric families,
which matters because §3 rules out judging this on surface proximity alone.

A competing hypothesis was tested first and **refuted**: self-contact (limbs touching, the
configuration that collapsed the geodesic method) correlates −0.29 with deform_mag, i.e.
contact predicts marginally *better* fits.

**Not a normalisation artefact.** `radial_med` is computed in a frame divided by max|coord|,
so it could merely track how far the extremities protrude. The frame-free ratio
radial/extent — a plain aspect ratio — predicts nearly as well (rho +0.75 / +0.74 / +0.61).
The signal is specimen **elongation**: stubby species are far from an elongated template.

**Held-out validation.** The correlation was found on bench50, so the gate was tested on two
disjoint 50-specimen sets drawn from the other 707 — `easy` (lowest radial_med, median 0.139)
and `rand` (uniform, median 0.244) — fitted with an identical recipe, with the reading fixed
before the run. Unpaired Mann-Whitney:

| metric | easy | rand | easy better by | p | family |
|---|---|---|---|---|---|
| deform_mag | 0.01307 | 0.02141 | **+38.9%** | 4.5e-13 | target-independent |
| edge_logratio | 0.33683 | 0.39013 | **+13.7%** | 5.6e-07 | target-independent |
| midline_dev | 0.01240 | 0.01577 | +21.4% | 4.9e-02 | target-independent |
| tri_quality | 0.68407 | 0.67451 | +1.4% | 1.8e-02 | target-independent |
| fscore@0.01 | 0.86864 | 0.62836 | **+38.2%** | 1.6e-14 | surface |
| chamfer_l2 | 0.00014 | 0.00035 | +58.8% | 2.3e-12 | surface |
| hausdorff_95 | 0.01763 | 0.02567 | +31.3% | 3.9e-09 | surface |
| normal_consistency | 0.79731 | 0.80914 | −1.5% | ns | surface |

It wins on **both** families, so the pre-registered "size artefact" branch is excluded. Sanity
check: `rand`'s fscore@0.02 (0.9166) reproduces bench50's M7 (0.9246), as a uniform draw
should.

**Quality versus coverage.** Pooling all 146 fitted specimens that have a rank (bench50 +
both held-out sets) and cutting at corpus percentiles of `radial_med`:

| coverage | cutoff | fscore@0.01 | deform_mag | edge_logratio | chamfer_l2 |
|---|---|---|---|---|---|
| top 10% | 0.163 | 0.8666 | 0.01325 | 0.3393 | 0.000145 |
| top 25% | 0.196 | 0.8490 | 0.01385 | 0.3387 | 0.000156 |
| **top 50%** | **0.237** | **0.8161** | **0.01494** | 0.3436 | 0.000176 |
| top 75% | 0.285 | 0.7756 | 0.01636 | 0.3535 | 0.000212 |
| all | 0.466 | 0.7113 | 0.01851 | 0.3689 | 0.000274 |

Monotone in every metric. **Top 50% versus the whole corpus: fscore@0.01 +14.7%, deform_mag
−19.3%, chamfer −35.8%.** This table is a pooled estimate across three differently-sampled
sets, not a single controlled experiment; the controlled statement is the easy-vs-rand test
above.

**Why the score means something.** ALL_ANTS_CLEAN — the corpus OmniAnt's shape space was
built from — has median `radial_med` **0.1521**, which is the **6.6th percentile** of the
worker distribution (worker median 0.2367). The workers are systematically more compact than
anything the template was constructed to represent, and `radial_med` measures exactly that
axis. The gate is not selecting "good scans"; it is selecting **specimens the existing shape
space can reach**.

That reframes the recommended operating point: register the top 50% by `radial_med`, use those
registrations to rebuild the shape space, and only then reconsider the remainder — because a
shape space built from the current all-corpus fits is a shape space built substantially from
failures, which is the most likely explanation for the folded geometry in §5.4.

---

## 6.3 A shape space rebuilt from gated registrations

`build_shape_space.py` is the headless equivalent of `smil_importer/pca.py`'s
`apply_pca_and_create_shapekeys`: PCA over registrations in correspondence, emitting a mean
shape, principal components as blendshapes, and the covariance of the projected betas.

**What is PCA'd, and why it differs from §5.4.** The correct object is each specimen's
*rest-space shaped geometry*, `v_shaped_i = v_template + deform_verts_i + shapedirs·betas_i`,
which is exactly what `smal_torch.__call__` composes before skinning — so `deform_verts` lives
in rest space and this expression is the specimen's shape with pose removed. §5.4 instead
PCA'd raw offset *fields*, mixing shape with whatever the offsets were doing to hide pose
error. (`log_beta_scales` and `betas_trans` act during skinning and are pose-space, so
limb-length variation expressed through them is **not** captured here — a real limitation.)

**Two bugs caught while building it**, both of the kind §7 is about:
* The mirror map was initially derived by matching a *fitted specimen* against the template
  (max match distance 7.68e-02) rather than from the template's own involution. Corrected, the
  involution is exact — 0.00e+00, valid on 100% of vertices — and the mean shape's folding fell
  from 2.285% to **0.675%**.
* The fold tolerance was initially the original template's 0.540%, which rejected 49 of 50
  directions — a standard the *mean of the registrations* (0.675%) already fails. Setting it
  from the data instead (p90 of the inputs) is the defensible requirement: the shape space must
  not generate meshes worse than the registrations it was built from.

**The finding that matters.** Measuring the input registrations' own validity:

| | folded adjacent-face fraction |
|---|---|
| original template | 0.540% |
| gated registrations, median | **2.888%** |
| gated registrations, p90 | 3.602% |
| gated registrations, max | 3.954% |
| **mean of the registrations** | **0.675%** |

Even the best-gated registrations produce rest-space geometry folding **5.3× more than the
template**. The free-form offsets are introducing self-intersecting geometry, in *every*
specimen, not just the failures. But averaging removes almost all of it (0.675%), which says
the folding is specimen-specific noise rather than shared shape — and is why a mean-shape
plus fold-gated basis is recoverable at all.

**Result.** 20 directions retained, all within the data-derived tolerance, written to
`3D_model_prep/SMIL_OmniAnt_gated.pkl` and verified to load and drive (`N_BETAS=20`, PC1 at
±3σ gives 0.089 max vertex displacement). The spectrum is **flat** — PC1 13.1%, PC1–10 50.8%,
41 components for 95% — the same high-dimensional residual §5.4 found, which remains the
strongest argument that these offsets encode correspondence error rather than biological shape.

*Not yet done:* refitting with this model to test whether it improves anything. Given §4's
finding that shape capacity is not the binding constraint, the prior should be that it will
not, and the value of this artefact is mainly as the input to a second co-registration round.

---

## 6.4 The root cause: these registrations contain no learnable shape structure

§3 said the suite was missing a metric for correspondence *correctness*. `probe_19` supplies
one, using the classical shape-model evaluation framework (Davies TMI 2002, Styner IPMI 2003)
that SOURCES.md had listed as not attempted.

**The idea.** Correspondence correctness is not measurable on one specimen without ground
truth, but it is measurable over a population. If template vertex *v* lands on the same
anatomical point in every specimen, the registered shapes form a low-dimensional manifold and
a shape model built from them **generalises**. If correspondence is scrambled, the model must
encode the permutation as well as the shape, so the same biological variation is spread over
far more directions and predicts nothing about a held-out specimen. Scrambled correspondence
is *expensive to represent*, and that cost is measurable without any labels.

**Leave-one-out reconstruction error**, on rest-space shaped geometry (pose removed), n=50 in
every row. `k=0` is the population spread, i.e. the error from simply predicting the mean:

| set | spread (k=0) | k=1 | k=10 | k=20 | in-sample PC1–10 |
|---|---|---|---|---|---|
| gated (top) | 0.01268 | 0.01282 | 0.01256 | 0.01241 | 39.7% |
| uniform sample | 0.02104 | 0.02138 | 0.02110 | 0.02090 | 39.3% |
| bench50 | 0.01996 | 0.02009 | 0.01970 | 0.01955 | 44.3% |

**The first mode is worse than predicting the mean. Twenty modes improve held-out error by
about 2%, while explaining 63–68% in-sample.** That is the signature of modes fitting
per-specimen noise.

**The instrument was validated before this was believed.** Fifty synthetic specimens sampled
from the template's own shape space — exact correspondence by construction, exactly 13
dimensional, same sample size:

| | gen@10 / spread | gen@20 / spread | in-sample PC1–10 |
|---|---|---|---|
| exact correspondence | **0.42** | **0.00** | 91.9% |
| + per-vertex noise 0.004 | 0.51 | 0.27 | 90.0% |
| **real registrations** | **0.99** | 0.98 | 39.7% |

The metric detects learnable structure when it is present, and reduces error to zero at 20
modes on 13-dimensional data. On the real registrations it finds none.

**The gate does not fix this.** Gated registrations appear to generalise better in absolute
terms (0.01256 vs 0.02110), but that is entirely lower diversity: normalised by each
population's own spread, all three sets sit at **0.99–1.00**. The gate improves the *fit*
(§6.2, decisively) without improving *correspondence consistency* at all.

**What this explains.** It is the common cause behind several separate observations:
* why the offset PCA spectrum is flat (§5.4) — the residual is noise, not shape;
* why enlarging the shape space 13 → 37 changed almost nothing (§4) — the added directions
  were noise directions;
* why co-registration produced folded blendshapes (§5.4) — it was extrapolating noise;
* why six correspondence-side interventions all failed to move the surface metrics (§8) — the
  surface metrics cannot see correspondence, and the correspondence was never established.

**What it means for scale.** Registering all 757 workers and rebuilding the shape space —
the plan §5.4 and this session both worked toward — **would not have worked**. More specimens
do not fix inconsistent correspondence; they average it further toward the mean. That is worth
stating plainly because it was the single most expensive item on the queue, and it was never
run. The measurement above cost ten minutes and removes it.

**The honest bottom line.** This pipeline currently produces *surface fits*, not
*registrations*. The distinction is exactly the one §1 opened with, and it is now measured at
the population level rather than inferred from edge distortion on single specimens. Nothing
downstream that depends on cross-specimen correspondence — shape spaces, beta regressors,
morphometrics — can be built on the current output at any corpus size.

---

## 6.5 Two consequences of §6.4, one tested and one not run

**The rebuilt shape space changes nothing (item 1, closed).** `SMIL_OmniAnt_gated.pkl` (20
directions from gated registrations) against the original 13-direction model, on the
`heldout_easy2` set — 50 specimens disjoint from bench50 *and* from both sets used earlier,
so no specimen that helped build the shape space is used to test it. Paired, n=50:

| metric | old (13) | gated (20) | Δ | p | |
|---|---|---|---|---|---|
| edge_logratio | 0.34022 | 0.34004 | +0.1% | 4.8e-01 | target-indep |
| deform_mag | 0.01483 | 0.01472 | +0.8% | 6.5e-02 | target-indep |
| midline_dev | 0.01399 | 0.01354 | +3.2% | 6.7e-01 | target-indep |
| chamfer_l2 | 0.00017 | 0.00017 | −0.5% | 6.7e-01 | surface |
| fscore@0.02 | 0.97121 | 0.97055 | −0.1% | 1.0e+00 | surface |

Every metric within ±3.2%, nothing significant. Exactly what §6.4 predicts: a shape space
built from registrations with no learnable structure adds nothing, and §4's "shape capacity is
not the binding constraint" is confirmed a second time by an independent route.

**Registering the full top 50% was not run, deliberately.** `run_top50.sh` is written, chunked
and ready (379 workers, ~2.5 h across both GPUs). It is not launched, because its purpose was
to feed a larger co-registration round and §6.4 shows that cannot work: more specimens do not
fix inconsistent correspondence, they average it toward the mean. A ten-minute measurement
removed the most expensive item on the queue. Running it would be spending hours to confirm
something already measured — and *that* judgement, rather than the run, is the useful output.

**What is being tried instead.** The soft partition (`--soft_partition <temperature>`),
implemented in `trainer_hierarchical.py`. The hard partition gives each target point entirely
to its argmin group, so a point equidistant between two legs commits fully to one and the fit
is pulled toward that commitment; §5.3 already showed that *freezing* such a commitment is
worse still. Softening is the E-step of the EM formulation used for robust statistical-shape-
model fitting: each point contributes to every group in proportion to how well that group
explains it, so uncertainty reduces influence rather than becoming a wrong commitment. Two
temperatures are running, and the pre-registered primary outcome is **probe 19 generalisation,
not surface metrics** — the claim is about correspondence, so it must be judged by the
instrument that measures correspondence.

---

## 6.6 The actual root cause: shape is stored in a parameter that cannot transfer

§6.4 reported that the registrations carry no learnable shape structure. Chasing *why*
produced the strongest and most actionable finding of this work, and it also **qualifies
§6.4's headline**.

**Step 1 — the soft partition does not help.** Pre-registered primary outcome was probe-19
generalisation, not surface metrics. All three land in the same place:

| arm | spread | gen@10 | gen/spread |
|---|---|---|---|
| hard (control) | 0.01996 | 0.01970 | 0.9870 |
| soft t=0.03 | 0.02019 | 0.01993 | 0.9871 |
| soft t=0.08 | 0.02031 | 0.02000 | 0.9846 |

Hard, frozen (§5.3) and soft assignments are indistinguishable. Per the pre-registration, the
problem is **upstream of the partition** — which is where the rest of this section looks.

**Step 2 — the variance is 100% free-form, which weakens §6.4 as stated.** Decomposing the
rest-space shape across the 50 specimens:

| component | variance | probe-19 gen@20/spread |
|---|---|---|
| parametric, `shapedirs·betas` | **0.0002** | 0.0008 (perfectly learnable) |
| free-form, `deform_verts` | **416.91** | 0.9794 (no structure) |
| free-form share of total | **100.0%** | |

So probe-19's null was measuring `deform_verts`, whose non-generalisation is close to
tautological for 30k per-specimen free parameters. **§6.4's finding is real but its
explanation was wrong**: the registrations lack learnable structure not because correspondence
is scrambled, but because *the parametric shape channel is carrying no signal at all.*

**Step 3 — the shape space is essentially unused.** Standard deviation of the fitted betas
across specimens, against a model prior sd of 0.28–1.52 per beta:

| run | betas sd | \|z\| mean | max \|z\| |
|---|---|---|---|
| **M7_handoff_midline** (the recommended arm) | **0.00042** | **0.0006** | 0.004 |
| **M1_sym** (its hierarchical base) | **0.00189** | 0.0026 | 0.037 |
| baseline | 0.25636 | 0.3738 | 2.057 |
| M4b_ceiling_free | 0.28360 | 0.4137 | 2.197 |

**Every specimen in the hierarchical arms receives essentially the same shape.** Betas shrink
monotonically down the schedule (H0 0.0087 → H1 0.0068 → H2 0.0059 → H3 0.0019). This is a
defect in the arm this report recommends, and it went unnoticed because no surface metric can
see it.

**Step 4 — what carries the variation instead.** Standard deviation across specimens:

| run | betas | `log_beta_scales` | ratio |
|---|---|---|---|
| M1_sym | 0.00189 | **0.13152** | 70× |
| M7_handoff_midline | 0.00042 | **0.17160** | 400× |
| baseline | 0.25636 | **0.91702** | 3.6× |
| M4b_ceiling_free | 0.28360 | **0.96504** | 3.4× |

In *every* arm, per-joint scaling carries more inter-specimen variation than the shape space —
by 70–400× in the hierarchical arms. The baseline drives per-joint scales from **0.106 to
4.542** (p05–p95), i.e. individual joints scaled by a tenth to four and a half times.

`log_beta_scales` is applied **during skinning**, per joint, per specimen. It is a pose-space
quantity. It is not part of the shape space, does not transfer between specimens, and is
invisible to any PCA over rest-space geometry — including probe 19's, and including the
`smil_importer` PCA that would be used to build a production model.

**Step 5 — and the model was built to prevent exactly this.** `SMIL_OmniAnt.pkl` ships
`scaledirs` with shape **(13, 55, 3)**: per-joint scale *blendshapes*, indexed by beta. The
model is designed so that betas drive joint scaling, keeping size variation inside the shape
space where it transfers.

**`scaledirs` is never referenced anywhere in the fitting path.** `smal_model/smal_torch.py`
does not use it. It is written by the Blender add-on (`smil_importer/model_build.py`) and
checked by one test, and that is all. The fitter instead optimises `log_beta_scales` as 165
free per-specimen parameters, which duplicate what `scaledirs·betas` was meant to supply — and
being free and essentially unregularised (`w_sym` only ties left to right), they win.

**The complete chain.** `scaledirs` unused → `log_beta_scales` free → per-joint scaling
out-competes betas by 70–400× → shape variation is stored per-specimen in a pose-space
parameter → the rest-space geometry carries no transferable shape → a shape space rebuilt from
the registrations is null (§6.5) → enlarging the shape space changes nothing (§4) → and no
downstream product that needs cross-specimen correspondence can be built.

This is one defect, it explains four previously separate observations, and it is concretely
fixable: **wire `scaledirs` into the forward pass so betas drive joint scaling, and penalise
or remove free `log_beta_scales`.** That is the next experiment, and probe 19 is the
instrument to judge it by — a successful fix should move gen/spread off 0.99 toward the 0.42
the synthetic control achieves.

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
