# Sources — papers read and code borrowed from

Everything below was actually consulted during this work, and the file/function that uses
each idea is named. Where an idea was rejected, the reason is given, because a citation
that only appears next to things that worked is not much use.

---

## Code this work builds on directly

| What | Where | How it is used here |
|---|---|---|
| **PyTorch3D** — Ravi et al., *Accelerating 3D Deep Learning with PyTorch3D*, arXiv:2007.08501 (2020) | installed pkg, v0.7.7 | `knn_points` (all correspondence operators), `sample_points_from_meshes` (area-weighted sampling — the fix in §2.4), `mesh_laplacian_smoothing`, `mesh_normal_consistency`, `chamfer_distance` (baseline only), and the rasterizer used for every 3D render in `render_3d.py`. |
| **SMAL** — Zuffi, Kanazawa, Jacobs, Black, *3D Menagerie: Modeling the 3D Shape and Pose of Animals*, CVPR 2017 | `smal_model/`, this repo | The parametric model class the whole pipeline fits. `SMAL3DFitter` in `fitter_3d/trainer.py` is reused unmodified by every experiment here. |
| **SMALify / SMALR** — Zuffi et al., *Lions and Tigers and Bears: Capturing Non-Rigid, 3D, Articulated Shape from Images*, CVPR 2018 | this repo's ancestry | Origin of the staged `init → pose/shape → deform` schedule that §2.1 shows is the core problem, and of the `model + free-form offsets` pattern whose cost §4 quantifies. |
| **SMPL** — Loper, Mahmood, Romero, Pons-Moll, Black, *SMPL: A Skinned Multi-Person Linear Model*, SIGGRAPH Asia 2015 | conceptual ancestor | Linear blend skinning + shape blendshapes, i.e. exactly the `shapedirs`/`weights`/`kintree_table` structure measured in probe 08. |

## Papers whose methods were implemented here

| Idea | Citation | Implemented in | Outcome |
|---|---|---|---|
| **Geman–McClure robust kernel**, and robust M-estimators for registration generally | Geman & McClure, *Statistical methods for tomographic image reconstruction*, Bull. ISI 1987; used for registration in Zhou, Park, Koltun, *Fast Global Registration*, ECCV 2016 | `robust_kernel()` in `trainer_moonshot.py` | **Mixed.** At a scale matched to the residual (A5: 0.20→0.10) it gave fscore@0.01 **+13.1%** at the pose checkpoint (48/50, p=2e-12). At too small a scale (E3: 0.05→0.02) it **starved the gradient** — 83% of points sat beyond the kernel scale in its zero-gradient region. It also systematically rejects **thin structures** (antennae, distal legs) as outliers — see §5. |
| **Graduated non-convexity / deterministic annealing** on the robust scale | Blake & Zisserman, *Visual Reconstruction*, MIT Press 1987; Black & Rangarajan, *On the unification of line processes, outlier rejection, and robust statistics*, IJCV 1996; Yang et al., *Graduated Non-Convexity for Robust Spatial Perception*, RA-L 2020 | `_current_scale()` in `trainer_moonshot.py` (geometric annealing) | Necessary but not sufficient — annealing to too small a final scale is what broke E3. |
| **Lowe ratio test + mutual nearest neighbours** for rejecting ambiguous correspondences | Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004 §7.1; mutual-NN filtering as in Rusu, Blodow, Beetz, *Fast Point Feature Histograms (FPFH) for 3D Registration*, ICRA 2009 | `mutual_nn_loss()` in `correspondence.py` | Direct attack on the six-identical-legs ambiguity: keeps only matches both sides agree on **and** that are unambiguous. |
| **Entropic optimal transport / Sinkhorn** as a soft correspondence | Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*, NeurIPS 2013; Feydy et al., *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences*, AISTATS 2019 | `sinkhorn_loss()` in `correspondence.py` (log-domain, batch-chunked) | Soft assignment so an ambiguous point spreads mass instead of committing to one wrong leg. |
| **Deterministic annealing on the assignment** (blur schedule) | Chui & Rangarajan, *A new point matching algorithm for non-rigid registration*, CVIU 2003 (TPS-RPM) | `corr_blur` → `corr_blur_end` annealing | Same continuation idea as GNC, applied to the transport plan rather than the residual. |
| **Basin hopping / multi-start over initial pose** | Wales & Doye, *Global Optimization by Basin-Hopping*, J. Phys. Chem. A 1997; multiple-initialisation practice in Bogo et al., *Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image*, ECCV 2016 | `optimise_multistart.py` | Motivated by arm E1, which showed the pose failure is a wrong-basin problem, not a step-size problem. |
| **As-rigid-as-possible / stiffness regularization against a rest shape** | Sorkine & Alexa, *As-Rigid-As-Possible Surface Modeling*, SGP 2007; Amberg, Romdhani, Vetter, *Optimal Step Nonrigid ICP*, CVPR 2007 (stiffness term) | `rest_edge_loss()` in `trainer_moonshot.py` | A cheap ARAP-lite: penalise deviation from the **template's own** rest edge lengths rather than from zero. Replaces the shrinkage bug in §2.4. Full ARAP (per-vertex rotation fitting via SVD) was not implemented — see "not attempted". |
| **Part-based / segment-aware registration**, fitting body before limbs | Anguelov et al., *SCAPE: Shape Completion and Animation of People*, SIGGRAPH 2005; Anguelov et al., *The Correlated Correspondence Algorithm for Unsupervised Registration of Nonrigid Surfaces*, NIPS 2004 | `trainer_hierarchical.py` | The main structural experiment: fit the body first to place the coxae, then partition the target by anatomical territory so a leg cannot be rewarded for matching a different leg. |
| **Gaussian shape prior via Mahalanobis distance** | standard in SMPL/SMAL fitting; here the covariance ships in the model file itself | `w_beta_prior` in `trainer_moonshot.py` | Activates `betas_prec`, which §2.4 shows was computed and never used. **I mis-scaled it by ~2 orders of magnitude on the first attempt** (§4.1). |
| **F-score at a distance threshold** as the primary surface metric, over chamfer | Knapitsch et al., *Tanks and Temples*, SIGGRAPH 2017; Tatarchenko et al., *What Do Single-view 3D Reconstruction Networks Learn?*, CVPR 2019 (chamfer is perturbed by outlier layout) | `metrics.py: surface_metrics()` | Why the report leads with fscore@0.01 rather than chamfer — several arms move the two in opposite directions, and that disagreement is diagnostic of outlier rejection. |

## Added in the part-field session (§12)

| Idea | Citation | Implemented in | Outcome |
|---|---|---|---|
| **PointNet++ segmentation architecture** (set abstraction + feature propagation, per-point logits) | Qi, Yi, Su, Guibas, *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*, NeurIPS 2017; PointNet — Qi et al., CVPR 2017 | `PartFieldNet` in `fitter_3d/partfield.py`, on the SA/FP layers already vendored in `fitter_3d/pointcloud2smil/pointnet2_utils.py` (yanx27/Pointnet_Pointnet2_pytorch lineage) | The `group_all` fourth abstraction level is the load-bearing part: it is the global-context branch, and it is precisely what a geodesic level set structurally lacks. |
| **Body-part segmentation as an intermediate representation for model fitting** | Omran, Lassner, Pons-Moll, Gehler, Schiele, *Neural Body Fitting: Unifying Deep Learning and Model-Based Human Pose and Shape Estimation*, 3DV 2018; Bogo et al., *FAUST*, CVPR 2014 (part labels from a registered template) | the whole of §12 | The reason a segmentation is worth predicting at all: it is a target-derived anchor that the fit cannot edit, unlike a fit-derived partition. |
| **Bootstrapping labels from a parametric model** (train on posed model samples, apply to real scans) | standard sim-to-real practice; SURREAL — Varol et al., *Learning from Synthetic Humans*, CVPR 2017 | `make_partfield_data.py` | Labels cost nothing: `weights.argmax` gives every template vertex its joint. The domain gap is closed by CT-specific augmentation (holes, debris blobs, and debris *bridges* that fuse limbs) rather than by annotation. |
| **Moment matching as a correspondence-free registration objective** | classical shape-moment alignment; the second-moment form used here avoids the SVD-based principal-axis extraction of e.g. Horn 1987 because it degenerates on near-isotropic parts | `fitter_3d/part_anchor_init.py` | Matches per-part centroid + second moment. Removes the freedom to choose *which* target point a source point explains, which is where chamfer's multimodality comes from. Aimed directly at §5's trapped-pose finding. |
| **Class-frequency reweighting for extreme imbalance** | inverse-frequency / inverse-sqrt weighting, standard in semantic segmentation (e.g. Paszke et al., *ENet*, arXiv:1606.02147 §median-frequency balancing) | `part_field_loss()` | Not cosmetic here: area-sampled points are ~61% body and 0.4–0.8% per distal-leg class, so an unweighted net scores well by never predicting a distal class — the same failure mode probe 13 found in the *fitter*. |

## Read and deliberately **not** used

| Method | Citation | Why not |
|---|---|---|
| **Coherent Point Drift / BCPD** | Myronenko & Song, *Point Set Registration: Coherent Point Drift*, TPAMI 2010; Hirose, *Acceleration of BCPD*, TPAMI 2021 | Its deformation model is a free displacement field, which discards the articulated SMIL parameterisation — the very thing registration here exists to produce. Its transferable idea (soft posterior correspondence) is obtained instead via Sinkhorn. |
| **Go-ICP**, **TEASER++**, **FGR** as a pipeline | Yang et al., TPAMI 2016; Yang, Shi, Carlone, *TEASER++*, T-RO 2020; Zhou et al., ECCV 2016 | Wrong regime: these solve globally-optimal **rigid** registration. This problem is non-rigid and cross-species. TEASER also needs putative correspondences, and `open3d` is not installed in this env. FGR's scaled Geman–McClure kernel *was* adopted (above). |
| **Functional maps** and deep variants | Ovsjanikov et al., SIGGRAPH 2012; Sharp et al., *DiffusionNet*, TOG 2022; unsupervised variants (ULRSSM etc.) | Intrinsic/LBO methods are **symmetry-ambiguous**, and a bilaterally symmetric ant is close to the worst case. The scans are also topologically noisy CT data, which corrupts the LBO spectrum. |
| **Neural / learned initialisers** (ArtEq, NFR) | Feng et al., *ArtEq*, CVPR 2023; *NFR*, arXiv:2505.22445 | Available weights are SMAL/SURREAL-trained — quadrupeds and humans, not hexapods. Retraining is a multi-day job, not a one-night one. |
| **PartField** for learned part segmentation | nv-tlabs/PartField, ICCV 2025 | Trained on man-made objects and humans; 5 mm insects are far out of distribution. The LBS-weight argmax segmentation already in `metrics.py` is free and anatomically exact for this template. **Superseded in §12** — the right move was not to reuse their weights but to take the idea and train a hexapod part field on labels the template generates for free. The objection was to the checkpoint, not to the concept. |

## Not attempted, and worth attempting next

- **Full ARAP** (per-vertex rotation via batched SVD) instead of the edge-length proxy —
  Sorkine & Alexa 2007. Only worth it if the rest-edge term leaves visible artefacts.
- **Co-registration / model bootstrapping**: alternate registration and shape-space rebuilding.
  Hirshberg et al., *Coregistration: Simultaneous Alignment and Modeling of Articulated 3D
  Shape*, ECCV 2012; Zuffi et al. CVPR 2017 iterate this 4 rounds. **This is the direct
  answer to §4** — if the shape space is too small, enlarge it from the data. It is the
  single highest-value follow-up and needs more than one night.
- **PCA specificity/generalisation** as population-level registration quality —
  Davies et al., *3D Statistical Shape Models Using Direct Optimisation of Description
  Length*, TMI 2002; Styner et al., *Evaluation of 3D Correspondence Methods for Model
  Building*, IPMI 2003. Requires a full 530-specimen run.

## Prior work on this exact codebase

Khaoula Jellal's two investigation deliverables (Google Drive, read in full during this
work) — the SDF-loss investigation and the penetration-loss investigation. Two results were
load-bearing here:

1. Her structural finding that **pose freezes after an early stage and later mesh metrics
   cannot see the damage** — independently reproduced and quantified in §2.1/§2.3.
2. Her weight-reduction methodology on the penetration loss (0.1/0.2 → 0.02/0.05 retained
   76% of the benefit for 27.5% of the cost) — the template for the `w_offset` sweep in §5,
   after `w_offset = 5.0` proved far too strong.

Her experimental discipline (pre-registered adoption criteria, 3-seed repeats, sign tests,
per-part regression gates, frozen-correspondence causal tests) is the protocol this work
follows.
