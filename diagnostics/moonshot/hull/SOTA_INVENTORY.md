# SOTA inventory — unsupervised part hierarchy and dense mesh registration

Produced by a five-way parallel literature sweep (convex decomposition; skeletal/medial;
unsupervised co-segmentation and part fields; non-rigid registration; symmetry and topology),
each followed by a deepening pass that fetched repositories and papers to verify code
availability. 66 unique methods after de-duplication.

**Read this next to [REPORT_HULL.md](REPORT_HULL.md), which measures the family and finds a
ceiling.** The short version of that report: a part decomposition can only address the 11.8%
of correspondence error that crosses a part boundary, so *every* method in the first four
sections below — including any perfect one — is capped at moving correspondence correctness
from 4.81% to 16.7%. The fifth section is the one that attacks the other 83%.

**Implemented and tested here** (all in `fitter_3d/hull_decomposition.py`):
CoACD as the over-decomposition; V-HACD/CoACD-style volumetric concavity and CoACD's `Hb`
surface deviation as merge criteria; **WCSeg (Asafi et al., SGP 2013)** in both its greedy
line-of-sight form and its actual spectral normalised-cut form.

**Deliberately not implemented, with reasons:**
- **PartField** (ICCV 2025) — the only surveyed method with a *native* part hierarchy and the
  strongest candidate on paper. Not tested because §5's ceiling bounds it identically, and
  because it is trained on ShapeNet/PartNet-lineage data where insects are far out of
  distribution. Worth revisiting only if the within-part problem is solved first.
- **SAMPart3D / P3-SAM / PartSAM** — granularity knobs rather than hierarchies, all with the
  same OOD risk (they distil web-image foundation features; ant CT scans are not web images).
- **V-HACD 4.0** — same family as CoACD, and §3.3 shows the binding constraint is that
  ACD cut planes do not follow anatomy, which V-HACD shares by construction.
- **RigNet / UniRig / Anymate** — produce a rooted skeleton tree, which is exactly the wanted
  structure, but all are supervised on rigged character datasets (bipeds/quadrupeds).

### Convexity / hull-separation — the family the user named

| method | year | venue | unsup. | hierarchy | code |
|---|---|---|---|---|---|
| **CoACD** | 2022 pape | SIGGRAPH 2022 / ACM TOG 41 | fully | Binary tree in mechanism, flat list in output, parent-child link genuinely dis | VERIFIED EMPIRICALLY, not just read. `pip download c |
| **HACD** | 2009 / 20 | 'A simple and efficient ap | fully | Yes — the strongest and cleanest hierarchy in the classical group, and the onl | Partially verified. Mamou's blog post confirms he re |
| **WCSeg** | 2013 / 20 | Asafi, Goren & Cohen-Or, ' | fully | Yes, and it is the most naturally hierarchical of the classical methods, thoug | Unverified. The TOG paper states 'our source code, d |
| **VisACD** | 2026 | Eurographics 2026 (Short P | fully | Yes in mechanism, no in output. The algorithm is a recursive binary split of a | VERIFIED: https://github.com/3dlg-hcvc/visacd. LICEN |
| **RL-ACD** | 2025 | SIGGRAPH Asia 2025 / ACM T | self-supervised | Yes in mechanism. The MDP is by definition a sequence of cutting actions appli | Unverified. The ACM DL page returned HTTP 403 to my  |
| **CID** | 2023 | IEEE Robotics and Automati | fully | Not as published — the paper outputs a flat partition via CID-FPS seeding plus | Project page https://ai4ce.github.io/CID/ (confirmed |
| **Learned convex-primitive family** | 2019-2021 | BSP-Net (CVPR 2020 Best St | self-supervised | Mostly no, with one real exception. BSP-Net's 'BSP tree' name overpromises: it | Unverified in detail — I did not fetch the repositor |
| **SEG-MAT** | 2020 (arX | IEEE TVCG — Lin, Liu, Li,  | fully | No — the released output is a FLAT segmentation with the part count determined | VERIFIED — README fetched at https://github.com/clin |

### Skeleton, medial axis and auto-rigging — hierarchy by construction

| method | year | venue | unsup. | hierarchy | code |
|---|---|---|---|---|---|
| **Skeletonization via Local Separators** | 2021 (LS, | ACM TOG 40(5) art. 187 (Bæ | fully | Yes, and I measured the shape of it. The vertex→separator map is a flat partit | VERIFIED BY INSTALLING AND RUNNING IT. Repo https:// |
| **skeletor** | 2020–2026 | Software library (Philipp  | fully | Yes, and more directly than LS: the `Skeleton.swc` table is a rooted parent-ch | VERIFIED on both GitHub and PyPI. Repo https://githu |
| **Point2Skeleton** | 2021 | CVPR 2021 (oral) — Lin, Li | fully | No. Output is a skeletal mesh with non-manifold branches, not a rooted tree; y | Verified — I read the README at https://github.com/c |
| **Coverage Axis++** | 2024 (CA+ | Computer Graphics Forum /  | fully | No. Output is a flat set of medial spheres with radii. A hierarchy requires yo | VERIFIED — README fetched at https://github.com/Fran |
| **Neural Skeleton** | 2023 | Shape Modeling Internation | self-supervised | No. Output is a medial point set / skeletal mesh, with no rooted tree and no p | Verified — I read the README at https://github.com/M |
| **MATTopo** | 2024 | ACM TOG / SIGGRAPH Asia 20 | fully | No directly, but it produces the richest raw material for building one: a medi | VERIFIED — README fetched at https://github.com/ning |
| **CSCD** | 2026 | WACV 2026 (Bardhan, Hebbal | fully | Produces a curve skeleton graph/tree inheriting the local-separator structure; | VERIFIED ABSENT — I fetched the project page at http |
| **UniRig** | 2025 | SIGGRAPH 2025 / ACM TOG (V | pretrained-zero-shot | Yes — a rooted tree by construction (the tokenization enforces validity), plus | Verified — I read the README at https://github.com/V |
| **Anymate** | 2025 (Any | SIGGRAPH 2025 Conference P | pretrained-zero-shot | Yes for both — rooted skeleton tree (joint positions + connectivity matrix, ro | Verified for both. Anymate: README read at https://g |
| **Contraction-family curve skeletons** | 2012 (MCF | SGP 2012 / CGF (Tagliasacc | fully | skeletor returns a Skeleton object with vertices, parent-child EDGES, an SWC t | Partly verified. skeletor: README read at https://gi |
| **ATM** | 2026 (arX | arXiv 2606.29167 (Liu, Xia | pretrained-zero-shot | Yes, in the only way that matters here — the hierarchy is the parametric model | CODE NOT RELEASED. VERIFIED: https://github.com/liu- |

### Unsupervised / zero-shot part fields — granularity knobs

| method | year | venue | unsup. | hierarchy | code |
|---|---|---|---|---|---|
| **PartField** | 2025 | ICCV 2025. Liu, Uy, Sharp  | pretrained-zero-shot | Yes — and this is the only method on the entire list where a hierarchical part | VERIFIED: https://github.com/nv-tlabs/PartField. LIC |
| **SAMPart3D** | 2024/2025 | arXiv:2411.07184, Nov 2024 | pretrained-zero-shot | Only in the weak sense of a continuous granularity knob. The scale-conditioned | Project page https://yhyang-myron.github.io/SAMPart3 |
| **P3-SAM** | 2025 | arXiv:2509.06784, Sept 202 | supervised | Yes at the output level: the paper reports hierarchical part segmentation as o | https://github.com/Tencent-Hunyuan/Hunyuan3D-Part, s |
| **PartSAM** | 2025 | arXiv:2509.21965 | pretrained-zero-shot | A granularity knob only — parts at different NMS thresholds are not guaranteed | UNVERIFIED / not available. The paper states 'our co |
| **DAE-Net** | 2024 | SIGGRAPH 2024; arXiv:2311. | fully | No. Output is a flat set of at most `branch_num` parts. There is no tree, no c | github.com/czq142857/DAE-Net (verified, README fetch |
| **SegviGen** | 2026 | SIGGRAPH 2026 (Journal tra | pretrained-zero-shot | No native hierarchy, but it has an unusual and potentially very useful granula | github.com/Nelipot-Lee/SegviGen (verified, README fe |
| **Classic unsupervised co-segmentation** | 2011-2016 | SIGGRAPH Asia 2011 (ACM TO | fully | No, and the classic pipeline offers no principled granularity control — the nu | Project page people.scs.carleton.ca/~olivervankaick/ |
| **HiT** | 2025 (v2  | arXiv:2510.27088 (Vora, Go | self-supervised | Yes, and more principled than anything else here: the hierarchy is architectur | github.com/aditya-vora/HiT (verified: real code, tra |

### Topology, symmetry and classical segmentation

| method | year | venue | unsup. | hierarchy | code |
|---|---|---|---|---|---|
| **Randomized Cuts partition function** | 2008 | ACM Transactions on Graphi | fully | Yes, explicitly — Section 7.2 gives a complete recursive algorithm (most-consi | None, confirmed twice. The official Princeton page ( |
| **Shape Diameter Function** | 2008 (ver | The Visual Computer 24(4): | fully | Two levels come free from one CGAL call (thickness class via output_cluster_id | THREE routes, all verified. (a) NOTHING NEW NEEDED — |
| **ToMATo** | 2013 | Journal of the ACM 60(6):4 | fully | Yes, implicitly and cheaply. The merge order is a monotone hierarchy: sweeping | Yes, verified. `gudhi.clustering.tomato.Tomato` in t |
| **Handle & tunnel loops via Reeb graphs** | 2013 | ACM Transactions on Graphi | fully | No, but it produces an ORDERED list of loops by geometric size, and cutting th | Yes, verified. ReebHanTun is distributed from Tamal  |
| **Symmetry Hierarchy of Man-Made Objects** | 2011 | Computer Graphics Forum 30 | fully | Yes — this is the only entry here whose entire output IS the hierarchy, and it | None found. Eurographics 2011 paper with a PDF hoste |
| **Symmetry Factored Embedding and Distance** | 2010 | ACM Transactions on Graphi | fully | Not directly. One run yields orbits — one level of structure. A hierarchy come | None. The Princeton project page offers only the pap |
| **E3Sym** | 2023 (rep | ICCV 2023 | self-supervised | No. Output is a set of reflection planes, i.e. a single global structural fact | Yes, minimal. https://github.com/renwuli/e3sym — off |
| **chi** | 2025 ICCV | ICCV 2025 (arXiv:2508.0550 | self-supervised | No — it produces a per-vertex feature and a left/right split, one level only.  | YES. https://github.com/TWeissberg/chirality (the pr |

### Non-rigid registration / dense correspondence — attacks the WITHIN-part error

| method | year | venue | unsup. | hierarchy | code |
|---|---|---|---|---|---|
| **ULRSSM** | 2023 / 20 | SIGGRAPH 2023, ACM TOG 42( | fully | No. Emits a dense point-to-point map only. Same remark as above: use the map t | Verified and directly fetched: https://github.com/do |
| **Smooth Shells** | 2020 | CVPR 2020 / NeurIPS 2020 ( | fully | Yes — but a SCALE hierarchy (coarse-to-fine frequency), not a PART hierarchy.  | Verified, repo fetched: https://github.com/marvin-ei |
| **Hybrid Functional Maps** | 2024 (CVP | CVPR 2024 (Bastian, Xie, N | fully | No hierarchy is produced. But this is the machinery that lets you PUSH the SMI | Verified. Project page https://hybridfmaps.github.io |
| **Diff3F** | 2024 (rep | CVPR 2024 (Dutt, Muralikri | pretrained-zero-shot | No hierarchy natively — a flat (V, 2048) feature field. Two ways to get one, b | VERIFIED via GitHub API. https://github.com/niladrid |
| **NFR** | 2025 (arX | arXiv 2505.22445 (Jiang, S | self-supervised | No hierarchy produced. But this is the natural HOST for one: the deformation s | VERIFIED via GitHub API and full file tree. https:// |
| **NFR / DFR** | 2023 (DFR | NeurIPS 2023 (Jiang, Sun,  | self-supervised | No. But it is the natural host for a hierarchy: the deformation stage is exact | Verified. NFR: https://github.com/rqhuang88/NFR — fe |
| **DPFM** | 2021 | 3DV 2021, Oral, Best Paper | fully | No. But the overlap mask is a segmentation-like output, and DeepShapeMatchingK | Repo exists at https://github.com/pvnieo/DPFM (PyTor |
| **DiffuMatch** | 2025 (arX | ICCV 2025 (Pierson, Li, Da | fully | No. | VERIFIED via GitHub API. https://github.com/daidedou |
| **DenoisFM** | 2025 | CVPR 2025 (Zhuravlev, Lähn | supervised | No, but the template-centric formulation is architecturally aligned with SMILi | Repo URL https://github.com/alekseizhuravlev/denoisi |
| **DenseMatcher** | 2024/2025 | ICLR 2025 (Zhu et al., TEA | supervised | No. Dense per-vertex correspondence only; hierarchy would again come from tran | VERIFIED via GitHub API. https://github.com/TEA-Lab/ |
| **G-MSM** | 2022/2023 | CVPR 2023 (Eisenberger, To | self-supervised | No part hierarchy. It builds a hierarchy over the DATASET (which shape mediate | VERIFIED via GitHub API. https://github.com/marvin-e |
| **AMM_NRR** | 2022/2023 | IEEE TPAMI (Yao, Deng, Xu, | fully | No. Purely geometric optimisation, no parts, no hierarchy, no semantics. | Verified, repo fetched: https://github.com/yaoyx689/ |
| **Unsupervised Spectral Basis Learning** | 2026 (arX | arXiv 2603.23383 (Luo, Che | fully | No. | VERIFIED to exist via GitHub API. https://github.com |
| **Hyper-Network Neural Functional Maps** | 2026 | ECCV 2026 (Dongliang Cao,  | fully | No. | NOT VERIFIED. No code URL was found on the arXiv abs |
| **MeshFM** | 2026 | arXiv:2607.27592 (Threedle | self-supervised | Partially. The paper demonstrates 'hierarchical part segmentation' with adjust | NOT RELEASED as of my check. Project page threedle.g |

### Other surveyed

| method | year | unsup. | hierarchy |
|---|---|---|---|
| NICP / NSR | 2024 | self-supervised | No. |
| BAE-Net / RIM-Net | 2019 / 20 | fully | BAE-Net: no, flat branches. RIM-Net: yes, but rigidly — a fixed BINARY |
| Learning Convex Decomposition via Featur | 2026 | self-supervised | Yes, and instrumentable in Python rather than C++. Algorithm 1 is an e |
| Lien & Amato Approximate Convex Decompos | 2004 / 20 | fully | Yes, and it is stated as a headline property rather than inferred: the |
| V-HACD 4.0 | 2016 / 4. | fully | Recursive by construction — maxRecursionDepth is a literal tree-depth  |
| Coverage Axis / Coverage Axis++ | 2022 (Cov | fully | No — and I verified this twice. The README states output is .obj files |
| Reeb graph / Morse-theoretic segmentatio | 2024 (Beg | fully | Yes, and this is its genuine strength: persistence-ordered cancellatio |
| Reeb graph segmentation driven by SDF /  | 2024 | fully | Single non-hierarchical segmentation per run, but the paper notes that |
| Randomized Cuts for 3D Mesh Analysis | 2008 | fully | Indirectly. The ranked list of most-consistent cuts is a natural nesti |
| Skeleton-based intrinsic symmetry detect | 2013 | fully | No. Output is a flat set of symmetric regions plus pairings; there is  |
| Multi-scale Partial Intrinsic Symmetry D | 2012 | fully | Multi-scale and overlapping, which is close to but not identical to a  |
| Robust Symmetry Detection via Riemannian | 2024 | fully | No. Output is a set of reflective planes with associated support, not  |
| Fast and Accurate Intrinsic Symmetry Det | 2018 | fully | No. Output is a pointwise involution (each vertex mapped to its mirror |
| Hybrid Functional Maps via DeepShapeMatc | 2024 (CVP | fully | No hierarchy. Same play as Diff3F: use the dense map to push the SMIL  |
| χ | 2025 (ICC | self-supervised | No. It is a per-vertex feature refinement. Its value is orthogonal: it |
| Synchronous Diffusion for Unsupervised S | 2024 (arX | fully | No. |