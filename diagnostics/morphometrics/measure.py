"""The measurement layer: turn a fitted run into a tidy morphometric table.

DESIGN, and why each choice is forced by something measured rather than chosen for convenience.

WHAT IS POSE-INVARIANT, AND WHAT IS NOT
A bone length is the distance between two ADJACENT joints, and rotating a joint does not change
the distance to its own parent. So `||J[i] - J[parent[i]]||` is invariant to the entire pose,
which matters because these are museum specimens posed differently in every scan. Joint
positions come from `J_regressor @ verts` -- the same gauge-invariant route probe_pose_recovery
used, and the reason it is trustworthy is that a joint position is a weighted average over
hundreds of vertices, so the within-part correspondence scrambling that destroys dense
correspondence (E7: 83% of error never crosses a part boundary) largely averages out.

PART DIMENSIONS NEED A PART-LOCAL FRAME, AND EIGENVALUE ORDER WILL NOT DO
Head width and head length require axes. Taking the head vertices' covariance eigenvectors and
sorting by eigenvalue silently SWAPS length and width for any ant whose head is wider than long
-- which is exactly the genera the analysis most wants to detect (Cephalotes). So instead each
part is Kabsch-aligned to the TEMPLATE's copy of that part, and extents are measured along the
template's fixed axes. A broad head then reports a large width, not a relabelled length.

Extents use the 2nd-98th percentile rather than min-max: these scans are damaged, and a single
stray vertex on a broken leg should not define a body dimension.

NO ABSOLUTE SIZE EXISTS
`load_meshes` centres each target and divides by max|coord|, and ALL_ANTS_CLEAN meshes arrive
pre-normalised (every one has bounding diagonal ~1.6) while worker mesh units are arbitrary per
scan. Absolute body size is therefore NOT recoverable from either corpus. Everything here is a
proportion, and size is removed explicitly with Mosimann log-shape-ratios (divide every
measurement by the geometric mean of the set, then log) rather than left implicit.

EXCLUSIONS, all verified against the model file
* `w_1_*`, `w_2_*` own ZERO vertices by dominant skinning weight. They are wing joints with no
  geometry; their "bone lengths" are rigidly tied to thorax geometry and merely re-measure
  thorax scale under an anatomical-sounding name. Dropped.
* `b_h` is COINCIDENT with `b_t` -- the head bone has length exactly 0.0. There is no head
  length bone, so head size is mesh-derived only.
* Pretarsi (7-15 verts), `b_a_5` (3 verts) and `an_3` (4 verts) own too little geometry for
  their joint position to be stable. Flagged `low_support` and excluded from the core set.
"""

import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

LOW_SUPPORT = 20  # dominant-weight vertex count below which a joint's position is unreliable

# Blocks kept in the default feature set, selected on GROUND-TRUTH RELIABILITY ALONE.
#
# Selection rule: BLOCK-level effective SNR > 1. Per-feature SNR is measured against the
# synthetic corpus and rescaled by (real spread / synthetic spread), then aggregated over the
# block as median x sqrt(n_features).
#
# THE sqrt(n) IS THE POINT, and getting it wrong once cost the gaster its place. Features are
# never used one at a time -- the classifier sees the whole block -- so several individually
# marginal but largely independent measurements combine into a usable one. Judging a block by
# its per-feature median silently penalises exactly the case where a body region is described by
# many modest measurements rather than a few sharp ones. Measured (calib_features.py):
#
#     block        R vs truth   corrected per-feature SNR   n   BLOCK SNR
#     head            0.958              2.92                3     5.53   keep
#     leg_prox        0.756              1.83               12     3.88   keep
#     mesosoma        0.849              1.49                3     3.27   keep
#     mandible        0.608              1.99                5     2.55   keep
#     gaster          0.597              0.77                8     2.31   keep
#     antenna         0.759              1.86                3     2.10   keep
#     leg_distal      0.136              0.44                6     0.90   DROP
#
# Only the distal leg fails, and it fails on every measure: R=0.136 against ground truth, and no
# genus signal whatever on its own (6.8%, p=0.72). Dropping it is what makes the selection worth
# doing -- a standardised PCA gives every column equal influence, so 6 noise columns cost real
# accuracy (all 40 features 42.4%, this 34-feature set 50.8%).
#
# THE GASTER'S ARTEFACT HYPOTHESIS WAS TESTED AND REFUTED, which is the other half of why it is
# here. The worry was preservation: abdominal distension varies with fixation, and fixation
# batch can track taxon. Using accession-number blocks as a batch proxy (5 batches, 119
# specimens, the largest spanning 30 genera so batch and taxon are not confounded), NO block
# predicts batch above chance -- gaster 31.9% against a 29.7% null (p=0.35), head p=0.64,
# mesosoma p=0.46. And residualising the gaster on batch does not damage its genus signal, it
# sharpens it: 50.8% -> 61.0%. The signal is biological.
CORE_BLOCKS = ("head", "mandible", "antenna", "leg_prox", "mesosoma", "gaster")

# Parts as vertex sets: the three tagmata plus the mandibles, which is how an ant is described.
#
# THE GASTER IS b_a_1..b_a_5 TAKEN AS ONE, deliberately. A part's dimensions come from rigidly
# aligning its vertex cloud to the template's copy, which is exactly pose-invariant only for a
# cloud driven by a single joint -- true of `b_h` and `b_t`, not of the gaster, which spans four
# flexion joints. Splitting it into segments would restore strict invariance, and was tried; it
# is not worth the cost. It doubles the part count, puts most of the extra columns on the
# petiole (72 verts) and postpetiole (126), and those are precisely the small, noisy structures
# that should not gain influence in a variance- or count-weighted feature set. One gaster, one
# set of three dimensions, and the residual pose sensitivity is declared rather than engineered
# away: it is the most likely reason the gaster block calibrates at corrected SNR 0.77 against
# the head's 2.92, and it is why the gaster sits outside CORE_BLOCKS.
PART_DEFS = {
    "head": lambda n: n == "b_h",
    "mesosoma": lambda n: n == "b_t",
    "gaster": lambda n: n.startswith("b_a_"),
    "mandible_r": lambda n: n == "ma_r",
    "mandible_l": lambda n: n == "ma_l",
}


def load_model():
    import config

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    Jr = np.asarray(dd["J_regressor"])
    if hasattr(Jr, "toarray"):
        Jr = Jr.toarray()
    jnames = [str(x) for x in dd["J_names"]]
    parents = np.asarray(dd["kintree_table"])[0]
    weights = np.asarray(dd["weights"])
    return dict(
        dd=dd,
        Jr=Jr,
        jnames=jnames,
        parents=parents,
        weights=weights,
        dominant=weights.argmax(1),
        v_template=np.asarray(dd["v_template"], dtype=np.float64),
    )


def group_of(name):
    if name.startswith("b_a_"):
        return "gaster"
    if name == "b_h":
        return "head"
    if name.startswith("ma"):
        return "mandible"
    if name.startswith("an_"):
        return "antenna"
    if name.startswith("w_"):
        return "wing"
    if name.startswith("l_"):
        seg = name.split("_")[2]
        return "leg_prox" if seg in ("co", "tr", "fe") else "leg_distal"
    return "mesosoma"


def bone_table(M, drop_wing=True, drop_zero=True):
    """One entry per real bone, named ANATOMICALLY rather than after its child joint.

    THE OFF-BY-ONE THIS FIXES. The bone from parent P to child C is the physical segment whose
    proximal joint is P -- the femur runs from the femur joint to the tibia joint. Naming that
    distance after C (the obvious thing, since kintree indexes bones by child) silently shifts
    every appendage measurement one joint distal: the "tibia" bone would be the femur's length
    and the "an_2" bone would be the scape's. Caught because the resulting table claimed
    Strumigenys had the SHORTEST mandibles of any genus measured, which is the reverse of the
    truth for a trap-jaw genus.

    So bones are named `seg_<parent>`. Bones descending from a body origin (`b_t`, `b_h`) are
    not segments at all -- they locate where an appendage ATTACHES -- and are named
    `attach_<child>` so they can never be mistaken for a limb length.

    VERIFIED EMPIRICALLY, not argued. In linear blend skinning a joint sits at the PROXIMAL end
    of the geometry it drives, so the test is: do a joint's dominantly-skinned vertices lie
    distal to it? Fraction of a joint's own vertices lying beyond it, along the bone from its
    parent: l_1_fe 0.99, l_1_ti 1.00, l_1_ta 1.00, an_1 1.00, an_2 0.99, ma 0.88, l_1_co 0.90.
    They do. Hence segment X spans joint X to X's child, and `seg_X` is correct.

    ONE ANOMALY, flagged rather than smoothed over: `l_*_tr_*` scores 0.06 -- its skinned
    vertices lie PROXIMAL to it, unlike every other leg joint -- and `seg_*_tr` is the longest
    leg bone in the template (0.346 vs the femur's 0.250), which no real ant trochanter is. The
    rig's trochanter placement does not correspond to the insect podomere, so `seg_*_tr` is
    marked `anomalous` and must not be reported as "trochanter length".
    """
    jn, par = M["jnames"], M["parents"]
    counts = np.bincount(M["dominant"], minlength=len(jn))
    J0 = M["Jr"] @ M["v_template"]
    out = []
    for j, p in enumerate(par):
        if p < 0 or p >= len(jn):
            continue
        if drop_wing and (jn[j].startswith("w_") or jn[p].startswith("w_")):
            continue
        L0 = np.linalg.norm(J0[j] - J0[p])
        if drop_zero and L0 < 1e-6:
            continue  # b_h <- b_t, a genuinely zero-length bone
        origin = jn[p] in ("b_t", "b_h")
        name = f"attach_{jn[j]}" if origin else f"seg_{jn[p]}"
        # support is limited by whichever endpoint owns less geometry: both joint positions
        # must be stable for their distance to mean anything
        low = min(counts[j], counts[p]) < LOW_SUPPORT
        out.append(
            dict(
                child=j,
                parent=int(p),
                name=name,
                joint=jn[j],
                is_attach=origin,
                group=group_of(jn[j]),
                low_support=bool(low),
                rest_len=L0,
            )
        )
    return out


def leaf_joints(M):
    """Joints with no child. Their size CANNOT come from a bone length -- there is no next joint.

    A mandible, a terminal antennal segment, a pretarsus: each is the end of its chain, so the
    only bone touching it is the one arriving from its parent, and that distance measures WHERE
    THE STRUCTURE ATTACHES, not how long it is. Using it as a length is exactly the error that
    made the first index table report Strumigenys as having the shortest mandibles of any genus.

    For these, length must come from the extent of the vertices the joint owns -- see
    `leaf_extents`. This is also why the free-form per-vertex deformation must stay small: a leaf
    measurement is read directly off the vertex cloud, so any offset the fitter is free to invent
    is added straight onto the measurement, with no skeleton to constrain it. E8b independently
    selected a 25x offset penalty on correspondence grounds; the same setting is what keeps these
    measurements meaningful, which is a useful convergence rather than a coincidence.
    """
    par = M["parents"]
    has_child = set(int(p) for p in par if p >= 0)
    return [j for j in range(len(M["jnames"])) if j not in has_child]


def leaf_extents(verts, M, lo=2.0, hi=98.0):
    """Length of each leaf structure, from its own vertex cloud rather than from a bone.

    Measured along the cloud's OWN first principal axis. For an elongate structure PC1 is
    unambiguously the long axis, so this needs no template alignment and no axis-role assignment
    -- the ordering instability that forced anatomical axis labelling for the head cannot arise
    for a single length. Percentile extent, not min-max, because these scans are damaged.
    """
    out = {}
    dom = M["dominant"]
    for j in leaf_joints(M):
        name = M["jnames"][j]
        if name.startswith("w_"):
            continue  # wing joints own no geometry at all
        idx = np.where(dom == j)[0]
        if idx.size < LOW_SUPPORT:
            continue  # an_3 (4 verts), pretarsi (7-15), b_a_5 (3): too little geometry to trust
        P = verts[:, idx]
        L = np.zeros(P.shape[0])
        for i in range(P.shape[0]):
            Q = P[i] - P[i].mean(0)
            _, _, Vt = np.linalg.svd(Q, full_matrices=False)
            t = Q @ Vt[0]
            L[i] = np.percentile(t, hi) - np.percentile(t, lo)
        out[f"leaflen_{name}"] = L
    return out


def joints(verts, Jr):
    """(n,V,3) -> (n,J,3) joint positions. Gauge-invariant; see module docstring."""
    return np.einsum("jv,nvc->njc", Jr, verts)


def bone_lengths(J, bones):
    return np.stack([np.linalg.norm(J[:, b["child"]] - J[:, b["parent"]], axis=-1) for b in bones], axis=1)


def kabsch(P, T):
    """Rotation R (3,3) best aligning centred P onto centred T:  T ~ P @ R."""
    H = P.T @ T
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt


def body_frame(M):
    """The template's anatomical frame: (lateral, antero-posterior, dorso-ventral), orthonormal.

    Derived from the model itself rather than assumed from coordinate order. `lateral` comes from
    the mean right-minus-left displacement over every bilateral joint pair, which is the one
    direction the skeleton defines unambiguously; `antero-posterior` from head centroid to gaster
    centroid, orthogonalised against it; dorso-ventral completes the right-handed triple.
    """
    jn, V = M["jnames"], M["v_template"]
    J = M["Jr"] @ V
    lat = np.zeros(3)
    for j, n in enumerate(jn):
        if n.endswith("_r") and n[:-2] + "_l" in jn:
            lat += J[j] - J[jn.index(n[:-2] + "_l")]
    lat /= np.linalg.norm(lat)
    dom = M["dominant"]
    hc = V[np.isin(dom, [j for j, n in enumerate(jn) if n == "b_h"])].mean(0)
    gc = V[np.isin(dom, [j for j, n in enumerate(jn) if n.startswith("b_a_")])].mean(0)
    ap = gc - hc
    ap -= (ap @ lat) * lat
    ap /= np.linalg.norm(ap)
    dv = np.cross(lat, ap)
    return np.stack([lat, ap, dv])  # rows: lateral, antero-posterior, dorso-ventral


# anatomical role of each part axis, in the order body_frame returns
ROLES = ("wid", "len", "hei")


def template_part_axes(M):
    """For each part: template vertex indices, centroid, and axes labelled ANATOMICALLY.

    The axes are the part's own principal directions -- but they are assigned to
    width/length/height by which body direction each is most aligned with, NOT by eigenvalue
    order. That distinction is not cosmetic: on this template the head's largest principal
    extent is its WIDTH (0.3241) and its length is second (0.3213), a 0.9% margin. Sorting by
    eigenvalue would label the head's width as its length, invert every cephalic index, and do
    so unstably, since a 1% shape change flips the order. Assignment is one-to-one (greedy on
    the largest |dot| first) so two axes can never claim the same role.
    """
    jn = M["jnames"]
    dom = M["dominant"]
    V = M["v_template"]
    BF = body_frame(M)
    out = {}
    for part, pred in PART_DEFS.items():
        js = [j for j, n in enumerate(jn) if pred(n)]
        idx = np.where(np.isin(dom, js))[0]
        if idx.size < 10:
            continue
        P = V[idx]
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
        # one-to-one assignment of the 3 part axes to the 3 body directions
        D = np.abs(Vt @ BF.T)  # (part_axis, body_dir)
        axes = [None, None, None]
        used_a, used_b = set(), set()
        for _ in range(3):
            a, b = np.unravel_index(np.argmax(np.where(D > -1, D, -1)), D.shape)
            axes[b] = Vt[a]
            used_a.add(a)
            used_b.add(b)
            D[a, :] = -1
            D[:, b] = -1
        out[part] = dict(idx=idx, centroid=c, axes=np.stack(axes), roles=ROLES)
    return out


def part_dims(verts, M, tpa, lo=2.0, hi=98.0):
    """Extents of each part along the TEMPLATE's part axes, after per-specimen rigid alignment.

    Returns {f'{part}_ax{k}': (n,)} plus the alignment residual as a quality flag.
    """
    V0 = M["v_template"]
    out = {}
    for part, info in tpa.items():
        idx, c0, ax = info["idx"], info["centroid"], info["axes"]
        T = V0[idx] - c0
        P = verts[:, idx]  # (n,m,3)
        Pc = P - P.mean(1, keepdims=True)
        dims = np.zeros((P.shape[0], 3))
        resid = np.zeros(P.shape[0])
        for i in range(P.shape[0]):
            R = kabsch(Pc[i], T)
            Q = Pc[i] @ R  # specimen part, rotated into the template's part frame
            proj = Q @ ax.T  # (m,3) coordinates along the three fixed template axes
            dims[i] = np.percentile(proj, hi, axis=0) - np.percentile(proj, lo, axis=0)
            # scale-free alignment residual: how rigid was this part, really?
            s = np.linalg.norm(T) / max(np.linalg.norm(Q), 1e-12)
            resid[i] = np.linalg.norm(Q * s - T) / max(np.linalg.norm(T), 1e-12)
        for k, role in enumerate(info.get("roles", ROLES)):
            out[f"{part}_{role}"] = dims[:, k]
        # NOT a quality metric, despite how it is computed. `resid` is the scale-free residual
        # after the best RIGID alignment of this part onto the template's copy of it, so it
        # measures how much this specimen's part SHAPE departs from the template -- a genuine
        # morphological quantity. Measured: it predicts genus at 32.2% LOO-1NN (null 8.8%),
        # while true quality proxies do not (scan `radial_med` 11.9% p=0.34, L-R asymmetry 6.8%).
        # It is scale-free already, so it must NOT go through the Mosimann log-ratio step with
        # the lengths; it is appended as its own block after normalisation.
        out[f"shapedev_{part}"] = resid
    return out


# ------------------------------------------------------------------ taxonomy
_CASENT = re.compile(r"(CASENT|OKENT|ANTWEB|UCDC|MCZ)[0-9A-Za-z]*", re.I)


def parse_taxonomy(label, corpus):
    """-> dict(specimen, genus, species, source). Handles both corpora's naming schemes."""
    stem = label[:-4] if label.endswith(".obj") else label
    if corpus == "clean":
        # ALL_ANTS_CLEAN: 'acanthomyrmex', 'aenictus-binghamii', 'pheidole-barumtaum-major', '01'
        if stem[0].isdigit():
            return dict(specimen=stem, genus=None, species=None, source="clean")
        parts = stem.replace(" (1)", "").split("-")
        g = parts[0].lower()
        sp = f"{g}_{parts[1].lower()}" if len(parts) > 1 else None
        return dict(specimen=stem, genus=g, species=sp, source="clean")
    # worker: 'Acanthomyrmex_cf.ferox_CASENT0878066_processed'
    stem = stem.replace("_processed", "")
    m = _CASENT.search(stem)
    code = m.group(0) if m else None
    head = stem[: m.start()].rstrip("_") if m else stem
    toks = head.split("_")
    g = toks[0].lower()
    sp = None
    if len(toks) > 1:
        e = toks[1].lower().replace("cf.", "").replace("aff.", "").replace("nr.", "")
        if e and e not in ("sp.", "sp", "indet."):
            sp = f"{g}_{e}"
    return dict(specimen=code or stem, genus=g, species=sp, source="worker")


# ------------------------------------------------------------------ assembly
def measure_run(run, corpus, M=None, bones=None, tpa=None, stage="Stage_3_deform_fine"):
    """Measure one run npz. Returns (rows: list[dict], colnames: list[str])."""
    M = M or load_model()
    bones = bones if bones is not None else bone_table(M)
    tpa = tpa or template_part_axes(M)
    p = os.path.join(MOON, "runs", run, f"{stage}.npz")
    if not os.path.isfile(p):
        cand = sorted(f for f in os.listdir(os.path.join(MOON, "runs", run)) if f.endswith(".npz"))
        if not cand:
            return [], [], []
        p = os.path.join(MOON, "runs", run, cand[-1])
    d = np.load(p)
    rows, cols, devcols = measure_verts(
        d["verts"].astype(np.float64), [str(x) for x in d["labels"]], corpus, M, bones, tpa, run=run
    )
    for i, r in enumerate(rows):
        r["betas"] = d["betas"][i].tolist()
        r["log_beta_scales"] = d["log_beta_scales"][i].tolist()
    return rows, cols, devcols


def measure_verts(verts, labels, corpus, M=None, bones=None, tpa=None, run=""):
    """Measure a vertex array directly. Used for fitted runs AND for ground truth, so that a
    calibration compares like with like -- the same code path, not a re-implementation."""
    M = M or load_model()
    bones = bones if bones is not None else bone_table(M)
    tpa = tpa or template_part_axes(M)
    J = joints(verts, M["Jr"])
    L = bone_lengths(J, bones)
    dims = part_dims(verts, M, tpa)
    leaves = leaf_extents(verts, M)  # leaf structures have no bone; see leaf_joints()

    rows = []
    for i, lab in enumerate(labels):
        r = dict(run=run, label=lab, **parse_taxonomy(lab, corpus))
        for k, b in enumerate(bones):
            r[b["name"]] = float(L[i, k])
        for k, v in dims.items():
            r[k] = float(v[i])
        for k, v in leaves.items():
            r[k] = float(v[i])
        rows.append(r)
    # `cols` are the LENGTH measurements, which are size-bearing and go through the Mosimann
    # log-ratio step. `devcols` are already scale-free and are appended afterwards.
    cols = [b["name"] for b in bones] + [k for k in dims if not k.startswith("shapedev_")] + list(leaves)
    devcols = sorted(k for k in dims if k.startswith("shapedev_"))
    return rows, cols, devcols


def bilateral_pairs(cols):
    """[(left_col, right_col, base_name)] for every measurement that exists on both sides.

    Two naming schemes occur: bone columns end in `_r`/`_l` (`bone_l_3_fe_r`), while part-derived
    columns carry the side in the middle (`mandible_r_ax0`, `shapedev_mandible_r`).
    """
    cs = set(cols)
    out, seen = [], set()
    for c in cols:
        lc, base = None, None
        if c.endswith("_r"):
            lc, base = c[:-2] + "_l", c[:-2]
        elif "_r_" in c:
            lc, base = c.replace("_r_", "_l_", 1), c.replace("_r_", "_", 1)
        if lc in cs and base not in seen:
            seen.add(base)
            out.append((lc, c, base))
    return out


def symmetrise(rows, cols, key="asym_median"):
    """Average left/right into one column per structure; return (new_cols, asymmetry_by_row).

    `key` names the per-row field the asymmetry is stored under. It is a parameter because this
    function is normally called TWICE -- once for the length columns, once for the shape-dev
    columns -- and the second call overwrites the first's field. That has caused two separate
    errors: a scratch analysis that read a Cohen's d of 1.45 on "asymmetry" that was really
    mandible-only, and a shipped CSV column reading 0.81 where the truth was 0.05. Prefer the
    RETURNED array, which is always the array for this call's columns.

    Bilateral averaging halves the variance of an independent measurement error, and the
    left-right DIFFERENCE on the same specimen is a free repeatability estimate -- the two sides
    are the same biological structure measured twice by the same pipeline, so their disagreement
    is measurement noise with no biological component beyond genuine fluctuating asymmetry.
    Measured on 128 workers: median 4.17%, p90 8.63%.
    """
    pairs = bilateral_pairs(cols)
    paired = {c for lc, rc, _ in pairs for c in (lc, rc)}
    for r in rows:
        a = []
        for lc, rc, base in pairs:
            vl, vr = r.get(lc), r.get(rc)
            if vl is None or vr is None:
                continue
            r[base] = 0.5 * (vl + vr)
            m = 0.5 * (abs(vl) + abs(vr))
            if m > 1e-12:
                a.append(abs(vl - vr) / m)
        r[key] = float(np.median(a)) if a else np.nan
    asym = np.array([r[key] for r in rows])
    return [c for c in cols if c not in paired] + [b for _, _, b in pairs], asym


def log_shape_ratios(X):
    """Mosimann: divide each row by its geometric mean, then log. Removes isometric size.

    Returns (Z, log_size). Columns must be strictly positive.
    """
    X = np.asarray(X, dtype=np.float64)
    logX = np.log(np.maximum(X, 1e-12))
    g = logX.mean(1, keepdims=True)  # log geometric mean = size
    return logX - g, g[:, 0]


def mesh_quality(rows, M):
    """Per-specimen registration quality, from the FIT ITSELF rather than from surface proximity.

    Three quantities, because they fail in different ways:

      deform      mean free-form per-vertex displacement / specimen extent. What the parametric
                  model could not express once pose and shape were exhausted -- its own admission
                  of failure. But it is a MEAN, so a fit with a few vertices yanked far out looks
                  as good as a uniformly slightly-off one.
      edge        median |log(fitted edge length / template edge length)| after removing the
                  per-specimen median, i.e. how much the surface had to STRETCH non-uniformly.
                  A spike shows up here even when the mean displacement is small.
      normal      1 - mean cosine between the normals of adjacent faces, minus the template's own
                  value. Spikes are exactly local normal reversals, so this is the sharpest
                  detector of geometry pulled outward at a point.

    A composite z-score of the three is the default filter. Chamfer is deliberately NOT part of
    it: surface proximity is the metric the moonshot reports showed is gamed by shrink-wrapping,
    and a fit can buy low chamfer with precisely the deformation these three penalise.
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    faces = torch.tensor(np.asarray(M["dd"]["f"]).astype(np.int64), device=dev)
    # unique undirected edges, and the face pairs adjacent across each
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e, _ = e.sort(1)
    ue, inv = torch.unique(e, dim=0, return_inverse=True)
    fid = torch.arange(faces.shape[0], device=dev).repeat(3)
    order = torch.argsort(inv)
    inv_s, fid_s = inv[order], fid[order]
    starts = torch.cat([torch.tensor([0], device=dev), (inv_s[1:] != inv_s[:-1]).nonzero().squeeze(1) + 1])
    pairable = torch.cat([starts[1:] - starts[:-1], torch.tensor([len(inv_s) - starts[-1]], device=dev)]) == 2
    fa, fb = fid_s[starts[pairable]], fid_s[starts[pairable] + 1]

    def face_normals(v):
        t = v[:, faces]
        n = torch.cross(t[:, :, 1] - t[:, :, 0], t[:, :, 2] - t[:, :, 0], dim=-1)
        return torch.nn.functional.normalize(n, dim=-1)

    V0 = torch.tensor(M["v_template"], dtype=torch.float32, device=dev).unsqueeze(0)
    e0 = (V0[:, ue[:, 0]] - V0[:, ue[:, 1]]).norm(dim=-1)
    n0 = face_normals(V0)
    tmpl_rough = float(1 - (n0[:, fa] * n0[:, fb]).sum(-1).mean())

    out, cache = {}, {}
    for r in rows:
        cache.setdefault(r["run"], []).append(r["label"])
    for run, labs in cache.items():
        z = np.load(os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz"))
        idx = {str(x): k for k, x in enumerate(z["labels"])}
        v = torch.tensor(z["verts"], dtype=torch.float32, device=dev)
        ext = (v - v.mean(1, keepdim=True)).abs().amax(dim=(1, 2))
        dm = torch.tensor(z["deform_verts"], dtype=torch.float32, device=dev).norm(dim=-1).mean(1) / ext
        el = (v[:, ue[:, 0]] - v[:, ue[:, 1]]).norm(dim=-1)
        lr = torch.log(el / e0.clamp_min(1e-12))
        edge = (lr - lr.median(dim=1, keepdim=True).values).abs().median(dim=1).values
        n = face_normals(v)
        rough = 1 - (n[:, fa] * n[:, fb]).sum(-1).mean(1) - tmpl_rough
        for lab in labs:
            k = idx[lab]
            out[lab] = (float(dm[k]), float(edge[k]), float(rough[k]))
        del v, n
        torch.cuda.empty_cache()
    return out


def quality_composite(rows, M):
    """Composite quality: z-scored deform + edge distortion + normal roughness, summed."""
    q = mesh_quality(rows, M)
    A = np.array([q[r["label"]] for r in rows])
    Z = (A - A.mean(0)) / np.maximum(A.std(0), 1e-12)
    return Z.sum(1), A


def quality_filter(rows, M, top_pct):
    """Boolean mask keeping the best `top_pct`% of rows by registration quality.

    Quality is the deform/edge/normal composite (see `mesh_quality`) -- what the fit needed after
    the pose and shape spaces were exhausted -- NOT chamfer to the target. Surface proximity is
    gamed by shrink-wrapping, so a fit can buy a good chamfer with exactly the deformation this
    penalises. §7 of the report measures the difference.
    """
    if top_pct is None or top_pct >= 100:
        return np.ones(len(rows), bool)
    q, _ = quality_composite(rows, M)
    return q <= np.percentile(q, top_pct)
