"""Unsupervised part hierarchy from CONVEXITY — "where do the hulls separate?"

WHY THIS EXISTS
---------------
Two target-derived partitions have been tried and both failed, for opposite reasons:

  (a) GEODESIC branch decomposition (probe 14) -- REJECTED. It looks for a *bottleneck* in
      the geodesic field. On ethanol-preserved workers the legs touch the body, and once
      they touch there is genuinely no thin neck to cut: 58% of specimens reach six
      branches against an 80% gate, and the decomposition agrees with itself across two
      surface samplings only 20% of the time.

  (b) A LEARNED part field (fitter_3d/partfield.py) -- 91% held-out accuracy, and it LOST
      decisively when substituted into the fitter, because it was frozen. A frozen
      assignment cannot recover from its own errors, whereas the circular fit-derived
      partition reassigns a mis-assigned point once the fit moves.

This file attacks the same problem with the criterion the geodesic method lacks:
**convexity**, not connectivity. A leg pressed flat against a thorax has no geodesic
bottleneck, but the *union* of the leg and the thorax is still strongly non-convex -- its
convex hull has to bridge the wedge of empty space between them, and that bridge is
measurable even when the two parts are welded together. That is the sense in which hulls
"separate": two parts belong together when their union is nearly as convex as the parts
were separately, and belong apart when merging them manufactures empty space.

That criterion is *local in convexity* but *global in extent*, which is exactly the axis on
which (a) was blind.

WHAT IT PRODUCES
----------------
A binary merge tree (dendrogram) over an over-decomposition of the target, whose merge
HEIGHT is the concavity created by that merge. Cutting the tree at any height or any k
yields a partition, so the hierarchy comes with a granularity knob rather than a fixed part
count -- which is what the hierarchical fitter wants, since it works coarse-to-fine (body
first, then legs, then distal segments).

THE PIPELINE
------------
  1. OVER-DECOMPOSE with CoACD (Wei et al., SIGGRAPH 2022): approximate convex decomposition
     driven by a Monte-Carlo tree search over cutting planes, minimising a concavity measure
     aware of the *volume* the hull adds, not just surface deviation. Run deliberately TIGHT
     so it over-segments -- the atoms are raw material, and an atom straddling two anatomical
     parts can never be un-straddled by merging. CoACD's own cut tree is not exposed by the
     library, and rebuilding the hierarchy by merging is both simpler and lets us swap the
     merge criterion, which is the actual experiment.

  2. ASSIGN every target surface sample to a hull by the exact convex-hull distance
     `max_i (a_i . x - b_i)` over that hull's facet planes (negative inside, positive
     outside). One matmul over all hulls at once, and exact -- unlike a nearest-hull-vertex
     heuristic, which mis-assigns points lying over large facets.

  3. MERGE agglomeratively over the ADJACENCY graph of those atoms, cheapest-concavity
     first, recording the merge height. Three interchangeable criteria (see `MERGE_CRITERIA`)
     because "where hulls separate" admits three defensible formalisations, and finding out
     which one holds on real scans is the point of the exercise.

PRE-REGISTERED GATES (fixed before any result was seen)
-------------------------------------------------------
Deliberately the gates that killed the geodesic method, so the comparison is fair.

G1  REPRODUCIBILITY. Two independent surface samplings of the same scan, decomposed
    independently, must agree on >= 90% of points after optimal label matching. Geodesic
    scored 20%. A partition that changes with the point sample cannot anchor a data term.
G2  PART RECOVERY. On the synthetic corpus, where ground-truth anatomical labels are exact,
    the cut at k = n_true_parts must reach adjusted Rand index >= 0.5 against the true
    parts, and >= 6 distinct legs must each be dominated by a distinct cluster.
G3  THE TOUCHING-LEG CLAIM. On the specimens where probe 14 collapsed to <= 2 geodesic
    branches, the tree must still separate >= 6 limb clusters. This is the specific claim
    convexity makes over connectivity; if it fails, the claim was wrong.
G4  COST. Under 60 s per specimen on CPU, so it can run over the 757-worker corpus.

Sources
-------
CoACD -- Wei, Liu, Chen, Wang, Ma, Zhou, "Approximate Convex Decomposition for 3D Meshes
  with Collision-Aware Concavity and Tree Search", SIGGRAPH 2022.
V-HACD -- Mamou & Ghorbel, "A simple and efficient approach for 3D mesh approximate convex
  decomposition", ICIP 2009 (the hierarchical-merge formulation this file's tree follows).
Weak convexity / visibility -- Asafi, Goren, Cohen-Or, "Weak Convex Decomposition by
  Lines-of-sight", SGP 2013: two points belong to the same part if the segment between them
  stays inside the shape. This is the `visibility` criterion, and it is the one designed
  specifically for parts that touch.
Shape Diameter Function -- Shapira, Shamir, Cohen-Or, "Consistent mesh partitioning and
  skeletonisation using the shape diameter function", The Visual Computer 2008.
"""

import heapq

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

MERGE_CRITERIA = ("concavity", "volume", "visibility", "hybrid", "spectral")


# ======================================================================================
# sampling and normalisation
# ======================================================================================
def normalise(v):
    """Centre and divide by max|coord| -- byte-identical to what `utils.load_meshes` does
    to every target mesh, so a decomposition computed here lives in the fitter's frame."""
    v = np.asarray(v, dtype=np.float64)
    c = v.mean(0)
    v = v - c
    s = np.abs(v).max()
    return v / s, c, s


def sample_surface(mesh, n, seed=0):
    """Area-weighted surface samples.

    Area-weighted, not per-vertex: the fitter's own data term was found to be biased by
    vertex-density sampling (REPORT §1), and a decomposition meant to shape that data term
    must be built on the same measure.
    """
    import trimesh

    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=int(seed))
    return np.asarray(pts, dtype=np.float64), np.asarray(fid)


# ======================================================================================
# step 1-2: over-decompose, and assign points to hulls
# ======================================================================================
def coacd_hulls(
    verts,
    faces,
    threshold=0.02,
    seed=0,
    max_hulls=-1,
    mcts_iterations=100,
    quiet=True,
    merge=False,
    preprocess_resolution=100,
):
    """Approximate convex decomposition. Returns a list of (hull_verts, hull_faces).

    TWO DEFAULTS ARE DELIBERATELY OVERRIDDEN, and both were measured to matter:

    `merge=False` (CoACD defaults to True). CoACD merges hulls *after* decomposing, which
    destroys the leaf set that hull membership is meant to define here, and the VisACD authors
    report it yields intersecting hulls in 35% of cases.

    `preprocess_resolution=100` (CoACD defaults to 50). The default is a voxel remesh coarse
    enough to weld a touching limb to the body BEFORE the algorithm sees the crease -- which
    would destroy exactly the geometry this whole approach exists to exploit.

    Measured effect of fixing both, plus tightening the threshold to 0.01, on how cleanly the
    atoms respect anatomy (4 synthetic specimens, 16-group labelling):

        defaults (merge=True, prep=50, thr 0.03)   99 atoms   purity 0.854   straddle 0.397
        merge=False + prep=100 + thr 0.01         450 atoms   purity 0.914   straddle 0.249

    Worth having; not sufficient. A quarter of the surface still spans more than one
    anatomical part, because CoACD optimises COLLISION concavity and its cutting planes have
    no reason to follow anatomy. See diagnostics/moonshot/hull/REPORT_HULL.md §3.
    """
    import coacd

    if quiet:
        coacd.set_log_level("error")
    m = coacd.Mesh(np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32))
    parts = coacd.run_coacd(
        m,
        threshold=threshold,
        max_convex_hull=max_hulls,
        preprocess_mode="auto",
        preprocess_resolution=int(preprocess_resolution),
        merge=bool(merge),
        mcts_nodes=20,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=3,
        seed=int(seed),
    )
    return [(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)) for v, f in parts]


def hull_planes(hv):
    """Outward facet planes (A, b) of the convex hull of `hv`, with A x - b <= 0 inside.

    Returns None when the point set is degenerate (fewer than 4 points, or coplanar), which
    happens for thin scan debris and must not abort a whole decomposition.
    """
    if len(hv) < 4:
        return None
    try:
        eq = ConvexHull(hv).equations  # rows [n_x, n_y, n_z, off], n.x + off <= 0 inside
    except Exception:
        return None
    return eq[:, :3], -eq[:, 3]


def assign_to_hulls(points, hulls, chunk=20000):
    """Label each point by the hull that best contains it.

    Distance to a convex hull is exactly `max_i (a_i . x - b_i)` over its facet planes:
    negative inside (equal to minus the distance to the nearest face), positive outside (a
    tight lower bound on the true distance). The argmin over hulls therefore assigns
    interior points to their container, and exterior points -- scan surface lying slightly
    outside every hull, common because CoACD only approximates -- to the hull they are
    closest to escaping. One matmul per chunk over all hull facets at once.
    """
    A_list, b_list, owner = [], [], []
    for i, (hv, _) in enumerate(hulls):
        pl = hull_planes(hv)
        if pl is None:
            continue  # degenerate (coplanar) hull: it owns no points
        A, b = pl
        A_list.append(A)
        b_list.append(b)
        owner.append(np.full(len(A), i, dtype=np.int64))
    n_h = len(hulls)
    if not A_list:
        return np.zeros(len(points), dtype=np.int64), 1
    A = np.concatenate(A_list, 0)
    b = np.concatenate(b_list, 0)
    owner = np.concatenate(owner, 0)
    # facets are appended hull by hull, so `owner` is already sorted and the per-hull maximum
    # is a segmented reduction. np.maximum.reduceat over those segments is ~10x faster than
    # np.maximum.at, which is a scalar-loop ufunc and dominated the runtime (3.9 s/specimen).
    starts = np.searchsorted(owner, np.arange(owner[-1] + 1), side="left")
    present = np.unique(owner)
    starts = starts[present]
    lab = np.empty(len(points), dtype=np.int64)
    for s in range(0, len(points), chunk):
        p = points[s : s + chunk]
        d = (p @ A.T - b[None, :]).T  # (F, P) signed plane distance
        seg = np.maximum.reduceat(d, starts, axis=0)  # (n_present, P) per-hull hull distance
        lab[s : s + chunk] = present[seg.argmin(0)]
    return lab, n_h


# ======================================================================================
# convexity measures
# ======================================================================================
def _hull_volume(p):
    """Volume of the convex hull of a point set; 0 if degenerate."""
    if len(p) < 4:
        return 0.0
    try:
        return float(ConvexHull(p).volume)
    except Exception:
        return 0.0


def _sample_hull_surface(p, n=1200, rng=None):
    """Area-weighted samples on the convex hull of `p`. Empty if degenerate."""
    if len(p) < 4:
        return np.zeros((0, 3))
    try:
        h = ConvexHull(p)
    except Exception:
        return np.zeros((0, 3))
    tri = h.points[h.simplices]  # (F, 3, 3)
    e0, e1 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
    tot = area.sum()
    if tot <= 0:
        return np.zeros((0, 3))
    rng = rng if rng is not None else np.random.default_rng(0)
    fi = rng.choice(len(tri), size=n, p=area / tot)
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    flip = (u + v).ravel() > 1
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    return tri[fi, 0] + u * e0[fi] + v * e1[fi]


def concavity(pts, n_hull_samples=1200, rng=None):
    """How much empty space the convex hull of `pts` adds, in shape units.

    Sample the hull's own surface and measure, for each sample, the distance to the nearest
    actual point. A tight, near-convex part leaves the hull hugging the surface and scores
    ~0. Merging a leg into the thorax makes the hull span the wedge between them, and those
    bridging facets sit far from any real surface. Reported at the 95th percentile rather
    than the max, so one spurious scan point cannot set the score.

    This is the surface-deviation half of CoACD's Hb concavity (Wei et al. 2022); the
    volumetric half is `volume_overhead`, kept separate so the two can be compared rather
    than blended by fiat.
    """
    if len(pts) < 4:
        return 0.0
    hs = _sample_hull_surface(pts, n_hull_samples, rng)
    if len(hs) == 0:
        return 0.0
    d, _ = cKDTree(pts).query(hs, k=1)
    return float(np.percentile(d, 95))


def bridge_cost(pa, pb, plane_a, plane_b, vol_a, vol_b, rng, n_mc=4000):
    """THE measure: how much empty space merging manufactures, relative to the smaller part.

    Merging A and B creates a region that is inside hull(A u B) but inside neither hull(A)
    nor hull(B). That region is the "bridge" -- literally the wedge of air the merged hull
    has to swallow. Estimated by Monte-Carlo over the union's bounding box, as a direct
    indicator (no subtraction of large nearly-equal volumes, so it stays stable).

    NORMALISED BY min(Va, Vb), AND THAT NORMALISATION IS THE WHOLE POINT. An absolute or
    Vu-relative measure is dominated by the larger cluster: absorbing a leg into a thorax
    barely changes the thorax's hull, so the greedy merge eats every appendage for free and
    the k=7 cut degenerates to one giant blob plus debris. Measured, before the fix: 5,425
    of 8,000 points in a single cluster. Dividing by the SMALLER part's own volume asks the
    scale-free question instead -- "is this bridge large compared to the thing being merged?"
    -- so a leg joining a thorax costs ~1 while two collinear leg segments cost ~0.

    This is the volumetric half of CoACD's concavity (Wei et al. 2022), and the same
    normalisation V-HACD uses to decide which pair of parts to merge next.
    """
    if plane_a is None or plane_b is None:
        return 1.0
    pu = np.concatenate([pa, pb], 0)
    pl_u = hull_planes(pu)
    if pl_u is None:
        return 1.0
    lo, hi = pu.min(0), pu.max(0)
    span = np.maximum(hi - lo, 1e-9)
    x = lo + rng.random((n_mc, 3)) * span
    in_u = ((x @ pl_u[0].T - pl_u[1]) <= 0).all(1)
    in_a = ((x @ plane_a[0].T - plane_a[1]) <= 0).all(1)
    in_b = ((x @ plane_b[0].T - plane_b[1]) <= 0).all(1)
    v_box = float(np.prod(span))
    bridge = float((in_u & ~in_a & ~in_b).mean()) * v_box
    denom = max(min(vol_a, vol_b), 1e-12)
    return float(bridge / denom)


class VisibilityOracle:
    """Mutual line-of-sight between surface points (Asafi et al., SGP 2013).

    Two surface points are mutually visible when the open segment between them does not
    cross the surface. A part is "weakly convex" when most of its point pairs see each
    other. Crucially this needs NO bottleneck: a leg lying flat against a thorax still fails
    the test, because a segment from the far side of the leg to the far side of the thorax
    must exit through the leg's surface and re-enter through the thorax's.

    Implemented over an Embree BVH on the original scan triangles. Non-watertightness is
    tolerated because the test counts *crossings of the surface*, not inside/outside.
    """

    def __init__(self, mesh, eps_frac=0.02):
        self.eps_frac = eps_frac
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector
        except Exception:
            from trimesh.ray.ray_triangle import RayMeshIntersector
        self.rmi = RayMeshIntersector(mesh)

    def fraction_visible(self, pa, pb, rng, n_pairs=160):
        """Fraction of sampled (a, b) pairs with unobstructed line of sight."""
        if len(pa) == 0 or len(pb) == 0:
            return 0.0
        n = int(min(n_pairs, max(len(pa), len(pb))))
        a = pa[rng.integers(0, len(pa), n)]
        b = pb[rng.integers(0, len(pb), n)]
        d = b - a
        L = np.linalg.norm(d, axis=1)
        ok = L > 1e-9
        if not ok.any():
            return 1.0
        a, d, L = a[ok], d[ok], L[ok]
        u = d / L[:, None]
        eps = self.eps_frac * L
        origins = a + u * eps[:, None]  # start off the source surface, so it is not self-hit
        loc, idx, _ = self.rmi.intersects_location(origins, u, multiple_hits=False)
        t = np.full(len(origins), np.inf)
        if len(idx):
            t[idx] = np.linalg.norm(loc - origins[idx], axis=1)
        return float((t >= (L - 2.0 * eps)).mean())


# ======================================================================================
# step 3: the merge tree
# ======================================================================================
class Hierarchy:
    """A binary merge tree over atoms, with cut-anywhere semantics.

    `linkage` is scipy-compatible: row i merges clusters `linkage[i, 0]` and `linkage[i, 1]`
    into cluster `n_atoms + i` at height `linkage[i, 2]`, containing `linkage[i, 3]` atoms.
    Heights are made monotone by max-ing with the children's, which a dendrogram requires
    and a raw concavity score does not automatically satisfy.
    """

    def __init__(self, linkage, atom_of_point, points, n_atoms, meta=None):
        self.linkage = np.asarray(linkage, dtype=np.float64).reshape(-1, 4)
        self.atom_of_point = np.asarray(atom_of_point, dtype=np.int64)
        self.points = np.asarray(points, dtype=np.float64)
        self.n_atoms = int(n_atoms)
        self.meta = meta or {}
        self._tree = cKDTree(self.points)

    # ---------------------------------------------------------------- cuts
    def cut_atoms(self, k):
        """Cluster id per ATOM for a k-cluster cut, ordered largest cluster first.

        Ordering by size makes the labelling deterministic across runs, which G1 needs:
        two samplings of the same scan must be comparable without an arbitrary permutation.
        """
        n = self.n_atoms
        n_apply = int(max(0, min(len(self.linkage), n - k)))
        groups = {i: [i] for i in range(n)}
        for r in range(n_apply):
            a, b = int(self.linkage[r, 0]), int(self.linkage[r, 1])
            groups[n + r] = groups.pop(a) + groups.pop(b)
        lab = np.full(n, -1, dtype=np.int64)
        for c, members in enumerate(sorted(groups.values(), key=len, reverse=True)):
            lab[np.asarray(members, dtype=np.int64)] = c
        return lab

    def cut(self, k):
        """Cluster id per POINT for a k-cluster cut."""
        return self.cut_atoms(k)[self.atom_of_point]

    def cut_height(self, h):
        """Cluster id per point, undoing every merge whose height exceeds `h`."""
        if not len(self.linkage):
            return self.cut(self.n_atoms)
        k = self.n_atoms - int((self.linkage[:, 2] <= h).sum())
        return self.cut(max(k, 1))

    def heights(self):
        return self.linkage[:, 2] if len(self.linkage) else np.zeros(0)

    def transfer(self, query, k=None, labels=None):
        """Label arbitrary query points by nearest decomposed point.

        This is how the decomposition reaches the fitter: the fitter resamples its target at
        every reassignment, so the partition must answer for points it has never seen.
        """
        lab = labels if labels is not None else self.cut(k)
        _, j = self._tree.query(np.asarray(query, dtype=np.float64), k=1)
        return lab[j]


def _components(nodes, nbr):
    seen, comps = set(), []
    for a in nodes:
        if a in seen:
            continue
        stack, comp = [a], []
        seen.add(a)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in nbr[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    return comps


def _bridge_components(nodes, nbr, pts_of):
    """Connect a disconnected adjacency graph, nearest component first.

    Worker scans carry a median of 34.5 vertex-connected components (REPORT §6), mostly
    debris holding 0.3% of the area. Without bridging, agglomeration returns a forest and
    `cut(k)` is dominated by debris singletons rather than by anatomy.
    """
    comps = _components(nodes, nbr)
    while len(comps) > 1:
        big = max(range(len(comps)), key=lambda i: len(comps[i]))
        bp = np.concatenate([pts_of[a] for a in comps[big]], 0)
        blab = np.concatenate([np.full(len(pts_of[a]), a) for a in comps[big]])
        bt = cKDTree(bp)
        best = None
        for ci, comp in enumerate(comps):
            if ci == big:
                continue
            cp = np.concatenate([pts_of[a] for a in comp], 0)
            clab = np.concatenate([np.full(len(pts_of[a]), a) for a in comp])
            d, j = bt.query(cp, k=1)
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), int(clab[i]), int(blab[j[i]]), ci)
        _, u, v, ci = best
        nbr[u].add(v)
        nbr[v].add(u)
        comps[big] = comps[big] + comps[ci]
        comps.pop(ci)


class _Cluster:
    """A node of the agglomeration: its points, and the hull quantities derived from them."""

    __slots__ = ("pts", "planes", "vol", "radius", "height", "members")

    def __init__(self, pts, members, height=0.0):
        self.pts = pts
        self.members = members
        self.height = height
        self.planes = hull_planes(pts)
        self.vol = _hull_volume(pts)
        # characteristic size: median distance to the centroid. Used to make the surface
        # concavity criterion scale-free in the same way bridge_cost is volume-scale-free.
        self.radius = float(np.median(np.linalg.norm(pts - pts.mean(0), axis=1))) if len(pts) else 0.0


class _CostModel:
    """Merge cost between two clusters, under the selected notion of 'hulls separate'.

    EVERY criterion here is normalised by the SMALLER part. That is not cosmetic: an
    un-normalised cost makes absorbing an appendage into the body nearly free, and greedy
    agglomeration then produces one giant cluster plus debris (measured: 5,425/8,000 points
    in one cluster at k=7).
    """

    def __init__(self, criterion, rng, vis, max_pts, vis_pairs, n_mc=4000):
        self.criterion = criterion
        self.rng = rng
        self.vis = vis
        self.max_pts = max_pts
        self.vis_pairs = vis_pairs
        self.n_mc = n_mc

    def union_pts(self, pa, pb):
        pu = np.concatenate([pa, pb], 0)
        if len(pu) > 2 * self.max_pts:
            pu = pu[self.rng.choice(len(pu), 2 * self.max_pts, replace=False)]
        return pu

    def _bridge(self, a, b):
        return bridge_cost(a.pts, b.pts, a.planes, b.planes, a.vol, b.vol, self.rng, self.n_mc)

    def _surface(self, a, b):
        """Added hull-surface deviation, in units of the smaller part's own radius."""
        pu = self.union_pts(a.pts, b.pts)
        cu = concavity(pu, rng=self.rng)
        base = max(concavity(a.pts, rng=self.rng), concavity(b.pts, rng=self.rng))
        r = max(min(a.radius, b.radius), 1e-9)
        return float(max(cu - base, 0.0) / r)

    def __call__(self, a, b):
        if self.criterion == "visibility":
            return 1.0 - self.vis.fraction_visible(a.pts, b.pts, self.rng, self.vis_pairs)
        if self.criterion == "concavity":
            return self._surface(a, b)
        if self.criterion == "volume":
            return self._bridge(a, b)
        # hybrid: the two are blind in opposite directions -- a bridge that is thin but wide
        # barely registers volumetrically, a long thin spur barely registers on the surface
        # deviation -- so take the conservative max.
        return max(self._bridge(a, b), self._surface(a, b))


def build_merge_tree(
    points,
    atom_of_point,
    n_atoms,
    criterion="hybrid",
    mesh=None,
    adjacency_eps=None,
    min_atom_pts=4,
    max_atom_pts=1200,
    seed=0,
    vis_pairs=160,
):
    """Agglomerate atoms cheapest-concavity-first over their adjacency graph.

    Only ADJACENT clusters may merge. Unrestricted, the cheapest merge is often two far-apart
    thin structures whose union hull happens to be small (two contralateral tarsi), which is
    anatomically absurd and destroys the tree. Adjacency after a merge is the union of the
    children's neighbours -- standard for a constrained agglomeration, and it keeps the loop
    at O(n_atoms^2) with small constants.

    Atoms too small to have a hull are folded into their nearest surviving atom and the
    survivors are RENUMBERED, so `Hierarchy.cut(k)` really does return k non-empty clusters.
    (Leaving them in as never-merging singletons silently inflates every k.)
    """
    if criterion not in MERGE_CRITERIA:
        raise ValueError(f"criterion must be one of {MERGE_CRITERIA}, got {criterion!r}")
    if criterion == "spectral":
        if mesh is None:
            raise ValueError("criterion='spectral' needs the source mesh for ray casting")
        return build_spectral_tree(points, atom_of_point, n_atoms, mesh, seed=seed)
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float64)
    atom_of_point = np.asarray(atom_of_point, dtype=np.int64)

    members = [np.nonzero(atom_of_point == a)[0] for a in range(n_atoms)]
    keep = [a for a in range(n_atoms) if len(members[a]) >= min_atom_pts]
    if len(keep) <= 1:
        return Hierarchy(np.zeros((0, 4)), np.zeros(len(points), dtype=np.int64), points, 1)

    # ---- renumber survivors to 0..n-1 and absorb the dropped atoms by proximity
    remap = {a: i for i, a in enumerate(keep)}
    aop = np.full(len(points), -1, dtype=np.int64)
    for a, i in remap.items():
        aop[members[a]] = i
    if (aop < 0).any():
        src = aop >= 0
        _, j = cKDTree(points[src]).query(points[~src], k=1)
        aop[~src] = aop[src][j]

    n = len(keep)
    pts_of = {}
    for i in range(n):
        idx = np.nonzero(aop == i)[0]
        if len(idx) > max_atom_pts:
            idx = rng.choice(idx, max_atom_pts, replace=False)
        pts_of[i] = points[idx]

    # ---- adjacency: atoms whose point sets come within eps
    if adjacency_eps is None:
        d, _ = cKDTree(points).query(points, k=2)
        adjacency_eps = 4.0 * float(np.median(d[:, 1]))
    trees = {i: cKDTree(pts_of[i]) for i in range(n)}
    nbr = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            # count_neighbors, NOT query_ball_tree. query_ball_tree returns one list PER POINT,
            # so `if tree.query_ball_tree(...)` is truthy whenever the source has any points at
            # all -- it was marking every pair adjacent (measured: mean degree 128 of 129
            # atoms, i.e. a complete graph), which silently voided the whole adjacency
            # constraint and let contralateral limbs merge.
            if trees[i].count_neighbors(trees[j], adjacency_eps) > 0:
                nbr[i].add(j)
                nbr[j].add(i)
    _bridge_components(list(range(n)), nbr, pts_of)

    vis = None
    if criterion == "visibility":
        if mesh is None:
            raise ValueError("criterion='visibility' needs the source mesh for ray casting")
        vis = VisibilityOracle(mesh)
    cost = _CostModel(criterion, rng, vis, max_atom_pts, vis_pairs)

    cl = {i: _Cluster(pts_of[i], [i]) for i in range(n)}
    active = set(range(n))
    cache, heap = {}, []

    def push(a, b):
        key = (min(a, b), max(a, b))
        c = cost(cl[a], cl[b])
        cache[key] = c
        heapq.heappush(heap, (c, key[0], key[1]))

    for i in range(n):
        for j in nbr[i]:
            if i < j:
                push(i, j)

    nxt = n
    linkage = []
    while len(active) > 1 and heap:
        c, a, b = heapq.heappop(heap)
        if a not in active or b not in active or b not in nbr[a]:
            continue
        if cache.get((min(a, b), max(a, b))) != c:
            continue  # stale entry, superseded by a re-push
        h = max(c, cl[a].height, cl[b].height)  # enforce a monotone dendrogram
        new = nxt
        nxt += 1
        linkage.append([a, b, h, len(cl[a].members) + len(cl[b].members)])
        cl[new] = _Cluster(cost.union_pts(cl[a].pts, cl[b].pts), cl[a].members + cl[b].members, height=h)
        nbr[new] = (nbr[a] | nbr[b]) - {a, b}
        for o in nbr[new]:
            nbr[o] = (nbr[o] - {a, b}) | {new}
        active.discard(a)
        active.discard(b)
        active.add(new)
        del cl[a], cl[b]
        for o in nbr[new]:
            if o in active:
                push(new, o)

    L = np.asarray(linkage, dtype=np.float64) if linkage else np.zeros((0, 4))
    return Hierarchy(
        L, aop, points, n, meta=dict(criterion=criterion, adjacency_eps=float(adjacency_eps), n_atoms_kept=n)
    )


# ======================================================================================
# the top-down alternative: recursive spectral bisection on weak-convexity affinity
# ======================================================================================
def visibility_affinity(points, atom_of_point, n_atoms, mesh, rng, n_pairs=140, sigma=0.35, max_pts=400):
    """Atom-by-atom weak-convexity affinity: how much of one atom can see the other.

    W[i, j] = fraction of sampled point pairs (one from each atom) whose connecting segment
    does not cross the surface, damped by a spatial Gaussian on the centroid distance. The
    damping keeps the normalised cut from pairing two far-apart atoms that happen to see
    each other across open space (a tarsus and the contralateral tarsi, under an ant held
    clear of any surface), which is a real failure of pure line-of-sight on non-watertight
    scans.
    """
    vis = VisibilityOracle(mesh)
    pts_of, cen = {}, np.zeros((n_atoms, 3))
    for a in range(n_atoms):
        idx = np.nonzero(atom_of_point == a)[0]
        if len(idx) > max_pts:
            idx = rng.choice(idx, max_pts, replace=False)
        pts_of[a] = points[idx]
        cen[a] = points[idx].mean(0) if len(idx) else 0.0
    W = np.eye(n_atoms)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if len(pts_of[i]) == 0 or len(pts_of[j]) == 0:
                continue
            W[i, j] = W[j, i] = vis.fraction_visible(pts_of[i], pts_of[j], rng, n_pairs)
    d = np.linalg.norm(cen[:, None] - cen[None, :], axis=-1)
    return W * np.exp(-(d**2) / (2.0 * sigma**2))


def _ncut_bisect(W):
    """Split a set of atoms by the sign of the Fiedler vector of the normalised Laplacian.

    Returns (mask, ncut_value). `ncut` is the normalised-cut objective of the split: LOW
    means the two halves barely see each other, i.e. this is a good place to separate.
    """
    n = len(W)
    if n < 2:
        return None, np.inf
    deg = W.sum(1)
    if not np.all(deg > 1e-12):
        deg = np.maximum(deg, 1e-12)
    dm = 1.0 / np.sqrt(deg)
    L = np.eye(n) - (W * dm[:, None]) * dm[None, :]  # symmetric normalised Laplacian
    try:
        w, v = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return None, np.inf
    if len(w) < 2:
        return None, np.inf
    fiedler = v[:, 1] * dm  # back to the unnormalised space
    # search the best threshold along the Fiedler ordering rather than taking sign(f): the
    # sign split is only optimal for a perfectly balanced cut, and limbs are small
    order = np.argsort(fiedler)
    best, best_m = np.inf, None
    tot = W.sum()
    for t in range(1, n):
        m = np.zeros(n, dtype=bool)
        m[order[:t]] = True
        cut = W[m][:, ~m].sum()
        assoc_a, assoc_b = W[m].sum(), W[~m].sum()
        if assoc_a <= 1e-12 or assoc_b <= 1e-12:
            continue
        nc = cut / assoc_a + cut / assoc_b
        if nc < best:
            best, best_m = nc, m.copy()
    _ = tot
    return best_m, float(best)


def build_spectral_tree(points, atom_of_point, n_atoms, mesh, seed=0, sigma=0.35, n_pairs=140, min_atoms=1):
    """Top-down hierarchy by recursive normalised-cut bisection on weak-convexity affinity.

    WHY TOP-DOWN. Greedy bottom-up merging was measured to fail on exactly the structure
    this project cares about: a leg is CONTIGUOUS with the thorax, so every individual
    leg-atom-to-body merge is locally cheap and the limb is absorbed one atom at a time.
    Measured at k=13 on both a synthetic specimen and a real Cephalotes: 6,500-7,400 of
    8,000 points ended in a single cluster under the volumetric, surface and hybrid
    criteria. No purely LOCAL criterion can prevent that, because locally there is nothing
    to see. A normalised cut is global -- it scores a whole partition of the affinity graph
    at once -- so the leg is separated as a unit or not at all.

    This is the formulation of Asafi, Goren and Cohen-Or (SGP 2013), whose entire premise is
    that line-of-sight weak convexity separates parts that TOUCH, which is precisely where
    the geodesic bottleneck method failed (REPORT §5.2).

    Splits are taken best-first (lowest normalised cut anywhere in the current forest), so
    `cut(k)` returns the k most confident separations rather than an arbitrary k.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float64)
    atom_of_point = np.asarray(atom_of_point, dtype=np.int64)

    members = [np.nonzero(atom_of_point == a)[0] for a in range(n_atoms)]
    keep = [a for a in range(n_atoms) if len(members[a]) >= 4]
    remap = {a: i for i, a in enumerate(keep)}
    aop = np.full(len(points), -1, dtype=np.int64)
    for a, i in remap.items():
        aop[members[a]] = i
    if (aop < 0).any():
        src = aop >= 0
        _, j = cKDTree(points[src]).query(points[~src], k=1)
        aop[~src] = aop[src][j]
    n = len(keep)
    if n <= 1:
        return Hierarchy(np.zeros((0, 4)), np.zeros(len(points), dtype=np.int64), points, 1)

    W = visibility_affinity(points, aop, n, mesh, rng, n_pairs=n_pairs, sigma=sigma)

    # ---- recursive best-first bisection, recording the split order
    splits = []
    heap = []
    groups = {0: np.arange(n)}
    nxt_g = 1

    def try_split(gid):
        idx = groups[gid]
        if len(idx) <= min_atoms:
            return
        m, nc = _ncut_bisect(W[np.ix_(idx, idx)])
        if m is None or not np.isfinite(nc):
            return
        heapq.heappush(heap, (nc, gid, idx[m].copy(), idx[~m].copy()))

    try_split(0)
    while heap:
        nc, gid, ia, ib = heapq.heappop(heap)
        if gid not in groups or len(ia) == 0 or len(ib) == 0:
            continue
        del groups[gid]
        ga, gb = nxt_g, nxt_g + 1
        nxt_g += 2
        groups[ga], groups[gb] = ia, ib
        splits.append((gid, ga, gb, nc))
        try_split(ga)
        try_split(gb)

    # ---- convert the top-down split order into a bottom-up scipy-style MERGE linkage.
    #
    # `Hierarchy.cut(k)` applies linkage rows 0 .. n-k-1, so the rows must be ordered
    # WORST-MERGE-FIRST for `cut(k)` to mean "keep the k most confident splits":
    #   rows first          collapse each unsplit leaf group's atoms into one node
    #   rows last           undo the splits, worst normalised cut first, best cut last
    # Children are always available when needed: the split heap pops in increasing ncut, and
    # a child split is only pushed once its parent has been popped, so every child carries a
    # larger split index than its parent and is therefore merged first.
    linkage = []
    node = {}
    nxt = n
    for gid, idx in groups.items():
        cur = int(idx[0])
        for a in idx[1:]:
            linkage.append([cur, int(a), 0.0, 0])
            cur = nxt
            nxt += 1
        node[gid] = cur
    for gid, ga, gb, nc in reversed(splits):
        linkage.append([node[ga], node[gb], 0.0, 0])
        node[gid] = nxt
        nxt += 1

    L = np.asarray(linkage, dtype=np.float64).reshape(-1, 4)
    if len(L):
        # strictly increasing so the dendrogram is monotone; the ORDER carries the meaning
        # here, not the magnitude, because a normalised cut is not a merge cost
        L[:, 2] = np.arange(1, len(L) + 1, dtype=np.float64)
        size = np.ones(n + len(L))
        for i, (a, b, _, _) in enumerate(L):
            size[n + i] = size[int(a)] + size[int(b)]
            L[i, 3] = size[n + i]
    return Hierarchy(L, aop, points, n, meta=dict(criterion="spectral", sigma=float(sigma), n_splits=len(splits)))


# ======================================================================================
# convenience: one call from a mesh to a hierarchy
# ======================================================================================
def decompose_mesh(
    mesh,
    n_points=20000,
    coacd_threshold=0.02,
    criterion="hybrid",
    seed=0,
    max_hulls=-1,
    mcts_iterations=100,
    normalise_frame=True,
):
    """Full pipeline on a trimesh object, in the loader's normalised frame."""
    import trimesh

    v = np.asarray(mesh.vertices, dtype=np.float64)
    if normalise_frame:
        v, _, _ = normalise(v)
    f = np.asarray(mesh.faces)
    nm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    pts, _ = sample_surface(nm, n_points, seed=seed)
    hulls = coacd_hulls(
        v, f, threshold=coacd_threshold, seed=seed, max_hulls=max_hulls, mcts_iterations=mcts_iterations
    )
    atom, n_atoms = assign_to_hulls(pts, hulls)
    tree = build_merge_tree(pts, atom, n_atoms, criterion=criterion, mesh=nm, seed=seed)
    tree.meta.update(n_hulls=len(hulls), n_points=int(n_points), coacd_threshold=float(coacd_threshold))
    return tree
