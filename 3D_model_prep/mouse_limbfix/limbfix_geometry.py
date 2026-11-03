"""Pure-numpy geometry for the SMIL mouse limb fix (no bpy), used by build_mouse_limbfix.py.

The source model's elbow (`ulna-radius`) pivot sits mid-chest: its forearm is about twice the labelled elbow-to-wrist
length in two independent 3D label sets (body-normalised). The repose model also moved the knee (`tibia`) pivot
without re-skinning, so part of the tibia's skin lies on the thigh side of the knee.

1. Elbow pivot (`elbow_on_forearm_line`), the source of the pivot plan's elbow rule: the new pivot stays on the source
   forearm line (wrist -> old elbow), at the point where forearm / upper-arm equals the labelled ratio
   (FOREARM_UPPERARM_RATIO). The plan stores the resulting pivots, so the build does not call it.
2. Skinning around a pivot (`repartition_pair`). Weights of a parent/child bone pair are re-split around the child
   pivot's bisector plane with a smoothstep band, gated by the distance to the child bone, so that a bone's skin lies
   on the distal side of its own pivot. The total weight of the pair per vertex is kept and every other bone's weight
   is untouched. A vertex that would exceed `max_weights` non-zero weights (2 = the add-on's "limit total 2"
   convention) gets the pair's weight on one side only.
3. Pivot plans (`apply_pivot_plan`): move the pivots a plan lists, then run its re-skin steps in order. The v5 model
   was built from plans/pivot_plan_v5.json, which moves the elbow, shoulder and knee pivots (each re-skinned) and
   puts the eyes on the eyeball centres.
"""

import numpy as np

# Mean of the median labelled forearm/upper-arm ratios of two 3D label sets.
FOREARM_UPPERARM_RATIO = 0.7175
# Half-width of the parent/child weight transition (model units); the source model's elbow and wrist bands span
# about +-0.04 u around the bisector plane (p5..p95 of mixed vertices).
TRANSITION_HALF_WIDTH = 0.04


def unit(v):
    return v / np.linalg.norm(v)


def elbow_on_forearm_line(shoulder, elbow, wrist, ratio=FOREARM_UPPERARM_RATIO, iters=60):
    """Point E on the segment wrist -> elbow with |wrist - E| / |E - shoulder| == ratio (bisection; the ratio grows
    monotonically from 0 at the wrist). Returns (E, t) with E = wrist + t * (elbow - wrist)."""

    def f(t):
        e = wrist + t * (elbow - wrist)
        return np.linalg.norm(wrist - e) / np.linalg.norm(e - shoulder) - ratio

    lo, hi = 0.0, 1.0
    if f(hi) < 0:
        raise ValueError(f"ratio {ratio} not reachable on the forearm line (max {f(hi) + ratio:.3f})")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    t = 0.5 * (lo + hi)
    return wrist + t * (elbow - wrist), t


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def segment_distance(V, a, b):
    ab = b - a
    t = np.clip(((V - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(V - (a + t[:, None] * ab), axis=1)


def repartition_pair(
    V, W, parent_pos, joint_pos, child_pos, p, j, half_width=TRANSITION_HALF_WIDTH, max_weights=2, radius=None
):
    """Re-split weights of bones p (parent) and j (child joint) around j's bisector plane.

    A vertex goes to the child only if it is distal of the plane AND close to the child's bone segment: the radius is
    the 95th percentile distance of the source's clearly-distal child-dominant vertices (the limb's own thickness),
    faded out over 2 * half_width. Without the proximity gate, skin far from the limb that the source happened to
    skin to the child (e.g. groin skin on Tail_00 + tibia) is split by the plane alone, leaving a seam that folds.

    V: (N, 3) rest vertices; W: (N, J) weights; *_pos: rest positions of the parent pivot, the joint pivot and the
    joint's child pivot. radius: explicit proximity radius (model units) instead of the estimate, for bones whose
    source skin is itself too spread to estimate a thickness from. Returns (W_new, info) with W_new a copy of W.
    """
    W = np.asarray(W)
    n = unit(unit(joint_pos - parent_pos) + unit(child_pos - joint_pos))
    s = (V - joint_pos) @ n  # > 0: distal (child side)
    dist = segment_distance(V, joint_pos, child_pos)
    if radius is None:
        limb = (W.argmax(1) == j) & (s > half_width)
        radius = float(np.percentile(dist[limb], 95))
    share = smoothstep((s + half_width) / (2.0 * half_width))  # child's share of the pair weight
    share = share * smoothstep((radius + 2.0 * half_width - dist) / (2.0 * half_width))
    pair = W[:, p] + W[:, j]
    region = pair > 0
    others = np.count_nonzero(np.delete(W, [p, j], axis=1) > 0, axis=1)
    hard = region & (others > max_weights - 2)  # would exceed max_weights non-zero weights: split hard
    share = np.where(hard, (share >= 0.5).astype(float), share)
    Wn = W.astype(np.float64).copy()
    Wn[region, j] = pair[region] * share[region]
    Wn[region, p] = pair[region] * (1.0 - share[region])
    Wn = Wn.astype(W.dtype)
    changed = np.abs(Wn[:, [p, j]] - W[:, [p, j]]).max(1) > 1e-6
    info = {
        "region_verts": int(region.sum()),
        "changed_verts": int(changed.sum()),
        "hard_assigned_verts": int((hard & changed).sum()),
        "plane_normal": n.tolist(),
        "child_radius_units": radius,
    }
    return Wn, info


def apply_pivot_plan(J, V, W, names, plan):
    """Move pivots and re-split skin as recorded in a pivot plan (e.g. plans/pivot_plan_v5.json).

    plan = {"pivots": {joint: [x, y, z]}, "max_weights": int,
            "reskin": [{"parent": p, "joint": j, "child": c, "half_width": w, "radius": r or null}, ...]}
    Re-skin steps run in the listed order on the moved pivots. Returns (J_new, W_new, info per step).
    """
    I = {nm: i for i, nm in enumerate(names)}
    Jn, Wn = np.array(J, copy=True), np.array(W, copy=True)
    for name, pos in plan["pivots"].items():
        Jn[I[name]] = pos
    info = {}
    for r in plan["reskin"]:
        p, j, c = I[r["parent"]], I[r["joint"]], I[r["child"]]
        Wn, info[f"{r['parent']}|{r['joint']}"] = repartition_pair(
            V,
            Wn,
            Jn[p],
            Jn[j],
            Jn[c],
            p,
            j,
            half_width=r["half_width"],
            max_weights=plan["max_weights"],
            radius=r.get("radius"),
        )
    return Jn, Wn, info
