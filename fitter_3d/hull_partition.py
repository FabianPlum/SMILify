"""Use the convexity hierarchy as a partition PRIOR — frozen structure, revisable naming.

THE DESIGN QUESTION THIS ANSWERS
--------------------------------
Two partitions have been tested and the comparison between them is the most useful thing
this project knows about partitions (REPORT §5.3):

  * The FIT-DERIVED partition (`TargetPartition`) assigns each target point the anatomical
    group of its nearest vertex on the current fitted mesh. It is circular -- it can only
    confirm what the fit believes -- but it is SELF-CORRECTING: a point given to the wrong
    leg is taken back once the fit moves.

  * The FROZEN target-derived partition (`PartFieldPartition`) is not circular, and it LOST
    decisively (part_leg_distal within_tau -18.5%, chamfer -47.6%, edge_logratio -22.9%),
    because its errors are permanent and its uncertainty becomes deletion or a wrong
    commitment rather than reduced weight.

The report's own reading of that pair: *"Circularity in the fit-derived partition is a real
flaw and also an error-correcting mechanism. Target-derived is defensible; FROZEN is the
wrong way to use it."*

So this class freezes the half that is a fact about the scan and leaves the half that is a
belief about anatomy free to move:

  FROZEN     which target points belong TOGETHER. That comes from the convexity hierarchy
             (`hull_decomposition.py`), is computed from the target alone, and is a genuine
             geometric fact -- points inside one near-convex chunk of surface are one piece
             of animal whatever the fit currently thinks.

  REVISABLE  WHICH ANATOMICAL GROUP each chunk is. That is recomputed at every reassignment
             by majority vote of the chunk's own points over their nearest-fitted-vertex
             groups -- i.e. by exactly the rule `TargetPartition` already uses, just applied
             per chunk instead of per point.

Nothing anatomical is frozen, so the failure mode that sank the part field cannot occur: a
chunk assigned to the wrong leg is reassigned wholesale as soon as the fit moves.

WHAT THIS BUYS OVER THE PER-POINT RULE
--------------------------------------
Voting per chunk makes the assignment PIECEWISE, and that is the anti-multimodality
mechanism the hierarchical fitter was built around, applied one level deeper. Under the
per-point rule a single leg's surface can be split three ways between neighbouring leg
groups, and each fragment then pulls its own group toward a different leg -- the exact
substitutable-leg ambiguity the staging exists to remove. A chunk must go somewhere as a
unit, so it either pulls a group onto a whole leg or does not pull it at all.

THE GRANULARITY KNOB IS THE HIERARCHY'S POINT
---------------------------------------------
`set_k` re-cuts the same tree, so a stage can use coarse chunks while placing the body and
fine chunks while fitting distal segments, without recomputing anything. That is what makes
this a hierarchy rather than a segmentation, and it is what the hierarchical schedule
(H0 body -> H1 legs -> H2 joint -> H3 deform) actually wants.

PRE-REGISTERED OUTCOME
----------------------
Judged on the DIRECT correspondence measure from the ground-truth round trip
(`SYNTHETIC_ROUNDTRIP.md`), not on surface metrics, because §3 establishes that surface
metrics cannot see correspondence and E6 establishes that the pipeline's correspondence is
~4.81% correct / 3.8 vertex-spacings of error on its own noise-free geometry.

KILL CONDITION, fixed before the run: if `correct_frac` on SYN_clean does not exceed the
4.81% baseline by more than the between-seed spread, the hierarchy does not inform matching
and this is dropped rather than re-weighted -- the same discipline that dropped E1 and E3.
"""

import numpy as np
import torch
from pytorch3d.ops import knn_points


class HullPartition:
    """`TargetPartition`-compatible partition backed by a convexity hierarchy.

    Interface is the three methods `HierarchicalStage` calls -- `set_probes`, `update`,
    `mask_for` -- so nothing in the fitter changes.

    Parameters
    ----------
    ref_pts : (B, R, 3) tensor
        The decomposition's own surface samples, in the fitter's target frame.
    hierarchies : list[Hierarchy]
        One per batch element, from `hull_decomposition.decompose_mesh`.
    k : int
        Cut size: how many chunks the target is broken into. Re-cuttable via `set_k`.
    min_majority : float
        If a chunk's winning group holds less than this fraction of its votes, the chunk is
        AMBIGUOUS and its points keep their own per-point labels instead. 0.0 disables the
        fallback (always vote). This is the soft edge of the mechanism: a chunk straddling
        two legs should not be forced onto one of them, and §5.3 showed that forcing is
        exactly how a target-derived partition loses.
    """

    def __init__(self, ref_pts, hierarchies, k, n_groups, device, min_majority=0.0):
        self.ref_pts = ref_pts.to(device)
        self.hierarchies = hierarchies
        self.n_groups = int(n_groups)
        self.device = device
        self.min_majority = float(min_majority)
        self.assign = None
        self.probe_pts = None
        self.probe_assign = None
        self.k = None
        self.n_ambiguous = 0
        self.set_k(k)

    # ------------------------------------------------------------------ granularity
    def set_k(self, k):
        """Re-cut the same tree at a different granularity. Cheap: no geometry is recomputed."""
        self.k = int(k)
        lab = [np.asarray(h.cut(self.k), dtype=np.int64) for h in self.hierarchies]
        self.ref_cluster = torch.as_tensor(np.stack(lab), device=self.device)  # (B, R)
        self.n_clusters = int(self.ref_cluster.max().item()) + 1

    def set_probes(self, probe_pts):
        self.probe_pts = probe_pts
        self.probe_assign = None

    # ------------------------------------------------------------------ assignment
    def _per_point_groups(self, fitted_verts, pts, vg):
        idx = knn_points(pts, fitted_verts, K=1).idx[..., 0]  # (B, P)
        return vg[idx]

    def _chunk_of(self, pts):
        idx = knn_points(pts, self.ref_pts, K=1).idx[..., 0]  # (B, P)
        return torch.gather(self.ref_cluster, 1, idx)

    def _vote(self, chunk, group):
        """Majority anatomical group per chunk, and the winning fraction.

        One scatter_add into a (B, n_clusters, n_groups) tally, so the whole vote is two
        kernel launches rather than a Python loop over chunks.
        """
        B, P = chunk.shape
        tally = torch.zeros(B, self.n_clusters, self.n_groups, device=chunk.device)
        flat = chunk * self.n_groups + group
        tally.view(B, -1).scatter_add_(1, flat, torch.ones(B, P, device=chunk.device))
        tot = tally.sum(-1)
        win, arg = tally.max(-1)
        frac = win / tot.clamp_min(1.0)
        return arg, frac  # (B, n_clusters) each

    def update(self, fitted_verts, tgt_pts):
        """Re-derive the naming from the CURRENT fit, keeping the chunking fixed."""
        with torch.no_grad():
            vg = self._vg
            grp = self._per_point_groups(fitted_verts, tgt_pts, vg)
            chunk = self._chunk_of(tgt_pts)
            arg, frac = self._vote(chunk, grp)
            voted = torch.gather(arg, 1, chunk)
            if self.min_majority > 0:
                ok = torch.gather(frac, 1, chunk) >= self.min_majority
                self.n_ambiguous = int((~ok).sum())
                self.assign = torch.where(ok, voted, grp)
            else:
                self.assign = voted

            churn = None
            if self.probe_pts is not None:
                pg = self._per_point_groups(fitted_verts, self.probe_pts, vg)
                pc = self._chunk_of(self.probe_pts)
                pnew = torch.gather(arg, 1, pc)
                if self.min_majority > 0:
                    pok = torch.gather(frac, 1, pc) >= self.min_majority
                    pnew = torch.where(pok, pnew, pg)
                if self.probe_assign is not None:
                    churn = float((pnew != self.probe_assign).float().mean().item())
                self.probe_assign = pnew
            return churn

    def mask_for(self, group_id):
        return self.assign == group_id

    # `HierarchicalStage` owns the template's vertex->group map; it is injected here rather
    # than duplicated, so the two can never drift apart.
    def bind_vertex_groups(self, vg):
        self._vg = torch.as_tensor(vg, device=self.device)
        return self


def build_hull_partition(
    targets,
    vertex_group,
    n_groups,
    device,
    k=13,
    n_points=20000,
    coacd_threshold=0.03,
    criterion="volume",
    seed=0,
    min_majority=0.0,
    verbose=True,
):
    """Decompose every target mesh and wrap the results as a `HullPartition`.

    Targets are pytorch3d `Meshes` already in the fitter's normalised frame, so the
    decomposition must NOT renormalise -- doing so would put the chunk cloud in a different
    frame from the target points it is looked up against.
    """
    import trimesh

    from fitter_3d.hull_decomposition import decompose_mesh

    hierarchies, ref = [], []
    for b in range(len(targets)):
        v = targets[b].verts_packed().detach().cpu().numpy().astype(np.float64)
        f = targets[b].faces_packed().detach().cpu().numpy()
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        h = decompose_mesh(
            tm,
            n_points=n_points,
            coacd_threshold=coacd_threshold,
            criterion=criterion,
            seed=seed,
            normalise_frame=False,  # already in the fitter's frame
        )
        hierarchies.append(h)
        ref.append(torch.tensor(h.points, dtype=torch.float32, device=device))
        if verbose:
            sizes = np.bincount(h.cut(k), minlength=k)
            print(
                f"[hull] target {b:2d}: {h.meta['n_hulls']:4d} hulls -> {h.n_atoms:4d} atoms, "
                f"k={k} chunk sizes {np.sort(sizes)[::-1][:8]}...",
                flush=True,
            )
    part = HullPartition(torch.stack(ref), hierarchies, k, n_groups, device, min_majority=min_majority)
    return part.bind_vertex_groups(vertex_group)
