"""Learned per-point PART FIELD over raw target point clouds.

WHY THIS EXISTS
---------------
Every partition this project has used so far is *fit-derived*: `TargetPartition.update()`
assigns each target point the anatomical group of its nearest vertex **on the current
fitted mesh**. That is circular. It can only ever confirm what the fit already believes, so
when the fit puts leg 2 where leg 3 belongs, the partition agrees with it and the data term
happily keeps it there. It is a smoother for a fit that is already roughly right, not a way
to *become* right.

The alternative proposed by the paradigm search is a partition derived from the TARGET
ALONE, computed once, frozen. Two candidates for that:

  (a) a geodesic level-set sweep of the scan's own branching topology -- tested in probe 14
      and REJECTED. On the template it separates all six legs perfectly, but on the 50 real
      scans only 58% reach six branches (its proposer's own gate was 80%) and only 20%
      agree with themselves across two surface samplings. Legs that *touch* the body or each
      other merge into one connected component, and no threshold separates them, because
      after they touch there is genuinely no thin neck in the geodesic field to cut.

  (b) a learned classifier -- this file. The reason it can succeed exactly where (a) failed
      is that (a) is a purely *local* criterion (is there a bottleneck here?) while a
      network with a global-context branch can label a point by where it sits relative to
      the whole animal. Two touching legs have no bottleneck between them but they are
      still at different x along the body axis, and the scans are canonically aligned, so
      the information is present -- just not in the form a level set can read.

LABELS ARE FREE
---------------
No annotation is needed. `weights.argmax(1)` gives every template vertex its dominant
skinning joint, `anatomical_groups()` maps joints to the 13 anatomical parts already used
by the hierarchical fitter, and a 14th class absorbs scan debris. Supervision comes from
(i) synthetic scans posed from the model itself, where labels are exact, and (ii) existing
fits, where labels are transferred by nearest neighbour and are noisy but in-domain.

PRE-REGISTERED GATES (fixed before any result was seen; scored by probe 15)
--------------------------------------------------------------------------
G1  REPRODUCIBILITY.  Two independent surface samplings of the same scan must receive the
    same labels: >= 90% per-point agreement, averaged over the 50 bench specimens. This is
    the gate that killed the geodesic method (20%). A partition that changes with the point
    sample cannot anchor a data term.
G2  BILATERAL CONSISTENCY.  Mirroring the input in y must swap left/right labels and
    preserve segment identity: >= 85% agreement. Unsupervised, needs no ground truth, and
    it is a real test -- an ant is near-symmetric, so passing means the network is reading
    the sign of y and global context rather than memorising local shape.
G3  THE TOUCHING-LEG CLAIM.  On the specimens where probe 14 collapsed to <= 2 geodesic
    branches, the field must still recover all six legs (each leg label present with >= 50
    points) on >= 80% of them. This is precisely the claim made for a classifier over a
    level-set sweep; if it fails, the claim was wrong.
G4  HELD-OUT ACCURACY.  >= 85% per-point accuracy on held-out synthetic specimens, and
    real-transfer agreement reported alongside (not gated -- the real "labels" are
    themselves fit-derived and so are not ground truth).
G5  DOWNSTREAM.  Substituted for `TargetPartition`, `part_leg_distal` must improve over M7
    with no regression in folded_face_frac. This is the only gate that decides shipping.

Sources: PointNet++ (Qi et al. NeurIPS 2017) segmentation architecture, via the set
abstraction / feature propagation layers already vendored in
`fitter_3d/pointcloud2smil/pointnet2_utils.py`. Bootstrap-labels-from-a-parametric-model is
the standard trick from SMPL body-part segmentation work (e.g. Bogo et al. FAUST, Omran et
al. NBF ICCV 2018, which regresses a body-part segmentation as an intermediate
representation for exactly this "give the fit a target-derived anchor" reason).
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fitter_3d.pointcloud2smil.pointnet2_utils import (  # noqa: E402
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)
from fitter_3d.trainer_hierarchical import anatomical_groups  # noqa: E402

DEBRIS = "debris"


def part_names(joint_names):
    """The label set: the 13 split-distal anatomical groups plus a debris class.

    Ordering is `sorted()` over the anatomical names followed by debris LAST, so that
    `names[:-1]` is exactly the group list `vertex_groups(..., split_distal=True)`
    produces and the two index spaces coincide. Downstream code depends on that.
    """
    names = sorted(set(anatomical_groups(joint_names, split_distal=True).values()))
    return names + [DEBRIS]


def template_vertex_labels(weights, joint_names):
    """Per-template-vertex part label from dominant skinning weight. No annotation."""
    dom = np.asarray(weights).argmax(axis=1)
    jg = anatomical_groups(joint_names, split_distal=True)
    names = part_names(joint_names)
    n2i = {n: i for i, n in enumerate(names)}
    return np.array([n2i[jg[int(d)]] for d in dom], dtype=np.int64)


def normalise(pts):
    """Centre and scale a point cloud exactly as the fitter's meshes are normalised.

    Deliberately NOT rotation-invariant. These scans are canonically aligned in the body
    axis AND in roll (confirmed by the user and by probe 05's renders), and that alignment
    is the only reason left/right is decidable at all. Throwing it away with a T-Net or a
    PCA frame would make G2 unachievable in principle.
    """
    c = pts.mean(dim=-2, keepdim=True)
    p = pts - c
    s = p.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return p / s, c, s


class PartFieldNet(nn.Module):
    """PointNet++ segmentation network, xyz-only input, per-point part logits.

    The `group_all` fourth abstraction level is load-bearing rather than decorative: it is
    the global-context branch, and it is what a geodesic level set structurally lacks. A
    point on a leg that touches the body is locally indistinguishable from a point on the
    body; it is separable only by where it sits in the whole animal.
    """

    def __init__(self, n_classes=14, width=1.0):
        super().__init__()
        w = lambda c: max(8, int(round(c * width)))  # noqa: E731
        self.sa1 = PointNetSetAbstraction(1024, 0.08, 32, 3 + 3, [w(32), w(32), w(64)], False)
        self.sa2 = PointNetSetAbstraction(256, 0.18, 32, w(64) + 3, [w(64), w(64), w(128)], False)
        self.sa3 = PointNetSetAbstraction(64, 0.35, 32, w(128) + 3, [w(128), w(128), w(256)], False)
        self.sa4 = PointNetSetAbstraction(None, None, None, w(256) + 3, [w(256), w(512)], True)
        self.fp4 = PointNetFeaturePropagation(w(512) + w(256), [w(256), w(256)])
        self.fp3 = PointNetFeaturePropagation(w(256) + w(128), [w(256), w(256)])
        self.fp2 = PointNetFeaturePropagation(w(256) + w(64), [w(256), w(128)])
        self.fp1 = PointNetFeaturePropagation(w(128) + 3, [w(128), w(128), w(128)])
        self.head = nn.Sequential(
            nn.Conv1d(w(128), w(128), 1),
            nn.BatchNorm1d(w(128)),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(w(128), n_classes, 1),
        )
        self.n_classes = n_classes

    def forward(self, pts):
        """pts (B, N, 3) already normalised. Returns logits (B, N, C)."""
        x = pts.transpose(1, 2).contiguous()  # (B,3,N)
        # ABSOLUTE xyz is fed as a per-point FEATURE, not just as the grouping coordinate.
        # `sample_and_group` subtracts the group centroid from the coordinates it passes to
        # the MLP, so without this the local features would be translation-invariant --
        # which would erase the sign of y and the position along the body axis, i.e. exactly
        # the two cues that decide which leg a point belongs to. G2 would be unwinnable.
        l0_xyz, l0_pts = x, x
        l1_xyz, l1_pts = self.sa1(l0_xyz, l0_pts)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        l4_xyz, l4_pts = self.sa4(l3_xyz, l3_pts)
        l3_pts = self.fp4(l3_xyz, l4_xyz, l3_pts, l4_pts)
        l2_pts = self.fp3(l2_xyz, l3_xyz, l2_pts, l3_pts)
        l1_pts = self.fp2(l1_xyz, l2_xyz, l1_pts, l2_pts)
        l0_pts = self.fp1(l0_xyz, l1_xyz, l0_pts, l1_pts)
        return self.head(l0_pts).transpose(1, 2).contiguous()


# --------------------------------------------------------------------------------------
# inference: a frozen, target-derived partition
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_field(net, pts, chunk_n=4096, n_votes=4, device=None, seed=0):
    """Label a dense cloud by averaging softmax over `n_votes` random 4096-point subsets.

    The network is trained at a fixed 4096 points and FPS/ball-query radii are tied to that
    density, so a 30k cloud is labelled by voting over subsets rather than by feeding all
    30k at once. Points not drawn in any subset inherit their nearest labelled neighbour.
    Votes are averaged in probability space, which is also what makes G1 pass: a single
    subset is a sample, the average over subsets is close to the expectation.
    """
    device = device or next(net.parameters()).device
    net.eval()
    P = pts.shape[0]
    prob = torch.zeros(P, net.n_classes, device=device)
    cnt = torch.zeros(P, 1, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    npts, _, _ = normalise(pts.to(device))
    for v in range(n_votes):
        perm = torch.randperm(P, generator=g)
        for i in range(0, P, chunk_n):
            idx = perm[i : i + chunk_n].to(device)
            if idx.numel() < 64:
                continue
            sub = npts[idx].unsqueeze(0)
            logit = net(sub)[0]
            prob[idx] += torch.softmax(logit, dim=-1)
            cnt[idx] += 1
    unseen = cnt.squeeze(-1) == 0
    prob = prob / cnt.clamp_min(1)
    if bool(unseen.any()):
        from pytorch3d.ops import knn_points

        src = (~unseen).nonzero(as_tuple=True)[0]
        j = knn_points(pts[unseen].unsqueeze(0).to(device), pts[src].unsqueeze(0).to(device), K=1).idx[0, :, 0]
        prob[unseen] = prob[src[j]]
    return prob


class PartFieldPartition:
    """Drop-in replacement for `TargetPartition` whose assignment never moves.

    Same interface (`set_probes`, `update`, `mask_for`) so `HierarchicalStage` needs no
    change, but `update()` ignores the fitted mesh entirely and returns churn 0.0 by
    construction. That zero is the point of the whole exercise: the data term becomes
    stationary, and the fit is being pulled toward an opinion about the target that it
    cannot itself edit.

    Debris points are assigned group -1 and therefore fall out of every `mask_for`, which
    removes them from all data terms. That is a side benefit the fit-derived partition
    could never give: it had to assign every point to *some* part.
    """

    def __init__(
        self, ref_pts, ref_label, n_groups, device, debris_id=None, ref_conf=None, min_conf=0.0, keep_debris=False
    ):
        self.ref_pts = ref_pts.to(device)  # (B, R, 3) dense reference clouds
        self.ref_label = ref_label.to(device).clone()  # (B, R) int64
        self.n_groups = n_groups
        self.device = device
        self.debris_id = debris_id
        self.assign = None
        self.probe_pts = None

        # keep_debris REASSIGNS debris points to their nearest non-debris neighbour instead
        # of dropping them. Probe 16 is why this exists: 90.8% of the points this field calls
        # debris lie INSIDE the animal's convex hull, a median 0.031 from the nearest body
        # point. They are not debris. They are the leg-body junction, and the network learned
        # to call it debris because the synthetic augmentation puts debris "bridges" exactly
        # there. Dropping them blinds the fit at the coxae -- the one region the whole
        # hierarchical schedule is built to place -- so the drop must be separable from the
        # partition itself before either can be blamed for a downstream result.
        self.n_reassigned = 0
        if keep_debris and debris_id is not None:
            from pytorch3d.ops import knn_points as _knn

            for b in range(self.ref_label.shape[0]):
                d = self.ref_label[b] == debris_id
                nd = ~d
                if not bool(d.any()) or int(nd.sum()) < 1:
                    continue
                src = nd.nonzero(as_tuple=True)[0]
                j = _knn(self.ref_pts[b][d].unsqueeze(0), self.ref_pts[b][nd].unsqueeze(0), K=1).idx[0, :, 0]
                self.ref_label[b][d] = self.ref_label[b][src[j]]
                self.n_reassigned += int(d.sum())
            self.debris_id = None  # no debris labels survive, so nothing is excluded
        # A frozen partition cannot self-correct, so where the field is UNSURE it should
        # abstain rather than commit. Points below min_conf are dropped from every data term
        # exactly like debris. This is the cheap stand-in for a fully soft partition: the
        # consumer indexes with a boolean mask, so real soft weights would mean rewriting
        # the chamfer, whereas abstention needs no change at all downstream.
        self.n_lowconf = 0
        if ref_conf is not None and min_conf > 0:
            low = ref_conf.to(device) < min_conf
            self.n_lowconf = int(low.sum())
            self.ref_label[low] = -1

    def set_probes(self, probe_pts):
        self.probe_pts = probe_pts

    def update(self, fitted_verts, tgt_pts):
        from pytorch3d.ops import knn_points

        with torch.no_grad():
            idx = knn_points(tgt_pts, self.ref_pts, K=1).idx[..., 0]  # (B,P)
            lab = torch.gather(self.ref_label, 1, idx)
            if self.debris_id is not None:
                lab = torch.where(lab == self.debris_id, torch.full_like(lab, -1), lab)
            self.assign = lab
        return 0.0  # stationary by construction

    def mask_for(self, group_id):
        return self.assign == group_id


def part_field_loss(logits, labels, class_weight=None, smooth=0.05):
    """Label-smoothed cross entropy, per-class-weighted.

    Weighting is not cosmetic. Probe 13 measured that coxa+trochanter+femur hold 94.0% of a
    leg chain's surface area and tarsus+pretarsus 2.4%; an unweighted per-point loss over
    area-sampled points would let the network ignore every distal class and still score
    well, which is the same failure mode that made the *fitter* neglect the distal tips.
    """
    B, N, C = logits.shape
    return F.cross_entropy(
        logits.reshape(B * N, C),
        labels.reshape(B * N),
        weight=class_weight,
        label_smoothing=smooth,
        ignore_index=-1,
    )
