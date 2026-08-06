"""VERIFY PROBE (adversarial): does PartAnchorInit move log_beta_scales when the correct
answer is 'do not move at all'?  CPU only, no GPU allocation.

Setup: the target IS the model's own rest mesh, and the part labels are PERFECT (nearest
rest vertex's anatomical group).  The unique correct solution is therefore
    joint_rot = 0, trans = 0, log_beta_scales = 0   (every joint scale exactly 1.000x)

Arms
  A  target area-sampled (exactly what optimise_hierarchical does)         [pipeline]
  B  target area-sampled, log_beta_scales EXCLUDED from the optimiser      [ablation]
  C  target = the template VERTICES themselves (identical point density)   [bias control]
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/fabi/dev/SMILify")

import pickle  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402

import config  # noqa: E402
from fitter_3d.part_anchor_init import PartAnchorInit  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_hierarchical import vertex_groups  # noqa: E402
from fitter_3d.trainer_moonshot import build_lr_joint_pairs, symmetry_penalty  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
dev = "cpu"

with open(config.SMAL_FILE, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
jnames = list(dd["J_names"])
vg, gnames = vertex_groups(np.asarray(dd["weights"]), jnames, split_distal=True)
vg_t = torch.tensor(vg)
MID = [i for i, n in enumerate(jnames) if n in ("b_t", "b_h") or n.startswith("b_a_")]
PAIRS = build_lr_joint_pairs(jnames)


class Stub:
    def __init__(self, assign):
        self.assign = assign


def run_arm(tag, area_sampled=True, free_scales=True, n_it=400):
    torch.manual_seed(0)
    np.random.seed(0)
    smal = SMAL3DFitter(batch_size=1, device=dev, shape_family=-1)
    with torch.no_grad():
        rest = smal(deform_verts=torch.zeros_like(smal.deform_verts))[0].clone()
    faces = smal.faces[0]
    if area_sampled:
        pts = sample_points_from_meshes(Meshes(verts=[rest], faces=[faces]), 8000).detach()
    else:
        pts = rest[None].clone().detach()
    idx = knn_points(pts, rest[None], K=1).idx[0, :, 0]
    assign = vg_t[idx][None]

    init = PartAnchorInit(smal, Stub(assign), vg, gnames, dev, n_it=n_it, log_every=10**9)
    if not free_scales:
        # drop the log_beta_scales param group entirely
        init.optimizer = torch.optim.Adam(
            [
                {"params": [smal.global_rot], "lr": 0.05},
                {"params": [smal.trans], "lr": 0.05},
                {"params": [smal.joint_rot], "lr": 0.05},
            ]
        )
    # step-size trace over the first few Adam steps
    []
    real_run = init.run
    hist = real_run(pts, scale=None) if n_it else []

    with torch.no_grad():
        lbs = smal.log_beta_scales[0]
        acc = smal.get_joint_scales()[0]
        raw = lbs.exp()
    print(f"\n=== ARM {tag} (area_sampled={area_sampled}, scales_free={free_scales}) ===")
    print(f"  anchor_loss  {hist[0]:.6e} -> {hist[-1]:.6e}")
    print(
        f"  RAW per-joint scale exp(lbs): {float(raw.min()):.3f}..{float(raw.max()):.3f}  |lbs| mean {float(lbs.abs().mean()):.4f}"
    )
    print(f"  ACCUM (get_joint_scales):     {float(acc.min()):.3f}..{float(acc.max()):.3f}")
    print(f"  symmetry_penalty(lbs)         {float(symmetry_penalty(smal.log_beta_scales, None, PAIRS)):.5f}")
    print(f"  joint_rot max deg             {float(smal.joint_rot[0].norm(dim=-1).max()) * 180 / np.pi:.1f}")
    print("  MIDLINE joints (invisible to symmetry_penalty):")
    for j in MID:
        print(
            f"    {jnames[j]:>7s}  raw {raw[j, 0]:.3f},{raw[j, 1]:.3f},{raw[j, 2]:.3f}   "
            f"accum {acc[j, 0]:.3f},{acc[j, 1]:.3f},{acc[j, 2]:.3f}"
        )
    return hist, float(lbs.abs().max())


# --- step-size check: how fast does Adam move lbs in the first steps?
torch.manual_seed(0)
np.random.seed(0)
smal = SMAL3DFitter(batch_size=1, device=dev, shape_family=-1)
with torch.no_grad():
    rest = smal(deform_verts=torch.zeros_like(smal.deform_verts))[0].clone()
pts = sample_points_from_meshes(Meshes(verts=[rest], faces=[smal.faces[0]]), 8000).detach()
idx = knn_points(pts, rest[None], K=1).idx[0, :, 0]
init = PartAnchorInit(smal, Stub(vg_t[idx][None]), vg, gnames, dev, n_it=0, log_every=10**9)
prev = smal.log_beta_scales.detach().clone()
print("step |  max|dlbs| this step   max|lbs| cumulative   (lr*0.5 = 0.025)")
for k in range(8):
    init.n_it = 1
    init.history = []
    init.run(pts)
    cur = smal.log_beta_scales.detach()
    print(f"  {k + 1}  |      {float((cur - prev).abs().max()):.5f}                {float(cur.abs().max()):.5f}")
    prev = cur.clone()

hA, _ = run_arm("A_pipeline", True, True)
hB, _ = run_arm("B_scales_frozen", True, False)
hC, _ = run_arm("C_same_density", False, True)
print(
    f"\nfinal anchor_loss  A(free scales) {hA[-1]:.3e}   B(frozen scales) {hB[-1]:.3e}   ratio {hB[-1] / hA[-1]:.2f}x"
)
