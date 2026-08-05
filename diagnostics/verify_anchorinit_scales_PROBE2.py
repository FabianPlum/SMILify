"""VERIFY PROBE 2: geometric magnitude of the scale drift left by PartAnchorInit.

Same ideal case as PROBE 1 (target == the model's own rest mesh, perfect labels, so the
correct answer is 'change nothing'), but this time tracking what the drift DOES to the
mesh: per-part extent ratio vs the true (target) part, chamfer to the target, and the
symmetry_penalty / rest-edge value that H0 will see at its iteration 0.

Note on scale semantics: SMAL.forward uses propagate_scaling=False, so batch_lbs cancels
the parent scale (s_par_inv @ R @ s) -- per-joint scales do NOT compound down the chain.
The effective scale of a joint is exp(log_beta_scales) itself, NOT
SMAL3DFitter.get_joint_scales() (which multiplies down the chain and is not on the forward
path).
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/fabi/dev/SMILify")

import pickle  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pytorch3d.loss import chamfer_distance  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402

import config  # noqa: E402
from fitter_3d.part_anchor_init import PartAnchorInit  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_hierarchical import rest_edge_loss, vertex_groups  # noqa: E402
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
# gaster vertices = vertices whose dominant joint is b_a_*
dom = np.asarray(dd["weights"]).argmax(axis=1)
GASTER_V = torch.tensor(np.isin(dom, [i for i, n in enumerate(jnames) if n.startswith("b_a_")]))
HEAD_V = torch.tensor(dom == jnames.index("b_h"))


class Stub:
    def __init__(self, assign):
        self.assign = assign


smal = SMAL3DFitter(batch_size=1, device=dev, shape_family=-1)
with torch.no_grad():
    rest = smal(deform_verts=torch.zeros_like(smal.deform_verts))[0].clone()
faces = smal.faces[0]
pts = sample_points_from_meshes(Meshes(verts=[rest], faces=[faces]), 8000).detach()
idx = knn_points(pts, rest[None], K=1).idx[0, :, 0]
init = PartAnchorInit(smal, Stub(vg_t[idx][None]), vg, gnames, dev, n_it=0, log_every=10**9)
extent = float((rest.amax(0) - rest.amin(0)).amax())


def report(it):
    with torch.no_grad():
        v = smal(deform_verts=torch.zeros_like(smal.deform_verts))[0]
        ch = float(chamfer_distance(v[None], pts)[0])
        lbs = smal.log_beta_scales[0]
        gx = float(
            (v[GASTER_V].amax(0) - v[GASTER_V].amin(0)).max() / (rest[GASTER_V].amax(0) - rest[GASTER_V].amin(0)).max()
        )
        hx = float((v[HEAD_V].amax(0) - v[HEAD_V].amin(0)).max() / (rest[HEAD_V].amax(0) - rest[HEAD_V].amin(0)).max())
        sym = float(symmetry_penalty(smal.log_beta_scales, None, PAIRS))
        ed = float(rest_edge_loss(v[None], rest[None], faces))
        disp = float((v - rest).norm(dim=-1).mean()) / extent
    print(
        f"  it={it:4d}  chamfer={ch:.3e}  meanDisp={100 * disp:5.2f}%ext  "
        f"gasterExtent={gx:.3f}x  headExtent={hx:.3f}x  maxRawScale={float(lbs.exp().max()):.3f} "
        f"minRawScale={float(lbs.exp().min()):.3f}  sym={sym:.5f}  restEdge={ed:.4f}"
    )


print("ideal case: target IS the rest mesh, perfect labels -> every number below should stay at its it=0 value")
report(0)
for chunk in [10, 40, 50, 100, 200]:
    init.n_it = chunk
    init.history = []
    init.run(pts)
    report(sum([10, 40, 50, 100, 200][: [10, 40, 50, 100, 200].index(chunk) + 1]))
