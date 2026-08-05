"""CPU-only verification of the --pf_body H0 data-term claim."""

import os
import sys
import pickle

import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/home/fabi/dev/SMILify")
sys.path.insert(0, "/home/fabi/dev/SMILify/diagnostics/moonshot")

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_hierarchical import vertex_groups  # noqa: E402
from fitter_3d.partfield import PartFieldNet, PartFieldPartition, predict_field, part_names  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
dev = "cpu"
NM = int(sys.argv[1]) if len(sys.argv) > 1 else 3

print("SMAL_FILE:", config.SMAL_FILE)
with open(config.SMAL_FILE, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()
jnames = list(dd["J_names"])
vg, gnames = vertex_groups(np.asarray(dd["weights"]), jnames, split_distal=True)
body_id = gnames.index("body")
body_mask = torch.tensor(vg == body_id)
print("groups:", gnames)
print("body vertex count:", int(body_mask.sum()), "of", len(vg))

# ---------------------------------------------------------------- (c) area weighting
v_t = torch.tensor(np.asarray(dd["v_template"]), dtype=torch.float32)
faces = torch.tensor(np.asarray(dd["f"]).astype(np.int64))
fb = body_mask[faces].all(dim=1)  # faces entirely inside the body group
fbody = faces[fb]
p0, p1, p2 = v_t[fbody[:, 0]], v_t[fbody[:, 1]], v_t[fbody[:, 2]]
area = 0.5 * torch.cross(p1 - p0, p2 - p0, dim=1).norm(dim=1)
w = torch.zeros(v_t.shape[0])
for k in range(3):
    w.index_add_(0, fbody[:, k], area / 3.0)
wb = w[body_mask]
wb_rel = (wb / wb.mean()).numpy()
ess = float(wb.sum() ** 2 / (wb**2).sum())
print(
    f"[c] body per-vertex area share rel. to uniform: "
    f"p1={np.percentile(wb_rel, 1):.3f} p50={np.percentile(wb_rel, 50):.3f} "
    f"p99={np.percentile(wb_rel, 99):.3f} max={wb_rel.max():.3f}  "
    f"Kish ESS={ess:.0f}/{int(body_mask.sum())} = {100 * ess / int(body_mask.sum()):.1f}%"
)

# ---------------------------------------------------------------- meshes + part field
import glob  # noqa: E402

files = sorted(glob.glob("/home/fabi/dev/SMILify/diagnostics/moonshot/bench50/*.obj"))[:NM]
_, targets = load_meshes(mesh_files=files, device=dev)
print("meshes:", [os.path.basename(f) for f in files])

pf_names = part_names(jnames)
assert pf_names[:-1] == list(gnames)
ck = torch.load("/home/fabi/dev/SMILify/diagnostics/moonshot/partfield/net_B_mirror.pt", map_location=dev)
net = PartFieldNet(n_classes=len(ck["names"]), width=ck.get("width", 1.0)).to(dev)
net.load_state_dict(ck["state"])
net.eval()
debris_id = pf_names.index("debris")

REF = 30000
ref_p, ref_l = [], []
for b in range(len(files)):
    p = sample_points_from_meshes(targets[b], REF)[0]
    prob = predict_field(net, p, seed=0)
    ref_p.append(p)
    ref_l.append(prob.argmax(-1))
    print(f"  {os.path.basename(files[b])}: debris {100 * float((ref_l[-1] == debris_id).float().mean()):.1f}%")
part = PartFieldPartition(torch.stack(ref_p), torch.stack(ref_l), len(gnames), dev, debris_id=debris_id)

# ---------------------------------------------------------------- H0 initial state
smal = SMAL3DFitter(batch_size=len(files), device=dev, shape_family=-1)
with torch.no_grad():
    fitted = smal()  # H0 iteration-0 model vertices
print("fitted verts:", tuple(fitted.shape), " max|coord| template:", float(v_t.abs().max()))

N = 8000
tgt_pts = sample_points_from_meshes(targets, N)
part.update(None, tgt_pts)

C = 0.25
c2 = C * C


def gm(d):
    return d * c2 / (d + c2)


def stats(name, d_st, d_ts):
    d = torch.cat([d_st.flatten(), d_ts.flatten()])
    r = d.sqrt()
    l2 = float(d_st.mean() + d_ts.mean())
    gml = float(gm(d_st).mean() + gm(d_ts).mean())
    infl = (c2 / (d + c2)) ** 2  # d(gm)/d(r^2) : per-residual gradient scaling
    print(
        f"  {name}: L2={l2:.5f}  GM={gml:.5f}  GM/L2={gml / max(l2, 1e-12):.3f}\n"
        f"      residual r: p50={float(r.median()):.4f} p95={float(r.quantile(0.95)):.4f} "
        f"p99={float(r.quantile(0.99)):.4f} max={float(r.max()):.4f}   frac r>c(0.25)={float((r > C).float().mean()) * 100:.2f}%\n"
        f"      GM gradient scaling: p50={float(infl.median()):.3f} p05={float(infl.quantile(0.05)):.3f} "
        f"min={float(infl.min()):.4f}\n"
        f"      share of TOTAL L2 loss from the worst 1% residuals = {float(d.topk(max(1, d.numel() // 100)).values.sum() / d.sum()) * 100:.1f}%"
    )
    return l2, gml


print("\n=== A) M9a control H0 (partitioned=False): GLOBAL area-sampled chamfer ===")
mesh = Meshes(verts=list(fitted), faces=[smal.faces[0]] * len(files))
src_pts = sample_points_from_meshes(mesh, N)
d_fwd = knn_points(src_pts, tgt_pts, K=1).dists[..., 0]
d_bwd = knn_points(tgt_pts, src_pts, K=1).dists[..., 0]
stats("global (what robust_kernel='gm' actually applies to)", d_fwd, d_bwd)

print("\n=== B) M9b H0 (--pf_body, partitioned=True): body verts vs body-labelled tgt ===")
src_g = fitted[:, body_mask, :]
allst, allts = [], []
for b in range(len(files)):
    tm = part.assign[b] == body_id
    tg = tgt_pts[b][tm].unsqueeze(0)
    sg = src_g[b].unsqueeze(0)
    print(f"  b{b}: {int(tm.sum())} body-labelled target pts of {N}; src {sg.shape[1]} body verts")
    allst.append(knn_points(sg, tg, K=1).dists[..., 0])
    allts.append(knn_points(tg, sg, K=1).dists[..., 0])
d_st = torch.cat([x.flatten() for x in allst]).unsqueeze(0)
d_ts = torch.cat([x.flatten() for x in allts]).unsqueeze(0)
l2_b, gm_b = stats("body-restricted, VERTEX source (code path actually taken)", d_st, d_ts)

print("\n=== C) same restriction but AREA-SAMPLED body source (isolates change (c)) ===")
# area-sample only the body sub-mesh
sub = Meshes(verts=list(fitted), faces=[fbody] * len(files))
src_area = sample_points_from_meshes(sub, N)
allst, allts = [], []
for b in range(len(files)):
    tm = part.assign[b] == body_id
    tg = tgt_pts[b][tm].unsqueeze(0)
    sg = src_area[b].unsqueeze(0)
    allst.append(knn_points(sg, tg, K=1).dists[..., 0])
    allts.append(knn_points(tg, sg, K=1).dists[..., 0])
d_st2 = torch.cat([x.flatten() for x in allst]).unsqueeze(0)
d_ts2 = torch.cat([x.flatten() for x in allts]).unsqueeze(0)
l2_c, gm_c = stats("body-restricted, AREA source", d_st2, d_ts2)
print(f"\n  vertex-source vs area-source L2 loss: {l2_b:.5f} vs {l2_c:.5f}  ratio {l2_b / l2_c:.3f}")
print(f"  vertex-source L2 vs GM on the SAME correspondence: {l2_b:.5f} vs {gm_b:.5f}  ratio {l2_b / gm_b:.3f}")
