"""PROBE: does a VOLUME objective give the distal legs any signal at all?

Volume IoU weights each anatomical part by its VOLUME. Surface chamfer weights it by AREA.
Thin structures (legs, antennae) have high area-to-volume. If the distal legs -- where all
the pose error lives -- carry a negligible volume fraction, a volume-IoU objective is
structurally BLIND exactly where the current objective already fails.
"""
import numpy as np, pickle

PKL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
with open(PKL, "rb") as f:
    u = pickle._Unpickler(f); u.encoding = "latin1"; dd = u.load()
de = lambda x: np.asarray(x.r if hasattr(x, "r") else x, dtype=np.float64)

V = de(dd["v_template"]); F = np.asarray(dd["f"], dtype=np.int64)
W = de(dd["weights"])                    # (Nv, J)
jnames = [str(x) for x in dd["J_names"]]
print("verts", V.shape, "faces", F.shape, "weights", W.shape)

owner = W.argmax(1)                      # dominant joint per vertex

# face area / signed volume
p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
svol = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
print(f"total area {area.sum():.4f}   total volume {svol.sum():.5f} (closed={abs(svol.sum())>0})")

# assign each face to the majority owner of its 3 verts
fo = owner[F]
face_owner = np.array([np.bincount(r).argmax() for r in fo])

def group(name):
    if name.startswith("b_a") or name in ("b_h", "root") or name.startswith("w_"):
        return "body"
    if name.startswith("an_"):
        return "antenna"
    if name.startswith("ma_"):
        return "mandible"
    parts = name.split("_")
    if len(parts) >= 3 and parts[0] == "l":
        s = parts[2]
        return "leg proximal (co/tr)" if s in ("co", "tr") else ("leg femur" if s == "fe" else "leg distal (ti/ta/pt)")
    return "other:" + name

gnames = ["root"] + jnames if len(jnames) == W.shape[1] - 1 else jnames
if len(gnames) != W.shape[1]:
    gnames = jnames
groups = np.array([group(gnames[i]) for i in range(len(gnames))])

rows = {}
for g in sorted(set(groups.tolist())):
    js = np.where(groups == g)[0]
    m = np.isin(face_owner, js)
    rows[g] = (area[m].sum(), svol[m].sum(), m.sum())

A, Vo = area.sum(), svol.sum()
print()
print(f"{'part group':<24s} {'faces':>7s} {'AREA %':>8s} {'VOL %':>8s}  {'vol/area ratio vs whole':>10s}")
for g, (a, v, n) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
    print(f"{g:<24s} {n:7d} {100*a/A:7.2f}% {100*v/Vo:7.2f}%   {(v/a)/(Vo/A):8.3f}x")

print()
legd = rows.get("leg distal (ti/ta/pt)", (0, 0, 0))
legp = rows.get("leg proximal (co/tr)", (0, 0, 0))
fem = rows.get("leg femur", (0, 0, 0))
allleg_a = legd[0] + legp[0] + fem[0]; allleg_v = legd[1] + legp[1] + fem[1]
print(f"ALL LEGS: area {100*allleg_a/A:.2f}%  volume {100*allleg_v/Vo:.2f}%")
print(f"DISTAL LEGS ONLY (ti/ta/pt, where cross-seed spread is 39-41 deg):")
print(f"   area {100*legd[0]/A:.2f}%   volume {100*legd[1]/Vo:.2f}%")
print(f"   -> a volume objective down-weights them by {(legd[0]/A)/(legd[1]/Vo):.2f}x relative to a surface objective")

# what does an ellipsoid proxy lose? cross-section circularity of distal segments
print()
print("ELLIPSOID GAUGE CHECK: cross-section aspect ratio of each segment "
      "(ellipsoid with equal minor axes has an unobservable axial-twist DOF)")
from numpy.linalg import eigh
for s in ["co", "fe", "ti", "ta"]:
    ars = []
    for k in (1, 2, 3):
        for sd in ("r", "l"):
            nm = f"l_{k}_{s}_{sd}"
            if nm not in gnames: continue
            ji = gnames.index(nm)
            vs = V[owner == ji]
            if len(vs) < 20: continue
            c = vs - vs.mean(0)
            w, _ = eigh(c.T @ c / len(c))
            w = np.sqrt(np.maximum(w, 0))[::-1]   # descending
            ars.append(w[1] / max(w[2], 1e-9))    # minor1/minor2
    if ars:
        print(f"   {s}: minor-axis ratio (1.0 = circular = twist unobservable) median {np.median(ars):.2f}  range {min(ars):.2f}-{max(ars):.2f}  n={len(ars)}")
