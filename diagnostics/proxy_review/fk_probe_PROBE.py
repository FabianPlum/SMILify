"""PROBE: is the cross-seed distal-leg disagreement a POSITION disagreement (the leg is
somewhere else -> volumetric IoU could help) or a ROTATION-ONLY / gauge disagreement (the
leg is in the same place, differently parameterised -> no search strategy can fix it)?

Pure-numpy forward kinematics on the template rest joints, driven by the stored per-seed
joint_rot. Per-joint scale and betas_trans are NOT applied (not stored in pose_basins.npz),
so this isolates the ROTATION-induced positional disagreement -- a lower bound.
"""
import numpy as np, pickle, os

PKL = "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"
with open(PKL, "rb") as f:
    u = pickle._Unpickler(f); u.encoding = "latin1"; dd = u.load()

def deChumpy(x):
    return np.asarray(x.r if hasattr(x, "r") else x, dtype=np.float64)

Jr = deChumpy(dd["J"])                       # (J,3) rest joint locations
par = np.asarray(dd["kintree_table"][0], dtype=np.int64)
jnames = [str(x) for x in dd["J_names"]]
J = Jr.shape[0]
print("joints", J, "parents[0]", par[0])

def rodrigues(v):
    # v (...,3) -> (...,3,3)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = v / np.maximum(th, 1e-12)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    s = np.sin(th)[..., None]; c = np.cos(th)[..., None]
    return I + s * K + (1 - c) * (K @ K)

def fk(rot_full):
    """rot_full (J,3) axis-angle incl. root -> (J,3) posed joint positions."""
    R = rodrigues(rot_full)
    G = [None] * J
    A = np.eye(4); A[:3, :3] = R[0]; A[:3, 3] = Jr[0]
    G[0] = A
    for i in range(1, J):
        p = par[i]
        A = np.eye(4); A[:3, :3] = R[i]; A[:3, 3] = Jr[i] - Jr[p]
        G[i] = G[p] @ A
    return np.stack([g[:3, 3] for g in G])

d = np.load("diagnostics/moonshot/out/pose_basins.npz", allow_pickle=True)
free = d["free"]; jn_p = [str(x) for x in d["joint_names"]]   # (S,B,54,3), excludes root
S, B = free.shape[0], free.shape[1]

# map probe joint order -> model joint order (probe excludes root, index 0)
assert jn_p == jnames[1:], (jn_p[:3], jnames[:4])

extent = Jr.max(0) - Jr.min(0)
diag = float(np.linalg.norm(extent))
print(f"template joint-cloud diagonal = {diag:.4f} model units")

pos = np.zeros((S, B, J, 3))
for s in range(S):
    for b in range(B):
        rf = np.zeros((J, 3)); rf[1:] = free[s, b]
        pos[s, b] = fk(rf)

def pair_mean(a):  # a (S,B,J,3) -> (B,J)
    acc = []
    for i in range(S):
        for j in range(i + 1, S):
            acc.append(np.linalg.norm(a[i] - a[j], axis=-1))
    return np.mean(acc, axis=0)

pd_ = pair_mean(pos)   # (B,J) in model units
segs = ["co", "tr", "fe", "ti", "ta", "pt"]
print()
print("CROSS-SEED JOINT-POSITION disagreement (rotation-only FK), by segment depth")
print("  seg    mean disp (units)   % of joint-cloud diagonal")
for s in segs:
    rows = [jnames.index(f"l_{k}_{s}_{sd}") for k in (1, 2, 3) for sd in ("r", "l")]
    m = pd_[:, rows].mean()
    print(f"  {s:>4s}   {m:10.5f}          {100*m/diag:6.2f}%")

# tip of each leg = pretarsus
tips = [jnames.index(f"l_{k}_pt_{sd}") for k in (1, 2, 3) for sd in ("r", "l")]
print()
print(f"leg-TIP position disagreement: mean {pd_[:,tips].mean():.5f} units = {100*pd_[:,tips].mean()/diag:.2f}% of diagonal")

# how much of the per-joint ROTATION spread is 'gauge' (child position unchanged)?
print()
print("Rotation spread vs induced child-position spread -- gauge check")
for s in ["fe", "ti", "ta"]:
    rows = [jnames.index(f"l_{k}_{s}_{sd}") for k in (1, 2, 3) for sd in ("r", "l")]
    childs = []
    for r in rows:
        c = [i for i in range(J) if par[i] == r]
        childs += c
    rotsp = np.degrees(pair_mean(np.concatenate([np.zeros((S, B, 1, 3)), free], axis=2))[:, rows]).mean()
    posc = pd_[:, childs].mean()
    # expected displacement if rotation were fully 'effective': theta * bone length
    L = np.mean([np.linalg.norm(Jr[c] - Jr[par[c]]) for c in childs])
    expected = np.radians(rotsp) * L
    print(f"  {s}: rot spread {rotsp:5.1f} deg, bone len {L:.4f}; child pos spread {posc:.5f} "
          f"vs {expected:.5f} if fully effective -> effectiveness {100*posc/max(expected,1e-9):5.1f}%")
