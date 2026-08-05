"""CPU-only probe: does the axis-angle lerp in make_partfield_data.py:229 straighten joints?"""

import os
import sys
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = "/home/fabi/dev/SMILify"
sys.path.insert(0, REPO)
import config

NPZ = os.path.join(REPO, "diagnostics/moonshot/runs/M7_handoff_midline/Stage_3_deform_fine.npz")
SEED = 20260805

fit = np.load(NPZ)
J = fit["joint_rot"]  # (50, 54, 3) axis-angle
G = fit["global_rot"]  # (50, 3)
n_spec = J.shape[0]

with open(config.SMAL_FILE, "rb") as fh:
    u = pickle._Unpickler(fh)
    u.encoding = "latin1"
    dd = u.load()
jnames = [n.decode() if isinstance(n, bytes) else n for n in dd["J_names"]]
# joint_rot excludes root -> names[1:]
jn = jnames[1 : 1 + J.shape[1]]
print("n joint_rot joints:", J.shape[1], "names[:6]:", jn[:6])

# ---- reproduce the train/test split exactly as the script does
rng = np.random.default_rng(SEED)
perm = rng.permutation(n_spec)
tr, te = np.sort(perm[:35]), np.sort(perm[35:])
print("train pool size", len(tr), "test pool", len(te))


def ang_of_aa(v):
    """true rotation angle in deg of an axis-angle vector (wraps norms > pi)."""
    n = np.linalg.norm(v, axis=-1)
    n = np.mod(n, 2 * np.pi)
    n = np.minimum(n, 2 * np.pi - n)
    return np.degrees(n)


raw = np.linalg.norm(J, axis=-1)
print("\n=== raw axis-angle NORMS (deg) over all 50x54 fits ===")
print(
    "  mean %.1f  median %.1f  max %.1f  frac>90deg %.3f  frac>180deg %.4f"
    % (
        np.degrees(raw).mean(),
        np.degrees(np.median(raw)),
        np.degrees(raw.max()),
        (np.degrees(raw) > 90).mean(),
        (np.degrees(raw) > 180).mean(),
    )
)
tr_ang = ang_of_aa(J)
print("=== TRUE rotation angles (deg, wrapped) ===")
print(
    "  mean %.1f  median %.1f  max %.1f  frac>90deg %.3f"
    % (tr_ang.mean(), np.median(tr_ang), tr_ang.max(), (tr_ang > 90).mean())
)

# ---- 4000 random (a,b,t) draws from the TRAIN pool, mirroring gen_synth
N = 4000
rs = np.random.default_rng(0)
a = rs.choice(tr, N)
b = rs.choice(tr, N)
t = rs.uniform(0, 1, N)

A = J[a]
B = J[b]  # (N,54,3)
mix = A * (1 - t)[:, None, None] + B * t[:, None, None]

# --- true rotation angle of the lerped vector vs endpoints
ang_mix = ang_of_aa(mix)
ang_A, ang_B = ang_of_aa(A), ang_of_aa(B)
ang_lin_endpoint = ang_A * (1 - t)[:, None] + ang_B * t[:, None]

# --- what SLERP would give
flatA = A.reshape(-1, 3)
flatB = B.reshape(-1, 3)
RA = Rot.from_rotvec(flatA)
RB = Rot.from_rotvec(flatB)
# slerp per-pair: R(t) = RA * (RA^-1 RB)^t
Rrel = RA.inv() * RB
rel = Rrel.as_rotvec()
tt = np.repeat(t, J.shape[1])[:, None]
Rslerp = RA * Rot.from_rotvec(rel * tt)
ang_slerp = np.degrees(np.linalg.norm(Rslerp.as_rotvec(), axis=-1)).reshape(N, -1)

# --- deviation between lerp rotation and the true geodesic
Rmix = Rot.from_rotvec(mix.reshape(-1, 3))
dev = np.degrees(np.linalg.norm((Rmix.inv() * Rslerp).as_rotvec(), axis=-1)).reshape(N, -1)

print("\n=== mean joint rotation angle (deg) over 4000 lerp draws, train pool ===")
print(
    "  endpoints, linear-in-angle reference : mean %.1f  median %.1f"
    % (ang_lin_endpoint.mean(), np.median(ang_lin_endpoint))
)
print("  SLERP (true geodesic)               : mean %.1f  median %.1f" % (ang_slerp.mean(), np.median(ang_slerp)))
print("  LERP as shipped                     : mean %.1f  median %.1f" % (ang_mix.mean(), np.median(ang_mix)))
print(
    "  ratio lerp/slerp  %.3f    lerp/linear-endpoint %.3f"
    % (ang_mix.mean() / ang_slerp.mean(), ang_mix.mean() / ang_lin_endpoint.mean())
)

print("\n=== geodesic deviation lerp-vs-slerp (deg) ===")
print(
    "  mean %.2f  median %.2f  p95 %.2f  max %.1f  frac>20 %.4f  frac>45 %.4f"
    % (dev.mean(), np.median(dev), np.percentile(dev, 95), dev.max(), (dev > 20).mean(), (dev > 45).mean())
)
per_sample_bad = (dev > 45).any(1).mean()
print("  fraction of SAMPLES with >=1 joint >45deg off the geodesic: %.3f" % per_sample_bad)
per_sample_bad20 = (dev > 20).any(1).mean()
print("  fraction of SAMPLES with >=1 joint >20deg off the geodesic: %.3f" % per_sample_bad20)

# ---- per-joint table, worst shrinkage
print("\n=== per-joint mean angle (deg): slerp -> lerp, worst 12 by absolute loss ===")
ms = ang_slerp.mean(0)
ml = ang_mix.mean(0)
order = np.argsort(-(ms - ml))
for i in order[:12]:
    print("  %-12s  slerp %6.1f  lerp %6.1f   ratio %.2f" % (jn[i], ms[i], ml[i], ml[i] / max(ms[i], 1e-9)))

# ---- how much does the jitter add back?
sd = J[:, :, :].std(0) * 0.06
print("\njitter sd (deg) per-component: mean %.2f max %.2f" % (np.degrees(sd).mean(), np.degrees(sd).max()))

# ---- global_rot
gmix = G[a] * (1 - t)[:, None] + G[b] * t[:, None]
gRA = Rot.from_rotvec(G[a])
gRB = Rot.from_rotvec(G[b])
gRs = gRA * Rot.from_rotvec((gRA.inv() * gRB).as_rotvec() * t[:, None])
gdev = np.degrees(np.linalg.norm((Rot.from_rotvec(gmix).inv() * gRs).as_rotvec(), axis=-1))
print(
    "\n=== global_rot lerp-vs-slerp deviation (deg): mean %.1f median %.1f max %.1f ==="
    % (gdev.mean(), np.median(gdev), gdev.max())
)
print("global_rot true angles: mean %.1f  max %.1f" % (ang_of_aa(G).mean(), ang_of_aa(G).max()))
