"""Probe 2: decompose the shrinkage into (a) interpolation-at-all vs (b) lerp-vs-slerp,
and check whether lerp leaves the geodesic segment between the two endpoint poses."""

import sys
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = "/home/fabi/dev/SMILify"
sys.path.insert(0, REPO)
import config

fit = np.load(REPO + "/diagnostics/moonshot/runs/M7_handoff_midline/Stage_3_deform_fine.npz")
J = fit["joint_rot"]
with open(config.SMAL_FILE, "rb") as fh:
    u = pickle._Unpickler(fh)
    u.encoding = "latin1"
    dd = u.load()
jnames = [n.decode() if isinstance(n, bytes) else n for n in dd["J_names"]]
jn = jnames[1 : 1 + J.shape[1]]

rng = np.random.default_rng(20260805)
perm = rng.permutation(J.shape[0])
tr = np.sort(perm[:35])

N = 6000
rs = np.random.default_rng(1)
a = rs.choice(tr, N)
b = rs.choice(tr, N)
t = rs.uniform(0, 1, N)
same = a == b
print("fraction of draws with a == b (interpolation is a no-op): %.3f" % same.mean())

A, B = J[a], J[b]
mix = A * (1 - t)[:, None, None] + B * t[:, None, None]
fA, fB, fM = A.reshape(-1, 3), B.reshape(-1, 3), mix.reshape(-1, 3)
RA, RB, RM = Rot.from_rotvec(fA), Rot.from_rotvec(fB), Rot.from_rotvec(fM)
tt = np.repeat(t, J.shape[1])[:, None]
RS = RA * Rot.from_rotvec((RA.inv() * RB).as_rotvec() * tt)


def geo(R1, R2):
    return np.degrees(np.linalg.norm((R1.inv() * R2).as_rotvec(), axis=-1))


dAB = geo(RA, RB)
dev_lerp = geo(RM, RS)
# distance of each candidate from the two endpoints
dMA, dMB = geo(RM, RA), geo(RM, RB)
dSA, dSB = geo(RS, RA), geo(RS, RB)
# "outside the segment": sum of distances to endpoints exceeds the endpoint separation
slack_lerp = (dMA + dMB) - dAB
slack_slerp = (dSA + dSB) - dAB
print("\n=== detour beyond the geodesic segment A->B (deg; 0 == on the shortest path) ===")
print(
    "  slerp : mean %.3f  p99 %.3f  max %.3f" % (slack_slerp.mean(), np.percentile(slack_slerp, 99), slack_slerp.max())
)
print(
    "  lerp  : mean %.3f  p99 %.3f  max %.3f  frac>30deg %.4f  frac>90deg %.4f"
    % (
        slack_lerp.mean(),
        np.percentile(slack_lerp, 99),
        slack_lerp.max(),
        (slack_lerp > 30).mean(),
        (slack_lerp > 90).mean(),
    )
)
per_s = (slack_lerp.reshape(N, -1) > 30).any(1)
print("  fraction of SAMPLES with >=1 joint detouring >30deg off the segment: %.3f" % per_s.mean())


# how much of the total shrinkage does slerp fix?
def ang(v):
    n = np.mod(np.linalg.norm(v, axis=-1), 2 * np.pi)
    return np.degrees(np.minimum(n, 2 * np.pi - n))


pool_mean = ang(J[tr]).mean()
pool_med = np.median(ang(J[tr]))
aS = np.degrees(np.linalg.norm(RS.as_rotvec(), axis=-1)).reshape(N, -1)
aM = ang(mix)
print("\n=== mean / median joint rotation angle (deg), train pool ===")
print("  the fits themselves        : mean %.2f  median %.2f" % (pool_mean, pool_med))
print("  slerp-interpolated corpus  : mean %.2f  median %.2f" % (aS.mean(), np.median(aS)))
print("  lerp-interpolated (shipped): mean %.2f  median %.2f" % (aM.mean(), np.median(aM)))
tot = pool_mean - aM.mean()
fixed = aS.mean() - aM.mean()
print(
    "  total shift vs fits: %.2f deg (%.1f%%);  removed by switching to slerp: %.2f deg (%.1f%% of the shift)"
    % (tot, 100 * tot / pool_mean, fixed, 100 * fixed / tot)
)
totm = pool_med - np.median(aM)
fixm = np.median(aS) - np.median(aM)
print(
    "  median shift %.2f deg (%.1f%%); slerp removes %.2f deg (%.1f%% of it)"
    % (totm, 100 * totm / pool_med, fixm, 100 * fixm / totm)
)

# per-joint, the six distal groups
print("\n=== per-joint, distal segments (ta/ti): fits -> slerp -> lerp (mean deg) ===")
poolj = ang(J[tr]).mean(0)
for i, n in enumerate(jn):
    if ("_ta_" in n) or ("_ti_" in n):
        print(
            "  %-10s fits %6.1f | slerp %6.1f (%.2f) | lerp %6.1f (%.2f of fits, %.2f of slerp)"
            % (
                n,
                poolj[i],
                aS.mean(0)[i],
                aS.mean(0)[i] / poolj[i],
                aM.mean(0)[i],
                aM.mean(0)[i] / poolj[i],
                aM.mean(0)[i] / aS.mean(0)[i],
            )
        )

# are the big deviations driven by the >180deg wrapped entries?
big = np.linalg.norm(fA, axis=-1) > np.pi
bigb = np.linalg.norm(fB, axis=-1) > np.pi
anybig = big | bigb
print("\nfrac of joint-pairs where an endpoint has |aa| > 180deg: %.4f" % anybig.mean())
print(
    "  mean lerp-vs-slerp deviation when a wrapped endpoint present: %.1f deg; otherwise %.2f deg"
    % (dev_lerp[anybig].mean(), dev_lerp[~anybig].mean())
)
print("  frac of >45deg deviations that involve a wrapped endpoint: %.3f" % (anybig[dev_lerp > 45].mean()))
print("  mean endpoint separation |A,B| = %.1f deg; frac of pairs >90deg apart: %.3f" % (dAB.mean(), (dAB > 90).mean()))
