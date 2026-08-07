"""PROBE: is a per-segment ellipsoid proxy DISJOINT from its target at the observed error
scale? IoU has exactly zero gradient for disjoint bodies. If the measured cross-seed
positional disagreement exceeds the segment radius, volume IoU cannot pull it back.
"""
import numpy as np, pickle
from numpy.linalg import eigh

with open("3D_model_prep/OmniAnt_25PCs_joint_limited.pkl", "rb") as f:
    u = pickle._Unpickler(f); u.encoding = "latin1"; dd = u.load()
de = lambda x: np.asarray(x.r if hasattr(x, "r") else x, dtype=np.float64)
V = de(dd["v_template"]); W = de(dd["weights"]); jn = [str(x) for x in dd["J_names"]]
owner = W.argmax(1)

# measured cross-seed positional disagreement per segment (from fk_probe)
meas = {"co": 0.0, "tr": 0.01679, "fe": 0.08846, "ti": 0.10832, "ta": 0.12681, "pt": 0.13485}

print(f"{'seg':>4s} {'semi-major':>11s} {'semi-minor(mean)':>17s} {'measured pos disagreement':>26s} {'disagreement / (2*minor)':>25s}")
for s in ["co", "tr", "fe", "ti", "ta", "pt"]:
    maj, mino = [], []
    for k in (1, 2, 3):
        for sd in ("r", "l"):
            nm = f"l_{k}_{s}_{sd}"
            if nm not in jn: continue
            vs = V[owner == jn.index(nm)]
            if len(vs) < 12: continue
            c = vs - vs.mean(0)
            w, _ = eigh(c.T @ c / len(c))
            w = np.sqrt(np.maximum(w, 0))[::-1]
            maj.append(2 * w[0]); mino.append(w[1] + w[2])   # ~semi-axes*2 for a uniform ellipsoid: sd*sqrt(5)
    if not maj: continue
    a = np.mean(maj) * np.sqrt(5) / 2      # semi-major of the equivalent uniform ellipsoid
    b = np.mean(mino) / 2 * np.sqrt(5) / 2  # mean semi-minor
    d = meas[s]
    print(f"{s:>4s} {a:11.4f} {b:17.4f} {d:26.4f} {d/(2*b):25.2f}")

print()
print("Interpretation: ratio > 1 means the proxy ellipsoid at one seed's solution and the")
print("proxy at another seed's solution are DISJOINT -- and so, at that error scale, the")
print("proxy is disjoint from the target segment too. IoU gradient there is exactly zero.")
