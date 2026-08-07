"""PROBE: at what smoothing width does a coarse-to-fine occupancy field MERGE adjacent legs?

Scale-space continuation only works if there is a range of smoothing widths that is
(a) wide enough to make the thin-segment occupancy non-flat at the observed error scale, and
(b) narrow enough that adjacent legs remain distinguishable.
If those two requirements do not overlap, continuation cannot resolve leg identity.
"""
import numpy as np, pickle
with open("3D_model_prep/OmniAnt_25PCs_joint_limited.pkl", "rb") as f:
    u = pickle._Unpickler(f); u.encoding = "latin1"; dd = u.load()
de = lambda x: np.asarray(x.r if hasattr(x, "r") else x, dtype=np.float64)
V = de(dd["v_template"]); W = de(dd["weights"]); jn = [str(x) for x in dd["J_names"]]
J = de(dd["J"])
owner = W.argmax(1)

def cen(nm): return V[owner == jn.index(nm)].mean(0)

print("Adjacent-leg separation in the REST template (model units):")
for s in ["co", "fe", "ti", "ta"]:
    ds = []
    for sd in ("r", "l"):
        for k in (1, 2):
            a, b = cen(f"l_{k}_{s}_{sd}"), cen(f"l_{k+1}_{s}_{sd}")
            ds.append(np.linalg.norm(a - b))
    # also left-right pair
    lr = [np.linalg.norm(cen(f"l_{k}_{s}_r") - cen(f"l_{k}_{s}_l")) for k in (1, 2, 3)]
    print(f"  {s}: fore-mid/mid-hind centroid gap  mean {np.mean(ds):.4f} (min {np.min(ds):.4f});"
          f"  left-right gap mean {np.mean(lr):.4f}")

# minimum surface-to-surface distance between distinct leg segments (worst case for merging)
import itertools
segs = {}
for k in (1, 2, 3):
    for sd in ("r", "l"):
        for s in ["ti", "ta"]:
            segs[f"l_{k}_{s}_{sd}"] = V[owner == jn.index(f"l_{k}_{s}_{sd}")]
mind = np.inf; pair = None
keys = list(segs)
for a, b in itertools.combinations(keys, 2):
    if a.split("_")[1] == b.split("_")[1] and a.split("_")[3] == b.split("_")[3]:
        continue  # same leg
    A, B = segs[a], segs[b]
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1).min()
    if d < mind: mind, pair = d, (a, b)
print(f"\nMinimum surface-to-surface gap between DIFFERENT legs' distal segments: {mind:.4f}  {pair}")

print(f"""
REQUIREMENT (a): smoothing sigma must be >= the observed positional disagreement
                 of the distal segments = ~0.108-0.135 units, otherwise the thin
                 segment's occupancy is flat/zero-gradient at that error scale.
REQUIREMENT (b): smoothing sigma must be << the separation between different legs
                 (min surface gap {mind:.4f} units) or the coarse level merges them.
OVERLAP: {'YES' if 0.135 < mind/2 else 'NO -- requirement (a) needs sigma >= 0.108, requirement (b) needs sigma << %.4f' % mind}
""")
