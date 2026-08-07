"""PROBE: can 'the target mesh's INTERIOR' even be defined for the worker scans?
The whole proposal is 'maximise volume IoU with the target mesh's interior'.
If the interior is ill-defined, the objective is ill-defined.
"""
import glob, numpy as np, trimesh, warnings
warnings.filterwarnings("ignore")

files = sorted(glob.glob("diagnostics/moonshot/bench50/*.obj"))[:12]
print(f"{'mesh':<44s} {'watertight':>10s} {'wind_consist':>12s} {'components':>10s} {'vol':>10s} {'hull_vol':>10s} {'vol/hull':>9s}")
rows = []
for f in files:
    m = trimesh.load(f, process=False)
    try:
        hv = m.convex_hull.volume
    except Exception:
        hv = float("nan")
    v = m.volume
    rows.append((m.is_watertight, m.is_winding_consistent, len(m.split(only_watertight=False)), v, hv))
    print(f"{f.split('/')[-1][:43]:<44s} {str(m.is_watertight):>10s} {str(m.is_winding_consistent):>12s} "
          f"{len(m.split(only_watertight=False)):>10d} {v:10.5f} {hv:10.5f} {v/hv if hv==hv and hv>0 else float('nan'):9.3f}")

wt = sum(r[0] for r in rows)
print(f"\nwatertight: {wt}/{len(rows)}")
neg = sum(1 for r in rows if r[3] <= 0)
print(f"non-positive 'volume' from divergence theorem: {neg}/{len(rows)}  <-- meaningless interior")
vh = [r[3]/r[4] for r in rows if r[4] == r[4] and r[4] > 0]
print(f"volume/hull ratio: median {np.median(vh):.3f}  min {min(vh):.3f}  max {max(vh):.3f}")
print("  (a plausible ant should be well under 0.5 of its convex hull; values near or above")
print("   that, or negative, indicate the divergence-theorem volume is not a real interior)")

# generalized winding number: the principled fallback. is it available?
try:
    from trimesh.proximity import ProximityQuery  # noqa
    print("\ntrimesh available. checking generalized winding number support...")
    m = trimesh.load(files[0], process=False)
    pts = m.bounding_box.sample_volume(2000)
    import time
    t = time.time()
    try:
        w = trimesh.proximity.max_tangent_sphere  # placeholder
        from trimesh.points import PointCloud  # noqa
        inside = m.contains(pts)
        print(f"  m.contains() on 2000 pts: {time.time()-t:.2f}s, inside fraction {inside.mean():.3f}"
              f"   (ray-based; unreliable on non-watertight input)")
    except Exception as e:
        print("  m.contains failed:", e)
except Exception as e:
    print("trimesh proximity unavailable:", e)
