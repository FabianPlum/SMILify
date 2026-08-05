"""Compare exported SMIL pkls: byte identity + semantic (key/dtype/bit-exact array) parity.

DIAGNOSTIC HARNESS for reviewing PR #88 — not part of the addon, never committed.
Run with the pytorch3d env python (numpy 1.26.4) so a successful load doubles as
the consumer-side compatibility test (matrix C).

Usage: python compare_exports.py <label:pathA:pathB> [...more pairs]
Every file mentioned is also load-tested individually.
"""

import hashlib
import pickle
import sys

import numpy as np

print(f"numpy {np.__version__}")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def semantic_diff(a, b):
    """Return list of human-readable differences between two SMIL data dicts."""
    diffs = []
    keys_a, keys_b = set(a), set(b)
    for k in sorted(keys_a - keys_b):
        diffs.append(f"key {k!r} only in A")
    for k in sorted(keys_b - keys_a):
        diffs.append(f"key {k!r} only in B")
    for k in sorted(keys_a & keys_b):
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            va, vb = np.asarray(va), np.asarray(vb)
            if va.dtype != vb.dtype:
                diffs.append(f"{k}: dtype {va.dtype} vs {vb.dtype}")
            elif va.shape != vb.shape:
                diffs.append(f"{k}: shape {va.shape} vs {vb.shape}")
            elif va.tobytes() != vb.tobytes():
                if va.dtype.kind in "fc" and np.allclose(va, vb, rtol=0, atol=0, equal_nan=True):
                    diffs.append(f"{k}: bit-different but value-equal (±0.0 / NaN payload)")
                else:
                    with np.errstate(all="ignore"):
                        try:
                            md = float(np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64))))
                        except Exception:
                            md = float("nan")
                    diffs.append(f"{k}: values differ (max abs diff {md:.3e})")
        else:
            if type(va) is not type(vb) or va != vb:
                diffs.append(f"{k}: {va!r} vs {vb!r}")
    return diffs


failures = 0

# Individual load test (matrix C)
all_files = []
for arg in sys.argv[1:]:
    _, pa, pb = arg.split(":")
    all_files += [pa, pb]
print("\n=== C: load test in this env ===")
loaded = {}
for path in dict.fromkeys(all_files):
    try:
        loaded[path] = load(path)
        print(f"[PASS] load {path}")
    except Exception as e:
        failures += 1
        print(f"[FAIL] load {path}: {type(e).__name__}: {e}")

# Pairwise comparisons
for arg in sys.argv[1:]:
    label, pa, pb = arg.split(":")
    print(f"\n=== {label} ===\n  A: {pa}\n  B: {pb}")
    ha, hb = sha256(pa), sha256(pb)
    print(f"  sha256 A: {ha}\n  sha256 B: {hb}")
    if ha == hb:
        print("  BYTE-IDENTICAL")
        continue
    print("  bytes differ -> semantic comparison:")
    if pa not in loaded or pb not in loaded:
        print("  (cannot compare semantically, load failed)")
        failures += 1
        continue
    diffs = semantic_diff(loaded[pa], loaded[pb])
    if not diffs:
        print("  SEMANTICALLY IDENTICAL (same keys, dtypes, bit-exact arrays; only pickle framing differs)")
    else:
        failures += 1
        for d in diffs:
            print(f"  - {d}")

print(f"\n=== {failures} failing comparison(s)/load(s) ===")
sys.exit(1 if failures else 0)
