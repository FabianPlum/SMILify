"""Pure-numpy self-consistency check for the issue-#56 axis remap.

No Blender needed. It mirrors ``_remap_bounds_to_model_frame`` from
``smil_importer.core_mesh`` and proves the property that matters: after the
remap, a rotation that is inside the authored *bone-local* box is inside the
exported *model-frame* box, and one that is outside stays outside. In other
words, the fitter (which reads model-frame components) enforces exactly the box
the user drew in the bone-local frame.

It also reproduces the two hand-analysed bones (l_3_tr_r, b_a_1), both of which
share the rest matrix B = [[1,0,0],[0,0,-1],[0,1,0]].

Run: python diagnostics/check_remap_math.py
"""

import numpy as np

RNG = np.random.default_rng(0)


# --- logic mirrored from core_mesh (kept in sync by hand; bpy-free) ----------
def is_signed_permutation(B, atol=1e-3):
    A = np.abs(np.asarray(B, dtype=np.float64))
    return (
        np.allclose(A.sum(axis=0), 1.0, atol=atol)
        and np.allclose(A.sum(axis=1), 1.0, atol=atol)
        and np.allclose(A.max(axis=0), 1.0, atol=atol)
        and np.allclose(A.max(axis=1), 1.0, atol=atol)
    )


def remap_bounds_to_model_frame(B, lo, hi):
    B = np.asarray(B, dtype=np.float64)
    if np.allclose(B, np.eye(3), atol=1e-3):
        return list(lo), list(hi)
    if not is_signed_permutation(B):
        return list(lo), list(hi)  # verbatim fallback (mixed axis)
    mlo, mhi = [0.0] * 3, [0.0] * 3
    for m in range(3):
        k = int(np.argmax(np.abs(B[m])))
        if B[m, k] >= 0:
            mlo[m], mhi[m] = lo[k], hi[k]
        else:
            mlo[m], mhi[m] = -hi[k], -lo[k]
    return mlo, mhi


# --- helpers -----------------------------------------------------------------
def random_signed_permutation(rng):
    perm = rng.permutation(3)
    signs = rng.choice([-1.0, 1.0], size=3)
    B = np.zeros((3, 3))
    for row, col in enumerate(perm):
        B[row, col] = signs[row]
    return B


def in_box(w, lo, hi):
    return np.all(w >= np.asarray(lo) - 1e-9) and np.all(w <= np.asarray(hi) + 1e-9)


# --- property test -----------------------------------------------------------
def test_membership_equivalence(trials=20000):
    """For random B (signed perm), local box, and local rotation w_local:
    membership in the local box == membership of B@w_local in the remapped box."""
    mismatches = 0
    for _ in range(trials):
        B = random_signed_permutation(RNG)
        lo = RNG.uniform(-1.5, 1.5, 3)
        hi = lo + RNG.uniform(0.0, 2.0, 3)  # ensure hi >= lo, allow asymmetric
        w_local = RNG.uniform(-2.5, 2.5, 3)
        w_model = B @ w_local

        mlo, mhi = remap_bounds_to_model_frame(B, lo.tolist(), hi.tolist())
        inside_local = in_box(w_local, lo, hi)
        inside_model = in_box(w_model, mlo, mhi)
        if inside_local != inside_model:
            mismatches += 1
    return mismatches


def show_analysed_bones():
    B = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    print("B (l_3_tr_r and b_a_1):\n", B, "\n signed-permutation?", is_signed_permutation(B))

    cases = {
        "l_3_tr_r (Y=Z symmetric)": ([-0.52, -0.5236, -0.5236], [0.79, 0.5236, 0.5236]),
        "b_a_1   (Y!=Z asymmetric)": ([-0.52, 0.0, 0.0], [0.79, 0.3491, 0.5236]),
    }
    for name, (lo, hi) in cases.items():
        mlo, mhi = remap_bounds_to_model_frame(B, lo, hi)
        print(f"\n{name}")
        print(f"  authored (bone-local) x/y/z : {list(zip(lo, hi))}")
        print(f"  exported (model-frame) x/y/z: {list(zip(mlo, mhi))}")


if __name__ == "__main__":
    print("=" * 66)
    print("Reproducing the two hand-analysed bones")
    print("=" * 66)
    show_analysed_bones()

    print("\n" + "=" * 66)
    print("Randomised membership-equivalence property")
    print("=" * 66)
    n = test_membership_equivalence()
    print(f"mismatches over 20000 random (B, box, rotation) trials: {n}")
    assert n == 0, "remap does NOT preserve box membership!"
    print("PASS: remapped model-frame box == authored bone-local box, exactly.")
