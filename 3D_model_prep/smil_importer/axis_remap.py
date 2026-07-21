"""Bone-local -> model-frame rotation-limit remap (issue #56).

Pure-numpy helpers, deliberately free of any ``bpy`` import so they can be unit
tested without Blender (see ``tests/test_axis_remap.py``). ``core_mesh`` imports
these and applies them inside ``export_joint_limits_to_npy``.

Background
----------
A Limit Rotation constraint is authored in a bone's *local* frame, but the
fitter's ``LimitPrior`` compares the pose's *model-frame* axis-angle components
against the exported bounds. A rotation vector authored in bone-local
coordinates ``w_local`` appears in the model frame as ``w_model = B @ w_local``,
where ``B``'s columns are the bone-local x/y/z axes expressed in model
coordinates (``B = rot3(bone.matrix_local)``).

For the common "clean axis" case - ``B`` a signed permutation (each row/column a
single +/-1) - an axis-aligned box stays axis-aligned under the change of basis,
so the per-axis [min,max] bounds can be remapped exactly by permuting rows and
swapping/negating the sign-flipped axes. For a genuinely rotated (mixed-axis)
``B`` the box becomes tilted and no per-axis representation is exact; we keep the
bounds verbatim and warn (the bounded issue-#56 caveat).
"""

import warnings

import numpy as np


def rot3(m4):
    """Upper-left 3x3 of a 4x4 matrix as a float64 numpy array.

    Accepts anything supporting ``m4[r][c]`` indexing (a Blender ``Matrix`` or a
    nested Python/numpy sequence). Columns are the bone-local x/y/z axes
    expressed in the armature/model frame - the same ``B`` the rotation-
    convention probe computes.
    """
    return np.array([[m4[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)


def is_signed_permutation(B, atol=1e-3):
    """True iff every row and column of ``B`` is a single +/-1 entry.

    Such a matrix only permutes and/or flips axes, so an axis-aligned box in one
    frame stays axis-aligned in the other.
    """
    A = np.abs(np.asarray(B, dtype=np.float64))
    return (
        np.allclose(A.sum(axis=0), 1.0, atol=atol)
        and np.allclose(A.sum(axis=1), 1.0, atol=atol)
        and np.allclose(A.max(axis=0), 1.0, atol=atol)
        and np.allclose(A.max(axis=1), 1.0, atol=atol)
    )


def remap_bounds_to_model_frame(B, lo, hi, bone_name=""):
    """Re-express per-axis bone-local limits ``lo``/``hi`` in the model frame.

    For a signed-permutation ``B``, model axis ``m`` equals exactly one bone-local
    axis ``k`` (optionally negated): a ``+`` keeps ``[lo_k, hi_k]``, a ``-`` gives
    ``[-hi_k, -lo_k]``. Identity ``B`` is a no-op. A mixed-axis ``B`` cannot be
    represented as a per-axis box, so the verbatim bounds are returned and a
    warning is emitted.

    Returns ``(model_lo, model_hi)`` as plain 3-element lists.
    """
    B = np.asarray(B, dtype=np.float64)
    if np.allclose(B, np.eye(3), atol=1e-3):
        return list(lo), list(hi)
    if not is_signed_permutation(B):
        warnings.warn(
            f"Bone '{bone_name}': rest orientation is a mixed-axis rotation, so its "
            f"bone-local rotation limits cannot be expressed as an axis-aligned box "
            f"in the model frame. Exporting the bounds verbatim; the fitter will treat "
            f"them as approximate model-frame bounds (issue #56 caveat).",
            stacklevel=2,
        )
        return list(lo), list(hi)

    mlo = [0.0, 0.0, 0.0]
    mhi = [0.0, 0.0, 0.0]
    for m in range(3):
        k = int(np.argmax(np.abs(B[m])))
        if B[m, k] >= 0:
            mlo[m], mhi[m] = lo[k], hi[k]
        else:
            mlo[m], mhi[m] = -hi[k], -lo[k]
    return mlo, mhi
