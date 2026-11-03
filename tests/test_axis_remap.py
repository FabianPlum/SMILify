"""Unit tests for the issue-#56 bone-local -> model-frame limit remap.

These exercise ``smil_importer.axis_remap`` directly. That module is deliberately
free of any ``bpy`` import, and we load it by file path here so the test needs
neither Blender nor the rest of the add-on (whose package ``__init__`` imports
``bpy``).
"""

import importlib.util
import itertools
from pathlib import Path

import numpy as np

# Load smil_importer/axis_remap.py in isolation (no package __init__ -> no bpy).
_MODULE_PATH = Path(__file__).resolve().parents[1] / "3D_model_prep" / "smil_importer" / "axis_remap.py"
_spec = importlib.util.spec_from_file_location("axis_remap", _MODULE_PATH)
axis_remap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(axis_remap)

is_signed_permutation = axis_remap.is_signed_permutation
remap_bounds_to_model_frame = axis_remap.remap_bounds_to_model_frame
remap_bounds_to_bone_frame = axis_remap.remap_bounds_to_bone_frame
limit_constraint_spec = axis_remap.limit_constraint_spec
rot3 = axis_remap.rot3

# B shared by the two hand-analysed bones (l_3_tr_r, b_a_1): local x->+x, y->+z,
# z->-y. A proper rotation (det +1) and a signed permutation.
B_TILT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _random_signed_permutation(rng):
    perm = rng.permutation(3)
    signs = rng.choice([-1.0, 1.0], size=3)
    B = np.zeros((3, 3))
    for row, col in enumerate(perm):
        B[row, col] = signs[row]
    return B


def _in_box(w, lo, hi):
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    return bool(np.all(w >= lo - 1e-9) and np.all(w <= hi + 1e-9))


def test_rot3_extracts_upper_left_3x3():
    m4 = [
        [1, 2, 3, 99],
        [4, 5, 6, 99],
        [7, 8, 9, 99],
        [0, 0, 0, 1],
    ]
    assert np.array_equal(rot3(m4), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_is_signed_permutation():
    assert is_signed_permutation(np.eye(3))
    assert is_signed_permutation(B_TILT)
    assert is_signed_permutation(np.diag([1.0, -1.0, 1.0]))
    # A genuine 45-deg rotation is NOT a signed permutation.
    c = np.cos(np.pi / 4)
    R = np.array([[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]])
    assert not is_signed_permutation(R)


def test_identity_is_noop():
    lo, hi = [-0.1, -0.2, -0.3], [0.4, 0.5, 0.6]
    mlo, mhi = remap_bounds_to_model_frame(np.eye(3), lo, hi, "id")
    assert mlo == lo and mhi == hi


def test_l_3_tr_r_symmetric_unchanged():
    # Y and Z identical & symmetric -> the tilt swaps them invisibly.
    lo = [-0.52, -0.5236, -0.5236]
    hi = [0.79, 0.5236, 0.5236]
    mlo, mhi = remap_bounds_to_model_frame(B_TILT, lo, hi, "l_3_tr_r")
    assert np.allclose(mlo, lo) and np.allclose(mhi, hi)


def test_b_a_1_asymmetric_swaps_and_sign_flips():
    # Authored bone-local x[-0.52,0.79] y[0,0.3491] z[0,0.5236].
    lo = [-0.52, 0.0, 0.0]
    hi = [0.79, 0.3491, 0.5236]
    mlo, mhi = remap_bounds_to_model_frame(B_TILT, lo, hi, "b_a_1")
    # model x = local x ; model y = -local z ; model z = local y.
    assert np.allclose(mlo, [-0.52, -0.5236, 0.0])
    assert np.allclose(mhi, [0.79, 0.0, 0.3491])


def test_mixed_axis_falls_back_verbatim_and_logs_when_authored():
    # Issue #56 review: the caveat goes through a plain `log` callable (default
    # print) instead of warnings.warn, so it is visible on EVERY export instead
    # of being deduped per session — and it fires only for authored limits.
    c = np.cos(np.pi / 4)
    R = np.array([[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]])
    lo, hi = [-0.1, -0.2, -0.3], [0.4, 0.5, 0.6]

    messages = []
    mlo, mhi = remap_bounds_to_model_frame(R, lo, hi, "skew", authored=True, log=messages.append)
    assert mlo == lo and mhi == hi
    assert len(messages) == 1 and "mixed-axis" in messages[0] and "skew" in messages[0]

    # Repeated calls keep logging (no warnings-module dedupe).
    remap_bounds_to_model_frame(R, lo, hi, "skew", authored=True, log=messages.append)
    assert len(messages) == 2


def test_mixed_axis_is_silent_for_unauthored_default_bounds():
    c = np.cos(np.pi / 4)
    R = np.array([[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]])
    lo, hi = [-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]

    messages = []
    mlo, mhi = remap_bounds_to_model_frame(R, lo, hi, "organic", authored=False, log=messages.append)
    assert mlo == lo and mhi == hi
    assert messages == []


def test_remap_preserves_box_membership_property():
    """The core guarantee: for a signed-permutation B, a bone-local rotation is
    inside the authored box iff B @ w is inside the remapped model-frame box."""
    rng = np.random.default_rng(0)
    for _ in range(5000):
        B = _random_signed_permutation(rng)
        lo = rng.uniform(-1.5, 1.5, 3)
        hi = lo + rng.uniform(0.0, 2.0, 3)  # hi >= lo, allows asymmetric
        w_local = rng.uniform(-2.5, 2.5, 3)
        w_model = B @ w_local

        mlo, mhi = remap_bounds_to_model_frame(B, lo.tolist(), hi.tolist(), "rnd")
        assert _in_box(w_local, lo, hi) == _in_box(w_model, mlo, mhi)


def test_remapped_bounds_stay_ordered():
    """min <= max must hold after any sign swap."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        B = _random_signed_permutation(rng)
        lo = rng.uniform(-1.5, 1.5, 3)
        hi = lo + rng.uniform(0.0, 2.0, 3)
        mlo, mhi = remap_bounds_to_model_frame(B, lo.tolist(), hi.tolist(), "ord")
        assert np.all(np.asarray(mlo) <= np.asarray(mhi) + 1e-12)


def _all_signed_permutations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            B = np.zeros((3, 3))
            for row, col in enumerate(perm):
                B[row, col] = signs[row]
            yield B


def test_bone_frame_inverse_round_trips_all_signed_permutations():
    """Issue #116: import recreates constraints with the inverse remap, so
    model -> bone-local -> model must be the identity for every clean-axis bone."""
    rng = np.random.default_rng(2)
    perms = list(_all_signed_permutations())
    assert len(perms) == 48
    for B in perms:
        lo = rng.uniform(-1.5, 1.5, 3)
        hi = lo + rng.uniform(0.0, 2.0, 3)
        blo, bhi = remap_bounds_to_bone_frame(B, lo.tolist(), hi.tolist())
        mlo, mhi = remap_bounds_to_model_frame(B, blo, bhi, "rt")
        assert np.allclose(mlo, lo) and np.allclose(mhi, hi)
    # Mixed-axis B: both directions keep the bounds verbatim, so the round trip still holds.
    c = np.cos(np.pi / 4)
    R = np.array([[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]])
    lo, hi = [-0.1, -0.2, -0.3], [0.4, 0.5, 0.6]
    assert remap_bounds_to_bone_frame(R, lo, hi) == (lo, hi)


def test_limit_constraint_spec_leaves_wide_open_axes_unticked():
    # Model z wide open; under B_TILT model z = local y, so local y stays unticked.
    lo, hi = [-0.52, -0.5236, -np.pi], [0.79, 0.0, np.pi]
    use, blo, bhi = limit_constraint_spec(B_TILT, lo, hi, wide_open=np.pi)
    assert use == [True, False, True]
    mlo, mhi = remap_bounds_to_model_frame(B_TILT, blo, bhi, "spec")
    assert np.allclose(mlo, lo) and np.allclose(mhi, hi)
    # Every axis wide open -> no constraint at all.
    assert limit_constraint_spec(B_TILT, [-np.pi] * 3, [np.pi] * 3, wide_open=np.pi) is None
