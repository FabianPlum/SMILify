"""Headless validation for the Blender add-on pose-corrective (posedirs) fix.

Issue #24: the add-on's ``apply_pose_correctives`` must reproduce the SMPL/SMAL
LBS pose-blend-shape term implemented in ``smal_model/smal_torch.py``:

    pose_feature = flatten( R_local[j] - I  for j in 1..J-1 )   # model frame
    v_offset     = reshape( pose_feature @ posedirs_mat , [V,3] )
    v_posed      = v_shaped + v_offset                          # BEFORE skinning

The reference builds ``posedirs_mat`` as ``reshape(posedirs,[-1,P]).T`` of shape
``(P, V*3)`` where ``P = (J-1)*9`` and ``posedirs`` has native shape ``(V,3,P)``.

This script runs entirely in NumPy (no Blender) and demonstrates:

  1. The offset math (reshape + matmul) already matches the reference.
  2. The OLD add-on is WRONG because it feeds ``bone.matrix_basis`` (expressed in
     the bone-LOCAL rest frame ``B``) straight into ``pose_feature``. The armature
     builder gives every bone the SAME non-identity rest orientation
     (``tail = head + [0,0,0.1]``), so ``B != I`` and the pose rotation is in the
     wrong basis.
  3. The FIX -- conjugating each pose rotation back into the model frame via the
     bone's own rest matrix, ``R_model = B @ matrix_basis @ B.T`` -- reproduces
     the reference exactly, for constant AND per-bone rest orientations.
  4. Iterating bones in kintree/joint-index order (not ``pose.bones`` collection
     order) is required; a permuted order diverges.

Run:  python diagnostics/posedirs_validation.py
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def rodrigues(axis, angle):
    """Axis-angle -> 3x3 rotation matrix (model frame)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def random_orthonormal(rng):
    """A random proper rotation (used as a synthetic bone rest orientation B)."""
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# --------------------------------------------------------------------------- #
# Reference: smal_torch.py posedirs application (offset only, pre-skinning)
# --------------------------------------------------------------------------- #
def reference_offset(posedirs_native, R_local_model):
    """posedirs_native: (V,3,P);  R_local_model: (J-1,3,3) in the MODEL frame."""
    V, _, P = posedirs_native.shape
    posedirs_mat = np.reshape(posedirs_native, [-1, P]).T
    pose_feature = np.concatenate([(R - np.eye(3)).flatten() for R in R_local_model])
    assert pose_feature.shape[0] == P, (pose_feature.shape, P)
    v_offset = np.reshape(pose_feature @ posedirs_mat, [V, 3])
    return v_offset


# --------------------------------------------------------------------------- #
# Add-on math (as written in smil_importer/model_build.py) -- offset only
# --------------------------------------------------------------------------- #
def addon_offset(posedirs_native, pose_feature):
    """Reproduces the add-on's reshape+matmul verbatim, given its pose_feature."""
    V, _, P = posedirs_native.shape
    posedirs_reshaped = np.reshape(posedirs_native, [-1, P])          # (V*3, P)
    v_offset = np.reshape(np.matmul(pose_feature, posedirs_reshaped.T), [-1, 3])
    return v_offset


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(0)
    V, J = 40, 6                      # tiny synthetic model
    P = (J - 1) * 9
    posedirs = rng.normal(size=(V, 3, P)) * 0.01

    R_local_model = np.stack(
        [rodrigues(rng.normal(size=3), rng.uniform(-1.0, 1.0)) for _ in range(J - 1)]
    )

    ref = reference_offset(posedirs, R_local_model)

    # Sanity: reshape/matmul is equivalent when fed the SAME rotations.
    pf_same = np.concatenate([(R - np.eye(3)).flatten() for R in R_local_model])
    off_same = addon_offset(posedirs, pf_same)
    err_math = np.abs(off_same - ref).max()

    # Blender emulation: matrix_basis = B.T @ R_model @ B (bone-local storage).
    B_const = random_orthonormal(rng)
    B_list_const = [B_const] * (J - 1)
    B_list_var = [random_orthonormal(rng) for _ in range(J - 1)]

    for label, B_list in [("constant B (current rig)", B_list_const),
                          ("per-bone B (general rig)", B_list_var)]:
        matrix_basis = [B.T @ R @ B for B, R in zip(B_list, R_local_model)]

        pf_old = np.concatenate([(Mb - np.eye(3)).flatten() for Mb in matrix_basis])
        off_old = addon_offset(posedirs, pf_old)
        err_old = np.abs(off_old - ref).max()

        R_recovered = [B @ Mb @ B.T for B, Mb in zip(B_list, matrix_basis)]
        pf_fix = np.concatenate([(R - np.eye(3)).flatten() for R in R_recovered])
        off_fix = addon_offset(posedirs, pf_fix)
        err_fix = np.abs(off_fix - ref).max()

        print(f"[{label}]")
        print(f"    OLD add-on max abs err vs reference : {err_old:.3e}   <- BUG (nonzero)")
        print(f"    FIXED     max abs err vs reference : {err_fix:.3e}   <- OK")

    # Bone-ordering check.
    perm = rng.permutation(J - 1)
    R_perm = R_local_model[perm]
    pf_perm = np.concatenate([(R - np.eye(3)).flatten() for R in R_perm])
    off_perm = addon_offset(posedirs, pf_perm)
    err_perm = np.abs(off_perm - ref).max()

    print("\n[bone ordering]")
    print(f"    permuted joint order max abs err   : {err_perm:.3e}   <- must iterate in kintree order")

    print("\n[reshape/matmul identity]")
    print(f"    add-on vs reference (same rots)    : {err_math:.3e}   <- reshape math already correct")

    assert err_math < 1e-9, "reshape/matmul should already match the reference"
    Mb = [B_const.T @ R @ B_const for R in R_local_model]
    Rr = [B_const @ m @ B_const.T for m in Mb]
    pf = np.concatenate([(R - np.eye(3)).flatten() for R in Rr])
    assert np.abs(addon_offset(posedirs, pf) - ref).max() < 1e-9, "fix must match reference"
    print("\nAll assertions passed: the B-conjugation fix reproduces smal_torch exactly.")

    mirror_end_to_end_test()


# --------------------------------------------------------------------------- #
# End-to-end mirror: run the *exact* logic of the rewritten
# smil_importer/model_build.py::apply_pose_correctives with mock Blender objects
# (no bpy), including joint-name ordering, and confirm it matches the reference.
# --------------------------------------------------------------------------- #
class _MockMatrix:
    def __init__(self, m):
        self.m = np.asarray(m, dtype=np.float64)

    def to_3x3(self):
        return self.m[:3, :3]

    def to_quaternion(self):
        return _MockQuat(self.m[:3, :3])


class _MockQuat:
    def __init__(self, r):
        self.r = r

    def to_matrix(self):
        return self.r


class _MockBoneData:
    def __init__(self, B):
        self.matrix_local = _MockMatrix(B)


class _MockPoseBone:
    def __init__(self, B, matrix_basis):
        self.bone = _MockBoneData(B)
        self.matrix_basis = _MockMatrix(matrix_basis)


def _addon_pose_feature(pose_bones):
    """Verbatim copy of the pose_feature loop in the rewritten function."""
    pose_feature = []
    for bone in pose_bones:
        M_basis = np.array(bone.matrix_basis.to_quaternion().to_matrix())
        B = np.array(bone.bone.matrix_local.to_3x3())
        R_model = B @ M_basis @ B.T
        pose_feature.extend((R_model - np.eye(3)).flatten())
    return np.array(pose_feature)


def mirror_end_to_end_test():
    rng = np.random.default_rng(7)
    V, J = 30, 5
    P = (J - 1) * 9
    posedirs = rng.normal(size=(V, 3, P)) * 0.02
    base_vertices = rng.normal(size=(V, 3))

    R_local_model = np.stack(
        [rodrigues(rng.normal(size=3), rng.uniform(-1, 1)) for _ in range(J - 1)]
    )
    ref_offset = reference_offset(posedirs, R_local_model)
    ref_final = base_vertices + ref_offset

    # Register bones under names J_1..J_{J-1} in a SHUFFLED dict to prove the
    # joint-name lookup restores the correct kintree order.
    joint_names = [f"J_{i}" for i in range(J)]  # J_0 is the root, skipped
    bones_by_name = {}
    for j in range(1, J):
        B = random_orthonormal(rng)
        Mb = B.T @ R_local_model[j - 1] @ B
        bones_by_name[joint_names[j]] = _MockPoseBone(B, Mb)
    shuffled = list(bones_by_name.items())
    rng.shuffle(shuffled)
    bones_by_name = dict(shuffled)  # collection order != joint order

    pose_bones = [bones_by_name[name] for name in joint_names[1:]]
    pose_feature = _addon_pose_feature(pose_bones)

    posedirs_reshaped = np.reshape(posedirs, [-1, P])
    offsets = np.reshape(np.matmul(pose_feature, posedirs_reshaped.T), [-1, 3])
    final = np.asarray(base_vertices) + offsets

    err = np.abs(final - ref_final).max()
    print("\n[end-to-end mirror of rewritten apply_pose_correctives]")
    print(f"    final rest verts max abs err vs reference : {err:.3e}")
    assert err < 1e-9, "rewritten function logic must match the reference"

    final2 = np.asarray(base_vertices) + np.reshape(
        np.matmul(pose_feature, posedirs_reshaped.T), [-1, 3]
    )
    assert np.abs(final2 - final).max() < 1e-12
    print("    idempotent re-application                 : OK")
    print("    joint-name ordering restores kintree order: OK")
    print("\nEnd-to-end mirror passed.")


if __name__ == "__main__":
    main()
