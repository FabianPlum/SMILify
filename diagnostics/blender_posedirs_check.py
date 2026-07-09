"""In-Blender verification for the pose-corrective (posedirs) fix -- issue #24.

HOW TO RUN
----------
1. Install/enable the ``smil_importer`` add-on in Blender (Edit > Preferences >
   Add-ons > Install..., pick the zip from ``python 3D_model_prep/build_addon.py``),
   OR make sure ``smil_importer`` is importable from Blender's Python.
2. Open Blender's *Scripting* workspace, load this file in the Text Editor, press
   *Run Script*. Read results in the system console (Window > Toggle System Console
   on Windows) or the Blender console.

WHAT IT CHECKS
--------------
It builds a small synthetic model with the add-on's OWN ``create_mesh_from_pkl`` +
``create_armature_and_weights`` (so every bone gets the real +Z rest orientation
that made ``matrix_basis`` live in the wrong frame). It then:

  * sets each non-root bone to a KNOWN model-frame rotation R_model via
    ``matrix_basis = B^-1 @ R_model @ B``  (B = bone rest matrix),
  * runs the fixed ``apply_pose_correctives``,
  * reads the resulting REST vertices back and subtracts v_template to recover the
    applied corrective offset,
  * compares against an independent NumPy reference computed straight from posedirs
    and the same R_model (the smal_torch formula).

PASS  = max abs error ~ 1e-6 or below (float rounding through mathutils).
It also re-runs to confirm idempotency.

If this passes, Blender's real matrix_local / matrix_basis / to_quaternion
conventions agree with the fix, and the add-on applies SMPL-style correctives
correctly. To test the actual SMPL *human* model instead of synthetic data,
replace the "SYNTHETIC MODEL" block with ``data = load_pkl_file(<path>)`` for
``basicModel_f_lbs_10_207_0_v1.0.0.pkl`` (needs the add-on's chumpy-aware loader),
convert arrays with ``np.array(...)``, and reuse the rest unchanged.
"""

import numpy as np
import bpy
from mathutils import Matrix

# --- locate the add-on's functions ---------------------------------------- #
try:
    from smil_importer.model_build import (
        apply_pose_correctives,
        create_mesh_from_pkl,
        create_armature_and_weights,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import smil_importer. Install/enable the add-on, or add its "
        "parent folder to sys.path before running this script."
    ) from exc


def rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def reference_offset(posedirs, R_local_model):
    """smal_torch formula: (V,3,P) posedirs, R_local_model list len J-1 (model frame)."""
    V, _, P = posedirs.shape
    posedirs_mat = np.reshape(posedirs, [-1, P]).T  # (P, V*3)
    pose_feature = np.concatenate([(R - np.eye(3)).flatten() for R in R_local_model])
    return np.reshape(pose_feature @ posedirs_mat, [V, 3])


def clean_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def build_synthetic_data(seed=0):
    rng = np.random.default_rng(seed)
    J = 4  # 1 root + 3 posed joints
    # joints along +X, so bones are non-trivially oriented (tail = head + +Z)
    Jloc = np.stack([np.array([i * 0.3, 0.0, 0.0]) for i in range(J)]).astype(np.float64)
    # a little cloud of vertices around the joints
    V_per = 6
    verts = []
    for j in range(J):
        verts.append(Jloc[j] + rng.normal(scale=0.05, size=(V_per, 3)))
    v_template = np.concatenate(verts, axis=0).astype(np.float64)
    V = v_template.shape[0]
    # trivial faces (triangles) just so a mesh can be built
    faces = [(i, (i + 1) % V, (i + 2) % V) for i in range(0, V - 2, 3)]
    # one-hot skin weights: each vertex bound to its joint
    weights = np.zeros((V, J), dtype=np.float64)
    for j in range(J):
        weights[j * V_per : (j + 1) * V_per, j] = 1.0
    # chain kintree 0->1->2->3
    kintree = np.array([[-1, 0, 1, 2], [0, 1, 2, 3]], dtype=np.int64)
    P = (J - 1) * 9
    posedirs = (rng.normal(size=(V, 3, P)) * 0.02).astype(np.float64)
    data = {
        "v_template": v_template,
        "f": faces,
        "J": Jloc,
        "weights": weights,
        "kintree_table": kintree,
        "posedirs": posedirs,
        "J_names": [f"J_{i}" for i in range(J)],
    }
    return data


def run():
    clean_scene()
    data = build_synthetic_data()
    J = len(data["J"])
    joint_names = data["J_names"]

    obj = create_mesh_from_pkl(data, base_name="TEST")
    armature = create_armature_and_weights(data, obj, base_name="TEST")
    assert armature is not None, "armature build failed"

    # Known model-frame rotations for joints 1..J-1
    rng = np.random.default_rng(1)
    R_model = [rodrigues(rng.normal(size=3), rng.uniform(-0.8, 0.8)) for _ in range(J - 1)]

    # Set each non-root bone's matrix_basis so its model-frame rotation == R_model.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    for k, name in enumerate(joint_names[1:]):
        pb = armature.pose.bones.get(name)
        B4 = pb.bone.matrix_local.to_3x3().to_4x4()  # rest orientation (bone-local -> model)
        Rm4 = Matrix(np.eye(4))
        for r in range(3):
            for c in range(3):
                Rm4[r][c] = float(R_model[k][r, c])
        pb.matrix_basis = B4.inverted() @ Rm4 @ B4  # inverse of B @ basis @ B^T
    bpy.ops.object.mode_set(mode="OBJECT")

    # Reference offset (independent of the add-on)
    ref = reference_offset(data["posedirs"], R_model)

    # Run the fixed add-on function
    apply_pose_correctives(obj, data["posedirs"], data["v_template"], joint_names)

    measured = np.array([np.array(v.co) for v in obj.data.vertices]) - data["v_template"]
    err = np.abs(measured - ref).max()
    print("\n================ posedirs in-Blender check ================")
    print(f"  vertices: {measured.shape[0]}   joints: {J}")
    print(f"  max abs error (add-on vs smal_torch reference): {err:.3e}")

    # Idempotency: run again, result must be identical
    apply_pose_correctives(obj, data["posedirs"], data["v_template"], joint_names)
    measured2 = np.array([np.array(v.co) for v in obj.data.vertices]) - data["v_template"]
    err_idem = np.abs(measured2 - measured).max()
    print(f"  idempotency (re-apply delta):                   {err_idem:.3e}")

    ok = err < 1e-6 and err_idem < 1e-9
    print("  RESULT:", "PASS ✅" if ok else "FAIL ❌")
    print("===========================================================\n")
    return ok


if __name__ == "__main__":
    run()
