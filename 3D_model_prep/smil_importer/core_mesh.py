"""Leaf geometry and export utilities.

scipy is an optional dependency, imported lazily so the add-on can
register even when scipy is absent. Operators that need it are gated on
dependencies.dependencies_installed().
"""

import bpy
import numpy as np
from mathutils import Vector

# Bone-local -> model-frame limit remap (issue #56). Kept in a bpy-free module so
# it can be unit tested without Blender (tests/test_axis_remap.py). Aliased to the
# previous private names for backward compatibility with existing importers
# (e.g. diagnostics/probe_joint_limits_axis_remap.py).
from .axis_remap import (
    rot3 as _rot3,
    remap_bounds_to_model_frame as _remap_bounds_to_model_frame,
)

try:
    from scipy.spatial import KDTree
except ImportError:  # pragma: no cover
    KDTree = None


def ensure_mesh(func):
    def wrapper(obj, *args, **kwargs):
        if not obj or obj.type != "MESH":
            raise TypeError("The selected object is not a mesh.")
        return func(obj, *args, **kwargs)

    return wrapper


def ensure_armature(func):
    def wrapper(obj, *args, **kwargs):
        if not obj or obj.type != "ARMATURE":
            raise TypeError("The selected object is not an armature.")
        return func(obj, *args, **kwargs)

    return wrapper


def mesh_to_numpy(obj):
    mesh = obj.data
    vertices = np.array([vert.co for vert in mesh.vertices], dtype=np.float32)
    faces = np.array(
        [poly.vertices for poly in mesh.polygons if len(poly.vertices) == 3],
        dtype=np.int32,
    )
    return vertices, faces


@ensure_mesh
def triangulate_mesh(obj):
    # SMPL / SMIL models assume tris-only topology
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")


@ensure_mesh
def export_vertices_to_npy(obj, filepath):
    vertices, _ = mesh_to_numpy(obj)
    np.save(filepath, vertices)
    return filepath, vertices


@ensure_mesh
def export_faces_to_npy(obj, filepath):
    _, faces = mesh_to_numpy(obj)
    np.save(filepath, faces)
    return filepath, faces


@ensure_mesh
def export_mesh_to_obj(obj, filepath):
    vertices, faces = mesh_to_numpy(obj)
    with open(filepath, "w") as file:
        for vert in vertices:
            file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return filepath


@ensure_mesh
def export_vertex_groups_to_npy(obj, filepath, clean_weights=False):
    """
    Exports the vertex group weights of an object to a .npy file.

    Parameters:
    obj (bpy.types.Object): The object containing the vertex groups.
    filepath (str): The path to the .npy file where the weights will be saved.
    clean_weights (bool): If True, clean and normalize vertex weights before exporting.

    Returns:
    tuple: The filepath and the weights array.
    """
    # Ensure we're working on the correct object
    bpy.context.view_layer.objects.active = obj

    # Clean and normalize weights if requested
    if clean_weights:
        # Switch to edit mode
        bpy.ops.object.mode_set(mode="EDIT")
        # Select all vertices
        bpy.ops.mesh.select_all(action="SELECT")
        # Clean weights
        bpy.ops.object.vertex_group_clean(group_select_mode="ALL", limit=0.001)
        # Normalize weights to sum to 1.0
        bpy.ops.object.vertex_group_normalize_all(lock_active=False)

        """
        NOTE: When using boundary weights to inform the joint regressor, smoothing the transition between weight groups
        may be necessary!
        """
        # Limit total number of weights per vertex
        # Originally, this was set to 1 but then we could not use bounadries between adjacent bones
        # to inform the joint regressor.
        bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=2)
        # Return to object mode
        bpy.ops.object.mode_set(mode="OBJECT")

    # Get the mesh and armature
    mesh = obj.data
    armature = obj.find_armature()
    bones = armature.data.bones if armature else []

    # Initialize the weights array
    num_vertices = len(mesh.vertices)
    num_bones = len(bones)
    weights = np.zeros((num_vertices, num_bones), dtype=np.float32)

    # Create a dictionary mapping bone names to their indices
    bone_index_map = {bone.name: idx for idx, bone in enumerate(bones)}

    # Populate the weights array
    for vertex in mesh.vertices:
        for group in vertex.groups:
            group_name = obj.vertex_groups[group.group].name
            if group_name in bone_index_map:
                bone_idx = bone_index_map[group_name]
                weights[vertex.index, bone_idx] = group.weight

    # Save the weights array to a .npy file
    np.save(filepath, weights)

    return filepath, weights


@ensure_armature
def export_joint_locations_to_npy(armature_obj, filepath):
    # get and export bone names and locations, based on bone head locations (so rotation origin, not the tail)
    joints = armature_obj.data.bones
    joint_locations = np.array([bone.head_local for bone in joints], dtype=np.float32)
    joint_names = [bone.name for bone in joints]
    np.save(filepath, joint_locations)
    return filepath, joint_locations, joint_names


@ensure_armature
def export_joint_hierarchy_to_npy(armature_obj, filepath):
    # get the bone hierarchy from armature relationships and get them into the format required for the SMPL kintree_table
    joints = armature_obj.data.bones
    hierarchy = [[-1, 0]]
    for bone in joints:
        if bone.parent:
            parent_index = joints.find(bone.parent.name)
            child_index = joints.find(bone.name)
            hierarchy.append([parent_index, child_index])
    hierarchy = np.array(hierarchy, dtype=np.int32).T
    np.save(filepath, hierarchy)
    return filepath, hierarchy


@ensure_armature
def export_joint_limits_to_npy(armature_obj, filepath, default_range=np.pi):
    """Export per-joint rotation limits as a ``(J, 3, 2)`` array (issue #56).

    Bones are iterated in ``armature.data.bones`` order - the SAME order used by
    :func:`export_joint_locations_to_npy` and :func:`export_joint_hierarchy_to_npy`,
    which is the order stored in ``J_names``/``J``/``kintree_table``. The returned
    array therefore lines up index-for-index with those keys.

    Layout::

        limits[j, axis, 0] = min angle (radians) for joint j, axis in {x, y, z}
        limits[j, axis, 1] = max angle (radians)

    For each bone the per-axis limits are read, in priority order, from:

    1. a ``LIMIT_ROTATION`` pose-bone constraint (``use_limit_x/y/z`` with
       ``min_x``/``max_x`` etc.), or
    2. the bone's IK rotation limits and locks (``lock_ik_x`` -> pinned to 0,
       ``use_ik_limit_x`` -> ``ik_min_x``/``ik_max_x``).

    Axes with no explicit limit fall back to ``[-default_range, +default_range]``
    (wide open). The root bone (index 0) is fixed at ``[0, 0]`` on every axis,
    matching the fitter's ``LimitPrior`` which ignores the root joint.

    FRAME: limits are authored in each bone's *local* frame (that is what a Limit
    Rotation constraint bounds), but the fitter's ``LimitPrior`` compares the
    pose's *model-frame* axis-angle components against them. When a bone's rest
    orientation is tilted relative to the model, a bound authored on bone-local Y
    would otherwise land on the wrong model axis. Each bone's bounds are therefore
    rotated into the model frame via ``B = rot3(bone.matrix_local)`` (see
    :func:`_remap_bounds_to_model_frame`). For the common "clean axis" case where
    ``B`` is a signed permutation this is exact; for a genuinely rotated ``B`` no
    per-axis box is exact, so the bounds are kept verbatim and a warning is emitted
    (bounded issue-#56 caveat). Values remain per-axis bounds in the axis-angle
    space of ``joint_rotations`` - the same convention as the legacy ``Ranges``.

    Returns ``(filepath, limits)`` where ``limits`` is a float32 ``(J, 3, 2)`` array.
    """
    bones = armature_obj.data.bones
    pose_bones = armature_obj.pose.bones
    num_joints = len(bones)
    limits = np.empty((num_joints, 3, 2), dtype=np.float32)

    for i, bone in enumerate(bones):
        lo = [-float(default_range)] * 3
        hi = [float(default_range)] * 3

        pbone = pose_bones.get(bone.name)
        if pbone is not None:
            limit_con = next((c for c in pbone.constraints if c.type == "LIMIT_ROTATION"), None)
            if limit_con is not None:
                if limit_con.use_limit_x:
                    lo[0], hi[0] = limit_con.min_x, limit_con.max_x
                if limit_con.use_limit_y:
                    lo[1], hi[1] = limit_con.min_y, limit_con.max_y
                if limit_con.use_limit_z:
                    lo[2], hi[2] = limit_con.min_z, limit_con.max_z
            else:
                # Fall back to IK rotation limits / locks.
                if pbone.lock_ik_x:
                    lo[0] = hi[0] = 0.0
                elif pbone.use_ik_limit_x:
                    lo[0], hi[0] = pbone.ik_min_x, pbone.ik_max_x
                if pbone.lock_ik_y:
                    lo[1] = hi[1] = 0.0
                elif pbone.use_ik_limit_y:
                    lo[1], hi[1] = pbone.ik_min_y, pbone.ik_max_y
                if pbone.lock_ik_z:
                    lo[2] = hi[2] = 0.0
                elif pbone.use_ik_limit_z:
                    lo[2], hi[2] = pbone.ik_min_z, pbone.ik_max_z

        # Limits are authored in the bone-local frame; rotate them into the
        # model frame so they line up with the axis-angle components the fitter
        # compares against (issue #56). B's columns are the bone-local axes in
        # model coordinates; a signed-permutation B just permutes/flips axes.
        B = _rot3(bone.matrix_local)
        lo, hi = _remap_bounds_to_model_frame(B, lo, hi, bone.name)

        limits[i, :, 0] = lo
        limits[i, :, 1] = hi

    # Root joint (index 0) is fixed; the fitter drops it via LimitPrior anyway.
    if num_joints > 0:
        limits[0] = 0.0

    # Guarantee min <= max even if a user authored an inverted constraint.
    lo_col = limits[..., 0].copy()
    hi_col = limits[..., 1].copy()
    swapped = lo_col > hi_col
    limits[..., 0] = np.where(swapped, hi_col, lo_col)
    limits[..., 1] = np.where(swapped, lo_col, hi_col)

    np.save(filepath, limits)
    return filepath, limits


@ensure_mesh
def export_y_axis_vertices_to_npy(obj, filepath):
    # retruns all vertex indices that lie on the y-axis (within some tolerance) for symmetry axis
    mesh = obj.data
    y_axis_vertices = np.array(
        [i for i, vert in enumerate(mesh.vertices) if np.isclose(vert.co.y, 0.0, atol=1e-3)],
        dtype=int,
    )
    np.save(filepath, y_axis_vertices)
    return filepath, y_axis_vertices


def find_nearest_neighbors(vertices, joint_locations, n):
    """
    Find the n nearest vertices to each joint location and calculate their influence
    (here referred to as weights) based on inverse distance.

    This function is used to compute a joint regressor matrix by finding the closest
    vertices to each joint and assigning weights based on inverse distance. The weights
    are normalized so they sum to 1 for each joint, creating a smooth influence region
    around each joint location.

    Args:
        vertices (np.ndarray): Array of vertex positions with shape (num_vertices, 3)
        joint_locations (np.ndarray): Array of joint positions with shape (num_joints, 3)
        n (int): Number of nearest vertices to consider for each joint

    Returns:
        tuple: (nearest_indices, nearest_weights)
            - nearest_indices (np.ndarray): Indices of n nearest vertices for each joint,
              shape (num_joints, n)
            - nearest_weights (np.ndarray): Normalized weights for each nearest vertex,
              shape (num_joints, n), where weights sum to 1 for each joint
    """
    nearest_indices = np.zeros((len(joint_locations), n), dtype=np.int32)
    nearest_weights = np.zeros((len(joint_locations), n), dtype=np.float32)
    for i, joint_loc in enumerate(joint_locations):
        distances = np.linalg.norm(vertices - joint_loc, axis=1)
        # get indices of n nearest vertices (argpartition is also fast as not the whole array is sorted here)
        nearest_indices[i] = np.argpartition(distances, n)[:n]
        # get the distances (slice array) of the n nearest vertices
        nearest_distances = distances[nearest_indices[i]]
        # the weight is the inverse of the distance
        weights = 1.0 / nearest_distances
        # normalize the weights so they sum to 1
        weights /= weights.sum()
        nearest_weights[i] = weights
    return nearest_indices, nearest_weights


def J_regressor_from_boundary_weights(
    vertices,
    joint_locations,
    n,
    kintree_table,
    vertex_weights,
    nn_for_leaf_bones=True,
    debug=False,
):
    """
    Find the weights of the vertices that are associated with both the current joint and the parent joint.
    As in the find_nearest_neighbors function, we use the inverse of the distance to calculate the influence of all vertices
    meeting the boundary weights criteria and normalise their influence.
    This implementation should then effectively use the ring of the vertices surrounding the joint
    to inform its placement. Using the inverese of the distance also ensures only positive weights are used.

    Args:
        vertices (np.ndarray): Array of vertex positions with shape (num_vertices, 3)
        joint_locations (np.ndarray): Array of joint positions with shape (num_joints, 3)
        n (int): Number of nearest vertices required to meet the boundary weights criteria, otherwise default to using the nearest_neighbors function for THAT joint
        kintree_table (np.ndarray): Array of shape (2, num_joints) containing the parent-child relationships between joints
        vertex_weights (np.ndarray): Array of shape (num_vertices, num_joints) containing the weights of the nearest vertices for each joint

    Returns:
        J_regressor (np.ndarray): (j,v) matrix, containing the weights of each vertex contributing to the location of each joint.
    """
    J_regressor = np.zeros((len(joint_locations), len(vertices)), dtype=np.float32)
    if debug:
        print("vertex_weights.shape: ", vertex_weights.shape)
        print("kintree_table.shape: ", kintree_table.shape)
        print("joint_locations.shape: ", joint_locations.shape)
        print("vertices.shape: ", vertices.shape)
        print("n: ", n)
        print("kintree_table: ", kintree_table)

    # compute the nearest neighbors so we can default to using them, when conditions are not met for bounadry weights
    nearest_indices, nearest_weights = find_nearest_neighbors(vertices, joint_locations, n)
    # Small epsilon to avoid division by zero
    epsilon = 1e-8

    for i in range(len(joint_locations)):
        # Find parent joint index
        parent_indices = np.where(kintree_table[1, :] == i)[0]

        # Extract parent index (should be only one)
        parent_index = kintree_table[0, parent_indices[0]]
        if debug:
            print(f"Joint {i}: parent_index = {parent_index}")

        joint_index = kintree_table[1, i]
        print(f"Joint {i}: joint_index = {joint_index}")
        # Check if this joint has children
        child_indices = np.where(kintree_table[0, :] == i)[0]
        has_children = len(child_indices) > 0

        # If nn_for_leaf_bones is True and this joint has no children, use nearest neighbor approach
        if nn_for_leaf_bones and not has_children:
            if debug:
                print(f"Joint {i}: Leaf joint with no children, using nearest neighbor approach")
            J_regressor[i, nearest_indices[i]] = nearest_weights[i]
            continue

        # Check if this is a root joint
        if parent_index == -1:
            if debug:
                print(f"Joint {i}: Root joint, using nearest neighbor approach")
            J_regressor[i, nearest_indices[i]] = nearest_weights[i]
            continue

        # Create boolean masks for parent and child weights
        parent_mask = vertex_weights[:, parent_index] > 0
        child_mask = vertex_weights[:, i] > 0

        # Create boundary mask where both parent and child have non-zero weights
        boundary_mask = parent_mask & child_mask

        # Count boundary vertices
        num_boundary_vertices = np.sum(boundary_mask)
        if debug:
            print(f"Joint {i}: {num_boundary_vertices} boundary vertices found")

        if num_boundary_vertices < n:
            if debug:
                print(
                    f"Joint {i}: Insufficient boundary vertices ({num_boundary_vertices} < {n}), using nearest neighbor approach"
                )
            J_regressor[i, nearest_indices[i]] = nearest_weights[i]
        else:
            if debug:
                print(f"Joint {i}: Using boundary weighting approach")
            # Calculate distances to all vertices
            distances = np.linalg.norm(vertices - joint_locations[i], axis=1)

            # Apply boundary mask to distances (non-boundary vertices become 0)
            boundary_distances = distances * boundary_mask.astype(np.float32)

            # Calculate inverse weights with epsilon protection
            inverse_weights = 1.0 / (boundary_distances + epsilon)

            # Set weights for non-boundary vertices to 0 (where boundary_distances was 0)
            inverse_weights[boundary_distances == 0] = 0

            # Normalize weights
            weight_sum = np.sum(inverse_weights)
            if weight_sum > 0:
                normalized_weights = inverse_weights / weight_sum
            else:
                if debug:
                    print(f"Joint {i}: Warning - all boundary weights are zero, using nearest neighbor approach")
                J_regressor[i, nearest_indices[i]] = nearest_weights[i]
                continue

            # Assign to J_regressor
            J_regressor[i, :] = normalized_weights

    return J_regressor


def check_J_regressor_alignment(J_regressor, joints, vertices, joint_names=None):
    """
    This function computes the discrepancy between the user-defined joint locations (joints)
    and the regressed joint locations from the mesh vertices and J_regressor

    Args:
        J_regressor (np.ndarray): (j,v) matrix, containing the weights of each vertex contributing to the location of each joint.
                                 This is just a weighted linear combination of vertex positions.
        joints (np.ndarray): (j,3) matrix, containing the x,y,z coordinates of each joint
        vertices (np.ndarray): (v,3) matrix, containing the x,y,z coordinates of each mesh vertex
        joint_names (list, optional): List of joint names for descriptive output

    Returns:
        tuple: (regressed_joints, discrepancies, mean_discrepancy)
            - regressed_joints (np.ndarray): The regressed joint positions using J_regressor
            - discrepancies (np.ndarray): Euclidean distances between original and regressed joints
            - mean_discrepancy (float): Mean discrepancy across all joints
    """
    # Compute regressed joint positions: J_regressor @ vertices
    # J_regressor shape: (j, v), vertices shape: (v, 3)
    # Result shape: (j, 3)
    regressed_joints = np.matmul(J_regressor, vertices)

    # Compute discrepancies (Euclidean distances)
    discrepancies = np.linalg.norm(joints - regressed_joints, axis=1)

    # Calculate model size for relative discrepancy reporting
    # Get the bounding box of the model
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    model_size = max_coords - min_coords
    longest_axis = np.max(model_size)

    # Compute relative discrepancies (as percentage of longest model axis)
    relative_discrepancies = discrepancies / longest_axis * 100

    # Compute mean discrepancy
    mean_discrepancy = np.mean(discrepancies)
    mean_relative_discrepancy = np.mean(relative_discrepancies)

    # Print detailed information
    print("\nJ_regressor alignment check:")
    print(f"  Original joints shape: {joints.shape}")
    print(f"  Regressed joints shape: {regressed_joints.shape}")
    print(f"  Model size: {model_size}")
    print(f"  Longest model axis: {longest_axis:.6f}")
    print(f"  Mean absolute discrepancy: {mean_discrepancy:.6f}")
    print(f"  Mean relative discrepancy: {mean_relative_discrepancy:.3f}% of model size")
    print(f"  Max relative discrepancy: {np.max(relative_discrepancies):.3f}% of model size")
    print(f"  Min relative discrepancy: {np.min(relative_discrepancies):.3f}% of model size")

    # Print individual joint discrepancies
    for i in range(len(joints)):
        joint_name = joint_names[i] if joint_names and i < len(joint_names) else f"Joint_{i}"
        print(f"  {joint_name}: absolute = {discrepancies[i]:.6f}, relative = {relative_discrepancies[i]:.3f}%")

    return regressed_joints, discrepancies, mean_discrepancy


@ensure_mesh
# @ensure_armature (careful, the mesh is the active object!)
def export_J_regressor_to_npy(
    mesh_obj,
    armature_obj,
    n,
    filepath=None,
    influence_type="inverse_distance",
    weights=None,
    kintree_table=None,
    export_as_csv=True,
):
    """
    Calculate or export the joint regressor matrix.

    Args:
    - mesh_obj: The mesh object
    - armature_obj: The armature object
    - n: Number of nearest vertices to consider for each joint
    - filepath: Optional path to save the regressor matrix. If None, only returns the matrix

    Returns:
    - tuple: (filepath if provided else None, J_regressor matrix)
    """
    vertices, _ = mesh_to_numpy(mesh_obj)
    joints = armature_obj.data.bones
    joint_locations = np.array([bone.head_local for bone in joints], dtype=np.float32)
    if influence_type == "inverse_distance" or influence_type is None:
        nearest_indices, nearest_weights = find_nearest_neighbors(vertices, joint_locations, n)
        J_regressor = np.zeros((len(joints), len(vertices)), dtype=np.float32)

        for i in range(len(joints)):
            J_regressor[i, nearest_indices[i]] = nearest_weights[i]

    elif influence_type == "boundary_weights":
        # Check if required parameters are available
        if kintree_table is None or weights is None:
            print("Warning: boundary_weights method requires kintree_table and weights parameters.")
            print("Falling back to inverse_distance method.")
            # Fall back to inverse_distance method
            nearest_indices, nearest_weights = find_nearest_neighbors(vertices, joint_locations, n)
            J_regressor = np.zeros((len(joints), len(vertices)), dtype=np.float32)
            for i in range(len(joints)):
                J_regressor[i, nearest_indices[i]] = nearest_weights[i]
        else:
            J_regressor = J_regressor_from_boundary_weights(vertices, joint_locations, n, kintree_table, weights)
    else:
        J_regressor = np.zeros((len(joints), len(vertices)), dtype=np.float32)
        raise ValueError(f"Invalid influence type: {influence_type}")

    # Check alignment between original joints and regressed joints
    joint_names = [bone.name for bone in joints]
    check_J_regressor_alignment(J_regressor, joint_locations, vertices, joint_names)

    if filepath:
        np.save(filepath, J_regressor)

    if export_as_csv and filepath:
        # Write the debug CSV next to the .npy (blend-relative, writable) instead of a
        # bare "test_J_reg.csv" that lands in Blender's read-only CWD. Never let a debug
        # dump crash the export.
        try:
            np.savetxt(filepath.replace(".npy", ".csv"), J_regressor, delimiter=",", fmt="%1.8f")
        except Exception as e:
            print(f"Warning: could not write J_regressor debug CSV ({e})")

    return J_regressor


@ensure_mesh
def export_posedirs(mesh_obj, start_frame, stop_frame, filepath):
    bpy.context.view_layer.objects.active = mesh_obj
    num_frames = stop_frame - start_frame + 1
    num_vertices = len(mesh_obj.data.vertices)
    posedirs = np.zeros((num_vertices, 3, num_frames), dtype=np.float32)

    for frame in range(start_frame, stop_frame + 1):
        bpy.context.scene.frame_set(frame)
        for i, vert in enumerate(mesh_obj.data.vertices):
            posedirs[i, :, frame - start_frame] = vert.co

    np.save(filepath, posedirs)
    return filepath, posedirs


def recalculate_joint_positions(vertex_positions, J_regressor):
    """
    Recalculate the positions of joints based on vertex positions and joint regressor weights.

    Args:
    - vertex_positions (np.ndarray): Array of vertex positions (N x 3)
    - J_regressor (np.ndarray): (normalised) joint regressor matrix (J x N)

    Returns:
    - joint_positions (np.ndarray): Updated joint positions (J x 3)
    """

    j, n = J_regressor.shape
    assert vertex_positions.shape[0] == n, "Number of vertices in vertex positions and weights must match."

    # Calculate joint positions using matrix multiplication: J_regressor @ vertex_positions
    # J_regressor shape: (j, n), vertex_positions shape: (n, 3)
    # Result shape: (j, 3)
    joint_positions = np.matmul(J_regressor, vertex_positions)

    return joint_positions


def apply_updated_joint_positions(obj, pkl_data):
    """
    Apply recalculated joint positions to the armature.

    Args:
    - obj (bpy.types.Object): The mesh object with the updated mean shape.
    - pkl_data (dict): Dictionary containing the joint weights information from the .pkl file.
    """
    # Get current vertex positions
    vertex_positions = np.array([np.array(v.co) for v in obj.data.vertices])

    # Calculate new joint positions
    joint_positions = recalculate_joint_positions(
        vertex_positions=vertex_positions, J_regressor=pkl_data["J_regressor"]
    )

    # Update the armature with the new joint positions
    armature = obj.find_armature()
    if not armature:
        print("No armature found for the selected mesh.")
        return

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")

    for i, bone in enumerate(armature.data.edit_bones):
        bone.head = joint_positions[i]
        # the bone tails all point upwards and bones are of equal length
        bone.tail = joint_positions[i] + [0, 0, 0.1]

    bpy.ops.object.mode_set(mode="OBJECT")
    print("Joint positions recalculated and updated.")


def compute_symmetric_pairs(vertices, axis="y", tolerance=0.01):
    """
    Compute symmetric pairs of vertices based on their coordinates and the specified symmetry axis.
    Allow for a specified percentage deviation (tolerance) from the exact mirrored position using KDTree.
    """
    sym_pairs = []
    sym_axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    tolerance_value = np.max(np.abs(vertices)) * tolerance

    # Reflect vertices along the symmetry axis
    reflected_vertices = vertices.copy()
    reflected_vertices[:, sym_axis_idx] *= -1

    # Build KDTree for the reflected vertices
    tree = KDTree(reflected_vertices)

    # Find symmetric pairs within the tolerance
    for idx, vertex in enumerate(vertices):
        dist, idx_sym = tree.query(vertex, distance_upper_bound=tolerance_value)
        if dist < tolerance_value:
            sym_pairs.append((idx, idx_sym))

    return np.array(sym_pairs)


def rebuild_symmetry_array(vertices_on_symmetry_axis, all_vertices, axis="y", tolerance=0.001):
    # Initialize the symmetry array
    symIdx = np.arange(len(all_vertices))

    # Set the indices for vertices on the symmetry axis to point to themselves
    for idx in vertices_on_symmetry_axis:
        symIdx[idx] = idx

    # Compute symmetrical vertex pairs
    symmetrical_vertex_pairs = compute_symmetric_pairs(all_vertices, axis, tolerance)

    # Set the indices for symmetrical vertex pairs
    for pair in symmetrical_vertex_pairs:
        symIdx[pair[0]] = pair[1]
        symIdx[pair[1]] = pair[0]

    return symIdx


def make_symmetrical(obj, pkl_data, center_tolerance=0.005):
    """
    Enforces the symmetry of the original model by updating the position of all vertices lying on the
    symmetry axis, finding corresponding vertices and mirroring their positions either left or right
    """

    print("Enforcing symmetry...")

    I = pkl_data["sym_verts"]
    v = pkl_data["v_template"]

    v = v - np.mean(v, axis=0)
    y = np.mean(v[I, 1])
    v[:, 1] = v[:, 1] - y
    v[I, 1] = 0

    left = v[:, 1] <= -center_tolerance
    right = v[:, 1] >= center_tolerance
    center = ~(left | right)

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    try:
        assert len(left_inds) == len(right_inds)
        print(len(left_inds), len(right_inds), len(center_inds))
    except AssertionError:
        print(
            f"Error enforcing symmetry: Unequal number of vertices on left ({len(left_inds)})",
            f"and right ({len(right_inds)}) sides. This may indicate an asymmetric mesh or",
            "incorrect symmetry axis.",
        )

    symIdx = rebuild_symmetry_array(vertices_on_symmetry_axis=I, all_vertices=v, axis="y", tolerance=0.001)

    # Check if the object has shape keys
    if obj.data.shape_keys:
        shape_keys = obj.data.shape_keys.key_blocks
    else:
        shape_keys = None

    for i, vertex in enumerate(obj.data.vertices):
        # enforce mesh centering
        if center[i]:
            new_position = Vector([vertex.co.x, 0, vertex.co.z])
        # mirror remaining vertices
        elif left[i]:
            corresponding_vertex = obj.data.vertices[symIdx[i]]
            new_position = Vector(
                [
                    corresponding_vertex.co.x,
                    -corresponding_vertex.co.y,
                    corresponding_vertex.co.z,
                ]
            )
        else:
            new_position = vertex.co

        # Update the main vertex position
        vertex.co = new_position

        # Also update all shape keys' vertex positions if they exist
        if shape_keys:
            for shape_key in shape_keys:
                shape_vertex = shape_keys[shape_key.name].data[i]
                if center[i]:
                    shape_vertex.co = Vector([shape_vertex.co.x, 0, shape_vertex.co.z])
                elif left[i]:
                    corresponding_shape_vertex = shape_keys[shape_key.name].data[symIdx[i]]
                    shape_vertex.co = Vector(
                        [
                            corresponding_shape_vertex.co.x,
                            -corresponding_shape_vertex.co.y,
                            corresponding_shape_vertex.co.z,
                        ]
                    )
                else:
                    shape_vertex.co = shape_vertex.co

        # Update the mesh to reflect the changes
        obj.data.update()


def cleanup_mesh(obj, center_tolerance=0.005):
    """
    Cleans up the mesh by merging vertices close to the symmetry axis
    and recalculating normals. Applies the same cleanup to all shapekeys.
    Removes all interior faces.
    """
    # Ensure we're working on the correct object
    bpy.context.view_layer.objects.active = obj

    # Apply the cleanup for the base mesh
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")

    # Select vertices with y coordinate close to 0 in the base mesh
    bpy.ops.object.mode_set(mode="OBJECT")
    for vertex in obj.data.vertices:
        if abs(vertex.co.y) < center_tolerance:
            vertex.select = True

    # Merge selected vertices by distance in the base mesh
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.remove_doubles(threshold=center_tolerance)

    # Recalculate mesh normals for the base mesh
    bpy.ops.mesh.normals_make_consistent(inside=False)

    # Ensure that the base mesh cleanup is applied before moving to shapekeys
    bpy.ops.object.mode_set(mode="OBJECT")

    # Remove interior faces
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.fill_holes(sides=0)
    bpy.ops.mesh.select_interior_faces()
    bpy.ops.mesh.delete(type="FACE")

    # Return to object mode
    bpy.ops.object.mode_set(mode="OBJECT")
