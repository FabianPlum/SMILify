bl_info = {
    "name": "SMIL Model Importer",
    "author": "Fabian Plum",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Tool Shelf",
    "description": "Import, configure, and export SMPL / SMIL models",
    "category": "Import-Export",
}

import bpy
import numpy as np
import pickle
import os
from scipy.spatial import KDTree
from mathutils import Vector
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
import tempfile
import csv
import bmesh
import time


# TODO if you are very bored, implement package installation with subprocesses
"""
# WINDOWS
# to install required packages, in case sklearn is not found

import pip
import sys
pip.main(['install', 'scikit-learn', 'matplotlib' '--target', (sys.exec_prefix) + '\\lib\\site-packages'])
"""

"""
# UBUNTU
# to install required packages, in case sklearn is not found
# here, we need to run the following from the command line instead, as blender does not want to pip install things while running
# so, go to the python executable that was shipped with blender and make sure the --target is correct

./python3.11 -m pip install matplotlib scikit-learn --target /home/USER/Downloads/blender-4.2.0-linux-x64/4.2/python/lib/python3.11/site-packages
"""


# global, so the model is not re-loaded
pkl_data = None

"""
SMIL-ify
"""

# before we do anything else, let's add some code from smal_model/smal_torch.py so we can load the model
# specifically old models that still use chumpy


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles legacy SMAL model files containing chumpy arrays"""

    def __init__(self, file, encoding="latin1"):
        """Initialize with latin1 encoding to handle legacy pickle files"""
        super().__init__(file, encoding=encoding)

    def find_class(self, module, name):
        """Override class lookup to handle chumpy arrays"""
        if module == "chumpy.ch" and name == "Ch":
            return self.ChumpyWrapper
        return super().find_class(module, name)

    class ChumpyWrapper:
        """Wrapper class that mimics chumpy array behavior but stores only numpy arrays"""

        def __init__(self, *args, **kwargs):
            """Initialize with data from args or empty array"""
            self.data = np.array(args[0]) if args else np.array([])

        def __array__(self):
            """Allow numpy array conversion via np.array(instance)"""
            return self.data

        def __setstate__(self, state):
            """Handle unpickling of chumpy arrays in various formats"""
            if isinstance(state, dict):
                # Handle old chumpy format where data is stored in 'x' key
                self.data = np.array(state.get("x", []))
            else:
                # Handle both tuple/list format and direct data format
                self.data = np.array(
                    state[0] if isinstance(state, (tuple, list)) else state
                )
            return self

        @property
        def r(self):
            """Mimic chumpy's .r property which returns the underlying data"""
            return self.data


# Decorators for type checking
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


# Convert mesh to numpy arrays
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


@ensure_mesh
def export_y_axis_vertices_to_npy(obj, filepath):
    # retruns all vertex indices that lie on the y-axis (within some tolerance) for symmetry axis
    mesh = obj.data
    y_axis_vertices = np.array(
        [
            i
            for i, vert in enumerate(mesh.vertices)
            if np.isclose(vert.co.y, 0.0, atol=1e-3)
        ],
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
    nearest_indices, nearest_weights = find_nearest_neighbors(
        vertices, joint_locations, n
    )
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
                print(
                    f"Joint {i}: Leaf joint with no children, using nearest neighbor approach"
                )
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
                    print(
                        f"Joint {i}: Warning - all boundary weights are zero, using nearest neighbor approach"
                    )
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
    print(f"\nJ_regressor alignment check:")
    print(f"  Original joints shape: {joints.shape}")
    print(f"  Regressed joints shape: {regressed_joints.shape}")
    print(f"  Model size: {model_size}")
    print(f"  Longest model axis: {longest_axis:.6f}")
    print(f"  Mean absolute discrepancy: {mean_discrepancy:.6f}")
    print(
        f"  Mean relative discrepancy: {mean_relative_discrepancy:.3f}% of model size"
    )
    print(
        f"  Max relative discrepancy: {np.max(relative_discrepancies):.3f}% of model size"
    )
    print(
        f"  Min relative discrepancy: {np.min(relative_discrepancies):.3f}% of model size"
    )

    # Print individual joint discrepancies
    for i in range(len(joints)):
        joint_name = (
            joint_names[i] if joint_names and i < len(joint_names) else f"Joint_{i}"
        )
        print(
            f"  {joint_name}: absolute = {discrepancies[i]:.6f}, relative = {relative_discrepancies[i]:.3f}%"
        )

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
    if influence_type == "inverse_distance" or influence_type == None:
        nearest_indices, nearest_weights = find_nearest_neighbors(
            vertices, joint_locations, n
        )
        J_regressor = np.zeros((len(joints), len(vertices)), dtype=np.float32)

        for i in range(len(joints)):
            J_regressor[i, nearest_indices[i]] = nearest_weights[i]

    elif influence_type == "boundary_weights":
        J_regressor = J_regressor_from_boundary_weights(
            vertices, joint_locations, n, kintree_table, weights
        )
    else:
        J_regressor = np.zeros((len(joints), len(vertices)), dtype=np.float32)
        raise ValueError(f"Invalid influence type: {influence_type}")

    # Check alignment between original joints and regressed joints
    joint_names = [bone.name for bone in joints]
    check_J_regressor_alignment(J_regressor, joint_locations, vertices, joint_names)

    if filepath:
        np.save(filepath, J_regressor)

    if export_as_csv:
        #np.savetxt(filepath.replace(".npy", ".csv"), J_regressor, delimiter=",")
        np.savetxt("test_J_reg.csv", J_regressor, delimiter=",", fmt="%1.8f")

    return J_regressor


"""
This is currently not supported.
The posedir created here captures the mesh shape at every frame
This is however NOT how the posedir is used in the original implementaiton and thus disabled for now.

Normally, posedirs are used to to apply shape corrections, based on joint rotations.
These are learned from the model data.
"""


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


def load_pkl_file(filepath):
    try:
        with open(filepath, "rb") as f:
            print("\nReading in contents of SMPL file...")
            data = CustomUnpickler(f).load()
            data_de_chumpied = {
                k: np.array(v) if isinstance(v, CustomUnpickler.ChumpyWrapper) else v
                for k, v in data.items()
            }
            print("\nContents of loaded SMPL file:")
            for key in data_de_chumpied:
                print(key)
                try:
                    if type(data_de_chumpied[key]) is not str:
                        print(data_de_chumpied[key].shape)
                except:
                    print(len(data_de_chumpied[key]))
        print("Loaded .pkl file successfully.")
        return data_de_chumpied
    except Exception as e:
        print(f"Failed to load .pkl file: {e}")
        return None


def load_npz_file(filepath):
    try:
        print("\nReading in contents of fitted model file...")
        data = np.load(filepath, allow_pickle=True)
        print("\nContents of loaded .npz file:")
        for key in data:
            print(key)
            if isinstance(data[key], np.ndarray):
                print(data[key].shape)
        print("Loaded .npz file successfully.")
        return data
    except Exception as e:
        print(f"Failed to load .npz file: {e}")
        return None


def apply_pose_correctives(obj, posedirs, base_vertices):
    """
    Apply pose-dependent corrective shape keys based on current armature pose.

    Args:
    - obj (bpy.types.Object): The mesh object to apply corrections to
    - posedirs (numpy.ndarray): Array of shape (num_vertices, 3, num_joints * 9) containing pose-dependent deformations
    - base_vertices (numpy.ndarray): Array of shape (num_vertices, 3) containing base vertex positions
    """
    # Find the armature
    armature = obj.find_armature()
    if not armature:
        print("No armature found. Cannot apply pose corrections.")
        return

    # Ensure we're in pose mode to read pose data
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    # Get pose bones (excluding root)
    pose_bones = armature.pose.bones[1:]  # Skip root bone

    # Store current vertex positions (these are the skinned positions without correctives)
    if "base_skinned_positions" not in obj:
        # Get current deformed vertex positions (from regular skinning)
        world_matrix = np.array(obj.matrix_world)
        world_matrix_inv = np.array(obj.matrix_world.inverted())
        base_skinned_positions = np.array(
            [np.array(obj.matrix_world @ v.co) for v in obj.data.vertices]
        )
        # Store these positions for future resets
        obj["base_skinned_positions"] = base_skinned_positions.tobytes()
        obj["world_matrix"] = world_matrix.tobytes()
        obj["world_matrix_inv"] = world_matrix_inv.tobytes()

    # Prepare pose feature vector
    pose_feature = []
    for bone in pose_bones:
        # Get bone's current rotation matrix in local space and convert to numpy
        R = np.array(bone.matrix_basis.to_3x3())
        # Compute difference from identity
        R_diff = R - np.eye(3)
        # Flatten and add to pose feature vector
        pose_feature.extend(R_diff.flatten())

    pose_feature = np.array(pose_feature)
    print(f"Generated pose feature vector of length: {len(pose_feature)}")

    # Reshape posedirs if needed
    if len(posedirs.shape) == 3:
        num_vertices, _, num_pose_basis = posedirs.shape
        posedirs_reshaped = np.reshape(posedirs, [-1, num_pose_basis])
    else:
        posedirs_reshaped = posedirs

    print(f"Posedirs shape: {posedirs_reshaped.shape}")
    print(f"Pose feature shape: {pose_feature.shape}")

    # Calculate vertex offsets
    vertex_offsets = np.reshape(np.matmul(pose_feature, posedirs_reshaped.T), [-1, 3])

    # Switch to object mode to modify vertices
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = obj

    # Restore base skinned positions
    base_skinned_positions = np.frombuffer(obj["base_skinned_positions"]).reshape(-1, 3)
    world_matrix = np.frombuffer(obj["world_matrix"]).reshape(4, 4)
    world_matrix_inv = np.frombuffer(obj["world_matrix_inv"]).reshape(4, 4)

    # Apply offsets to vertices
    for idx, offset in enumerate(vertex_offsets):
        # Get the base skinned position
        skinned_pos = base_skinned_positions[idx]
        # Add corrective offset to the skinned position
        final_pos = skinned_pos + offset
        # Update vertex position (convert back to local space)
        local_pos = world_matrix_inv @ np.append(final_pos, 1.0)
        obj.data.vertices[idx].co = local_pos[:3]

        # Print debug info for first vertex
        if idx == 0:
            print(f"First vertex:")
            print(f"  Base position: {base_vertices[idx]}")
            print(f"  Base skinned position: {skinned_pos}")
            print(f"  Pose offset: {offset}")
            print(f"  Final position: {final_pos}")
            print(f"  Local position: {local_pos[:3]}")

    obj.data.update()
    print("Applied pose-dependent corrective shape keys")


def create_mesh_from_pkl(data):
    # read in the .pkl file with mesh data stored similar to obj files
    # (tris triplets and faces with vertex indices)
    if "v_template" not in data or "f" not in data:
        print("No 'verts' or 'faces' key found in the .pkl file.")
        return None

    verts = data["v_template"]
    faces = data["f"]

    mesh = bpy.data.meshes.new(name="SMPL_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name="SMPL_Object", object_data=mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return obj


def create_armature_and_weights(data, obj):
    """
    Create an armature based on the joint locations and assign weights to the mesh vertices.

    Args:
    - data (dict): Dictionary containing the contents of the .pkl file.
    - obj (bpy.types.Object): The newly created mesh object.
    """
    if "J" not in data or "weights" not in data or "kintree_table" not in data:
        print("No 'J', 'weights', or 'kintree_table' key found in the .pkl file.")
        return None

    joints = data["J"]
    # the default SMAL / SMPL models don't have J_names, so let's generate them if they are absent
    if "J_names" not in data:
        data["J_names"] = [f"J_{i}" for i in range(joints.shape[0])]
        print(data["J_names"])
    joint_names = data["J_names"]
    weights = data["weights"]
    kintree_table = data["kintree_table"]

    # Create armature
    bpy.ops.object.add(type="ARMATURE", enter_editmode=True)
    armature = bpy.context.object
    armature.name = "SMPL_Armature"
    armature.show_in_front = True

    # Add bones based on hierarchy
    bones = []
    for i, (parent_idx, child_idx, bone_name) in enumerate(
        zip(kintree_table[0], kintree_table[1], joint_names)
    ):
        bone = armature.data.edit_bones.new(bone_name)
        bone.head = joints[child_idx]
        bone.tail = joints[child_idx] + np.array([0, 0, 0.1])
        bones.append(bone)

        # in some cases when the parent_idx has been stored as -1 this causes an integer overflow
        # to avoid this leading to some weird errors, if the parent_idx is out of range, set it to -1 here.
        if parent_idx > len(joint_names):
            parent_idx = -1

        if parent_idx != -1:
            bone.parent = armature.data.edit_bones[joint_names[parent_idx]]

    bpy.ops.object.mode_set(mode="OBJECT")

    # Parent mesh to armature
    obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE")

    # Assign vertex weights
    for i, vertex_weights in enumerate(weights):
        for j, (weight, bone_name) in enumerate(zip(vertex_weights, joint_names)):
            if weight > 0:
                vertex_group = obj.vertex_groups.get(bone_name)
                if vertex_group is None:
                    vertex_group = obj.vertex_groups.new(name=bone_name)
                vertex_group.add([i], weight, "ADD")


def create_shapekeys(data, obj):
    """
    Create shapekeys from deformation vertices in the new mesh object.

    Args:
    - data (dict): Dictionary containing the contents of the .npz file.
    - obj (bpy.types.Object): The newly created mesh object.
    """
    if "verts" not in data or "labels" not in data:
        print("No 'verts' or 'labels' key found in the .npz file.")
        return

    deform_verts = data["verts"]
    target_shape_names = data["labels"]

    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis")

    for i, deform in enumerate(deform_verts):
        shape_key_name = target_shape_names[i]
        shape_key = obj.shape_key_add(name=shape_key_name)
        for vert_index, vert in enumerate(deform):
            shape_key.data[vert_index].co = vert

    # Sort shape keys alphabetically
    sort_shape_keys(obj)

    # Here, as the individual shapekeys are entirely independent of each other,
    # the covariance matrix is simply a [n, n] identity matrix
    num_shapes = deform_verts.shape[0]
    cov = np.eye(num_shapes)
    print(cov.shape)
    # Likewise, the mean_betas are 1/n for all shapes
    mean_betas = np.ones(num_shapes) / num_shapes

    print(f"Created {len(deform_verts)} shapekeys.")
    return cov, mean_betas


def apply_pca_and_create_shapekeys(
    scans,
    obj,
    num_components=10,
    overwrite_mesh=False,
    std_range=1,
    labels=None,
    output_dir=None,
):
    n, v, _ = scans.shape
    # Reshape the scans into (n, v*3)
    scans_reshaped = scans.reshape(n, v * 3)

    # Perform PCA
    pca = PCA(n_components=num_components)
    pca.fit(scans_reshaped)

    # Mean shape
    mean_shape = pca.mean_.reshape(v, 3)

    # get covariance matrix
    transformed_betas = pca.transform(scans_reshaped)
    COV = EmpiricalCovariance(assume_centered=False).fit(transformed_betas)
    cov_out = COV.covariance_
    mean_betas = COV.location_

    if overwrite_mesh:
        # Overwrite the mesh vertex coordinates with the mean shape
        for vert_index, vert in enumerate(mean_shape):
            obj.data.vertices[vert_index].co = vert
        # then add a basis shape key
        shape_key = obj.shape_key_add(name="Basis")
    else:
        # Add the mean shape as a shapekey
        if not obj.data.shape_keys:
            obj.shape_key_add(name="Basis")
        shape_key = obj.data.shape_keys.key_blocks["Basis"]
        for vert_index, vert in enumerate(mean_shape):
            shape_key.data[vert_index].co = vert

    # Principal components (reshape each component back to (v, 3))
    shapekeys = [component.reshape(v, 3) for component in pca.components_]

    # Standard deviations of the principal components
    std_devs = np.sqrt(pca.explained_variance_)

    # Add shapekeys as shape keys with min and max range
    for i, (shapekey, std_dev) in enumerate(zip(shapekeys, std_devs)):
        shape_key_name = f"PC_{i + 1}"
        shape_key = obj.shape_key_add(name=shape_key_name)

        # Calculate min and max range for the shape key
        min_range = -std_range * std_dev
        max_range = std_range * std_dev

        # Update the shape key vertex positions
        for j, vertex in enumerate(shapekey):
            shape_key.data[j].co = mean_shape[j] + vertex

        # Set min and max range for the shape key
        shape_key.slider_min = min_range
        shape_key.slider_max = max_range

    print(
        f"Created {num_components} PCA shapekeys with custom min and max ranges based on standard deviations."
    )
    # Optional: export XY (PC1, PC2) scatter data and PCA stats
    try:
        if output_dir is not None:
            if labels is None or len(labels) != scans.shape[0]:
                labels = [f"sample_{i}" for i in range(scans.shape[0])]
            # XY coordinates for first two PCs
            pc_xy_path = os.path.join(output_dir, "smil_shape_PC_xy.csv")
            with open(pc_xy_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["label", "PC1", "PC2"]) 
                for i, lab in enumerate(labels):
                    pc1 = transformed_betas[i, 0] if transformed_betas.shape[1] > 0 else 0.0
                    pc2 = transformed_betas[i, 1] if transformed_betas.shape[1] > 1 else 0.0
                    writer.writerow([lab, pc1, pc2])

            # PCA stats
            stats_path = os.path.join(output_dir, "smil_shape_PCA_stats.txt")
            with open(stats_path, "w") as f:
                f.write("PCA stats for shape-derived PCs\n")
                f.write(f"n_samples: {scans_reshaped.shape[0]}\n")
                f.write(f"n_features: {scans_reshaped.shape[1]}\n")
                f.write(f"n_components: {num_components}\n")
                f.write(f"explained_variance_ratio: {pca.explained_variance_ratio_.tolist()}\n")
                f.write(f"explained_variance: {pca.explained_variance_.tolist()}\n")
                f.write(f"singular_values: {pca.singular_values_.tolist()}\n")
                f.write(f"mean_l2_norm: {float(np.linalg.norm(pca.mean_))}\n")
                # Add per-shape PC weights (scores) needed to reproduce each input shape
                f.write("\npc_weights_per_shape (scores):\n")
                header = ",".join(["label"] + [f"PC{i+1}" for i in range(min(num_components, transformed_betas.shape[1]))])
                f.write(header + "\n")
                for i, lab in enumerate(labels):
                    weights = transformed_betas[i, : num_components]
                    weights_str = ",".join([f"{w}" for w in weights.tolist()])
                    f.write(f"{lab},{weights_str}\n")
            print(f"Shape PCA XY exported to {pc_xy_path}; stats to {stats_path}")
    except Exception as e:
        print(f"Failed exporting shape PCA XY/stats: {e}")
    return cov_out, mean_betas


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
    assert (
        vertex_positions.shape[0] == n
    ), "Number of vertices in vertex positions and weights must match."

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
    armature = bpy.data.objects.get("SMPL_Armature")
    if not armature:
        print("SMPL_Armature not found.")
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


def rebuild_symmetry_array(
    vertices_on_symmetry_axis, all_vertices, axis="y", tolerance=0.001
):
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
            f"incorrect symmetry axis.",
        )

    symIdx = rebuild_symmetry_array(
        vertices_on_symmetry_axis=I, all_vertices=v, axis="y", tolerance=0.001
    )

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
                    corresponding_shape_vertex = shape_keys[shape_key.name].data[
                        symIdx[i]
                    ]
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


def export_smpl_model(obj, export_path, pkl_data=None):
    """
    Export the updated model as a new SMPL file with the shapekeys stored in the model's shapedirs.

    Args:
    - obj (bpy.types.Object): The mesh object with the updated vertex locations and shapekeys.
    - pkl_data (dict): Dictionary containing the original SMPL data.
    - export_path (str): The file path where the new SMPL file will be saved.
    """

    if pkl_data is None:
        # create a new pkl data dictionary when new models are exported
        pkl_data = {
            "f": [],
            "J_regressor": [],
            "kintree_table": [],
            "J": [],
            "bs_style": "lbs",
            "weights": [],
            "posedirs": np.empty(
                0
            ),  # ignore for now as we currently don't have corrective shapekeys in our models
            "v_template": [],
            "shapedirs": [],
            "bs_type": "lrotmin",
            "sym_verts": [],
        }

        # new models most likely have weight paiting / assignment issues that need to be resolved
        # if your model looks weirdly spiky when loading into fitter_3d/optimise.py, more likely
        # than not, your weight painting is the culprit.
        # first, run "clean", than run "limit total" to one vertex group
        #
        # UPDATE: Ehhhh, idk,for some later tests I have found that using smoothing with limit set to 2
        # can actually help with getting clean weights and correct J-regressor results.
        # We'll monitor this and provide guidance once we release this addon.

        clean_weights = True
    else:
        clean_weights = False

    # Update "v_template" with the newly computed vertex locations of the mesh
    updated_vertices = np.array([np.array(v.co) for v in obj.data.vertices])
    pkl_data["v_template"] = updated_vertices
    print(pkl_data["v_template"].shape)

    # update all changed elements due to topoly changes
    # filepaths for temporary output files, used during debugging, this can be removed in the next big refactor
    faces_npy_path = bpy.path.abspath("//test_faces.npy")
    vertex_groups_npy_path = bpy.path.abspath("//test_vertex_groups.npy")
    joint_locations_npy_path = bpy.path.abspath("//test_joint_locations.npy")
    j_regressor_npy_path = bpy.path.abspath("//test_joint_regressor.npy")
    y_axis_vertices_npy_path = bpy.path.abspath("//test_y_axis_vertices.npy")
    joint_hierarchy_npy_path = bpy.path.abspath("//test_joint_hierarchy.npy")

    pkl_data["f"] = export_faces_to_npy(obj, faces_npy_path)[1]
    print(pkl_data["f"].shape)
    pkl_data["weights"] = export_vertex_groups_to_npy(
        obj, vertex_groups_npy_path, clean_weights=clean_weights
    )[1]
    print(pkl_data["weights"].shape)
    pkl_data["sym_verts"] = export_y_axis_vertices_to_npy(
        obj, y_axis_vertices_npy_path
    )[1]
    print(pkl_data["sym_verts"].shape)

    armature_obj = next(
        (obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None
    )
    if not armature_obj:
        print("No armature object found.")
        return

    print("Found armature object:", armature_obj.name)

    pkl_data["kintree_table"] = export_joint_hierarchy_to_npy(
        armature_obj, joint_hierarchy_npy_path
    )[1]
    pkl_data["J"], pkl_data["J_names"] = export_joint_locations_to_npy(
        armature_obj, joint_locations_npy_path
    )[1:]
    pkl_data["J_regressor"] = export_J_regressor_to_npy(
        obj,
        armature_obj,
        10,
        j_regressor_npy_path,
        weights=pkl_data["weights"],
        kintree_table=pkl_data["kintree_table"],
        influence_type="boundary_weights",
    )

    # Update "shapedirs" with the content of the shapekeys
    num_vertices = len(updated_vertices)
    try:
        num_shapekeys = (
            len(obj.data.shape_keys.key_blocks) - 1
        )  # Exclude the "Basis" shape key
        shapedirs = np.zeros(
            (num_vertices, 3, num_shapekeys)
        )  # add 1 for one base shapekey
        for i, shape_key in enumerate(
            obj.data.shape_keys.key_blocks[1:], start=0
        ):  # Exclude the "Basis" shape key
            for j, vert in enumerate(shape_key.data):
                shapedirs[j, :, i] = np.array(vert.co) - updated_vertices[j]
    except AttributeError:
        print("No shapekeys found.")
        shapedirs = np.zeros((num_vertices, 3))

    pkl_data["shapedirs"] = shapedirs
    print(shapedirs.shape)

    # Write out the new pkl file to the same location as the input pkl file with the user-specified name
    output_path = os.path.join(
        os.path.dirname(export_path), bpy.context.scene.smpl_tool.output_filename
    )
    try:
        with open(output_path, "wb") as f:
            pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"New SMPL file saved successfully at {output_path}.")
    except Exception as e:
        print(f"Failed to save new SMPL file: {e}")


"""
GUI-ify
"""


class SMPL_PT_Panel(bpy.types.Panel):
    bl_label = "SMIL Model Importer"
    bl_idname = "SMPL_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SMPL"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        smpl_tool = scene.smpl_tool

        layout.prop(smpl_tool, "pkl_filepath")
        layout.prop(smpl_tool, "npz_filepath")
        layout.prop(smpl_tool, "shapekeys_from_PCA")
        layout.prop(smpl_tool, "number_of_PC")
        layout.prop(smpl_tool, "clean_mesh")
        layout.prop(smpl_tool, "merging_threshold")
        layout.prop(smpl_tool, "regress_joints")
        layout.prop(smpl_tool, "symmetrise")

        layout.operator("smpl.import_model", text="Import SMIL Model")

        # Add section for pose correctives
        layout.separator()
        layout.label(text="Advanced processing options:")
        layout.operator("smpl.recompute_joint_positions", text="Recompute joint positions")
        layout.operator("smpl.load_all_unposed_meshes", text="Load all unposed registered meshes")
        layout.operator("smpl.generate_from_unposed", text="Generate SMIL model from unposed meshes")
        
        layout.separator()
        layout.prop(smpl_tool, "output_filename")
        layout.operator("smpl.export_model", text="Export SMIL Model")

        # Add section for pose correctives
        layout.separator()
        layout.label(text="Apply corrective shape keys:")
        # Add note about pose correctives availability
        box = layout.box()
        box.label(
            text="Note: Only available when pose correctives are provided via posedirs",
            icon="INFO",
        )
        layout.operator("smpl.apply_pose_correctives", text="Apply Pose Correctives")


def store_smpl_data(context, data, obj=None):
    """Store SMPL data in a temporary file and save the path"""
    if obj is None:
        obj = context.active_object

    if obj:
        # Create a temporary file to store the data
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"smpl_data_{obj.name}.pkl")

        # Save the data to the temporary file
        with open(temp_path, "wb") as f:
            pickle.dump(data, f)

        # Store only the path in the object's custom properties
        obj["smpl_data_path"] = temp_path
        obj["has_smpl_data"] = True
        context.scene.smpl_tool.has_smpl_data = True


def get_smpl_data(context):
    """Retrieve SMPL data from the temporary file"""
    obj = context.active_object
    if obj and "smpl_data_path" in obj:
        temp_path = obj["smpl_data_path"]
        if os.path.exists(temp_path):
            with open(temp_path, "rb") as f:
                return pickle.load(f)
    return None


def get_joint_distances(armature_obj):
    """Calculate distances between all joint pairs in the armature."""
    joints = armature_obj.data.bones
    distances = []

    # Calculate distances between all joint pairs
    for i, bone1 in enumerate(joints):
        for j, bone2 in enumerate(joints[i + 1 :], i + 1):
            dist = (bone1.head_local - bone2.head_local).length
            distances.append([bone1.name, bone2.name, dist])

    return distances


def get_joint_distances_from_positions(joint_positions, joint_names):
    """Calculate distances between all joint pairs from joint positions."""
    distances = []

    # Calculate distances between all joint pairs
    for i, pos1 in enumerate(joint_positions):
        for j, pos2 in enumerate(joint_positions[i + 1 :], i + 1):
            dist = np.linalg.norm(pos1 - pos2)
            distances.append([joint_names[i], joint_names[j], dist])

    return distances


def export_joint_distances(context, filepath):
    """Export joint distances to a CSV file, including distances for each shape key."""
    armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if not armature:
        return False, "No armature found"

    mesh_obj = context.active_object
    if not mesh_obj or mesh_obj.type != "MESH":
        return False, "No mesh object selected"

    # Get joint names from armature
    joint_names = [bone.name for bone in armature.data.bones]

    # Recalculate J_regressor for current mesh state
    # This ensures it works even if mesh topology has changed
    J_regressor = export_J_regressor_to_npy(mesh_obj, armature, 10)

    # Check if reference measurements are available
    smpl_tool = context.scene.smpl_tool
    reference_measurements = {}
    reference_joint_pair = []

    if smpl_tool.has_reference_data:
        reference_measurements = get_reference_measurements(context)

        # Parse joint pair from reference_joint_pair
        joint_pair = smpl_tool.reference_joint_pair

        # Try to extract joint names from the joint pair string
        # Format is typically "joint1 to joint2 [unit]"
        if "to" in joint_pair:
            parts = joint_pair.split("to")
            if len(parts) >= 2:
                joint1 = parts[0].strip()
                joint2 = parts[1].split("[")[0].strip()
                reference_joint_pair = [joint1, joint2]

        # Verify reference joints exist in the armature
        if len(reference_joint_pair) == 2:
            for joint in reference_joint_pair:
                if joint not in joint_names:
                    print(f"Warning: Reference joint '{joint}' not found in armature")
                    reference_joint_pair = []

    # Prepare data for CSV
    all_data = []

    # Add header row with scaling info if reference data is available
    if reference_joint_pair and reference_measurements:
        all_data.append(
            [
                "Shape",
                "Joint1",
                "Joint2",
                "Distance",
                "Scaling Factor",
                "Scaled Distance in mm",
            ]
        )
    else:
        all_data.append(["Shape", "Joint1", "Joint2", "Distance"])

    # Get base mesh distances using depsgraph evaluation
    depsgraph = context.evaluated_depsgraph_get()

    # Store original shape key values
    original_values = {}
    if mesh_obj.data.shape_keys:
        for key in mesh_obj.data.shape_keys.key_blocks:
            original_values[key.name] = key.value
            key.value = 0.0  # Reset all to 0

    # Update mesh to ensure we start from basis
    mesh_obj.data.update()

    # Get evaluated mesh for base shape
    eval_obj = mesh_obj.evaluated_get(depsgraph)

    # Get vertex positions from evaluated mesh
    vertex_positions = np.array([np.array(v.co) for v in eval_obj.data.vertices])

    # Calculate joint positions using J_regressor
    joint_positions = recalculate_joint_positions(vertex_positions, J_regressor)

    # Calculate distances between all joint pairs
    base_distances = []
    for i, pos1 in enumerate(joint_positions):
        for j, pos2 in enumerate(joint_positions[i + 1 :], i + 1):
            dist = np.linalg.norm(pos1 - pos2)
            base_distances.append([joint_names[i], joint_names[j], dist])

    # Add base mesh distances
    for row in base_distances:
        if reference_joint_pair and reference_measurements:
            # scaling factor is not applicable to base mesh
            scaling_factor = "N/A"
            scaled_distance = "N/A"
            all_data.append(["Base"] + row + [scaling_factor, scaled_distance])
        else:
            all_data.append(["Base"] + row)

    # Get distances for each shape key
    if mesh_obj.data.shape_keys and len(mesh_obj.data.shape_keys.key_blocks) > 1:
        # For each shape key
        for key in mesh_obj.data.shape_keys.key_blocks[1:]:  # Skip basis
            # Reset all shape keys to 0 first
            for k in mesh_obj.data.shape_keys.key_blocks:
                k.value = 0.0

            # Set this shape key to 1.0
            key.value = 1.0

            # Force a complete update of the mesh
            mesh_obj.data.update()
            context.view_layer.update()

            # Get the evaluated object with this shape key applied
            depsgraph.update()
            eval_obj = mesh_obj.evaluated_get(depsgraph)

            # Get vertex positions from evaluated mesh
            vertex_positions = np.array(
                [np.array(v.co) for v in eval_obj.data.vertices]
            )

            # Calculate joint positions using J_regressor
            joint_positions = recalculate_joint_positions(vertex_positions, J_regressor)

            # Calculate distances between joints
            key_distances = []
            for i, pos1 in enumerate(joint_positions):
                for j, pos2 in enumerate(joint_positions[i + 1 :], i + 1):
                    dist = np.linalg.norm(pos1 - pos2)
                    key_distances.append([joint_names[i], joint_names[j], dist])

            # Calculate scaling factor if reference data is available
            scaling_factor = 1.0
            # Clean the key name as it may contain file endings
            key_name = key.name.split(".")[0]

            if (
                reference_joint_pair
                and reference_measurements
                and key_name in reference_measurements
            ):
                # Find the distance between reference joints for this shape key
                ref_joint_idx1 = (
                    joint_names.index(reference_joint_pair[0])
                    if reference_joint_pair[0] in joint_names
                    else -1
                )
                ref_joint_idx2 = (
                    joint_names.index(reference_joint_pair[1])
                    if reference_joint_pair[1] in joint_names
                    else -1
                )

                if ref_joint_idx1 >= 0 and ref_joint_idx2 >= 0:
                    # Calculate the current distance between reference joints
                    current_dist = np.linalg.norm(
                        joint_positions[ref_joint_idx1]
                        - joint_positions[ref_joint_idx2]
                    )

                    # Get the reference distance
                    reference_dist = reference_measurements.get(key_name, 0.0)

                    if current_dist > 0 and reference_dist > 0:
                        # Calculate scaling factor
                        scaling_factor = reference_dist / current_dist
                        print(
                            f"Shape key {key.name}: Scaling factor = {scaling_factor} (Reference: {reference_dist}, Current: {current_dist})"
                        )

            # Add to data with shape key name and apply scaling if needed
            for dist_data in key_distances:
                if reference_joint_pair and reference_measurements:
                    scaled_distance = dist_data[2] * scaling_factor
                    all_data.append(
                        [key.name] + dist_data + [scaling_factor, scaled_distance]
                    )
                else:
                    all_data.append([key.name] + dist_data)

        # Restore original values
        for key_name, value in original_values.items():
            if key_name in mesh_obj.data.shape_keys.key_blocks:
                mesh_obj.data.shape_keys.key_blocks[key_name].value = value
            mesh_obj.data.update()

    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_data)
        return True, f"Distances exported to {filepath}"
    except Exception as e:
        return False, f"Failed to export distances: {str(e)}"


def calculate_mesh_measurements(obj):
    """Calculate surface area and volume of a mesh object using Blender's internal functions."""
    bpy.context.view_layer.objects.active = obj

    # Get mesh data
    mesh = obj.data

    # Calculate surface area
    surface_area = sum(p.area for p in mesh.polygons)

    # Calculate volume
    # We need to ensure the mesh is manifold for accurate volume calculation
    bm = bmesh.new()
    bm.from_mesh(mesh)
    volume = bm.calc_volume()
    bm.free()

    return abs(surface_area), abs(volume)


def export_mesh_measurements(context, filepath):
    """Export mesh surface area and volume measurements to a CSV file, including measurements for each shape key."""
    obj = context.active_object
    if not obj or obj.type != "MESH":
        return False, "No mesh object selected"

    # Check if reference measurements are available
    smpl_tool = context.scene.smpl_tool
    reference_measurements = {}
    reference_joint_pair = []

    if smpl_tool.has_reference_data:
        reference_measurements = get_reference_measurements(context)

        # Parse joint pair from reference_joint_pair
        joint_pair = smpl_tool.reference_joint_pair

        # Try to extract joint names from the joint pair string
        if "to" in joint_pair:
            parts = joint_pair.split("to")
            if len(parts) >= 2:
                joint1 = parts[0].strip()
                joint2 = parts[1].split("[")[0].strip()
                reference_joint_pair = [joint1, joint2]

        # Verify reference joints exist in the armature
        armature = next(
            (obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None
        )
        if armature and len(reference_joint_pair) == 2:
            joint_names = [bone.name for bone in armature.data.bones]
            for joint in reference_joint_pair:
                if joint not in joint_names:
                    print(f"Warning: Reference joint '{joint}' not found in armature")
                    reference_joint_pair = []

    try:
        # Prepare data for CSV
        all_data = []

        # Add header row with scaling info if reference data is available
        if reference_joint_pair and reference_measurements:
            all_data.append(
                ["Shape", "Measurement", "Value", "Scaling Factor", "Scaled Value"]
            )
        else:
            all_data.append(["Shape", "Measurement", "Value"])

        # Calculate base measurements
        surface_area, volume = calculate_mesh_measurements(obj)

        # Base measurements are not scaled
        if reference_joint_pair and reference_measurements:
            all_data.append(["Base", "Surface Area", surface_area, "N/A", "N/A"])
            all_data.append(["Base", "Volume", volume, "N/A", "N/A"])
        else:
            all_data.append(["Base", "Surface Area", surface_area])
            all_data.append(["Base", "Volume", volume])

        # Get measurements for each shape key
        if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 1:
            # Store original values
            original_values = {}
            for key in obj.data.shape_keys.key_blocks[1:]:  # Skip basis
                original_values[key.name] = key.value
                key.value = 0.0

            # Update mesh to ensure we start from basis
            obj.data.update()
            context.view_layer.update()  # Force a full update

            # Create a temporary object for measurements
            temp_mesh = bpy.data.meshes.new("TempMeasurementMesh")
            temp_obj = bpy.data.objects.new("TempMeasurementObj", temp_mesh)
            context.collection.objects.link(temp_obj)

            # If we have reference measurements, we need to calculate joint distances
            joint_distances = {}
            if reference_joint_pair and reference_measurements and armature:
                # Recalculate J_regressor for current mesh state
                J_regressor = export_J_regressor_to_npy(obj, armature, 10)
                joint_names = [bone.name for bone in armature.data.bones]

                # Get indices of reference joints
                ref_joint_idx1 = (
                    joint_names.index(reference_joint_pair[0])
                    if reference_joint_pair[0] in joint_names
                    else -1
                )
                ref_joint_idx2 = (
                    joint_names.index(reference_joint_pair[1])
                    if reference_joint_pair[1] in joint_names
                    else -1
                )

                if ref_joint_idx1 >= 0 and ref_joint_idx2 >= 0:
                    # For each shape key, calculate the joint distance
                    for key in obj.data.shape_keys.key_blocks[1:]:  # Skip basis
                        # Reset all shape keys to 0 first
                        for k in obj.data.shape_keys.key_blocks[1:]:
                            k.value = 0.0

                        # Set this shape key to 1.0
                        key.value = 1.0
                        obj.data.update()

                        # Get vertex positions with this shape key applied
                        vertex_positions = np.array(
                            [np.array(v.co) for v in obj.data.vertices]
                        )

                        # Calculate joint positions using J_regressor
                        joint_positions = recalculate_joint_positions(
                            vertex_positions, J_regressor
                        )

                        # Calculate distance between reference joints
                        current_dist = np.linalg.norm(
                            joint_positions[ref_joint_idx1]
                            - joint_positions[ref_joint_idx2]
                        )
                        joint_distances[key.name] = current_dist

                        # Reset this shape key
                        key.value = 0.0
                        obj.data.update()

            # For each shape key
            for key in obj.data.shape_keys.key_blocks[1:]:  # Skip basis
                # Reset all shape keys to 0 first
                for k in obj.data.shape_keys.key_blocks[1:]:
                    k.value = 0.0

                # Set this shape key to 1.0
                key.value = 1.0

                # Force a complete update of the mesh
                obj.data.update()
                context.view_layer.update()

                # Get the evaluated object
                depsgraph = context.evaluated_depsgraph_get()
                eval_obj = obj.evaluated_get(depsgraph)

                # Copy the evaluated mesh to our temporary mesh
                temp_mesh.clear_geometry()
                temp_mesh.from_pydata(
                    [v.co[:] for v in eval_obj.data.vertices],
                    [],
                    [p.vertices[:] for p in eval_obj.data.polygons],
                )
                temp_mesh.update()

                # Calculate measurements on the temporary object
                key_surface_area, key_volume = calculate_mesh_measurements(temp_obj)

                # Calculate scaling factor if reference data is available
                scaling_factor = 1.0
                # Clean the key name as it may contain file endings
                key_name = key.name.split(".")[0]

                if (
                    reference_joint_pair
                    and reference_measurements
                    and key_name in reference_measurements
                    and key.name in joint_distances
                ):

                    # Get the reference distance
                    reference_dist = reference_measurements.get(key_name, 0.0)
                    current_dist = joint_distances[key.name]

                    if current_dist > 0 and reference_dist > 0:
                        # Calculate linear scaling factor
                        scaling_factor = reference_dist / current_dist
                        print(
                            f"Shape key {key.name}: Scaling factor = {scaling_factor}"
                        )

                        # Scale surface area (s²) and volume (s³)
                        scaled_surface_area = key_surface_area * (scaling_factor**2)
                        scaled_volume = key_volume * (scaling_factor**3)

                        # Add to data with shape key name and scaled values
                        all_data.append(
                            [
                                key.name,
                                "Surface Area",
                                key_surface_area,
                                scaling_factor,
                                scaled_surface_area,
                            ]
                        )
                        all_data.append(
                            [
                                key.name,
                                "Volume",
                                key_volume,
                                scaling_factor,
                                scaled_volume,
                            ]
                        )
                    else:
                        # Add unscaled values if we can't calculate scaling
                        all_data.append(
                            [key.name, "Surface Area", key_surface_area, "N/A", "N/A"]
                        )
                        all_data.append([key.name, "Volume", key_volume, "N/A", "N/A"])
                else:
                    # Add to data without scaling
                    if reference_joint_pair and reference_measurements:
                        all_data.append(
                            [key.name, "Surface Area", key_surface_area, "N/A", "N/A"]
                        )
                        all_data.append([key.name, "Volume", key_volume, "N/A", "N/A"])
                    else:
                        all_data.append([key.name, "Surface Area", key_surface_area])
                        all_data.append([key.name, "Volume", key_volume])

            # Remove temporary object
            bpy.data.objects.remove(temp_obj)
            bpy.data.meshes.remove(temp_mesh)

            # Restore original values
            for key_name, value in original_values.items():
                obj.data.shape_keys.key_blocks[key_name].value = value
            obj.data.update()
            context.view_layer.update()

        # Export to CSV
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_data)

        return True, f"Measurements exported to {filepath}"
    except Exception as e:
        return False, f"Failed to export measurements: {str(e)}"


def sort_shape_keys(obj):
    """
    Sort all shape keys alphabetically (except for the Basis shape key which stays first).

    Args:
    - obj (bpy.types.Object): The mesh object with shape keys to sort
    """
    if not obj.data.shape_keys or len(obj.data.shape_keys.key_blocks) <= 2:
        # No need to sort if there are 0, 1, or 2 shape keys (basis + 1)
        return

    # Get shape key names (excluding Basis)
    shape_keys = obj.data.shape_keys.key_blocks
    names = [key.name for key in shape_keys[1:]]

    # Sort the names
    sorted_names = sorted(names)

    # Rearrange shape keys using the proper Blender API
    for i, name in enumerate(sorted_names, 1):
        # Get current index of this shape key
        current_idx = shape_keys.find(name)
        # Move to the correct position (i) using the proper API
        if current_idx != i:
            # We need to use the shape_key_move operator
            bpy.context.view_layer.objects.active = obj
            obj.active_shape_key_index = current_idx
            # Move the shape key up or down until it's in the right position
            if current_idx < i:
                # Need to move down
                for _ in range(i - current_idx):
                    bpy.ops.object.shape_key_move(type="DOWN")
            else:
                # Need to move up
                for _ in range(current_idx - i):
                    bpy.ops.object.shape_key_move(type="UP")

    print(f"Sorted {len(sorted_names)} shape keys alphabetically")


def load_reference_measurements(filepath):
    """
    Load reference measurements from a CSV file.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        tuple: (joint_pair, measurements_dict) where joint_pair is a string describing the measured joints
               and measurements_dict is a dictionary mapping shape names to measurement values
    """
    try:
        measurements = {}
        joint_pair = ""

        with open(filepath, "r") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            # Extract joint pair from header
            if len(header) >= 2:
                # expecting the joint pair to be in the format "Joint1 to Joint2"
                joint_pair = header[1]

            # Read measurements
            for row in reader:
                if len(row) >= 2:
                    shape_name = row[0]
                    try:
                        measurement = float(row[1])
                        measurements[shape_name] = measurement
                    except ValueError:
                        print(
                            f"Warning: Could not convert measurement for {shape_name} to float"
                        )

        return joint_pair, measurements
    except Exception as e:
        print(f"Error loading reference measurements: {e}")
        return "", {}


def get_reference_measurements(context):
    """Get the reference measurements from the temporary file"""
    if "reference_measurements_path" in context.scene:
        temp_path = context.scene["reference_measurements_path"]
        if os.path.exists(temp_path):
            with open(temp_path, "rb") as f:
                return pickle.load(f)
    return {}


class SMPL_OT_ImportModel(bpy.types.Operator):
    bl_idname = "smpl.import_model"
    bl_label = "Import SMIL Model"

    def execute(self, context):
        scene = context.scene
        smpl_tool = scene.smpl_tool

        try:
            pkl_filepath = bpy.path.abspath(smpl_tool.pkl_filepath)
            data = load_pkl_file(pkl_filepath)
            if data:
                obj = create_mesh_from_pkl(data)
                if obj:
                    obj["SMIL_TYPE"] = "SMIL_model_from_direct_npz_import"
                    # Store SMPL data in the object
                    store_smpl_data(context, data, obj=obj)

                    create_armature_and_weights(data, obj)
                    if not smpl_tool.npz_filepath:
                        self.report(
                            {"INFO"},
                            "No .npz file provided, skipping shapekey creation.",
                        )
                        return {"FINISHED"}

                    npz_filepath = bpy.path.abspath(smpl_tool.npz_filepath)
                    if not os.path.exists(npz_filepath):
                        self.report(
                            {"INFO"},
                            "Could not find .npz file, skipping shapekey creation.",
                        )
                        return {"FINISHED"}

                    npz_data = load_npz_file(npz_filepath)
                    verts_data = npz_data["verts"]

                    if verts_data.shape[1] != len(obj.data.vertices):
                        self.report({"ERROR"}, "Vertex count mismatch.")
                        return {"CANCELLED"}

                    if smpl_tool.shapekeys_from_PCA:
                        output_dir = os.path.dirname(pkl_filepath)
                        labels = (
                            list(npz_data["labels"]) if "labels" in npz_data else None
                        )
                        cov, mean_betas = apply_pca_and_create_shapekeys(
                            verts_data,
                            obj,
                            smpl_tool.number_of_PC,
                            overwrite_mesh=True,
                            labels=labels,
                            output_dir=output_dir,
                        )

                    else:
                        cov, mean_betas = create_shapekeys(npz_data, obj)

                    data["shape_cov"] = cov
                    data["shape_mean_betas"] = mean_betas
                    # Update the stored data with the new shape info
                    store_smpl_data(context, data, obj=obj)

                    if smpl_tool.symmetrise:
                        make_symmetrical(obj, data)

                    if smpl_tool.regress_joints:
                        apply_updated_joint_positions(obj, data)

                    if smpl_tool.clean_mesh:
                        cleanup_mesh(obj, center_tolerance=smpl_tool.merging_threshold)

                    self.report({"INFO"}, "SMIL Model imported successfully.")
                    return {"FINISHED"}
                else:
                    self.report({"ERROR"}, "Failed to create mesh from .pkl file.")
                    return {"CANCELLED"}
            else:
                self.report({"ERROR"}, "Failed to load .pkl file.")
                return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to import SMPL Model: {e}")
            return {"CANCELLED"}


class SMPL_OT_GenerateFromUnposed(bpy.types.Operator):
    bl_idname = "smpl.generate_from_unposed"
    bl_label = "Generate SMIL model from unposed meshes"
    bl_description = "Generates a new SMIL model by using all loaded unposed meshes as shapekeys"

    def execute(self, context):
        scene = context.scene
        smpl_tool = scene.smpl_tool

        # 1. Find all tagged objects
        unposed_meshes = [
            obj
            for obj in bpy.data.objects
            if obj.get("SMIL_TYPE") == "unposed_registered_mesh"
        ]
        if not unposed_meshes:
            self.report({"ERROR"}, "No unposed registered meshes found in the scene.")
            return {"CANCELLED"}

        self.report(
            {"INFO"}, f"Found {len(unposed_meshes)} unposed meshes to use as shapekeys."
        )

        try:
            # 2. Load base pkl file to create the new model on
            pkl_filepath = bpy.path.abspath(smpl_tool.pkl_filepath)
            data = load_pkl_file(pkl_filepath)
            if not data:
                self.report({"ERROR"}, "Failed to load base .pkl file.")
                return {"CANCELLED"}

            obj = create_mesh_from_pkl(data)
            if not obj:
                self.report({"ERROR"}, "Failed to create mesh from .pkl file.")
                return {"CANCELLED"}

            # 3. Get vertex data from scene objects
            verts_list = []
            labels_list = []
            depsgraph = context.evaluated_depsgraph_get()

            for unposed_obj in unposed_meshes:
                eval_obj = unposed_obj.evaluated_get(depsgraph)

                # Ensure vertex counts match before proceeding
                if len(eval_obj.data.vertices) != len(obj.data.vertices):
                    self.report(
                        {
                            "ERROR"
                        },
                        f"Vertex count mismatch between base model and '{unposed_obj.name}'. Skipping.",
                    )
                    continue

                mesh_verts = np.array([v.co[:] for v in eval_obj.data.vertices])
                verts_list.append(mesh_verts)
                labels_list.append(unposed_obj.name)

            if not verts_list:
                self.report(
                    {"ERROR"}, "No valid unposed meshes found with matching vertex counts."
                )
                bpy.data.objects.remove(obj)
                return {"CANCELLED"}

            verts_data = np.array(verts_list)
            mean_shape = np.mean(verts_data, axis=0)

            npz_data = {"verts": verts_data, "labels": labels_list}

            # The rest is similar to SMPL_OT_ImportModel
            obj["SMIL_TYPE"] = "SMIL_model_from_unposed_meshes"
            store_smpl_data(context, data, obj=obj)

            create_armature_and_weights(data, obj)

            # Overwrite the base mesh geometry with the mean shape of the unposed meshes.
            # This is crucial for the shapekeys to be based on the correct average shape.
            for i, v_co in enumerate(mean_shape):
                obj.data.vertices[i].co = v_co

            if smpl_tool.shapekeys_from_PCA:
                output_dir = os.path.dirname(pkl_filepath)
                cov, mean_betas = apply_pca_and_create_shapekeys(
                    verts_data,
                    obj,
                    smpl_tool.number_of_PC,
                    overwrite_mesh=True,
                    labels=labels_list,
                    output_dir=output_dir,
                )
            else:
                cov, mean_betas = create_shapekeys(npz_data, obj)

            data["shape_cov"] = cov
            data["shape_mean_betas"] = mean_betas
            # Update the stored data with the new shape info
            store_smpl_data(context, data, obj=obj)

            if smpl_tool.symmetrise:
                make_symmetrical(obj, data)

            if smpl_tool.regress_joints:
                apply_updated_joint_positions(obj, data)

            if smpl_tool.clean_mesh:
                cleanup_mesh(obj, center_tolerance=smpl_tool.merging_threshold)

            self.report(
                {"INFO"}, "SMIL Model generated from unposed meshes successfully."
            )
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to generate SMIL Model: {e}")
            return {"CANCELLED"}


class SMPL_OT_ExportModel(bpy.types.Operator):
    bl_idname = "smpl.export_model"
    bl_label = "Export SMIL Model"

    def execute(self, context):
        # Get SMPL data from the active object
        data = get_smpl_data(context)
        if data is None:
            self.report(
                {"INFO"},
                "No SMPL model data found. Attempting to export selected mesh as a new SMPL model.",
            )

        scene = context.scene
        smpl_tool = scene.smpl_tool

        try:
            obj = bpy.context.active_object
            if not obj or obj.type != "MESH":
                self.report({"ERROR"}, "No valid mesh object selected.")
                return {"CANCELLED"}

            export_smpl_model(
                obj, pkl_data=data, export_path=bpy.path.abspath(smpl_tool.pkl_filepath)
            )

            self.report({"INFO"}, "SMPL Model exported successfully.")
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to export SMPL Model: {str(e)}")
            return {"CANCELLED"}


class SMPL_OT_LoadAllUnposedMeshes(bpy.types.Operator):
    bl_idname = "smpl.load_all_unposed_meshes"
    bl_label = "Load all unposed registered meshes"
    bl_description = "Load and rig all registered meshes from the .npz file, offsetting them in the viewport."

    def execute(self, context):
        scene = context.scene
        smpl_tool = scene.smpl_tool
        wm = context.window_manager

        # Load PKL data (for rigging info)
        pkl_filepath = bpy.path.abspath(smpl_tool.pkl_filepath)
        pkl_data = load_pkl_file(pkl_filepath)
        if not pkl_data:
            self.report({"ERROR"}, "Failed to load .pkl file.")
            return {'CANCELLED'}

        # Load NPZ data (for registered meshes)
        npz_filepath = bpy.path.abspath(smpl_tool.npz_filepath)
        if not os.path.exists(npz_filepath):
            self.report({"ERROR"}, "Could not find .npz file.")
            return {'CANCELLED'}
        npz_data = load_npz_file(npz_filepath)
        if npz_data is None or "verts" not in npz_data:
            self.report({"ERROR"}, "No 'verts' key found in .npz file.")
            return {'CANCELLED'}

        verts_array = npz_data["verts"]  # shape (N, V, 3)
        labels = npz_data["labels"] if "labels" in npz_data else [f"mesh_{i}" for i in range(len(verts_array))]
        faces = pkl_data["f"]
        weights = pkl_data["weights"]
        joints = pkl_data["J"]
        kintree_table = pkl_data["kintree_table"]
        joint_names = pkl_data["J_names"] if "J_names" in pkl_data else [f"J_{i}" for i in range(joints.shape[0])]
        J_regressor = np.copy(pkl_data["J_regressor"]) if "J_regressor" in pkl_data else None
        global_rots = npz_data["global_rot"] if "global_rot" in npz_data else None  # (N, 3)
        joint_rots = npz_data["joint_rot"] if "joint_rot" in npz_data else None    # (N, J-1, 3)
        translations = npz_data["trans"] if "trans" in npz_data else None          # (N, 3)

        n_meshes = len(verts_array)
        # --- Compute mean shape and mean joint locations ---
        mean_shape = np.mean(verts_array, axis=0)  # (V, 3)
        mean_joints = J_regressor @ mean_shape  # (J, 3)
        # Build child lookup for each joint from kintree_table
        num_joints = mean_joints.shape[0]
        # array to store scaling and translation to morph from mean shape to target shapes
        transform_data = np.zeros((num_joints, n_meshes * 6))
        children = [[] for _ in range(num_joints)]
        for parent, child in zip(kintree_table[0], kintree_table[1]):
            if parent >= 0:
                children[parent].append(child)
        wm.progress_begin(0, n_meshes)
        try:
            for i, verts in enumerate(verts_array):
                wm.progress_update(i)

                # Apply translation from npz if it exists
                if translations is not None and i < len(translations):
                    verts = verts - translations[i]

                # Always use a fresh copy of the original J_regressor
                J_reg = np.copy(J_regressor) if J_regressor is not None else None
                # Build a data dict for this mesh
                mesh_data = {
                    "v_template": verts,
                    "f": faces,
                    "weights": weights,
                    "J": joints.copy(),  # will be updated below
                    "kintree_table": kintree_table,
                    "J_names": joint_names,
                    "J_regressor": J_reg,
                }
                obj = create_mesh_from_pkl(mesh_data)
                if obj is None:
                    self.report({"WARNING"}, f"Failed to create mesh for {labels[i]}")
                    continue
                obj.name = str(labels[i])
                obj["SMIL_TYPE"] = "unposed_registered_mesh"
                # Rig the mesh
                create_armature_and_weights(mesh_data, obj)
                # Find the armature just created (should be the active object and of type 'ARMATURE')
                armature = bpy.context.active_object if bpy.context.active_object and bpy.context.active_object.type == 'ARMATURE' else None
                if armature is None:
                    # Try to find the most recently created armature
                    armatures = [a for a in bpy.data.objects if a.type == 'ARMATURE']
                    armature = armatures[-1] if armatures else None
                if armature is not None:
                    # Name the armature with SMIL_ prefix
                    armature.name = f"SMIL_{labels[i]}"
                    # --- Control Hierarchy Setup ---
                    # Main parent for all controls of this mesh
                    snap_controls_parent_name = f"Snap_Controls_{armature.name}"
                    snap_controls_parent = bpy.data.objects.new(snap_controls_parent_name, None)
                    snap_controls_parent.location = (0, i, 0)
                    context.collection.objects.link(snap_controls_parent)

                    # Parent for IK controls, shaped as a sphere
                    controls_parent_name = f"IK_Controls_{armature.name}"
                    controls_parent = bpy.data.objects.new(controls_parent_name, None)
                    controls_parent.empty_display_type = 'SPHERE'
                    controls_parent.empty_display_size = 0.8
                    controls_parent.parent = snap_controls_parent
                    controls_parent.location = (0, 0, 0)  # Relative to snap_controls_parent
                    context.collection.objects.link(controls_parent)

                    # Parent armature to the main control and reset its location
                    armature.parent = snap_controls_parent
                    armature.location = (0, 0, 0)
                    
                    # Store offset for world space calculations
                    armature_offset = snap_controls_parent.location
                    
                    # Move the mesh to the origin relative to the armature
                    obj.location = (0, 0, 0)
                # Update joint locations using J_reg and current mesh vertices
                if J_reg is not None and armature is not None:
                    vertex_positions = verts
                    mesh_joints = J_reg @ vertex_positions  # (J, 3)
                    bpy.context.view_layer.objects.active = armature
                    bpy.ops.object.mode_set(mode="EDIT")

                    edit_bones = armature.data.edit_bones

                    # First, set all head positions, as they are needed for tail calculations
                    for j, bone in enumerate(edit_bones):
                        bone.head = mesh_joints[j]

                    # Then, set tail positions based on children
                    for j, bone in enumerate(edit_bones):
                        child_indices = children[j]
                        num_children = len(child_indices)

                        if num_children == 0:  # Leaf bone
                            bone.tail = bone.head + Vector((0, 0, 0.1))
                        elif num_children == 1:  # Single child
                            child_bone = edit_bones[child_indices[0]]
                            bone.tail = child_bone.head
                        else:  # Multiple children
                            # Calculate the mean position of children heads
                            child_head_vectors = [edit_bones[child_idx].head for child_idx in child_indices]
                            mean_pos = sum(child_head_vectors, Vector()) / num_children
                            bone.tail = mean_pos

                    bpy.ops.object.mode_set(mode="OBJECT")

                # --- PER-BONE LENGTH NORMALIZATION (HIERARCHICAL) ---
                if armature is not None:
                    bpy.context.view_layer.objects.active = armature
                    bpy.ops.object.mode_set(mode="POSE")
                    pose_bones = armature.pose.bones
                    # For each joint, compute mean and mesh distances to direct children
                    raw_scales = np.ones(num_joints)
                    min_dist = 1e-6  # Avoid division by zero
                    for j in range(num_joints):
                        if j == 0:
                            continue  # Do not scale the root bone
                        child_indices = children[j]
                        if not child_indices:
                            continue  # Skip scaling for joints with no children
                        mesh_dists = [np.linalg.norm(mesh_joints[j] - mesh_joints[c]) for c in child_indices]
                        mean_dists = [np.linalg.norm(mean_joints[j] - mean_joints[c]) for c in child_indices]
                        ratios = []
                        for md, mmd in zip(mesh_dists, mean_dists):
                            if mmd > min_dist:
                                ratios.append(md / mmd)
                        if ratios:
                            raw_scales[j] = np.mean(ratios)
                        else:
                            raw_scales[j] = 1.0
                    # Now compute hierarchical scales
                    final_scales = np.ones(num_joints)
                    # Build parent lookup for each joint
                    parent_lookup = {child: parent for parent, child in zip(kintree_table[0], kintree_table[1]) if parent >= 0}
                    for j in range(1, num_joints):  # skip root
                        # Compute cumulative product of all ancestor scales
                        cumulative = 1.0
                        parent = parent_lookup.get(j, None)
                        while parent is not None and parent > 0:
                            cumulative *= final_scales[parent]
                            parent = parent_lookup.get(parent, None)
                        if raw_scales[j] > 0:
                            final_scales[j] = raw_scales[j] / cumulative
                        else:
                            final_scales[j] = 1.0
                        pose_bones[j].scale = Vector([1.0 / final_scales[j]] * 3)

                    # Store the inverse of the applied scale (which is final_scales)
                    # The applied scale is 1.0 / final_scales[j]
                    scale_col_start = i * 6
                    transform_data[:, scale_col_start : scale_col_start + 3] = np.tile(
                        final_scales.reshape(-1, 1), (1, 3)
                    )
                    
                    bpy.ops.object.mode_set(mode="POSE")
                    
                # --- END PER-BONE LENGTH NORMALIZATION (HIERARCHICAL) ---
                bpy.context.view_layer.update()

                # --- IK Rig Setup ---

                # 1. Get armature and bone data
                kintree_table = pkl_data['kintree_table']
                joint_names = pkl_data['J_names']
                num_joints = len(joint_names)

                # 2. Identify leaf bones
                parent_indices = set(kintree_table[0, :])
                leaf_bone_indices = [i for i in range(num_joints) if i not in parent_indices]

                # 3. Batch-calculate all IK target positions in Pose Mode
                bpy.context.view_layer.objects.active = armature
                
                target_positions = {}
                for bone_idx in range(num_joints):
                    # Do not create targets for leaf bones
                    if bone_idx in leaf_bone_indices:
                        continue
                    pose_bone = armature.pose.bones[bone_idx]
                    world_tail_pos = armature.matrix_world @ pose_bone.tail
                    target_positions[pose_bone.name] = world_tail_pos

                # 4. Batch-create all IK target empties in Object Mode
                bpy.ops.object.mode_set(mode='OBJECT')
                
                ik_targets = {}
                for bone_name, pos in target_positions.items():
                    ik_target = bpy.data.objects.new(f"IK_Target_{armature.name}_{bone_name}", None)
                    # Set location relative to the parent to avoid double transformation
                    ik_target.location = pos - armature_offset
                    ik_target.empty_display_size = 0.05
                    ik_target.parent = controls_parent
                    context.collection.objects.link(ik_target)
                    ik_targets[bone_name] = ik_target

                # 5. Batch-apply all constraints in Pose Mode
                bpy.context.view_layer.objects.active = armature
                bpy.ops.object.mode_set(mode='POSE')

                for bone_idx in range(num_joints):
                    pose_bone = armature.pose.bones[bone_idx]
                    ik_target = ik_targets.get(pose_bone.name)

                    if not ik_target:
                        continue

                    ik_constraint = pose_bone.constraints.new('IK')
                    ik_constraint.target = ik_target
                    ik_constraint.chain_count = 1
                    ik_constraint.influence = 1.0

                # 6. Return to Object Mode
                bpy.ops.object.mode_set(mode='OBJECT')

                # --- Hierarchical Joint Alignment to Mean Shape ---
                # This aligns the posed rig to the mean shape's proportions.

                # A. Calculate tail target positions from the mean shape
                mean_shape_tail_targets = {}
                for j in range(num_joints):
                    if j in leaf_bone_indices:
                        continue

                    child_indices = children[j]
                    if not child_indices:
                        continue

                    if len(child_indices) == 1:
                        target_pos = Vector(mean_joints[child_indices[0]])
                    else:
                        child_head_vectors = [Vector(mean_joints[child_idx]) for child_idx in child_indices]
                        target_pos = sum(child_head_vectors, Vector()) / len(child_indices)

                    bone_name = joint_names[j]
                    mean_shape_tail_targets[bone_name] = target_pos + armature_offset

                # B. Move the IK empties to their new target locations to pose the rig
                for bone_name, ik_empty in ik_targets.items():
                    if bone_name in mean_shape_tail_targets:
                        ik_empty.location = mean_shape_tail_targets[bone_name] - armature_offset

                # C. For bones with siblings, prepare to snap their heads to the mean shape's head position
                parent_lookup = {child: parent for parent, child in zip(kintree_table[0], kintree_table[1]) if parent >= 0}
                snap_target_data = {} # {bone_name: target_world_pos}
                
                # Manually align the root bone's head to the mean shape's root joint position
                root_bone_name = joint_names[0]
                snap_target_data[root_bone_name] = Vector(mean_joints[0]) + armature_offset

                # Calculate and store translation for the root bone
                translation = mean_joints[0] - mesh_joints[0]
                trans_col_start = i * 6 + 3
                transform_data[0, trans_col_start : trans_col_start + 3] = translation
                
                for j in range(1, num_joints): # Skip root bone
                    parent_idx = parent_lookup.get(j)
                    if parent_idx is not None and len(children[parent_idx]) > 1:
                        bone_name = joint_names[j]
                        snap_target_data[bone_name] = Vector(mean_joints[j]) + armature_offset
                        # Calculate and store translation for joints with siblings
                        translation = mean_joints[j] - mesh_joints[j]
                        trans_col_start = i * 6 + 3
                        transform_data[j, trans_col_start : trans_col_start + 3] = translation
                        
                # D. Batch-create snap targets and constraints
                if snap_target_data:
                    snap_targets = {}
                    for bone_name, target_pos in snap_target_data.items():
                        snap_target = bpy.data.objects.new(f"Snap_Target_{armature.name}_{bone_name}", None)
                        snap_target.location = target_pos - armature_offset
                        snap_target.empty_display_size = 0.02
                        snap_target.parent = snap_controls_parent
                        context.collection.objects.link(snap_target)
                        snap_targets[bone_name] = snap_target

                    # Apply constraints
                    bpy.context.view_layer.objects.active = armature
                    bpy.ops.object.mode_set(mode='POSE')
                    for bone_name, snap_target_empty in snap_targets.items():
                        pose_bone = armature.pose.bones.get(bone_name)
                        if pose_bone:
                            copy_loc_constraint = pose_bone.constraints.new('COPY_LOCATION')
                            copy_loc_constraint.target = snap_target_empty
                    bpy.ops.object.mode_set(mode='OBJECT')
        finally:
            wm.progress_end()
            
        # --- Export morph data to CSV ---
        try:
            output_path = os.path.join(
                os.path.dirname(pkl_filepath), "smil_morph_data.csv"
            )
            with open(output_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Header
                header = ["joint_name"]
                for label in labels:
                    header.extend(
                        [
                            f"{label}_scale_x",
                            f"{label}_scale_y",
                            f"{label}_scale_z",
                            f"{label}_translation_x",
                            f"{label}_translation_y",
                            f"{label}_translation_z",
                        ]
                    )
                writer.writerow(header)
                # Data rows
                for j in range(num_joints):
                    row = [joint_names[j]] + transform_data[j, :].tolist()
                    writer.writerow(row)
            print(f"Morph data exported to {output_path}")

            # --- Also export PCA of morph data in same layout ---
            try:
                # Build feature matrix X with one row per mesh and features per joint (scale xyz + translation xyz)
                features_per_joint = 6
                X = np.zeros((n_meshes, num_joints * features_per_joint), dtype=np.float32)
                for i in range(n_meshes):
                    # collect features for mesh i across all joints
                    for j in range(num_joints):
                        start_feat = j * features_per_joint
                        end_feat = start_feat + features_per_joint
                        start_src = i * features_per_joint
                        end_src = start_src + features_per_joint
                        X[i, start_feat:end_feat] = transform_data[j, start_src:end_src]

                # Determine number of components respecting limits
                requested_components = int(getattr(smpl_tool, "number_of_PC", 1))
                n_components = max(1, min(requested_components, X.shape[0], X.shape[1]))

                pca = PCA(n_components=n_components)
                pca.fit(X)
                components = pca.components_  # (k, num_joints*6)

                pc_output_path = os.path.join(
                    os.path.dirname(pkl_filepath), "smil_morph_PC_data.csv"
                )
                with open(pc_output_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    # Header: joint_name, then for each PC six columns matching the original naming pattern
                    header_pc = ["joint_name"]
                    for k in range(n_components):
                        pc_label = f"PC_{k+1}"
                        header_pc.extend(
                            [
                                f"{pc_label}_scale_x",
                                f"{pc_label}_scale_y",
                                f"{pc_label}_scale_z",
                                f"{pc_label}_translation_x",
                                f"{pc_label}_translation_y",
                                f"{pc_label}_translation_z",
                            ]
                        )
                    writer.writerow(header_pc)

                    # Rows per joint, values sliced from component loadings
                    for j in range(num_joints):
                        row = [joint_names[j]]
                        start_feat = j * features_per_joint
                        end_feat = start_feat + features_per_joint
                        for k in range(n_components):
                            row.extend(components[k, start_feat:end_feat].tolist())
                        writer.writerow(row)

                print(
                    f"Morph PCA data (k={n_components}) exported to {pc_output_path}. Explained variance ratios: {pca.explained_variance_ratio_}"
                )

                # Export XY coordinates (PC1, PC2 scores) and PCA stats
                try:
                    # Scores for each mesh
                    scores = pca.transform(X)  # shape (n_meshes, k)
                    pc_xy_path = os.path.join(os.path.dirname(pkl_filepath), "smil_morph_PC_xy.csv")
                    with open(pc_xy_path, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["label", "PC1", "PC2"]) 
                        for i, lab in enumerate(labels):
                            pc1 = scores[i, 0] if scores.shape[1] > 0 else 0.0
                            pc2 = scores[i, 1] if scores.shape[1] > 1 else 0.0
                            writer.writerow([lab, pc1, pc2])

                    stats_path = os.path.join(os.path.dirname(pkl_filepath), "smil_morph_PCA_stats.txt")
                    with open(stats_path, "w") as f:
                        f.write("PCA stats for morph (scale/translation) PCs\n")
                        f.write(f"n_samples: {X.shape[0]}\n")
                        f.write(f"n_features: {X.shape[1]}\n")
                        f.write(f"n_components: {n_components}\n")
                        f.write(f"explained_variance_ratio: {pca.explained_variance_ratio_.tolist()}\n")
                        f.write(f"explained_variance: {pca.explained_variance_.tolist()}\n")
                        f.write(f"singular_values: {pca.singular_values_.tolist()}\n")
                        # mean vector may be large; just record L2 norm
                        f.write(f"mean_l2_norm: {float(np.linalg.norm(pca.mean_))}\n")
                    print(f"Morph PCA XY exported to {pc_xy_path}; stats to {stats_path}")
                except Exception as e:
                    print(f"Failed exporting morph PCA XY/stats: {e}")
            except Exception as e:
                print(f"Failed to export morph PCA data: {e}")
        except Exception as e:
            print(f"Failed to export morph data: {e}")
            
        self.report({"INFO"}, f"Loaded and rigged {n_meshes} meshes.")
        return {'FINISHED'}


class SMPL_OT_RecomputeJointPositions(bpy.types.Operator):
    bl_idname = "smpl.recompute_joint_positions"
    bl_label = "Recompute joint positions"
    bl_description = "Recompute the J_regressor and update joint locations for the selected armature only."

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, 'Select a mesh object with an armature.')
            return {'CANCELLED'}
        armature = obj.find_armature()
        if not armature:
            self.report({'ERROR'}, 'Selected mesh has no armature.')
            return {'CANCELLED'}
        # Recompute J_regressor for this mesh+armature
        J_regressor = export_J_regressor_to_npy(obj, armature, 10)
        vertex_positions = np.array([np.array(v.co) for v in obj.data.vertices])
        joint_positions = np.matmul(J_regressor, vertex_positions)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="EDIT")
        for j, bone in enumerate(armature.data.edit_bones):
            bone.head = joint_positions[j]
            bone.tail = joint_positions[j] + [0, 0, 0.1]
        bpy.ops.object.mode_set(mode="OBJECT")
        self.report({'INFO'}, 'Joint positions updated for selected armature.')
        return {'FINISHED'}


class SMPLProperties(bpy.types.PropertyGroup):
    pkl_filepath: bpy.props.StringProperty(
        name="PKL Filepath",
        description="Path to the .pkl file",
        default="",
        subtype="FILE_PATH",
    )

    npz_filepath: bpy.props.StringProperty(
        name="NPZ Filepath",
        description="Path to the .npz file",
        default="",
        subtype="FILE_PATH",
    )

    shapekeys_from_PCA: bpy.props.BoolProperty(
        name="shapekeys from PCA",
        description="Generate shapekeys from PCA",
        default=True,
    )

    number_of_PC: bpy.props.IntProperty(
        name="Number of Principal Components",
        description="Number of principal components for PCA",
        default=20,
    )

    regress_joints: bpy.props.BoolProperty(
        name="Regress Joints", description="Regress joint positions", default=True
    )

    clean_mesh: bpy.props.BoolProperty(
        name="Auto Clean-up Mesh",
        description="Merges overlapping vertices and removes inward facing faces",
        default=True,
    )

    merging_threshold: bpy.props.FloatProperty(
        name="Minimal vertex distance",
        description="Minimal distance between vertices on centre line during mesh cleanup",
        default=0.001,
    )

    symmetrise: bpy.props.BoolProperty(
        name="Symmetrise", description="Symmetrise the model", default=True
    )

    # Add properties to store SMPL data
    has_smpl_data: bpy.props.BoolProperty(default=False)
    v_template: bpy.props.FloatVectorProperty(size=3)  # This will store the shape
    posedirs: bpy.props.FloatVectorProperty(
        size=3
    )  # This will store the pose correctives

    # Add to SMPLProperties class:
    output_filename: bpy.props.StringProperty(
        name="Output Filename",
        description="Name of the output SMPL model file",
        default="SMPL_fit.pkl",
    )

    # Add properties for reference measurements CSV
    reference_csv_filepath: bpy.props.StringProperty(
        name="Reference CSV Filepath",
        description="Path to the CSV file containing reference measurements",
        default="",
        subtype="FILE_PATH",
    )

    reference_joint_pair: bpy.props.StringProperty(
        name="Reference Joint Pair",
        description="Joint pair used for reference measurements (read from CSV)",
        default="",
        options={"SKIP_SAVE"},
    )

    has_reference_data: bpy.props.BoolProperty(default=False)


class SMPL_OT_ApplyPoseCorrectivesOperator(bpy.types.Operator):
    bl_idname = "smpl.apply_pose_correctives"
    bl_label = "Apply Pose Correctives"
    bl_description = (
        "Apply pose-dependent corrective shape keys based on current armature pose"
    )

    @classmethod
    def poll(cls, context):
        # Only enable if we have an active mesh object with an armature
        obj = context.active_object
        if not (
            obj
            and obj.type == "MESH"
            and obj.find_armature()
            and "has_smpl_data" in obj
            and "smpl_data_path" in obj
        ):
            return False

        # Check if posedirs exists and is not empty
        data = get_smpl_data(context)
        if not data or "posedirs" not in data:
            return False

        # Check if posedirs is not empty (has actual data)
        posedirs = data["posedirs"]
        return isinstance(posedirs, np.ndarray) and posedirs.size > 0

    def execute(self, context):
        obj = context.active_object
        try:
            # Get the original data
            data = get_smpl_data(context)
            if not data or "posedirs" not in data or "v_template" not in data:
                self.report(
                    {"ERROR"}, "No SMPL data found. Please import a SMPL model first."
                )
                return {"CANCELLED"}

            apply_pose_correctives(obj, data["posedirs"], data["v_template"])
            self.report({"INFO"}, "Applied pose correctives successfully.")
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to apply pose correctives: {str(e)}")
            return {"CANCELLED"}


class SMPL_OT_ExportJointDistances(bpy.types.Operator):
    bl_idname = "smpl.export_joint_distances"
    bl_label = "Export Joint Distances"
    bl_description = "Export distances between all joints to a CSV file"

    @classmethod
    def poll(cls, context):
        return any(obj.type == "ARMATURE" for obj in bpy.data.objects)

    def execute(self, context):
        # Generate filename based on active mesh
        mesh_obj = context.active_object
        if mesh_obj and mesh_obj.type == "MESH":
            filename = f"{mesh_obj.name}_joint_distances.csv"
        else:
            filename = "joint_distances.csv"

        filepath = os.path.join(os.path.dirname(bpy.data.filepath), filename)

        success, message = export_joint_distances(context, filepath)
        if success:
            self.report({"INFO"}, message)
            return {"FINISHED"}
        else:
            self.report({"ERROR"}, message)
            return {"CANCELLED"}


class SMPL_PT_MorphometryPanel(bpy.types.Panel):
    bl_label = "SMIL Morphometry"
    bl_idname = "SMPL_PT_MorphometryPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SMPL"

    def draw(self, context):
        layout = self.layout
        smpl_tool = context.scene.smpl_tool

        # Reference measurements section
        box = layout.box()
        box.label(text="Reference Measurements:")
        box.prop(smpl_tool, "reference_csv_filepath")
        box.operator("smpl.load_reference_measurements", text="Load Reference CSV")

        # Display loaded reference info
        if smpl_tool.has_reference_data:
            info_box = box.box()
            info_box.label(text="Loaded Reference Data:", icon="INFO")
            info_box.label(text=f"Joint Pair: {smpl_tool.reference_joint_pair}")

            # Get number of measurements
            measurements = get_reference_measurements(context)
            if measurements:
                info_box.label(text=f"Number of Shapes: {len(measurements)}")

        # Add measurement export buttons
        box = layout.box()
        box.label(text="Export Measurements:")

        # Show shape key count if available
        obj = context.active_object
        if obj and obj.type == "MESH" and obj.data.shape_keys:
            shape_key_count = len(obj.data.shape_keys.key_blocks) - 1  # Exclude basis
            if shape_key_count > 0:
                box.label(
                    text=f"Will include measurements for {shape_key_count} shape keys",
                    icon="SHAPEKEY_DATA",
                )

        box.operator("smpl.export_joint_distances", text="Joint Distances")
        box.operator("smpl.export_mesh_measurements", text="Surface Area & Volume")


class SMPL_OT_ExportMeshMeasurements(bpy.types.Operator):
    bl_idname = "smpl.export_mesh_measurements"
    bl_label = "Export Mesh Measurements"
    bl_description = "Export surface area and volume measurements to a CSV file"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == "MESH"

    def execute(self, context):
        # Generate filename based on active mesh
        mesh_obj = context.active_object
        filename = f"{mesh_obj.name}_measurements.csv"
        filepath = os.path.join(os.path.dirname(bpy.data.filepath), filename)

        success, message = export_mesh_measurements(context, filepath)
        if success:
            self.report({"INFO"}, message)
            return {"FINISHED"}
        else:
            self.report({"ERROR"}, message)
            return {"CANCELLED"}


class SMPL_OT_LoadReferenceMeasurements(bpy.types.Operator):
    bl_idname = "smpl.load_reference_measurements"
    bl_label = "Load Reference Measurements"
    bl_description = "Load reference measurements from a CSV file"

    def execute(self, context):
        scene = context.scene
        smpl_tool = scene.smpl_tool

        filepath = bpy.path.abspath(smpl_tool.reference_csv_filepath)
        if not os.path.exists(filepath):
            self.report({"ERROR"}, f"File not found: {filepath}")
            return {"CANCELLED"}

        joint_pair, measurements = load_reference_measurements(filepath)

        if not measurements:
            self.report({"ERROR"}, "Failed to load measurements or file is empty")
            return {"CANCELLED"}

        # Store the data in scene properties
        smpl_tool.reference_joint_pair = joint_pair
        smpl_tool.has_reference_data = True

        # Store measurements in a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "reference_measurements.pkl")
        with open(temp_path, "wb") as f:
            pickle.dump(measurements, f)

        context.scene["reference_measurements_path"] = temp_path

        self.report(
            {"INFO"},
            f"Loaded {len(measurements)} reference measurements for {joint_pair}",
        )
        return {"FINISHED"}


# Update the classes tuple to include new classes
classes = (
    SMPL_PT_Panel,
    SMPL_PT_MorphometryPanel,
    SMPL_OT_ImportModel,
    SMPL_OT_GenerateFromUnposed,
    SMPL_OT_ExportModel,
    SMPL_OT_ApplyPoseCorrectivesOperator,
    SMPL_OT_ExportJointDistances,
    SMPL_OT_ExportMeshMeasurements,
    SMPL_OT_LoadReferenceMeasurements,
    SMPL_OT_LoadAllUnposedMeshes,
    SMPL_OT_RecomputeJointPositions,  # <-- Add here
    SMPLProperties,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.smpl_tool = bpy.props.PointerProperty(type=SMPLProperties)


def unregister():
    # Clean up temporary files
    for obj in bpy.data.objects:
        if "smpl_data_path" in obj:
            temp_path = obj["smpl_data_path"]
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    # Clean up reference measurements file
    if hasattr(bpy.context.scene, "reference_measurements_path"):
        temp_path = bpy.context.scene.reference_measurements_path
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.smpl_tool


if __name__ == "__main__":
    register()
