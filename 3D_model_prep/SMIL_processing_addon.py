bl_info = {
    "name": "SMIL Model Importer",
    "author": "Fabian Plum",
    "version": (1, 0, 2),
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
    def __init__(self, file, encoding='latin1'):
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
                self.data = np.array(state.get('x', []))
            else:
                # Handle both tuple/list format and direct data format
                self.data = np.array(state[0] if isinstance(state, (tuple, list)) else state)
            return self

        @property
        def r(self):
            """Mimic chumpy's .r property which returns the underlying data"""
            return self.data
        

# Decorators for type checking
def ensure_mesh(func):
    def wrapper(obj, *args, **kwargs):
        if not obj or obj.type != 'MESH':
            raise TypeError("The selected object is not a mesh.")
        return func(obj, *args, **kwargs)

    return wrapper


def ensure_armature(func):
    def wrapper(obj, *args, **kwargs):
        if not obj or obj.type != 'ARMATURE':
            raise TypeError("The selected object is not an armature.")
        return func(obj, *args, **kwargs)

    return wrapper


# Convert mesh to numpy arrays
def mesh_to_numpy(obj):
    mesh = obj.data
    vertices = np.array([vert.co for vert in mesh.vertices], dtype=np.float32)
    faces = np.array([poly.vertices for poly in mesh.polygons if len(poly.vertices) == 3], dtype=np.int32)
    return vertices, faces


@ensure_mesh
def triangulate_mesh(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')


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
    with open(filepath, 'w') as file:
        for vert in vertices:
            file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return filepath


@ensure_mesh
def export_vertex_groups_to_npy(obj, filepath):
    """
    Exports the vertex group weights of an object to a .npy file.
    
    Parameters:
    obj (bpy.types.Object): The object containing the vertex groups.
    filepath (str): The path to the .npy file where the weights will be saved.
    
    Returns:
    tuple: The filepath and the weights array.
    """
    # Ensure we're working on the correct object
    bpy.context.view_layer.objects.active = obj

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
    joints = armature_obj.data.bones
    joint_locations = np.array([bone.head_local for bone in joints], dtype=np.float32)
    joint_names = [bone.name for bone in joints]
    np.save(filepath, joint_locations)
    return filepath, joint_locations, joint_names


@ensure_armature
def export_joint_hierarchy_to_npy(armature_obj, filepath):
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
    mesh = obj.data
    y_axis_vertices = np.array([i for i, vert in enumerate(mesh.vertices) if np.isclose(vert.co.y, 0.0, atol=1e-3)],
                               dtype=int)
    np.save(filepath, y_axis_vertices)
    return filepath, y_axis_vertices


def find_nearest_neighbors(vertices, joint_locations, n):
    nearest_indices = np.zeros((len(joint_locations), n), dtype=np.int32)
    nearest_weights = np.zeros((len(joint_locations), n), dtype=np.float32)
    for i, joint_loc in enumerate(joint_locations):
        distances = np.linalg.norm(vertices - joint_loc, axis=1)
        nearest_indices[i] = np.argpartition(distances, n)[:n]
        nearest_distances = distances[nearest_indices[i]]
        weights = 1.0 / nearest_distances
        weights /= weights.sum()
        nearest_weights[i] = weights
    return nearest_indices, nearest_weights


@ensure_mesh
# @ensure_armature (careful, the mesh is the active object!)
def export_J_regressor_to_npy(mesh_obj, armature_obj, n, filepath=None):
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
    nearest_indices, nearest_weights = find_nearest_neighbors(vertices, joint_locations, n)
    J_regressor = np.zeros((len(joints), len(vertices)), dtype=np.float32)
    for i in range(len(joints)):
        J_regressor[i, nearest_indices[i]] = nearest_weights[i]
    
    if filepath:
        np.save(filepath, J_regressor)
        return filepath, J_regressor
    return None, J_regressor


"""
This is currently not supported.
The posedir created here captures the mesh shape at every frame
This is however not how the posedir is used in the original implementaiton and thus disabled for now.
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


# Export all model elements as a dict following the SMPL convention, contained in a .pkl file
def export_smpl_model(start_frame=0, stop_frame=1):
    smpl_dict = {
        "f": [],
        "J_regressor": [],
        "kintree_table": [],
        "J": [],
        "bs_style": "lbs",
        "weights": [],
        "posedirs": [],
        "v_template": [],
        "shapedirs": [],
        "bs_type": "lrotmin",
        "sym_verts": []
    }

    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        print("No valid mesh object selected for testing.")
        return

    print("Selected mesh object:", obj.name)
    triangulate_mesh(obj)

    vertices_npy_path = bpy.path.abspath('//test_vertices.npy')
    faces_npy_path = bpy.path.abspath('//test_faces.npy')
    vertex_groups_npy_path = bpy.path.abspath('//test_vertex_groups.npy')
    joint_locations_npy_path = bpy.path.abspath('//test_joint_locations.npy')
    joint_hierarchy_npy_path = bpy.path.abspath('//test_joint_hierarchy.npy')
    j_regressor_npy_path = bpy.path.abspath('//test_joint_regressor.npy')
    y_axis_vertices_npy_path = bpy.path.abspath('//test_y_axis_vertices.npy')
    posedirs_npy_path = bpy.path.abspath('//test_posedirs.npy')

    smpl_file_path = bpy.path.abspath('//smpl_ATTA.pkl')

    smpl_dict["v_template"] = export_vertices_to_npy(obj, vertices_npy_path)[1]
    smpl_dict["f"] = export_faces_to_npy(obj, faces_npy_path)[1]
    smpl_dict["weights"] = export_vertex_groups_to_npy(obj, vertex_groups_npy_path)[1]
    smpl_dict["sym_verts"] = export_y_axis_vertices_to_npy(obj, y_axis_vertices_npy_path)[1]

    num_verts = smpl_dict["v_template"].shape[0]
    num_joints = smpl_dict["weights"].shape[1]

    print("Vertices: ", num_verts)
    print("Joints: ", num_joints)

    smpl_dict["shapedirs"] = np.zeros((num_verts, 3))

    smpl_dict["posedirs"] = np.empty(0)  # leave empty if there are no pose blend shapes

    print(smpl_dict["posedirs"].shape)

    armature_obj = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
    if not armature_obj:
        print("No armature object found.")
        return

    print("Found armature object:", armature_obj.name)

    smpl_dict["kintree_table"] = export_joint_hierarchy_to_npy(armature_obj, joint_hierarchy_npy_path)[1]
    smpl_dict["J"], smpl_dict["J_names"] = export_joint_locations_to_npy(armature_obj, joint_locations_npy_path)[1:]
    smpl_dict["J_regressor"] = export_J_regressor_to_npy(obj, armature_obj, 20, j_regressor_npy_path)[1]

    with open(smpl_file_path, "wb") as f:
        pickle.dump(smpl_dict, f)

    print(f"SMPL model exported successfully to {smpl_file_path}")


def load_pkl_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            print("\nReading in contents of SMPL file...")
            data = CustomUnpickler(f).load()
            data_de_chumpied = {k: np.array(v) if isinstance(v, CustomUnpickler.ChumpyWrapper) else v 
                              for k, v in data.items()}
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
    Apply pose-dependent corrective blend shapes based on current armature pose.
    
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
    bpy.ops.object.mode_set(mode='POSE')
    
    # Get pose bones (excluding root)
    pose_bones = armature.pose.bones[1:]  # Skip root bone
    
    # Store current vertex positions (these are the skinned positions without correctives)
    if "base_skinned_positions" not in obj:
        # Get current deformed vertex positions (from regular skinning)
        world_matrix = np.array(obj.matrix_world)
        world_matrix_inv = np.array(obj.matrix_world.inverted())
        base_skinned_positions = np.array([np.array(obj.matrix_world @ v.co) for v in obj.data.vertices])
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
    bpy.ops.object.mode_set(mode='OBJECT')
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
    print("Applied pose-dependent corrective blend shapes")


def create_mesh_from_pkl(data):
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
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
    armature = bpy.context.object
    armature.name = "SMPL_Armature"
    armature.show_in_front = True

    # Add bones based on hierarchy
    bones = []
    for i, (parent_idx, child_idx, bone_name) in enumerate(zip(kintree_table[0], kintree_table[1], joint_names)):
        print(bone_name)
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

    bpy.ops.object.mode_set(mode='OBJECT')

    # Parent mesh to armature
    obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE')

    # Assign vertex weights
    for i, vertex_weights in enumerate(weights):
        for j, (weight, bone_name) in enumerate(zip(vertex_weights, joint_names)):
            if weight > 0:
                vertex_group = obj.vertex_groups.get(bone_name)
                if vertex_group is None:
                    vertex_group = obj.vertex_groups.new(name=bone_name)
                vertex_group.add([i], weight, 'ADD')


def create_blendshapes(data, obj):
    """
    Create blendshapes from deformation vertices in the new mesh object.

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
    
    # Here, as the individual blendshapes are entirely independent of each other,
    # the covariance matrix is simply a [n, n] identity matrix     
    num_shapes = deform_verts.shape[0]   
    cov = np.eye(num_shapes)
    print(cov.shape)
    # Likewise, the mean_betas are 1/n for all shapes
    mean_betas = np.ones(num_shapes) / num_shapes

    print(f"Created {len(deform_verts)} blendshapes.")
    return cov, mean_betas


def apply_pca_and_create_blendshapes(scans, obj, num_components=10, overwrite_mesh=False, std_range=1):
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
        # Add the mean shape as a blendshape
        if not obj.data.shape_keys:
            obj.shape_key_add(name="Basis")
        shape_key = obj.data.shape_keys.key_blocks["Basis"]
        for vert_index, vert in enumerate(mean_shape):
            shape_key.data[vert_index].co = vert

    # Principal components (reshape each component back to (v, 3))
    blendshapes = [component.reshape(v, 3) for component in pca.components_]

    # Standard deviations of the principal components
    std_devs = np.sqrt(pca.explained_variance_)

    # Add blendshapes as shape keys with min and max range
    for i, (blendshape, std_dev) in enumerate(zip(blendshapes, std_devs)):
        shape_key_name = f"PC_{i + 1}"
        shape_key = obj.shape_key_add(name=shape_key_name)

        # Calculate min and max range for the shape key
        min_range = -std_range * std_dev
        max_range = std_range * std_dev

        # Update the shape key vertex positions
        for j, vertex in enumerate(blendshape):
            shape_key.data[j].co = mean_shape[j] + vertex

        # Set min and max range for the shape key
        shape_key.slider_min = min_range
        shape_key.slider_max = max_range

    print(f"Created {num_components} PCA blendshapes with custom min and max ranges based on standard deviations.")

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
    assert vertex_positions.shape[0] == n, "Number of vertices in vertex positions and weights must match."

    # Initialize the joint positions array
    joint_positions = np.zeros((j, 3))

    # Calculate the position of each joint
    for i in range(j):
        joint_positions[i] = np.sum(J_regressor[i, :, None] * vertex_positions, axis=0)

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
    joint_positions = recalculate_joint_positions(vertex_positions=vertex_positions, 
                                                  J_regressor=pkl_data["J_regressor"])

    # Update the armature with the new joint positions
    armature = bpy.data.objects.get("SMPL_Armature")
    if not armature:
        print("SMPL_Armature not found.")
        return

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')

    for i, bone in enumerate(armature.data.edit_bones):
        bone.head = joint_positions[i]
        # the bone tails all point upwards and bones are of equal length
        bone.tail = joint_positions[i] + [0, 0, 0.1]

    bpy.ops.object.mode_set(mode='OBJECT')
    print("Joint positions recalculated and updated.")


def compute_symmetric_pairs(vertices, axis='y', tolerance=0.01):
    """
    Compute symmetric pairs of vertices based on their coordinates and the specified symmetry axis.
    Allow for a specified percentage deviation (tolerance) from the exact mirrored position using KDTree.
    """
    sym_pairs = []
    sym_axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
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


def rebuild_symmetry_array(vertices_on_symmetry_axis, all_vertices, axis='y', tolerance=0.001):
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
        print(f"Error enforcing symmetry: Unequal number of vertices on left ({len(left_inds)})", 
              f"and right ({len(right_inds)}) sides. This may indicate an asymmetric mesh or",
              f"incorrect symmetry axis.")

    symIdx = rebuild_symmetry_array(vertices_on_symmetry_axis=I,
                                    all_vertices=v, axis='y',
                                    tolerance=0.001)

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
            new_position = Vector([corresponding_vertex.co.x, -corresponding_vertex.co.y, corresponding_vertex.co.z])
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
                    shape_vertex.co = Vector([corresponding_shape_vertex.co.x, -corresponding_shape_vertex.co.y,
                                              corresponding_shape_vertex.co.z])
                else:
                    shape_vertex.co = shape_vertex.co

        # Update the mesh to reflect the changes
        obj.data.update()


def cleanup_mesh(obj, center_tolerance=0.005):
    """
    Cleans up the mesh by merging vertices close to the symmetry axis
    and recalculating normals. Applies the same cleanup to all blendshapes.
    Removes all interior faces.
    """
    # Ensure we're working on the correct object
    bpy.context.view_layer.objects.active = obj

    # Apply the cleanup for the base mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')

    # Select vertices with y coordinate close to 0 in the base mesh
    bpy.ops.object.mode_set(mode='OBJECT')
    for vertex in obj.data.vertices:
        if abs(vertex.co.y) < center_tolerance:
            vertex.select = True

    # Merge selected vertices by distance in the base mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(threshold=center_tolerance)

    # Recalculate mesh normals for the base mesh
    bpy.ops.mesh.normals_make_consistent(inside=False)

    # Ensure that the base mesh cleanup is applied before moving to blendshapes
    bpy.ops.object.mode_set(mode='OBJECT')

    # Remove interior faces
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.fill_holes(sides=0)
    bpy.ops.mesh.select_interior_faces()
    bpy.ops.mesh.delete(type='FACE')

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')


def export_smpl_model(obj, pkl_data, export_path):
    """
    Export the updated model as a new SMPL file with the blendshapes stored in the model's shapedirs.

    Args:
    - obj (bpy.types.Object): The mesh object with the updated vertex locations and blendshapes.
    - pkl_data (dict): Dictionary containing the original SMPL data.
    - export_path (str): The file path where the new SMPL file will be saved.
    """
    # Update "v_template" with the newly computed vertex locations of the mesh
    updated_vertices = np.array([np.array(v.co) for v in obj.data.vertices])
    print(pkl_data["v_template"].shape, updated_vertices.shape)
    pkl_data["v_template"] = updated_vertices

    # update all changed elements due to topoly changes

    vertices_npy_path = bpy.path.abspath('//test_vertices.npy')
    faces_npy_path = bpy.path.abspath('//test_faces.npy')
    vertex_groups_npy_path = bpy.path.abspath('//test_vertex_groups.npy')
    joint_locations_npy_path = bpy.path.abspath('//test_joint_locations.npy')
    j_regressor_npy_path = bpy.path.abspath('//test_joint_regressor.npy')
    y_axis_vertices_npy_path = bpy.path.abspath('//test_y_axis_vertices.npy')

    pkl_data["f"] = export_faces_to_npy(obj, faces_npy_path)[1]
    print(pkl_data["weights"].shape)
    pkl_data["weights"] = export_vertex_groups_to_npy(obj, vertex_groups_npy_path)[1]
    print(pkl_data["weights"].shape)
    pkl_data["sym_verts"] = export_y_axis_vertices_to_npy(obj, y_axis_vertices_npy_path)[1]

    armature_obj = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
    if not armature_obj:
        print("No armature object found.")
        return

    print("Found armature object:", armature_obj.name)

    pkl_data["J"], pkl_data["J_names"] = export_joint_locations_to_npy(armature_obj, joint_locations_npy_path)[1:]
    pkl_data["J_regressor"] = export_J_regressor_to_npy(obj, armature_obj, 20, j_regressor_npy_path)[1]

    # Update "shapedirs" with the content of the blendshapes
    num_blendshapes = len(obj.data.shape_keys.key_blocks) - 1  # Exclude the "Basis" shape key
    num_vertices = len(updated_vertices)
    shapedirs = np.zeros((num_vertices, 3, num_blendshapes))  # add 1 for one base blendshape

    for i, shape_key in enumerate(obj.data.shape_keys.key_blocks[1:], start=0):  # Exclude the "Basis" shape key
        for j, vert in enumerate(shape_key.data):
            shapedirs[j, :, i] = np.array(vert.co) - updated_vertices[j]

    pkl_data["shapedirs"] = shapedirs
    print(shapedirs.shape)

    # Write out the new pkl file to the same location as the input pkl file with the user-specified name
    output_path = os.path.join(os.path.dirname(export_path), bpy.context.scene.smpl_tool.output_filename)
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"New SMPL file saved successfully at {output_path}.")
    except Exception as e:
        print(f"Failed to save new SMPL file: {e}")


"""
GUI-ify
"""


class SMPL_PT_Panel(bpy.types.Panel):
    bl_label = "SMPL Model Importer"
    bl_idname = "SMPL_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SMPL'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        smpl_tool = scene.smpl_tool

        layout.prop(smpl_tool, "pkl_filepath")
        layout.prop(smpl_tool, "npz_filepath")
        layout.prop(smpl_tool, "blendshapes_from_PCA")
        layout.prop(smpl_tool, "number_of_PC")
        layout.prop(smpl_tool, "clean_mesh")
        layout.prop(smpl_tool, "merging_threshold")
        layout.prop(smpl_tool, "regress_joints")
        layout.prop(smpl_tool, "symmetrise")

        layout.operator("smpl.import_model", text="Import SMPL Model")
        
        # Add output filename field before export button
        layout.prop(smpl_tool, "output_filename")
        layout.operator("smpl.export_model", text="Export SMPL Model")
        
        # Add section for pose correctives
        layout.separator()
        layout.label(text="Apply corrective blend shapes:")
        # Add note about pose correctives availability
        box = layout.box()
        box.label(text="Note: Only available when pose correctives are provided via posedirs", icon='INFO')
        layout.operator("smpl.apply_pose_correctives", text="Apply Pose Correctives")


def store_smpl_data(context, data):
    """Store SMPL data in a temporary file and save the path"""
    obj = context.active_object
    if obj:
        # Create a temporary file to store the data
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"smpl_data_{obj.name}.pkl")
        
        # Save the data to the temporary file
        with open(temp_path, 'wb') as f:
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
            with open(temp_path, 'rb') as f:
                return pickle.load(f)
    return None

class SMPL_OT_ImportModel(bpy.types.Operator):
    bl_idname = "smpl.import_model"
    bl_label = "Import SMPL Model"

    def execute(self, context):
        scene = context.scene
        smpl_tool = scene.smpl_tool

        try:
            pkl_filepath = bpy.path.abspath(smpl_tool.pkl_filepath)
            data = load_pkl_file(pkl_filepath)
            if data:
                obj = create_mesh_from_pkl(data)
                if obj:
                    # Store SMPL data in the object
                    store_smpl_data(context, data)
                    
                    create_armature_and_weights(data, obj)
                    if not smpl_tool.npz_filepath:
                        self.report({'INFO'}, "No .npz file provided, skipping blendshape creation.")
                        return {'FINISHED'}
                        
                    npz_filepath = bpy.path.abspath(smpl_tool.npz_filepath)
                    if not os.path.exists(npz_filepath):
                        self.report({'INFO'}, "Could not find .npz file, skipping blendshape creation.")
                        return {'FINISHED'}
                    
                    npz_data = load_npz_file(npz_filepath)
                    verts_data = npz_data['verts']

                    if verts_data.shape[1] != len(obj.data.vertices):
                        self.report({'ERROR'}, "Vertex count mismatch.")
                        return {'CANCELLED'}

                    if smpl_tool.blendshapes_from_PCA:
                        cov, mean_betas = apply_pca_and_create_blendshapes(verts_data, 
                                                                           obj, 
                                                                           smpl_tool.number_of_PC, 
                                                                           overwrite_mesh=True)
                                                                               
                    else:
                        cov, mean_betas = create_blendshapes(npz_data, obj)
                        
                    data["shape_cov"] = cov
                    data["shape_mean_betas"] = mean_betas

                    if smpl_tool.symmetrise:
                        make_symmetrical(obj, data)

                    if smpl_tool.regress_joints:
                        apply_updated_joint_positions(obj, data)

                    if smpl_tool.clean_mesh:
                        cleanup_mesh(obj, center_tolerance=smpl_tool.merging_threshold)

                    self.report({'INFO'}, "SMPL Model imported successfully.")
                    return {'FINISHED'}
                else:
                    self.report({'ERROR'}, "Failed to create mesh from .pkl file.")
                    return {'CANCELLED'}
            else:
                self.report({'ERROR'}, "Failed to load .pkl file.")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to import SMPL Model: {e}")
            return {'CANCELLED'}


class SMPL_OT_ExportModel(bpy.types.Operator):
    bl_idname = "smpl.export_model"
    bl_label = "Export SMPL Model"

    def execute(self, context):
        # Get SMPL data from the active object
        data = get_smpl_data(context)
        if data is None:
            self.report({'ERROR'}, "No SMPL model data found. Please import a SMPL model first.")
            return {'CANCELLED'}

        scene = context.scene
        smpl_tool = scene.smpl_tool

        try:
            obj = bpy.context.active_object
            if not obj or obj.type != 'MESH':
                self.report({'ERROR'}, "No valid mesh object selected.")
                return {'CANCELLED'}

            export_smpl_model(obj, data, export_path=bpy.path.abspath(smpl_tool.pkl_filepath))

            self.report({'INFO'}, "SMPL Model exported successfully.")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to export SMPL Model: {str(e)}")
            return {'CANCELLED'}


class SMPLProperties(bpy.types.PropertyGroup):
    pkl_filepath: bpy.props.StringProperty(
        name="PKL Filepath",
        description="Path to the .pkl file",
        default="",
        subtype='FILE_PATH'
    )

    npz_filepath: bpy.props.StringProperty(
        name="NPZ Filepath",
        description="Path to the .npz file",
        default="",
        subtype='FILE_PATH'
    )

    blendshapes_from_PCA: bpy.props.BoolProperty(
        name="Blendshapes from PCA",
        description="Generate blendshapes from PCA",
        default=True
    )

    number_of_PC: bpy.props.IntProperty(
        name="Number of Principal Components",
        description="Number of principal components for PCA",
        default=20
    )

    regress_joints: bpy.props.BoolProperty(
        name="Regress Joints",
        description="Regress joint positions",
        default=True
    )

    clean_mesh: bpy.props.BoolProperty(
        name="Auto Clean-up Mesh",
        description="Merges overlapping vertices and removes inward facing faces",
        default=True
    )

    merging_threshold: bpy.props.FloatProperty(
        name="Minimal vertex distance",
        description="Minimal distance between vertices on centre line during mesh cleanup",
        default=0.001
    )

    symmetrise: bpy.props.BoolProperty(
        name="Symmetrise",
        description="Symmetrise the model",
        default=True
    )
    
    # Add properties to store SMPL data
    has_smpl_data: bpy.props.BoolProperty(default=False)
    v_template: bpy.props.FloatVectorProperty(size=3)  # This will store the shape
    posedirs: bpy.props.FloatVectorProperty(size=3) # This will store the pose correctives
    
    # Add to SMPLProperties class:
    output_filename: bpy.props.StringProperty(
        name="Output Filename",
        description="Name of the output SMPL model file",
        default="SMPL_fit.pkl"
    )

class SMPL_OT_ApplyPoseCorrectivesOperator(bpy.types.Operator):
    bl_idname = "smpl.apply_pose_correctives"
    bl_label = "Apply Pose Correctives"
    bl_description = "Apply pose-dependent corrective blend shapes based on current armature pose"
    
    @classmethod
    def poll(cls, context):
        # Only enable if we have an active mesh object with an armature
        obj = context.active_object
        if not (obj and obj.type == 'MESH' and obj.find_armature() and "has_smpl_data" in obj and "smpl_data_path" in obj):
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
                self.report({'ERROR'}, "No SMPL data found. Please import a SMPL model first.")
                return {'CANCELLED'}
            
            apply_pose_correctives(obj, data["posedirs"], data["v_template"])
            self.report({'INFO'}, "Applied pose correctives successfully.")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to apply pose correctives: {str(e)}")
            return {'CANCELLED'}


def get_joint_distances(armature_obj):
    """Calculate distances between all joint pairs in the armature."""
    joints = armature_obj.data.bones
    distances = []
    
    # Calculate distances between all joint pairs
    for i, bone1 in enumerate(joints):
        for j, bone2 in enumerate(joints[i+1:], i+1):
            dist = (bone1.head_local - bone2.head_local).length
            distances.append([bone1.name, bone2.name, dist])
            
    return distances

def get_joint_distances_from_positions(joint_positions, joint_names):
    """Calculate distances between all joint pairs from joint positions."""
    distances = []
    
    # Calculate distances between all joint pairs
    for i, pos1 in enumerate(joint_positions):
        for j, pos2 in enumerate(joint_positions[i+1:], i+1):
            dist = np.linalg.norm(pos1 - pos2)
            distances.append([joint_names[i], joint_names[j], dist])
            
    return distances

def export_joint_distances(context, filepath):
    """Export joint distances to a CSV file, including distances for each shape key."""
    armature = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
    if not armature:
        return False, "No armature found"
    
    mesh_obj = context.active_object
    if not mesh_obj or mesh_obj.type != 'MESH':
        return False, "No mesh object selected"
    
    # Get joint names from armature
    joint_names = [bone.name for bone in armature.data.bones]
    
    # Get base mesh distances
    base_distances = get_joint_distances(armature)
    
    # Prepare data for CSV
    all_data = [['Base'] + row for row in base_distances]
    
    # Recalculate J_regressor for current mesh state
    # This ensures it works even if mesh topology has changed
    _, J_regressor = export_J_regressor_to_npy(mesh_obj, armature, 20)
    
    # Get distances for each shape key
    if mesh_obj.data.shape_keys and len(mesh_obj.data.shape_keys.key_blocks) > 1:
        # Store original values
        original_values = {}
        for key in mesh_obj.data.shape_keys.key_blocks[1:]:  # Skip basis
            original_values[key.name] = key.value
            key.value = 0.0
        
        # Update mesh to ensure we start from basis
        mesh_obj.data.update()
        
        # For each shape key
        for key in mesh_obj.data.shape_keys.key_blocks[1:]:  # Skip basis
            # Set this shape key to 1.0
            key.value = 1.0
            mesh_obj.data.update()
            
            # Get vertex positions with this shape key applied
            vertex_positions = np.array([np.array(v.co) for v in mesh_obj.data.vertices])
            
            # Calculate joint positions using J_regressor
            joint_positions = recalculate_joint_positions(vertex_positions, J_regressor)
            
            # Calculate distances between joints
            key_distances = get_joint_distances_from_positions(joint_positions, joint_names)
            
            # Add to data with shape key name
            for i, dist_data in enumerate(key_distances):
                all_data.append([key.name] + dist_data)
            
            # Reset this shape key
            key.value = 0.0
            mesh_obj.data.update()
        
        # Restore original values
        for key_name, value in original_values.items():
            mesh_obj.data.shape_keys.key_blocks[key_name].value = value
        mesh_obj.data.update()
    
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Shape', 'Joint1', 'Joint2', 'Distance'])
            writer.writerows(all_data)
        return True, f"Distances exported to {filepath}"
    except Exception as e:
        return False, f"Failed to export distances: {str(e)}"

# Add new operator classes

class SMPL_OT_ExportJointDistances(bpy.types.Operator):
    bl_idname = "smpl.export_joint_distances"
    bl_label = "Export Joint Distances"
    bl_description = "Export distances between all joints to a CSV file"
    
    @classmethod
    def poll(cls, context):
        return any(obj.type == 'ARMATURE' for obj in bpy.data.objects)
    
    def execute(self, context):
        # Generate filename based on active mesh
        mesh_obj = context.active_object
        if mesh_obj and mesh_obj.type == 'MESH':
            filename = f"{mesh_obj.name}_joint_distances.csv"
        else:
            filename = "joint_distances.csv"
            
        filepath = os.path.join(os.path.dirname(bpy.data.filepath), filename)
        
        success, message = export_joint_distances(context, filepath)
        if success:
            self.report({'INFO'}, message)
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

# Add new panel class

class SMPL_PT_MorphometryPanel(bpy.types.Panel):
    bl_label = "SMAL Morphometry"
    bl_idname = "SMPL_PT_MorphometryPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SMPL'
    
    def draw(self, context):
        layout = self.layout
        
        # Add measurement export buttons
        box = layout.box()
        box.label(text="Export Measurements:")
        
        # Show shape key count if available
        obj = context.active_object
        if obj and obj.type == 'MESH' and obj.data.shape_keys:
            shape_key_count = len(obj.data.shape_keys.key_blocks) - 1  # Exclude basis
            if shape_key_count > 0:
                box.label(text=f"Will include measurements for {shape_key_count} shape keys", icon='SHAPEKEY_DATA')
        
        box.operator("smpl.export_joint_distances", text="Joint Distances")
        box.operator("smpl.export_mesh_measurements", text="Surface Area & Volume")

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
    if not obj or obj.type != 'MESH':
        return False, "No mesh object selected"
    
    try:
        # Prepare data for CSV
        all_data = []
        
        # Calculate base measurements
        surface_area, volume = calculate_mesh_measurements(obj)
        all_data.append(['Base', 'Surface Area', surface_area])
        all_data.append(['Base', 'Volume', volume])
        
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
                    [p.vertices[:] for p in eval_obj.data.polygons]
                )
                temp_mesh.update()
                
                # Calculate measurements on the temporary object
                key_surface_area, key_volume = calculate_mesh_measurements(temp_obj)
                
                # Add to data with shape key name
                all_data.append([key.name, 'Surface Area', key_surface_area])
                all_data.append([key.name, 'Volume', key_volume])
            
            # Remove temporary object
            bpy.data.objects.remove(temp_obj)
            bpy.data.meshes.remove(temp_mesh)
            
            # Restore original values
            for key_name, value in original_values.items():
                obj.data.shape_keys.key_blocks[key_name].value = value
            obj.data.update()
            context.view_layer.update()
        
        # Export to CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Shape', 'Measurement', 'Value'])
            writer.writerows(all_data)
            
        return True, f"Measurements exported to {filepath}"
    except Exception as e:
        return False, f"Failed to export measurements: {str(e)}"

class SMPL_OT_ExportMeshMeasurements(bpy.types.Operator):
    bl_idname = "smpl.export_mesh_measurements"
    bl_label = "Export Mesh Measurements"
    bl_description = "Export surface area and volume measurements to a CSV file"
    
    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'
    
    def execute(self, context):
        # Generate filename based on active mesh
        mesh_obj = context.active_object
        filename = f"{mesh_obj.name}_measurements.csv"
        filepath = os.path.join(os.path.dirname(bpy.data.filepath), filename)
        
        success, message = export_mesh_measurements(context, filepath)
        if success:
            self.report({'INFO'}, message)
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

# Update the classes tuple to include new classes
classes = (
    SMPL_PT_Panel,
    SMPL_PT_MorphometryPanel,  # Add the new panel
    SMPL_OT_ImportModel,
    SMPL_OT_ExportModel,
    SMPL_OT_ApplyPoseCorrectivesOperator,
    SMPL_OT_ExportJointDistances,  # Add the new operators
    SMPL_OT_ExportMeshMeasurements,  # Add the new operator
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
    
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.smpl_tool


if __name__ == "__main__":
    register()
