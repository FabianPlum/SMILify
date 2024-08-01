bl_info = {
    "name": "SMPL Model Importer",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Tool Shelf",
    "description": "Import, configure, and export SMPL models",
    "category": "Import-Export",
}

import bpy
import numpy as np
import pickle
import os
from scipy.spatial import KDTree
from mathutils import Vector
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
SMIL-ify
"""

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
    np.save(filepath, joint_locations)
    return filepath, joint_locations

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
    y_axis_vertices = np.array([i for i, vert in enumerate(mesh.vertices) if np.isclose(vert.co.y, 0.0, atol=1e-3)], dtype=int)
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
#@ensure_armature (careful, the mesh is the active object!)
def export_J_regressor_to_npy(mesh_obj, armature_obj, n, filepath):
    vertices, _ = mesh_to_numpy(mesh_obj)
    joints = armature_obj.data.bones
    joint_locations = np.array([bone.head_local for bone in joints], dtype=np.float32)
    nearest_indices, nearest_weights = find_nearest_neighbors(vertices, joint_locations, n)
    J_regressor = np.zeros((len(joints), len(vertices)), dtype=np.float32)
    for i in range(len(joints)):
        J_regressor[i, nearest_indices[i]] = nearest_weights[i]
    np.save(filepath, J_regressor)
    return filepath, J_regressor

"""
This is currently not supported.
The posedir created here captures the mesh shape at every frame
This is however not hwo the posedir is used in the original implementaiton and thus disabled for now.
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
    
    """ 
    #Let's be adventurous and remove the poses and instead add fun shapes 
    
    smpl_dict["posedirs"] = export_posedirs(obj, start_frame, stop_frame, posedirs_npy_path)[1]

    smpl_dict["shapedirs"] = np.zeros((num_verts, 3))
    
    """
    
    smpl_dict["shapedirs"] = np.zeros((num_verts, 3))

    smpl_dict["posedirs"] = np.empty(0) # leave empty if there are no pose blend shapes
    
    print(smpl_dict["posedirs"].shape)
    

    armature_obj = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
    if not armature_obj:
        print("No armature object found.")
        return

    print("Found armature object:", armature_obj.name)

    smpl_dict["kintree_table"] = export_joint_hierarchy_to_npy(armature_obj, joint_hierarchy_npy_path)[1]
    smpl_dict["J"] = export_joint_locations_to_npy(armature_obj, joint_locations_npy_path)[1]
    smpl_dict["J_regressor"] = export_J_regressor_to_npy(obj, armature_obj, 20, j_regressor_npy_path)[1]

    with open(smpl_file_path, "wb") as f:
        pickle.dump(smpl_dict, f)

    print(f"SMPL model exported successfully to {smpl_file_path}")

def load_pkl_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            print("\nReading in contents of SMPL file...")
            data = pickle.load(f, encoding='latin1')
            print("\nContents of loaded SMPL file:")
            for key in data:
                print(key)
                if type(data[key]) is not str:
                    print(data[key].shape)
        print("Loaded .pkl file successfully.")
        return data
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
    weights = data["weights"]
    kintree_table = data["kintree_table"]

    # Create armature
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
    armature = bpy.context.object
    armature.name = "SMPL_Armature"
    armature.show_in_front = True

    # Add bones based on hierarchy
    bones = []
    for i, (parent_idx, child_idx) in enumerate(zip(kintree_table[0], kintree_table[1])):
        bone = armature.data.edit_bones.new(f"Bone_{child_idx}")
        bone.head = joints[child_idx]
        bone.tail = joints[child_idx] + np.array([0, 0, 0.1])
        bones.append(bone)
        if parent_idx != -1:
            bone.parent = armature.data.edit_bones[f"Bone_{parent_idx}"]

    bpy.ops.object.mode_set(mode='OBJECT')

    # Parent mesh to armature
    obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE')

    # Assign vertex weights
    for i, vertex_weights in enumerate(weights):
        for j, weight in enumerate(vertex_weights):
            if weight > 0:
                vertex_group = obj.vertex_groups.get(f"Bone_{j}")
                if vertex_group is None:
                    vertex_group = obj.vertex_groups.new(name=f"Bone_{j}")
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

    print(f"Created {len(deform_verts)} blendshapes.")

            
def apply_pca_and_create_blendshapes(scans, obj, num_components=10, overwrite_mesh=False, std_range=2):
    n, v, _ = scans.shape
    # Reshape the scans into (n, v*3)
    scans_reshaped = scans.reshape(n, v * 3)
    
    # Perform PCA
    pca = PCA(n_components=num_components)
    pca.fit(scans_reshaped)
    
    # Mean shape
    mean_shape = pca.mean_.reshape(v, 3)
    
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
        shape_key_name = f"PC_{i+1}"
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
                                

def recalculate_joint_positions(obj, pkl_data):
    """
    Recalculate the positions of joints based on the mean shape and vertex weights.

    Args:
    - obj (bpy.types.Object): The mesh object with the updated mean shape.
    - pkl_data (dict): Dictionary containing the joint weights information from the .pkl file.
    """
    mean_shape = np.array([np.array(v.co) for v in obj.data.vertices])
    weights = pkl_data["J_regressor"]

    j, n = weights.shape
    assert mean_shape.shape[0] == n, "Number of vertices in mean shape and weights must match."
    
    # Initialize the joint positions array
    joint_positions = np.zeros((j, 3))
    
    # Calculate the position of each joint
    for i in range(j):
        joint_positions[i] = np.sum(weights[i, :, None] * mean_shape, axis=0)

    # Update the armature with the new joint positions
    armature = bpy.data.objects.get("SMPL_Armature")
    if not armature:
        print("SMPL_Armature not found.")
        return
    
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')

    for i, bone in enumerate(armature.data.edit_bones):
        bone.head = joint_positions[i]
        bone.tail = joint_positions[i] + [0,0,0.1]

    bpy.ops.object.mode_set(mode='OBJECT')
    print("Joint positions recalculated and updated.")
    
    return joint_positions

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
        print("WHOOPS")

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
                    shape_vertex.co = Vector([corresponding_shape_vertex.co.x, -corresponding_shape_vertex.co.y, corresponding_shape_vertex.co.z])
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


def export_smpl_model(obj, pkl_data, export_path, joint_positions):
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
    
    pkl_data["J"] = joint_positions
    
    # update all changed elements due to topoly changes
    
    vertices_npy_path = bpy.path.abspath('//test_vertices.npy')
    faces_npy_path = bpy.path.abspath('//test_faces.npy')
    vertex_groups_npy_path = bpy.path.abspath('//test_vertex_groups.npy')
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

    pkl_data["J_regressor"] = export_J_regressor_to_npy(obj, armature_obj, 20, j_regressor_npy_path)[1]

    # Update "shapedirs" with the content of the blendshapes
    num_blendshapes = len(obj.data.shape_keys.key_blocks) - 1  # Exclude the "Basis" shape key
    num_vertices = len(updated_vertices)
    shapedirs = np.zeros((num_vertices, 3, num_blendshapes + 1)) # add 1 for one base blendshape

    for i, shape_key in enumerate(obj.data.shape_keys.key_blocks[1:], start=0):  # Exclude the "Basis" shape key
        for j, vert in enumerate(shape_key.data):
            shapedirs[j, :, i + 1] = np.array(vert.co) - updated_vertices[j]

    pkl_data["shapedirs"] = shapedirs

    # Write out the new pkl file to the same location as the input pkl file with the name SMPL_fit.pkl
    output_path = os.path.join(os.path.dirname(export_path), "SMPL_fit.pkl")
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
        layout.prop(smpl_tool, "regress_joints")
        layout.prop(smpl_tool, "export_model")
        layout.prop(smpl_tool, "symmetrise")

        layout.operator("smpl.import_model", text="Import SMPL Model")
        layout.operator("smpl.export_model", text="Export SMPL Model")

class SMPL_OT_ImportModel(bpy.types.Operator):
    bl_idname = "smpl.import_model"
    bl_label = "Import SMPL Model"

    def execute(self, context):
        scene = context.scene
        smpl_tool = scene.smpl_tool

        try:
            # Ensure the file paths are absolute
            pkl_filepath = bpy.path.abspath(smpl_tool.pkl_filepath)
            npz_filepath = bpy.path.abspath(smpl_tool.npz_filepath)

            pkl_data = load_pkl_file(pkl_filepath)
            if pkl_data:
                obj = create_mesh_from_pkl(pkl_data)
                if obj:
                    create_armature_and_weights(pkl_data, obj)
                    npz_data = load_npz_file(npz_filepath)
                    verts_data = npz_data['verts']

                    if verts_data.shape[1] != len(obj.data.vertices):
                        self.report({'ERROR'}, "Vertex count mismatch.")
                        return {'CANCELLED'}

                    if smpl_tool.blendshapes_from_PCA:
                        apply_pca_and_create_blendshapes(verts_data, obj, smpl_tool.number_of_PC, overwrite_mesh=True)
                    else:
                        create_blendshapes(npz_data, obj)

                    if smpl_tool.symmetrise:
                        make_symmetrical(obj, pkl_data)

                    if smpl_tool.regress_joints:
                        recalculate_joint_positions(obj, pkl_data)

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
        scene = context.scene
        smpl_tool = scene.smpl_tool

        try:
            pkl_data = load_pkl_file(smpl_tool.pkl_filepath)
            obj = bpy.context.active_object
            if not obj or obj.type != 'MESH':
                self.report({'ERROR'}, "No valid mesh object selected.")
                return {'CANCELLED'}

            joint_positions = recalculate_joint_positions(obj, pkl_data)
            export_smpl_model(obj, pkl_data, export_path=smpl_tool.pkl_filepath, joint_positions=joint_positions)

            self.report({'INFO'}, "SMPL Model exported successfully.")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to export SMPL Model: {e}")
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

    export_model: bpy.props.BoolProperty(
        name="Export Model",
        description="Export the final model",
        default=True
    )

    symmetrise: bpy.props.BoolProperty(
        name="Symmetrise",
        description="Symmetrise the model",
        default=True
    )

classes = (
    SMPL_PT_Panel,
    SMPL_OT_ImportModel,
    SMPL_OT_ExportModel,
    SMPLProperties,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.smpl_tool = bpy.props.PointerProperty(type=SMPLProperties)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.smpl_tool

if __name__ == "__main__":
    register()

