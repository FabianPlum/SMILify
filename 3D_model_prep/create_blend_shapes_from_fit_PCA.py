import bpy
import numpy as np
import pickle
import os
from scipy.spatial import KDTree
from mathutils import Vector

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

./python3.11 -m pip install matplotlib scikit-learn --target /home/fabi/Downloads/blender-4.2.0-linux-x64/4.2/python/lib/python3.11/site-packages

# if you are very bored, implement this with subprocess
"""

# User settings
blendshapes_from_PCA = True
number_of_PC = 20
USE_DEFORM = False
REGRESS_JOINTS = True
EXPORT_MODEL = True
SYMMETRISE = True

"""
# Windows -> can use local paths
# Update these file paths to your actual file locations
pkl_filepath = "smpl_ATTA.pkl"
npz_filepath = "../fit3d_results/Stage3.npz"

"""
# Ubuntu -> use absolute paths
# Update these file paths to your actual file locations
pkl_filepath = "/home/fabi/dev/SMILify/3D_model_prep/smpl_ATTA.pkl"
# ATTA ONLY
npz_filepath = "/home/fabi/dev/SMILify/fit3d_results_ref/Stage_3_deform_fine.npz"
# ALL SPECIES
#npz_filepath = "/home/fabi/dev/SMILify/Fitter_RESULTS/fit3d_results_ALL_ANTS_ALL_METHODS/Stage3.npz"


try:
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
except:
    print("nWARNING: Module sklearn not found!" + 
          "Un-comment pip install at the top of the script!\n")
    blendshapes_from_PCA = False

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
    
    if USE_DEFORM:
        deform_verts = data["deform_verts"]
    else:
        deform_verts = data["verts"]
    target_shape_names = data["labels"]

    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis")

    for i, deform in enumerate(deform_verts):
        shape_key_name = target_shape_names[i]
        shape_key = obj.shape_key_add(name=shape_key_name)
        for vert_index, vert in enumerate(deform):
            if USE_DEFORM:
                shape_key.data[vert_index].co = np.array(obj.data.vertices[vert_index].co) + vert 
            else:
                shape_key.data[vert_index].co = vert

    print(f"Created {len(deform_verts)} blendshapes.")

            
def apply_pca_and_create_blendshapes(scans, obj, num_components=10, overwrite_mesh=False, std_range=2):
    if USE_DEFORM:
        base_mesh = np.array([np.array(i.co) for i in obj.data.vertices])
        scans += base_mesh
    
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
    print(pkl_data["J"].shape, joint_positions.shape)

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


def main(pkl_filepath, npz_filepath):        
    """ Main function to execute the workflow. """
    pkl_data = load_pkl_file(pkl_filepath)
    if pkl_data:
        obj = create_mesh_from_pkl(pkl_data)
        if obj:
            create_armature_and_weights(pkl_data, obj)
            try:
                npz_data = load_npz_file(npz_filepath)
                if USE_DEFORM:
                    verts_data = npz_data['deform_verts']
                else:
                    verts_data = npz_data['verts']
                
                if verts_data.shape[1] != len(obj.data.vertices):
                    print(f"Error: Vertex count mismatch between mesh ({len(obj.data.vertices)}) and deformation data ({verts_data.shape[1]}).")
                    return
                
                if blendshapes_from_PCA:
                    apply_pca_and_create_blendshapes(verts_data, obj, number_of_PC, overwrite_mesh=True)
                else:
                    create_blendshapes(npz_data, obj)
                
                if SYMMETRISE:
                    make_symmetrical(obj, pkl_data)
                
                if REGRESS_JOINTS:
                    joint_positions = recalculate_joint_positions(obj, pkl_data)
                else:
                    joint_positions = pkl_data["J"]
                
                # clean up the mesh at the end of the process
                # otherwise the weights and J_regressor are incorrect
                cleanup_mesh(obj, center_tolerance=0.005)
                
                # Afterwards update the weight and regressors to ensure consistent
                # vertex IDs and array shapes. This is handled in export_smpl_model()
                    
                if EXPORT_MODEL:
                    export_smpl_model(obj, pkl_data, export_path=pkl_filepath, joint_positions=joint_positions)
                 
            except Exception as e:
                print(f"Failed to load or process blendshapes data: {e}")
        else:
            print("Failed to create mesh from .pkl file.")
    else:
        print("Failed to load .pkl file.")

if __name__ == "__main__":
    main(pkl_filepath, npz_filepath)
