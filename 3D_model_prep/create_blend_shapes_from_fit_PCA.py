import bpy
import numpy as np
import pickle

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
blendshapes_from_PCA = False
number_of_PC = 10
USE_DEFORM = False

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
npz_filepath = "/home/fabi/dev/SMILify/fit3d_results/Stage3.npz"
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
            data = pickle.load(f, encoding='latin1')
            print("\nContents of loaded SMPL file:")
            for key in data:
                print(key)
        print("Loaded .pkl file successfully.")
        return data
    except Exception as e:
        print(f"Failed to load .pkl file: {e}")
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


def apply_pca_and_create_blendshapes(scans, obj, num_components=10):
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
    
    # Principal components (reshape each component back to (v, 3))
    blendshapes = [component.reshape(v, 3) for component in pca.components_]
    
    shape_key = obj.shape_key_add(name="Basis")        
    for vert_index, vert in enumerate(mean_shape):
        shape_key.data[vert_index].co = vert
    
    # Add blendshapes as shape keys
    for i, blendshape in enumerate(blendshapes):
        shape_key = obj.shape_key_add(name=f"PC_{i+1}")        
        for j, vertex in enumerate(blendshape):
            shape_key.data[j].co = mean_shape[j] + vertex
                                
                
def apply_pca_and_create_blendshapes_ALT(data, obj, num_components=10):
    """ Apply PCA on deformation vertices and create blendshapes based on the principal components. """
    # Reshape data for PCA (flattening the vertex coordinates into a single dimension per sample)
    data_reshaped = data.reshape(-1, data.shape[0], order="F")

    # Initialize and fit PCA
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(data_reshaped)
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance by component:", explained_variance)

    if not obj.data.shape_keys:
        shape_key = obj.shape_key_add(name="Basis")
        # create new mean shape
        mean_shape = np.mean(data, axis=0)
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        ax.scatter(mean_shape[:,0], 
                   mean_shape[:,1], 
                   mean_shape[:,2])
        plt.savefig('mean_shape.png')
        plt.close()
        
        for vert_index, vert in enumerate(mean_shape):
            shape_key.data[vert_index].co = vert
    
    for i in range(num_components):
        pc_reshaped = principal_components[:, i].reshape(-1, 3, order="F")
        shape_key_name = f"PC_{i+1}"
        shape_key = obj.shape_key_add(name=shape_key_name)
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        
        print(np.abs(pc_reshaped / np.max(np.abs(pc_reshaped))))
        
        # use colouration of element to indicate magnitude of change for each vertex (regardless of direction)
        ax.scatter(mean_shape[:,0], 
                   mean_shape[:,1], 
                   mean_shape[:,2],
                   c=np.mean(np.abs(pc_reshaped / np.max(np.abs(pc_reshaped))), axis=1))
        
        print(shape_key_name +'.png')
        plt.savefig(shape_key_name +'.png')
        plt.close()
        
        for vert_index, vert in enumerate(pc_reshaped):
            shape_key.data[vert_index].co = vert 

    print(f"Created {num_components} PCA blendshapes based on explained variance.")


def main(pkl_filepath, npz_filepath):        
    """ Main function to execute the workflow. """
    pkl_data = load_pkl_file(pkl_filepath)
    if pkl_data:
        obj = create_mesh_from_pkl(pkl_data)
        if obj:
            create_armature_and_weights(pkl_data, obj)
            try:
                npz_data = np.load(npz_filepath, allow_pickle=True)
                if USE_DEFORM:
                    verts_data = npz_data['deform_verts']
                else:
                    verts_data = npz_data['verts']
                
                if verts_data.shape[1] != len(obj.data.vertices):
                    print(f"Error: Vertex count mismatch between mesh ({len(obj.data.vertices)}) and deformation data ({verts_data.shape[1]}).")
                    return
                
                if blendshapes_from_PCA:
                    apply_pca_and_create_blendshapes(verts_data, obj, number_of_PC)
                else:
                    create_blendshapes(npz_data, obj)
                    
            except Exception as e:
                print(f"Failed to load or process blendshapes data: {e}")
        else:
            print("Failed to create mesh from .pkl file.")
    else:
        print("Failed to load .pkl file.")

if __name__ == "__main__":
    main(pkl_filepath, npz_filepath)
