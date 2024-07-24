import bpy
import numpy as np
import pickle
from sklearn.decomposition import PCA
from mathutils import Vector

# User settings
blendshapes_from_PCA = True
number_of_PC = 10
# Update these file paths to your actual file locations
pkl_filepath = "/home/fabi/dev/SMILify/3D_model_prep/smpl_ATTA.pkl"
npz_filepath = "/home/fabi/dev/SMILify/fit3d_results/Stage3.npz"

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
                    
                
def apply_pca_and_create_blendshapes(data, obj, num_components=10):
    """ Apply PCA on deformation vertices and create blendshapes based on the principal components. """
    # Reshape data for PCA (flattening the vertex coordinates into a single dimension per sample)
    data_reshaped = data.reshape(-1, data.shape[0])

    # Initialize and fit PCA
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(data_reshaped)
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance by component:", explained_variance)

    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis")
    
    for i in range(num_components):
        pc_reshaped = principal_components[:, i].reshape(-1, 3)
        shape_key_name = f"PC_{i+1}"
        shape_key = obj.shape_key_add(name=shape_key_name)
        for vert_index, vert in enumerate(pc_reshaped):
            shape_key.data[vert_index].co = vert
        
    """

    # Create blendshapes for the first num_components principal components
    for i in range(num_components):
        shape_key_name = f"PC_{i+1}"
        shape_key = obj.shape_key_add(name=shape_key_name)
        pc_shape = principal_components[:, i].reshape(-1, 3)  # reshape to (v, 3)
        print(principal_components[:,i].shape)
        print(pc_shape.shape)

        # Ensure vertex count matches
        if len(obj.data.vertices) != pc_shape.shape[0]:
            print("Error: The number of vertices does not match the number of PCA component entries.")
            return
        
        for vert_index, vert in enumerate(obj.data.shape_keys.key_blocks["Basis"].data):
            shape_key.data[vert_index].co = vert
            # Convert NumPy array to Vector before adding
            additional_vector = Vector((pc_shape[vert_index][0], pc_shape[vert_index][1], pc_shape[vert_index][2]))
            vert.co += additional_vector
    
    """
         

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
                verts_data = npz_data['verts']
                # deform_data = npz_data ['deform_verts']
                
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
