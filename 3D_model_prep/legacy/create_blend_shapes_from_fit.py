import bpy
import numpy as np
import pickle

# Update these file paths to your actual file locations
pkl_filepath = "D:/SMAL/SMILify/3D_model_prep/smpl_ATTA.pkl"
npz_filepath = "D:/SMAL/SMILify/Fitter_RESULTS/fit3d_results_ALL_ANTS_ALL_METHODS/Stage3.npz"

def load_pkl_file(filepath):
    """
    Load the .pkl file containing the SMPL model.

    Args:
    - filepath (str): Path to the .pkl file.

    Returns:
    - dict: Dictionary containing the contents of the .pkl file.
    """
    try:
        with open(filepath, 'rb') as f:
            print("\nContents of loaded SMPL file:")
            data = pickle.load(f, encoding='latin1')
            for key in data:
                print(key)
        print("Loaded .pkl file successfully.")
        return data
    except Exception as e:
        print(f"Failed to load .pkl file: {e}")
        return None

def create_mesh_from_pkl(data):
    """
    Create a new mesh object from the vertices and faces in the SMPL .pkl data.

    Args:
    - data (dict): Dictionary containing the contents of the .pkl file.

    Returns:
    - bpy.types.Object: The newly created mesh object.
    """
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

def load_npz_file(filepath):
    """
    Load the .npz file and list its contents.

    Args:
    - filepath (str): Path to the .npz file.

    Returns:
    - dict: Dictionary containing the contents of the .npz file.
    """
    try:
        data = np.load(filepath)
        print("Loaded .npz file successfully.")
        print("Contents:")
        for key in data:
            print(key)
        return data
    except Exception as e:
        print(f"Failed to load .npz file: {e}")
        return None

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

def main(pkl_filepath, npz_filepath):
    """
    Main function to load the SMPL model and blendshapes and apply them to the mesh.

    Args:
    - pkl_filepath (str): Path to the .pkl file containing the SMPL model.
    - npz_filepath (str): Path to the .npz file containing the blendshapes.
    """
    pkl_data = load_pkl_file(pkl_filepath)
    if pkl_data:
        obj = create_mesh_from_pkl(pkl_data)
        if obj:
            create_armature_and_weights(pkl_data, obj)
            npz_data = load_npz_file(npz_filepath)
            if npz_data:
                create_blendshapes(npz_data, obj)
            else:
                print("Failed to load blendshapes.")
        else:
            print("Failed to create mesh from .pkl file.")
    else:
        print("Failed to load .pkl file.")

if __name__ == "__main__":
    main(pkl_filepath, npz_filepath)
