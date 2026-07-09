"""Build Blender mesh/armature/shape-keys from model data and export."""

import os
import pickle

import bpy
import numpy as np

from .core_mesh import export_J_regressor_to_npy, export_faces_to_npy, export_joint_hierarchy_to_npy, export_joint_locations_to_npy, export_vertex_groups_to_npy, export_y_axis_vertices_to_npy
from .measurements import sort_shape_keys
from .unpickler import CustomUnpickler

def load_pkl_file(filepath):
    try:
        with open(filepath, "rb") as f:
            print("\nReading in contents of SMPL file...")
            data = CustomUnpickler(f).load()
            data_de_chumpied = {
                k: np.array(v) if isinstance(v, CustomUnpickler.ChumpyWrapper) else v for k, v in data.items()
            }
            print("\nContents of loaded SMPL file:")
            for key in data_de_chumpied:
                print(key)
                try:
                    if type(data_de_chumpied[key]) is not str:
                        print(data_de_chumpied[key].shape)
                except Exception:
                    try:
                        print(len(data_de_chumpied[key]))
                    except Exception:
                        # if it's not a numpy array or a list, just print the type
                        print(type(data_de_chumpied[key]))
        print("Loaded .pkl file successfully.")

        try:
            # Check for new morph PCA entries
            if "scaledirs" in data_de_chumpied:
                print(f"Found scaledirs with shape: {data_de_chumpied['scaledirs'].shape}")
            if "transdirs" in data_de_chumpied:
                print(f"Found transdirs with shape: {data_de_chumpied['transdirs'].shape}")
        except Exception:
            print("No valid scaledirs or transdirs found.")

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
            print("First vertex:")
            print(f"  Base position: {base_vertices[idx]}")
            print(f"  Base skinned position: {skinned_pos}")
            print(f"  Pose offset: {offset}")
            print(f"  Final position: {final_pos}")
            print(f"  Local position: {local_pos[:3]}")

    obj.data.update()
    print("Applied pose-dependent corrective shape keys")


def create_mesh_from_pkl(data, base_name="SMPL"):
    # read in the .pkl file with mesh data stored similar to obj files
    # (tris triplets and faces with vertex indices)
    if "v_template" not in data or "f" not in data:
        print("No 'verts' or 'faces' key found in the .pkl file.")
        return None

    verts = data["v_template"]
    faces = data["f"]

    mesh = bpy.data.meshes.new(name=f"{base_name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name=base_name, object_data=mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return obj


def create_armature_and_weights(data, obj, base_name="SMPL"):
    """
    Create an armature based on the joint locations and assign weights to the mesh vertices.

    Args:
    - data (dict): Dictionary containing the contents of the .pkl file.
    - obj (bpy.types.Object): The newly created mesh object.
    - base_name (str): Base name used for armature data and object (defaults to "SMPL").

    Returns:
    - bpy.types.Object: The created armature object, or None on failure.
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
    armature.name = f"{base_name}_Armature"
    armature.data.name = f"{base_name}_Armature"
    armature.show_in_front = True

    # Add bones based on hierarchy
    bones = []
    for i, (parent_idx, child_idx, bone_name) in enumerate(zip(kintree_table[0], kintree_table[1], joint_names)):
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

    return armature


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


def create_shapekeys_from_pkl_shapedirs(data, obj):
    """
    Create shapekeys from shapedirs stored in pkl data.

    Args:
    - data (dict): Dictionary containing pkl data with 'shapedirs'
    - obj (bpy.types.Object): The mesh object

    Returns:
    - tuple: (cov, mean_betas) covariance matrix and mean betas
    """
    if "shapedirs" not in data:
        print("No 'shapedirs' key found in the pkl file.")
        return None, None

    shapedirs = data["shapedirs"]

    # shapedirs has shape (num_vertices, 3, num_shapekeys)
    # Check if shapedirs is not empty
    if shapedirs.size == 0:
        print("shapedirs is empty.")
        return None, None

    if len(shapedirs.shape) != 3:
        print(f"Unexpected shapedirs shape: {shapedirs.shape}. Expected (V, 3, K).")
        return None, None

    num_vertices, _, num_shapekeys = shapedirs.shape

    # Get base vertex positions
    base_vertices = np.array([np.array(v.co) for v in obj.data.vertices])

    # Verify vertex count matches
    if base_vertices.shape[0] != num_vertices:
        print(f"Vertex count mismatch: mesh has {base_vertices.shape[0]} vertices, shapedirs expects {num_vertices}.")
        return None, None

    # Create basis shape key if it doesn't exist
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis")

    # PCA-derived betas can fall well outside Blender's default 0..1 slider range.
    # Widen the range so animation imports load weights verbatim instead of clipping.
    pkl_shape_key_slider_min = -10.0
    pkl_shape_key_slider_max = 10.0

    # Create shapekeys from shapedirs
    for i in range(num_shapekeys):
        shape_key_name = f"Shape_{i}"
        shape_key = obj.shape_key_add(name=shape_key_name)

        # Apply displacements from shapedirs
        for vert_index in range(num_vertices):
            displacement = shapedirs[vert_index, :, i]
            shape_key.data[vert_index].co = base_vertices[vert_index] + displacement

        shape_key.slider_min = pkl_shape_key_slider_min
        shape_key.slider_max = pkl_shape_key_slider_max
        shape_key.value = 0.0

    # Create covariance matrix and mean betas
    # For independent shapekeys, use identity matrix
    cov = np.eye(num_shapekeys)
    mean_betas = np.ones(num_shapekeys) / num_shapekeys

    print(f"Created {num_shapekeys} shapekeys from pkl shapedirs.")
    return cov, mean_betas


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
            "posedirs": np.empty(0),  # ignore for now as we currently don't have corrective shapekeys in our models
            "v_template": [],
            "shapedirs": [],
            "bs_type": "lrotmin",
            "sym_verts": [],
            "scaledirs": [],  # optional PCA components for joint scaling variation
            "transdirs": [],  # optional PCA components for joint translation variation
            "static_joint_locs": False,  # whether joint locations are static (False by default, overwritten by object property if set)
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
    pkl_data["weights"] = export_vertex_groups_to_npy(obj, vertex_groups_npy_path, clean_weights=clean_weights)[1]
    print(pkl_data["weights"].shape)
    pkl_data["sym_verts"] = export_y_axis_vertices_to_npy(obj, y_axis_vertices_npy_path)[1]
    print(pkl_data["sym_verts"].shape)

    armature_obj = obj.find_armature()
    if not armature_obj:
        print("No armature object found for the selected mesh.")
        return

    print("Found armature object:", armature_obj.name)

    pkl_data["kintree_table"] = export_joint_hierarchy_to_npy(armature_obj, joint_hierarchy_npy_path)[1]
    pkl_data["J"], pkl_data["J_names"] = export_joint_locations_to_npy(armature_obj, joint_locations_npy_path)[1:]

    # Check if model has static joint locations
    if obj.get("static_joint_locs", False) or bpy.context.scene.smpl_tool.force_static_joint_locs:
        # Keep J_regressor as all zeroes for static joint models
        num_joints = len(pkl_data["J"])
        num_vertices = len(obj.data.vertices)
        pkl_data["J_regressor"] = np.zeros((num_joints, num_vertices), dtype=np.float32)
        pkl_data["static_joint_locs"] = True
        print("Static joint locations: J_regressor kept as all zeroes (not recomputed)")
    else:
        # Get the selected J_regressor method from the scene
        smpl_tool = bpy.context.scene.smpl_tool
        pkl_data["J_regressor"] = export_J_regressor_to_npy(
            obj,
            armature_obj,
            10,
            j_regressor_npy_path,
            weights=pkl_data["weights"],
            kintree_table=pkl_data["kintree_table"],
            influence_type=smpl_tool.j_regressor_method,
        )

    # Update "shapedirs" with the content of the shapekeys
    num_vertices = len(updated_vertices)
    try:
        num_shapekeys = len(obj.data.shape_keys.key_blocks) - 1  # Exclude the "Basis" shape key
        shapedirs = np.zeros((num_vertices, 3, num_shapekeys))  # add 1 for one base shapekey
        for i, shape_key in enumerate(obj.data.shape_keys.key_blocks[1:], start=0):  # Exclude the "Basis" shape key
            for j, vert in enumerate(shape_key.data):
                shapedirs[j, :, i] = np.array(vert.co) - updated_vertices[j]
    except AttributeError:
        print("No shapekeys found.")
        shapedirs = np.zeros((num_vertices, 3))

    pkl_data["shapedirs"] = shapedirs
    print(shapedirs.shape)

    # Check if scaledirs and transdirs exist and include them in export
    # These will fail when a model is exported from a mesh for the first time, which is fine we just need to catch the error and continue
    try:
        if "scaledirs" in pkl_data:
            print(f"Including scaledirs in export with shape: {pkl_data['scaledirs'].shape}")
        if "transdirs" in pkl_data:
            print(f"Including transdirs in export with shape: {pkl_data['transdirs'].shape}")
    except Exception:
        print("No scaledirs or transdirs found.")

    # Write out the new pkl file to the same location as the input pkl file with the user-specified name
    output_path = os.path.join(os.path.dirname(export_path), bpy.context.scene.smpl_tool.output_filename)
    try:
        with open(output_path, "wb") as f:
            pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"New SMPL file saved successfully at {output_path}.")
    except Exception as e:
        print(f"Failed to save new SMPL file: {e}")

