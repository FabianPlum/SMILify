"""Blender operators for import/export/generation/animation."""

import os
import csv
import pickle
import tempfile

import bpy
import numpy as np
from mathutils import Vector

try:
    from sklearn.decomposition import PCA
except ImportError:  # pragma: no cover
    PCA = None

from . import state
from . import dependencies
from .core_mesh import apply_updated_joint_positions, cleanup_mesh, export_J_regressor_to_npy, make_symmetrical
from .measurements import export_joint_distances, export_mesh_measurements, get_smpl_data, load_reference_measurements, store_smpl_data
from .model_build import apply_pose_correctives, create_armature_and_weights, create_mesh_from_pkl, create_shapekeys, create_shapekeys_from_pkl_shapedirs, export_smpl_model, load_npz_file, load_pkl_file
from .pca import apply_entangled_pca_and_create_shapekeys, apply_pca_and_create_shapekeys
from .state import clear_morph_pca_globals

class SMPL_OT_ImportModel(bpy.types.Operator):
    bl_idname = "smpl.import_model"
    bl_label = "Import SMIL Model"

    def execute(self, context):
        if not dependencies.dependencies_installed():
            self.report({"ERROR"}, dependencies.MISSING_MESSAGE)
            return {"CANCELLED"}
        scene = context.scene
        smpl_tool = scene.smpl_tool

        try:
            pkl_filepath = bpy.path.abspath(smpl_tool.pkl_filepath)
            base_name = os.path.splitext(os.path.basename(pkl_filepath))[0] or "SMPL"
            data = load_pkl_file(pkl_filepath)
            if data:
                obj = create_mesh_from_pkl(data, base_name=base_name)
                if obj:
                    obj["SMIL_TYPE"] = "SMIL_model_from_direct_npz_import"

                    # Check if the loaded pkl has static_joint_locs set
                    if data.get("static_joint_locs", False):
                        obj["static_joint_locs"] = True
                        print("Loaded model with static joint locations")

                    # Store SMPL data in the object
                    store_smpl_data(context, data, obj=obj)

                    create_armature_and_weights(data, obj, base_name=base_name)

                    # Check if npz file is provided and exists
                    npz_filepath = bpy.path.abspath(smpl_tool.npz_filepath) if smpl_tool.npz_filepath else None
                    npz_exists = npz_filepath and os.path.exists(npz_filepath)

                    if npz_exists:
                        # Load from npz file (existing behavior with PCA option)
                        npz_data = load_npz_file(npz_filepath)
                        verts_data = npz_data["verts"]

                        if verts_data.shape[1] != len(obj.data.vertices):
                            self.report({"ERROR"}, "Vertex count mismatch.")
                            return {"CANCELLED"}

                        if smpl_tool.shapekeys_from_PCA:
                            output_dir = os.path.dirname(pkl_filepath)
                            labels = list(npz_data["labels"]) if "labels" in npz_data else None
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
                    else:
                        # No npz file - try to load shapekeys from pkl shapedirs
                        if "shapedirs" in data and data["shapedirs"].size > 0:
                            self.report(
                                {"INFO"},
                                "No .npz file provided, loading shapekeys from pkl shapedirs.",
                            )
                            cov, mean_betas = create_shapekeys_from_pkl_shapedirs(data, obj)

                            if cov is None or mean_betas is None:
                                self.report(
                                    {"WARNING"},
                                    "Failed to load shapekeys from pkl shapedirs.",
                                )
                                return {"FINISHED"}

                            # create_shapekeys_from_pkl_shapedirs returns trivial
                            # identity defaults — only useful when the pkl has no
                            # real PCA stats. If the pkl already carries learned
                            # shape_cov/shape_mean_betas, keep those.
                            existing_cov = data.get("shape_cov")
                            existing_mean = data.get("shape_mean_betas")
                            if isinstance(existing_cov, np.ndarray) and existing_cov.size > 0:
                                cov = existing_cov
                            if isinstance(existing_mean, np.ndarray) and existing_mean.size > 0:
                                mean_betas = existing_mean
                        else:
                            self.report(
                                {"INFO"},
                                "No .npz file or shapedirs data available, skipping shapekey creation.",
                            )
                            return {"FINISHED"}

                    data["shape_cov"] = cov
                    data["shape_mean_betas"] = mean_betas

                    # Handle static joint locations
                    if smpl_tool.force_static_joint_locs:
                        # Set J_regressor to all zeroes for static joint locations
                        num_joints = data["J"].shape[0]
                        num_vertices = len(obj.data.vertices)
                        data["J_regressor"] = np.zeros((num_joints, num_vertices), dtype=np.float32)
                        obj["static_joint_locs"] = True
                        data["static_joint_locs"] = True
                        print("Static joint locations enabled - J_regressor set to zeroes")

                    # Update the stored data with the new shape info
                    store_smpl_data(context, data, obj=obj)

                    if smpl_tool.symmetrise:
                        make_symmetrical(obj, data)

                    if smpl_tool.regress_joints:
                        # Skip joint regression for static joint models
                        if not obj.get("static_joint_locs", False):
                            apply_updated_joint_positions(obj, data)
                        else:
                            print("Skipping joint regression - model has static joint locations")

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
        if not dependencies.dependencies_installed():
            self.report({"ERROR"}, dependencies.MISSING_MESSAGE)
            return {"CANCELLED"}
        scene = context.scene
        smpl_tool = scene.smpl_tool

        # 1. Find all tagged objects
        unposed_meshes = [obj for obj in bpy.data.objects if obj.get("SMIL_TYPE") == "unposed_registered_mesh"]
        if not unposed_meshes:
            self.report({"ERROR"}, "No unposed registered meshes found in the scene.")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Found {len(unposed_meshes)} unposed meshes to use as shapekeys.")

        try:
            # 2. Load base pkl file to create the new model on
            pkl_filepath = bpy.path.abspath(smpl_tool.pkl_filepath)
            base_name = os.path.splitext(os.path.basename(pkl_filepath))[0] or "SMPL"
            data = load_pkl_file(pkl_filepath)
            if not data:
                self.report({"ERROR"}, "Failed to load base .pkl file.")
                return {"CANCELLED"}

            obj = create_mesh_from_pkl(data, base_name=base_name)
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
                        {"ERROR"},
                        f"Vertex count mismatch between base model and '{unposed_obj.name}'. Skipping.",
                    )
                    continue

                mesh_verts = np.array([v.co[:] for v in eval_obj.data.vertices])
                verts_list.append(mesh_verts)
                labels_list.append(unposed_obj.name)

            if not verts_list:
                self.report({"ERROR"}, "No valid unposed meshes found with matching vertex counts.")
                bpy.data.objects.remove(obj)
                return {"CANCELLED"}

            verts_data = np.array(verts_list)
            mean_shape = np.mean(verts_data, axis=0)

            npz_data = {"verts": verts_data, "labels": labels_list}

            # The rest is similar to SMPL_OT_ImportModel
            obj["SMIL_TYPE"] = "SMIL_model_from_unposed_meshes"

            # Check if the base pkl has static_joint_locs set
            if data.get("static_joint_locs", False):
                obj["static_joint_locs"] = True
                print("Using base model with static joint locations")

            store_smpl_data(context, data, obj=obj)

            create_armature_and_weights(data, obj, base_name=base_name)

            # Overwrite the base mesh geometry with the mean shape of the unposed meshes.
            # This is crucial for the shapekeys to be based on the correct average shape.
            for i, v_co in enumerate(mean_shape):
                obj.data.vertices[i].co = v_co

            if smpl_tool.shapekeys_from_PCA:
                output_dir = os.path.dirname(pkl_filepath)

                if smpl_tool.separate_pcas:
                    # Use separate PCAs (original behavior)
                    cov, mean_betas = apply_pca_and_create_shapekeys(
                        verts_data,
                        obj,
                        smpl_tool.number_of_PC,
                        overwrite_mesh=True,
                        labels=labels_list,
                        output_dir=output_dir,
                    )
                else:
                    # Use entangled PCA (new behavior)
                    # Collect scale and translation data from unposed meshes
                    scale_data_list = []
                    translation_data_list = []

                    for unposed_obj in unposed_meshes:
                        if "scale_data" in unposed_obj and "translation_data" in unposed_obj:
                            # Reconstruct scale data
                            scale_data = np.array(unposed_obj["scale_data"], dtype=np.float32)

                            # Debug: Check scale data
                            print(
                                f"Scale data for {unposed_obj.name}: shape={scale_data.shape}, values={scale_data[:10]}..."
                            )

                            # Validate scale data shape
                            if len(scale_data) != 55:
                                print(
                                    f"ERROR: Expected 55 joints but got {len(scale_data)} scale values for {unposed_obj.name}"
                                )
                                print("This suggests the scale data was stored incorrectly.")
                                # Skip this mesh
                                continue

                            # Reconstruct translation data
                            translation_data = np.array(unposed_obj["translation_data"], dtype=np.float32)
                            num_joints = len(scale_data)

                            # Debug: Check what we retrieved
                            print(f"Retrieved translation data length: {len(translation_data)} for {unposed_obj.name}")
                            print(f"Expected length: {num_joints * 3} = {num_joints * 3}")

                            # Validate translation data shape
                            expected_translation_size = num_joints * 3
                            if len(translation_data) != expected_translation_size:
                                print(
                                    f"ERROR: Expected {expected_translation_size} translation values but got {len(translation_data)} for {unposed_obj.name}"
                                )
                                print("This suggests the translation data was stored incorrectly.")
                                # Skip this mesh
                                continue

                            translation_data = translation_data.reshape(num_joints, 3)

                            # Debug: Check translation data
                            print(f"Translation data for {unposed_obj.name}: shape={translation_data.shape}")
                            if np.any(np.isnan(translation_data)) or np.any(np.isinf(translation_data)):
                                print(f"Warning: NaN or Inf values in translation data for {unposed_obj.name}")
                                print(
                                    f"Translation data range: {np.min(translation_data)} to {np.max(translation_data)}"
                                )

                            scale_data_list.append(scale_data)
                            translation_data_list.append(translation_data)
                        else:
                            self.report(
                                {"WARNING"},
                                f"No scale/translation data found for {unposed_obj.name}. Using separate PCAs instead.",
                            )
                            # Fall back to separate PCAs
                            cov, mean_betas = apply_pca_and_create_shapekeys(
                                verts_data,
                                obj,
                                smpl_tool.number_of_PC,
                                overwrite_mesh=True,
                                labels=labels_list,
                                output_dir=output_dir,
                            )
                            break
                    else:
                        # All meshes have scale/translation data, proceed with entangled PCA
                        scale_data = np.array(scale_data_list)
                        translation_data = np.array(translation_data_list)

                        # Debug: Check for NaN values in scale data
                        if np.any(np.isnan(scale_data)):
                            print("Warning: NaN values found in scale_data array")
                        if np.any(np.isnan(translation_data)):
                            print("Warning: NaN values found in translation_data array")
                        if np.any(np.isnan(verts_data)):
                            print("Warning: NaN values found in verts_data array")

                        cov, mean_betas, scaledirs, transdirs = apply_entangled_pca_and_create_shapekeys(
                            verts_data,
                            scale_data,
                            translation_data,
                            obj,
                            smpl_tool.number_of_PC,
                            overwrite_mesh=True,
                            labels=labels_list,
                            output_dir=output_dir,
                        )

                        # Store the entangled PCA components
                        data["scaledirs"] = scaledirs
                        data["transdirs"] = transdirs
            else:
                cov, mean_betas = create_shapekeys(npz_data, obj)

            data["shape_cov"] = cov
            data["shape_mean_betas"] = mean_betas

            # Handle static joint locations
            if smpl_tool.force_static_joint_locs:
                # Set J_regressor to all zeroes for static joint locations
                num_joints = data["J"].shape[0]
                num_vertices = len(obj.data.vertices)
                data["J_regressor"] = np.zeros((num_joints, num_vertices), dtype=np.float32)
                obj["static_joint_locs"] = True
                data["static_joint_locs"] = True
                print("Static joint locations enabled - J_regressor set to zeroes")

            # Update the stored data with the new shape info
            store_smpl_data(context, data, obj=obj)

            # Check if Transformation PCA components are available from LoadAllUnposedMeshes
            # Only use global variables when separate PCAs are enabled
            if smpl_tool.separate_pcas:
                if state.computed_scaledirs is not None and state.computed_transdirs is not None:
                    # Additional safety check for array shapes
                    if (
                        isinstance(state.computed_scaledirs, np.ndarray)
                        and isinstance(state.computed_transdirs, np.ndarray)
                        and len(state.computed_scaledirs.shape) == 3
                        and len(state.computed_transdirs.shape) == 3
                    ):
                        data["scaledirs"] = state.computed_scaledirs
                        data["transdirs"] = state.computed_transdirs
                        print("Added Transformation PCA components to generated model:")
                        print(f"  scaledirs shape: {state.computed_scaledirs.shape}")
                        print(f"  transdirs shape: {state.computed_transdirs.shape}")
                        # Update the stored data with the morph PCA info
                        store_smpl_data(context, data, obj=obj)
                    else:
                        print(
                            "Transformation PCA components found but with invalid shapes. Run 'Load all unposed registered meshes' again."
                        )
                else:
                    print(
                        "No Transformation PCA components found. Run 'Load all unposed registered meshes' first to compute them."
                    )

            if smpl_tool.symmetrise:
                make_symmetrical(obj, data)

            if smpl_tool.regress_joints:
                # Skip joint regression for static joint models
                if not obj.get("static_joint_locs", False):
                    apply_updated_joint_positions(obj, data)
                else:
                    print("Skipping joint regression - model has static joint locations")

            if smpl_tool.clean_mesh:
                cleanup_mesh(obj, center_tolerance=smpl_tool.merging_threshold)

            self.report({"INFO"}, "SMIL Model generated from unposed meshes successfully.")
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to generate SMIL Model: {e}")
            return {"CANCELLED"}


class SMPL_OT_ExportModel(bpy.types.Operator):
    bl_idname = "smpl.export_model"
    bl_label = "Export SMIL Model"

    def execute(self, context):
        if not dependencies.dependencies_installed():
            self.report({"ERROR"}, dependencies.MISSING_MESSAGE)
            return {"CANCELLED"}
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

            export_smpl_model(obj, pkl_data=data, export_path=bpy.path.abspath(smpl_tool.pkl_filepath))

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
        if not dependencies.dependencies_installed():
            self.report({"ERROR"}, dependencies.MISSING_MESSAGE)
            return {"CANCELLED"}
        scene = context.scene
        smpl_tool = scene.smpl_tool
        wm = context.window_manager

        # Load PKL data (for rigging info)
        pkl_filepath = bpy.path.abspath(smpl_tool.pkl_filepath)
        pkl_data = load_pkl_file(pkl_filepath)
        if not pkl_data:
            self.report({"ERROR"}, "Failed to load .pkl file.")
            return {"CANCELLED"}

        # Load NPZ data (for registered meshes)
        npz_filepath = bpy.path.abspath(smpl_tool.npz_filepath)
        if not os.path.exists(npz_filepath):
            self.report({"ERROR"}, "Could not find .npz file.")
            return {"CANCELLED"}
        npz_data = load_npz_file(npz_filepath)
        if npz_data is None or "verts" not in npz_data:
            self.report({"ERROR"}, "No 'verts' key found in .npz file.")
            return {"CANCELLED"}

        verts_array = npz_data["verts"]  # shape (N, V, 3)
        labels = npz_data["labels"] if "labels" in npz_data else [f"mesh_{i}" for i in range(len(verts_array))]
        faces = pkl_data["f"]
        weights = pkl_data["weights"]
        joints = pkl_data["J"]
        kintree_table = pkl_data["kintree_table"]
        joint_names = pkl_data["J_names"] if "J_names" in pkl_data else [f"J_{i}" for i in range(joints.shape[0])]
        J_regressor = np.copy(pkl_data["J_regressor"]) if "J_regressor" in pkl_data else None
        npz_data["global_rot"] if "global_rot" in npz_data else None  # (N, 3)
        npz_data["joint_rot"] if "joint_rot" in npz_data else None  # (N, J-1, 3)
        translations = npz_data["trans"] if "trans" in npz_data else None  # (N, 3)

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
                label_base_name = f"SMIL_{labels[i]}"
                obj = create_mesh_from_pkl(mesh_data, base_name=str(labels[i]))
                if obj is None:
                    self.report({"WARNING"}, f"Failed to create mesh for {labels[i]}")
                    continue
                obj["SMIL_TYPE"] = "unposed_registered_mesh"
                # Rig the mesh
                armature = create_armature_and_weights(mesh_data, obj, base_name=label_base_name)
                if armature is None:
                    armature = obj.find_armature()
                if armature is not None:
                    # --- Control Hierarchy Setup ---
                    # Main parent for all controls of this mesh
                    snap_controls_parent_name = f"Snap_Controls_{armature.name}"
                    snap_controls_parent = bpy.data.objects.new(snap_controls_parent_name, None)
                    snap_controls_parent.location = (0, i, 0)
                    context.collection.objects.link(snap_controls_parent)

                    # Parent for IK controls, shaped as a sphere
                    controls_parent_name = f"IK_Controls_{armature.name}"
                    controls_parent = bpy.data.objects.new(controls_parent_name, None)
                    controls_parent.empty_display_type = "SPHERE"
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
                    parent_lookup = {
                        child: parent for parent, child in zip(kintree_table[0], kintree_table[1]) if parent >= 0
                    }
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

                    # Store scale data in mesh object for later use in entangled PCA
                    # Handle zero scales by setting them to 1.0
                    final_scales_clean = final_scales.copy()
                    zero_mask = (final_scales_clean == 0) | np.isnan(final_scales_clean) | np.isinf(final_scales_clean)
                    if np.any(zero_mask):
                        print(
                            f"Warning: Found {np.sum(zero_mask)} zero/invalid scale values for {labels[i]}, setting to 1.0"
                        )
                        final_scales_clean[zero_mask] = 1.0

                    # Debug: Check what we're about to store
                    print(
                        f"About to store scale data for {labels[i]}: shape={final_scales_clean.shape}, values={final_scales_clean[:10]}..."
                    )
                    print(f"final_scales range: {np.min(final_scales_clean)} to {np.max(final_scales_clean)}")
                    print(f"final_scales has NaN: {np.any(np.isnan(final_scales_clean))}")
                    print(f"final_scales has Inf: {np.any(np.isinf(final_scales_clean))}")

                    # Store scale data as a list to avoid byte conversion issues
                    obj["scale_data"] = final_scales_clean.tolist()

                    # Verify what was stored
                    stored_data = np.array(obj["scale_data"], dtype=np.float32)
                    print(f"Verification - stored data shape: {stored_data.shape}, values: {stored_data[:10]}...")
                    print(f"Stored data matches original: {np.array_equal(stored_data, final_scales_clean)}")

                    bpy.ops.object.mode_set(mode="POSE")

                # --- END PER-BONE LENGTH NORMALIZATION (HIERARCHICAL) ---
                bpy.context.view_layer.update()

                # --- IK Rig Setup ---

                # 1. Get armature and bone data
                kintree_table = pkl_data["kintree_table"]
                joint_names = pkl_data["J_names"]
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
                bpy.ops.object.mode_set(mode="OBJECT")

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
                bpy.ops.object.mode_set(mode="POSE")

                for bone_idx in range(num_joints):
                    pose_bone = armature.pose.bones[bone_idx]
                    ik_target = ik_targets.get(pose_bone.name)

                    if not ik_target:
                        continue

                    ik_constraint = pose_bone.constraints.new("IK")
                    ik_constraint.target = ik_target
                    ik_constraint.chain_count = 1
                    ik_constraint.influence = 1.0

                # 6. Return to Object Mode
                bpy.ops.object.mode_set(mode="OBJECT")

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
                parent_lookup = {
                    child: parent for parent, child in zip(kintree_table[0], kintree_table[1]) if parent >= 0
                }
                snap_target_data = {}  # {bone_name: target_world_pos}

                # Manually align the root bone's head to the mean shape's root joint position
                root_bone_name = joint_names[0]
                snap_target_data[root_bone_name] = Vector(mean_joints[0]) + armature_offset

                # Calculate and store translation for the root bone
                translation = mean_joints[0] - mesh_joints[0]
                trans_col_start = i * 6 + 3
                transform_data[0, trans_col_start : trans_col_start + 3] = translation

                # Initialize translation data array for this mesh
                mesh_translation_data = np.zeros((num_joints, 3))
                mesh_translation_data[0] = translation

                for j in range(1, num_joints):  # Skip root bone
                    parent_idx = parent_lookup.get(j)
                    # Calculate translation for all joints, not just those with siblings
                    translation = mean_joints[j] - mesh_joints[j]
                    mesh_translation_data[j] = translation

                    # Only create snap targets for joints with siblings
                    if parent_idx is not None and len(children[parent_idx]) > 1:
                        bone_name = joint_names[j]
                        snap_target_data[bone_name] = Vector(mean_joints[j]) + armature_offset
                        trans_col_start = i * 6 + 3
                        transform_data[j, trans_col_start : trans_col_start + 3] = translation

                # Store translation data in mesh object for later use in entangled PCA
                # Debug: Check for NaN values in translation data
                if np.any(np.isnan(mesh_translation_data)):
                    print(f"Warning: NaN values found in mesh_translation_data for {labels[i]}")
                    print(f"Translation data shape: {mesh_translation_data.shape}")
                    print(f"Translation data: {mesh_translation_data}")
                    # Replace NaN with 0.0 (no translation)
                    mesh_translation_data = np.nan_to_num(mesh_translation_data, nan=0.0)

                print(f"About to store translation data for {labels[i]}: shape={mesh_translation_data.shape}")
                print(f"Translation data range: {np.min(mesh_translation_data)} to {np.max(mesh_translation_data)}")

                # Store translation data as a flat list to avoid reshaping issues
                flat_translation_data = mesh_translation_data.flatten()
                obj["translation_data"] = flat_translation_data.tolist()

                # Debug: Verify the storage
                print(f"Stored translation data length: {len(obj['translation_data'])} (should be {55 * 3} = 165)")
                print(f"Original shape: {mesh_translation_data.shape}, Flattened length: {len(flat_translation_data)}")

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
                    bpy.ops.object.mode_set(mode="POSE")
                    for bone_name, snap_target_empty in snap_targets.items():
                        pose_bone = armature.pose.bones.get(bone_name)
                        if pose_bone:
                            copy_loc_constraint = pose_bone.constraints.new("COPY_LOCATION")
                            copy_loc_constraint.target = snap_target_empty
                    bpy.ops.object.mode_set(mode="OBJECT")
        finally:
            wm.progress_end()

        # --- Export morph data to CSV ---
        try:
            output_path = os.path.join(os.path.dirname(pkl_filepath), "smil_morph_data.csv")
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

                # Store PCA components in pkl_data for export
                # Reshape components to separate scale and translation data
                # components shape: (k, num_joints*6) -> reshape to (k, num_joints, 6)
                components_reshaped = components.reshape(n_components, num_joints, 6)

                # Extract scale and translation components
                # Scale data: components[:, :, 0:3] (first 3 columns)
                # Translation data: components[:, :, 3:6] (last 3 columns)
                scaledirs = components_reshaped[:, :, 0:3]  # (k, num_joints, 3)
                transdirs = components_reshaped[:, :, 3:6]  # (k, num_joints, 3)

                # Store in pkl_data for later export
                pkl_data["scaledirs"] = scaledirs
                pkl_data["transdirs"] = transdirs

                # Also store in global variables for use by other operators
                state.computed_scaledirs = scaledirs
                state.computed_transdirs = transdirs

                print("Stored PCA components in pkl_data:")
                print(f"  scaledirs shape: {scaledirs.shape}")
                print(f"  transdirs shape: {transdirs.shape}")
                print("Also stored in global variables for use by other operators")

                pc_output_path = os.path.join(os.path.dirname(pkl_filepath), "smil_morph_PC_data.csv")
                with open(pc_output_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    # Header: joint_name, then for each PC six columns matching the original naming pattern
                    header_pc = ["joint_name"]
                    for k in range(n_components):
                        pc_label = f"PC_{k + 1}"
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
        return {"FINISHED"}


class SMPL_OT_RecomputeJointPositions(bpy.types.Operator):
    bl_idname = "smpl.recompute_joint_positions"
    bl_label = "Recompute joint positions"
    bl_description = "Recompute the J_regressor and update joint locations for the selected armature only."

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object with an armature.")
            return {"CANCELLED"}
        armature = obj.find_armature()
        if not armature:
            self.report({"ERROR"}, "Selected mesh has no armature.")
            return {"CANCELLED"}

        # Check if joint locations are set to be static
        if obj.get("static_joint_locs", False):
            self.report(
                {"WARNING"}, "Joint locations are set to be static for this model. Joint recomputation is disabled."
            )
            return {"CANCELLED"}

        # Recompute J_regressor for this mesh+armature using selected method
        smpl_tool = context.scene.smpl_tool

        # For boundary_weights method, try to get required data
        kintree_table = None
        weights = None
        if smpl_tool.j_regressor_method == "boundary_weights":
            # Try to get kintree_table and weights from stored data
            if hasattr(obj, "get"):
                if "kintree_table" in obj:
                    kintree_table = obj["kintree_table"]
                if "weights" in obj:
                    weights = obj["weights"]

        J_regressor = export_J_regressor_to_npy(
            obj, armature, 10, influence_type=smpl_tool.j_regressor_method, weights=weights, kintree_table=kintree_table
        )
        vertex_positions = np.array([np.array(v.co) for v in obj.data.vertices])
        joint_positions = np.matmul(J_regressor, vertex_positions)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="EDIT")
        for j, bone in enumerate(armature.data.edit_bones):
            bone.head = joint_positions[j]
            bone.tail = joint_positions[j] + [0, 0, 0.1]
        bpy.ops.object.mode_set(mode="OBJECT")
        self.report({"INFO"}, "Joint positions updated for selected armature.")
        return {"FINISHED"}


class SMPL_OT_ClearMorphPCA(bpy.types.Operator):
    bl_idname = "smpl.clear_morph_pca"
    bl_label = "Clear Transformation PCA components"
    bl_description = "Clear the globally stored Transformation PCA components"

    def execute(self, context):
        clear_morph_pca_globals()
        self.report({"INFO"}, "Transformation PCA components cleared.")
        return {"FINISHED"}


def _resolve_shape_key_for_beta(obj, beta_index):
    """Return the shape key block that corresponds to beta_index, or None.

    Prefers 'Shape_<i>' (direct shapedir mapping as written by create_shapekeys_from_pkl_shapedirs),
    falls back to 'PC_<i+1>' (1-indexed PCA mapping from create_shapekeys), then to positional
    order skipping the Basis key.
    """
    if not obj.data.shape_keys:
        return None
    keys = obj.data.shape_keys.key_blocks
    for candidate in (f"Shape_{beta_index}", f"PC_{beta_index + 1}"):
        if candidate in keys:
            return keys[candidate]
    non_basis = [k for k in keys if k.name != "Basis"]
    if beta_index < len(non_basis):
        return non_basis[beta_index]
    return None


def _apply_betas_to_shape_keys(obj, betas, frame=None):
    """Set shape-key values from a flat betas vector, optionally keyframing at `frame`."""
    for i, value in enumerate(betas):
        key = _resolve_shape_key_for_beta(obj, i)
        if key is None:
            break
        key.value = float(value)
        if frame is not None:
            key.keyframe_insert(data_path="value", frame=frame)


def _load_animation_files(npz_path):
    """Load .npz + sidecar .json from a path (sidecar path derived by suffix swap)."""
    import json

    npz_data = np.load(npz_path)
    json_path = os.path.splitext(npz_path)[0] + ".json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Sidecar .json not found next to {npz_path}")
    with open(json_path, "r") as f:
        sidecar = json.load(f)
    return npz_data, sidecar


def _find_mesh_with_armature(context):
    """Return (mesh_obj, armature_obj) for the active object, or (None, None)."""
    obj = context.active_object
    if obj is None:
        return None, None
    if obj.type == "ARMATURE":
        for child in obj.children:
            if child.type == "MESH":
                return child, obj
        return None, obj
    if obj.type == "MESH":
        armature = obj.find_armature()
        if armature is not None:
            return obj, armature
    return None, None


class SMPL_OT_ImportAnimation(bpy.types.Operator):
    bl_idname = "smpl.import_animation"
    bl_label = "Import Inference Animation"
    bl_description = (
        "Import a SMIL inference animation (.npz + sidecar .json) onto the active "
        "SMIL rig. Drives per-bone rotation/scale, root translation, and (when "
        "skeleton is static) per-frame shape-key weights."
    )

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.npz", options={"HIDDEN"})

    static_shape: bpy.props.BoolProperty(
        name="Static shape (use clip-averaged betas)",
        description=(
            "Apply only the clip-averaged betas once at frame 0 instead of per-frame "
            "shape keyframes. Forced on when static_joint_locs is False in the sidecar."
        ),
        default=False,
    )
    apply_joint_scales: bpy.props.BoolProperty(
        name="Apply per-joint scale",
        description="Keyframe bone.scale from log_beta_scales (exp-applied).",
        default=True,
    )
    create_cameras: bpy.props.BoolProperty(
        name="Create cameras from sidecar",
        default=True,
    )

    # PyTorch3D and Blender disagree on natural scene scale: inference clips look
    # tiny in a default Blender scene. Multiplying the rig's root transform and
    # the camera world positions by IMPORT_SCALE brings them to a comfortable
    # working size without touching focal length, per-bone rotation/scale, or
    # the camera object's own size.
    IMPORT_SCALE = 10.0

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    # ------------------------------------------------------------------ main

    def execute(self, context):
        try:
            npz_data, sidecar = _load_animation_files(bpy.path.abspath(self.filepath))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load animation files: {e}")
            return {"CANCELLED"}

        mesh_obj, armature = _find_mesh_with_armature(context)
        if armature is None:
            self.report({"ERROR"}, "Select a SMIL mesh or its armature before importing.")
            return {"CANCELLED"}

        poses = npz_data["poses"]  # (F, N_JOINTS, 3)
        trans = npz_data["trans"]  # (F, 3)
        betas_avg = npz_data["betas"]  # (N_BETAS,)
        betas_per_frame = npz_data["betas_per_frame"] if "betas_per_frame" in npz_data.files else None
        log_beta_scales = npz_data["log_beta_scales"] if "log_beta_scales" in npz_data.files else None
        # Optional global mesh scale (root-centered). When present, the inference
        # renderer applied: rendered_v = (v - J0) * mesh_scale + trans. We must
        # mirror that with armature.scale and an offset to armature.location.
        mesh_scale = npz_data["mesh_scale"] if "mesh_scale" in npz_data.files else None
        fps = float(npz_data["fps"]) if "fps" in npz_data.files else float(sidecar.get("fps", 30.0))

        n_frames, n_joints, _ = poses.shape
        joint_names = sidecar.get("joint_names", [])

        # Joint-count validation against the active armature.
        armature_bone_names = [b.name for b in armature.pose.bones]
        missing = [name for name in joint_names if name not in armature_bone_names]
        if missing:
            self.report(
                {"ERROR"},
                f"Armature is missing {len(missing)} bones from the animation "
                f"(first: {missing[:3]}). Import a matching SMIL model first.",
            )
            return {"CANCELLED"}

        # Branch on skeleton mode (see docs/animation export plan).
        static_joint_locs = bool(sidecar.get("static_joint_locs", False))
        effective_static_shape = self.static_shape or not static_joint_locs

        if not static_joint_locs:
            # Apply averaged betas statically, then recompute joint locations once.
            if mesh_obj is not None:
                _apply_betas_to_shape_keys(mesh_obj, betas_avg, frame=None)
                if mesh_obj.get("static_joint_locs", False) is False:
                    prev_active = context.view_layer.objects.active
                    context.view_layer.objects.active = mesh_obj
                    try:
                        bpy.ops.smpl.recompute_joint_positions()
                    except Exception as e:
                        self.report({"WARNING"}, f"Joint recomputation failed: {e}")
                    finally:
                        context.view_layer.objects.active = prev_active
            self.report(
                {"INFO"},
                "static_joint_locs=False: applied averaged betas and recomputed joints; "
                "per-frame shape animation is disabled for this clip.",
            )

        # Configure scene frame range and fps.
        scene = context.scene
        scene.frame_start = 0
        scene.frame_end = n_frames - 1
        scene.render.fps = max(1, int(round(fps)))

        # Inference cameras are assumed square; default the render output to 1080x1080
        # so viewport/render framing matches the camera intrinsics out of the box.
        scene.render.resolution_x = 1080
        scene.render.resolution_y = 1080
        scene.render.resolution_percentage = 100

        # Per-frame keyframing.
        context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")

        # Ensure axis-angle rotation mode on all pose bones we drive.
        for bone_name in joint_names:
            armature.pose.bones[bone_name].rotation_mode = "AXIS_ANGLE"

        # Cache each bone's rest rotation (armature space). SMAL pose rotations
        # are expressed in a frame that is world-aligned at rest; Blender's
        # rotation_axis_angle is in the bone's local rest frame. SMIL armatures
        # build every bone with head→tail along world +Z (see
        # create_armature_and_weights), so each bone's rest rotation R_0 is
        # non-trivial and we must conjugate: R_basis = R_0ᵀ · R_smal · R_0,
        # which for axis-angle reduces to angle unchanged, axis = R_0ᵀ · axis.
        rest_rot_inv_per_joint = []
        for bone_name in joint_names:
            R_rest = np.array(armature.pose.bones[bone_name].bone.matrix_local.to_3x3(), dtype=np.float64)
            rest_rot_inv_per_joint.append(R_rest.T)  # orthonormal: T == inv

        # Rest position of the root joint (joints[0]) in armature-local space.
        # Needed when mesh_scale is present: the inference renderer applies
        # rendered_v = (v - J0) * mesh_scale + trans. To replicate this with
        # armature.scale = s and armature.location = L, the visible vertex
        # becomes L + s * v, which equals (v - J0) * s + trans iff
        # L = trans - s * J0.
        root_joint_rest = np.array(armature.pose.bones[joint_names[0]].bone.head_local, dtype=np.float64)

        for f in range(n_frames):
            scene.frame_set(f)

            # Per-joint axis-angle rotation.
            for j, bone_name in enumerate(joint_names):
                aa = poses[f, j]  # (3,) world-aligned axis-angle
                angle = float(np.linalg.norm(aa))
                if angle > 1e-8:
                    axis_world = aa / angle
                    axis_local = rest_rot_inv_per_joint[j] @ axis_world
                else:
                    axis_local = np.array([0.0, 0.0, 1.0])
                pb = armature.pose.bones[bone_name]
                pb.rotation_axis_angle = (angle, float(axis_local[0]), float(axis_local[1]), float(axis_local[2]))
                pb.keyframe_insert(data_path="rotation_axis_angle", frame=f)

                if self.apply_joint_scales and log_beta_scales is not None:
                    if log_beta_scales.ndim == 3 and log_beta_scales.shape[1] == n_joints:
                        s = np.exp(log_beta_scales[f, j])
                        pb.scale = (float(s[0]), float(s[1]), float(s[2]))
                        pb.keyframe_insert(data_path="scale", frame=f)

            # Root translation (and global mesh scale, when present).
            # IMPORT_SCALE multiplies both the rig's world scale and its world
            # translation so the visible vertex (location + scale * vertex)
            # ends up at IMPORT_SCALE * (inference vertex), matching what the
            # cameras (also scaled below) expect.
            if mesh_scale is not None:
                s = float(mesh_scale[f])
                loc = self.IMPORT_SCALE * (trans[f].astype(np.float64) - s * root_joint_rest)
                rig_scale = self.IMPORT_SCALE * s
            else:
                loc = self.IMPORT_SCALE * trans[f].astype(np.float64)
                rig_scale = self.IMPORT_SCALE
            armature.scale = (rig_scale, rig_scale, rig_scale)
            armature.keyframe_insert(data_path="scale", frame=f)
            armature.location = (float(loc[0]), float(loc[1]), float(loc[2]))
            armature.keyframe_insert(data_path="location", frame=f)

            # Per-frame shape keys (only in static-skeleton mode).
            if mesh_obj is not None and not effective_static_shape and betas_per_frame is not None:
                _apply_betas_to_shape_keys(mesh_obj, betas_per_frame[f], frame=f)

        bpy.ops.object.mode_set(mode="OBJECT")

        # Static-shape single keyframe (applies to both user-forced and skeleton-forced paths).
        if effective_static_shape and mesh_obj is not None:
            _apply_betas_to_shape_keys(mesh_obj, betas_avg, frame=0)

        # Cameras.
        created_cams = []
        if self.create_cameras:
            created_cams = self._create_cameras(sidecar.get("cameras", []))

        # Group armature + cameras under a single empty so the whole imported
        # scene can be rotated/oriented as one. The mesh is intentionally left
        # parented to the armature — re-parenting it here would break the
        # Armature modifier's deformation chain.
        scene_root = bpy.data.objects.new(name="SMIL_Animation_Root", object_data=None)
        scene_root.empty_display_type = "ARROWS"
        context.collection.objects.link(scene_root)
        for child in (armature, *created_cams):
            child.parent = scene_root
            # Empty is at identity, so leaving matrix_parent_inverse as identity
            # preserves each child's existing world transform.

        scene.frame_set(0)
        self.report(
            {"INFO"},
            f"Imported {n_frames} frames at {fps:.2f} fps (static_shape={'on' if effective_static_shape else 'off'}).",
        )
        return {"FINISHED"}

    # ------------------------------------------------------------------ cameras

    def _create_cameras(self, cameras):
        import math
        from mathutils import Matrix

        created = []
        for cam in cameras:
            name = str(cam.get("view_name", "smil_cam"))
            R = np.array(cam.get("R", np.eye(3).tolist()), dtype=np.float64).reshape(3, 3)
            t = np.array(cam.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            fov_deg = float(cam.get("fov", 45.0))

            cam_data = bpy.data.cameras.new(name=name)
            cam_data.angle = math.radians(fov_deg)
            cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)
            bpy.context.collection.objects.link(cam_obj)

            # PyTorch3D's FoVPerspectiveCameras uses the row-vector world→view
            # convention: p_view = p_world @ R + T. In column-vector form this
            # is p_view = Rᵀ · p_world + T, so the world-to-view matrix is
            # [Rᵀ | T] and camera-to-world is [R | -R · T].
            #
            # Additionally, PyTorch3D camera local axes are (+X left, +Y up,
            # +Z forward) while Blender's are (+X right, +Y up, -Z forward).
            # Right-multiplying by diag(-1, 1, -1) flips the camera's local X
            # and Z so it looks the same direction in Blender as in PyTorch3D.
            cam_axis_flip = np.diag([-1.0, 1.0, -1.0])
            mat = np.eye(4)
            mat[:3, :3] = R @ cam_axis_flip
            # Scale the world-space camera position (not the camera object's
            # own scale) by IMPORT_SCALE so cameras stay framed on the rig
            # after the rig itself has been scaled up.
            mat[:3, 3] = self.IMPORT_SCALE * (-R @ t)
            cam_obj.matrix_world = Matrix(mat.tolist())
            created.append(cam_obj)
        return created


class SMPL_OT_ExportAnimationGLTF(bpy.types.Operator):
    bl_idname = "smpl.export_animation_gltf"
    bl_label = "Export Animated Model as glTF"
    bl_description = (
        "Export the imported SMIL animation (rig, mesh, cameras) as a glTF 2.0 "
        "file via Blender's built-in exporter. Available once Import SMIL "
        "Animation has produced a SMIL_Animation_Root empty."
    )

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.glb;*.gltf", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        # poll() drives the UI auto-grey-out — true only when a previous
        # import succeeded and left the root empty in place.
        return bpy.data.objects.get("SMIL_Animation_Root") is not None

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "smil_animation.glb"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        root = bpy.data.objects.get("SMIL_Animation_Root")
        if root is None:
            self.report(
                {"WARNING"},
                "No imported SMIL animation found (SMIL_Animation_Root missing).",
            )
            return {"CANCELLED"}

        prev_active = context.view_layer.objects.active
        prev_selected = [o for o in context.view_layer.objects if o.select_get()]
        try:
            bpy.ops.object.select_all(action="DESELECT")
            export_objs = [root] + list(root.children_recursive)
            for obj in export_objs:
                obj.select_set(True)
            context.view_layer.objects.active = root

            filepath = bpy.path.abspath(self.filepath)
            bpy.ops.export_scene.gltf(
                filepath=filepath,
                use_selection=True,
                export_animations=True,
                export_apply=False,
            )
        except Exception as e:
            self.report({"WARNING"}, f"glTF export failed: {e}")
            return {"CANCELLED"}
        finally:
            try:
                bpy.ops.object.select_all(action="DESELECT")
            except Exception:
                pass
            for obj in prev_selected:
                try:
                    obj.select_set(True)
                except Exception:
                    pass
            try:
                context.view_layer.objects.active = prev_active
            except Exception:
                pass

        self.report({"INFO"}, f"Exported glTF animation to {filepath}")
        return {"FINISHED"}


class SMPL_OT_ApplyPoseCorrectivesOperator(bpy.types.Operator):
    bl_idname = "smpl.apply_pose_correctives"
    bl_label = "Apply Pose Correctives"
    bl_description = "Apply pose-dependent corrective shape keys based on current armature pose"

    @classmethod
    def poll(cls, context):
        # Only enable if we have an active mesh object with an armature
        obj = context.active_object
        if not (
            obj and obj.type == "MESH" and obj.find_armature() and "has_smpl_data" in obj and "smpl_data_path" in obj
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
                self.report({"ERROR"}, "No SMPL data found. Please import a SMPL model first.")
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

