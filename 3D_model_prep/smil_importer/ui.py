"""N-panel UI (View3D > SMPL / Morphometry)."""

import bpy

from .measurements import get_reference_measurements
from .state import get_morph_pca_status


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
        layout.prop(smpl_tool, "force_static_joint_locs")

        layout.operator("smpl.import_model", text="Direct Import SMIL Model")

        # Add section for pose correctives
        layout.separator()
        layout.label(text="Advanced processing options:")
        layout.prop(smpl_tool, "j_regressor_method")
        layout.operator("smpl.recompute_joint_positions", text="Recompute joint positions")
        layout.operator("smpl.load_all_unposed_meshes", text="Load all unposed registered meshes")
        layout.prop(smpl_tool, "separate_pcas")
        layout.operator("smpl.generate_from_unposed", text="Generate SMIL model from unposed meshes")

        # Add morph PCA status indicator
        morph_available, morph_status = get_morph_pca_status()
        if morph_available:
            status_box = layout.box()
            status_box.label(text="Transformation PCA components:", icon="CHECKMARK")
            status_box.label(text=morph_status)
        else:
            status_box = layout.box()
            status_box.label(text="Transformation PCA components:", icon="INFO")
            status_box.label(text=morph_status)

        # Add clear button if components are available
        if morph_available:
            layout.operator("smpl.clear_morph_pca", text="Clear Transformation PCA components")

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

        layout.separator()
        layout.operator("smpl.import_animation", text="Import SMIL Animation (.npz)")
        # Stays greyed out until SMPL_OT_ExportAnimationGLTF.poll() finds
        # SMIL_Animation_Root in the scene.
        layout.operator("smpl.export_animation_gltf", text="Export animated model as glTF")


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
