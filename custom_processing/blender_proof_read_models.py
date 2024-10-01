import bpy
import os
import addon_utils
from bpy.props import StringProperty, IntProperty
from bpy.types import Panel, Operator

def export_mesh_to_obj(obj, filepath):
    """
    Exports a mesh object to an OBJ file.

    Args:
        obj (bpy.types.Object): The Blender mesh object to export.
        filepath (str): The file path where the OBJ file will be saved.

    Returns:
        str: The file path of the exported OBJ file.

    Raises:
        TypeError: If the selected object is not a mesh.
        ValueError: If a face is not a triangle.
    """
    if obj.type != 'MESH':
        raise TypeError("The selected object is not a mesh.")

    # Convert mesh to triangles
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = obj.data
    vertices = []
    faces = []

    for vert in mesh.vertices:
        vertices.append(vert.co)

    for poly in mesh.polygons:
        if len(poly.vertices) == 3:
            faces.append(poly.vertices)
        else:
            raise ValueError(f"Face with vertices {poly.vertices} is not a triangle and will be skipped.")

    with open(filepath, 'w') as file:
        for vert in vertices:
            file.write(f"v {vert.x} {vert.y} {vert.z}\n")

        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    return filepath

def ensure_addon_enabled(addon_name):
    """
    Ensures that the specified Blender addon is enabled.

    Args:
        addon_name (str): The name of the addon to enable.

    Returns:
        bool: True if the addon is enabled successfully, False otherwise.
    """
    if addon_utils.check(addon_name)[0]:
        print(f"Addon {addon_name} is already enabled.")
        return True
    try:
        addon_utils.enable(addon_name, default_set=True)
        if addon_utils.check(addon_name)[0]:
            print(f"Enabled addon: {addon_name}")
            return True
        else:
            print(f"Failed to enable addon: {addon_name}")
            return False
    except Exception as e:
        print(f"Error enabling addon {addon_name}: {str(e)}")
        return False

class MODEL_LOADER_PT_Panel(Panel):
    """Panel for the Model Loader addon."""

    bl_label = "Model Loader"
    bl_idname = "MODEL_LOADER_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Model Loader'

    def draw(self, context):
        """
        Draws the panel layout.

        Args:
            context (bpy.types.Context): The current Blender context.
        """
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "input_folder")
        layout.prop(scene, "output_folder")

        row = layout.row()
        row.operator("model_loader.load_previous_model", text="Previous Model")
        row.operator("model_loader.load_next_model", text="Next Model")
        layout.operator("model_loader.export_model")

        layout.label(text=f"Current Model: {scene.current_model_name}")
        layout.label(text=f"Model {scene.current_model_index + 1} of {scene.total_models}")

class MODEL_LOADER_OT_LoadNextModel(Operator):
    """Operator to load the next model."""

    bl_idname = "model_loader.load_next_model"
    bl_label = "Load Next Model"

    def execute(self, context):
        """
        Executes the operator to load the next model.

        Args:
            context (bpy.types.Context): The current Blender context.

        Returns:
            set: {'FINISHED'} if successful, {'CANCELLED'} otherwise.
        """
        return load_model(context, 1)

class MODEL_LOADER_OT_LoadPreviousModel(Operator):
    """Operator to load the previous model."""

    bl_idname = "model_loader.load_previous_model"
    bl_label = "Load Previous Model"

    def execute(self, context):
        """
        Executes the operator to load the previous model.

        Args:
            context (bpy.types.Context): The current Blender context.

        Returns:
            set: {'FINISHED'} if successful, {'CANCELLED'} otherwise.
        """
        return load_model(context, -1)

def load_model(context, direction):
    """
    Loads the next or previous model based on the direction.

    Args:
        context (bpy.types.Context): The current Blender context.
        direction (int): 1 for next model, -1 for previous model.

    Returns:
        set: {'FINISHED'} if successful, {'CANCELLED'} otherwise.
    """
    scene = context.scene
    if not scene.input_folder:
        return {'CANCELLED'}

    model_files = [f for f in os.listdir(scene.input_folder) if f.lower().endswith('.obj')]
    if not model_files:
        return {'CANCELLED'}

    scene.current_model_index = (scene.current_model_index + direction) % len(model_files)
    if scene.current_model_index < 0:
        scene.current_model_index = len(model_files) - 1

    # Remove previous model if exists
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Load new model
    model_path = os.path.join(scene.input_folder, model_files[scene.current_model_index])
    try:
        bpy.ops.wm.obj_import(filepath=model_path)
    except Exception as e:
        return {'CANCELLED'}

    scene.current_model_name = model_files[scene.current_model_index]
    scene.total_models = len(model_files)

    return {'FINISHED'}

class MODEL_LOADER_OT_ExportModel(Operator):
    """Operator to export the current model."""

    bl_idname = "model_loader.export_model"
    bl_label = "Export Model"

    def execute(self, context):
        """
        Executes the operator to export the current model.

        Args:
            context (bpy.types.Context): The current Blender context.

        Returns:
            set: {'FINISHED'} if successful, {'CANCELLED'} otherwise.
        """
        scene = context.scene
        if not scene.output_folder:
            self.report({'ERROR'}, "Please specify an output folder")
            return {'CANCELLED'}

        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object selected")
            return {'CANCELLED'}

        # Apply all transforms
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        output_path = os.path.join(scene.output_folder, f"{scene.current_model_name}_processed.obj")
        export_mesh_to_obj(obj, output_path)
        self.report({'INFO'}, f"Model exported to {output_path}")

        return {'FINISHED'}

def register():
    """Registers the addon classes and properties."""
    if not hasattr(bpy.types.Scene, "input_folder"):
        bpy.types.Scene.input_folder = StringProperty(
            name="Input Folder",
            description="Folder containing .obj files",
            subtype='DIR_PATH'
        )
    if not hasattr(bpy.types.Scene, "output_folder"):
        bpy.types.Scene.output_folder = StringProperty(
            name="Output Folder",
            description="Folder to save processed .obj files",
            subtype='DIR_PATH'
        )
    if not hasattr(bpy.types.Scene, "current_model_name"):
        bpy.types.Scene.current_model_name = StringProperty(default="")
    if not hasattr(bpy.types.Scene, "current_model_index"):
        bpy.types.Scene.current_model_index = IntProperty(default=0)
    if not hasattr(bpy.types.Scene, "total_models"):
        bpy.types.Scene.total_models = IntProperty(default=0)

    for cls in [MODEL_LOADER_PT_Panel, MODEL_LOADER_OT_LoadNextModel, MODEL_LOADER_OT_LoadPreviousModel, MODEL_LOADER_OT_ExportModel]:
        if not hasattr(bpy.types, cls.__name__):
            bpy.utils.register_class(cls)

def unregister():
    """Unregisters the addon classes and properties."""
    del bpy.types.Scene.input_folder
    del bpy.types.Scene.output_folder
    del bpy.types.Scene.current_model_name
    del bpy.types.Scene.current_model_index
    del bpy.types.Scene.total_models

    bpy.utils.unregister_class(MODEL_LOADER_PT_Panel)
    bpy.utils.unregister_class(MODEL_LOADER_OT_LoadNextModel)
    bpy.utils.unregister_class(MODEL_LOADER_OT_LoadPreviousModel)
    bpy.utils.unregister_class(MODEL_LOADER_OT_ExportModel)

if __name__ == "__main__":
    register()