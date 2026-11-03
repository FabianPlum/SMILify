"""SMIL Model Importer — installable Blender add-on.

Import, configure and export SMPL / SMIL parametric models.

This package replaces the former single 4000-line ``SMIL_processing_addon.py``
script. Functionality is split into focused modules:

    dependencies  scipy / scikit-learn detection and one-click installer
    state         shared Transformation-PCA components
    unpickler     legacy chumpy-aware .pkl loader
    core_mesh     leaf geometry / export utilities (scipy, lazy)
    pca           PCA shape-space construction (scikit-learn, lazy)
    model_build   build Blender mesh / armature / shape-keys, export model
    measurements  joint distances, mesh measurements, SMPL data store
    properties    add-on PropertyGroup
    operators     import / export / generate / animation operators
    ui            N-panel UI
    visualization viewport overlays (joint-limit wedges)

scipy and scikit-learn are imported lazily, so the add-on always registers even
when they are missing; the preferences panel then offers to install them.
"""

bl_info = {
    "name": "SMIL Model Importer",
    "author": "Fabian Plum",
    "version": (2, 3, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > SMIL",
    "description": "Import, configure, and export SMPL / SMIL models",
    "warning": "Requires scipy and scikit-learn (installable from add-on preferences)",
    "category": "Import-Export",
}

import os

import bpy

from . import dependencies
from . import properties
from . import operators
from . import ui
from . import visualization

# Registration order: dependency-installer + preferences first, then UI panels,
# operators, and finally the PropertyGroup that Scene.smpl_tool points at.
classes = (
    dependencies.SMIL_OT_InstallDependencies,
    dependencies.SMILAddonPreferences,
    ui.SMIL_PT_Panel,
    ui.SMIL_PT_VisualizationPanel,
    ui.SMIL_PT_MorphometryPanel,
    operators.SMPL_OT_ImportModel,
    operators.SMPL_OT_GenerateFromUnposed,
    operators.SMPL_OT_ExportModel,
    operators.SMPL_OT_ApplyPoseCorrectivesOperator,
    operators.SMPL_OT_ExportJointDistances,
    operators.SMPL_OT_ExportMeshMeasurements,
    operators.SMPL_OT_LoadReferenceMeasurements,
    operators.SMPL_OT_LoadAllUnposedMeshes,
    operators.SMPL_OT_RecomputeJointPositions,
    operators.SMPL_OT_ClearMorphPCA,
    operators.SMPL_OT_ImportAnimation,
    operators.SMPL_OT_ExportAnimationGLTF,
    properties.SMPLProperties,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.smpl_tool = bpy.props.PointerProperty(type=properties.SMPLProperties)
    # Issue #56 (review): the overlay checkbox is a Scene property (saved in
    # the .blend) but the draw handler is session state. Install a load_post
    # handler so the overlay is restored when a file is opened, and sync the
    # already-loaded scene now (covers add-on re-enable mid-session).
    visualization.register_handlers()


def unregister():
    # Remove the load_post handler and any active viewport overlay before
    # classes disappear.
    visualization.unregister_handlers()
    visualization.disable_overlay()

    # Clean up temporary files written by operators.
    for obj in bpy.data.objects:
        if "smpl_data_path" in obj:
            temp_path = obj["smpl_data_path"]
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # Clean up reference measurements file.
    if "reference_measurements_path" in bpy.context.scene:
        temp_path = bpy.context.scene["reference_measurements_path"]
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.smpl_tool


if __name__ == "__main__":
    register()
