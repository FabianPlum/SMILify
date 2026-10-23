"""Blender PropertyGroup for the add-on UI."""

import bpy

from . import visualization


class SMPLProperties(bpy.types.PropertyGroup):
    pkl_filepath: bpy.props.StringProperty(
        name="PKL Filepath",
        description="Path to the .pkl file",
        default="",
        subtype="FILE_PATH",
    )

    npz_filepath: bpy.props.StringProperty(
        name="NPZ Filepath",
        description="Path to the .npz file",
        default="",
        subtype="FILE_PATH",
    )

    shapekeys_from_PCA: bpy.props.BoolProperty(
        name="shapekeys from PCA",
        description="Generate shapekeys from PCA",
        default=True,
    )

    number_of_PC: bpy.props.IntProperty(
        name="Number of Principal Components",
        description="Number of principal components for PCA",
        default=20,
    )

    regress_joints: bpy.props.BoolProperty(name="Regress Joints", description="Regress joint positions", default=True)

    clean_mesh: bpy.props.BoolProperty(
        name="Auto Clean-up Mesh",
        description="Merges overlapping vertices and removes inward facing faces",
        default=True,
    )

    merging_threshold: bpy.props.FloatProperty(
        name="Minimal vertex distance",
        description="Minimal distance between vertices on centre line during mesh cleanup",
        default=0.001,
    )

    symmetrise: bpy.props.BoolProperty(name="Symmetrise", description="Symmetrise the model", default=True)

    # Add property for separate PCAs
    separate_pcas: bpy.props.BoolProperty(
        name="Perform separate PCAs for shape, scale, and translation",
        description="When enabled, performs separate PCAs for shape, scale, and translation. When disabled, performs entangled PCA combining all three.",
        default=True,
    )

    # Add property for J_regressor computation method
    j_regressor_method: bpy.props.EnumProperty(
        name="J_regressor Computation Method",
        description="Choose the method for computing joint regressor weights",
        items=[
            ("inverse_distance", "Inverse Distance", "Use inverse distance weighting to nearest vertices"),
            ("boundary_weights", "Boundary Weights", "Use boundary weights based on parent-child joint relationships"),
        ],
        default="inverse_distance",
    )

    # Add properties to store SMPL data
    has_smpl_data: bpy.props.BoolProperty(default=False)
    v_template: bpy.props.FloatVectorProperty(size=3)  # This will store the shape
    posedirs: bpy.props.FloatVectorProperty(size=3)  # This will store the pose correctives

    # Add to SMPLProperties class:
    output_filename: bpy.props.StringProperty(
        name="Output Filename",
        description="Name of the output SMPL model file",
        default="SMPL_fit.pkl",
    )

    # Add properties for reference measurements CSV
    reference_csv_filepath: bpy.props.StringProperty(
        name="Reference CSV Filepath",
        description="Path to the CSV file containing reference measurements",
        default="",
        subtype="FILE_PATH",
    )

    reference_joint_pair: bpy.props.StringProperty(
        name="Reference Joint Pair",
        description="Joint pair used for reference measurements (read from CSV)",
        default="",
        options={"SKIP_SAVE"},
    )

    has_reference_data: bpy.props.BoolProperty(default=False)

    # Add property for force static joint locations
    force_static_joint_locs: bpy.props.BoolProperty(
        name="Force Static Joint Locations",
        description="Joint locations will not be affected by shape keys. J_regressor will be set to all zeroes. Useful for models with root bone at world origin or when joint locations should remain constant.",
        default=False,
    )

    # Issue #116: the exporter rebuilds 'joint_limits' from constraints, so import
    # has to recreate them or an unchanged round trip opens every joint.
    import_joint_limits: bpy.props.BoolProperty(
        name="Import Joint Limits",
        description=(
            "Recreate the .pkl's 'joint_limits' as local-space 'Limit Rotation' bone "
            "constraints, so the panel, the viewport overlay and a re-export match the "
            "file. Axes at the wide-open default range below are left unconstrained. "
            "Untick to import the rig without constraints"
        ),
        default=True,
    )

    # Issue #56: export user-defined per-joint rotation limits.
    # Deliberately opt-out (default=True): authored constraints always land in
    # the .pkl, and unconstrained rigs get the wide-open default range, which
    # keeps the limit prior inactive - so the key is harmless when unused.
    export_joint_limits: bpy.props.BoolProperty(
        name="Export Joint Limits",
        description=(
            "Read per-bone rotation limits and store them in the exported .pkl under the "
            "'joint_limits' key. Limits act as pose priors for the optimisation fitter and "
            "neural inference. For each bone the limits are read from a 'Limit Rotation' "
            "pose-bone constraint if present, otherwise from the bone's IK rotation "
            "limits/locks. Axes with no explicit limit use the wide-open default range "
            "below; the root bone is fixed."
        ),
        default=True,
    )

    # Visualization section (SMIL panel > Visualization).
    show_joint_limit_overlay: bpy.props.BoolProperty(
        name="Show Joint Limit Overlay",
        description=(
            "Draw a translucent wedge in the viewport for each enabled axis of every "
            "bone's Limit Rotation constraint (X red, Y green, Z blue). Selects exactly "
            "the constraints 'Export Joint Limits' reads (muted, influence-0 and "
            "non-local-space constraints are skipped), so preview and export always "
            "agree on WHICH limits apply. Wedges show the authored bone-local angles; "
            "the export additionally remaps them into the model frame. The IK fallback "
            "and default range are not shown"
        ),
        default=False,
        update=visualization.toggle_overlay_update,
    )

    joint_limit_default_range: bpy.props.FloatProperty(
        name="Default Joint Limit Range",
        description=(
            "Half-range used for axes with no explicit limit (displayed in degrees, "
            "matching the Limit Rotation constraint fields; stored in radians). Such "
            "axes are exported as [-value, +value]. The default 180 deg means "
            "effectively unconstrained, so the limit prior stays inactive until real "
            "limits are set. WARNING: 0 exports every unset axis as locked, which "
            "freezes the pose wherever the limit loss is active"
        ),
        default=3.141592653589793,
        min=0.0,
        soft_min=0.017453292519943295,  # 1 degree: keep the slider off the all-locked 0
        subtype="ANGLE",
    )
