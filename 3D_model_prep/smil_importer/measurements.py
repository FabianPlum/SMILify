"""Joint distances, mesh measurements, reference data and the SMPL data store."""

import os
import csv
import base64
import pickle

import bpy
import bmesh
import numpy as np

# Custom property key under which encoded SMPL data is stored on objects.
SMPL_DATA_PROP = "smpl_data_b64"

from .core_mesh import export_J_regressor_to_npy, recalculate_joint_positions


def _encode_smpl_data(data):
    """Serialize a SMPL/SMIL data dict for storage in a Blender string property."""
    return base64.b64encode(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _decode_smpl_data(encoded):
    """Inverse of _encode_smpl_data. Raises on corruption — caller decides recovery."""
    return pickle.loads(base64.b64decode(encoded.encode("ascii")))


def store_smpl_data(context, data, obj=None):
    """Embed the SMPL/SMIL data dict on the mesh object so it survives .blend reopen.

    The legacy implementation wrote to a system temp file and only kept the path
    on the object, which silently lost entangled morph PCA, shape PCA stats and
    other metadata after a reboot or when sharing the project. The embedded
    custom property travels with the .blend and is the authoritative source.
    """
    if obj is None:
        obj = context.active_object

    if not obj:
        return

    try:
        obj[SMPL_DATA_PROP] = _encode_smpl_data(data)
        obj["has_smpl_data"] = True
        context.scene.smpl_tool.has_smpl_data = True
    except Exception as e:
        print(f"Failed to embed SMPL data on {obj.name!r}: {e}")


def get_smpl_data(context):
    """Retrieve SMPL data, preferring the embedded copy over the legacy temp file."""
    obj = context.active_object
    if obj is None:
        return None

    if SMPL_DATA_PROP in obj:
        try:
            return _decode_smpl_data(obj[SMPL_DATA_PROP])
        except Exception as e:
            print(f"Failed to decode embedded SMPL data on {obj.name!r}: {e}")

    # Legacy fallback: old projects only stored a temp-file path.
    if "smpl_data_path" in obj:
        temp_path = obj["smpl_data_path"]
        if os.path.exists(temp_path):
            with open(temp_path, "rb") as f:
                return pickle.load(f)
    return None


def get_joint_distances(armature_obj):
    """Calculate distances between all joint pairs in the armature."""
    joints = armature_obj.data.bones
    distances = []

    # Calculate distances between all joint pairs
    for i, bone1 in enumerate(joints):
        for j, bone2 in enumerate(joints[i + 1 :], i + 1):
            dist = (bone1.head_local - bone2.head_local).length
            distances.append([bone1.name, bone2.name, dist])

    return distances


def get_joint_distances_from_positions(joint_positions, joint_names):
    """Calculate distances between all joint pairs from joint positions."""
    distances = []

    # Calculate distances between all joint pairs
    for i, pos1 in enumerate(joint_positions):
        for j, pos2 in enumerate(joint_positions[i + 1 :], i + 1):
            dist = np.linalg.norm(pos1 - pos2)
            distances.append([joint_names[i], joint_names[j], dist])

    return distances


def export_joint_distances(context, filepath):
    """Export joint distances to a CSV file, including distances for each shape key."""
    mesh_obj = context.active_object
    if not mesh_obj or mesh_obj.type != "MESH":
        return False, "No mesh object selected"

    armature = mesh_obj.find_armature()
    if not armature:
        return False, "No armature found for the selected mesh"

    # Get joint names from armature
    joint_names = [bone.name for bone in armature.data.bones]

    # Recalculate J_regressor for current mesh state using selected method
    # This ensures it works even if mesh topology has changed
    # Uses the 10 nearest vertices, consider exposing this as a parameter
    smpl_tool = context.scene.smpl_tool
    if mesh_obj.get("static_joint_locs", False):
        J_regressor = export_J_regressor_to_npy(mesh_obj, armature, 10, influence_type=smpl_tool.j_regressor_method)

    # Check if reference measurements are available
    reference_measurements = {}
    reference_joint_pair = []

    if smpl_tool.has_reference_data:
        reference_measurements = get_reference_measurements(context)

        # Parse joint pair from reference_joint_pair
        joint_pair = smpl_tool.reference_joint_pair

        # Try to extract joint names from the joint pair string
        # Format is typically "joint1 to joint2 [unit]"
        if "to" in joint_pair:
            parts = joint_pair.split("to")
            if len(parts) >= 2:
                joint1 = parts[0].strip()
                joint2 = parts[1].split("[")[0].strip()
                reference_joint_pair = [joint1, joint2]

        # Verify reference joints exist in the armature
        if len(reference_joint_pair) == 2:
            for joint in reference_joint_pair:
                if joint not in joint_names:
                    print(f"Warning: Reference joint '{joint}' not found in armature")
                    reference_joint_pair = []

    # Prepare data for CSV
    all_data = []

    # Add header row with scaling info if reference data is available
    if reference_joint_pair and reference_measurements:
        all_data.append(
            [
                "Shape",
                "Joint1",
                "Joint2",
                "Distance",
                "Scaling Factor",
                "Scaled Distance in mm",
            ]
        )
    else:
        all_data.append(["Shape", "Joint1", "Joint2", "Distance"])

    # Get base mesh distances using depsgraph evaluation
    depsgraph = context.evaluated_depsgraph_get()

    # Store original shape key values
    original_values = {}
    if mesh_obj.data.shape_keys:
        for key in mesh_obj.data.shape_keys.key_blocks:
            original_values[key.name] = key.value
            key.value = 0.0  # Reset all to 0

    # Update mesh to ensure we start from basis
    mesh_obj.data.update()

    # Get evaluated mesh for base shape
    eval_obj = mesh_obj.evaluated_get(depsgraph)

    # Get vertex positions from evaluated mesh
    vertex_positions = np.array([np.array(v.co) for v in eval_obj.data.vertices])

    # Calculate joint positions using J_regressor
    if mesh_obj.get("static_joint_locs", False):
        joint_positions = np.array([bone.head_local for bone in armature.data.bones])
    else:
        joint_positions = recalculate_joint_positions(vertex_positions, J_regressor)

    # Calculate distances between all joint pairs
    base_distances = []
    for i, pos1 in enumerate(joint_positions):
        for j, pos2 in enumerate(joint_positions[i + 1 :], i + 1):
            dist = np.linalg.norm(pos1 - pos2)
            base_distances.append([joint_names[i], joint_names[j], dist])

    # Add base mesh distances
    for row in base_distances:
        if reference_joint_pair and reference_measurements:
            # scaling factor is not applicable to base mesh
            scaling_factor = "N/A"
            scaled_distance = "N/A"
            all_data.append(["Base"] + row + [scaling_factor, scaled_distance])
        else:
            all_data.append(["Base"] + row)

    # Get distances for each shape key
    if mesh_obj.data.shape_keys and len(mesh_obj.data.shape_keys.key_blocks) > 1:
        # For each shape key
        for key in mesh_obj.data.shape_keys.key_blocks[1:]:  # Skip basis
            # Reset all shape keys to 0 first
            for k in mesh_obj.data.shape_keys.key_blocks:
                k.value = 0.0

            # Set this shape key to 1.0
            key.value = 1.0

            # Force a complete update of the mesh
            mesh_obj.data.update()
            context.view_layer.update()

            # Get the evaluated object with this shape key applied
            depsgraph.update()
            eval_obj = mesh_obj.evaluated_get(depsgraph)

            # Get vertex positions from evaluated mesh
            vertex_positions = np.array([np.array(v.co) for v in eval_obj.data.vertices])

            # Calculate joint positions using J_regressor
            if mesh_obj.get("static_joint_locs", False):
                joint_positions = np.array([bone.head_local for bone in armature.data.bones])
            else:
                joint_positions = recalculate_joint_positions(vertex_positions, J_regressor)

            # Calculate distances between joints
            key_distances = []
            for i, pos1 in enumerate(joint_positions):
                for j, pos2 in enumerate(joint_positions[i + 1 :], i + 1):
                    dist = np.linalg.norm(pos1 - pos2)
                    key_distances.append([joint_names[i], joint_names[j], dist])

            # Calculate scaling factor if reference data is available
            scaling_factor = 1.0
            # Clean the key name as it may contain file endings
            key_name = key.name.split(".")[0]

            if reference_joint_pair and reference_measurements and key_name in reference_measurements:
                # Find the distance between reference joints for this shape key
                ref_joint_idx1 = (
                    joint_names.index(reference_joint_pair[0]) if reference_joint_pair[0] in joint_names else -1
                )
                ref_joint_idx2 = (
                    joint_names.index(reference_joint_pair[1]) if reference_joint_pair[1] in joint_names else -1
                )

                if ref_joint_idx1 >= 0 and ref_joint_idx2 >= 0:
                    # Calculate the current distance between reference joints
                    current_dist = np.linalg.norm(joint_positions[ref_joint_idx1] - joint_positions[ref_joint_idx2])

                    # Get the reference distance
                    reference_dist = reference_measurements.get(key_name, 0.0)

                    if current_dist > 0 and reference_dist > 0:
                        # Calculate scaling factor
                        scaling_factor = reference_dist / current_dist
                        print(
                            f"Shape key {key.name}: Scaling factor = {scaling_factor} (Reference: {reference_dist}, Current: {current_dist})"
                        )

            # Add to data with shape key name and apply scaling if needed
            for dist_data in key_distances:
                if reference_joint_pair and reference_measurements:
                    scaled_distance = dist_data[2] * scaling_factor
                    all_data.append([key.name] + dist_data + [scaling_factor, scaled_distance])
                else:
                    all_data.append([key.name] + dist_data)

        # Restore original values
        for key_name, value in original_values.items():
            if key_name in mesh_obj.data.shape_keys.key_blocks:
                mesh_obj.data.shape_keys.key_blocks[key_name].value = value
            mesh_obj.data.update()

    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_data)
        return True, f"Distances exported to {filepath}"
    except Exception as e:
        return False, f"Failed to export distances: {str(e)}"


def calculate_mesh_measurements(obj):
    """Calculate surface area and volume of a mesh object using Blender's internal functions."""
    bpy.context.view_layer.objects.active = obj

    # Get mesh data
    mesh = obj.data

    # Calculate surface area
    surface_area = sum(p.area for p in mesh.polygons)

    # Calculate volume
    # We need to ensure the mesh is manifold for accurate volume calculation
    bm = bmesh.new()
    bm.from_mesh(mesh)
    volume = bm.calc_volume()
    bm.free()

    return abs(surface_area), abs(volume)


def export_mesh_measurements(context, filepath):
    """Export mesh surface area and volume measurements to a CSV file, including measurements for each shape key."""
    obj = context.active_object
    if not obj or obj.type != "MESH":
        return False, "No mesh object selected"

    # Check if reference measurements are available
    smpl_tool = context.scene.smpl_tool
    reference_measurements = {}
    reference_joint_pair = []

    if smpl_tool.has_reference_data:
        reference_measurements = get_reference_measurements(context)

        # Parse joint pair from reference_joint_pair
        joint_pair = smpl_tool.reference_joint_pair

        # Try to extract joint names from the joint pair string
        if "to" in joint_pair:
            parts = joint_pair.split("to")
            if len(parts) >= 2:
                joint1 = parts[0].strip()
                joint2 = parts[1].split("[")[0].strip()
                reference_joint_pair = [joint1, joint2]

        # Verify reference joints exist in the armature
        armature = obj.find_armature()
        if armature and len(reference_joint_pair) == 2:
            joint_names = [bone.name for bone in armature.data.bones]
            for joint in reference_joint_pair:
                if joint not in joint_names:
                    print(f"Warning: Reference joint '{joint}' not found in armature")
                    reference_joint_pair = []

    try:
        # Prepare data for CSV
        all_data = []

        # Add header row with scaling info if reference data is available
        if reference_joint_pair and reference_measurements:
            all_data.append(["Shape", "Measurement", "Value", "Scaling Factor", "Scaled Value"])
        else:
            all_data.append(["Shape", "Measurement", "Value"])

        # Calculate base measurements
        surface_area, volume = calculate_mesh_measurements(obj)

        # Base measurements are not scaled
        if reference_joint_pair and reference_measurements:
            all_data.append(["Base", "Surface Area", surface_area, "N/A", "N/A"])
            all_data.append(["Base", "Volume", volume, "N/A", "N/A"])
        else:
            all_data.append(["Base", "Surface Area", surface_area])
            all_data.append(["Base", "Volume", volume])

        # Get measurements for each shape key
        if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 1:
            # Store original values
            original_values = {}
            for key in obj.data.shape_keys.key_blocks[1:]:  # Skip basis
                original_values[key.name] = key.value
                key.value = 0.0

            # Update mesh to ensure we start from basis
            obj.data.update()
            context.view_layer.update()  # Force a full update

            # Create a temporary object for measurements
            temp_mesh = bpy.data.meshes.new("TempMeasurementMesh")
            temp_obj = bpy.data.objects.new("TempMeasurementObj", temp_mesh)
            context.collection.objects.link(temp_obj)

            # If we have reference measurements, we need to calculate joint distances
            joint_distances = {}
            if reference_joint_pair and reference_measurements and armature:
                # Recalculate J_regressor for current mesh state using selected method
                smpl_tool = context.scene.smpl_tool
                J_regressor = export_J_regressor_to_npy(obj, armature, 10, influence_type=smpl_tool.j_regressor_method)
                joint_names = [bone.name for bone in armature.data.bones]

                # Get indices of reference joints
                ref_joint_idx1 = (
                    joint_names.index(reference_joint_pair[0]) if reference_joint_pair[0] in joint_names else -1
                )
                ref_joint_idx2 = (
                    joint_names.index(reference_joint_pair[1]) if reference_joint_pair[1] in joint_names else -1
                )

                if ref_joint_idx1 >= 0 and ref_joint_idx2 >= 0:
                    # For each shape key, calculate the joint distance
                    for key in obj.data.shape_keys.key_blocks[1:]:  # Skip basis
                        # Reset all shape keys to 0 first
                        for k in obj.data.shape_keys.key_blocks[1:]:
                            k.value = 0.0

                        # Set this shape key to 1.0
                        key.value = 1.0
                        obj.data.update()

                        # Get vertex positions with this shape key applied
                        vertex_positions = np.array([np.array(v.co) for v in obj.data.vertices])

                        # Calculate joint positions using J_regressor
                        joint_positions = recalculate_joint_positions(vertex_positions, J_regressor)

                        # Calculate distance between reference joints
                        current_dist = np.linalg.norm(joint_positions[ref_joint_idx1] - joint_positions[ref_joint_idx2])
                        joint_distances[key.name] = current_dist

                        # Reset this shape key
                        key.value = 0.0
                        obj.data.update()

            # For each shape key
            for key in obj.data.shape_keys.key_blocks[1:]:  # Skip basis
                # Reset all shape keys to 0 first
                for k in obj.data.shape_keys.key_blocks[1:]:
                    k.value = 0.0

                # Set this shape key to 1.0
                key.value = 1.0

                # Force a complete update of the mesh
                obj.data.update()
                context.view_layer.update()

                # Get the evaluated object
                depsgraph = context.evaluated_depsgraph_get()
                eval_obj = obj.evaluated_get(depsgraph)

                # Copy the evaluated mesh to our temporary mesh
                temp_mesh.clear_geometry()
                temp_mesh.from_pydata(
                    [v.co[:] for v in eval_obj.data.vertices],
                    [],
                    [p.vertices[:] for p in eval_obj.data.polygons],
                )
                temp_mesh.update()

                # Calculate measurements on the temporary object
                key_surface_area, key_volume = calculate_mesh_measurements(temp_obj)

                # Calculate scaling factor if reference data is available
                scaling_factor = 1.0
                # Clean the key name as it may contain file endings
                key_name = key.name.split(".")[0]

                if (
                    reference_joint_pair
                    and reference_measurements
                    and key_name in reference_measurements
                    and key.name in joint_distances
                ):
                    # Get the reference distance
                    reference_dist = reference_measurements.get(key_name, 0.0)
                    current_dist = joint_distances[key.name]

                    if current_dist > 0 and reference_dist > 0:
                        # Calculate linear scaling factor
                        scaling_factor = reference_dist / current_dist
                        print(f"Shape key {key.name}: Scaling factor = {scaling_factor}")

                        # Scale surface area (s²) and volume (s³)
                        scaled_surface_area = key_surface_area * (scaling_factor**2)
                        scaled_volume = key_volume * (scaling_factor**3)

                        # Add to data with shape key name and scaled values
                        all_data.append(
                            [
                                key.name,
                                "Surface Area",
                                key_surface_area,
                                scaling_factor,
                                scaled_surface_area,
                            ]
                        )
                        all_data.append(
                            [
                                key.name,
                                "Volume",
                                key_volume,
                                scaling_factor,
                                scaled_volume,
                            ]
                        )
                    else:
                        # Add unscaled values if we can't calculate scaling
                        all_data.append([key.name, "Surface Area", key_surface_area, "N/A", "N/A"])
                        all_data.append([key.name, "Volume", key_volume, "N/A", "N/A"])
                else:
                    # Add to data without scaling
                    if reference_joint_pair and reference_measurements:
                        all_data.append([key.name, "Surface Area", key_surface_area, "N/A", "N/A"])
                        all_data.append([key.name, "Volume", key_volume, "N/A", "N/A"])
                    else:
                        all_data.append([key.name, "Surface Area", key_surface_area])
                        all_data.append([key.name, "Volume", key_volume])

            # Remove temporary object
            bpy.data.objects.remove(temp_obj)
            bpy.data.meshes.remove(temp_mesh)

            # Restore original values
            for key_name, value in original_values.items():
                obj.data.shape_keys.key_blocks[key_name].value = value
            obj.data.update()
            context.view_layer.update()

        # Export to CSV
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_data)

        return True, f"Measurements exported to {filepath}"
    except Exception as e:
        return False, f"Failed to export measurements: {str(e)}"


def sort_shape_keys(obj):
    """
    Sort all shape keys alphabetically (except for the Basis shape key which stays first).

    Args:
    - obj (bpy.types.Object): The mesh object with shape keys to sort
    """
    if not obj.data.shape_keys or len(obj.data.shape_keys.key_blocks) <= 2:
        # No need to sort if there are 0, 1, or 2 shape keys (basis + 1)
        return

    # Get shape key names (excluding Basis)
    shape_keys = obj.data.shape_keys.key_blocks
    names = [key.name for key in shape_keys[1:]]

    # Sort the names
    sorted_names = sorted(names)

    # Rearrange shape keys using the proper Blender API
    for i, name in enumerate(sorted_names, 1):
        # Get current index of this shape key
        current_idx = shape_keys.find(name)
        # Move to the correct position (i) using the proper API
        if current_idx != i:
            # We need to use the shape_key_move operator
            bpy.context.view_layer.objects.active = obj
            obj.active_shape_key_index = current_idx
            # Move the shape key up or down until it's in the right position
            if current_idx < i:
                # Need to move down
                for _ in range(i - current_idx):
                    bpy.ops.object.shape_key_move(type="DOWN")
            else:
                # Need to move up
                for _ in range(current_idx - i):
                    bpy.ops.object.shape_key_move(type="UP")

    print(f"Sorted {len(sorted_names)} shape keys alphabetically")


def load_reference_measurements(filepath):
    """
    Load reference measurements from a CSV file.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        tuple: (joint_pair, measurements_dict) where joint_pair is a string describing the measured joints
               and measurements_dict is a dictionary mapping shape names to measurement values
    """
    try:
        measurements = {}
        joint_pair = ""

        with open(filepath, "r") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            # Extract joint pair from header
            if len(header) >= 2:
                # expecting the joint pair to be in the format "Joint1 to Joint2"
                joint_pair = header[1]

            # Read measurements
            for row in reader:
                if len(row) >= 2:
                    shape_name = row[0]
                    try:
                        measurement = float(row[1])
                        measurements[shape_name] = measurement
                    except ValueError:
                        print(f"Warning: Could not convert measurement for {shape_name} to float")

        return joint_pair, measurements
    except Exception as e:
        print(f"Error loading reference measurements: {e}")
        return "", {}


def get_reference_measurements(context):
    """Get the reference measurements from the temporary file"""
    if "reference_measurements_path" in context.scene:
        temp_path = context.scene["reference_measurements_path"]
        if os.path.exists(temp_path):
            with open(temp_path, "rb") as f:
                return pickle.load(f)
    return {}
