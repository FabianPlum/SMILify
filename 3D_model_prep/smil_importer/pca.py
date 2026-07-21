"""PCA-based shape-space construction (scikit-learn, imported lazily)."""

import os
import csv

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.covariance import EmpiricalCovariance
except ImportError:  # pragma: no cover
    PCA = None
    EmpiricalCovariance = None


def apply_pca_and_create_shapekeys(
    scans,
    obj,
    num_components=10,
    overwrite_mesh=False,
    std_range=1,
    labels=None,
    output_dir=None,
):
    n, v, _ = scans.shape
    # Reshape the scans into (n, v*3)
    scans_reshaped = scans.reshape(n, v * 3)

    # Perform PCA
    pca = PCA(n_components=num_components)
    pca.fit(scans_reshaped)

    # Mean shape
    mean_shape = pca.mean_.reshape(v, 3)

    # get covariance matrix
    transformed_betas = pca.transform(scans_reshaped)
    COV = EmpiricalCovariance(assume_centered=False).fit(transformed_betas)
    cov_out = COV.covariance_
    mean_betas = COV.location_

    if overwrite_mesh:
        # Overwrite the mesh vertex coordinates with the mean shape
        for vert_index, vert in enumerate(mean_shape):
            obj.data.vertices[vert_index].co = vert
        # then add a basis shape key
        shape_key = obj.shape_key_add(name="Basis")
    else:
        # Add the mean shape as a shapekey
        if not obj.data.shape_keys:
            obj.shape_key_add(name="Basis")
        shape_key = obj.data.shape_keys.key_blocks["Basis"]
        for vert_index, vert in enumerate(mean_shape):
            shape_key.data[vert_index].co = vert

    # Principal components (reshape each component back to (v, 3))
    shapekeys = [component.reshape(v, 3) for component in pca.components_]

    # Standard deviations of the principal components
    std_devs = np.sqrt(pca.explained_variance_)

    # Add shapekeys as shape keys with min and max range
    for i, (shapekey, std_dev) in enumerate(zip(shapekeys, std_devs)):
        shape_key_name = f"PC_{i + 1}"
        shape_key = obj.shape_key_add(name=shape_key_name)

        # Calculate min and max range for the shape key
        min_range = -std_range * std_dev
        max_range = std_range * std_dev

        # Update the shape key vertex positions
        for j, vertex in enumerate(shapekey):
            shape_key.data[j].co = mean_shape[j] + vertex

        # Set min and max range for the shape key
        shape_key.slider_min = min_range
        shape_key.slider_max = max_range

    print(f"Created {num_components} PCA shapekeys with custom min and max ranges based on standard deviations.")
    # Optional: export XY (PC1, PC2) scatter data and PCA stats
    try:
        if output_dir is not None:
            if labels is None or len(labels) != scans.shape[0]:
                labels = [f"sample_{i}" for i in range(scans.shape[0])]
            # XY coordinates for first two PCs
            pc_xy_path = os.path.join(output_dir, "smil_shape_PC_xy.csv")
            with open(pc_xy_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["label", "PC1", "PC2"])
                for i, lab in enumerate(labels):
                    pc1 = transformed_betas[i, 0] if transformed_betas.shape[1] > 0 else 0.0
                    pc2 = transformed_betas[i, 1] if transformed_betas.shape[1] > 1 else 0.0
                    writer.writerow([lab, pc1, pc2])

            # PCA stats
            stats_path = os.path.join(output_dir, "smil_shape_PCA_stats.txt")
            with open(stats_path, "w") as f:
                f.write("PCA stats for shape-derived PCs\n")
                f.write(f"n_samples: {scans_reshaped.shape[0]}\n")
                f.write(f"n_features: {scans_reshaped.shape[1]}\n")
                f.write(f"n_components: {num_components}\n")
                f.write(f"explained_variance_ratio: {pca.explained_variance_ratio_.tolist()}\n")
                f.write(f"explained_variance: {pca.explained_variance_.tolist()}\n")
                f.write(f"singular_values: {pca.singular_values_.tolist()}\n")
                f.write(f"mean_l2_norm: {float(np.linalg.norm(pca.mean_))}\n")
                # Add per-shape PC weights (scores) needed to reproduce each input shape
                f.write("\npc_weights_per_shape (scores):\n")
                header = ",".join(
                    ["label"] + [f"PC{i + 1}" for i in range(min(num_components, transformed_betas.shape[1]))]
                )
                f.write(header + "\n")
                for i, lab in enumerate(labels):
                    weights = transformed_betas[i, :num_components]
                    weights_str = ",".join([f"{w}" for w in weights.tolist()])
                    f.write(f"{lab},{weights_str}\n")
            print(f"Shape PCA XY exported to {pc_xy_path}; stats to {stats_path}")
    except Exception as e:
        print(f"Failed exporting shape PCA XY/stats: {e}")
    return cov_out, mean_betas


def apply_entangled_pca_and_create_shapekeys(
    scans,
    scale_data,
    translation_data,
    obj,
    num_components=10,
    overwrite_mesh=False,
    std_range=1,
    labels=None,
    output_dir=None,
):
    import csv

    """
    Apply PCA to combined shape, scale, and translation features, then create shape keys.
    
    Args:
        scans: Vertex data (n, v, 3)
        scale_data: Scale data (n, j) - single scale factor per joint
        translation_data: Translation data (n, j, 3)
        obj: Blender mesh object
        num_components: Number of PCA components
        overwrite_mesh: Whether to overwrite mesh with mean shape
        std_range: Standard deviation range for shape keys
        labels: Labels for each sample
        output_dir: Output directory for CSV files
    
    Returns:
        tuple: (cov_out, mean_betas, scaledirs, transdirs)
    """
    n, v, _ = scans.shape
    n_joints = scale_data.shape[1]

    # Reshape vertex data to (n, v*3)
    vertex_features = scans.reshape(n, v * 3)

    # Reshape translation data to (n, j*3)
    translation_features = translation_data.reshape(n, n_joints * 3)

    # Combine all features: [vertex_features, scale_features, translation_features]
    combined_features = np.concatenate([vertex_features, scale_data, translation_features], axis=1)

    # Debug: Check each feature type separately with detailed statistics
    print("=== FEATURE RANGES BEFORE NORMALIZATION ===")
    print("Vertex features:")
    print(f"  Shape: {vertex_features.shape}")
    print(f"  Range: {np.min(vertex_features):.6f} to {np.max(vertex_features):.6f}")
    print(f"  Mean: {np.mean(vertex_features):.6f}, Std: {np.std(vertex_features):.6f}")
    print(f"  Min abs: {np.min(np.abs(vertex_features)):.6f}, Max abs: {np.max(np.abs(vertex_features)):.6f}")

    print("Scale data:")
    print(f"  Shape: {scale_data.shape}")
    print(f"  Range: {np.min(scale_data):.6f} to {np.max(scale_data):.6f}")
    print(f"  Mean: {np.mean(scale_data):.6f}, Std: {np.std(scale_data):.6f}")
    print(f"  Min abs: {np.min(np.abs(scale_data)):.6f}, Max abs: {np.max(np.abs(scale_data)):.6f}")

    print("Translation features:")
    print(f"  Shape: {translation_features.shape}")
    print(f"  Range: {np.min(translation_features):.6f} to {np.max(translation_features):.6f}")
    print(f"  Mean: {np.mean(translation_features):.6f}, Std: {np.std(translation_features):.6f}")
    print(f"  Min abs: {np.min(np.abs(translation_features)):.6f}, Max abs: {np.max(np.abs(translation_features)):.6f}")

    # Check for extreme values in each feature type
    vertex_extreme = np.sum(np.abs(vertex_features) > 1e6)
    scale_extreme = np.sum(np.abs(scale_data) > 1e6)
    translation_extreme = np.sum(np.abs(translation_features) > 1e6)

    if vertex_extreme > 0:
        print(f"Warning: Extreme vertex values found: {vertex_extreme} values")
    if scale_extreme > 0:
        print(f"Warning: Extreme scale values found: {scale_extreme} values")
    if translation_extreme > 0:
        print(f"Warning: Extreme translation values found: {translation_extreme} values")

    # Check if normalization is needed by comparing feature magnitudes
    vertex_magnitude = np.std(vertex_features)
    scale_magnitude = np.std(scale_data)
    translation_magnitude = np.std(translation_features)

    print("=== NORMALIZATION ASSESSMENT ===")
    print("Feature standard deviations:")
    print(f"  Vertex: {vertex_magnitude:.6f}")
    print(f"  Scale: {scale_magnitude:.6f}")
    print(f"  Translation: {translation_magnitude:.6f}")

    max_magnitude = max(vertex_magnitude, scale_magnitude, translation_magnitude)
    min_magnitude = min(vertex_magnitude, scale_magnitude, translation_magnitude)
    magnitude_ratio = max_magnitude / min_magnitude if min_magnitude > 0 else float("inf")

    print(f"Magnitude ratio (max/min): {magnitude_ratio:.2f}")
    if magnitude_ratio > 100:
        print("Normalization is RECOMMENDED - large magnitude differences detected")
    elif magnitude_ratio > 10:
        print("Normalization is ADVISABLE - moderate magnitude differences detected")
    else:
        print("Normalization may not be necessary - similar magnitudes")
    print("=== END FEATURE ANALYSIS ===")

    # Check for NaN values and handle them
    nan_mask = np.isnan(combined_features)
    if np.any(nan_mask):
        print(f"Warning: Found {np.sum(nan_mask)} NaN values in combined features")
        print(f"NaN locations: {np.where(nan_mask)}")
        # Replace NaN values with 0 (or could use mean imputation)
        combined_features = np.nan_to_num(combined_features, nan=0.0)
        print("Replaced NaN values with 0.0")

    # Skip normalization since feature magnitudes are similar (ratio: 5.60)
    print(f"Combined features shape: {combined_features.shape}")
    print(f"Combined features range: {np.min(combined_features):.6f} to {np.max(combined_features):.6f}")
    print("Skipping normalization - feature magnitudes are similar")

    # Perform PCA on normalized features
    pca = PCA(n_components=num_components)
    pca.fit(combined_features)

    # Since we didn't normalize, the PCA components are already in the original scale
    original_mean = pca.mean_  # This is the mean of the original data

    # Components are already in the correct scale since no normalization was applied
    pca_components_denorm = pca.components_

    # Debug: Check the magnitude of components
    print(
        f"PCA components magnitude range: {np.min(np.abs(pca_components_denorm)):.6f} to {np.max(np.abs(pca_components_denorm)):.6f}"
    )

    # Separate the original mean into shape, scale, and translation parts
    vertex_mean = original_mean[: v * 3].reshape(v, 3)
    # the scale and translation mean are already used to compute the mean mesh so they don't get re-applied.
    original_mean[v * 3 : v * 3 + n_joints]
    original_mean[v * 3 + n_joints :].reshape(n_joints, 3)

    # Get covariance matrix from transformed data
    transformed_betas = pca.transform(combined_features)

    # Debug: Check transformed_betas
    print(f"transformed_betas shape: {transformed_betas.shape}")
    print(f"transformed_betas range: {np.min(transformed_betas):.6f} to {np.max(transformed_betas):.6f}")
    print(
        f"transformed_betas sample values: {transformed_betas[0, :5] if transformed_betas.shape[1] > 5 else transformed_betas[0, :]}"
    )

    COV = EmpiricalCovariance(assume_centered=False).fit(transformed_betas)
    cov_out = COV.covariance_
    mean_betas = COV.location_

    # Update mesh with mean shape
    if overwrite_mesh:
        for vert_index, vert in enumerate(vertex_mean):
            obj.data.vertices[vert_index].co = vert
        if not obj.data.shape_keys:
            obj.shape_key_add(name="Basis")
    else:
        if not obj.data.shape_keys:
            obj.shape_key_add(name="Basis")
        shape_key = obj.data.shape_keys.key_blocks["Basis"]
        for vert_index, vert in enumerate(vertex_mean):
            shape_key.data[vert_index].co = vert

    # Separate PCA components into shape, scale, and translation parts
    shape_components = pca_components_denorm[:, : v * 3].reshape(num_components, v, 3)
    scale_components = pca_components_denorm[:, v * 3 : v * 3 + n_joints]  # (num_components, n_joints)
    translation_components = pca_components_denorm[:, v * 3 + n_joints :].reshape(num_components, n_joints, 3)

    # Create shape keys from shape components
    std_devs = np.sqrt(pca.explained_variance_)

    for i, (shapekey, std_dev) in enumerate(zip(shape_components, std_devs)):
        shape_key_name = f"PC_{i + 1}"
        shape_key = obj.shape_key_add(name=shape_key_name)

        min_range = -std_range * std_dev
        max_range = std_range * std_dev

        for j, vertex in enumerate(shapekey):
            shape_key.data[j].co = vertex_mean[j] + vertex

        shape_key.slider_min = min_range
        shape_key.slider_max = max_range

    print(
        f"Created {num_components} entangled PCA shapekeys with custom min and max ranges based on standard deviations."
    )

    # Prepare scaledirs and transdirs for export
    # Scale components: tile single values to 3D for compatibility
    scaledirs = np.tile(scale_components[:, :, np.newaxis], (1, 1, 3))  # (num_components, n_joints, 3)
    transdirs = translation_components  # (num_components, n_joints, 3)

    # Export XY coordinates and PCA stats if output_dir is provided
    try:
        if output_dir is not None:
            if labels is None or len(labels) != scans.shape[0]:
                labels = [f"sample_{i}" for i in range(scans.shape[0])]

            # XY coordinates for first two PCs
            pc_xy_path = os.path.join(output_dir, "smil_entangled_PC_xy.csv")

            # Debug: Check what we're about to write
            print("About to write PC XY data:")
            print(f"  Number of labels: {len(labels)}")
            print(f"  transformed_betas shape: {transformed_betas.shape}")
            print(f"  First few PC1 values: {transformed_betas[:3, 0] if transformed_betas.shape[1] > 0 else 'No PC1'}")
            print(f"  First few PC2 values: {transformed_betas[:3, 1] if transformed_betas.shape[1] > 1 else 'No PC2'}")

            with open(pc_xy_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["label", "PC1", "PC2"])
                for i, lab in enumerate(labels):
                    pc1 = transformed_betas[i, 0] if transformed_betas.shape[1] > 0 else 0.0
                    pc2 = transformed_betas[i, 1] if transformed_betas.shape[1] > 1 else 0.0
                    writer.writerow([lab, pc1, pc2])
                    print(f"  Writing: {lab}, {pc1}, {pc2}")

            # PCA stats
            stats_path = os.path.join(output_dir, "smil_entangled_PCA_stats.txt")
            with open(stats_path, "w") as f:
                f.write("PCA stats for entangled (shape+scale+translation) PCs\n")
                f.write(f"n_samples: {combined_features.shape[0]}\n")
                f.write(f"n_features: {combined_features.shape[1]}\n")
                f.write(f"n_components: {num_components}\n")
                f.write(f"explained_variance_ratio: {pca.explained_variance_ratio_.tolist()}\n")
                f.write(f"explained_variance: {pca.explained_variance_.tolist()}\n")
                f.write(f"singular_values: {pca.singular_values_.tolist()}\n")
                f.write(f"mean_l2_norm: {float(np.linalg.norm(original_mean))}\n")
                f.write("\npc_weights_per_shape (scores):\n")
                header = ",".join(
                    ["label"] + [f"PC{i + 1}" for i in range(min(num_components, transformed_betas.shape[1]))]
                )
                f.write(header + "\n")
                for i, lab in enumerate(labels):
                    weights = transformed_betas[i, :num_components]
                    weights_str = ",".join([f"{w}" for w in weights.tolist()])
                    f.write(f"{lab},{weights_str}\n")
            print(f"Entangled PCA XY exported to {pc_xy_path}; stats to {stats_path}")
    except Exception as e:
        print(f"Failed exporting entangled PCA XY/stats: {e}")

    # --- Export entangled morph data to CSV ---
    try:
        entangled_output_path = os.path.join(output_dir, "smil_morph_PC_data_entangled.csv")

        # Get joint names from the object's stored data
        joint_names = None

        # Try multiple methods to get joint names
        try:
            # Method 1: Check if object has J_names stored as custom property
            if hasattr(obj, "get") and "J_names" in obj:
                joint_names = obj["J_names"]
                print(f"Retrieved joint names from object custom property: {len(joint_names)} names")

            # Method 2: Try to get from armature
            if joint_names is None:
                armature = obj.find_armature()
                if armature and hasattr(armature, "data") and hasattr(armature.data, "bones"):
                    joint_names = [bone.name for bone in armature.data.bones]
                    print(f"Retrieved joint names from armature: {len(joint_names)} names")

            # Method 3: Try to get from stored SMIL data
            if joint_names is None and hasattr(obj, "get") and "smpl_data" in obj:
                smpl_data = obj["smpl_data"]
                if hasattr(smpl_data, "get") and "J_names" in smpl_data:
                    joint_names = smpl_data["J_names"]
                    print(f"Retrieved joint names from SMIL data: {len(joint_names)} names")

        except Exception as e:
            print(f"Error retrieving joint names: {e}")

        # Fallback to generic names if we can't get proper joint names
        if joint_names is None or len(joint_names) != n_joints:
            joint_names = [f"joint_{j}" for j in range(n_joints)]
            print(
                f"Warning: Using generic joint names for entangled morph data export (expected {n_joints}, got {len(joint_names) if joint_names else 0})"
            )
        else:
            print(f"Successfully retrieved {len(joint_names)} joint names for entangled morph data export")

        with open(entangled_output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Header: joint_name, then for each PC six columns matching the original naming pattern
            header_entangled = ["joint_name"]
            for k in range(num_components):
                pc_label = f"PC_{k + 1}"
                header_entangled.extend(
                    [
                        f"{pc_label}_scale_x",
                        f"{pc_label}_scale_y",
                        f"{pc_label}_scale_z",
                        f"{pc_label}_translation_x",
                        f"{pc_label}_translation_y",
                        f"{pc_label}_translation_z",
                    ]
                )
            writer.writerow(header_entangled)

            # Data rows: one per joint, with PCA component values
            for j in range(n_joints):
                row = [joint_names[j]]
                for k in range(num_components):
                    # Add scale components (3 values)
                    row.extend(scaledirs[k, j, :].tolist())
                    # Add translation components (3 values)
                    row.extend(transdirs[k, j, :].tolist())
                writer.writerow(row)

        print(f"Entangled morph data exported to {entangled_output_path}")
    except Exception as e:
        print(f"Failed exporting entangled morph data: {e}")

    return cov_out, mean_betas, scaledirs, transdirs
