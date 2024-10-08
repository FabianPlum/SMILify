import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
import math
import addon_utils
import numpy as np
import os
import time
import sys
import random
import json

def ensure_addon_enabled(addon_name):
    """
    Ensures that the specified Blender addon is enabled.

    Args:
        addon_name (str): The name of the addon to enable.

    Returns:
        None
    """
    if not addon_utils.check(addon_name)[0]:
        addon_utils.enable(addon_name, default_set=True)
        print(f"Enabled addon: {addon_name}")

def apply_modifiers(obj, edge_split_angle=1.5708, weld_merge_threshold=None, dissolve_angle_limit=0.01, fill_holes_sides=0):
    """
    Applies a series of modifiers and mesh operations to the given object to simplify and clean up the mesh.

    Args:
        obj (bpy.types.Object): The Blender object to modify.
        edge_split_angle (float): The angle threshold for the Edge Split modifier (in radians).
        weld_merge_threshold (float): If explicitly provided, the distance threshold for the Weld modifier. 
                                      If None, weld_merge_threshold is calculated based on the object's dimensions.
        dissolve_angle_limit (float): The angle limit for the Limited Dissolve operation.
        fill_holes_sides (int): The maximum number of sides a hole can have to be filled. 0 means no limit.

    Returns:
        None
    """
    if weld_merge_threshold is None:
        # Get the bounding box size
        bbox_size = obj.dimensions
        print(f"Bounding box size: {bbox_size}")

        # Calculate the weld_merge_threshold based on object size
        max_dimension = max(bbox_size)
        weld_merge_threshold = max_dimension * 0.002  # 0.2% of the largest dimension

        print(f"Calculated weld_merge_threshold: {weld_merge_threshold}")

        # Update the weld_merge_threshold parameter
        if weld_merge_threshold > 0:
            weld_merge_threshold = weld_merge_threshold
        else:
            print("Warning: Calculated weld_merge_threshold is 0 or negative. Using default value.")
            weld_merge_threshold = 2
    else:
        weld_merge_threshold = 2

    # Apply Edge Split Modifier
    edge_split = obj.modifiers.new(name="EdgeSplit", type='EDGE_SPLIT')
    edge_split.split_angle = edge_split_angle
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Apply Weld Modifier
    weld = obj.modifiers.new(name="Weld", type='WELD')
    weld.merge_threshold = weld_merge_threshold
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Weld")

    # Switch to Edit Mode for mesh operations
    bpy.ops.object.mode_set(mode='EDIT')

    # Create a BMesh
    bm = bmesh.from_edit_mesh(obj.data)

    # Fill Holes operation using bmesh.ops
    bmesh.ops.holes_fill(bm, edges=bm.edges, sides=fill_holes_sides)

    # Limited Dissolve operation
    bmesh.ops.dissolve_limit(bm, angle_limit=dissolve_angle_limit, use_dissolve_boundaries=False, verts=bm.verts, edges=bm.edges)

    # Update the mesh
    bmesh.update_edit_mesh(obj.data)

    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Free the BMesh
    bm.free()

def clean_internal_geometry(obj, ray_density=1000, secondary_rays=50, random_seed=0):
    """
    Cleans internal geometry of the object using ray casting.

    Args:
        obj (bpy.types.Object): The Blender object to clean.
        ray_density (int): The density of primary rays to cast.
        secondary_rays (int): The number of secondary rays to cast for each primary ray.
        random_seed (int): Seed for random number generation to ensure consistent results.

    Returns:
        None
    """
    # Set the random seed for numpy and Python's random module
    np.random.seed(random_seed)
    random.seed(random_seed)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')  # Ensure we're in Object mode
    
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    # Get the bounding box in world space
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_min = Vector(map(min, zip(*bbox_corners)))
    bbox_max = Vector(map(max, zip(*bbox_corners)))
    
    # Calculate bounding sphere
    center = (bbox_max + bbox_min) / 2
    radius = (bbox_max - bbox_min).length * 2 # four as large, so hard to sample corners are hit
    
    def cast_ray(origin, direction):
        """
        Casts a ray from the given origin in the given direction and returns hit information.

        Args:
            origin (Vector): The starting point of the ray.
            direction (Vector): The direction of the ray.

        Returns:
            tuple: A tuple containing hit (bool), location (Vector), and face_index (int).
        """
        hit, loc, norm, face_index = obj.ray_cast(obj.matrix_world.inverted() @ origin, direction)
        return hit, face_index
    
    def add_face_and_connected(face, vertices_to_keep):
        """
        Adds the given face and its immediately connected faces to the set of vertices to keep.

        Args:
            face (BMFace): The face to add.
            vertices_to_keep (set): The set of vertex indices to keep.

        Returns:
            None
        """
        for vert in face.verts:
            vertices_to_keep.add(vert.index)
        for edge in face.edges:
            for linked_face in edge.link_faces:
                if linked_face != face:
                    for vert in linked_face.verts:
                        vertices_to_keep.add(vert.index)
    
    # Set to store indices of vertices to keep
    vertices_to_keep = set()
    
    # Generate spherical distribution of rays
    phi = np.linspace(0, 2 * np.pi, int(np.sqrt(ray_density)))
    theta = np.linspace(0, np.pi, int(np.sqrt(ray_density)))
    
    for p in phi:
        for t in theta:
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)
            
            origin = center + Vector((x, y, z))
            main_direction = (center - origin).normalized()
            
            # Cast main ray
            hit, face_index = cast_ray(origin, main_direction)
            if hit and face_index < len(bm.faces):
                face = bm.faces[face_index]
                add_face_and_connected(face, vertices_to_keep)
            
            # Cast secondary rays
            for _ in range(secondary_rays):
                # Generate random offset angles
                azimuth_offset = np.random.uniform(-np.pi/9, np.pi/9)  # ±20 degrees
                elevation_offset = np.random.uniform(-np.pi/9, np.pi/9)  # ±20 degrees
                
                # Apply rotation to the main direction
                offset_direction = main_direction.copy()
                offset_direction.rotate(mathutils.Euler((elevation_offset, 0, azimuth_offset)))
                
                hit, face_index = cast_ray(origin, offset_direction)
                if hit and face_index < len(bm.faces):
                    face = bm.faces[face_index]
                    add_face_and_connected(face, vertices_to_keep)
    
    # Select vertices to keep
    for vert in bm.verts:
        vert.select = vert.index in vertices_to_keep
    
    # Invert selection
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='INVERT')
    
    # Delete unselected vertices
    bm.verts.ensure_lookup_table()
    verts_to_remove = [v for v in bm.verts if not v.select]
    bmesh.ops.delete(bm, geom=verts_to_remove, context='VERTS')
    
    # Update the mesh
    bpy.ops.object.mode_set(mode='OBJECT')  # Ensure we're in Object mode before updating
    bm.to_mesh(mesh)
    mesh.update()
    
    # Clean up
    bm.free()

    print(f"Vertices kept: {len(vertices_to_keep)}")
    print(f"Total vertices after cleaning: {len(mesh.vertices)}")

def find_largest_component(obj):
    """
    Finds and keeps only the largest connected component of the mesh.

    Args:
        obj (bpy.types.Object): The Blender object to process.

    Returns:
        None
    """
    bpy.ops.object.mode_set(mode='EDIT')
    
    mesh = bmesh.from_edit_mesh(obj.data)
    
    # Select all vertices
    for v in mesh.verts:
        v.select = False
    mesh.verts.ensure_lookup_table()
    
    # Find the largest coherent mesh
    unvisited = set(mesh.verts)
    largest_component = set()
    
    while unvisited:
        current = unvisited.pop()
        component = set([current])
        to_visit = set([current])
        
        while to_visit:
            current = to_visit.pop()
            for edge in current.link_edges:
                neighbor = edge.other_vert(current)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    component.add(neighbor)
                    to_visit.add(neighbor)
        
        if len(component) > len(largest_component):
            largest_component = component
    
    # Select the largest component
    for v in largest_component:
        v.select = True
    
    # Update the mesh
    bmesh.update_edit_mesh(obj.data)
    
    # Invert selection and delete vertices
    bpy.ops.mesh.select_all(action='INVERT')
    bpy.ops.mesh.delete(type='VERT')
    
    bpy.ops.object.mode_set(mode='OBJECT')

def export_mesh_to_obj(obj, filepath):
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

def count_holes(obj):
    """
    Counts the number of holes in the given mesh object.

    Args:
        obj (bpy.types.Object): The Blender object to analyze.

    Returns:
        int: The number of holes in the mesh.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    
    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    
    hole_count = 0
    visited_edges = set()
    
    for start_edge in boundary_edges:
        if start_edge not in visited_edges:
            # Start of a new hole
            current_edge = start_edge
            is_hole = True
            loop_edges = []
            
            while current_edge not in visited_edges:
                visited_edges.add(current_edge)
                loop_edges.append(current_edge)
                
                # Find the next edge in the boundary loop
                next_vert = current_edge.verts[1] if current_edge.verts[0] in current_edge.link_faces[0].verts else current_edge.verts[0]
                next_edges = [e for e in next_vert.link_edges if e in boundary_edges and e != current_edge]
                
                if not next_edges:
                    # We've reached an open end, not a hole
                    is_hole = False
                    break
                
                current_edge = next_edges[0]
                
                if current_edge == start_edge:
                    # We've completed a loop
                    break
            
            if is_hole:
                hole_count += 1
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return hole_count

def calculate_face_size_cov(obj):
    """
    Calculates the coefficient of variation of face sizes in the given mesh object.

    Args:
        obj (bpy.types.Object): The Blender object to analyze.

    Returns:
        float: The standard deviation of face sizes.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    
    face_areas = [f.calc_area() for f in bm.faces]
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return np.round(np.std(face_areas) / np.mean(face_areas), 3)

def calculate_mesh_smoothness(obj):
    """
    Calculates the average angle between face normals as a measure of mesh smoothness.

    Args:
        obj (bpy.types.Object): The Blender object to analyze.

    Returns:
        float: The average angle between face normals in degrees.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    
    total_angle = 0
    total_comparisons = 0
    
    for face in bm.faces:
        for edge in face.edges:
            adjacent_face = [f for f in edge.link_faces if f != face]
            if adjacent_face:
                angle = face.normal.angle(adjacent_face[0].normal)
                total_angle += math.degrees(angle)
                total_comparisons += 1
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    if total_comparisons > 0:
        average_angle = total_angle / total_comparisons
        return np.round(average_angle, 3)
    else:
        return 0.0

def decimate_mesh(obj, max_vertices):
    """
    Decimates the mesh to reduce the number of vertices.

    Args:
        obj (bpy.types.Object): The Blender object to decimate.
        max_vertices (int): The target maximum number of vertices.

    Returns:
        int: The number of remaining vertices after decimation.
    """
    print("Applying mesh decimation...")
    initial_vertices = len(obj.data.vertices)
    current_vertices = initial_vertices
    iteration_count = 0
    last_vertex_count = current_vertices

    while current_vertices > max_vertices:
        modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
        modifier.decimate_type = 'COLLAPSE'
        modifier.use_symmetry = False
        modifier.use_collapse_triangulate = True

        # Apply decimation with a ratio of 0.5 if more than twice the target vertices
        if current_vertices / max_vertices > 2:
            modifier.ratio = 0.5
        else:
            modifier.ratio = max_vertices / current_vertices

        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.modifier_apply(modifier="Decimate")
        except RuntimeError as e:
            if "Modifiers cannot be applied to multi-user data" in str(e):
                print("Making mesh data single-user and retrying...")
                bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)
                bpy.ops.object.modifier_apply(modifier="Decimate")
            else:
                raise

        current_vertices = len(obj.data.vertices)
        print(f"Current vertices after decimation: {current_vertices}")

        # Edge case detection
        iteration_count += 1
        if current_vertices >= last_vertex_count or iteration_count > 10:
            print(f"Decimation stopped after {iteration_count} iterations.")
            break
        last_vertex_count = current_vertices

    return current_vertices

def reduce_vertices_by_distance(obj, target_vertices=1000000, max_iterations=100):
    """
    Iteratively increases the merge distance to reduce the number of vertices.

    Args:
        obj (bpy.types.Object): The Blender object to process.
        target_vertices (int): The target number of vertices.
        max_iterations (int): Maximum number of iterations to attempt.

    Returns:
        int: The final number of vertices.
    """
    initial_vertices = len(obj.data.vertices)
    if initial_vertices <= target_vertices:
        return initial_vertices

    bpy.context.view_layer.objects.active = obj
    
    for i in range(max_iterations):
        merge_distance = 1 * (2 ** i)  # Exponentially increase merge distance
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        current_vertices = len(obj.data.vertices)
        print(f"Iteration {i+1}: Merge distance = {merge_distance:.6f}, Vertices = {current_vertices}")
        
        if current_vertices <= target_vertices:
            break
    
    return current_vertices

def process_stl(stl_path, output_dir=None, max_vertices=20000, ray_density=1000, secondary_rays=5, random_seed=42):
    """
    Processes an STL file by importing, cleaning, simplifying, and decimating the mesh.

    Args:
        stl_path (str): The file path of the STL file to process.
        output_dir (str, optional): The directory to save the processed mesh. If None, saves in the same directory as the input file.
        max_vertices (int): The maximum number of vertices to keep after decimation.
        ray_density (int): The density of primary rays for internal geometry cleaning.
        secondary_rays (int): The number of secondary rays for internal geometry cleaning.
        random_seed (int): Seed for random number generation to ensure consistent results.

    Returns:
        int: The number of remaining vertices after processing.
    """
    # Import the STL file
    bpy.ops.wm.stl_import(filepath=stl_path)
    obj = bpy.context.selected_objects[0]
    
    # Reduce vertices if necessary
    initial_vertices = len(obj.data.vertices)
    if initial_vertices > 2000000:
        print(f"Initial vertex count: {initial_vertices}. Reducing vertices...")
        reduced_vertices = reduce_vertices_by_distance(obj)
        print(f"Reduced vertex count: {reduced_vertices}")
    
    # Find and keep only the largest component
    find_largest_component(obj)

    # Set the origin to the center of mass
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')

    # Set the object's location to the world origin
    obj.location = Vector((0, 0, 0))

    # Clean internal geometry using ray casting
    print("Cleaning internal geometry...")
    clean_internal_geometry(obj, ray_density, secondary_rays, random_seed)

    # Apply simplification modifiers
    print("Applying simplification modifiers...")
    apply_modifiers(obj, fill_holes_sides=0)  # 0 means fill all holes regardless of size

    # Reapply the largest component to remove floating artifacts
    find_largest_component(obj)
    
    # Apply mesh decimation
    remaining_vertices = decimate_mesh(obj, max_vertices)
    
    # Get mesh data
    mesh = obj.data
    
    # Convert vertices to numpy array for easier computation
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(vertices.T)
    
    # Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    
    # Create rotation matrix to align principal axis with X-axis
    rotation_matrix = Matrix(eigenvectors).to_4x4().inverted()
    
    # Apply rotation
    obj.matrix_world = rotation_matrix @ obj.matrix_world
    
    # Apply the rotation to make it permanent
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    
    print("Aligned model with X-axis based on principal component analysis.")
    
    # Get mesh data again after the initial alignment
    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])

    # Calculate the variance along Y and Z axes
    y_variance = np.var(vertices[:, 1])
    z_variance = np.var(vertices[:, 2])

    # Determine if we need to rotate 90 degrees around X-axis
    if y_variance < z_variance:
        rotation_matrix = Matrix.Rotation(np.pi/2, 4, 'X')
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 90 degrees around X-axis to put legs down.")

    # Ensure the "up" direction is positive Z
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_center = (z_min + z_max) / 2
    z_median = np.median(vertices[:, 2])

    if z_median < z_center:
        rotation_matrix = Matrix.Rotation(np.pi, 4, 'X')
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Flipped model 180 degrees around X-axis to ensure positive Z is up.")

    # Apply the rotations to make them permanent
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # After ensuring positive Z is up
    print("Ensured positive Z is up.")

    # Get mesh data again
    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])

    # Divide the model into slices along the X-axis
    num_slices = 20
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    slice_width = (x_max - x_min) / num_slices
    
    slice_densities = []
    for i in range(num_slices):
        slice_start = x_min + i * slice_width
        slice_end = slice_start + slice_width
        slice_vertices = vertices[(vertices[:, 0] >= slice_start) & (vertices[:, 0] < slice_end)]
        
        # Calculate the density of the slice (number of vertices / volume)
        slice_volume = slice_width * (slice_vertices[:, 1].max() - slice_vertices[:, 1].min()) * (slice_vertices[:, 2].max() - slice_vertices[:, 2].min())
        slice_density = len(slice_vertices) / slice_volume if slice_volume > 0 else 0
        slice_densities.append(slice_density)

    # The end with lower density is likely to be the antennae end (head)
    head_end = 'start' if np.mean(slice_densities[:3]) < np.mean(slice_densities[-3:]) else 'end'

    if head_end == 'end':
        rotation_matrix = Matrix.Rotation(np.pi, 4, 'Z')
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 180 degrees around Z-axis to ensure head is in positive X direction.")

    # Apply the rotation to make it permanent
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Report the number of remaining vertices
    remaining_vertices = len(obj.data.vertices)
    print(f"Number of remaining vertices: {remaining_vertices}")
    
    # After all processing steps and before exporting
    hole_count = count_holes(obj)
    print(f"Number of holes in the processed mesh: {hole_count}")

    # Calculate standard deviation of face sizes
    face_size_cov = calculate_face_size_cov(obj)
    print(f"Coefficient of variation of face sizes: {face_size_cov}")

    # Calculate mesh smoothness
    mesh_smoothness = calculate_mesh_smoothness(obj)
    print(f"Average angle between face normals: {mesh_smoothness} degrees")

    # Determine the output directory
    if output_dir is None:
        output_dir = os.path.dirname(stl_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if output_dir is None:
        output_dir = os.path.dirname(stl_path)
    os.makedirs(output_dir, exist_ok=True)
    
    original_file_name = os.path.splitext(os.path.basename(stl_path))[0]
    export_path = os.path.join(output_dir, f"{original_file_name}_processed.obj")
    
    print(f"Exporting the processed mesh to {export_path}...")
    export_mesh_to_obj(obj, export_path)
    print("Mesh exported successfully.")

    # Update the corresponding JSON file with vertex and hole count
    json_path = os.path.splitext(stl_path)[0] + '.json'
    if os.path.exists(json_path):
        print(f"Updating JSON file: {json_path}")
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        json_data['processed_vertex_count'] = remaining_vertices
        json_data['processed_hole_count'] = hole_count
        json_data['processed_face_size_cov'] = face_size_cov
        json_data['processed_mesh_smoothness'] = mesh_smoothness
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print("JSON file updated successfully.")
    else:
        print(f"Warning: Corresponding JSON file not found at {json_path}")
    
    return remaining_vertices, hole_count, face_size_cov, mesh_smoothness  # Return vertex count, hole count, face size cov, and mesh smoothness

def main():
    start_time = time.time()  # Start the timer

    # Check if the script is run from Blender's text editor
    if bpy.context.space_data is not None and bpy.context.space_data.type == 'TEXT_EDITOR':
        # Running within Blender
        stl_path = bpy.path.abspath("/home/fabi/dev/SMILify/custom_processing/antscan_data/Acanthomyrmex_glabfemoralis_CASENT0744002/Acanthomyrmex_glabfemoralis_CASENT0744002.stl")  # Update this path
        stl_path = bpy.path.abspath("/home/fabi/dev/SMILify/custom_processing/antscan_data/Platythyrea_MG01_CASENT0840864-D4/Platythyrea_MG01_CASENT0840864-D4.stl")  # Update this path
        output_dir = "/home/fabi/dev/SMILify/custom_processing/antscan_processed"  # if not provided, saves in the same directory as the input file
    else:
        # Running as a standalone script
        if len(sys.argv) < 3:
            print("Usage: blender --background --python prepare_antscan_data_for_mesh_fitting.py -- <input_stl_path> <output_dir>")
            sys.exit(1)
        stl_path = sys.argv[-2]
        output_dir = sys.argv[-1]

    vertex_count, hole_count, face_size_cov, mesh_smoothness = process_stl(stl_path, output_dir=output_dir, max_vertices=50000, ray_density=1000, secondary_rays=10000, random_seed=0)
    print(f"Processed STL file. Final vertex count: {vertex_count}")
    print(f"Number of holes in the processed mesh: {hole_count}")
    print(f"Coefficient of variation of face sizes: {face_size_cov}")
    print(f"Mesh smoothness (average angle between face normals): {mesh_smoothness} degrees")

    end_time = time.time()  # Stop the timer
    processing_time = end_time - start_time
    print(f"Total processing time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    main()
    
