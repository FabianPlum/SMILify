import bpy
import bmesh
import mathutils
from mathutils import Vector
import math
import addon_utils
import numpy as np
import os
import time

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

def apply_modifiers(obj, edge_split_angle=1.5708, weld_merge_threshold=None, dissolve_angle_limit=0.01):
    """
    Applies a series of modifiers to the given object to simplify and clean up the mesh.

    Args:
        obj (bpy.types.Object): The Blender object to modify.
        edge_split_angle (float): The angle threshold for the Edge Split modifier (in radians).
        weld_merge_threshold (float): If explicitly provided, the distance threshold for the Weld modifier. 
                                      If None, weld_merge_threshold is calculated based on the object's dimensions.
        dissolve_angle_limit (float): The angle limit for the Limited Dissolve operation.

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

    # Apply Limited Dissolve in Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.dissolve_limited(angle_limit=dissolve_angle_limit)
    bpy.ops.object.mode_set(mode='OBJECT')

def clean_internal_geometry(obj, ray_density=1000, secondary_rays=50):
    """
    Cleans internal geometry of the object using ray casting.

    Args:
        obj (bpy.types.Object): The Blender object to clean.
        ray_density (int): The density of primary rays to cast.
        secondary_rays (int): The number of secondary rays to cast for each primary ray.

    Returns:
        None
    """
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
        return hit, loc, face_index
    
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
            hit, loc, face_index = cast_ray(origin, main_direction)
            if hit and face_index < len(bm.faces):
                face = bm.faces[face_index]
                add_face_and_connected(face, vertices_to_keep)
            
            # Cast secondary rays
            for _ in range(secondary_rays):
                # Generate random offset angles
                azimuth_offset = np.random.uniform(-np.pi/18, np.pi/18)  # ±10 degrees
                elevation_offset = np.random.uniform(-np.pi/18, np.pi/18)  # ±10 degrees
                
                # Apply rotation to the main direction
                offset_direction = main_direction.copy()
                offset_direction.rotate(mathutils.Euler((elevation_offset, 0, azimuth_offset)))
                
                hit, loc, face_index = cast_ray(origin, offset_direction)
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

def process_stl(stl_path, output_dir=None, max_vertices=20000, ray_density=1000, secondary_rays=5):
    """
    Processes an STL file by importing, cleaning, simplifying, and decimating the mesh.

    Args:
        stl_path (str): The file path of the STL file to process.
        output_dir (str, optional): The directory to save the processed mesh. If None, saves in the same directory as the input file.
        max_vertices (int): The maximum number of vertices to keep after decimation.
        ray_density (int): The density of primary rays for internal geometry cleaning.
        secondary_rays (int): The number of secondary rays for internal geometry cleaning.

    Returns:
        int: The number of remaining vertices after processing.
    """
    # Import the STL file
    bpy.ops.wm.stl_import(filepath=stl_path)
    obj = bpy.context.selected_objects[0]
    
    # Set the origin to the center of mass
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    
    # Set the object's location to the world origin
    obj.location = Vector((0, 0, 0))
    
    # Find and keep only the largest component
    find_largest_component(obj)

    # Clean internal geometry using ray casting
    print("Cleaning internal geometry...")
    clean_internal_geometry(obj, ray_density, secondary_rays)

    # Apply simplification modifiers
    print("Applying simplification modifiers...")
    apply_modifiers(obj)

    # Reapply the largest component to remove floating artifacts
    find_largest_component(obj)
    
    # Apply mesh decimation
    print("Applying mesh decimation...")
    initial_vertices = len(obj.data.vertices)
    current_vertices = initial_vertices

    if current_vertices > max_vertices:
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
    
    # Calculate the dimensions of the bounding box
    dimensions = obj.dimensions
    
    # Determine the longest axis
    longest_axis = max(range(3), key=lambda i: dimensions[i])
    
    # Calculate rotation to align longest axis with X-axis
    if longest_axis == 0:  # Already aligned with X-axis
        rotation = (0, 0, 0)
    elif longest_axis == 1:  # Y is longest, rotate -90 degrees around Z
        rotation = (0, 0, -math.pi/2)
    else:  # Z is longest, rotate 90 degrees around Y
        rotation = (0, math.pi/2, 0)
    
    # Apply the rotation
    obj.rotation_euler = rotation
    
    # Apply the rotation to make it permanent
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    
    # Report the number of remaining vertices
    remaining_vertices = len(obj.data.vertices)
    print(f"Number of remaining vertices: {remaining_vertices}")
    
    # Determine the output directory
    if output_dir is None:
        output_dir = os.path.dirname(stl_path)
    os.makedirs(output_dir, exist_ok=True)
    
    original_file_name = os.path.splitext(os.path.basename(stl_path))[0]
    export_path = os.path.join(output_dir, f"{original_file_name}_processed.obj")
    
    print(f"Exporting the processed mesh to {export_path}...")
    export_mesh_to_obj(obj, export_path)
    print("Mesh exported successfully.")
    
    return remaining_vertices

def main():
    start_time = time.time()  # Start the timer

    # Example usage
    stl_path = "/home/fabi/dev/SMILify/custom_processing/antscan_data/Platythyrea_MG01_CASENT0840864-D4/Platythyrea_MG01_CASENT0840864-D4.stl"
    output_dir = "/home/fabi/dev/SMILify/custom_processing/antscan_processed"
    vertex_count = process_stl(stl_path, output_dir=output_dir, max_vertices=50000, ray_density=10000, secondary_rays=2000)
    print(f"Processed STL file. Final vertex count: {vertex_count}")

    end_time = time.time()  # Stop the timer
    processing_time = end_time - start_time
    print(f"Total processing time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    main()
