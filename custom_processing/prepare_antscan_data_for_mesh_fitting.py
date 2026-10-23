"""
Prepares AntScan STL meshes for mesh fitting.

Pipeline (process_stl):
    1. Merge near-duplicate vertices of very dense scans (reduce_vertices_by_distance).
    2. Keep the largest connected component of the input scan.
    3. Remove internal geometry by ray casting from a surrounding sphere, keeping the surface
       visible from outside (clean_internal_geometry). Islands separated by sparse ray
       coverage are reconnected along short paths of the original mesh
       (bridge_nearby_islands); components below a minimum face count are removed.
    4. Edge split, weld, hole filling and limited dissolve (apply_modifiers). The weld distance
       adapts to the bounding-box size and to the local edge length; a weld that destroys more
       than a set fraction of faces raises an error.
    5. Remove small components, decimate to a vertex budget, and align the mesh to its principal
       axes with legs down, positive Z up and the head along positive X.
    6. Export a triangulated OBJ and write mesh statistics into the JSON file next to the input.

Usage:
    blender --background --python prepare_antscan_data_for_mesh_fitting.py -- <input_stl_path> <output_dir>
"""

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
from collections import deque


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


def _median_edge_length(obj):
    """
    Computes the median edge length of the object's current mesh.

    This provides a measure of local mesh density that, unlike overall
    bounding-box size, reflects how finely or sparsely the mesh is
    triangulated.

    Args:
        obj (bpy.types.Object): The Blender object to measure.

    Returns:
        float: The median edge length, or 0.0 if the mesh has no edges.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    lengths = sorted(edge.calc_length() for edge in bm.edges)
    bpy.ops.object.mode_set(mode="OBJECT")

    if not lengths:
        return 0.0
    mid = len(lengths) // 2
    if len(lengths) % 2:
        return lengths[mid]
    return (lengths[mid - 1] + lengths[mid]) / 2


def apply_modifiers(
    obj,
    edge_split_angle=1.5708,
    weld_merge_threshold=None,
    dissolve_angle_limit=0.01,
    fill_holes_sides=0,
    min_island_faces=4,
    max_weld_face_loss_pct=20.0,
):
    """
    Applies a series of modifiers and mesh operations to simplify and clean
    the given object.

    Args:
        obj (bpy.types.Object): The Blender object to modify.
        edge_split_angle (float): Angle threshold for the Edge Split modifier
            (radians).
        weld_merge_threshold (float): Distance threshold for the Weld
            modifier. If None, it is the smaller of 0.2% of the largest
            bounding-box dimension and 30% of the median edge length.
        dissolve_angle_limit (float): Angle limit for the Limited Dissolve
            operation.
        fill_holes_sides (int): Maximum number of sides a hole may have to be
            filled. 0 means no limit.
        min_island_faces (int): Connected components with fewer faces than
            this are removed before welding.
        max_weld_face_loss_pct (float): A RuntimeError is raised if the Weld
            step removes more than this percentage of faces.

    Returns:
        None
    """
    if weld_merge_threshold is None:
        # Weld distance: the smaller of a bounding-box-relative and an edge-length-relative bound.
        bbox_size = obj.dimensions
        max_dimension = max(bbox_size)
        bbox_threshold = max_dimension * 0.002  # 0.2 % of the largest dimension
        print(f"Bounding box size: {bbox_size}, bbox-based threshold: {bbox_threshold}")

        median_edge = _median_edge_length(obj)
        local_threshold = median_edge * 0.3  # 30 % of local median edge length
        print(f"Median edge length: {median_edge}, local-density-based threshold: {local_threshold}")

        if median_edge > 0:
            weld_merge_threshold = min(bbox_threshold, local_threshold)
        else:
            weld_merge_threshold = bbox_threshold

        if weld_merge_threshold <= 0:
            print("Warning: Calculated weld_merge_threshold is 0 or negative. Using default value.")
            weld_merge_threshold = 2

        print(f"Using weld_merge_threshold: {weld_merge_threshold}")

    # Apply Edge Split modifier
    edge_split = obj.modifiers.new(name="EdgeSplit", type="EDGE_SPLIT")
    edge_split.split_angle = edge_split_angle
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Remove small components before welding.
    removed_islands = filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) before Weld")

    # Apply Weld modifier
    face_count_before_weld = len(obj.data.polygons)
    weld = obj.modifiers.new(name="Weld", type="WELD")
    weld.merge_threshold = weld_merge_threshold
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Weld")
    face_count_after_weld = len(obj.data.polygons)

    if face_count_before_weld > 0:
        face_loss_pct = 100 * (face_count_before_weld - face_count_after_weld) / face_count_before_weld
        print(f"Weld: {face_count_before_weld} -> {face_count_after_weld} faces ({face_loss_pct:.1f}% loss)")
        if face_loss_pct > max_weld_face_loss_pct:
            raise RuntimeError(
                f"Weld step destroyed {face_loss_pct:.1f}% of faces "
                f"({face_count_before_weld} -> {face_count_after_weld}), exceeding the "
                f"{max_weld_face_loss_pct}% threshold. This indicates that Weld merged "
                f"vertices across disconnected mesh regions rather than within a single "
                f"surface. Aborting rather than continuing with a corrupted mesh."
            )

    # Switch to Edit Mode for mesh operations
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)

    # Fill holes
    bmesh.ops.holes_fill(bm, edges=bm.edges, sides=fill_holes_sides)

    # Limited Dissolve
    bmesh.ops.dissolve_limit(
        bm,
        angle_limit=dissolve_angle_limit,
        use_dissolve_boundaries=False,
        verts=bm.verts,
        edges=bm.edges,
    )

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")
    bm.free()


def _bfs_path(start_vert, target_verts, max_hops=60):
    """
    Performs a breadth-first search through the mesh edge graph starting from
    start_vert, stopping as soon as a vertex belonging to target_verts is
    reached or max_hops is exceeded.

    Args:
        start_vert (BMVert): Vertex from which the search begins.
        target_verts (set): Set of BMVert objects that constitute valid targets.
        max_hops (int): Maximum number of edge hops permitted before giving up.

    Returns:
        list[BMVert] or None: The path from start_vert to the reached target
        (inclusive of both ends), or None if no path exists within max_hops.
    """
    parent = {start_vert: None}
    depth = {start_vert: 0}
    queue = deque([start_vert])
    while queue:
        current = queue.popleft()
        if current in target_verts and current is not start_vert:
            path = []
            v = current
            while v is not None:
                path.append(v)
                v = parent[v]
            return path
        if depth[current] >= max_hops:
            continue
        for edge in current.link_edges:
            neighbor = edge.other_vert(current)
            if neighbor not in parent:
                parent[neighbor] = current
                depth[neighbor] = depth[current] + 1
                queue.append(neighbor)
    return None


def bridge_nearby_islands(
    bm,
    vertices_to_keep,
    min_bridge_island_faces=50,
    prefilter_gap_multiplier=25,
    max_bridge_hops=20,
    patch_rings=2,
):
    """
    Reconnects islands of the kept surface that are joined in the original mesh.

    The faces whose vertices are all in vertices_to_keep form islands. For each
    pair of islands with at least min_bridge_island_faces faces whose bounding
    boxes and closest boundary vertices lie within prefilter_gap_multiplier
    median edge lengths, a breadth-first search through the original mesh runs
    from the boundary vertex of the first island closest to the second island
    to the nearest vertex of the second island. If the path has at most
    max_bridge_hops edges, its vertices and patch_rings rings of neighbours are
    added to vertices_to_keep, restoring existing faces of the original mesh.

    Args:
        bm (BMesh): The complete mesh before any vertices are deleted.
        vertices_to_keep (set): Set of vertex indices marked to survive;
            mutated in place.
        min_bridge_island_faces (int): Minimum face count required for an
            island to be considered a bridging candidate.
        prefilter_gap_multiplier (float): Maximum bounding-box gap and closest
            boundary-vertex distance between two islands, as a multiple of the
            median edge length.
        max_bridge_hops (int): Maximum number of edges in a bridging path.
        patch_rings (int): Number of vertex rings added around the path.

    Returns:
        int: Number of island pairs that were bridged.
    """
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    kept_faces = [f for f in bm.faces if all(v.index in vertices_to_keep for v in f.verts)]
    unvisited = set(kept_faces)
    islands = []
    while unvisited:
        seed = unvisited.pop()
        island_faces = {seed}
        to_visit = [seed]
        while to_visit:
            current = to_visit.pop()
            for edge in current.edges:
                for linked_face in edge.link_faces:
                    if linked_face in unvisited:
                        unvisited.remove(linked_face)
                        island_faces.add(linked_face)
                        to_visit.append(linked_face)
        island_verts = set()
        for f in island_faces:
            island_verts.update(f.verts)
        islands.append((island_verts, island_faces))

    real_islands = [(verts, faces) for verts, faces in islands if len(faces) >= min_bridge_island_faces]
    if len(real_islands) < 2:
        return 0

    def boundary_verts(verts, faces):
        """Vertices on edges shared with faces outside the given face subset."""
        result = set()
        for f in faces:
            for edge in f.edges:
                linked_in_island = sum(1 for lf in edge.link_faces if lf in faces)
                if linked_in_island < len(edge.link_faces):
                    result.update(edge.verts)
        return result if result else verts

    real_islands = [(verts, boundary_verts(verts, faces)) for verts, faces in real_islands]

    edge_lengths = sorted(e.calc_length() for e in bm.edges)
    median_edge = edge_lengths[len(edge_lengths) // 2] if edge_lengths else 1.0
    prefilter_threshold = prefilter_gap_multiplier * median_edge

    def bbox(verts):
        xs = [v.co.x for v in verts]
        ys = [v.co.y for v in verts]
        zs = [v.co.z for v in verts]
        return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))

    boxes = [bbox(boundary) for _, boundary in real_islands]

    def bbox_gap(box_a, box_b):
        gap = 0.0
        for axis in range(3):
            lo = max(box_a[0][axis], box_b[0][axis]) - min(box_a[1][axis], box_b[1][axis])
            gap += max(lo, 0.0) ** 2
        return math.sqrt(gap)

    bridged = 0
    for i in range(len(real_islands)):
        for j in range(i + 1, len(real_islands)):
            island_a_verts, boundary_a = real_islands[i]
            island_b_verts, boundary_b = real_islands[j]

            if bbox_gap(boxes[i], boxes[j]) > prefilter_threshold:
                continue

            best_dist, best_vert = None, None
            for va in boundary_a:
                for vb in boundary_b:
                    d = (va.co - vb.co).length
                    if best_dist is None or d < best_dist:
                        best_dist, best_vert = d, va

            if best_dist is None or best_dist > prefilter_threshold:
                continue

            path = _bfs_path(best_vert, island_b_verts, max_hops=max_bridge_hops)
            if path is None:
                continue

            patch_verts = set(path)
            frontier = set(path)
            for _ in range(patch_rings):
                next_frontier = set()
                for v in frontier:
                    for edge in v.link_edges:
                        neighbor = edge.other_vert(v)
                        if neighbor not in patch_verts:
                            patch_verts.add(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier

            for v in patch_verts:
                vertices_to_keep.add(v.index)

            bridged += 1
            print(
                f"Bridged islands ({len(island_a_verts)} and {len(island_b_verts)} verts): "
                f"gap={best_dist:.2f}, path_hops={len(path) - 1}, patch_verts={len(patch_verts)}"
            )

    return bridged


def clean_internal_geometry(
    obj,
    ray_density=1000,
    secondary_rays=50,
    random_seed=0,
    keep_rings=2,
    min_island_faces=4,
    min_bridge_island_faces=50,
    max_bridge_hops=20,
):
    """
    Removes internal geometry from the object by ray casting.

    Args:
        obj (bpy.types.Object): The Blender object to clean.
        ray_density (int): Density of primary rays.
        secondary_rays (int): Number of secondary rays cast for each primary ray.
        random_seed (int): Seed for random-number generation, ensuring
            reproducible results.
        keep_rings (int): Number of face rings kept around each ray-hit face.
        min_island_faces (int): Components with fewer faces than this are
            removed after ray-cast cleaning.
        min_bridge_island_faces (int): Minimum face count of an island
            considered by bridge_nearby_islands.
        max_bridge_hops (int): Maximum path length used by
            bridge_nearby_islands.

    Returns:
        None
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Bounding box in world space
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_min = Vector(map(min, zip(*bbox_corners)))
    bbox_max = Vector(map(max, zip(*bbox_corners)))

    center = (bbox_max + bbox_min) / 2
    radius = (bbox_max - bbox_min).length * 2  # ray origins lie on a sphere of twice the bounding-box diagonal

    def cast_ray(origin, direction):
        """Cast a ray and return hit status together with the face index."""
        hit, loc, norm, face_index = obj.ray_cast(obj.matrix_world.inverted() @ origin, direction)
        return hit, face_index

    def add_face_and_connected(face, vertices_to_keep):
        """Add the vertices of face and of all faces within keep_rings rings of it."""
        frontier = {face}
        visited_faces = {face}
        for _ in range(keep_rings):
            next_frontier = set()
            for f in frontier:
                for edge in f.edges:
                    for linked_face in edge.link_faces:
                        if linked_face not in visited_faces:
                            visited_faces.add(linked_face)
                            next_frontier.add(linked_face)
            frontier = next_frontier
            if not frontier:
                break
        for f in visited_faces:
            for vert in f.verts:
                vertices_to_keep.add(vert.index)

    vertices_to_keep = set()

    # Spherical distribution of primary rays
    phi = np.linspace(0, 2 * np.pi, int(np.sqrt(ray_density)))
    theta = np.linspace(0, np.pi, int(np.sqrt(ray_density)))

    for p in phi:
        for t in theta:
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)

            origin = center + Vector((x, y, z))
            main_direction = (center - origin).normalized()

            hit, face_index = cast_ray(origin, main_direction)
            if hit and face_index < len(bm.faces):
                face = bm.faces[face_index]
                add_face_and_connected(face, vertices_to_keep)

            for _ in range(secondary_rays):
                azimuth_offset = np.random.uniform(-np.pi / 9, np.pi / 9)  # +/- 20 degrees
                elevation_offset = np.random.uniform(-np.pi / 9, np.pi / 9)

                offset_direction = main_direction.copy()
                offset_direction.rotate(mathutils.Euler((elevation_offset, 0, azimuth_offset)))

                hit, face_index = cast_ray(origin, offset_direction)
                if hit and face_index < len(bm.faces):
                    face = bm.faces[face_index]
                    add_face_and_connected(face, vertices_to_keep)

    # Reconnect kept islands along short paths of the original mesh.
    bridged = bridge_nearby_islands(
        bm,
        vertices_to_keep,
        min_bridge_island_faces=min_bridge_island_faces,
        max_bridge_hops=max_bridge_hops,
    )
    print(f"Bridged {bridged} nearby island pairs via original-mesh shortest path")

    # Select vertices to keep
    for vert in bm.verts:
        vert.select = vert.index in vertices_to_keep

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="INVERT")

    bm.verts.ensure_lookup_table()
    verts_to_remove = [v for v in bm.verts if not v.select]
    bmesh.ops.delete(bm, geom=verts_to_remove, context="VERTS")

    bpy.ops.object.mode_set(mode="OBJECT")
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()

    print(f"Vertices kept (pre-island-filter): {len(vertices_to_keep)}")
    print(f"Total vertices after ray-cast cleaning: {len(obj.data.vertices)}")

    removed_islands = filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces)")
    print(f"Total vertices after island filtering: {len(obj.data.vertices)}")


def find_largest_component(obj):
    """
    Retains only the largest connected component of the mesh.

    Args:
        obj (bpy.types.Object): The Blender object to process.

    Returns:
        None
    """
    bpy.ops.object.mode_set(mode="EDIT")

    mesh = bmesh.from_edit_mesh(obj.data)

    for v in mesh.verts:
        v.select = False
    mesh.verts.ensure_lookup_table()

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

    for v in largest_component:
        v.select = True

    bmesh.update_edit_mesh(obj.data)

    bpy.ops.mesh.select_all(action="INVERT")
    bpy.ops.mesh.delete(type="VERT")

    bpy.ops.object.mode_set(mode="OBJECT")


def filter_small_components(obj, min_faces=4):
    """
    Removes every connected component (by face adjacency) with fewer than
    min_faces faces and keeps all others.

    Args:
        obj (bpy.types.Object): The Blender object to process.
        min_faces (int): Minimum face count required for a component to survive.

    Returns:
        int: Number of components that were removed.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    unvisited = set(bm.faces)
    removed_components = 0
    verts_to_remove = set()

    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        to_visit = [seed]

        while to_visit:
            current = to_visit.pop()
            for edge in current.edges:
                for linked_face in edge.link_faces:
                    if linked_face in unvisited:
                        unvisited.remove(linked_face)
                        component.add(linked_face)
                        to_visit.append(linked_face)

        if len(component) < min_faces:
            removed_components += 1
            for face in component:
                verts_to_remove.update(face.verts)

    if verts_to_remove:
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=list(verts_to_remove), context="VERTS")
        bmesh.update_edit_mesh(obj.data)

    bpy.ops.object.mode_set(mode="OBJECT")
    return removed_components


def export_mesh_to_obj(obj, filepath):
    """
    Exports the given mesh object to an OBJ file after converting all faces to
    triangles.

    Args:
        obj (bpy.types.Object): Mesh object to export.
        filepath (str): Destination path for the OBJ file.

    Returns:
        str: The filepath that was written.
    """
    if obj.type != "MESH":
        raise TypeError("The selected object is not a mesh.")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")

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

    with open(filepath, "w") as file:
        for vert in vertices:
            file.write(f"v {vert.x} {vert.y} {vert.z}\n")
        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    return filepath


def count_holes(obj):
    """
    Counts the number of holes present in the given mesh object.

    Args:
        obj (bpy.types.Object): The Blender object to analyse.

    Returns:
        int: Number of holes in the mesh.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]

    hole_count = 0
    visited_edges = set()

    for start_edge in boundary_edges:
        if start_edge not in visited_edges:
            current_edge = start_edge
            is_hole = True
            loop_edges = []

            while current_edge not in visited_edges:
                visited_edges.add(current_edge)
                loop_edges.append(current_edge)

                next_vert = (
                    current_edge.verts[1]
                    if current_edge.verts[0] in current_edge.link_faces[0].verts
                    else current_edge.verts[0]
                )
                next_edges = [e for e in next_vert.link_edges if e in boundary_edges and e != current_edge]

                if not next_edges:
                    is_hole = False
                    break

                current_edge = next_edges[0]

                if current_edge == start_edge:
                    break

            if is_hole:
                hole_count += 1

    bpy.ops.object.mode_set(mode="OBJECT")
    return hole_count


def calculate_face_size_cov(obj):
    """
    Calculates the coefficient of variation of face areas in the given mesh.

    Args:
        obj (bpy.types.Object): The Blender object to analyse.

    Returns:
        float: Coefficient of variation of face areas, rounded to three decimal
            places.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    face_areas = [f.calc_area() for f in bm.faces]

    bpy.ops.object.mode_set(mode="OBJECT")

    return np.round(np.std(face_areas) / np.mean(face_areas), 3)


def calculate_mesh_smoothness(obj):
    """
    Calculates the average angle between adjacent face normals as a measure of
    mesh smoothness.

    Args:
        obj (bpy.types.Object): The Blender object to analyse.

    Returns:
        float: Average angle between face normals in degrees, rounded to three
            decimal places, or 0.0 if no comparisons are possible.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

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

    bpy.ops.object.mode_set(mode="OBJECT")

    if total_comparisons > 0:
        average_angle = total_angle / total_comparisons
        return np.round(average_angle, 3)
    return 0.0


def decimate_mesh(obj, max_vertices):
    """
    Iteratively decimates the mesh until the vertex count falls to or below
    max_vertices.

    Args:
        obj (bpy.types.Object): The Blender object to decimate.
        max_vertices (int): Target maximum number of vertices.

    Returns:
        int: Number of remaining vertices after decimation.
    """
    print("Applying mesh decimation...")
    initial_vertices = len(obj.data.vertices)
    current_vertices = initial_vertices
    iteration_count = 0
    last_vertex_count = current_vertices

    while current_vertices > max_vertices:
        modifier = obj.modifiers.new(name="Decimate", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.use_symmetry = False
        modifier.use_collapse_triangulate = True

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

        iteration_count += 1
        if current_vertices >= last_vertex_count or iteration_count > 10:
            print(f"Decimation stopped after {iteration_count} iterations.")
            break
        last_vertex_count = current_vertices

    return current_vertices


def reduce_vertices_by_distance(obj, target_vertices=1000000, max_iterations=100):
    """
    Iteratively increases the merge distance in order to reduce the number of
    vertices.

    Args:
        obj (bpy.types.Object): The Blender object to process.
        target_vertices (int): Desired maximum number of vertices.
        max_iterations (int): Maximum number of merge iterations.

    Returns:
        int: Final number of vertices.
    """
    initial_vertices = len(obj.data.vertices)
    if initial_vertices <= target_vertices:
        return initial_vertices

    bpy.context.view_layer.objects.active = obj

    for i in range(max_iterations):
        merge_distance = 1 * (2**i)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        bpy.ops.object.mode_set(mode="OBJECT")

        current_vertices = len(obj.data.vertices)
        print(f"Iteration {i + 1}: Merge distance = {merge_distance:.6f}, Vertices = {current_vertices}")

        if current_vertices <= target_vertices:
            break

    return current_vertices


def process_stl(
    stl_path,
    output_dir=None,
    max_vertices=20000,
    ray_density=1000,
    secondary_rays=5,
    random_seed=42,
    min_island_faces=4,
    keep_rings=2,
    min_bridge_island_faces=50,
    max_bridge_hops=20,
):
    """
    Processes an STL file by importing, cleaning, simplifying and decimating
    the mesh.

    Args:
        stl_path (str): Path of the STL file to process.
        output_dir (str, optional): Directory in which to save the processed
            mesh. If None, the mesh is saved next to the input file.
        max_vertices (int): Maximum number of vertices retained after
            decimation.
        ray_density (int): Density of primary rays used for internal-geometry
            cleaning.
        secondary_rays (int): Number of secondary rays used for internal-
            geometry cleaning.
        random_seed (int): Seed for random-number generation.
        min_island_faces (int): Minimum face count of a connected component,
            used by every small-component filter in the pipeline.
        keep_rings (int): Number of face rings kept around each ray-hit face.
        min_bridge_island_faces (int): Minimum face count of an island
            considered by bridge_nearby_islands.
        max_bridge_hops (int): Maximum path length used by
            bridge_nearby_islands.

    Returns:
        tuple: (remaining_vertices, hole_count, face_size_cov, mesh_smoothness)
    """
    # Import the STL file
    bpy.ops.wm.stl_import(filepath=stl_path)
    obj = bpy.context.selected_objects[0]

    # Reduce vertices if the mesh is extremely dense
    initial_vertices = len(obj.data.vertices)
    if initial_vertices > 2000000:
        print(f"Initial vertex count: {initial_vertices}. Reducing vertices...")
        reduced_vertices = reduce_vertices_by_distance(obj)
        print(f"Reduced vertex count: {reduced_vertices}")

    # Keep only the largest connected component
    find_largest_component(obj)

    # Centre the object
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")
    obj.location = Vector((0, 0, 0))

    # Remove internal geometry
    print("Cleaning internal geometry...")
    clean_internal_geometry(
        obj,
        ray_density,
        secondary_rays,
        random_seed,
        keep_rings=keep_rings,
        min_island_faces=min_island_faces,
        min_bridge_island_faces=min_bridge_island_faces,
        max_bridge_hops=max_bridge_hops,
    )

    # Apply simplification modifiers
    print("Applying simplification modifiers...")
    apply_modifiers(
        obj,
        fill_holes_sides=0,
        min_island_faces=min_island_faces,
    )

    # Remove small components left by welding and hole filling.
    removed_islands = filter_small_components(obj, min_faces=min_island_faces)
    print(f"Removed {removed_islands} debris islands (< {min_island_faces} faces) after apply_modifiers")

    # Decimate
    remaining_vertices = decimate_mesh(obj, max_vertices)

    # Align principal axis with the X-axis via PCA
    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    cov_matrix = np.cov(vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    rotation_matrix = Matrix(eigenvectors).to_4x4().inverted()
    obj.matrix_world = rotation_matrix @ obj.matrix_world
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    print("Aligned model with X-axis based on principal component analysis.")

    # Orient legs downward
    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    y_variance = np.var(vertices[:, 1])
    z_variance = np.var(vertices[:, 2])
    if y_variance < z_variance:
        rotation_matrix = Matrix.Rotation(np.pi / 2, 4, "X")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 90 degrees around X-axis to put legs down.")

    # Ensure positive Z is up
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    z_center = (z_min + z_max) / 2
    z_median = np.median(vertices[:, 2])
    if z_median < z_center:
        rotation_matrix = Matrix.Rotation(np.pi, 4, "X")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Flipped model 180 degrees around X-axis to ensure positive Z is up.")

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    print("Ensured positive Z is up.")

    # Determine head direction by comparing slice densities along X
    mesh = obj.data
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    num_slices = 20
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    slice_width = (x_max - x_min) / num_slices

    slice_densities = []
    for i in range(num_slices):
        slice_start = x_min + i * slice_width
        slice_end = slice_start + slice_width
        slice_vertices = vertices[(vertices[:, 0] >= slice_start) & (vertices[:, 0] < slice_end)]
        slice_volume = (
            slice_width
            * (slice_vertices[:, 1].max() - slice_vertices[:, 1].min())
            * (slice_vertices[:, 2].max() - slice_vertices[:, 2].min())
        )
        slice_density = len(slice_vertices) / slice_volume if slice_volume > 0 else 0
        slice_densities.append(slice_density)

    head_end = "start" if np.mean(slice_densities[:3]) < np.mean(slice_densities[-3:]) else "end"
    if head_end == "end":
        rotation_matrix = Matrix.Rotation(np.pi, 4, "Z")
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        print("Rotated model 180 degrees around Z-axis to ensure head is in positive X direction.")

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    remaining_vertices = len(obj.data.vertices)
    print(f"Number of remaining vertices: {remaining_vertices}")

    hole_count = count_holes(obj)
    print(f"Number of holes in the processed mesh: {hole_count}")

    face_size_cov = calculate_face_size_cov(obj)
    print(f"Coefficient of variation of face sizes: {face_size_cov}")

    mesh_smoothness = calculate_mesh_smoothness(obj)
    print(f"Average angle between face normals: {mesh_smoothness} degrees")

    if output_dir is None:
        output_dir = os.path.dirname(stl_path)
    os.makedirs(output_dir, exist_ok=True)

    original_file_name = os.path.splitext(os.path.basename(stl_path))[0]
    export_path = os.path.join(output_dir, f"{original_file_name}_processed.obj")

    print(f"Exporting the processed mesh to {export_path}...")
    export_mesh_to_obj(obj, export_path)
    print("Mesh exported successfully.")

    # Update accompanying JSON metadata if present
    json_path = os.path.splitext(stl_path)[0] + ".json"
    if os.path.exists(json_path):
        print(f"Updating JSON file: {json_path}")
        with open(json_path, "r") as f:
            json_data = json.load(f)

        json_data["processed_vertex_count"] = remaining_vertices
        json_data["processed_hole_count"] = hole_count
        json_data["processed_face_size_cov"] = face_size_cov
        json_data["processed_mesh_smoothness"] = mesh_smoothness

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print("JSON file updated successfully.")
    else:
        print(f"Warning: Corresponding JSON file not found at {json_path}")

    return remaining_vertices, hole_count, face_size_cov, mesh_smoothness


def main():
    start_time = time.time()

    if bpy.context.space_data is not None and bpy.context.space_data.type == "TEXT_EDITOR":
        # Running inside the Blender text editor
        stl_path = bpy.path.abspath(
            "/home/fabi/dev/SMILify/custom_processing/antscan_data/"
            "Acanthomyrmex_glabfemoralis_CASENT0744002/"
            "Acanthomyrmex_glabfemoralis_CASENT0744002.stl"
        )
        stl_path = bpy.path.abspath(
            "/home/fabi/dev/SMILify/custom_processing/antscan_data/"
            "Platythyrea_MG01_CASENT0840864-D4/"
            "Platythyrea_MG01_CASENT0840864-D4.stl"
        )
        output_dir = "/home/fabi/dev/SMILify/custom_processing/antscan_processed"
    else:
        # Running as a standalone script
        if len(sys.argv) < 3:
            print(
                "Usage: blender --background --python "
                "prepare_antscan_data_for_mesh_fitting.py -- "
                "<input_stl_path> <output_dir>"
            )
            sys.exit(1)
        stl_path = sys.argv[-2]
        output_dir = sys.argv[-1]

    vertex_count, hole_count, face_size_cov, mesh_smoothness = process_stl(
        stl_path,
        output_dir=output_dir,
        max_vertices=50000,
        ray_density=1000,
        secondary_rays=10000,
        random_seed=0,
    )
    print(f"Processed STL file. Final vertex count: {vertex_count}")
    print(f"Number of holes in the processed mesh: {hole_count}")
    print(f"Coefficient of variation of face sizes: {face_size_cov}")
    print(f"Mesh smoothness (average angle between face normals): {mesh_smoothness} degrees")

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Total processing time: {processing_time:.2f} seconds")


if __name__ == "__main__":
    main()
