import numpy as np
import config
from scipy.spatial import KDTree


def compute_symmetric_pairs(vertices, axis="y", tolerance=0.01):
    """
    Compute symmetric pairs of vertices based on their coordinates and the specified symmetry axis.
    Allow for a specified percentage deviation (tolerance) from the exact mirrored position using KDTree.
    """
    sym_pairs = []
    sym_axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    tolerance_value = np.max(np.abs(vertices)) * tolerance

    # Reflect vertices along the symmetry axis
    reflected_vertices = vertices.copy()
    reflected_vertices[:, sym_axis_idx] *= -1

    # Build KDTree for the reflected vertices
    tree = KDTree(reflected_vertices)

    # Find symmetric pairs within the tolerance
    for idx, vertex in enumerate(vertices):
        dist, idx_sym = tree.query(vertex, distance_upper_bound=tolerance_value)
        if dist < tolerance_value:
            sym_pairs.append((idx, idx_sym))

    return np.array(sym_pairs)


def rebuild_symmetry_array(vertices_on_symmetry_axis, all_vertices, axis="y", tolerance=0.001):
    # Initialize the symmetry array
    symIdx = np.arange(len(all_vertices))

    # Set the indices for vertices on the symmetry axis to point to themselves
    for idx in vertices_on_symmetry_axis:
        symIdx[idx] = idx

    # Compute symmetrical vertex pairs
    symmetrical_vertex_pairs = compute_symmetric_pairs(all_vertices, axis, tolerance)

    # Set the indices for symmetrical vertex pairs
    for pair in symmetrical_vertex_pairs:
        symIdx[pair[0]] = pair[1]
        symIdx[pair[1]] = pair[0]

    return symIdx


def align_smal_template_to_symmetry_axis(v, I):
    # I: indices of the vertices on the symmetry axis (the model's sym_verts)
    v = v - np.mean(v)
    y = np.mean(v[I, 1])
    v[:, 1] = v[:, 1] - y
    v[I, 1] = 0

    """
    # ORIGINALLY
    
    left = v[:, 1] < 0
    right = v[:, 1] > 0
    center = v[:, 1] == 0
    """

    # WIP!!!
    center_tolerance = 0.01
    left = v[:, 1] <= -center_tolerance
    right = v[:, 1] >= center_tolerance
    center = ~(left | right)

    symIdx = rebuild_symmetry_array(vertices_on_symmetry_axis=I, all_vertices=v, axis="y", tolerance=0.001)

    v[left[symIdx]] = np.array([1, -1, 1]) * v[left]

    if config.DEBUG:
        print("\n left:", left)
        print("\n right:", right)
        print("\n center:", center, "\n")

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    if config.DEBUG:
        print("LEFT ELEMS:", left_inds.shape)
        print("RIGHT ELEMS:", right_inds.shape)
        print("CENTER ELEMS:", center_inds.shape, "\n")

    if len(left_inds) != len(right_inds):
        raise ValueError(
            f"Symmetry alignment found {len(left_inds)} left vs {len(right_inds)} right vertices; "
            "the template is not left/right symmetric about its sym_verts plane (y axis). "
            "Fix the model's sym_verts / mesh symmetry or set config.ignore_sym = True."
        )

    return v, left_inds, right_inds, center_inds
