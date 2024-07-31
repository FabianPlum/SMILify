import os
import pickle as pkl
import numpy as np
import config
from scipy.spatial import KDTree

# from smal_model.smpl_webuser.serialization import load_model



def compute_symmetric_pairs(vertices, axis='y', tolerance=0.01):
    """
    Compute symmetric pairs of vertices based on their coordinates and the specified symmetry axis.
    Allow for a specified percentage deviation (tolerance) from the exact mirrored position using KDTree.
    """
    sym_pairs = []
    sym_axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
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

def rebuild_symmetry_array(vertices_on_symmetry_axis, all_vertices, axis='y', tolerance=0.001):
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

def align_smal_template_to_symmetry_axis(v, sym_file=None, I=None):
    # These are the indexes of the points that are on the symmetry axis
    # This is hard-coded for the dog model, so it doesn't apply to our stuff
    if I is None:
        I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32, 37, 55, 119, 120, 163, 209, 210, 211, 213, 216, 227, 326, 395, 452, 578, 910, 959, 964,
             975, 976, 977, 1172, 1175, 1176, 1178, 1194, 1243, 1739, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803,
             1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821,
             1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839,
             1840, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858,
             1859, 1860, 1861, 1862, 1863, 1870, 1919, 1960, 1961, 1965, 1967, 2003]

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


    if sym_file is not None:
        with open(sym_file, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            symIdx = u.load()
    else:
        # DEV -> compute vertex pairs:
        symIdx = rebuild_symmetry_array(vertices_on_symmetry_axis=I,
                                        all_vertices=v, axis='y',
                                        tolerance=0.001)

        print(symIdx)
        print(symIdx.shape)

    v[left[symIdx]] = np.array([1, -1, 1]) * v[left]

    if config.DEBUG:
        print("\n left:", left)
        print("\n right:", right)
        print("\n center:", center, "\n")

    """
    for e, elem in enumerate(zip(left, right, center, v)):
        print(e, elem)
    """

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    if config.DEBUG:
        print("LEFT ELEMS:", left_inds.shape)
        print("RIGHT ELEMS:", right_inds.shape)
        print("CENTER ELEMS:", center_inds.shape, "\n")

    try:
        assert (len(left_inds) == len(right_inds))
    except:
        import pdb;
        pdb.set_trace()

    return v, left_inds, right_inds, center_inds

# Legacy
# def get_smal_template(model_name, data_name, shape_family_id=-1):
#     model = load_model(model_name)
#     nBetas = len(model.betas.r)

#     with open(data_name, 'rb') as f:
#         u = pkl._Unpickler(f)
#         u.encoding = 'latin1'
#         data = u.load()

#     # Select average zebra/horse
#     # betas = data['cluster_means'][2][:nBetas]
#     betas = data['cluster_means'][shape_family_id][:nBetas]
#     model.betas[:] = betas

#     if shape_family_id == -1:
#         model.betas[:] = np.zeros_like(betas)

#     v = model.r.copy()
#     return v
