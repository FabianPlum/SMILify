import bpy
import numpy as np
import os
import pickle

# Decorators for type checking
def ensure_mesh(func):
    def wrapper(obj, *args, **kwargs):
        if not obj or obj.type != 'MESH':
            raise TypeError("The selected object is not a mesh.")
        return func(obj, *args, **kwargs)
    return wrapper

def ensure_armature(func):
    def wrapper(obj, *args, **kwargs):
        if not obj or obj.type != 'ARMATURE':
            raise TypeError("The selected object is not an armature.")
        return func(obj, *args, **kwargs)
    return wrapper

# Convert mesh to numpy arrays
def mesh_to_numpy(obj):
    mesh = obj.data
    vertices = np.array([vert.co for vert in mesh.vertices], dtype=np.float32)
    faces = np.array([poly.vertices for poly in mesh.polygons if len(poly.vertices) == 3], dtype=np.int32)
    return vertices, faces

@ensure_mesh
def triangulate_mesh(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')

@ensure_mesh
def export_vertices_to_npy(obj, filepath):
    vertices, _ = mesh_to_numpy(obj)
    np.save(filepath, vertices)
    return filepath, vertices

@ensure_mesh
def export_faces_to_npy(obj, filepath):
    _, faces = mesh_to_numpy(obj)
    np.save(filepath, faces)
    return filepath, faces

@ensure_mesh
def export_mesh_to_obj(obj, filepath):
    vertices, faces = mesh_to_numpy(obj)
    with open(filepath, 'w') as file:
        for vert in vertices:
            file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return filepath

@ensure_mesh
def export_vertex_groups_to_npy(obj, filepath):
    mesh = obj.data
    vertex_groups = obj.vertex_groups
    num_groups = len(vertex_groups)
    weights = np.zeros((len(mesh.vertices), num_groups), dtype=np.float32)
    for i, vert in enumerate(mesh.vertices):
        for j, group in enumerate(vertex_groups):
            try:
                weights[i, j] = group.weight(i)
            except RuntimeError:
                weights[i, j] = 0.0
    np.save(filepath, weights)
    return filepath, weights

@ensure_armature
def export_joint_locations_to_npy(armature_obj, filepath):
    joints = armature_obj.data.bones
    joint_locations = np.array([bone.head_local for bone in joints], dtype=np.float32)
    np.save(filepath, joint_locations)
    return filepath, joint_locations

@ensure_armature
def export_joint_hierarchy_to_npy(armature_obj, filepath):
    joints = armature_obj.data.bones
    hierarchy = [[-1, 0]]
    for bone in joints:
        if bone.parent:
            parent_index = joints.find(bone.parent.name)
            child_index = joints.find(bone.name)
            hierarchy.append([parent_index, child_index])
    hierarchy = np.array(hierarchy, dtype=np.int32).T
    np.save(filepath, hierarchy)
    return filepath, hierarchy

@ensure_mesh
def export_y_axis_vertices_to_npy(obj, filepath):
    mesh = obj.data
    y_axis_vertices = np.array([i for i, vert in enumerate(mesh.vertices) if np.isclose(vert.co.y, 0.0, atol=1e-3)], dtype=int)
    np.save(filepath, y_axis_vertices)
    return filepath, y_axis_vertices

def find_nearest_neighbors(vertices, joint_locations, n):
    nearest_indices = np.zeros((len(joint_locations), n), dtype=np.int32)
    nearest_weights = np.zeros((len(joint_locations), n), dtype=np.float32)
    for i, joint_loc in enumerate(joint_locations):
        distances = np.linalg.norm(vertices - joint_loc, axis=1)
        nearest_indices[i] = np.argpartition(distances, n)[:n]
        nearest_distances = distances[nearest_indices[i]]
        weights = 1.0 / nearest_distances
        weights /= weights.sum()
        nearest_weights[i] = weights
    return nearest_indices, nearest_weights

@ensure_mesh
#@ensure_armature (careful, the mesh is the active object!)
def export_J_regressor_to_npy(mesh_obj, armature_obj, n, filepath):
    vertices, _ = mesh_to_numpy(mesh_obj)
    joints = armature_obj.data.bones
    joint_locations = np.array([bone.head_local for bone in joints], dtype=np.float32)
    nearest_indices, nearest_weights = find_nearest_neighbors(vertices, joint_locations, n)
    J_regressor = np.zeros((len(joints), len(vertices)), dtype=np.float32)
    for i in range(len(joints)):
        J_regressor[i, nearest_indices[i]] = nearest_weights[i]
    np.save(filepath, J_regressor)
    return filepath, J_regressor

"""
This is currently not supported.
The posedir created here captures the mesh shape at every frame
This is however not hwo the posedir is used in the original implementaiton and thus disabled for now.
"""
@ensure_mesh
def export_posedirs(mesh_obj, start_frame, stop_frame, filepath):
    bpy.context.view_layer.objects.active = mesh_obj
    num_frames = stop_frame - start_frame + 1
    num_vertices = len(mesh_obj.data.vertices)
    posedirs = np.zeros((num_vertices, 3, num_frames), dtype=np.float32)

    for frame in range(start_frame, stop_frame + 1):
        bpy.context.scene.frame_set(frame)
        for i, vert in enumerate(mesh_obj.data.vertices):
            posedirs[i, :, frame - start_frame] = vert.co

    np.save(filepath, posedirs)
    return filepath, posedirs


# Export all model elements as a dict following the SMPL convention, contained in a .pkl file
def export_smpl_model(start_frame=0, stop_frame=1):
    smpl_dict = {
        "f": [],
        "J_regressor": [],
        "kintree_table": [],
        "J": [],
        "bs_style": "lbs",
        "weights": [],
        "posedirs": [],
        "v_template": [],
        "shapedirs": [],
        "bs_type": "lrotmin",
        "sym_verts": []
    }

    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        print("No valid mesh object selected for testing.")
        return

    print("Selected mesh object:", obj.name)
    triangulate_mesh(obj)

    vertices_npy_path = bpy.path.abspath('//test_vertices.npy')
    faces_npy_path = bpy.path.abspath('//test_faces.npy')
    vertex_groups_npy_path = bpy.path.abspath('//test_vertex_groups.npy')
    joint_locations_npy_path = bpy.path.abspath('//test_joint_locations.npy')
    joint_hierarchy_npy_path = bpy.path.abspath('//test_joint_hierarchy.npy')
    j_regressor_npy_path = bpy.path.abspath('//test_joint_regressor.npy')
    y_axis_vertices_npy_path = bpy.path.abspath('//test_y_axis_vertices.npy')
    posedirs_npy_path = bpy.path.abspath('//test_posedirs.npy')

    smpl_file_path = bpy.path.abspath('//smpl_ATTA.pkl')

    smpl_dict["v_template"] = export_vertices_to_npy(obj, vertices_npy_path)[1]
    smpl_dict["f"] = export_faces_to_npy(obj, faces_npy_path)[1]
    smpl_dict["weights"] = export_vertex_groups_to_npy(obj, vertex_groups_npy_path)[1]
    smpl_dict["sym_verts"] = export_y_axis_vertices_to_npy(obj, y_axis_vertices_npy_path)[1]

    num_verts = smpl_dict["v_template"].shape[0]
    num_joints = smpl_dict["weights"].shape[1]

    print("Vertices: ", num_verts)
    print("Joints: ", num_joints)
    
    """ 
    #Let's be adventurous and remove the poses and instead add fun shapes 
    
    smpl_dict["posedirs"] = export_posedirs(obj, start_frame, stop_frame, posedirs_npy_path)[1]

    smpl_dict["shapedirs"] = np.zeros((num_verts, 3))
    
    """
    
    smpl_dict["shapedirs"] = np.zeros((num_verts, 3))

    smpl_dict["posedirs"] = np.empty(0) # leave empty if there are no pose blend shapes
    
    print(smpl_dict["posedirs"].shape)
    

    armature_obj = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
    if not armature_obj:
        print("No armature object found.")
        return

    print("Found armature object:", armature_obj.name)

    smpl_dict["kintree_table"] = export_joint_hierarchy_to_npy(armature_obj, joint_hierarchy_npy_path)[1]
    smpl_dict["J"] = export_joint_locations_to_npy(armature_obj, joint_locations_npy_path)[1]
    smpl_dict["J_regressor"] = export_J_regressor_to_npy(obj, armature_obj, 20, j_regressor_npy_path)[1]

    with open(smpl_file_path, "wb") as f:
        pickle.dump(smpl_dict, f)

    print(f"SMPL model exported successfully to {smpl_file_path}")

# Run tests
def test_export_functions():
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        print("No valid mesh object selected for testing.")
        return

    # Triangulate the mesh before testing
    triangulate_mesh(obj)

    vertices_npy_path = bpy.path.abspath('//test_vertices.npy')
    faces_npy_path = bpy.path.abspath('//test_faces.npy')
    obj_path = bpy.path.abspath('//test_mesh.obj')
    vertex_groups_npy_path = bpy.path.abspath('//test_vertex_groups.npy')
    y_axis_vertices_npy_path = bpy.path.abspath('//test_y_axis_vertices.npy')

    # Test export_vertices_to_npy
    try:
        vertices_path = export_vertices_to_npy(obj, vertices_npy_path)[0]
        assert os.path.exists(vertices_path), "Vertices .npy file not found!"
        print(f"Vertices exported successfully to {vertices_path}")
    except Exception as e:
        print(f"export_vertices_to_npy failed: {e}")

    # Test export_faces_to_npy
    try:
        faces_path = export_faces_to_npy(obj, faces_npy_path)[0]
        assert os.path.exists(faces_path), "Faces .npy file not found!"
        print(f"Faces exported successfully to {faces_path}")
    except Exception as e:
        print(f"export_faces_to_npy failed: {e}")

    # Test export_mesh_to_obj
    try:
        obj_file_path = export_mesh_to_obj(obj, obj_path)
        assert os.path.exists(obj_file_path), "OBJ file not found!"
        print(f"Mesh exported successfully to {obj_file_path}")
    except Exception as e:
        print(f"export_mesh_to_obj failed: {e}")

    # Test export_vertex_groups_to_npy
    try:
        vertex_groups_path = export_vertex_groups_to_npy(obj, vertex_groups_npy_path)[0]
        assert os.path.exists(vertex_groups_path), "Vertex groups .npy file not found!"
        print(f"Vertex groups exported successfully to {vertex_groups_path}")
    except Exception as e:
        print(f"export_vertex_groups_to_npy failed: {e}")

    # Test export_y_axis_vertices_to_npy
    try:
        y_axis_vertices_path = export_y_axis_vertices_to_npy(obj, y_axis_vertices_npy_path)[0]
        assert os.path.exists(y_axis_vertices_path), "Y-axis vertices .npy file not found!"
        print(f"Y-axis vertices exported successfully to {y_axis_vertices_path}")
    except Exception as e:
        print(f"export_y_axis_vertices_to_npy failed: {e}")

    # Check for armature object
    armature_obj = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)

    if not armature_obj:
        print("No armature object found for testing.")
        return

    joint_locations_npy_path = bpy.path.abspath('//test_joint_locations.npy')
    joint_hierarchy_npy_path = bpy.path.abspath('//test_joint_hierarchy.npy')
    j_regressor_npy_path = bpy.path.abspath('//test_J_regressor.npy')

    # Test export_joint_locations_to_npy
    try:
        joint_locations_path = export_joint_locations_to_npy(armature_obj, joint_locations_npy_path)[0]
        assert os.path.exists(joint_locations_path), "Joint locations .npy file not found!"
        print(f"Joint locations exported successfully to {joint_locations_path}")
    except Exception as e:
        print(f"export_joint_locations_to_npy failed: {e}")

    # Test export_joint_hierarchy_to_npy
    try:
        joint_hierarchy_path = export_joint_hierarchy_to_npy(armature_obj, joint_hierarchy_npy_path)[0]
        assert os.path.exists(joint_hierarchy_path), "Joint hierarchy .npy file not found!"
        print(f"Joint hierarchy exported successfully to {joint_hierarchy_path}")
    except Exception as e:
        print(f"export_joint_hierarchy_to_npy failed: {e}")

    # Test export_J_regressor_to_npy
    try:
        j_regressor_path = export_J_regressor_to_npy(obj, armature_obj, 5, j_regressor_npy_path)[0]  # Assuming 5 closest vertices
        assert os.path.exists(j_regressor_path), "J regressor .npy file not found!"
        print(f"J regressor exported successfully to {j_regressor_path}")
    except Exception as e:
        print(f"export_J_regressor_to_npy failed: {e}")

# Export SMPL-style model
export_smpl_model()
