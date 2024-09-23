import sys, os
import cv2
import imageio
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import config
import json
from smal_fitter import SMALFitter
import torch
from optimize_to_joints import ImageExporter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Define a function to read the JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def return_placeholder_data(input_image=None, num_joints=55):
    image_size = (512, 512)
    # pass a placeholder to the SMALFitter class as we are not actually going to provide any normal input data
    if input_image is not None:
        # Load image
        img_data = imageio.v2.imread(os.path.join(input_image))

        # convert value range to 0 - 1
        img_data = img_data / 255.0
        rgb = torch.FloatTensor(img_data)[None, ...].permute(0, 3, 1, 2)

        # dynamically use the size of the loaded image
        image_size = rgb.shape[2:]

        filenames = [input_image]
    else:
        rgb = torch.zeros((1, 3, image_size[0], image_size[1]))
        filenames = ["PLACEHOLDER"]
    sil = torch.zeros((1, 1, image_size[0], image_size[1]))
    joints = torch.zeros((1, num_joints, 2))
    visibility = torch.zeros((1, num_joints))

    return (rgb, sil, joints, visibility), filenames


def map_joint_order(joint_names_smil, joint_names_input, joints):
    # map joint names with correct ids regardless of order
    new_joint_locs = np.zeros((len(joint_names_smil), 3), float)

    for o, orig_joint in enumerate(joint_names_smil):
        for m, mapped_joints in enumerate(joint_names_input):
            if orig_joint == mapped_joints:
                new_joint_locs[o] = joints[m]  # flip x and y

    return new_joint_locs


def compute_rodrigues_from_upvector_change(curr_upvector, prev_upvector=np.array([0, 0, 1])):
    """
    Computes the Rodrigues vector from the change in the upvector between two consecutive frames.
    :param prev_upvector: The upvector from the previous frame (a numpy array or list of size 3).
    :param curr_upvector: The upvector from the current frame (a numpy array or list of size 3).
    :return: A Rodrigues vector representing the rotation from prev_upvector to curr_upvector.
    """
    prev_upvector = np.array(prev_upvector)
    curr_upvector = np.array(curr_upvector)

    # Normalize vectors
    prev_upvector = prev_upvector / np.linalg.norm(prev_upvector)
    curr_upvector = curr_upvector / np.linalg.norm(curr_upvector)

    # Compute the rotation axis using cross product
    rotation_axis = np.cross(prev_upvector, curr_upvector)

    # Compute the angle using dot product
    angle = np.arccos(np.clip(np.dot(prev_upvector, curr_upvector), -1.0, 1.0))

    if np.isclose(angle, 0):
        return np.zeros(3)  # No rotation, return zero vector

    # Normalize the rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Rodrigues vector is angle * axis
    rodrigues_vector = angle * rotation_axis

    return rodrigues_vector


def unreal_to_pytorch_coord(cc):
    """
    Function to convert Unreal engine coordinates to pytorch coordinates
    unreal     X forward, Y right, Z up
    pytorch3d  X left, Y up, Z forward
    :param cc: [x,y,z] Unreal engine coordinates
    :return: pt_cc pytorch formatted coordinates
    """
    pt_cc = np.array([-cc[1], cc[2], cc[0]])
    return pt_cc


def ue_view_proj_to_pytorch3d_transform(view_proj_matrix_tensor):
    """
    Converts a view projection matrix from Unreal Engine 5 to the rotation and translation matrices
    used by a pytorch3d.renderer.OpenGLPerspectiveCameras object.

    Args:
        view_proj_matrix_tensor (torch.Tensor): A (4, 4) matrix representing the view projection matrix
                                                from Unreal Engine 5.

    Returns:
        R (torch.Tensor): A (3, 3) rotation matrix for OpenGLPerspectiveCameras.
        T (torch.Tensor): A (3,) translation vector for OpenGLPerspectiveCameras.
    """

    # Step 1: Decompose the view-projection matrix into view and projection matrices
    view_matrix = torch.inverse(view_proj_matrix_tensor)

    # Step 2: Extract the rotation and translation components
    rotation_matrix = view_matrix[:3, :3]
    translation_vector = view_matrix[:3, 3]

    # Step 3: Adjust for the coordinate system difference between UE5 and PyTorch3D
    conversion_matrix = torch.tensor([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ], dtype=rotation_matrix.dtype)  # Ensure the conversion matrix has the same dtype as rotation_matrix

    # Apply the conversion matrix
    rotation_matrix = conversion_matrix @ rotation_matrix
    # translation_vector = conversion_matrix @ translation_vector

    # Step 4: Adjust the translation vector for PyTorch3D's coordinate system
    translation_vector = torch.tensor([-translation_vector[1],
                                       translation_vector[2],
                                       translation_vector[0]])

    return rotation_matrix, translation_vector


def extract_view_proj_matrix_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extracting the view projection matrix
    view_proj_matrix = data['iterationData']['camera']['View Projection Matrix']

    # Constructing the matrix from the JSON data
    view_proj_matrix_tensor = torch.tensor([
        [view_proj_matrix['xPlane']['x'], view_proj_matrix['yPlane']['x'], view_proj_matrix['zPlane']['x'],
         view_proj_matrix['wPlane']['x']],
        [view_proj_matrix['xPlane']['y'], view_proj_matrix['yPlane']['y'], view_proj_matrix['zPlane']['y'],
         view_proj_matrix['wPlane']['y']],
        [view_proj_matrix['xPlane']['z'], view_proj_matrix['yPlane']['z'], view_proj_matrix['zPlane']['z'],
         view_proj_matrix['wPlane']['z']],
        [view_proj_matrix['xPlane']['w'], view_proj_matrix['yPlane']['w'], view_proj_matrix['zPlane']['w'],
         view_proj_matrix['wPlane']['w']],
    ])

    return view_proj_matrix_tensor


def quaternion_to_rodrigues(quaternion):
    """
    Converts a quaternion from Unreal Engine's coordinate system to a Rodrigues vector in PyTorch3D's coordinate system.
    :param quaternion: A quaternion represented as a list or array [w, x, y, z] from Unreal Engine 5.
    :return: A Rodrigues vector (numpy array of size 3) in PyTorch3D's coordinate system.
    """
    w, x, y, z = quaternion

    # Coordinate system transformation from Unreal Engine to PyTorch3D
    # UE5: (X, Y, Z) -> PyTorch3D: (-Z, X, Y)
    # Apply this transformation to the quaternion
    quaternion_pytorch3d = [w, x, y, z]

    # Compute the angle theta
    theta = 2 * np.arccos(w)

    if np.isclose(theta, 0) or np.isclose(theta, 2 * np.pi):
        # Small rotation, return zero vector
        return np.zeros(3)

    # Compute the sin(theta/2) to scale the vector part
    sin_theta_over_2 = np.sin(theta / 2)

    # Rodrigues vector is scaled axis (angle * axis)
    rodrigues_vector = theta * np.array(quaternion_pytorch3d[1:]) / sin_theta_over_2

    return rodrigues_vector


def cam_quternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Converts a quaternion to a rotation matrix.
    """
    # Normalize the quaternion
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    # Convert to rotation matrix
    rotation_matrix = np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ])

    return rotation_matrix


def unreal_euler_to_pytorch3d_rodrigues(pitch, yaw, roll, return_R=False):
    """
    Converts Unreal Engine Euler angles (pitch, yaw, roll) to a Rodrigues vector
    in the PyTorch3D coordinate system. If set return_R == True, the function will instead return
    the rotation matrix of the input (as is required for the camera object of the fitter)

    Unreal Engine uses:
    - Pitch: rotation around X-axis
    - Yaw: rotation around Z-axis
    - Roll: rotation around Y-axis

    PyTorch3D uses a right-handed coordinate system:
    - X: left
    - Y: up
    - Z: forward

    Args:
        pitch (float): Rotation around X-axis in degrees (Unreal Engine).
        yaw (float): Rotation around Z-axis in degrees (Unreal Engine).
        roll (float): Rotation around Y-axis in degrees (Unreal Engine).

    Returns:
        rodrigues_vector (np.array): A Rodrigues vector in PyTorch3D coordinate system.
    """

    # Step 1: Convert Euler angles from degrees to radians
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    # Step 2: Create the rotation matrices in Unreal Engine's coordinate system
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    R_y = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Step 3: Combine the rotation matrices
    # Unreal Engine applies rotations in ZYX order: R = R_z * R_y * R_x
    R_unreal = R_z @ R_y @ R_x

    # Step 4: Adjust for the coordinate system transformation to PyTorch3D
    # Unreal Engine to PyTorch3D: +X -> -Y, +Y -> +Z, +Z -> +X
    conversion_matrix = np.array([
        [0, 1, 0],  # X' = Y
        [0, 0, 1],  # Y' = Z
        [1, 0, 0]  # Z' = X
    ])

    # Apply the conversion matrix
    R_pytorch3d = conversion_matrix @ R_unreal @ conversion_matrix.T

    # when requiring only the rotation matrix, not the Rodrigues vector
    if return_R:
        return R_pytorch3d

    # Step 5: Convert the adjusted rotation matrix to a Rodrigues vector
    theta = np.arccos(np.clip((np.trace(R_pytorch3d) - 1) / 2, -1, 1))

    if np.isclose(theta, 0):
        # Small rotation, return zero vector
        return np.zeros(3)
    else:
        rodrigues_vector = (1 / (2 * np.sin(theta))) * np.array([
            R_pytorch3d[2, 1] - R_pytorch3d[1, 2],
            R_pytorch3d[0, 2] - R_pytorch3d[2, 0],
            R_pytorch3d[1, 0] - R_pytorch3d[0, 1]
        ]) * theta

    return rodrigues_vector


def set_equal_axis_lengths(ax):
    """Set equal lengths for all axes in a 3D plot."""
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    centers = np.mean(extents, axis=1)
    max_range = np.max(extents[:, 1] - extents[:, 0]) / 2

    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - max_range, ctr + max_range)


def plot_camera_view(camera_translation, camera_rotation_matrix, point_3d, quiver_length=1.0, fov=60.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D point
    ax.scatter(point_3d[0], point_3d[1], point_3d[2], c='r', marker='o', label='3D Point')

    # Plot the camera position
    ax.scatter(camera_translation[0], camera_translation[1], camera_translation[2], c='b', marker='^',
               label='Camera Position')

    # Define camera's coordinate system axes
    camera_x_axis = camera_rotation_matrix[:, 0]  # Directly take the first column
    camera_y_axis = camera_rotation_matrix[:, 1]  # Directly take the second column
    camera_z_axis = camera_rotation_matrix[:, 2]  # Directly take the third column

    # Normalize the axes
    camera_x_axis /= np.linalg.norm(camera_x_axis)
    camera_y_axis /= np.linalg.norm(camera_y_axis)
    camera_z_axis /= np.linalg.norm(camera_z_axis)

    # Plot the camera's coordinate system
    ax.quiver(camera_translation[0], camera_translation[1], camera_translation[2],
              camera_x_axis[0], camera_x_axis[1], camera_x_axis[2],
              length=quiver_length, color='r', label='Camera X-axis')

    ax.quiver(camera_translation[0], camera_translation[1], camera_translation[2],
              camera_y_axis[0], camera_y_axis[1], camera_y_axis[2],
              length=quiver_length, color='g', label='Camera Y-axis')

    ax.quiver(camera_translation[0], camera_translation[1], camera_translation[2],
              camera_z_axis[0], camera_z_axis[1], camera_z_axis[2],
              length=quiver_length, color='b', label='Camera Z-axis')

    # Connect the camera to the 3D point
    ax.plot([camera_translation[0], point_3d[0]],
            [camera_translation[1], point_3d[1]],
            [camera_translation[2], point_3d[2]], 'k--', label='Camera to Point')

    # Plot the cone representing the FOV
    plot_fov_cone(ax, camera_translation, camera_z_axis, fov, 200)

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Camera Viewing Orientation and 3D Point')

    # Set equal axis lengths
    set_equal_axis_lengths(ax)

    ax.legend()
    plt.show()


def plot_fov_cone(ax, origin, direction, fov, length):
    """Plot a cone representing the camera's field of view (FOV)."""
    # Convert FOV from degrees to radians
    fov_rad = np.radians(fov)

    # Define the radius of the base of the cone
    radius = np.tan(fov_rad / 2) * length

    # Create a circle in the plane orthogonal to the direction vector
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)

    # Create the cone's base in 3D
    base_circle = np.array([circle_x, circle_y, np.zeros_like(circle_x)])

    # Rotate the circle to align with the direction vector
    if not np.allclose(direction, [0, 0, 1]):
        # Find rotation matrix to align z-axis with direction
        v = np.cross([0, 0, 1], direction)
        c = np.dot([0, 0, 1], direction)
        s = np.linalg.norm(v)
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + k + k @ k * ((1 - c) / (s ** 2))
        base_circle = R @ base_circle

    # Translate the circle to the tip of the cone
    cone_base = origin[:, np.newaxis] + direction[:, np.newaxis] * length + base_circle

    # Create the surface of the cone
    cone_vertices = [list(zip(cone_base[0, :], cone_base[1, :], cone_base[2, :]))]

    # Plot the cone's surface
    ax.add_collection3d(Poly3DCollection(cone_vertices, color='b', alpha=0.3))

    # Plot lines from the origin to the base of the cone
    for i in range(cone_base.shape[1]):
        ax.plot([origin[0], cone_base[0, i]], [origin[1], cone_base[1, i]], [origin[2], cone_base[2, i]], 'b-',
                alpha=0.3)


def unreal_to_opengl_rotation_matrix(yaw, pitch, roll):
    """Convert Unreal Engine yaw, pitch, roll to an OpenGL-compatible rotation matrix."""
    # Convert degrees to radians
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # Rotation around the Unreal Z-axis (Yaw -> corresponds to -Y-axis in OpenGL)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Rotation around the Unreal Y-axis (Pitch -> corresponds to X-axis in OpenGL)
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation around the Unreal X-axis (Roll -> corresponds to Z-axis in OpenGL)
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Combine rotations: R = R_yaw * R_pitch * R_roll
    R = R_yaw @ R_pitch @ R_roll

    # Convert to OpenGL coordinate system:
    # Unreal's (X, Y, Z) -> OpenGL's (X, Y, Z) with handedness change (invert Z)
    handedness_conversion = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    R = handedness_conversion @ R

    return R


def parse_projection_components(iteration_data_file):
    ####### DOUBLE CHECK THESE VALUES GO INTO THE RIGHT MATRIX ELEMENTS ############

    # converts Unreal view projection into rotation and translation components
    input_matrix = iteration_data_file["iterationData"]["camera"]["View Matrix"]
    w = input_matrix["wPlane"]
    x = input_matrix["xPlane"]
    y = input_matrix["yPlane"]
    z = input_matrix["zPlane"]
    # now, assign the respective transposed values to the rotation...
    cam_rot = np.array([[x["x"], y["x"], z["x"]],
                        [x["y"], y["y"], z["y"]],
                        [x["z"], y["z"], z["z"]]])
    # and the translation
    """
    cam_trans = np.array([iteration_data_file["iterationData"]["camera"]["Location"]["x"],
                          iteration_data_file["iterationData"]["camera"]["Location"]["y"],
                          iteration_data_file["iterationData"]["camera"]["Location"]["z"]])
    """
    cam_trans = np.array([w["x"],
                          w["y"],
                          w["z"]])
    # There. Tried to do it differently, had a break down, now it works.
    # Bon appetit
    return cam_rot, cam_trans


def parse_camera_intrinsics(img, iteration_data_file, FOV=60):
    # first get the image resolution from the batch data file and the current FOV from the iteration data file
    img_cv = cv2.imread(img)
    res_px_X = img_cv.shape[1]
    res_px_Y = img_cv.shape[0]

    # then compute the image centre and focal length in x and y respectively
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    cx = res_px_X / 2
    cy = res_px_Y / 2

    fx = cx / np.tan(np.radians(FOV) / 2)
    fy = cy / np.tan(np.radians(FOV) / 2)

    return cx, cy, fx, fy


if __name__ == '__main__':
    """
    STEP 1 - LOAD replicAnt generated SMIL data
    """
    # Read the JSON file
    json_file_path = "data/replicAnt_trials/TEST_ANGLES/TEST_ANGLES_00.json"
    input_image = json_file_path.split(".")[0] + ".JPG"
    data = read_json_file(json_file_path)

    # Extract shape and pose parameters
    shape_betas = data['iterationData']['subject Data'][0]["1"]["shape betas"]
    pose_data = data['iterationData']['subject Data'][0]["1"]['keypoints']

    joint_angles = []
    joint_names = []

    # TODO - Fix joint angle translation issues.
    root_rotation = np.array([pose_data["b_t"]["eulerAngles"]["y"],
                              pose_data["b_t"]["eulerAngles"]["z"],
                              pose_data["b_t"]["eulerAngles"]["x"]], np.float64)

    prev_upvector = np.array([0, 1, 0])  # Placeholder for the previous upvector

    for key in pose_data:
        joint_names.append(key)

        rodrigues_angle = unreal_euler_to_pytorch3d_rodrigues(pitch=pose_data[key]["eulerAngles"]["y"],
                                                              yaw=pose_data[key]["eulerAngles"]["z"],
                                                              roll=pose_data[key]["eulerAngles"]["x"])

        print(key, rodrigues_angle)
        joint_angles.append(rodrigues_angle)

    np_joint_angles = np.array(joint_angles)

    # map joints, in case the order differs. The root bone is expected to be the first entry
    np_joint_angles_mapped = map_joint_order(config.dd["J_names"], joint_names, np_joint_angles)

    # Convert shape betas to a NumPy array
    if len(shape_betas) != 0:
        shape_betas = np.zeros(20)
    else:
        shape_betas = np.array(shape_betas)

    # Display the extracted data
    print("Shape Betas:", shape_betas.shape)
    print("Pose Data:", np_joint_angles_mapped.shape)

    """
    STEP 1b - GET CAMERA INFO
    """

    R = torch.tensor(unreal_to_opengl_rotation_matrix(pitch=data['iterationData']['camera']["Rotation"]["pitch"],
                                                      yaw=data['iterationData']['camera']["Rotation"]["yaw"],
                                                      roll=data['iterationData']['camera']["Rotation"]["roll"]))

    camera_translation = [data['iterationData']['camera']["Location"]["x"],
                          data['iterationData']['camera']["Location"]["y"],
                          data['iterationData']['camera']["Location"]["z"]]

    T = torch.tensor([-camera_translation[1],
                      camera_translation[2],
                      camera_translation[0]])

    point_3d = [-pose_data["b_t"]["3DPos"]["y"],
                pose_data["b_t"]["3DPos"]["z"],
                pose_data["b_t"]["3DPos"]["x"]]

    camera_fov = [data['iterationData']['camera']["FOV"]]

    """
    STEP 2 - LOAD SMIL MODEL
    """

    data_json, filenames = return_placeholder_data(
        input_image=input_image,
        num_joints=len(np_joint_angles_mapped))  # in the shape of the default convention returned by the dataloaders

    print("\nINFO: Preparing SMIL model...")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not config.ignore_hardcoded_body:
        assert config.SHAPE_FAMILY >= 0, "Shape family should be greater than 0"

        use_unity_prior = config.SHAPE_FAMILY == 1 and not config.FORCE_SMAL_PRIOR
    else:
        use_unity_prior = False

    if not use_unity_prior and not config.ALLOW_LIMB_SCALING:
        print(
            "WARNING: Limb scaling is only recommended for the new Unity prior. TODO: add a regularizer to constrain scale parameters.")
        config.ALLOW_LIMB_SCALING = False

    model = SMALFitter(device, data_json, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior)

    """
    STEP 3 - APPLY SHAPE BETAS AND JOINT ROTATIONS
    """

    # model parameters
    # model.betas = torch.nn.Parameter(torch.Tensor(shape_betas).to(device))
    model.joint_rotations = torch.nn.Parameter(
        torch.Tensor(np_joint_angles_mapped[1:]).reshape((1, np_joint_angles_mapped[1:].shape[0], 3)).to(device))

    # camera parameters
    # TODO fix camera compute!

    cam_rot, cam_trans = parse_projection_components(data)

    cx, cy, fx, fy = parse_camera_intrinsics(input_image, data, FOV=camera_fov[0])

    print(parse_camera_intrinsics(input_image, data, FOV=camera_fov[0]))

    cam_intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])

    R = cam_rot
    T = np.reshape(cam_trans, (3, 1))
    C = cam_intrinsics
    root_location = [pose_data["b_t"]["3DPos"]["x"],
                     pose_data["b_t"]["3DPos"]["y"],
                     pose_data["b_t"]["3DPos"]["z"]]


    X = np.reshape(np.array(root_location), (3, -1))

    # given the above data, it should be possible to project the 3D points into the corresponding image,
    # so they land in the correct position on the image
    P = C @ np.hstack([R, T])  # projection matrix
    X_hom = np.vstack([X, np.ones(X.shape[1])])  # 3D points in homogenous coordinates

    print(X_hom)

    X_hom = P @ X_hom  # project the 3D points

    print(X_hom)

    X_2d = X_hom[:2, :] / X_hom[2, :]  # convert them back to 2D pixel space

    print(X_2d)

    # now forgetting about getting the projection, but just getting the 3D loc of the subject in camera space

    P_base = np.hstack([R, T])
    X_hom = np.vstack([X, np.ones(X.shape[1])])  # 3D points in homogenous coordinates
    X_hom = P_base @ X_hom  # project the 3D points

    print(X_hom) # now we got our reprojected X location [X,Y,Z] with respect to the camera
    # the reference is to the image centre with
    # X + = right
    # Y + = up
    # Z + = away from camera

    # next, we want the rotation (of the model's root (b_t), so model.global_rotation

    # TODO get rotation relative to camera

    """

    model.fov = torch.nn.Parameter(torch.Tensor(camera_fov).to(device))
    model.renderer.cameras.fov = camera_fov[0]

    model_loc = torch.tensor([-pose_data["b_t"]["3DPos"]["y"],
                              pose_data["b_t"]["3DPos"]["z"],
                              pose_data["b_t"]["3DPos"]["x"]])

    # as a test, let's keep the cam where it is and move the model

    print(
        model.renderer.cameras.T)  # located on the ground and slightly moved forward to look at subject from the front
    print(model_loc)

    new_model_loc = (model_loc - T) * 0.01  # because the scaling is off)
    print(new_model_loc)
    
    """

    """
    model.renderer.cameras.R = R.clone().detach().to(device).unsqueeze(0)
    model.renderer.cameras.T = T.clone().detach().to(device).unsqueeze(0)
    """
    
    print("model.trans")
    print(model.trans.shape)
    print(model.trans)
    
    X_hom_pytorch = np.array([-X_hom[1], X_hom[2], X_hom[0]], dtype=float)
    X_hom_pytorch = X_hom_pytorch.reshape(1, 3)  # Reshape to (1, 3)
    model.trans = torch.nn.Parameter(torch.tensor(X_hom_pytorch, dtype=torch.float32).to(device))

    print("model.trans")
    print(model.trans.shape)
    print(model.trans)

    # camera -> X+ left, Y+ up, Z+ forward (away from camera)
    # the model is located at the origin
    # model.renderer.cameras.T = torch.tensor([0.5, 0.5, 3.0]).clone().detach().to(device).unsqueeze(0)

    # so not in the rotation of the root bone but HERE we should store the combined rotation of
    # the model with respect to the camera
    """
    # we start out with this (from SMALFitter)
    global_rotation_np = eul_to_axis(np.array([-np.pi / 2, 0, -np.pi / 2]))
    global_rotation = torch.from_numpy(global_rotation_np).float().to(device).unsqueeze(0).repeat(self.num_images,
                                                                                                  1)  # Global Init (Head-On)
    
    so from the camera we have its translation in world space and its rotation in world space
    - then we can get its position relative to the subject (placing the subject at 0,0,0 as is convention)
    - the negative position should then be the 
    
    
    """

    """
    STEP 4 - RENDER POSED MESH
    """

    """
    plot_camera_view(camera_translation=T,
                     camera_rotation_matrix=R,
                     point_3d=point_3d,
                     quiver_length=100,
                     fov=camera_fov)
    """

    print("\nINFO: Rendering output")
    image_exporter = ImageExporter("LOCAL_TEST", [os.path.basename(input_image)])

    image_exporter.stage_id = 0
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter)
