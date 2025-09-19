import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import imageio
import sys
from scipy.spatial.transform import Rotation

sys.path.append(os.path.dirname(sys.path[0]))
from smal_fitter import SMALFitter
import config

from optimize_to_joints import ImageExporter
from nibabel import eulerangles

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)


"""
General / Helper functions
"""

def _reorder_dirs_array(dirs: np.ndarray, bones_len: int, what: str = "input array") -> np.ndarray:
    """
    Reorder PCA direction tensors to shape (num_joints, 3, num_components).

    Detects axes using:
    - one axis with size 3 -> channel axis
    - one axis with size == bones_len -> joint axis
    - remaining axis -> component axis
    """
    if not isinstance(dirs, np.ndarray):
        raise ValueError(f"{what} must be a numpy array, got {type(dirs)}")
    if dirs.ndim != 3:
        raise ValueError(f"{what} must be 3D but has shape {dirs.shape}")

    shape = dirs.shape
    channel_axes = [i for i, s in enumerate(shape) if s == 3]
    if len(channel_axes) != 1:
        raise ValueError(f"{what}: expected exactly one axis of size 3 (channels), got shape {shape}")
    channel_axis = channel_axes[0]

    joint_axes = [i for i, s in enumerate(shape) if s == bones_len]
    if len(joint_axes) != 1:
        raise ValueError(f"{what}: expected exactly one axis of size bones_len={bones_len} (joints), got shape {shape}")
    joint_axis = joint_axes[0]

    pc_axis = [i for i in range(3) if i not in (channel_axis, joint_axis)]
    if len(pc_axis) != 1:
        raise ValueError(f"{what}: could not determine PC axis from shape {shape}")
    pc_axis = pc_axis[0]

    return np.transpose(dirs, (joint_axis, channel_axis, pc_axis))


def sample_pca_transforms_from_dirs(dd: dict, scale_weights, trans_weights, bone_names=None, check_strict: bool = True):
    """
    Compute per-bone PCA scale and translation using SMIL/SMAL PCA dirs and weights.

    Inputs:
    - dd: dict loaded from SMIL .pkl containing 'scaledirs', 'transdirs', and 'J_names'
    - scale_weights: array-like of length num_components for scale PCs
    - trans_weights: array-like of length num_components for translation PCs
    - bone_names: optional list of joint/bone names; defaults to dd['J_names']
    - check_strict: if True, enforce that scaledirs/transdirs share identical (J,C)

    Returns:
    - translation: np.ndarray of shape (J, 3), ordered by bone_names (default dd['J_names'])
    - scale: np.ndarray of shape (J, 3), ordered by bone_names (default dd['J_names']), values are (1 + weighted sum)

    Shape expectations (after normalization):
    - scaledirs: (J, 3, C)
    - transdirs: (J, 3, C)
    - len(scale_weights) == C == len(trans_weights)
    - len(bone_names) == J
    """
    if dd is None:
        raise ValueError("dd must be provided and contain PCA dirs")

    if 'scaledirs' not in dd:
        raise KeyError("scaledirs not found in dd")
    if 'transdirs' not in dd:
        raise KeyError("transdirs not found in dd")

    scaledirs_raw = dd['scaledirs']
    transdirs_raw = dd['transdirs']

    # Prepare bone names first so we know expected joint axis size
    if bone_names is None:
        if 'J_names' not in dd:
            raise KeyError("J_names not found in dd and bone_names not provided")
        bone_names = dd['J_names']
    bones_len = len(bone_names)

    # Convert to numpy if needed (in case of lists)
    scaledirs_raw = np.asarray(scaledirs_raw)
    transdirs_raw = np.asarray(transdirs_raw)

    # reorder scaledirs and transdirs to match the expected shape (J, 3, C)
    scaledirs = _reorder_dirs_array(scaledirs_raw, bones_len=bones_len, what='scaledirs')
    transdirs = _reorder_dirs_array(transdirs_raw, bones_len=bones_len, what='transdirs')

    num_joints_s, _, num_pcs_s = scaledirs.shape
    num_joints_t, _, num_pcs_t = transdirs.shape

    # Validate joint counts
    if len(bone_names) != num_joints_s or len(bone_names) != num_joints_t:
        raise ValueError(
            f"Bone names length ({len(bone_names)}) must match num_joints of dirs (scale={num_joints_s}, trans={num_joints_t})"
        )

    # Convert weights
    scale_weights = np.asarray(scale_weights, dtype=np.float64).reshape(-1)
    trans_weights = np.asarray(trans_weights, dtype=np.float64).reshape(-1)

    # Check component counts
    if scale_weights.shape[0] != num_pcs_s:
        raise ValueError(
            f"scale_weights length ({scale_weights.shape[0]}) must equal num scale PCs ({num_pcs_s})"
        )
    if trans_weights.shape[0] != num_pcs_t:
        raise ValueError(
            f"trans_weights length ({trans_weights.shape[0]}) must equal num translation PCs ({num_pcs_t})"
        )

    if check_strict and (num_pcs_s != num_pcs_t or num_joints_s != num_joints_t):
        raise ValueError(
            f"scaledirs and transdirs must share same (num_joints, num_pcs). Got scale=({num_joints_s},{num_pcs_s}), trans=({num_joints_t},{num_pcs_t})"
        )

    # Weighted sums over components -> (J, 3)
    # translation: direct sum over PCs
    translation_sum = np.tensordot(transdirs, trans_weights, axes=([2], [0]))  # (J, 3)
    # scale: base of 1.0 + weighted sum
    scale_sum = np.tensordot(scaledirs, scale_weights, axes=([2], [0]))  # (J, 3)
    scale_sum = 1.0 + scale_sum

    return translation_sum.astype(np.float32), scale_sum.astype(np.float32)

"""
Unreal data parsing functions
"""


def parse_projection_components(iteration_data_file):
    """
    Parse Unreal Engine view projection matrix into rotation and translation components.
    
    Args:
        iteration_data_file (dict): JSON data containing Unreal Engine iteration data.
                                   Must have structure: {"iterationData": {"camera": {"View Matrix": {...}}}}
                                   where View Matrix contains wPlane, xPlane, yPlane, zPlane components.
    
    Returns:
        tuple: (cam_rot, cam_trans)
            - cam_rot (np.ndarray): 3x3 rotation matrix extracted from view matrix
            - cam_trans (np.ndarray): 3x1 translation vector extracted from view matrix
    """
    # converts Unreal view projection into rotation and translation components
    input_matrix = iteration_data_file["iterationData"]["camera"]["View Matrix"]
    w = input_matrix["wPlane"]
    x = input_matrix["xPlane"]
    y = input_matrix["yPlane"]
    z = input_matrix["zPlane"]
    # now, assign the respective transposed values to the rotation...
    cam_rot = np.array(
        [[x["x"], y["x"], z["x"]], [x["y"], y["y"], z["y"]], [x["z"], y["z"], z["z"]]]
    )
    # and the translation
    cam_trans = np.array([w["x"], w["y"], w["z"]])
    # There. Tried to do it differently, had a break down, now it works.
    # Bon appetit
    return cam_rot, cam_trans


def parse_camera_intrinsics(batch_data_file, iteration_data_file):
    """
    Parse camera intrinsic parameters from Unreal Engine data files.
    
    Calculates camera intrinsic parameters (principal point and focal length) from
    image resolution and field of view data. Uses the standard pinhole camera model.
    
    Args:
        batch_data_file (dict): JSON data containing batch information including image resolution.
                               Must have structure: {"Image Resolution": {"x": int, "y": int}}
        iteration_data_file (dict): JSON data containing iteration-specific camera data.
                                   Must have structure: {"iterationData": {"camera": {"FOV": float}}}
    
    Returns:
        tuple: (cx, cy, fx, fy)
            - cx (float): Principal point x-coordinate (image center x)
            - cy (float): Principal point y-coordinate (image center y) 
            - fx (float): Focal length in x-direction (pixels)
            - fy (float): Focal length in y-direction (pixels)
    
    Note:
        Assumes square pixels (fx = fy). The principal point is assumed to be at
        the center of the image (cx = width/2, cy = height/2).
    """
    # first get the image resolution from the batch data file and the current FOV from the iteration data file
    res_px_X = batch_data_file["Image Resolution"]["x"]
    res_px_Y = batch_data_file["Image Resolution"]["y"]
    FOV = iteration_data_file["iterationData"]["camera"]["FOV"]

    # then compute the image centre and focal length in x and y respectively
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    cx = res_px_X / 2
    cy = res_px_Y / 2

    fx = cx / np.tan(np.radians(FOV) / 2)
    fy = cy / np.tan(np.radians(FOV) / 2)

    return cx, cy, fx, fy


def return_placeholder_data(input_image=None, num_joints=55, pose_data=None, keypoints_2d=None, keypoint_visibility=None, silhouette=None):
    """
    Create placeholder data for SMALFitter initialization from Unreal Engine pose data.
    
    Prepares input data in the format expected by SMALFitter class. Can load an actual
    image or create placeholder tensors. Supports both raw pose_data and preprocessed keypoints.
    
    Args:
        input_image (str, optional): Path to input image file. If None, creates placeholder
                                    tensors with default size (512, 512). Default is None.
        num_joints (int, optional): Number of joints in the pose data. Default is 55.
        pose_data (dict, optional): Dictionary containing 2D joint positions from Unreal data.
                                   Expected format: {joint_name: {"2DPos": {"x": float, "y": float}}}
                                   If None and keypoints_2d is None, creates zero tensors.
        keypoints_2d (np.ndarray, optional): Preprocessed normalized 2D keypoint coordinates 
                                            of shape (num_joints, 2) with values in [0, 1].
        keypoint_visibility (np.ndarray, optional): Visibility array of shape (num_joints,).
        silhouette (np.ndarray, optional): Silhouette array of shape (H, W).
    
    Returns:
        tuple: ((rgb, sil, joints, visibility), filenames)
            - rgb (torch.Tensor): RGB image tensor of shape (1, 3, H, W) with values in [0, 1]
            - sil (torch.Tensor): Silhouette tensor of shape (1, 1, H, W) filled with zeros if silhouette is None, otherwise the silhouette tensor
            - joints (torch.Tensor): Joint positions tensor of shape (1, num_joints, 2) in pixel coordinates
            - visibility (torch.Tensor): Joint visibility tensor of shape (1, num_joints)
            - filenames (list): List containing the input image filename or ["PLACEHOLDER"]
    
    Note:
        Prioritizes keypoints_2d/keypoint_visibility over pose_data if both are provided.
        Converts normalized coordinates to pixel coordinates based on loaded image size.
    
    """
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

    if silhouette is not None:
        sil = torch.FloatTensor(silhouette)[None, None, ...]
    else:
        sil = torch.zeros((1, 1, image_size[0], image_size[1]))

    # Prioritize preprocessed keypoints_2d over raw pose_data
    if keypoints_2d is not None and keypoint_visibility is not None:
        # Convert normalized coordinates [0,1] to pixel coordinates
        # keypoints_2d from load_SMIL_Unreal_sample is already in [y_norm, x_norm] format
        # which matches what draw_joints expects [y, x], so just scale to pixels
        pixel_coords = keypoints_2d.copy()
        pixel_coords[:, 0] = pixel_coords[:, 0] * image_size[0]  # y coordinates  
        pixel_coords[:, 1] = pixel_coords[:, 1] * image_size[1]  # x coordinates
        
        joints = torch.tensor(pixel_coords.reshape(1, num_joints, 2), dtype=torch.float32)
        visibility = torch.tensor(keypoint_visibility.reshape(1, num_joints), dtype=torch.float32)
    elif pose_data is not None:
        # NOTE: This does not actually display the ground truth points directly.
        # Setting the visibility of the joints to 1 simply means, that the joints of the posed model are displayed.
        display_points_2D = [
            [pose_data[key]["2DPos"]["y"], pose_data[key]["2DPos"]["x"]]
            for key in pose_data.keys()
        ]
        joints = torch.tensor(
            np.array(display_points_2D).reshape(1, num_joints, 2), dtype=torch.float32
        )
        visibility = torch.ones((1, num_joints))
    else:
        joints = torch.zeros((1, num_joints, 2))
        visibility = torch.zeros((1, num_joints))

    return (rgb, sil, joints, visibility), filenames


def map_joint_order(joint_names_smil, joint_names_input, joints):
    """
    Map joint data to SMIL joint order regardless of input joint ordering.
    
    Args:
        joint_names_smil (list): List of joint names in the order expected by SMIL model.
        joint_names_input (list): List of joint names from the input data (Unreal Engine).
        joints (np.ndarray): Array of shape (N, 3) containing joint positions/angles
                            corresponding to joint_names_input order.
    
    Returns:
        np.ndarray: Array of shape (len(joint_names_smil), 3) with joint data reordered
                   to match SMIL joint name order. Unmatched joints are filled with zeros.
    
    Note:
        The root bone is expected to be the first entry in joint_names_smil.
        Joints that don't have a matching name in the input data will be set to zero.
    """
    # map joint names with correct ids regardless of order
    new_joint_locs = np.zeros((len(joint_names_smil), 3), float)

    for o, orig_joint in enumerate(joint_names_smil):
        for m, mapped_joints in enumerate(joint_names_input):
            if orig_joint == mapped_joints:
                new_joint_locs[o] = joints[m]  # flip x and y

    return new_joint_locs

def get_joint_angles_from_pose_data(pose_data):
    """
    Extract joint rotation angles from Unreal Engine pose data in quaternion format.
    
    Converts quaternion rotations from Unreal Engine pose data to angle-axis representation
    suitable for SMIL model. Handles the root bone (b_t) specially by setting its rotation
    to zero since global rotation is handled separately.
    
    Args:
        pose_data (dict): Dictionary containing joint pose data from Unreal Engine.
                          Expected format: {joint_name: {"quaternion": {"x": float, "y": float, 
                                                                        "z": float, "w": float}}}
                          where quaternions are in scalar-last format (x, y, z, w).
    
    Returns:
        np.ndarray: Array of shape (N, 3) containing angle-axis rotation vectors for each joint.
                    Each row represents [rx, ry, rz] where the rotation is around the axis
                    defined by the normalized vector with magnitude equal to the rotation angle.
    
    Note:
        - Uses scipy.spatial.transform.Rotation for quaternion to euler conversion
        - Converts to ZYX euler angles then to angle-axis representation
        - Applies coordinate system transformations: z=-z, x=-x for non-root bones
        - Root bone (b_t) rotation is set to zero as it's handled by global rotation
    """

    joint_angles = []
    joint_names = []

    for key in pose_data:
        joint_names.append(key)

        rot = Rotation.from_quat(
            [
                pose_data[key]["quaternion"]["x"],
                pose_data[key]["quaternion"]["y"],
                pose_data[key]["quaternion"]["z"],
                pose_data[key]["quaternion"]["w"],
            ],
            scalar_first=False,
        )

        rot_eul = rot.as_euler("zyx", degrees=False)

        # ignore root bone:
        if key != "b_t":
            theta, vector = eulerangles.euler2angle_axis(
                z=-rot_eul[0], y=rot_eul[1], x=-rot_eul[2]
            )
        else:
            # no rotation for root bone, that's handled in the global rotation
            theta, vector = eulerangles.euler2angle_axis(z=0, y=0, x=0)

        rodrigues_angle = vector * theta

        joint_angles.append(rodrigues_angle)

    return np.array(joint_angles), joint_names


"""
Plotting / Debugging functions
"""


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Function based on https://stackoverflow.com/questions/13685386/
    matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_3D_points(pose_data, input_image):
    """
    check 3D points look okay in matplotlib
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    display_points_names = [key for key in pose_data.keys()]
    display_points_3D = [
        [
            pose_data[key]["3DPos"]["x"],
            pose_data[key]["3DPos"]["y"],
            pose_data[key]["3DPos"]["z"],
        ]
        for key in pose_data.keys()
    ]

    for i, xyz in enumerate(display_points_3D):
        ax.scatter(xyz[0], xyz[1], xyz[2], marker="o", s=10)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    # use custom function to ensure equal axis proportions
    set_axes_equal(ax)

    # opens external plot
    plt.title(input_image.split("/")[-1])
    plt.show()


def plot_3D_projected_points(
    pose_data, input_image, cam_rot, cam_trans, fx, fy, cx, cy
):
    """
    Plots the 3D points and their projected 2D points on the image
    """
    R = cam_rot
    T = np.reshape(np.array(cam_trans), (3, 1))
    C = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    fig = plt.figure()
    ax = fig.add_subplot()

    display_img = cv2.imread(input_image)

    # Display the image as background
    ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))

    display_points_names = [key for key in pose_data.keys()]
    display_points_3D = [
        [
            pose_data[key]["3DPos"]["x"],
            pose_data[key]["3DPos"]["y"],
            pose_data[key]["3DPos"]["z"],
        ]
        for key in pose_data.keys()
    ]
    display_points_2D = [
        [pose_data[key]["2DPos"]["x"], pose_data[key]["2DPos"]["y"]]
        for key in pose_data.keys()
    ]

    for i, x in enumerate(display_points_3D):
        X = np.reshape(np.array(x), (3, -1))

        # given the above data, it should be possible to project the 3D points into the corresponding image,
        # so they land in the correct position on the image
        P = C @ np.hstack([R, T])  # projection matrix
        X_hom = np.vstack(
            [X, np.ones(X.shape[1])]
        )  # 3D points in homogenous coordinates

        X_hom = P @ X_hom  # project the 3D points

        X_2d = (
            X_hom[:2, :] / X_hom[2, :]
        )  # convert them back to 2D pixel space by dividing by the z coordinate

        gt_x_2d = display_points_2D[i][0]
        gt_y_2d = display_points_2D[i][1]

        ax.scatter(gt_x_2d, gt_y_2d, marker="o", s=20)
        ax.scatter(X_2d[0], display_img.shape[1] - X_2d[1], marker="^", s=8)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")

    ax.set_xlim([0, display_img.shape[0]])
    ax.set_ylim([0, display_img.shape[1]])
    ax.set_aspect("equal")

    ax.invert_yaxis()

    # opens external plot
    plt.title(input_image.split("/")[-1] + " projected")
    plt.show()


def create_sphere_meshes_at_points(
    points, radius=0.05, device="cpu", color=None, use_rainbow=True, mirror_x=True
):
    """
    Creates a PyTorch3D mesh of a sphere for each 3D point.

    Args:
        points (numpy.ndarray): Array of shape (N, 3) containing 3D point coordinates.
        radius (float): Radius of each sphere. Default is 0.05.
        device (str): Device to place tensors on ('cpu' or 'cuda'). Default is 'cpu'.
        color (list or numpy.ndarray): RGB color for all spheres or array of shape (N, 3) for individual colors.
                                      Values should be in range [0, 1]. Default is red [1.0, 0.0, 0.0].
        use_rainbow (bool): If True, use a rainbow color scheme instead of the provided color. Default is True.
        mirror_x (bool): If True, mirror the x coordinates. Default is True.

    Returns:
        Meshes: A single PyTorch3D Meshes object containing all spheres.
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    if len(points.shape) == 1:
        # Single point, reshape to (1, 3)
        points = points.reshape(1, 3)

    # Mirror x coordinates if requested
    if mirror_x:
        points = points.copy()
        points[:, 0] = -points[:, 0]

    # Default color is red if not specified
    if color is None:
        color = [1.0, 0.0, 0.0]  # Red

    # Generate rainbow colors if requested
    if use_rainbow and points.shape[0] > 1:
        # Create a rainbow color map
        rainbow_colors = np.zeros((points.shape[0], 3))
        for i in range(points.shape[0]):
            # Calculate hue value between 0 and 1 based on point index
            hue = i / (points.shape[0] - 1)

            # Convert HSV to RGB (simplified version)
            # This creates a rainbow from red->yellow->green->cyan->blue->magenta->red
            h = hue * 6.0
            x = 1.0 - abs(h % 2.0 - 1.0)

            if h < 1.0:
                rainbow_colors[i] = [1.0, x, 0.0]
            elif h < 2.0:
                rainbow_colors[i] = [x, 1.0, 0.0]
            elif h < 3.0:
                rainbow_colors[i] = [0.0, 1.0, x]
            elif h < 4.0:
                rainbow_colors[i] = [0.0, x, 1.0]
            elif h < 5.0:
                rainbow_colors[i] = [x, 0.0, 1.0]
            else:
                rainbow_colors[i] = [1.0, 0.0, x]

        color = rainbow_colors

    # Convert to tensor and move to device
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)

    # Create a base sphere
    base_sphere = ico_sphere(level=2, device=device)
    base_verts = base_sphere.verts_padded()[0]  # Shape: (V, 3)
    base_faces = base_sphere.faces_padded()[0]  # Shape: (F, 3)

    all_verts = []
    all_faces = []
    all_colors = []

    for i in range(points.shape[0]):
        # Scale the sphere
        verts = base_verts * radius

        # Translate the sphere to the point location
        verts = verts + points_tensor[i]

        # Set color for this sphere
        if isinstance(color, np.ndarray) and color.shape[0] == points.shape[0]:
            sphere_color = torch.tensor(color[i], dtype=torch.float32, device=device)
        else:
            sphere_color = torch.tensor(color, dtype=torch.float32, device=device)

        # Add to lists
        all_verts.append(verts)

        # Offset faces for each new sphere
        faces_offset = base_faces + (i * base_verts.shape[0])
        all_faces.append(faces_offset)

        # Create vertex colors for this sphere
        verts_rgb = sphere_color.expand(verts.shape[0], 3)
        all_colors.append(verts_rgb)

    # Concatenate all vertices, faces, and colors
    all_verts = torch.cat(all_verts, dim=0)
    all_faces = torch.cat(all_faces, dim=0)
    all_colors = torch.cat(all_colors, dim=0)

    # Create textures
    textures = TexturesVertex(verts_features=[all_colors])

    # Create the mesh with all spheres
    spheres_mesh = Meshes(verts=[all_verts], faces=[all_faces], textures=textures)

    return spheres_mesh


def compute_keypoint_visibility(keypoints_2d, mask, image_width, image_height):
    """
    Compute keypoint visibility based on mask and image bounds.
    
    Args:
        keypoints_2d (np.array): Normalized 2D keypoint coordinates [0, 1]
        mask (np.array): Binary mask of the object (None if not available, already dilated when loaded)
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
        
    Returns:
        np.array: Visibility array (1 for visible, 0 for not visible)
    """
    num_keypoints = len(keypoints_2d)
    visibility = np.ones(num_keypoints)
    
    # Convert normalized coordinates to pixel coordinates
    pixel_coords = keypoints_2d.copy()
    pixel_coords[:, 0] *= image_height  # x coordinate
    pixel_coords[:, 1] *= image_width   # y coordinate
    
    # Check if keypoints are outside image bounds
    for i, (x, y) in enumerate(pixel_coords):
        if x < 0 or x >= image_height or y < 0 or y >= image_width:
            visibility[i] = 0
    
    # If mask is available, check visibility within mask
    if mask is not None:
        # Mask is already dilated when loaded, so use it directly
        for i, (x, y) in enumerate(pixel_coords):
            if visibility[i] == 1:  # Only check if not already outside bounds
                x_int, y_int = int(round(x)), int(round(y))
                if 0 <= x_int < image_height and 0 <= y_int < image_width:
                    if mask[x_int, y_int] == 0:
                        visibility[i] = 0
    
    return visibility


def load_SMIL_Unreal_sample(json_file_path, 
                        plot_tests=False, 
                        propagate_scaling=True, 
                        translation_factor=0.01,
                        load_image=True,
                        verbose=False):
    """
    Load a SMIL sample from replicAnt generated SMIL data and return the loaded image and SMIL data

    Args:
        json_file_path (str): Path to the JSON file containing the SMIL data
        plot_tests (bool): Whether to plot the tests
        propagate_scaling (bool): Whether to propagate scaling to the child joints
        translation_factor (float): The factor to multiply the translation by
        verbose (bool): Whether to print verbose output

    Returns:
        tuple: (x_output, y_output)
            - x_output (dict): Dictionary containing the input data
                - input_image (str): Input image path
                - input_image_data (np.array): Input image data
                - input_image_mask (np.array): Input image mask
            - y_output (dict): Dictionary containing the output data
                - pose_data (dict): Raw pose data
                - joint_angles (np.array): Joint angles
                - joint_names (list): Joint names
                - cam_rot (torch.tensor): Camera rotation
                - cam_trans (torch.tensor): Camera translation
                - cam_rot_orig (np.array): Original camera rotation from Unreal data
                - cam_trans_orig (np.array): Original camera translation from Unreal data
                - cx (float): Image centre x
                - cy (float): Image centre y
                - fx (float): Focal length x
                - fy (float): Focal length y
                - cam_fov (float): Camera FOV
                - scale_weights (np.array): Scale weights
                - trans_weights (np.array): Translation weights
                - root_loc (np.array): Root location
                - root_rot (np.array): Root rotation
                - translation_factor (float): for scaling sampled translations
                - propagate_scaling (bool): Whether to propagate scaling to the child joints

    """

    """
    STEP 1 - LOAD replicAnt generated SMIL data
    """
    x_output = {}
    y_output = {}

    # get the batch data file path
    batch_data_file_path = json_file_path.replace(
        json_file_path.split("/")[-1],
        "_BatchData_" + json_file_path.split("/")[-2] + ".json",
    )

    # get the input image path
    input_image = json_file_path.split(".")[0] + ".JPG"

    if load_image:
        # load the image
        input_image_data = imageio.v2.imread(input_image)
    else:
        input_image_data = None

    x_output["input_image"] = input_image
    x_output["input_image_data"] = input_image_data
    
    # load the input image mask
    try:
        input_image_mask = imageio.v2.imread(input_image.replace(".JPG", "_ID.png"))
        # convert to binary mask (in replicAnt data, the mask is in the red channel)
        input_image_mask = cv2.threshold(input_image_mask, 0, 255, cv2.THRESH_BINARY)[1]
        # Extract single channel if multi-channel
        if len(input_image_mask.shape) > 2:
            input_image_mask = input_image_mask[:, :, 0]
        
        # Dilate the mask to improve quality (Unreal generated masks are very thin)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        input_image_mask = cv2.dilate(input_image_mask, kernel, iterations=2)
        
        x_output["input_image_mask"] = input_image_mask
    except FileNotFoundError:
        input_image_mask = None
        x_output["input_image_mask"] = None

    # load the json data
    with open(json_file_path, "r") as file:
        data = json.load(file)

    with open(batch_data_file_path, "r") as file:
        batch_data_file = json.load(file)

    # Extract pose data contained in "keypoints"
    pose_data = data["iterationData"]["subject Data"][0]["1"]["keypoints"]
    y_output["pose_data"] = pose_data

    # get camera data into correct format: 
    # extrinsics: rotation and translation
    cam_rot, cam_trans = parse_projection_components(data)
    y_output["cam_rot_orig"] = cam_rot
    y_output["cam_trans_orig"] = cam_trans

    # intrinsics: image centre, focal length, and field of view
    cx, cy, fx, fy = parse_camera_intrinsics(
        batch_data_file=batch_data_file, iteration_data_file=data
    )

    cam_fov = [data["iterationData"]["camera"]["FOV"]]

    y_output["cam_fov"] = cam_fov
    y_output["cx"] = cx
    y_output["cy"] = cy
    y_output["fx"] = fx
    y_output["fy"] = fy

    if verbose:
        print("INFO: Camera rotation", cam_rot)
        print("INFO: Camera translation", cam_trans)
        print("INFO: Image centre x", cx)
        print("INFO: Image centre y", cy)
        print("INFO: Focal length x", fx)
        print("INFO: Focal length y", fy)
        print("INFO: Camera FOV", cam_fov)

    # read Scale and Translation weights from iteration file, if available
    try:
        scale_weights = data["iterationData"]["subject Data"][0]["1"]["ScaleWeights"]
        trans_weights = data["iterationData"]["subject Data"][0]["1"]["TranslationWeights"]
        if verbose:
            print("INFO: Scale and translation weights found.")
    except KeyError:
        scale_weights = None
        trans_weights = None
        if verbose:
            print("INFO: No scale and translation weights")

    y_output["scale_weights"] = scale_weights
    y_output["trans_weights"] = trans_weights
    y_output["translation_factor"] = translation_factor
    y_output["propagate_scaling"] = propagate_scaling

    # Extract shape and pose parameters
    try:
        shape_betas = data["iterationData"]["subject Data"][0]["1"]["shape betas"]
        if isinstance(shape_betas, dict):
            shape_betas_temp = []
            for key, value in shape_betas.items():
                shape_betas_temp.append(value)
            shape_betas = shape_betas_temp
    except KeyError:
        shape_betas = []


    joint_angles, joint_names = get_joint_angles_from_pose_data(pose_data)

    # map joints, in case the order differs. The root bone is expected to be the first entry
    np_joint_angles_mapped = map_joint_order(
        config.dd["J_names"], joint_names, joint_angles
    )

    y_output["joint_angles"] = np_joint_angles_mapped
    y_output["joint_names"] = config.dd["J_names"] # should be identical to joint_names from the loaded data but just to be safe

    # Convert shape betas to a NumPy array
    if len(shape_betas) == 0:
        shape_betas = np.zeros(config.dd["shapedirs"].shape[2])
    else:
        shape_betas = np.array(shape_betas)

    y_output["shape_betas"] = shape_betas

    # Check if the loaded shape betas have the same number of dimensions as the model
    if shape_betas.shape[0] != config.dd["shapedirs"].shape[2]:
        print("\nERROR: Shape betas have the wrong number of dimensions")
        print("INFO: Shape betas from data:", len(shape_betas))
        print("INFO: Model shape betas:", config.dd["shapedirs"].shape[2])
        exit()

    if verbose:
        # Print the extracted shape and pose data
        print("Shape Betas:", shape_betas.shape)
        print("Pose Data:", np_joint_angles_mapped.shape)

    # Convert to PyTorch tensors and extract R, T components
    # Mirror the rotation matrix for x-axis
    mirror_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Apply mirroring to rotation matrix
    mirrored_rot = mirror_matrix @ cam_rot.T @ mirror_matrix.T
    R = torch.tensor(mirrored_rot, dtype=torch.float32)

    # Mirror the translation vector for x-axis
    T = torch.tensor([-cam_trans[0], cam_trans[1], cam_trans[2]], dtype=torch.float32)

    y_output["cam_rot"] = R
    y_output["cam_trans"] = T

    # get the model location
    model_loc = np.array(
        [
            -pose_data["b_t"]["3DPos"]["x"],
            pose_data["b_t"]["3DPos"]["y"],
            pose_data["b_t"]["3DPos"]["z"],
        ],
        dtype=np.float32,
    )

    rot = Rotation.from_quat(
        [
            pose_data["b_t"]["globalRotation"]["x"],
            pose_data["b_t"]["globalRotation"]["y"],
            pose_data["b_t"]["globalRotation"]["z"],
            pose_data["b_t"]["globalRotation"]["w"],
        ],
        scalar_first=False,
    )

    rot_eul = rot.as_euler("zyx", degrees=False)

    # Z is actually Z here (so YAW)
    # Y is PITCH
    # X is ROLL

    theta, vector = eulerangles.euler2angle_axis(
        z=-rot_eul[0] + np.pi, y=-rot_eul[1], x=rot_eul[2]
    )

    global_rotation_np = vector * theta

    y_output["root_loc"] = model_loc
    y_output["root_rot"] = global_rotation_np

    # Extract 2D keypoint coordinates for training
    keypoints_2d = []
    keypoint_names = []
    
    # Get image dimensions for normalization
    image_width = batch_data_file["Image Resolution"]["x"]
    image_height = batch_data_file["Image Resolution"]["y"]
    
    for key in pose_data.keys():
        keypoint_names.append(key)
        # Note: y and x are swapped to account for coordinate system differences
        # Normalize coordinates to [0, 1] range based on image dimensions
        norm_x = pose_data[key]["2DPos"]["y"] / image_height
        norm_y = pose_data[key]["2DPos"]["x"] / image_width
        keypoints_2d.append([norm_x, norm_y])
    
    # Map keypoints to SMIL joint order (create a 2D-specific mapping)
    mapped_keypoints_2d = np.zeros((len(config.dd["J_names"]), 2), float)
    for o, orig_joint in enumerate(config.dd["J_names"]):
        for m, mapped_joint in enumerate(keypoint_names):
            if orig_joint == mapped_joint:
                mapped_keypoints_2d[o] = keypoints_2d[m]  # Assign normalized 2D coordinates
    
    y_output["keypoints_2d"] = mapped_keypoints_2d  # Normalized 2D coordinates [0, 1]

    # Compute keypoint visibility based on mask and image bounds
    # Only compute when image and mask are loaded to ensure efficiency
    if load_image and x_output["input_image_mask"] is not None:
        # Compute visibility using the loaded mask and image dimensions
        y_output["keypoint_visibility"] = compute_keypoint_visibility(
            mapped_keypoints_2d, 
            x_output["input_image_mask"], 
            image_width, 
            image_height
        )
    else:
        # Fallback: only check image bounds, treat all joints as visible if within bounds
        visibility = np.ones(len(config.dd["J_names"]))
        for i, (norm_x, norm_y) in enumerate(mapped_keypoints_2d):
            pixel_x = norm_x * image_height
            pixel_y = norm_y * image_width
            if pixel_x < 0 or pixel_x >= image_height or pixel_y < 0 or pixel_y >= image_width:
                visibility[i] = 0
        y_output["keypoint_visibility"] = visibility

    # --- Re-parameterize scene: place model at origin with zero rotation ---
    # PyTorch3D uses row-vector convention: X_cam = X_world R + T
    # With X_world = X_model R_model + t_model, the equivalent camera extrinsics are:
    #   R_cam_new = R_model R_cam
    #   T_cam_new = t_model R_cam + T_cam
    # Derive model rotation directly from Unreal quaternion and mirror to PyTorch3D
    rot_model_ue = Rotation.from_quat(
        [
            -pose_data["b_t"]["globalRotation"]["x"],
            -pose_data["b_t"]["globalRotation"]["y"],
            -pose_data["b_t"]["globalRotation"]["z"],
            pose_data["b_t"]["globalRotation"]["w"],
        ],
        scalar_first=False,
    )
    R_model_ue = rot_model_ue.as_matrix().astype(np.float32)
    R_model_p3d = mirror_matrix @ R_model_ue @ mirror_matrix.T

    t_model_p3d = y_output["root_loc"].astype(np.float32)

    R_cam_p3d = y_output["cam_rot"].cpu().numpy() if isinstance(y_output["cam_rot"], torch.Tensor) else np.array(y_output["cam_rot"], dtype=np.float32)
    T_cam_p3d = y_output["cam_trans"].cpu().numpy() if isinstance(y_output["cam_trans"], torch.Tensor) else np.array(y_output["cam_trans"], dtype=np.float32)

    # Compose camera relative to the (now-origin) model using PyTorch3D row-vector convention
    R_cam_new = R_model_p3d @ R_cam_p3d
    T_cam_new = t_model_p3d @ R_cam_p3d + T_cam_p3d

    # Apply -180° yaw correction around the model's up axis (z) to align Unreal/PyTorch3D conventions
    yaw_offset = np.pi
    cos_y = np.cos(yaw_offset).astype(np.float32)
    sin_y = np.sin(yaw_offset).astype(np.float32)
    Rz = np.array([[cos_y, -sin_y, 0.0], 
                   [sin_y, cos_y, 0.0], 
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    R_cam_new = Rz @ R_cam_new

    # Update outputs: camera now encodes the relative transform; model at origin with zero rotation
    y_output["cam_rot"] = torch.tensor(R_cam_new, dtype=torch.float32)
    y_output["cam_trans"] = torch.tensor(T_cam_new, dtype=torch.float32)
    y_output["root_loc"] = np.zeros_like(t_model_p3d, dtype=np.float32)
    y_output["root_rot"] = np.zeros(3, dtype=np.float32)

    if verbose:
        print("\nINFO: Sucessfully loaded data from", json_file_path)

    return x_output, y_output


def Render_SMAL_Model_from_Unreal_data(x,y,device,verbose=False):
    """
    Render a SMAL model from Unreal data
    
    Args:
        x (dict): Dictionary containing the input data
        y (dict): Dictionary containing the output data
    
    """

    # Use processed normalized coordinates instead of raw pose_data for better accuracy
    data_json, filenames = return_placeholder_data(
        input_image=x["input_image"], 
        num_joints=len(y["joint_angles"]), 
        keypoints_2d=y["keypoints_2d"],
        keypoint_visibility=y["keypoint_visibility"],
        silhouette=x["input_image_mask"]
    )

    # Some code from the original SMALFitter to set up the model
    if not config.ignore_hardcoded_body:
        assert config.SHAPE_FAMILY >= 0, "Shape family should be greater than 0"

        use_unity_prior = config.SHAPE_FAMILY == 1 and not config.FORCE_SMAL_PRIOR
    else:
        use_unity_prior = False

    if not use_unity_prior and not config.ALLOW_LIMB_SCALING:
        config.ALLOW_LIMB_SCALING = False

    model = SMALFitter(
        device, data_json, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior
    )

    # model parameters
    model.betas = torch.nn.Parameter(torch.Tensor(y["shape_betas"]).to(device))

    # Check if scaledirs and transdirs exist in the model data
    if "scaledirs" in config.dd:
        if verbose:
            print(f"Found scaledirs in model with shape: {config.dd['scaledirs'].shape}")
        scaledirs_found = True
    else:
        if verbose:
            print("No scaling components (scaledirs) found in model data")
        scaledirs_found = False
    if "transdirs" in config.dd:
        if verbose:
            print(f"Found transdirs in model with shape: {config.dd['transdirs'].shape}")
        transdirs_found = True
    else:
        if verbose:
            print("No translation components (transdirs) found in model data")
        transdirs_found = False

    if scaledirs_found and transdirs_found and y["scale_weights"] is not None and y["trans_weights"] is not None:
        translation_out, scale_out = sample_pca_transforms_from_dirs(config.dd, y["scale_weights"], y["trans_weights"])

        # Scale remains log-space; add batch dimension
        model.log_beta_scales = torch.nn.Parameter(torch.from_numpy(np.log(scale_out))[None, ...].float().to(device))
        model.betas_trans = torch.nn.Parameter(torch.from_numpy(translation_out * y["translation_factor"])[None, ...].float().to(device))
        model.propagate_scaling = y["propagate_scaling"]


    else:
        print("No scaling or translation components found in model data")

    # set model joint rotations
    model.joint_rotations = torch.nn.Parameter(
        torch.Tensor(y["joint_angles"][1:])
        .reshape((1, y["joint_angles"][1:].shape[0], 3))
        .to(device)
    )


    """
    STEP 4 - Set up pytorch3d scene with the SMIL model
    """

    try:
        R = y["cam_rot"].clone().detach().to(device).unsqueeze(0)
        T = y["cam_trans"].clone().detach().to(device).unsqueeze(0)
    except AttributeError:
        R = torch.tensor(y["cam_rot"]).clone().detach().to(device).unsqueeze(0)
        T = torch.tensor(y["cam_trans"]).clone().detach().to(device).unsqueeze(0)

    model.fov = torch.nn.Parameter(torch.Tensor(y["cam_fov"]).to(device))
    model.renderer.cameras.fov = torch.nn.Parameter(torch.Tensor(y["cam_fov"]).to(device))
    model.renderer.cameras.R = torch.nn.Parameter(torch.Tensor(R).to(device))
    model.renderer.cameras.T = torch.nn.Parameter(torch.Tensor(T).to(device))


    global_rotation = torch.nn.Parameter(
        torch.from_numpy(y["root_rot"]).float().to(device).unsqueeze(0)
    )

    model.global_rotation = global_rotation
    model.trans = torch.nn.Parameter(torch.Tensor(np.array([y["root_loc"]])).to(device))


    """
    STEP 5 - RENDER POSED MESH
    """

    print("\nINFO: Rendering output")
    image_exporter = ImageExporter("LOCAL_TEST", [os.path.basename(x["input_image"])])
    image_exporter.stage_id = 0
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter, apply_UE_transform=True)




def plot_loaded_data_tests(x,y,device):


    # plot 3D points
    plot_3D_points(pose_data=y["pose_data"], input_image=x["input_image"])

    # check, the 3D points align with their corresponding projected 2D points
    plot_3D_projected_points(
        pose_data=y["pose_data"],
        input_image=x["input_image"],
        cam_rot=y["cam_rot_orig"],
        cam_trans=y["cam_trans_orig"],
        fx=y["fx"],
        fy=y["fy"],
        cx=y["cx"],
        cy=y["cy"],
    )


    # Create sphere meshes for keypoints - no need to mirror here as the function will do it
    keypoints_3d = np.array(
        [
            [
                y["pose_data"][key]["3DPos"]["x"],
                y["pose_data"][key]["3DPos"]["y"],
                y["pose_data"][key]["3DPos"]["z"],
            ]
            for key in y["pose_data"].keys()
        ]
    )

    cameras = FoVPerspectiveCameras(
        device=device, R=y["cam_rot"].unsqueeze(0), T=y["cam_trans"].unsqueeze(0), fov=y["cam_fov"], degrees=True, znear=0.01, zfar=5000
    )

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used.
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the ant is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    # Lights can also be moved
    lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=device)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    # Change specular color and change material shininess
    materials = Materials(
        device=device, specular_color=[[1.0, 1.0, 1.0]], shininess=10.0
    )

    # visualize keypoints with mirroring enabled
    sphere_meshes = create_sphere_meshes_at_points(
        keypoints_3d, radius=0.25, device=device, use_rainbow=True, mirror_x=True
    )

    # requires same types of materials for both meshes
    # combined_meshes = join_meshes_as_scene([mesh, sphere_meshes])

    # render the spheres
    sphere_images = renderer(
        sphere_meshes, lights=lights, materials=materials, cameras=cameras
    )

    plt.figure(figsize=(10, 10))

    # Read input image
    display_img = cv2.imread(x["input_image"])
    # Convert to RGB
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    # Get render dimensions
    render_height, render_width = sphere_images[0].shape[:2]

    # Resize input image to match render dimensions
    display_img = cv2.resize(display_img, (render_width, render_height))

    # Get render as numpy array and create mask
    render = sphere_images[0, ..., :3].cpu().numpy()
    mask = ~np.all(render == 1.0, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    # Blend render with input image using mask
    blended = np.where(mask, render, display_img / 255.0)

    plt.imshow(blended)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    """
    IMPORTANT: Verify the data has been generated with the same SMIL model that is referenced in the config.py file
    """
    # set the device to use (first available GPU by default)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Read the JSON file
    json_file_path = (
        "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-Masked/replicAnt-x-SMIL-OmniAnt-Masked_0000.json"
    )

    # Load the SMIL data
    x,y = load_SMIL_Unreal_sample(json_file_path, 
                                  plot_tests=False, 
                                  propagate_scaling=True, 
                                  translation_factor=0.01)

    # Verify things plot correctly
    plot_loaded_data_tests(x,y,device)

    # Render the SMAL model based on the loaded data
    Render_SMAL_Model_from_Unreal_data(x,y,device)