import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import imageio
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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


def _detect_pose_root(pose_data):
    """Return the name of the hierarchical root joint in a replicAnt pose_data dict.

    Strategy:
      1. If ``config.ROOT_JOINT`` is a key in pose_data, use it (legacy
         bug-skeleton datasets generated against the active SMAL model rely
         on this — preserves backwards compatibility).
      2. Otherwise fall back to the first key, which by Unreal serialisation
         convention is the root of the skeleton hierarchy (e.g. ``'Mskel'``
         for the mouse rig, ``'b_t'`` for the bug rig). This makes the loader
         independent of which SMAL model happens to be loaded in ``config``.
    """
    if config.ROOT_JOINT in pose_data:
        return config.ROOT_JOINT
    return next(iter(pose_data))


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
    suitable for SMIL model. Handles the root bone specially by setting its rotation
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
        - Root bone rotation is set to zero as it's handled by global rotation
    """

    joint_angles = []
    joint_names = []
    root_key = _detect_pose_root(pose_data)

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
        if key != root_key:
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


def refine_visibility_with_depth(
    visibility,
    keypoints_2d_normalized,
    keypoints_3d_world_raw,
    camera_location_world_raw,
    depth_image,
    image_width,
    image_height,
    depth_max_cm=1000.0,
    depth_tolerance_cm=5.0,
    depth_neighborhood=1,
):
    """
    Refine a per-joint visibility array with a replicAnt depth-buffer
    self-occlusion check. Mirrors the convention used in the sungaya
    pipeline so the two stay byte-equivalent.

    Encoding: replicAnt's depth pass packs a Euclidean camera-to-surface
    distance (in cm) into the RED channel of an RGBA uint8 PNG via a
    linear map `surface_cm = (R / 255) * depth_max_cm`. A joint is
    occluded when the keypoint's true distance to the camera exceeds
    the front-most surface distance (over a small neighborhood) by more
    than `depth_tolerance_cm`.

    Coordinates here are in raw Unreal world frame (cm, no PyTorch3D
    x-mirror). The (R/255)*range mapping is preserved, and distances
    are coordinate-frame agnostic as long as keypoint and camera live
    in the same frame.

    The refinement is monotone: it can only turn 1.0 -> 0.0, never the
    other way. Joints already at 0.0 and joints whose 3D ground truth
    is missing (NaN) are skipped.

    Args:
        visibility (np.array): Current visibility per joint (1.0 or 0.0). Refined in place AND returned.
        keypoints_2d_normalized (np.array): (n_joints, 2) — same axis-swapped
            (norm_x = 2DPos.y / H, norm_y = 2DPos.x / W) convention as the
            rest of this module. (row_idx, col_idx) = (norm_x * H, norm_y * W).
        keypoints_3d_world_raw (np.array): (n_joints, 3) — raw Unreal 3DPos
            in cm. NaN for joints with no ground-truth 3D position.
        camera_location_world_raw (np.array): (3,) — raw Unreal camera
            `Location` in cm.
        depth_image (np.array): (H, W, 4) uint8 RGBA loaded via imageio. Depth
            is in channel 0. Pass None to skip the refinement (visibility
            returned unchanged).
        image_width (int), image_height (int): Reference resolution that
            `keypoints_2d_normalized` was normalised against.
        depth_max_cm (float): Encoded range for the depth pass (cm).
        depth_tolerance_cm (float): Margin added to the surface distance
            before declaring a keypoint occluded. NB the depth pass is
            8-bit over `depth_max_cm` so one LSB ≈ depth_max_cm/255 cm
            (~3.92 cm at the 1000 cm default) — set tolerance to at
            least one LSB plus the expected interior-joint offset, or
            interior skeletal joints will get clipped.
        depth_neighborhood (int): Half-window in pixels for the surface
            min-depth lookup. `0` samples exactly one pixel; `1` uses 3x3.

    Returns:
        np.array: Updated visibility array (same object as input).
    """
    if depth_image is None:
        return visibility
    if depth_image.ndim != 3 or depth_image.shape[2] < 1:
        return visibility
    if (
        depth_image.shape[0] != image_height
        or depth_image.shape[1] != image_width
    ):
        return visibility

    cam_loc = np.asarray(camera_location_world_raw, dtype=np.float64)

    for j in range(len(visibility)):
        if visibility[j] == 0.0:
            continue
        p3 = keypoints_3d_world_raw[j]
        if not np.isfinite(p3).all():
            continue

        norm_x, norm_y = keypoints_2d_normalized[j]
        if not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0):
            continue
        pixel_row = int(np.clip(norm_x * image_height, 0, image_height - 1))
        pixel_col = int(np.clip(norm_y * image_width, 0, image_width - 1))

        if depth_neighborhood <= 0:
            r_val = int(depth_image[pixel_row, pixel_col, 0])
        else:
            r0 = max(0, pixel_row - depth_neighborhood)
            r1 = min(image_height, pixel_row + depth_neighborhood + 1)
            c0 = max(0, pixel_col - depth_neighborhood)
            c1 = min(image_width, pixel_col + depth_neighborhood + 1)
            r_val = int(depth_image[r0:r1, c0:c1, 0].min())

        surface_cm = (r_val / 255.0) * depth_max_cm
        dist_cm = float(np.linalg.norm(p3.astype(np.float64) - cam_loc))
        if dist_cm > (surface_cm + depth_tolerance_cm):
            visibility[j] = 0.0

    return visibility


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
    root_key = _detect_pose_root(pose_data)
    model_loc = np.array(
        [
            -pose_data[root_key]["3DPos"]["x"],
            pose_data[root_key]["3DPos"]["y"],
            pose_data[root_key]["3DPos"]["z"],
        ],
        dtype=np.float32,
    )

    rot = Rotation.from_quat(
        [
            pose_data[root_key]["globalRotation"]["x"],
            pose_data[root_key]["globalRotation"]["y"],
            pose_data[root_key]["globalRotation"]["z"],
            pose_data[root_key]["globalRotation"]["w"],
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

    # Extract 3D keypoints from pose_data before transformation
    keypoints_3d = []
    keypoint_names_3d = []
    for key in pose_data.keys():
        keypoint_names_3d.append(key)
        # Apply Unreal to PyTorch3D coordinate transformation (mirror x-axis)
        x_ue = pose_data[key]["3DPos"]["x"]
        y_ue = pose_data[key]["3DPos"]["y"] 
        z_ue = pose_data[key]["3DPos"]["z"]
        keypoints_3d.append([-x_ue, y_ue, z_ue])  # Mirror x-axis for Unreal to PyTorch3D
    
    # Map 3D keypoints to SMIL joint order
    mapped_keypoints_3d = np.zeros((len(config.dd["J_names"]), 3), float)
    for o, orig_joint in enumerate(config.dd["J_names"]):
        for m, mapped_joint in enumerate(keypoint_names_3d):
            if orig_joint == mapped_joint:
                mapped_keypoints_3d[o] = keypoints_3d[m]
    
    y_output["keypoints_3d_original"] = mapped_keypoints_3d.copy()  # Store original for debugging

    # --- Re-parameterize scene: place model at origin with zero rotation ---
    # PyTorch3D uses row-vector convention: X_cam = X_world R + T
    # With X_world = X_model R_model + t_model, the equivalent camera extrinsics are:
    #   R_cam_new = R_model R_cam
    #   T_cam_new = t_model R_cam + T_cam
    # Derive model rotation directly from Unreal quaternion and mirror to PyTorch3D
    rot_model_ue = Rotation.from_quat(
        [
            -pose_data[root_key]["globalRotation"]["x"],
            -pose_data[root_key]["globalRotation"]["y"],
            -pose_data[root_key]["globalRotation"]["z"],
            pose_data[root_key]["globalRotation"]["w"],
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
    
    # Transform 3D keypoints to match the model transformation
    # Apply the inverse transformation to move keypoints from world coordinates 
    # to model-centered coordinates (same transformation applied to the model)
    # X_model_centered = (X_world - t_model) @ R_model^T @ Rz^T
    # where Rz is the -180° yaw correction
    
    # First, apply the inverse yaw correction (transpose of Rz)
    Rz_inv = Rz.T
    
    # Transform keypoints: translate by -t_model, then rotate by inverse of R_model, then apply inverse yaw correction
    keypoints_3d_transformed = []
    for kp_3d in mapped_keypoints_3d:
        # Translate to model origin
        kp_translated = kp_3d - t_model_p3d
        # Apply inverse model rotation (R_model^T)
        kp_rotated = kp_translated @ R_model_p3d.T
        # Apply inverse yaw correction  
        kp_final = kp_rotated @ Rz_inv
        keypoints_3d_transformed.append(kp_final)
    
    y_output["keypoints_3d"] = np.array(keypoints_3d_transformed, dtype=np.float32)

    if verbose:
        print("\nINFO: Sucessfully loaded data from", json_file_path)

    return x_output, y_output


def load_SMIL_Unreal_multiview_sample(
    data_path: str,
    frame_index: int,
    camera_indices: list = None,
    propagate_scaling: bool = True,
    translation_factor: float = 0.1,
    load_images: bool = True,
    canonical_frame: bool = True,
    depth_occlusion_check: bool = True,
    depth_max_cm: float = 1000.0,
    depth_tolerance_cm: float = 5.0,
    depth_neighborhood: int = 1,
    verbose: bool = False
):
    """
    Load a multi-view SMIL sample from replicAnt flat-directory dataset.

    Handles multi-camera replicAnt data where each frame/camera has separate JSON and image files.
    File naming: {dataset_name}_{frame_idx:05d}_CAM{camera_id}.{ext}

    Args:
        data_path (str): Root directory of the multi-camera dataset
        frame_index (int): Frame index to load (0-9999)
        camera_indices (list, optional): List of camera IDs to load (e.g., [1,2,3]).
                                        If None, loads all 12 cameras.
        propagate_scaling (bool): Whether to propagate scaling to child joints
        translation_factor (float): Uniform scale applied at load time to all
            world-frame translations (`root_loc`, `cam_trans_per_view`,
            `keypoints_3d`, `keypoints_3d_world`, `canonical_to_world_t`).
            Default 0.1 matches the SMAL mouse mesh's native size to the
            Unreal data frame, so trainer-side `mesh + trans` (no extra
            rescaling) projects correctly with `use_ue_scaling=False`.
            See "Scale Unification" in MULTIVIEW_REPLICANT_INTEGRATION_DESIGN.md.
        load_images (bool): Whether to load image data
        depth_occlusion_check (bool): If True (default), refine per-view
            visibility with the replicAnt depth-buffer self-occlusion test.
            AND-composed with the existing ID-mask check. Falls back
            silently to id-mask-only visibility for any view whose
            `_Depth_CAM{id}.png` is absent.
        depth_max_cm (float): Encoded range of the depth pass in cm.
            replicAnt defaults to 1000 cm; tune only if the depth pass was
            re-exported with a different range.
        depth_tolerance_cm (float): Margin added to the surface distance
            before declaring a joint occluded. Default of 5 cm is one
            depth-LSB (~3.92 cm at the 1000 cm range) plus ~1 cm
            interior-joint slack; bump if interior joints still get
            clipped.
        depth_neighborhood (int): Half-window in pixels for the surface
            min-depth sample. 0 = exact pixel; 1 = 3x3 patch (default,
            matches sungaya).
        canonical_frame (bool): If True (default), re-express all camera
            extrinsics and model pose in the canonical camera's frame
            (lowest CAM ID -> R=I, t=0). If False, leave everything in
            raw PyTorch3D-mirrored world frame. See the canonical-frame
            block below for the transformation, conventions, and the
            decision to store the FORWARD transform under
            `canonical_to_world`.
        verbose (bool): Whether to print verbose output

    Returns:
        tuple: (x_output, y_output)
            - x_output (dict): Per-camera input data
                - image_data: List of image arrays per camera
                - image_paths: List of image file paths
                - input_image_mask: List of ID mask arrays per camera
                - mask_paths: List of mask file paths
                - num_views: Number of valid cameras loaded
                - camera_ids: List of camera IDs loaded
            - y_output (dict): Shared and per-view parameters
                - Shared data (identical across cameras):
                  - joint_angles, shape_betas, root_loc, root_rot, etc.
                  - keypoints_3d: (n_joints, 3) GT 3D in canonical frame
                    when canonical_frame=True, otherwise world frame.
                    Mapped to model J_names order; zeros for missing.
                  - keypoints_3d_world: same shape; always raw
                    PyTorch3D-mirrored world frame (kept for round-trip
                    checks and convention-switch flexibility).
                  - canonical_to_world: tuple (R_0, t_0) — the FORWARD
                    transform world -> canonical (x_can = x_w @ R_0 + t_0).
                    Invert at decode time: x_w = (x_can - t_0) @ R_0.T.
                  - canonical_cam_id: int — CAM ID used as canonical, or
                    -1 when canonical_frame=False.
                - Per-camera data:
                  - keypoints_2d_per_view: List of [n_joints, 2] arrays
                  - keypoint_visibility_per_view: List of [n_joints] visibility arrays
                  - cam_rot_per_view: List of rotation matrices
                  - cam_trans_per_view: List of translation vectors
                  - fx_per_view, fy_per_view, cx_per_view, cy_per_view: Lists of intrinsics
    """
    x_output = {}
    y_output = {}

    data_path = Path(data_path)

    # Detect dataset name from _BatchData_*.json files
    batch_files = list(data_path.glob("_BatchData_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No _BatchData_*.json found in {data_path}")

    batch_data_path = batch_files[0]
    dataset_name = batch_data_path.stem.replace("_BatchData_", "")

    with open(batch_data_path, "r") as f:
        batch_data_file = json.load(f)

    # Determine camera indices to load
    if camera_indices is None:
        # Auto-detect: look for CAM files for this frame
        cam_files = list(data_path.glob(f"{dataset_name}_{frame_index:05d}_CAM*.json"))
        camera_indices = sorted([
            int(re.search(r'CAM(\d+)', f.name).group(1))
            for f in cam_files
        ])

    if not camera_indices:
        raise FileNotFoundError(
            f"No camera files found for frame {frame_index:05d} in {data_path}"
        )

    if verbose:
        print(f"Loading frame {frame_index:05d} with cameras: {camera_indices}")

    # Load shared data from first camera (pose/shape identical across all cameras)
    first_camera_json = data_path / f"{dataset_name}_{frame_index:05d}_CAM{camera_indices[0]}.json"
    with open(first_camera_json, "r") as f:
        first_camera_data = json.load(f)

    # Extract shared pose and shape data
    pose_data = first_camera_data["iterationData"]["subject Data"][0]["1"]["keypoints"]

    # Extract shape betas (same across all cameras)
    try:
        shape_betas = first_camera_data["iterationData"]["subject Data"][0]["1"]["shape betas"]
        if isinstance(shape_betas, dict):
            shape_betas = [v for v in shape_betas.values()]
    except KeyError:
        shape_betas = []

    if len(shape_betas) == 0:
        shape_betas = np.zeros(config.dd["shapedirs"].shape[2])
    else:
        shape_betas = np.array(shape_betas)

    y_output["shape_betas"] = shape_betas

    # Extract joint angles from pose data (same across cameras)
    joint_angles, joint_names = get_joint_angles_from_pose_data(pose_data)
    np_joint_angles_mapped = map_joint_order(
        config.dd["J_names"], joint_names, joint_angles
    )
    y_output["joint_angles"] = np_joint_angles_mapped
    y_output["joint_names"] = config.dd["J_names"]

    # Extract scale and translation weights if available
    try:
        scale_weights = first_camera_data["iterationData"]["subject Data"][0]["1"]["ScaleWeights"]
        trans_weights = first_camera_data["iterationData"]["subject Data"][0]["1"]["TranslationWeights"]
    except KeyError:
        scale_weights = None
        trans_weights = None

    y_output["scale_weights"] = scale_weights
    y_output["trans_weights"] = trans_weights
    y_output["translation_factor"] = translation_factor
    y_output["propagate_scaling"] = propagate_scaling

    # Extract shared global rotation and translation from pose data
    root_key = _detect_pose_root(pose_data)
    model_loc = np.array(
        [
            -pose_data[root_key]["3DPos"]["x"],
            pose_data[root_key]["3DPos"]["y"],
            pose_data[root_key]["3DPos"]["z"],
        ],
        dtype=np.float32,
    )

    rot = Rotation.from_quat(
        [
            pose_data[root_key]["globalRotation"]["x"],
            pose_data[root_key]["globalRotation"]["y"],
            pose_data[root_key]["globalRotation"]["z"],
            pose_data[root_key]["globalRotation"]["w"],
        ],
        scalar_first=False,
    )
    rot_eul = rot.as_euler("zyx", degrees=False)
    theta, vector = eulerangles.euler2angle_axis(
        z=-rot_eul[0] + np.pi, y=-rot_eul[1], x=rot_eul[2]
    )
    global_rotation_np = vector * theta

    y_output["root_loc"] = model_loc
    y_output["root_rot"] = global_rotation_np
    y_output["pose_data"] = pose_data  # Keep raw pose data for compatibility

    # Raw-Unreal-frame 3D keypoints (no x-mirror). Needed for the depth
    # occlusion check, where the camera Location is read from the JSON
    # un-mirrored and distances must be computed in a consistent frame.
    # Missing joints stay NaN so the depth helper skips them.
    keypoints_3d_unreal_raw = np.full(
        (len(config.dd["J_names"]), 3), np.nan, dtype=np.float32
    )
    if depth_occlusion_check:
        kp3d_raw_by_name = {}
        for k, kp in pose_data.items():
            p3 = kp.get("3DPos")
            if p3 is not None:
                kp3d_raw_by_name[k] = np.array(
                    [p3["x"], p3["y"], p3["z"]], dtype=np.float32
                )
        for o, j in enumerate(config.dd["J_names"]):
            if j in kp3d_raw_by_name:
                keypoints_3d_unreal_raw[o] = kp3d_raw_by_name[j]

    # Per-camera lists
    image_data = []
    image_paths = []
    mask_data = []
    mask_paths = []
    depth_paths = []

    keypoints_2d_per_view = []
    keypoint_visibility_per_view = []
    keypoint_in_dataset_per_view = []
    cam_rot_per_view = []
    cam_trans_per_view = []
    fx_per_view = []
    fy_per_view = []
    cx_per_view = []
    cy_per_view = []
    fov_per_view = []

    image_width = batch_data_file["Image Resolution"]["x"]
    image_height = batch_data_file["Image Resolution"]["y"]

    # Load per-camera data
    for cam_id in camera_indices:
        json_path = data_path / f"{dataset_name}_{frame_index:05d}_CAM{cam_id}.json"

        with open(json_path, "r") as f:
            cam_data = json.load(f)

        # Load image
        image_path = json_path.with_suffix(".JPG")
        if load_images and image_path.exists():
            img = imageio.v2.imread(str(image_path))
            image_data.append(img)
        else:
            image_data.append(None)
        image_paths.append(str(image_path))

        # Load ID mask for visibility computation
        mask_path = json_path.with_name(
            f"{dataset_name}_{frame_index:05d}_ID_CAM{cam_id}.png"
        )
        if mask_path.exists():
            id_mask = imageio.v2.imread(str(mask_path))
            # Convert to binary mask
            if len(id_mask.shape) > 2:
                id_mask = id_mask[:, :, 0]
            id_mask = cv2.threshold(id_mask, 0, 255, cv2.THRESH_BINARY)[1]
            mask_data.append(id_mask)
        else:
            mask_data.append(None)
        mask_paths.append(str(mask_path))

        # Load depth pass for the self-occlusion refinement. Optional —
        # if the file is missing we fall back to id-mask-only visibility
        # for this view per the design decision.
        depth_path = json_path.with_name(
            f"{dataset_name}_{frame_index:05d}_Depth_CAM{cam_id}.png"
        )
        depth_paths.append(str(depth_path))
        depth_image = None
        if depth_occlusion_check and depth_path.exists():
            depth_image = imageio.v2.imread(str(depth_path))

        # Extract per-camera 2D keypoints. The shared `pose_data` above was
        # read from the first camera's JSON; 2DPos is per-camera, so we re-
        # read from the current camera's own keypoints dict here.
        cam_pose = cam_data["iterationData"]["subject Data"][0]["1"]["keypoints"]
        keypoints_2d = []
        keypoint_names = []
        for key in cam_pose.keys():
            keypoint_names.append(key)
            # Intentional axis swap (norm_x <- y/H, norm_y <- x/W) — mirrors the
            # single-view path above. The downstream ID-mask lookup pixel-indexes
            # against these values with the same swap, so don't "fix" one side
            # without updating the other.
            try:
                norm_x = cam_pose[key]["2DPos"]["y"] / image_height
                norm_y = cam_pose[key]["2DPos"]["x"] / image_width
            except (KeyError, TypeError):
                # Fallback if 2D positions not in this camera's data
                norm_x = 0.5
                norm_y = 0.5
            keypoints_2d.append([norm_x, norm_y])

        # Map to SMIL joint order. Track which model J_names actually
        # have a matching dataset entry this view; model-only joints
        # have no GT 2D (left at the [0, 0] sentinel) and must NOT be
        # treated as visible — see the visibility initialisation below.
        mapped_keypoints_2d = np.zeros((len(config.dd["J_names"]), 2), float)
        in_dataset_this_view = np.zeros(len(config.dd["J_names"]), dtype=bool)
        for o, orig_joint in enumerate(config.dd["J_names"]):
            for m, mapped_joint in enumerate(keypoint_names):
                if orig_joint == mapped_joint:
                    mapped_keypoints_2d[o] = keypoints_2d[m]
                    in_dataset_this_view[o] = True

        keypoints_2d_per_view.append(mapped_keypoints_2d)
        keypoint_in_dataset_per_view.append(in_dataset_this_view)

        # Initialise visibility from the in-dataset bitmap so model-only
        # joints stay invisible regardless of where their [0, 0]
        # sentinel happens to land on the ID/depth pass. The bounds /
        # mask / depth steps below can only ever decrease visibility.
        visibility = in_dataset_this_view.astype(np.float64)
        if mask_data[-1] is not None:
            id_mask = mask_data[-1]
            for i, (norm_x, norm_y) in enumerate(mapped_keypoints_2d):
                if visibility[i] == 0.0:
                    continue  # not in dataset — never visible
                # Check bounds
                if not (0 <= norm_x <= 1.0 and 0 <= norm_y <= 1.0):
                    visibility[i] = 0.0
                    continue
                # Check ID mask at keypoint location
                pixel_x = int(np.clip(norm_x * image_height, 0, image_height - 1))
                pixel_y = int(np.clip(norm_y * image_width, 0, image_width - 1))
                if id_mask[pixel_x, pixel_y] == 0:
                    visibility[i] = 0.0
        else:
            # Fallback: only check bounds
            for i, (norm_x, norm_y) in enumerate(mapped_keypoints_2d):
                if visibility[i] == 0.0:
                    continue  # not in dataset — never visible
                if not (0 <= norm_x <= 1.0 and 0 <= norm_y <= 1.0):
                    visibility[i] = 0.0

        # Depth-buffer self-occlusion refinement (AND with id-mask result).
        # No-op when depth_image is None (missing file or check disabled).
        if depth_occlusion_check and depth_image is not None:
            cam_loc_unreal_raw = np.array(
                [
                    cam_data["iterationData"]["camera"]["Location"]["x"],
                    cam_data["iterationData"]["camera"]["Location"]["y"],
                    cam_data["iterationData"]["camera"]["Location"]["z"],
                ],
                dtype=np.float32,
            )
            refine_visibility_with_depth(
                visibility=visibility,
                keypoints_2d_normalized=mapped_keypoints_2d,
                keypoints_3d_world_raw=keypoints_3d_unreal_raw,
                camera_location_world_raw=cam_loc_unreal_raw,
                depth_image=depth_image,
                image_width=image_width,
                image_height=image_height,
                depth_max_cm=depth_max_cm,
                depth_tolerance_cm=depth_tolerance_cm,
                depth_neighborhood=depth_neighborhood,
            )

        keypoint_visibility_per_view.append(visibility)

        # Extract per-camera camera parameters
        cam_rot, cam_trans = parse_projection_components(cam_data)
        cx, cy, fx, fy = parse_camera_intrinsics(batch_data_file, cam_data)
        fov = cam_data["iterationData"]["camera"]["FOV"]

        # Mirror rotation matrix for PyTorch3D
        mirror_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mirrored_rot = mirror_matrix @ cam_rot.T @ mirror_matrix.T
        R = torch.tensor(mirrored_rot, dtype=torch.float32)
        T = torch.tensor([-cam_trans[0], cam_trans[1], cam_trans[2]], dtype=torch.float32)

        cam_rot_per_view.append(R)
        cam_trans_per_view.append(T)
        fx_per_view.append(fx)
        fy_per_view.append(fy)
        cx_per_view.append(cx)
        cy_per_view.append(cy)
        fov_per_view.append(fov)

        if verbose:
            print(f"  CAM{cam_id}: fx={fx:.2f}, fy={fy:.2f}, visible_joints={int(np.sum(visibility))}/{len(visibility)}")

    # ------------------------------------------------------------------
    # 3D ground-truth keypoints in raw world frame (PyTorch3D-mirrored,
    # same convention as cam_rot_per_view / cam_trans_per_view / model_loc).
    # Track which J_names actually have a dataset entry — joints not in
    # `pose_data` (model-only joints) must NOT participate in the canonical-
    # frame transform; they end up zeroed-out after all conversions to
    # match the SLEAP convention of (0,0,0) as the "no GT" sentinel.
    # Downstream consumers (viz, trainer) recognise (0,0,0) as missing.
    # ------------------------------------------------------------------
    kp3d_by_name = {}
    for k, kp in pose_data.items():
        p3 = kp.get("3DPos")
        if p3 is not None:
            kp3d_by_name[k] = np.array(
                [-p3["x"], p3["y"], p3["z"]], dtype=np.float32
            )
    keypoints_3d_world = np.zeros(
        (len(config.dd["J_names"]), 3), dtype=np.float32
    )
    keypoint_3d_in_dataset = np.zeros(len(config.dd["J_names"]), dtype=bool)
    for o, j in enumerate(config.dd["J_names"]):
        if j in kp3d_by_name:
            keypoints_3d_world[o] = kp3d_by_name[j]
            keypoint_3d_in_dataset[o] = True

    # ------------------------------------------------------------------
    # Canonical-camera-frame transformation.
    #
    # Pick the lowest CAM ID (camera_indices[0] after the sort above) as
    # the canonical camera. Define a new world frame in which that
    # camera's extrinsics are identity — i.e. the canonical frame IS
    # cam-0's view — and re-express everything relative to it.
    #
    # Forward transform world -> canonical:
    #     x_can = x_world @ R_0 + t_0
    # where (R_0, t_0) are cam-0's row-vector PyTorch3D-mirrored
    # extrinsics. Derivation (row-vector convention throughout):
    #     R_v'                  = R_0.T @ R_v
    #     t_v'                  = t_v - t_0 @ R_v'
    #     root_loc'             = root_loc @ R_0 + t_0
    #     R_root_can_col        = R_0.T @ R_root_col   (scipy col-vec form)
    #     keypoints_3d_can      = keypoints_3d_world @ R_0 + t_0
    #
    # Storage choice: y_output emits `canonical_to_world = (R_0, t_0)`
    # as the FORWARD transform — not its inverse. We picked forward
    # because (R_0, t_0) IS the artifact computed at the canonical site,
    # storing it avoids drift from re-deriving, and the canonical
    # frame's definition stays visible at the storage point. Consumers
    # that need world-frame coordinates apply the inverse explicitly:
    #     x_world = (x_canonical - t_0) @ R_0.T
    # ------------------------------------------------------------------
    if canonical_frame:
        canonical_cam_idx = 0  # lowest CAM ID, first in sorted camera_indices
        R_0 = cam_rot_per_view[canonical_cam_idx].clone()
        t_0 = cam_trans_per_view[canonical_cam_idx].clone()
        R_0_np = R_0.numpy()
        t_0_np = t_0.numpy()

        for v in range(len(camera_indices)):
            R_v = cam_rot_per_view[v]
            t_v = cam_trans_per_view[v]
            R_v_can = R_0.T @ R_v
            t_v_can = t_v - t_0 @ R_v_can
            cam_rot_per_view[v] = R_v_can
            cam_trans_per_view[v] = t_v_can

        model_loc = (model_loc @ R_0_np + t_0_np).astype(np.float32)

        R_root_col = Rotation.from_rotvec(global_rotation_np).as_matrix()
        R_root_can_col = R_0_np.T @ R_root_col
        global_rotation_np = (
            Rotation.from_matrix(R_root_can_col).as_rotvec().astype(np.float32)
        )

        keypoints_3d = (keypoints_3d_world @ R_0_np + t_0_np).astype(np.float32)

        canonical_to_world = (
            R_0_np.astype(np.float32),
            t_0_np.astype(np.float32),
        )
        canonical_cam_id = int(camera_indices[canonical_cam_idx])
    else:
        keypoints_3d = keypoints_3d_world.copy()
        canonical_to_world = (
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        canonical_cam_id = -1  # sentinel: no canonical frame applied

    # ------------------------------------------------------------------
    # Scale unification (Phase 1b). Apply `translation_factor` uniformly
    # to all world-frame translations and 3D coordinates. Default 0.1
    # makes mesh-native and data-frame units coincide so that downstream
    # consumers can use `mesh + trans` (no `(mesh - root) * 10 + trans`
    # hack). Rotations are scale-invariant and untouched. All canonical-
    # frame relationships are preserved because uniform scaling commutes
    # with the canonical-frame transform.
    # See "Scale Unification" in MULTIVIEW_REPLICANT_INTEGRATION_DESIGN.md.
    # ------------------------------------------------------------------
    s = float(translation_factor)
    if s != 1.0:
        model_loc = (model_loc * s).astype(np.float32)
        keypoints_3d = (keypoints_3d * s).astype(np.float32)
        keypoints_3d_world = (keypoints_3d_world * s).astype(np.float32)
        canonical_to_world = (
            canonical_to_world[0],
            (canonical_to_world[1] * s).astype(np.float32),
        )
        for v in range(len(camera_indices)):
            cam_trans_per_view[v] = cam_trans_per_view[v] * s

    # ------------------------------------------------------------------
    # Zero out 3D keypoints for joints with no dataset GT.
    # Reason: when `canonical_frame=True`, a zero-padded row in
    # `keypoints_3d_world` becomes `0 @ R_0 + t_0 = t_0` after the
    # canonical-frame transform, i.e. lands at the canonical camera's
    # position — far from the mesh cluster and a meaningless 3D target.
    # Apply the (0,0,0) sentinel AFTER all transformations so downstream
    # consumers (visualisers, trainers) can detect missing joints with the
    # standard `~np.all(kp3d == 0, axis=1)` check — the SLEAP convention.
    # `keypoints_3d_world` is also zeroed for the same joints so the inverse
    # `(x_can - t_0) @ R_0.T` round-trip still maps missing rows to zero.
    # ------------------------------------------------------------------
    if (~keypoint_3d_in_dataset).any():
        keypoints_3d[~keypoint_3d_in_dataset] = 0.0
        keypoints_3d_world[~keypoint_3d_in_dataset] = 0.0

    # Overwrite the earlier raw-frame assignments with (possibly)
    # canonical-frame values. pose_data is left untouched for
    # backwards compatibility with consumers that re-derive from it.
    y_output["root_loc"] = model_loc
    y_output["root_rot"] = global_rotation_np
    y_output["keypoints_3d"] = keypoints_3d
    y_output["keypoints_3d_world"] = keypoints_3d_world
    y_output["canonical_to_world"] = canonical_to_world
    y_output["canonical_cam_id"] = canonical_cam_id

    # Populate x_output
    x_output["image_data"] = image_data
    x_output["image_paths"] = image_paths
    x_output["input_image_mask"] = mask_data
    x_output["mask_paths"] = mask_paths
    x_output["depth_paths"] = depth_paths
    x_output["num_views"] = len(camera_indices)
    x_output["camera_ids"] = camera_indices

    # Populate y_output with per-view data
    y_output["keypoints_2d_per_view"] = keypoints_2d_per_view
    y_output["keypoint_visibility_per_view"] = keypoint_visibility_per_view
    y_output["keypoint_in_dataset_per_view"] = keypoint_in_dataset_per_view
    y_output["cam_rot_per_view"] = cam_rot_per_view
    y_output["cam_trans_per_view"] = cam_trans_per_view
    y_output["fx_per_view"] = fx_per_view
    y_output["fy_per_view"] = fy_per_view
    y_output["cx_per_view"] = cx_per_view
    y_output["cy_per_view"] = cy_per_view
    y_output["fov_per_view"] = fov_per_view

    if verbose:
        print(f"Successfully loaded frame {frame_index:05d} with {len(camera_indices)} cameras")

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
        "/media/fabi/Data/Mausb/Mausb_03.json"
    )

    # Load the SMIL data
    x,y = load_SMIL_Unreal_sample(json_file_path, 
                                  plot_tests=False, 
                                  propagate_scaling=True, 
                                  translation_factor=0.01)

    # Verify things plot correctly
    #plot_loaded_data_tests(x,y,device)

    # Render the SMAL model based on the loaded data
    Render_SMAL_Model_from_Unreal_data(x,y,device)