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


def return_placeholder_data(input_image=None, num_joints=55, pose_data=None):
    """
    Create placeholder data for SMALFitter initialization from Unreal Engine pose data.
    
    Prepares input data in the format expected by SMALFitter class. Can load an actual
    image or create placeholder tensors. If pose_data is provided, extracts 2D joint
    positions and sets visibility flags for visualization.
    
    Args:
        input_image (str, optional): Path to input image file. If None, creates placeholder
                                    tensors with default size (512, 512). Default is None.
        num_joints (int, optional): Number of joints in the pose data. Default is 55.
        pose_data (dict, optional): Dictionary containing 2D joint positions from Unreal data.
                                   Expected format: {joint_name: {"2DPos": {"x": float, "y": float}}}
                                   If None, creates zero tensors for joints and visibility.
                                   Default is None.
    
    Returns:
        tuple: ((rgb, sil, joints, visibility), filenames)
            - rgb (torch.Tensor): RGB image tensor of shape (1, 3, H, W) with values in [0, 1]
            - sil (torch.Tensor): Silhouette tensor of shape (1, 1, H, W) filled with zeros
            - joints (torch.Tensor): Joint positions tensor of shape (1, num_joints, 2)
            - visibility (torch.Tensor): Joint visibility tensor of shape (1, num_joints)
            - filenames (list): List containing the input image filename or ["PLACEHOLDER"]
    
    Note:
        When pose_data is provided, joint positions are extracted and visibility is set to 1
        for all joints. The x-coordinate is negated to account for coordinate system differences.
        Setting visibility to 1 means the joints of the posed model will be displayed.
    
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
    sil = torch.zeros((1, 1, image_size[0], image_size[1]))
    if pose_data is None:
        joints = torch.zeros((1, num_joints, 2))
        visibility = torch.zeros((1, num_joints))
    else:
        # NOTE: This does not actually display the ground truth points directtly.
        # Setting the visibility of the joints to 1 simply means, that the joints of the posed model are displayed.
        display_points_2D = [
            [-pose_data[key]["2DPos"]["x"], pose_data[key]["2DPos"]["y"]]
            for key in pose_data.keys()
        ]
        joints = torch.tensor(
            np.array(display_points_2D).reshape(1, num_joints, 2), dtype=torch.float32
        )
        visibility = torch.ones((1, num_joints))

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

        print(key, rodrigues_angle)
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

    for i, name in enumerate(display_points_names):
        print(name, display_points_3D[i])

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


if __name__ == "__main__":
    """
    STEP 1 - LOAD replicAnt generated SMIL data
    """
    # Read the JSON file
    json_file_path = (
        "data/replicAnt_trials/replicAnt-x-SMIL-demo/replicAnt-x-SMIL-demo_00.json"
    )
    # Generate additional plots for debugging
    plot_tests = True

    # get the batch data file path
    batch_data_file_path = json_file_path.replace(
        json_file_path.split("/")[-1],
        "_BatchData_" + json_file_path.split("/")[-2] + ".json",
    )

    # get the input image path
    input_image = json_file_path.split(".")[0] + ".JPG"

    # load the json data
    with open(json_file_path, "r") as file:
        data = json.load(file)

    with open(batch_data_file_path, "r") as file:
        batch_data_file = json.load(file)

    # Extract pose data contained in "keypoints"
    pose_data = data["iterationData"]["subject Data"][0]["1"]["keypoints"]
    
    # get camera data into correct format: 
    # extrinsics: rotation and translation
    cam_rot, cam_trans = parse_projection_components(data)

    print("INFO: Camera rotation", cam_rot)
    print("INFO: Camera translation", cam_trans)

    # intrinsics: image centre, focal length, and field of view
    cx, cy, fx, fy = parse_camera_intrinsics(
        batch_data_file=batch_data_file, iteration_data_file=data
    )

    print("INFO: Image centre x", cx)
    print("INFO: Image centre y", cy)
    print("INFO: Focal length x", fx)
    print("INFO: Focal length y", fy)

    cam_fov = [data["iterationData"]["camera"]["FOV"]]

    print("INFO: Camera FOV", cam_fov)

    """
    STEP 2 - Verify parsed data is correct
    """

    if plot_tests:
        # plot 3D points
        plot_3D_points(pose_data=pose_data, input_image=input_image)

        # check, the 3D points align with their corresponding projected 2D points
        plot_3D_projected_points(
            pose_data=pose_data,
            input_image=input_image,
            cam_rot=cam_rot,
            cam_trans=cam_trans,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    """
    STEP 3 - Set up basic pytorch3d scene with 3D points as spheres
    """

    # set the device to use (first available GPU by default)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract shape and pose parameters
    shape_betas = data["iterationData"]["subject Data"][0]["1"]["shape betas"]
    pose_data = data["iterationData"]["subject Data"][0]["1"]["keypoints"]

    joint_angles, joint_names = get_joint_angles_from_pose_data(pose_data)

    # map joints, in case the order differs. The root bone is expected to be the first entry
    np_joint_angles_mapped = map_joint_order(
        config.dd["J_names"], joint_names, joint_angles
    )

    # Convert shape betas to a NumPy array
    if len(shape_betas) == 0:
        shape_betas = np.zeros(20)
    else:
        shape_betas = np.array(shape_betas)

    # Display the extracted data
    print("Shape Betas:", shape_betas.shape)
    print("Pose Data:", np_joint_angles_mapped.shape)

    # setting pose data to None supresses displaing joint locations in the rendered image
    data_json, filenames = return_placeholder_data(
        input_image=input_image, num_joints=len(np_joint_angles_mapped), pose_data=pose_data
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

    # CURRENTLY SHAPE BETAS ARE DISABLED IN UNREAL / set to zero
    # model parameters
    model.betas = torch.nn.Parameter(torch.Tensor(shape_betas).to(device))

    # set model joint rotations
    model.joint_rotations = torch.nn.Parameter(
        torch.Tensor(np_joint_angles_mapped[1:])
        .reshape((1, np_joint_angles_mapped[1:].shape[0], 3))
        .to(device)
    )

    # Convert to PyTorch tensors and extract R, T components
    # Mirror the rotation matrix for x-axis
    mirror_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Apply mirroring to rotation matrix
    mirrored_rot = mirror_matrix @ cam_rot.T @ mirror_matrix.T
    R = torch.tensor(mirrored_rot, dtype=torch.float32)

    # Mirror the translation vector for x-axis
    T = torch.tensor([-cam_trans[0], cam_trans[1], cam_trans[2]], dtype=torch.float32)

    R = R.clone().detach().to(device).unsqueeze(0)
    T = T.clone().detach().to(device).unsqueeze(0)

    cameras = FoVPerspectiveCameras(
        device=device, R=R, T=T, fov=cam_fov, degrees=True, znear=0.01, zfar=5000
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

    # get the model location
    model_loc = np.array(
        [
            -pose_data["b_t"]["3DPos"]["x"],
            pose_data["b_t"]["3DPos"]["y"],
            pose_data["b_t"]["3DPos"]["z"],
        ],
        dtype=np.float32,
    )


    if plot_tests:

        # Create sphere meshes for keypoints - no need to mirror here as the function will do it
        keypoints_3d = np.array(
            [
                [
                    pose_data[key]["3DPos"]["x"],
                    pose_data[key]["3DPos"]["y"],
                    pose_data[key]["3DPos"]["z"],
                ]
                for key in pose_data.keys()
            ]
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
        display_img = cv2.imread(input_image)
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

    """
    STEP 4 - Set up pytorch3d scene with the SMIL model
    """

    model.fov = torch.nn.Parameter(torch.Tensor(cam_fov).to(device))
    model.renderer.cameras.fov = torch.nn.Parameter(torch.Tensor(cam_fov).to(device))
    model.renderer.cameras.R = torch.nn.Parameter(torch.Tensor(R).to(device))
    model.renderer.cameras.T = torch.nn.Parameter(torch.Tensor(T).to(device))

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

    global_rotation = torch.nn.Parameter(
        torch.from_numpy(global_rotation_np).float().to(device).unsqueeze(0)
    )

    model.global_rotation = global_rotation
    model.trans = torch.nn.Parameter(torch.Tensor(np.array([model_loc])).to(device))

    """
    STEP 5 - RENDER POSED MESH
    """

    print("\nINFO: Rendering output")
    image_exporter = ImageExporter("LOCAL_TEST", [os.path.basename(input_image)])
    image_exporter.stage_id = 0
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter, apply_UE_transform=True)
