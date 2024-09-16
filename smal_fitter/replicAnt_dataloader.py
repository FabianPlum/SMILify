import sys
import os

# Add the base path of the repository to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import imageio.v2 as imageio
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from smal_fitter import SMALFitter
from optimize_to_joints import ImageExporter
import config


def load_json_file(json_file_path):
    """
    Load the replicAnt data from the JSON file.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def get_iteration_data(json_file_path):
    """
    Get the iteration data from the replicAnt JSON file.

    This function loads the JSON file, extracts relevant information including
    image, camera parameters, and subject data.

    Args:
        json_file_path (str): Path to the JSON file containing replicAnt information.

    Returns:
        tuple: A tuple containing:
            - image (np.ndarray): Image as a numpy array with shape [height, width, color channels].
            - camera_angles (dict): Camera rotation angles (pitch, yaw, roll).
            - camera_translation (dict): Camera location (x, y, z).
            - camera_intrinsics (dict): Camera intrinsic parameters (FOV).
            - subject_3d_locations (dict): 3D locations of subject keypoints.
            - subject_rotations (dict): Rotation angles of subject keypoints.
            - camera_view_matrix (np.ndarray): 4x4 camera view matrix.
            - global_subject_rotation (dict): Global rotation of the subject (roll, pitch, yaw).

    Raises:
        FileNotFoundError: If the JSON file or image file is not found.
        KeyError: If expected keys are not found in the input data.
        json.JSONDecodeError: If the JSON file is invalid.
    """
    try:
        # Load JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Construct image path
        image_path = os.path.abspath(json_file_path.split(".")[0] + ".JPG")

        # Load image
        try:
            image = imageio.imread(image_path)
        except FileNotFoundError:
            print(f"Warning: Image file not found: {image_path}")
            image = None

        # Extract camera data
        camera_data = data['iterationData']['camera']
        camera_angles = camera_data['Rotation']
        camera_translation = camera_data['Location']
        camera_intrinsics = {'FOV': camera_data['FOV']}
        
        # Handle View Matrix
        if 'View Matrix' in camera_data:
            view_matrix = camera_data['View Matrix']
            camera_view_matrix = np.array([
                [view_matrix['xPlane']['x'], view_matrix['yPlane']['x'], view_matrix['zPlane']['x'], view_matrix['wPlane']['x']],
                [view_matrix['xPlane']['y'], view_matrix['yPlane']['y'], view_matrix['zPlane']['y'], view_matrix['wPlane']['y']],
                [view_matrix['xPlane']['z'], view_matrix['yPlane']['z'], view_matrix['zPlane']['z'], view_matrix['wPlane']['z']],
                [view_matrix['xPlane']['w'], view_matrix['yPlane']['w'], view_matrix['zPlane']['w'], view_matrix['wPlane']['w']]
            ])
        else:
            print("Warning: View Matrix not found in camera data. Using identity matrix.")
            camera_view_matrix = np.eye(4)

        # Extract subject data
        subject_data = data['iterationData']['subject Data'][0]['1']
        subject_3d_locations = {}
        subject_rotations = {}
        subject_global_rotations = {}

        for key, value in subject_data['keypoints'].items():
            subject_3d_locations[key] = value['3DPos']
            subject_rotations[key] = value['eulerAngles']

        return (image, camera_angles, camera_translation, 
                camera_intrinsics, subject_3d_locations, subject_rotations, 
                camera_view_matrix)

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except KeyError as e:
        raise KeyError(f"Missing expected key in input data: {e}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Invalid JSON file: {json_file_path}", "", 0)
    

def visualize_3d_scene(camera_translation, camera_intrinsics, subject_3d_locations, global_subject_rotation, camera_view_matrix):
    """
    Visualize the 3D scene with subject keypoints, root bone coordinate system, and camera.

    Args:
        camera_translation (dict): Camera location (x, y, z).
        camera_intrinsics (dict): Camera intrinsic parameters (FOV).
        subject_3d_locations (dict): 3D locations of subject keypoints.
        global_subject_rotation (dict): Global rotation of the subject (pitch, yaw, roll).
        camera_view_matrix (numpy.ndarray): 4x4 camera view matrix.

    Returns:
        None. Displays a 3D plot.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot subject keypoints
    colors = plt.cm.jet(np.linspace(0, 1, len(subject_3d_locations)))
    for i, ((key, location), color) in enumerate(zip(subject_3d_locations.items(), colors)):
        ax.scatter(location['x'], location['y'], location['z'], c=[color], label=key)

    # Plot root bone coordinate system
    root_location = subject_3d_locations['b_t']
    root_rotation = subject_rotations['b_t']
    plot_coordinate_system(ax, [root_location['x'], root_location['y'], root_location['z']], 
                           root_rotation, scale=5)

    # Plot camera
    plot_camera(ax, camera_translation, camera_intrinsics['FOV'], camera_view_matrix)

    # Set equal aspect ratio
    ax.set_box_aspect((1, 1, 1))

    # Set equal axis limits
    max_range = np.max(np.ptp([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()], axis=1)) / 2.0
    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title('3D Scene Visualization')
    plt.tight_layout()
    plt.show()

def plot_coordinate_system(ax, origin, rotation, scale=1.0):
    """
    Plot a coordinate system at the given origin with the specified rotation.

    Args:
        ax (Axes3D): The 3D axes to plot on.
        origin (list): The origin of the coordinate system [x, y, z].
        rotation (dict): The rotation angles {'pitch': pitch, 'yaw': yaw, 'roll': roll} in degrees.
        scale (float): The scale of the coordinate system axes.

    Returns:
        None. Plots the coordinate system on the given axes.
    """

    R = rotation_matrix([rotation['x'], rotation['y'], rotation['z']])

    for i, (color, label) in enumerate(zip(['r', 'g', 'b'], ['X', 'Y', 'Z'])):
        axis = np.array([0, 0, 0])
        # invert the Y axis
        if label == 'Y':
            axis[i] = -scale
        else:
            axis[i] = scale
        rotated_axis = np.dot(R, axis)
        ax.quiver(origin[0], origin[1], origin[2],
                  rotated_axis[0]*scale, rotated_axis[1]*scale, rotated_axis[2]*scale,
                  color=color, label=f'{label} axis')

def plot_camera(ax, location, fov, view_matrix):
    """
    Plot the camera and its field of view using the camera view matrix.

    Args:
        ax (Axes3D): The 3D axes to plot on.
        location (dict): Camera location {x, y, z}.
        rotation (dict): Camera rotation angles {pitch, yaw, roll} in degrees.
        fov (float): Field of view in degrees.
        view_matrix (numpy.ndarray): 4x4 camera view matrix.

    Returns:
        None. Plots the camera on the given axes.
    """
    cam_loc = np.array([location['x'], location['y'], location['z']])
    ax.scatter(*cam_loc, c='k', s=100, label='Camera')

    # Extract rotation matrix from view matrix
    R = view_matrix[:3, :3].T  # Transpose because view matrix is inverse of camera matrix

    # Camera looks along positive Z in Unreal Engine convention
    camera_dir = R[:, 2]

    # Plot camera direction
    ax.quiver(*cam_loc, *camera_dir, color='k', length=50, arrow_length_ratio=0.1)

    # Create camera frustum
    fov_rad = np.radians(fov)
    aspect_ratio = 1.0  # Assuming square image for simplicity
    near = 10
    far = 200
    
    frustum_height_near = 2 * np.tan(fov_rad / 2) * near
    frustum_width_near = frustum_height_near * aspect_ratio
    frustum_height_far = 2 * np.tan(fov_rad / 2) * far
    frustum_width_far = frustum_height_far * aspect_ratio
    
    frustum_corners = np.array([
        [frustum_width_near/2, frustum_height_near/2, near],
        [-frustum_width_near/2, frustum_height_near/2, near],
        [-frustum_width_near/2, -frustum_height_near/2, near],
        [frustum_width_near/2, -frustum_height_near/2, near],
        [frustum_width_far/2, frustum_height_far/2, far],
        [-frustum_width_far/2, frustum_height_far/2, far],
        [-frustum_width_far/2, -frustum_height_far/2, far],
        [frustum_width_far/2, -frustum_height_far/2, far]
    ])
    
    frustum_corners = np.dot(frustum_corners, R.T) + cam_loc

    frustum_faces = [
        [frustum_corners[0], frustum_corners[1], frustum_corners[2], frustum_corners[3]],
        [frustum_corners[4], frustum_corners[5], frustum_corners[6], frustum_corners[7]],
        [frustum_corners[0], frustum_corners[1], frustum_corners[5], frustum_corners[4]],
        [frustum_corners[2], frustum_corners[3], frustum_corners[7], frustum_corners[6]],
        [frustum_corners[1], frustum_corners[2], frustum_corners[6], frustum_corners[5]],
        [frustum_corners[0], frustum_corners[3], frustum_corners[7], frustum_corners[4]]
    ]

    frustum = Poly3DCollection(frustum_faces, alpha=0.2, facecolor='cyan', edgecolor='b')
    ax.add_collection3d(frustum)

def rotation_matrix(angles):
    """
    Compute the rotation matrix from Euler angles.

    Args:
        angles (list): Rotation angles [roll, pitch, yaw] in degrees.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    roll, pitch, yaw = np.radians(angles)
    Rx = rotation_matrix_x(np.pi/2 - roll) # temporary fix for replicAnt data
    Ry = rotation_matrix_y(pitch)
    Rz = rotation_matrix_z(yaw)
    return np.dot(Rz, np.dot(Ry, Rx))

def rotation_matrix_x(angle):
    """Rotation matrix around X axis."""
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])

def rotation_matrix_y(angle):
    """Rotation matrix around Y axis."""
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])

def rotation_matrix_z(angle):
    """Rotation matrix around Z axis."""
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def unreal_to_pytorch3d_coords(coords):
    """Convert Unreal Engine coordinates to PyTorch3D coordinates."""
    return np.array([-coords['y'], coords['z'], coords['x']])

def return_placeholder_data(input_image=None, num_joints=54):
    image_size = (512, 512)
    # pass a placeholder to the SMALFitter class as we are not actually going to provide any normal input data
    if input_image is not None:
        # Load image
        try:
            img_data = imageio.v2.imread(os.path.join(input_image))
        except FileNotFoundError:
            print(f"Warning: Image file not found: {input_image}")
            img_data = None
        except AttributeError:
            try:
                img_data = imageio.imread(os.path.join(input_image))
            except FileNotFoundError:
                print(f"Warning: Image file not found: {input_image}")
                img_data = None

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

def unreal_euler_to_pytorch3d_rodrigues(x, y, z):
    """
    Convert Unreal Engine Euler angles to PyTorch3D Rodrigues vector.
    
    Args:
        x (float): Rotation around X-axis in degrees.
        y (float): Rotation around Y-axis in degrees.
        z (float): Rotation around Z-axis in degrees.
    
    Returns:
        numpy.ndarray: Rodrigues vector in PyTorch3D coordinate system.
    """
    # Convert degrees to radians
    x, y, z = np.radians([x, y, z])
    
    # Create rotation matrix
    Rx = rotation_matrix_x(x)
    Ry = rotation_matrix_y(y)
    Rz = rotation_matrix_z(z)
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # Convert to Rodrigues vector
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if np.isclose(theta, 0):
        return np.zeros(3)
    else:
        rodrigues = (1 / (2 * np.sin(theta))) * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) * theta
    
    # Convert to PyTorch3D coordinate system
    return np.array([-rodrigues[1], rodrigues[2], rodrigues[0]])

def convert_camera_parameters(camera_translation, camera_view_matrix):
    """
    Convert camera parameters from Unreal Engine to PyTorch3D convention.

    Args:
        camera_translation (dict): Camera location in Unreal Engine coordinates.
        camera_view_matrix (np.ndarray): 4x4 camera view matrix from Unreal Engine.

    Returns:
        tuple: (camera_location, camera_rotation) in PyTorch3D convention.
    """
    camera_location = unreal_to_pytorch3d_coords(camera_translation)
    
    # Convert UE5 view matrix to OpenGL/PyTorch3D convention
    opengl_from_ue = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    
    # Invert the view matrix as it's typically done for camera transformations
    inverted_view_matrix = np.linalg.inv(camera_view_matrix[:3, :3])
    
    opengl_view_matrix = opengl_from_ue @ inverted_view_matrix @ opengl_from_ue.T
    
    # The rotation matrix is now directly usable (no need for transpose)
    camera_rotation = opengl_view_matrix

    return camera_location, camera_rotation

if __name__ == '__main__':
    """
    STEP 1 - LOAD replicAnt generated SMIL data
    """
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")
    
    # Construct the path to the JSON file
    json_file_path = os.path.abspath(os.path.join(current_dir, "..", "data", "replicAnt_trials", "TEST_ANGLES", "TEST_ANGLES_00.json"))
    
    # Check if the file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}")
    else:
        try:
            # Try to open and read the file
            with open(json_file_path, 'r') as file:
                data = json.load(file)
                print("JSON data loaded successfully")
            
            # Get the iteration data
            image, camera_angles, camera_translation, camera_intrinsics, subject_3d_locations, subject_rotations, camera_view_matrix = get_iteration_data(json_file_path)

            # Check if image is None before visualization
            if image is None:
                print("Warning: Image not loaded. Proceeding with visualization without image.")

            # Visualize the 3D scene
            visualize_3d_scene(camera_translation, camera_intrinsics, subject_3d_locations, subject_rotations, camera_view_matrix)


        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            if isinstance(e, FileNotFoundError):
                print(f"File permissions: {oct(os.stat(json_file_path).st_mode)[-3:]}")

    """
    STEP 2 - LOAD SMIL MODEL AND RENDER OUTPUT
    """
    print("\nINFO: Loading SMIL model and rendering output...")

    # Initialize SMIL model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_path = json_file_path.replace('.json', '.JPG')
    data_json, filenames = return_placeholder_data(input_image=image_path, num_joints=54)
    model = SMALFitter(device, data_json, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior=False)

    # Apply shape betas and joint rotations
    shape_betas = data['iterationData']['subject Data'][0]["1"]["shape betas"]
    model.betas = torch.nn.Parameter(torch.tensor(shape_betas, device=device).float().unsqueeze(0))

    # Handle global rotation separately
    global_rotation = unreal_euler_to_pytorch3d_rodrigues(subject_rotations['b_t']['x'], subject_rotations['b_t']['y'], subject_rotations['b_t']['z'])
    model.global_rotation = torch.nn.Parameter(torch.tensor(global_rotation, device=device).float().unsqueeze(0))

    # TODO get and apply subject angles, akin to visualise_replicAnt_x_SMIL_data.py
    
    # Set camera parameters
    camera_data = data['iterationData']['camera']
    fov = torch.tensor([camera_data['FOV']], device=device)
    model.fov = torch.nn.Parameter(fov)
    model.renderer.cameras.fov = fov

    """
    # TODO: Fix camera parameter setting as currently the model goes out of view when this is applied
    
    # Convert camera parameters
    camera_location, camera_rotation = convert_camera_parameters(camera_translation, camera_view_matrix)

    # Convert to torch tensor and add batch dimension
    R = torch.from_numpy(camera_rotation).to(device).float().unsqueeze(0)
    T = torch.from_numpy(camera_location).to(device).float().unsqueeze(0)

    model.renderer.cameras.R = R
    model.renderer.cameras.T = T

    # Set model translation (root position)
    root_position = unreal_to_pytorch3d_coords(subject_3d_locations['b_t'])
    model.trans = torch.nn.Parameter(torch.tensor(root_position, device=device).float().unsqueeze(0))
    """

    # Render output
    print("\nINFO: Rendering output")
    
    # Create a new output directory at the base of the repo
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "output", "REPLICANT_TEST")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the basename of the input file for the output filename
    output_filename = os.path.basename(image_path)
    
    image_exporter = ImageExporter(output_dir, [output_filename])
    image_exporter.stage_id = 0
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter)

    print(f"Rendering complete. Check the output directory: {output_dir}")

