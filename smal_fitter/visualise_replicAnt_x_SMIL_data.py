import sys, os
import imageio
from typing_extensions import override

sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import config
import json
from smal_fitter import SMALFitter
import torch
from optimize_to_joints import ImageExporter


# Define a function to read the JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def euler_to_rodrigues(roll, pitch, yaw):
    # Convert to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))

    # Rodrigues vector
    theta = np.arccos((np.trace(R) - 1) / 2)
    v = np.array([(R[2, 1] - R[1, 2]),
                  (R[0, 2] - R[2, 0]),
                  (R[1, 0] - R[0, 1])]) / (2 * np.sin(theta))

    rodrigues_vector = theta * v

    return rodrigues_vector


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


if __name__ == '__main__':
    """
    STEP 1 - LOAD replicAnt generated SMIL data
    """
    # Read the JSON file
    json_file_path = "data/replicAnt_trials/TEST_ANGLES/TEST_ANGLES_09.json"
    input_image = json_file_path.split(".")[0] + ".JPG"
    data = read_json_file(json_file_path)

    # Extract shape and pose parameters
    shape_betas = data['iterationData']['subject Data'][0]["1"]["shape betas"]
    pose_data = data['iterationData']['subject Data'][0]["1"]['keypoints']

    joint_angles = []
    joint_names = []

    for key in pose_data:
        joint_names.append(key)
        joint_angles.append(euler_to_rodrigues(roll=pose_data[key]["eulerAngles"]["x"],
                                               pitch=pose_data[key]["eulerAngles"]["y"],
                                               yaw=pose_data[key]["eulerAngles"]["z"]))

    joint_angles = np.array(joint_angles)

    # Convert shape betas to a NumPy array
    shape_betas = np.array(shape_betas)

    # Display the extracted data
    print("Shape Betas:", shape_betas.shape)
    print("Pose Data:", joint_angles.shape)

    """
    STEP 2 - LOAD SMIL MODEL
    """

    data_json, filenames = return_placeholder_data(
        input_image=input_image,
        num_joints=len(joint_angles))  # in the shape of the default convention returned by the dataloaders

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

    model.betas = torch.nn.Parameter(torch.Tensor(shape_betas).to(device))
    model.joint_rotations = torch.nn.Parameter(torch.Tensor(joint_angles[1:]).reshape((1, 54, 3)).to(device))

    """
    STEP 4 - RENDER POSED MESH
    """

    print("\nINFO: Rendering output")
    image_exporter = ImageExporter("LOCAL_TEST", ["test_replicAnt_x_smil.jpg"])

    image_exporter.stage_id = 0
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter)
