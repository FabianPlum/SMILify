import torch
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# Add the parent directories to the path to import modules
# not very pretty, but it works.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Unreal2Pytorch3D import load_SMIL_Unreal_sample, Render_SMAL_Model_from_Unreal_data
from utils import eul_to_axis

# Import rotation utilities from PyTorch3D
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d


# ----------------------- ROTATION CONVERSION UTILS ----------------------- #

def axis_angle_to_rotation_6d(axis_angle):
    """
    Converts axis-angle representation to 6D rotation representation.
    Args:
        axis_angle: Tensor of shape (..., 3) or numpy array
    Returns:
        Tensor of shape (..., 6) or numpy array
    """
    if isinstance(axis_angle, np.ndarray):
        axis_angle_tensor = torch.from_numpy(axis_angle)
        is_numpy = True
    else:
        axis_angle_tensor = axis_angle
        is_numpy = False
    
    rotation_matrix = axis_angle_to_matrix(axis_angle_tensor)
    rotation_6d = matrix_to_rotation_6d(rotation_matrix)
    
    if is_numpy:
        return rotation_6d.numpy()
    return rotation_6d

class replicAntSMILDataset(torch.utils.data.Dataset):    
    def __init__(self, data_path, use_ue_scaling=True, rotation_representation='axis_angle'):
        """
        Initialize replicAnt SMIL Dataset.
        
        Args:
            data_path: Path to the dataset directory
            use_ue_scaling: Whether this dataset expects UE scaling (default True for replicAnt data)
            rotation_representation: '6d' or 'axis_angle' for joint rotations (default: 'axis_angle')
        """
        self.data_path = data_path
        self.use_ue_scaling = use_ue_scaling
        self.rotation_representation = rotation_representation
        self.data_json_paths = []
        for file in os.listdir(self.data_path):
            if file.endswith('.json') and not file.startswith('_BatchData'):
                self.data_json_paths.append(os.path.join(self.data_path, file))

        # sort the data json paths so when iterating over the dataset, the order is consistent
        self.data_json_paths.sort()

    def __getitem__(self, idx):
        x, y = load_SMIL_Unreal_sample(self.data_json_paths[idx], 
                                       plot_tests=False, 
                                       propagate_scaling=True, 
                                       translation_factor=0.01)

        # x contains the input image path and the input image data
        # y contains the processed SMIL data

        # Convert rotation representations if needed
        if self.rotation_representation == '6d':
            # Convert root rotation (global rotation) from axis-angle to 6D
            if 'root_rot' in y:
                y['root_rot'] = axis_angle_to_rotation_6d(y['root_rot'])
            
            # Convert joint angles from axis-angle to 6D
            if 'joint_angles' in y:
                y['joint_angles'] = axis_angle_to_rotation_6d(y['joint_angles'])

        return x, y

    def __len__(self):
        return len(self.data_json_paths)
    
    def get_ue_scaling_flag(self):
        """
        Get the UE scaling flag for this dataset.
        
        Returns:
            bool: Whether this dataset expects UE scaling
        """
        return self.use_ue_scaling


if __name__ == "__main__":
    # provide path to a replicAnt SMIL dataset
    data_path = "data/replicAnt_trials/replicAnt-x-SMIL-TEX"
    synthDataset = replicAntSMILDataset(data_path)
    print("Number of samples in the dataset: ", len(synthDataset))

    # access a sample from the dataset
    # structure: Dataset [sample_idx] -> [0] for x or [1] for y -> ["key"]
    data, labels = synthDataset[0]
    print("First sample: ", data["input_image"])
    print("First sample camera rotation: ", labels["cam_rot"])

    # Render the SMAL model based on the loaded data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Render_SMAL_Model_from_Unreal_data(data, labels, device)

    # example dataloaders for training, validation, and testing
    # lets split the dataset into three parts (train 70%, test 15%, validation 15%)
    test_size = 0.15
    val_size = 0.15
    BATCH_SIZE = 4

    test_amount, val_amount = int(synthDataset.__len__() * test_size), int(synthDataset.__len__() * val_size)

    # this function will automatically randomly split your dataset but you could also implement the split yourself
    train_set, val_set, test_set = torch.utils.data.random_split(synthDataset, [
                (synthDataset.__len__() - (test_amount + val_amount)), 
                test_amount, 
                val_amount
    ])


    train_dataloader = torch.utils.data.DataLoader(
                train_set,
                batch_size=BATCH_SIZE,
                shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=BATCH_SIZE,
                shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=BATCH_SIZE,
                shuffle=True,
    )
    
    # print the number of samples in each dataset
    print("\nNumber of samples in train set: ", len(train_set))
    print("Number of samples in val set: ", len(val_set))
    print("Number of samples in test set: ", len(test_set))

    # print the number of batches in each dataloader
    print("\nNumber of batches in train dataloader: ", len(train_dataloader))
    print("Number of batches in val dataloader: ", len(val_dataloader))
    print("Number of batches in test dataloader: ", len(test_dataloader))