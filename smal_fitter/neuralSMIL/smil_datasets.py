import torch
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues in multiprocessing
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
    def __init__(self, data_path, use_ue_scaling=True, rotation_representation='axis_angle', backbone_name='resnet152'):
        """
        Initialize replicAnt SMIL Dataset.
        
        Args:
            data_path: Path to the dataset directory
            use_ue_scaling: Whether this dataset expects UE scaling (default True for replicAnt data)
            rotation_representation: '6d' or 'axis_angle' for joint rotations (default: 'axis_angle')
            backbone_name: Backbone name to determine keypoint scaling (default: 'resnet152')
        """
        self.data_path = data_path
        self.use_ue_scaling = use_ue_scaling
        self.rotation_representation = rotation_representation
        self.backbone_name = backbone_name
        self.data_json_paths = []
        for file in os.listdir(self.data_path):
            if file.endswith('.json') and not file.startswith('_BatchData'):
                self.data_json_paths.append(os.path.join(self.data_path, file))

        # sort the data json paths so when iterating over the dataset, the order is consistent
        self.data_json_paths.sort()
        
        # Detect input resolution from batch data file
        self.original_resolution = self._detect_input_resolution()
        
        # Determine target resolution based on backbone
        if backbone_name.startswith('vit'):
            self.target_resolution = 224  # ViT expects 224x224
        else:
            self.target_resolution = self.original_resolution  # ResNet can handle original resolution

    def _detect_input_resolution(self):
        """Detect the input resolution from the batch data file."""
        import json
        import os
        
        # Look for batch data file
        batch_files = [f for f in os.listdir(self.data_path) if f.startswith('_BatchData') and f.endswith('.json')]
        
        if batch_files:
            batch_file_path = os.path.join(self.data_path, batch_files[0])
            try:
                with open(batch_file_path, 'r') as f:
                    batch_data = json.load(f)
                
                if 'Image Resolution' in batch_data:
                    resolution = batch_data['Image Resolution']
                    # Return the resolution (assuming square images, take x or y)
                    return resolution.get('x', resolution.get('y', 512))
            except Exception as e:
                print(f"Warning: Could not load batch data file {batch_file_path}: {e}")
        
        # Fallback to default resolution
        print("Warning: Could not detect input resolution, using default 512")
        return 512

    def get_input_resolution(self):
        """Get the detected input resolution."""
        return self.original_resolution
    
    def get_target_resolution(self):
        """Get the target resolution based on backbone."""
        return self.target_resolution
    
    def get_ue_scaling_flag(self):
        """Get the UE scaling flag."""
        return self.use_ue_scaling

    def __getitem__(self, idx):
        # Use optimized loading with reduced I/O operations
        x, y = load_SMIL_Unreal_sample(
            self.data_json_paths[idx], 
            plot_tests=False, 
            propagate_scaling=True, 
            translation_factor=0.01,
            load_image=True,  # Ensure image is loaded
            verbose=False  # Reduce output
        )

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