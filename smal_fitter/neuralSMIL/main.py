import torch
import os
import sys
import numpy as np

# Add the parent directories to the path to import modules
# not very pretty, but it works.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smil_datasets import replicAntSMILDataset
from smil_image_regressor import SMILImageRegressor
import config


def create_placeholder_data_batch(batch_size=1, image_size=512):
    """
    Create placeholder data batch for SMALFitter initialization.
    
    Args:
        batch_size: Batch size
        image_size: Image size (assumed square)
        
    Returns:
        RGB tensor for rgb_only mode
    """
    # For rgb_only=True, we only need the RGB tensor
    rgb = torch.zeros((batch_size, 3, image_size, image_size))
    return rgb


def demonstrate_smil_regressor():
    """
    Demonstrate the SMIL Image Regressor network.
    """
    print("=== SMIL Image Regressor Demonstration ===")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create placeholder data for SMALFitter initialization
    placeholder_data = create_placeholder_data_batch(batch_size=1)
    
    # Initialize the SMIL Image Regressor
    print("Initializing SMIL Image Regressor...")
    model = SMILImageRegressor(
        device=device,
        data_batch=placeholder_data,
        batch_size=1,
        shape_family=config.SHAPE_FAMILY,
        use_unity_prior=False,
        rgb_only=True,
        freeze_backbone=True,
        hidden_dim=512
    ).to(device)
    
    print(f"Model initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with a sample from the dataset
    print("\nTesting with sample data...")
    data_path = "data/replicAnt_trials/replicAnt-x-SMIL-TEX"
    dataset = replicAntSMILDataset(data_path)
    
    if len(dataset) > 0:
        # Get a sample
        x_data, y_data = dataset[0]
        
        if x_data['input_image_data'] is not None:
            print(f"Sample image shape: {x_data['input_image_data'].shape}")
            
            # Predict SMIL parameters
            model.eval()
            with torch.no_grad():
                predicted_params = model.predict_from_image(x_data['input_image_data'])
            
            print("\nPredicted SMIL parameters:")
            for key, value in predicted_params.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} - {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # Show some parameter values
            print(f"\nSample parameter values:")
            print(f"  Global rotation: {predicted_params['global_rot'][0].cpu().numpy()}")
            print(f"  Shape betas (first 5): {predicted_params['betas'][0, :5].cpu().numpy()}")
            print(f"  Translation: {predicted_params['trans'][0].cpu().numpy()}")
            print(f"  Camera FOV: {predicted_params['fov'][0].cpu().numpy()}")
            
        else:
            print("No image data available in sample")
    else:
        print("Dataset is empty")
    
    print("\n=== Demonstration completed ===")


if __name__ == "__main__":
    # set the device to use (first available GPU by default)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Demonstrate the SMIL Image Regressor
    demonstrate_smil_regressor()
    
    print("\n" + "="*50)
    print("To train the model, run:")
    print("python train_smil_regressor.py")
    print("="*50)
    
    # Original dataset loading code (for reference)
    print("\n--- Original Dataset Loading (for reference) ---")
    
    # provide path to a replicAnt SMIL dataset
    data_path = "data/replicAnt_trials/replicAnt-x-SMIL-TEX"
    synthDataset = replicAntSMILDataset(data_path)
    print("Number of samples in the dataset: ", len(synthDataset))

    # split the dataset into three parts (train 70%, test 15%, validation 15%)
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