# SMIL Image Regressor

A neural network implementation for predicting SMIL (Statistical Model of Insect Locomotion) parameters from RGB images. This implementation extends the existing SMALFitter class and uses a frozen ResNet152 backbone as a feature extractor.

## Overview

The SMIL Image Regressor learns to predict the following SMIL parameters from input images:

- **Global rotation**: 3D rotation of the root joint
- **Joint rotations**: 3D rotations for all joints (excluding root)
- **Shape parameters (betas)**: Shape deformation parameters
- **Translation**: 3D translation of the model
- **Camera FOV**: Field of view parameter
- **Joint scales**: Per-joint scaling parameters (if available)
- **Joint translations**: Per-joint translation parameters (if available)

## Architecture

The network consists of:

1. **ResNet152 Backbone**: Pre-trained on ImageNet, frozen during training
2. **Feature Extraction**: 2048-dimensional feature vector from ResNet152
3. **Regression Head**: Two fully connected layers (2048 → 512 → 256)
4. **Parameter Output**: Final layer that outputs all SMIL parameters

## Files

- `smil_image_regressor.py`: Main network implementation
- `train_smil_regressor.py`: Training script
- `main.py`: Demonstration script
- `smil_datasets.py`: Dataset loading utilities
- `README.md`: This documentation

## Usage

### Basic Usage

```python
from smil_image_regressor import SMILImageRegressor
import torch

# Create placeholder data for initialization
placeholder_data = create_placeholder_data_batch(batch_size=1)

# Initialize the model
model = SMILImageRegressor(
    device='cuda',
    data_batch=placeholder_data,
    batch_size=1,
    shape_family=config.SHAPE_FAMILY,
    use_unity_prior=False,
    rgb_only=True,
    freeze_backbone=True,
    hidden_dim=512
)

# Predict from an image
import numpy as np
image_data = np.random.rand(512, 512, 3) * 255  # Example image
predicted_params = model.predict_from_image(image_data)
```

### Training

To train the model on your dataset:

```bash
python train_smil_regressor.py
```

The training script will:
1. Load the replicAnt SMIL dataset
2. Split it into train/validation/test sets
3. Train the model with Adam optimizer
4. Save the best model based on validation loss
5. Generate training plots

### Demonstration

To see the model in action:

```bash
python main.py
```

This will:
1. Initialize the model
2. Load a sample from the dataset
3. Predict SMIL parameters
4. Display the results

## Dataset Format

The model expects data in the format provided by `replicAntSMILDataset`:

- **Input (x)**: Dictionary containing:
  - `input_image`: Path to the image file
  - `input_image_data`: Image as numpy array (H, W, C)

- **Target (y)**: Dictionary containing:
  - `joint_angles`: Joint rotation angles
  - `shape_betas`: Shape parameters
  - `root_loc`: Root location
  - `root_rot`: Root rotation
  - `cam_fov`: Camera field of view
  - `scale_weights`: Joint scale weights (optional)
  - `trans_weights`: Joint translation weights (optional)

## Model Parameters

The model can be configured with the following parameters:

- `freeze_backbone`: Whether to freeze ResNet152 weights (default: True)
- `hidden_dim`: Hidden dimension for FC layers (default: 512)
- `batch_size`: Batch size for processing
- `shape_family`: SMIL shape family ID
- `use_unity_prior`: Whether to use unity prior

## Training Configuration

The training script uses the following default settings:

- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: MSE loss for all parameters
- **Batch Size**: 4
- **Epochs**: 50
- **Data Split**: 70% train, 15% validation, 15% test

## Output

The model outputs a dictionary with the following keys:

```python
{
    'global_rot': torch.Tensor,      # (batch_size, 3)
    'joint_rot': torch.Tensor,       # (batch_size, N_POSE, 3)
    'betas': torch.Tensor,           # (batch_size, N_BETAS)
    'trans': torch.Tensor,           # (batch_size, 3)
    'fov': torch.Tensor,             # (batch_size, 1)
    'log_beta_scales': torch.Tensor, # (batch_size, N_JOINTS, 3) [optional]
    'betas_trans': torch.Tensor,     # (batch_size, N_JOINTS, 3) [optional]
}
```

## Integration with SMALFitter

The model extends `SMALFitter` and inherits all its functionality:

- SMIL model loading and rendering
- Parameter optimization
- Visualization capabilities
- Loss computation

You can use the predicted parameters directly with the SMALFitter's rendering pipeline:

```python
# Set predicted parameters to the model
model.set_smil_parameters(predicted_params, batch_idx=0)

# Generate visualization
image_exporter = ImageExporter("output", ["sample"])
model.generate_visualization(image_exporter)
```

## Requirements

- PyTorch
- Torchvision
- NumPy
- OpenCV
- Matplotlib
- tqdm

## Notes

- The ResNet152 backbone is frozen by default to leverage pre-trained features
- Images are automatically resized to 512x512 pixels
- The model handles both standard SMAL and SMIL model configurations
- Joint scales and translations are only predicted if available in the model data







