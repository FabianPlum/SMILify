# SMIL Image Regressor

A neural network implementation for predicting SMIL (Statistical Model of Insect Locomotion) parameters from RGB images. This implementation extends the existing SMALFitter class and supports multiple backbone architectures including ResNet and Vision Transformer (ViT) models. The architecture and training paradigm choices are based on [AniMer](https://github.com/luoxue-star/AniMer).

## Overview

The SMIL Image Regressor learns to predict the following SMIL parameters from input images:

- **Global rotation**: 3D rotation of the root joint (axis-angle or 6D representation)
- **Joint rotations**: 3D rotations for all joints (excluding root)
- **Shape parameters (betas)**: Shape deformation parameters
- **Translation**: 3D translation of the model
- **Camera parameters**: FOV, rotation matrix, and translation
- **Joint scales**: Per-joint scaling parameters (if available)
- **Joint translations**: Per-joint translation parameters (if available)

## Key Features

- **Multiple Backbone Support**: ResNet50/101/152 and Vision Transformer (ViT) models
- **Flexible Regression Heads**: MLP or Transformer Decoder architectures
- **Advanced Training**: Loss curriculum, learning rate scheduling, 3D keypoint supervision
- **Optimized Data Pipeline**: HDF5-based dataset preprocessing for faster training
- **Ground Truth Testing**: Comprehensive validation against known parameters
- **Joint Filtering**: Configurable ignored joints to handle model misalignments

## Architecture Options

### Backbone Networks
1. **ResNet Models**: ResNet50/101/152 (512x512 input, 2048 features)
2. **Vision Transformers**: ViT Base/Large (224x224 input, 768/1024 features)

### Regression Heads
1. **MLP Head**: Two fully connected layers with dropout
2. **Transformer Decoder**: Advanced attention-based decoder with iterative refinement

### Loss Functions
- **Parameter Loss**: MSE for global rotation, joint rotations, shape, translation, camera
- **2D Keypoint Loss**: MSE between predicted and ground truth 2D keypoints
- **3D Keypoint Loss**: MSE between predicted and ground truth 3D joint positions  
- **Silhouette Loss**: Binary cross-entropy for rendered vs. ground truth silhouettes

## Files

### Core Implementation
- `smil_image_regressor.py`: Main network implementation with multiple backbone support
- `transformer_decoder.py`: Transformer-based regression head implementation
- `train_smil_regressor.py`: Advanced training script with curriculum learning
- `training_config.py`: Comprehensive training configuration and hyperparameters

### Dataset Management
- `smil_datasets.py`: Unified dataset interface for JSON and HDF5 formats
- `dataset_preprocessing.py`: HDF5 dataset preprocessing for optimized training
- `optimized_dataset.py`: High-performance HDF5 dataset loader
- `preprocess_dataset.py`: CLI script for dataset preprocessing

### Testing and Validation
- `test_smil_regressor_ground_truth.py`: Ground truth validation and 3D keypoint alignment testing
- `example_usage.py`: Example usage patterns and demonstrations

## Quick Start Guide

### 1. Dataset Preprocessing (Recommended)

First, preprocess your JSON dataset to optimized HDF5 format for faster training:

```bash
# Basic preprocessing
python preprocess_dataset.py input_dataset/ optimized_dataset.h5

# Advanced preprocessing with custom settings
python preprocess_dataset.py input_dataset/ optimized_dataset.h5 \
    --silhouette_threshold 0.15 \
    --backbone vit_large_patch16_224 \
    --min_visible_keypoints 10 \
    --num_workers 8
```

### 2. Training

Train the model using either preprocessed HDF5 or original JSON datasets:

```bash
# Train with optimized dataset (recommended - faster)
python train_smil_regressor.py --data_path optimized_dataset.h5 --batch-size 8

# Train with original JSON dataset
python train_smil_regressor.py --dataset simple --batch-size 8

# Advanced training with custom configuration
python train_smil_regressor.py \
    --data_path optimized_dataset.h5 \
    --batch-size 8 \
    --backbone vit_large_patch16_224 \
    --learning-rate 1e-6 \
    --num-epochs 100
```

### 3. Ground Truth Testing

Validate model accuracy against known ground truth parameters:

```bash
# Test with 3D keypoint alignment
python test_smil_regressor_ground_truth.py --test-3d-keypoints

# Test specific sample
python test_smil_regressor_ground_truth.py --sample-index 5 --tolerance-3d-keypoints 0.15
```

### 4. Basic Python Usage

```python
from smil_image_regressor import SMILImageRegressor
from smil_datasets import UnifiedSMILDataset
import torch

# Load dataset (automatically detects HDF5 vs JSON)
dataset = UnifiedSMILDataset.from_path("path/to/dataset", 
                                      rotation_representation='6d',
                                      backbone_name='vit_large_patch16_224')

# Get a sample
x_data, y_data = dataset[0]

# Initialize model with ViT backbone
model = SMILImageRegressor(
    device='cuda',
    data_batch=x_data,
    batch_size=1,
    shape_family=config.SHAPE_FAMILY,
    backbone_name='vit_large_patch16_224',
    head_type='transformer_decoder',
    rotation_representation='6d'
)

# Predict from image
predicted_params = model.predict_from_image(x_data['input_image_data'])
```

## Configuration

### Training Configuration (`training_config.py`)

The training system uses a comprehensive configuration system:

```python
# Dataset configuration
DATA_PATHS = {
    'simple': "/path/to/simple/dataset",
    'simple100k_local': "/path/to/large/dataset"
}

# Model configuration
MODEL_CONFIG = {
    'backbone_name': 'vit_large_patch16_224',  # or 'resnet152'
    'head_type': 'transformer_decoder',        # or 'mlp'
    'freeze_backbone': True,
    'rotation_representation': '6d'            # or 'axis_angle'
}

# Loss curriculum - weights change during training
LOSS_CURRICULUM = {
    'base_weights': {
        'global_rot': 0.001,
        'joint_rot': 0.001,
        'keypoint_2d': 0.0,    # Starts at 0, increases later
        'keypoint_3d': 0.0,    # Starts at 0, increases later
        'silhouette': 0.0      # Starts at 0, increases later
    },
    'curriculum_stages': [
        (10, {'keypoint_2d': 0.01, 'keypoint_3d': 0.02}),  # Stage 1
        (30, {'keypoint_2d': 0.02, 'keypoint_3d': 0.05})   # Stage 2
    ]
}

# Ignored joints configuration
IGNORED_JOINTS_CONFIG = {
    'ignored_joint_names': ['b_a_5'],  # Joints to ignore during training
    'verbose_ignored_joints': True
}
```

### Advanced Features

#### Dataset Optimization
- **HDF5 Preprocessing**: Convert JSON datasets to optimized HDF5 format
- **JPEG Compression**: Efficient image storage with configurable quality
- **Chunked Storage**: Optimized for batch loading during training
- **Automatic Filtering**: Remove low-quality samples based on silhouette coverage

#### Training Optimizations
- **Loss Curriculum**: Gradual introduction of different loss components
- **Learning Rate Scheduling**: Automatic reduction at key epochs
- **Mixed Precision**: 16-bit training for faster convergence
- **Visualization**: Training progress visualization and keypoint alignment plots

#### Joint Filtering System
Configure joints to ignore during training to handle model misalignments:

```python
IGNORED_JOINTS_CONFIG = {
    'ignored_joint_names': [
        'an_1_r', 'an_1_l',  # Antenna tips (often misaligned)
        'b_a_5'              # Specific problematic joint
    ]
}
```

## Dataset Formats

### Original JSON Format
The model supports the original replicAnt SMIL JSON format:

- **Input (x)**: Dictionary containing:
  - `input_image`: Path to the image file
  - `input_image_data`: Image as numpy array (H, W, C) in range [0, 1]
  - `input_image_mask`: Silhouette mask as numpy array (H, W)

- **Target (y)**: Dictionary containing:
  - `joint_angles`: Joint rotation angles (N_JOINTS, 3)
  - `shape_betas`: Shape parameters (N_BETAS,)
  - `root_loc`: Root location (3,)
  - `root_rot`: Root rotation (3,)
  - `cam_fov`: Camera field of view (scalar)
  - `cam_rot`: Camera rotation matrix (3, 3)
  - `cam_trans`: Camera translation (3,)
  - `keypoints_2d`: 2D keypoint coordinates (N_JOINTS, 2) normalized [0, 1]
  - `keypoints_3d`: 3D keypoint coordinates (N_JOINTS, 3)
  - `keypoint_visibility`: Keypoint visibility flags (N_JOINTS,)
  - `scale_weights`: Joint scale weights (optional)
  - `trans_weights`: Joint translation weights (optional)

### Optimized HDF5 Format
For faster training, datasets can be preprocessed to HDF5 format:

- **Benefits**: 
  - 10-12x faster data loading
  - JPEG compression for efficient storage
  - Pre-computed visibility and filtering
  - Chunked storage optimized for batch sizes

- **Structure**:
  ```
  dataset.h5
  ├── metadata/          # Dataset statistics and configuration
  ├── images/           # RGB images (JPEG) and silhouette masks
  ├── parameters/       # All SMIL parameters (pose, shape, camera)
  ├── keypoints/        # 2D/3D keypoints and visibility
  └── auxiliary/        # Original paths and statistics
  ```

## Model Architecture Parameters

### Backbone Configuration
- `backbone_name`: Network architecture
  - ResNet: `'resnet50'`, `'resnet101'`, `'resnet152'`
  - ViT: `'vit_base_patch16_224'`, `'vit_large_patch16_224'`
- `freeze_backbone`: Freeze backbone weights (default: True)
- `input_resolution`: Input image size (auto-selected based on backbone)

### Regression Head Configuration
- `head_type`: Regression head architecture
  - `'mlp'`: Simple MLP with dropout
  - `'transformer_decoder'`: Advanced transformer decoder with iterative refinement
- `hidden_dim`: Hidden dimension for regression head (auto-selected based on backbone)

### Training Parameters
- `rotation_representation`: Rotation parameterization
  - `'axis_angle'`: 3-parameter axis-angle representation
  - `'6d'`: 6D rotation representation (more stable)
- `batch_size`: Training batch size (4-16 depending on GPU memory)
- `learning_rate`: Base learning rate (1.25e-6, AniMer-style conservative)
- `num_epochs`: Training epochs (100+ recommended)

## Performance Benchmarks

### Dataset Loading Speed
- **JSON Dataset**: ~2 iterations/second
- **Optimized HDF5**: ~24 iterations/second (**12x faster**)

### Memory Usage (ViT Large + Transformer Decoder)
- **GPU Memory**: ~2-3GB for batch size 8
- **Training Speed**: ~24 it/s on RTX 4090 with optimized dataset

### Model Accuracy (Ground Truth Test)
- **Parameter Loss**: < 0.01 for ground truth parameters
- **2D Keypoint Error**: < 2 pixels on average
- **3D Keypoint Alignment**: Mean error varies by joint complexity

## Model Output

The model outputs a dictionary with SMIL parameters:

```python
{
    'global_rot': torch.Tensor,      # (batch_size, 3/6) - root rotation
    'joint_rot': torch.Tensor,       # (batch_size, N_POSE, 3/6) - joint rotations  
    'betas': torch.Tensor,           # (batch_size, N_BETAS) - shape parameters
    'trans': torch.Tensor,           # (batch_size, 3) - translation
    'fov': torch.Tensor,             # (batch_size, 1) - camera FOV
    'cam_rot': torch.Tensor,         # (batch_size, 3, 3) - camera rotation matrix
    'cam_trans': torch.Tensor,       # (batch_size, 3) - camera translation
    'log_beta_scales': torch.Tensor, # (batch_size, N_JOINTS, 3) [optional]
    'betas_trans': torch.Tensor,     # (batch_size, N_JOINTS, 3) [optional]
}
```

*Note: Rotation dimensions depend on `rotation_representation` setting (3 for axis-angle, 6 for 6D)*

## Integration with SMALFitter

The model extends `SMALFitter` and inherits all functionality:

```python
# Direct parameter setting and rendering
model.set_smil_parameters(predicted_params, batch_idx=0)

# Generate 3D mesh and 2D projections
vertices, faces = model.get_mesh_vertices_faces()
rendered_image = model.render_rgb_image()
rendered_silhouette = model.render_silhouette()

# Compute losses for training
loss_dict = model.compute_prediction_loss(predicted_params, target_params)
```

## Requirements

### Core Dependencies
- **PyTorch** (≥1.10) with CUDA support
- **PyTorch3D** for 3D rendering and transformations
- **Torchvision** for backbone networks
- **NumPy** for numerical operations
- **OpenCV** for image processing
- **H5py** for HDF5 dataset support

### Training & Visualization
- **tqdm** for progress bars
- **Matplotlib** for plotting and visualization
- **Pillow** for image I/O
- **scikit-learn** for data splitting


## Latest Improvements Over Initial Implementation (just a dev log)

- **2x faster training** with HDF5 dataset optimization, but more importanly much more stable training due to massively reduced I/O operations
- **Multiple backbone support** (ResNet + ViT architectures)
- **Advanced regression heads** with transformer decoder
- **Comprehensive loss curriculum** with 2D/3D keypoint supervision
- **Robust training pipeline** with automatic checkpointing and visualization
- **Ground truth validation** for accuracy verification
- **Joint filtering system** for handling model misalignments








