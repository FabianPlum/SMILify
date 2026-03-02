# SMIL Image Regressor

A neural network framework for predicting SMIL (Statistical Model of Insect Locomotion) parameters from RGB images. Supports both single-view and multi-view reconstruction, with multiple backbone architectures and regression heads. The architecture and training paradigm are based on [AniMer](https://github.com/luoxue-star/AniMer).

## Overview

The regressors learn to predict the following SMIL parameters from input images:

- **Global rotation**: 3D rotation of the root joint (axis-angle or 6D representation)
- **Joint rotations**: 3D rotations for all joints (excluding root)
- **Shape parameters (betas)**: Shape deformation parameters
- **Translation**: 3D translation of the model
- **Camera parameters**: FOV, rotation matrix, and translation
- **Joint scales / translations**: Per-joint scaling and translation parameters (if enabled)

## Key Features

- **Multiple Backbone Support**: ResNet50/101/152 and Vision Transformer (ViT) models
- **Flexible Regression Heads**: MLP or Transformer Decoder with Iterative Error Feedback (IEF)
- **Multi-View Training**: Cross-attention fusion across synchronized camera views with per-view camera heads
- **Triangulation Consistency Loss**: Differentiable triangulation of GT 2D keypoints through predicted cameras for geometric self-supervision
- **Distributed Training**: `torchrun` / SLURM / `mp.spawn` multi-GPU training via DDP
- **JSON Config System**: Reproducible, version-controlled experiments via dataclass-backed JSON configs
- **Advanced Training**: Loss curriculum, learning rate scheduling, 3D keypoint supervision
- **Optimized Data Pipeline**: HDF5-based dataset preprocessing for faster training
- **Joint Filtering**: Configurable ignored joints and per-joint importance weighting

## Architecture

### Backbone Networks
1. **ResNet Models**: ResNet50/101/152 (512×512 input, 2048-dim features)
2. **Vision Transformers**: ViT Base/Large (224×224 input, 768/1024-dim features + 196 patch tokens)

### Regression Heads
1. **MLP Head**: Two fully connected layers with dropout
2. **Transformer Decoder**: Cross-attends over spatial patch tokens (single-view) or per-view fused features (multi-view), with Iterative Error Feedback (IEF) iterations — current predictions are encoded into the query token so the decoder conditions on its own previous output (HMR/SPIN/AniMer style)

### Multi-View Architecture
The `MultiViewSMILImageRegressor` adds:
- Per-view ViT backbone (shared weights) to extract 196 patch tokens per view
- Cross-attention layers to fuse per-view features into view-aware representations `(B, V, 1024)`
- Transformer decoder cross-attending over all per-view features (not a single pooled token)
- Separate camera prediction head per canonical view slot
- Shared body parameter prediction head

### Loss Functions
- **Parameter Loss**: MSE for global rotation, joint rotations, shape, translation, camera
- **2D Keypoint Loss**: MSE between projected and ground truth 2D keypoints (per-view, visibility-weighted)
- **3D Keypoint Loss**: MSE between predicted and ground truth 3D joint positions
- **Silhouette Loss**: Binary cross-entropy for rendered vs. ground truth silhouettes
- **Triangulation Consistency Loss**: Differentiable DLT triangulation of GT 2D keypoints through predicted cameras, compared to body model 3D predictions — gradients flow into camera heads
- **Regularization**: Joint angle, limb scale, and limb translation regularizers with curriculum weighting

## Files

### Core Implementation
- `smil_image_regressor.py`: Single-view network (ResNet/ViT backbones, MLP/Transformer head)
- `multiview_smil_regressor.py`: Multi-view network with cross-view attention and per-view camera heads
- `transformer_decoder.py`: Transformer-based regression head with IEF
- `train_smil_regressor.py`: Single-view training script
- `train_multiview_regressor.py`: Multi-view training script (DDP-capable)
- `training_config.py`: **Deprecated** — use `configs/` and JSON configs instead

### Configuration System
- `configs/`: Unified dataclass-based config system (see `configs/README.md`)
  - `base_config.py`, `singleview_config.py`, `multiview_config.py`: Config dataclasses
  - `config_utils.py`: JSON load/save, deep merge, validation
  - `examples/`: Example JSON configs for single-view and multi-view

### Dataset Management
- `smil_datasets.py`: Unified dataset interface for JSON and HDF5 formats
- `sleap_data/`: SLEAP multi-view HDF5 dataset loader and collation
- `dataset_preprocessing.py`: HDF5 dataset preprocessing CLI
- `optimized_dataset.py`: High-performance HDF5 dataset loader

### Testing and Validation
- `test_smil_regressor_ground_truth.py`: Ground truth validation and 3D keypoint alignment
- `run_multiview_inference.py`: Multi-view inference on a trained checkpoint
- `benchmark_model.py`: Benchmarking script for multi-view models

## Quick Start

### 1. Dataset Preprocessing (Recommended)

```bash
# Basic preprocessing
python dataset_preprocessing.py input_dataset/ optimized_dataset.h5

# With custom SMAL model (required when not using default)
python dataset_preprocessing.py input_dataset/ output.h5 \
    --smal-file "3D_model_prep/SMILy_Mouse.pkl"
```

### 2. Single-View Training

```bash
# With JSON config (recommended)
python train_smil_regressor.py --config configs/examples/singleview_baseline.json

# CLI overrides on top of JSON config
python train_smil_regressor.py \
    --config configs/examples/singleview_baseline.json \
    --batch-size 4 \
    --num-epochs 500

# Legacy (no config file — uses training_config.py defaults)
python train_smil_regressor.py --data_path optimized_dataset.h5 --batch-size 8
```

### 3. Multi-View Training

```bash
# With JSON config (recommended)
python train_multiview_regressor.py --config configs/examples/multiview_sticks.json

# JSON config with CLI dataset override
python train_multiview_regressor.py \
    --config configs/examples/multiview_sticks.json \
    --dataset_path /path/to/other_dataset.h5

# Distributed training (single node, 4 GPUs)
torchrun --nproc_per_node=4 train_multiview_regressor.py \
    --config configs/examples/multiview_sticks.json

# SLURM / multi-node: set RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
# before running — torchrun or the SLURM launcher sets these automatically
```

### 4. Multi-View Inference

```bash
python run_multiview_inference.py --checkpoint multiview_checkpoints/best_model.pth
```

### 5. Ground Truth Testing

```bash
python test_smil_regressor_ground_truth.py --test-3d-keypoints

# With custom SMAL model
python test_smil_regressor_ground_truth.py \
    --smal-file "3D_model_prep/SMILy_Mouse.pkl" \
    --test-3d-keypoints
```

## Configuration System

Configuration is managed through JSON files backed by Python dataclasses. See `configs/README.md` for full documentation.

**Precedence (highest to lowest)**:
1. CLI arguments
2. JSON config file (`--config path/to/config.json`)
3. Mode-specific defaults (`SingleViewConfig` / `MultiViewConfig`)
4. Base defaults (`BaseTrainingConfig`)
5. Legacy `config.py` (SMAL paths, `SHAPE_FAMILY`, `N_BETAS`, `N_POSE` — unchanged)

### JSON Config Structure

Every JSON config must include `"mode": "singleview"` or `"mode": "multiview"`. Example (multi-view):

```json
{
  "mode": "multiview",

  "num_views_to_use": null,
  "min_views_per_sample": 2,
  "cross_attention_layers": 2,
  "cross_attention_heads": 8,

  "smal_model": {
    "smal_file": "3D_model_prep/SMILy_STICK.pkl",
    "shape_family": null
  },

  "dataset": {
    "data_path": "3D_Sticks_full.h5",
    "train_ratio": 0.85,
    "val_ratio": 0.05,
    "test_ratio": 0.1,
    "dataset_fraction": 0.1
  },

  "model": {
    "backbone_name": "vit_large_patch16_224",
    "freeze_backbone": true,
    "head_type": "transformer_decoder",
    "transformer_depth": 6,
    "transformer_heads": 8,
    "transformer_ief_iters": 3
  },

  "optimizer": {
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "lr_schedule": {
      "0": 5e-5,
      "10": 3e-5,
      "60": 1e-5,
      "150": 2e-6
    }
  },

  "loss_curriculum": {
    "base_weights": {
      "keypoint_2d": 0.1,
      "keypoint_3d": 0.25,
      "triangulation_consistency": 0.0
    },
    "curriculum_stages": {
      "10": {"triangulation_consistency": 0.1},
      "30": {"keypoint_3d": 1.0, "triangulation_consistency": 0.001},
      "50": {"keypoint_3d": 2.0, "triangulation_consistency": 0.01}
    }
  },

  "training": {
    "batch_size": 16,
    "num_epochs": 300,
    "seed": 42,
    "rotation_representation": "6d"
  },

  "output": {
    "checkpoint_dir": "multiview_checkpoints",
    "save_checkpoint_every": 10
  }
}
```

Curriculum epoch keys are strings in JSON and automatically converted to integers on load.

### SMAL Model Override

To use a non-default SMAL/SMIL model file, specify `"smal_model"` in your JSON config or pass `--smal-file` on the CLI:

```bash
python train_multiview_regressor.py \
    --config configs/examples/multiview_sticks.json \
    --smal-file "3D_model_prep/SMILy_Mouse.pkl"
```

This reloads `config.py` globals (`dd`, `N_POSE`, `N_BETAS`) to match the specified model before any dataset or network construction.

## Distributed Training

`train_multiview_regressor.py` supports three launch modes:

| Mode | How | Notes |
|------|-----|-------|
| **Single GPU** | `python train_multiview_regressor.py ...` | Default; `--num_gpus 1` |
| **Single-node multi-GPU** | `torchrun --nproc_per_node=N ...` or `--num_gpus N` | `mp.spawn` fallback if torchrun not detected |
| **Multi-node** | `torchrun` / SLURM with `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` env vars | IPv4-only TCP store to avoid IPv6 issues on HPC |

A DDP barrier is inserted after each epoch's checkpointing and visualization to prevent rank desynchronization.

## Dataset Formats

### SLEAP Multi-View HDF5 Format

Multi-view training uses SLEAP-exported HDF5 files containing synchronized frames from multiple cameras. The `SLEAPMultiViewDataset` loader handles:
- Per-view RGB images and silhouette masks
- Per-view 2D keypoints with visibility flags
- GT camera parameters (rotation, translation, FOV) per view
- 3D keypoint ground truth

### Optimized Single-View HDF5 Format

```
dataset.h5
├── metadata/       # Dataset statistics and configuration
├── images/         # RGB images (JPEG) and silhouette masks
├── parameters/     # All SMIL parameters (pose, shape, camera)
├── keypoints/      # 2D/3D keypoints and visibility
└── auxiliary/      # Original paths and statistics
```

Benefits: 10–12× faster data loading, JPEG compression, chunked storage optimized for batch sizes.

## Model Output

```python
{
    'global_rot': torch.Tensor,       # (B, 3/6)         root rotation
    'joint_rot': torch.Tensor,        # (B, N_POSE, 3/6) joint rotations
    'betas': torch.Tensor,            # (B, N_BETAS)      shape parameters
    'trans': torch.Tensor,            # (B, 3)            translation
    'fov': torch.Tensor,              # (B, 1)            camera FOV
    'cam_rot': torch.Tensor,          # (B, 3, 3)         camera rotation
    'cam_trans': torch.Tensor,        # (B, 3)            camera translation
    'log_beta_scales': torch.Tensor,  # (B, N_JOINTS, 3)  [optional]
    'betas_trans': torch.Tensor,      # (B, N_JOINTS, 3)  [optional]
}
```

Rotation dimensions depend on `rotation_representation`: 3 for axis-angle, 6 for 6D.

## Checkpoints

Checkpoints include a `config` key with all parameters needed for inference:

```python
checkpoint = torch.load('best_model.pth')
# checkpoint['config'] contains:
#   model_config, rotation_representation, scale_trans_mode/config,
#   shape_family, smal_file
```

`run_multiview_inference.py` and `run_inference.py` prefer `checkpoint['config']` and fall back to `training_config.py` defaults for older checkpoints.

## Known Architectural Issues and Status

| # | Issue | Status |
|---|-------|--------|
| 1 | IEF loop had no actual feedback (identical input each iteration) | **Fixed** — current predictions encoded into query token |
| 2 | Multiview decoder cross-attended to a single pooled token (wasted capacity) | **Fixed** — decoder now cross-attends over per-view features `(B, V, 1024)` |
| 3 | Mean-pool bottleneck before body prediction | **Resolved** by fix #2 |
| 4 | NaN clamping broke gradient flow | **Fixed** — replaced with `torch.nan_to_num` |
| 5 | No multi-view geometric consistency for camera heads | **Fixed** — differentiable triangulation consistency loss implemented |
| 9 | `init_pose = zeros` invalid for 6D rotation representation | **Fixed** — initialised to 6D identity `[1,0,0,1,0,0]` per joint |

## Requirements

### Core Dependencies
- **PyTorch** (≥2.0) with CUDA support
- **PyTorch3D** for 3D rendering and transformations
- **Torchvision** for backbone networks
- **timm** for ViT backbones
- **NumPy** / **OpenCV** / **H5py**

### Training & Visualization
- `tqdm`, `matplotlib`, `Pillow`, `scikit-learn`, `imageio`

Install: `pip install -r requirements.txt`
