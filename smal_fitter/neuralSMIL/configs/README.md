# SMILify Unified Configuration System

This package provides a single, type-safe configuration system for both **single-view** and **multi-view** SMIL image regressor training. It replaces the previous mix of `training_config.py`, in-script defaults, and CLI-only overrides with JSON config files and dataclasses.

## Precedence (highest to lowest)

1. **CLI arguments** (e.g. `--batch-size 4`, `--config my.json`)
2. **JSON config file** (when using `--config path/to/config.json`)
3. **Mode-specific defaults** (`SingleViewConfig` / `MultiViewConfig`)
4. **Base defaults** (`BaseTrainingConfig` in `base_config.py`)
5. **Legacy `config.py`** (only for SMAL model paths, `SHAPE_FAMILY`, `N_BETAS`, `N_POSE`, etc.—unchanged for fitter_3d compatibility)

## Quick start

### Single-view training with a JSON config

```bash
python train_smil_regressor.py --config configs/examples/singleview_baseline.json
```

### Multi-view training with a JSON config

```bash
python train_multiview_regressor.py --config configs/examples/multiview_6cam.json
```

### Multi-GPU training

```bash
python train_multiview_regressor.py --config configs/examples/multiview_mouse_UNET_long.json --num_gpus 2
```

### Override from CLI

```bash
python train_smil_regressor.py --config configs/examples/singleview_baseline.json --batch-size 8 --dataset my_data.h5
```

### Without a config file (legacy behavior)

If you omit `--config`, the scripts fall back to the previous behavior (e.g. `TrainingConfig` / `MultiViewTrainingConfig` and their defaults). No JSON file is required.

## JSON config requirements

- The JSON file **must** include a top-level `"mode"` field: `"singleview"` or `"multiview"`.
- Optional: include a top-level `"smal_model"` section to override values otherwise sourced from `config.py`:
  - `"smal_file"`: SMAL/SMIL model pickle path. When used, callers should reload `config` so `dd`, `N_POSE`, `N_BETAS`, etc. match that file.
  - `"shape_family"`: integer shape family passed into SMAL/SMIL code (overrides `config.SHAPE_FAMILY` for the run).
- Curriculum parameters (loss weights and learning rate schedules) use **string keys** for epoch thresholds in JSON (e.g. `"10": 3e-5`). These are converted to integer keys when loaded.
- See `examples/` for full example configs.

## Top-level JSON sections

### `smal_model` — SMAL/SMIL model overrides (optional)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `smal_file` | string or null | `null` | Path to the SMAL/SMIL model `.pkl` file. Overrides `config.SMAL_FILE` at runtime; derived globals (`dd`, `N_POSE`, `N_BETAS`, joints) are reloaded automatically. |
| `shape_family` | int or null | `null` | Shape family index passed into SMAL/SMIL code. |

### `dataset` — Data paths and splits

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_path` | string | *required* | Path to the HDF5 dataset file. Images and 2D keypoints must be undistorted (see note below). |
| `train_ratio` | float | `0.85` | Fraction of data used for training. |
| `val_ratio` | float | `0.05` | Fraction of data used for validation. |
| `test_ratio` | float | `0.1` | Fraction of data used for testing. |
| `dataset_fraction` | float | `0.5` | Fraction of training split sampled per epoch (a different random subset is drawn each epoch for diversity). |

**Undistortion:** All input images and 2D keypoints must be in undistorted (ideal pinhole) space. PyTorch3D's `FoVPerspectiveCameras` does not model lens distortion, so raw barrel-distorted images would cause a mismatch between rendered projections and ground truth keypoints. Undistort images with `cv2.undistort()` and use `reprojections.h5` (ideal pinhole projections of triangulated 3D points) as 2D keypoint ground truth.

### `model` — Network architecture

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backbone_name` | string | `"vit_large_patch16_224"` | Feature extraction backbone. See [Supported backbones](#supported-backbones). |
| `freeze_backbone` | bool | `true` | Freeze pretrained backbone weights (fine-tune only the head). |
| `hidden_dim` | int | `1024` | Decoder hidden dimension. Auto-adjusted per backbone if not set explicitly (see table below). |
| `head_type` | string | `"transformer_decoder"` | Prediction head type: `"mlp"` or `"transformer_decoder"`. |
| `input_resolution` | int or null | `null` | Input image size. `null` = auto-detect from backbone (224 for ViT, 512 for ResNet/UNet). |
| `use_unity_prior` | bool | `false` | Use legacy Unity quadruped body prior. |
| `rgb_only` | bool | `false` | Use only RGB channels (ignore alpha/mask channel if present). |

#### Transformer decoder options (used when `head_type = "transformer_decoder"`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transformer_depth` | int | `6` | Number of transformer decoder layers. |
| `transformer_heads` | int | `8` | Number of attention heads per layer. |
| `transformer_dim_head` | int | `64` | Dimension per attention head. |
| `transformer_mlp_dim` | int | `1024` | Feed-forward network hidden dimension. |
| `transformer_dropout` | float | `0.1` | Dropout rate in attention and FFN layers. |
| `transformer_ief_iters` | int | `3` | Number of iterative error feedback (IEF) refinement iterations. |
| `transformer_trans_scale_factor` | int | `1` | Scale factor applied to predicted `betas_trans`. |

#### Supported backbones

| Backbone name | Type | Feature dim | Default resolution | Spatial dim | Notes |
|---------------|------|-------------|-------------------|-------------|-------|
| `resnet50` | CNN | 2048 | 512 | - | Fully convolutional; resolution configurable. |
| `resnet101` | CNN | 2048 | 512 | - | Fully convolutional; resolution configurable. |
| `resnet152` | CNN | 2048 | 512 | - | Fully convolutional; resolution configurable. |
| `vit_base_patch16_224` | ViT | 768 | 224 | 196 patches | Fixed 224px input (patch size 16). |
| `vit_large_patch16_224` | ViT | 1024 | 224 | 196 patches | Fixed 224px input (patch size 16). |
| `unet_efficientnet_b0` | UNet | 320 | 512 | 64 | Encoder-decoder with skip connections. |
| `unet_efficientnet_b3` | UNet | 384 | 512 | 128 | Encoder-decoder with skip connections. |
| `unet_resnet34` | UNet | 512 | 512 | 128 | Encoder-decoder with skip connections. |
| `unet_mobilenet_v3` | UNet | 960 | 512 | 64 | Encoder-decoder with skip connections; lightweight. |

**Auto-adjusted `hidden_dim`** (when not set explicitly in JSON):

| Backbone family | hidden_dim |
|-----------------|-----------|
| `vit_base_*` | 768 |
| `vit_large_*` | 1024 |
| `resnet*` | 2048 |
| `unet_efficientnet_b0` | 512 |
| `unet_efficientnet_b3` | 512 |
| `unet_resnet34` | 512 |
| `unet_mobilenet_v3` | 256 |

**Choosing a backbone:**
- **ViT** backbones are small-resolution (224px) but produce rich global features with patch-level spatial tokens. Good for when input images can be downscaled.
- **ResNet** backbones are fully convolutional and work at arbitrary resolution. No decoder or spatial tokens — the head sees only the global average-pooled feature.
- **UNet** backbones pair an encoder (EfficientNet, ResNet, MobileNet) with a feature pyramid decoder, producing both a global feature vector and high-resolution spatial feature maps. Best for large input resolutions where spatial detail matters. More VRAM-intensive — consider `backbone_chunk_size` and `use_mixed_precision` for multi-view training with many cameras.

### `optimizer` — Optimizer and learning rate

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | float | `5e-5` | Base learning rate (epoch 0). |
| `weight_decay` | float | `1e-4` | AdamW weight decay. |
| `gradient_clip_norm` | float | `1.0` | Max gradient norm for clipping (0 = disabled). |
| `optimizer_type` | string | `"adamw"` | Optimizer algorithm. |
| `lr_schedule` | dict | *(see defaults)* | Epoch-to-LR mapping for learning rate curriculum. Keys are epoch thresholds (strings in JSON), values are learning rates. The highest applicable threshold wins. |

Example:
```json
"lr_schedule": {
  "0": 5e-5,
  "50": 3e-5,
  "100": 1e-5,
  "300": 2e-6
}
```

### `loss_curriculum` — Loss weights and progressive training

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_weights` | dict | *(see below)* | Initial loss weights applied from epoch 0. |
| `curriculum_stages` | dict | *(see defaults)* | Epoch-to-weight-update mapping. Each stage partially overrides `base_weights` from that epoch onward. |

#### Available loss terms

| Loss name | Description | Typical range |
|-----------|-------------|---------------|
| `global_rot` | Global rotation error | 0.0 |
| `joint_rot` | Per-joint rotation error | 0.001 |
| `betas` | Shape parameter error | 0.0005 |
| `trans` | Translation error | 0.0005 |
| `fov` | Field-of-view error | 0.001 |
| `cam_rot` | Camera rotation error (multi-view) | 0.01 |
| `cam_trans` | Camera translation error (multi-view) | 0.01 |
| `log_beta_scales` | Per-limb scale parameter error | 0.0005 |
| `betas_trans` | Per-limb translation parameter error | 0.0005 |
| `keypoint_2d` | 2D keypoint reprojection error | 0.1 |
| `keypoint_3d` | 3D keypoint error | 0.25 |
| `silhouette` | Silhouette overlap loss | 0.0 |
| `joint_angle_regularization` | Joint angle prior/regularization | 0.001 |
| `limb_scale_regularization` | Limb scale regularization | 0.01 |
| `limb_trans_regularization` | Limb translation regularization | 1.0 |
| `triangulation_consistency` | Cross-view triangulation consistency (multi-view). **Disabled by default — see note below.** | 0.0 |
| `ief_intermediate` | Intermediate IEF iteration loss | 0.0 |

> **Why `triangulation_consistency` is disabled.**
> This loss triangulates GT 2D keypoints with the predicted cameras (DLT) and compares the result to the network's own predicted 3D joints (detached). Gradients flow only into the camera heads.
>
> In practice this term is **redundant with `keypoint_2d`**: both encode the same geometric constraint (GT 2D, predicted cameras, and predicted 3D must be mutually consistent) — they differ only in functional form (DLT pseudo-inverse vs. forward projection). When one is satisfied, the other is too.
>
> It becomes **actively harmful when GT 2D labels are noisy** (e.g. from a 2D pose estimator). The 3D GT is typically cleaner because triangulation across many views averages out per-view noise. The `keypoint_3d` loss anchors the body model to that cleaner signal. But `triangulation_consistency` re-triangulates the noisy 2D with still-learning cameras and pushes the cameras to match the body model — injecting 2D noise directly into camera supervision.
>
> When **GT camera calibration is available** (`use_gt_camera_init: true`) with direct camera supervision (`cam_rot`, `cam_trans`, `fov` losses), the system is already geometrically consistent by construction — the 3D GT was produced by the same camera solve. Any consistency loss is a no-op at best.
>
> **Leave this at 0.0** unless all of the following hold: (1) you have clean 2D keypoints, (2) no GT camera calibration, and (3) no 3D keypoints. Even then, its value is marginal since it carries no information beyond `keypoint_2d`.

Curriculum stages example:
```json
"curriculum_stages": {
  "10": { "keypoint_2d": 0.1, "joint_angle_regularization": 1e-5 },
  "50": { "keypoint_3d": 2.0, "limb_scale_regularization": 0.001 },
  "200": { "fov": 1e-7, "cam_rot": 1e-8, "cam_trans": 1e-8 }
}
```

### `training` — Training hyperparameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int | `8` | Samples per GPU per step. |
| `num_epochs` | int | `1000` | Total training epochs. |
| `seed` | int | `1234` | Random seed for reproducibility. |
| `rotation_representation` | string | `"6d"` | Rotation parameterization: `"6d"` (continuous) or `"axis_angle"`. |
| `num_workers` | int | `8` | DataLoader worker processes. |
| `pin_memory` | bool | `true` | Pin DataLoader memory for faster GPU transfer. |
| `prefetch_factor` | int | `4` | Batches prefetched per worker. |
| `resume_checkpoint` | string or null | `null` | Path to checkpoint `.pth` file to resume from. |
| `reset_ief_token_embedding` | bool | `false` | Re-initialize IEF token embeddings when resuming (useful when changing `transformer_ief_iters`). |
| `use_gt_camera_init` | bool | `true` | Use ground-truth camera parameters as initialization base and predict deltas (multi-view). |
| `use_mixed_precision` | bool | `false` | Enable FP16 mixed precision training via `torch.cuda.amp`. Roughly halves activation memory and speeds up Tensor Core operations. Weight updates remain FP32 for numerical stability. |
| `backbone_chunk_size` | int or null | `null` | Maximum number of images processed through the backbone in a single forward pass. `null` = process all views at once. Set to a smaller value (e.g. `6`) to reduce peak VRAM when using many views with high-resolution inputs. Mathematically equivalent to unchunked — only affects memory, not results. |

### `scale_trans_beta` — Per-limb scale/translation handling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"entangled_with_betas"` | How per-limb scale/translation betas are handled: `"ignore"` (zero out), `"separate"` (predict independently), or `"entangled_with_betas"` (predict jointly with shape betas). |

### `mesh_scaling` — Global mesh scale

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allow_mesh_scaling` | bool | `true` | Allow the model to predict a global mesh scale factor. |
| `init_mesh_scale` | float | `1.0` | Initial mesh scale value. |
| `use_log_scale` | bool | `true` | Predict scale in log-space for numerical stability. |

### `joint_importance` — Per-joint loss weighting

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable per-joint importance weighting. |
| `important_joint_names` | list[string] | `[]` | Joint names to upweight (e.g. `["Nose", "paw_L_tip"]`). Must match joint names in the SMAL model. |
| `weight_multiplier` | float | `10.0` | Multiplier applied to important joints in 2D/3D keypoint losses. |

### `ignored_joint_locations` — Loss-level joint exclusion

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable joint exclusion from keypoint losses. |
| `ignored_joint_names` | list[string] | `[]` | Joint names excluded from 2D and 3D keypoint loss computation. Joints remain in the dataset but are not supervised. |

### `ignored_joints` — Data-level joint exclusion

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ignored_joint_names` | list[string] | `[]` | Joints removed entirely during data preprocessing. |
| `verbose` | bool | `true` | Log which joints are ignored. |

### `multi_dataset` — Multi-dataset training

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable multi-dataset training. |
| `datasets` | list[object] | `[]` | List of dataset entries (see below). |
| `validation_split_strategy` | string | `"per_dataset"` | Validation split strategy: `"per_dataset"` or `"combined"`. |

Each dataset entry:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `""` | Display name. |
| `path` | string | `""` | Path to HDF5 file. |
| `type` | string | `"optimized_hdf5"` | Dataset type: `"replicant"`, `"sleap"`, `"optimized_hdf5"`, or `"auto"`. |
| `weight` | float | `1.0` | Sampling weight relative to other datasets. |
| `enabled` | bool | `true` | Whether this dataset is active. |
| `available_labels` | dict | *(all true)* | Which label types are available (e.g. `"keypoint_3d": false` if no 3D annotations). |

### `output` — Checkpoints and visualization

| Field | Type | Default (single-view) | Default (multi-view) | Description |
|-------|------|-----------------------|----------------------|-------------|
| `checkpoint_dir` | string | `"checkpoints"` | `"multiview_checkpoints"` | Directory for saved model checkpoints. |
| `plots_dir` | string | `"plots"` | `"plots"` | Directory for training metric plots. |
| `visualizations_dir` | string | `"visualizations"` | `"multiview_visualizations"` | Directory for validation visualizations. |
| `train_visualizations_dir` | string | `"visualizations_train"` | - | Directory for training-set visualizations. |
| `singleview_visualizations_dir` | string | - | `"multiview_singleview_renders"` | Per-view rendered overlays (multi-view only). |
| `save_checkpoint_every` | int | `10` | `10` | Save a checkpoint every N epochs. |
| `generate_visualizations_every` | int | `10` | `10` | Generate visualizations every N epochs. |
| `plot_history_every` | int | `10` | - | Update training history plots every N epochs. |
| `num_visualization_samples` | int | `10` | `3` | Number of samples to visualize. |

## Multi-view only fields

These fields are set at the **top level** of the JSON (not inside a section) for multiview configs:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_views_to_use` | int or null | `null` | Maximum views per sample. `null` = use all available views in the dataset. |
| `min_views_per_sample` | int | `2` | Minimum views required per training sample. |
| `cross_attention_layers` | int | `2` | Number of cross-view attention layers for multi-view feature fusion. |
| `cross_attention_heads` | int | `8` | Number of attention heads in cross-view attention. |
| `cross_attention_dropout` | float | `0.1` | Dropout in cross-view attention layers. |

## Memory optimization for multi-view training

When training with many camera views (e.g. 18) at high resolution (e.g. 512px) with a UNet backbone, peak VRAM can exceed GPU capacity. Two options reduce memory usage without affecting training quality:

**`backbone_chunk_size`** (in `training`): Instead of pushing all `batch_size * num_views` images through the backbone at once, process them in chunks. For example, with `batch_size: 2` and 18 views, the backbone normally sees 36 images simultaneously. Setting `backbone_chunk_size: 6` processes 6 at a time, reducing peak backbone VRAM by ~6x.

**`use_mixed_precision`** (in `training`): Uses FP16 for forward pass activations and convolutions via `torch.cuda.amp`, roughly halving activation memory. Weight updates remain in FP32 for stability. Also accelerates training on GPUs with Tensor Cores (e.g. RTX 3090/4090, A100).

Example for a VRAM-constrained setup:
```json
"training": {
  "batch_size": 2,
  "use_mixed_precision": true,
  "backbone_chunk_size": 6
}
```

## Curriculum keys in JSON

JSON does not allow integer keys. Use **string** keys for epoch-based dicts; they are converted to integers on load:

- **`lr_schedule`**: `"0": 5e-5`, `"10": 3e-5`, ...
- **`curriculum_stages`**: `"1": { "keypoint_2d": 0.1 }`, `"25": { ... }`, ...

## Example configs

| File | Mode | Description |
|------|------|-------------|
| `singleview_baseline.json` | singleview | Full single-view config with ViT backbone, transformer decoder, and loss/LR curriculum. |
| `multiview_baseline.json` | multiview | Multi-view config with cross-attention and multi-view output directories. |
| `multiview_sticks.json` | multiview | Stick insect with UNet EfficientNet-B3 backbone, 512px. |
| `multiview_sticks_UNET.json` | multiview | Stick insect UNet variant. |
| `multiview_sticks_UNET_continue.json` | multiview | Continuation training from a checkpoint. |
| `multiview_mouse_UNET_long.json` | multiview | 18-camera mouse with UNet EfficientNet-B3, mixed precision, and backbone chunking for VRAM optimization. |

## Using the examples

```bash
# Single-view
python train_smil_regressor.py --config configs/examples/singleview_baseline.json

# Multi-view
python train_multiview_regressor.py --config configs/examples/multiview_baseline.json --dataset_path /path/to/multiview.h5

# Multi-view with mixed precision (CLI override)
python train_multiview_regressor.py --config configs/examples/multiview_sticks.json --use_mixed_precision
```

Copy an example to your project and edit; keep the `mode` field and the structure above so the loader and legacy bridge continue to work.

## Public API

| Symbol | Description |
|--------|-------------|
| `load_config(config_file=..., cli_overrides=...)` | Load and merge JSON + CLI into a `SingleViewConfig` or `MultiViewConfig`. |
| `load_from_json(path)` | Load raw dict from JSON (with epoch key conversion). |
| `save_config_json(config, path)` | Serialize a config dataclass to JSON. |
| `validate_json_mode(path)` | Check that the JSON has a valid `mode` field. |
| `SingleViewConfig` | Default config for single-view training. |
| `MultiViewConfig` | Default config for multi-view training (adds cross-attention and multi-view output paths). |
| `BaseTrainingConfig` | Shared base; use mode-specific classes in practice. |
| `ConfigurationError` | Raised for invalid or incompatible config. |

## Structure

- **`base_config.py`** — Shared dataclasses: `DatasetConfig`, `ModelConfig`, `OptimizerConfig`, `LossCurriculumConfig`, `ScaleTransBetaConfig`, `MeshScalingConfig`, `JointImportanceConfig`, `IgnoredJointLocationsConfig`, `IgnoredJointsConfig`, `MultiDatasetConfig`, `OutputConfig`, `TrainingHyperparameters`, `SmalModelConfig`, and `BaseTrainingConfig`.
- **`singleview_config.py`** — `SingleViewConfig` (minimal extension of base).
- **`multiview_config.py`** — `MultiViewConfig`, `MultiViewOutputConfig`, and multi-view-specific fields (cross-attention, output dirs).
- **`config_utils.py`** — JSON load/save, deep merge into dataclasses, mode validation.
- **`examples/`** — Example JSON configs.

## Backward compatibility

- **Legacy scripts**: `train_smil_regressor.py` and `train_multiview_regressor.py` convert the new config to the dict format expected by their existing `main()` via `to_legacy_dict()` and `to_multiview_legacy_dict()`. No change to the rest of the training loop is required.
- **`config.py`**: Unchanged; still used for SMAL file paths, shape family, joint counts, and GPU settings by `fitter_3d` and `optimize_to_joints`.
- **`training_config.py`**: Deprecated; new setups should use this package and JSON configs.
- **New fields** (`use_mixed_precision`, `backbone_chunk_size`) default to `false`/`null`, so existing configs and scripts work without changes.
