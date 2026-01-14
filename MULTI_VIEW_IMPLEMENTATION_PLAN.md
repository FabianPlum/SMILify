# Multi-View SLEAP Dataset Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding multi-view support to the SMILify pipeline. The goal is to leverage multiple synchronized camera views to predict a single set of SMIL body parameters (pose and shape) while predicting separate camera parameters for each view. This approach is theoretically sound because all cameras observe the same animal at the same instant, so a single set of body parameters should minimize the 2D keypoint loss across all perspectives when camera parameters are correctly estimated.

**Critical Constraint**: All changes must be additive and backward-compatible with the existing single-view workflow.

**Recent Expansion (GT 3D + Calibrated Cameras)**: The multi-view pipeline now optionally ingests **ground truth 3D keypoints** and **per-camera calibration** (intrinsics + extrinsics) from SLEAP/Anipose exports. This enables:
- Direct **3D keypoint supervision** (`keypoint_3d` loss)
- Direct **camera supervision** (`fov`, `cam_rot`, `cam_trans` losses)
- Robust camera conversion to PyTorch3D including **aspect ratio** handling
- A dataset-level `world_scale` to keep 3D coordinates + camera translations consistent with SMILify world units

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Pipeline Changes](#2-data-pipeline-changes)
3. [Model Architecture Changes](#3-model-architecture-changes)
4. [Loss Computation Changes](#4-loss-computation-changes)
5. [Training Pipeline Changes](#5-training-pipeline-changes)
6. [Permutation Equivariance](#6-permutation-equivariance)
7. [File Structure](#7-file-structure)
8. [Implementation Phases](#8-implementation-phases)
9. [Testing Strategy](#9-testing-strategy)
10. [Risk Assessment](#10-risk-assessment)
11. [Design Decisions (Resolved)](#11-design-decisions-resolved)

---

## 1. Architecture Overview

### Current Architecture (Single-View)

```
Input Image (1, 3, H, W)
        │
        ▼
   ┌─────────────┐
   │  Backbone   │  (ResNet/ViT)
   │  (frozen)   │
   └─────────────┘
        │
        ▼
   Feature Vector (1, feature_dim)
        │
        ▼
   ┌─────────────┐
   │ Predictor   │  (MLP or Transformer Decoder)
   │    Head     │
   └─────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Predicted Parameters:              │
   │  - global_rot (1, 6)                │
   │  - joint_rot (1, N_POSE, 6)         │
   │  - betas (1, N_BETAS)               │
   │  - trans (1, 3)                     │
   │  - fov (1, 1)                       │
   │  - cam_rot (1, 3, 3)                │
   │  - cam_trans (1, 3)                 │
   │  - mesh_scale (optional, 1, 1)      │
   └─────────────────────────────────────┘
```

### Proposed Multi-View Architecture

**Key Principle**: A single set of body parameters (pose, shape) is predicted from the combined information of ALL input views. The model then predicts separate camera parameters for each view. During loss computation, the unified body + each view's camera produces rendered 2D keypoints that are compared against that view's ground truth.

```
Input Images (K available views, 3, H, W)
        │
        │   ┌────────────────────────────────────────────────────┐
        │   │  If K > num_views_to_use: randomly sample views    │
        │   │  If K < num_views_to_use: use all K with masking   │
        │   └────────────────────────────────────────────────────┘
        │
        ▼ (process each selected view through SHARED backbone)
   ┌─────────────────────────────────────────────────────────────┐
   │                     Backbone (frozen)                        │
   │    View 0 ──► features_0                                     │
   │    View 1 ──► features_1     (shared weights for all views)  │
   │    View 2 ──► features_2                                     │
   └─────────────────────────────────────────────────────────────┘
        │
        ▼
   Feature Vectors (N_active_views, feature_dim)
        │
        ▼ (concatenate along feature dimension)
   ┌─────────────────────────────────────────────────────────┐
   │   Feature Fusion: Cross-Attention between views         │
   │   Output: (1, N_active_views, feature_dim)              │
   │                                                         │
   │   Cross-attention allows views to share information     │
   │   NOTE: For missing views, features are masked          │
   └─────────────────────────────────────────────────────────┘
        │
        ├────────────────────────────────────────────────────────┐
        │                                                        │
        ▼                                                        ▼
   ┌────────────────────┐                          ┌─────────────────────────┐
   │     Body Head      │                          │     Camera Head(s)      │
   │                    │                          │     (one per view)      │
   │  Predicts SINGLE   │                          │                         │
   │  set of body       │                          │  Predicts SEPARATE      │
   │  params from ALL   │                          │  camera params for      │
   │  concatenated      │                          │  each active view       │
   │  view features     │                          │                         │
   └─────────┬──────────┘                          └────────────┬────────────┘
             │                                                  │
             ▼                                                  ▼
   ┌─────────────────────┐                   ┌───────────────────────────────────┐
   │  UNIFIED Body Params│                   │  Per-View Camera Params           │
   │  (from ALL views)   │                   │                                   │
   │  - global_rot       │                   │  View 0: fov_0, cam_rot_0, cam_t_0│
   │  - joint_rot        │                   │  View 1: fov_1, cam_rot_1, cam_t_1│
   │  - betas            │                   │  View 2: fov_2, cam_rot_2, cam_t_2│
   │  - trans            │                   │  ...                              │
   │  - scale/trans PCA  │                   │                                   │
   └─────────────────────┘                   └───────────────────────────────────┘
```

**Loss Computation Flow** (per batch sample):

```
For each active view v:
    1. Take UNIFIED body params (same for all views)
    2. Take camera params for view v  
    3. Render SMIL mesh → project to 2D keypoints using camera v
    4. Compare rendered 2D keypoints vs GT keypoints for view v
    5. Weight loss by per-keypoint visibility (more visible = higher weight)
    
Total 2D loss = weighted average of per-view losses
```

**Additional Supervision (Optional, when available)**:
- **3D keypoint loss**: compares predicted canonical joints (3D) to GT 3D keypoints (world/model space after `world_scale`)
- **Camera parameter losses**: supervises each camera head with GT:
  - `fov` (vertical FOV in degrees)
  - `cam_rot` (3x3 rotation matrix)
  - `cam_trans` (3-vector translation)

---

## 2. Data Pipeline Changes

### 2.1 New Files to Create

#### `smal_fitter/sleap_data/preprocess_sleap_multiview_dataset.py`

**Purpose**: Preprocess multi-view SLEAP data into optimized HDF5 format where each sample contains all synchronized views.

**Key Design Decisions**:

1. **Sample Grouping**: Samples are grouped by `(session_name, frame_idx)` to ensure time synchronization
2. **Variable Views**: Handle cases where some frames may have missing views (store visibility mask, NO padding)
3. **Fixed View Order**: Establish a canonical view ordering per session for consistent training
4. **Store All Views**: The preprocessor stores ALL available views per sample; view selection/sampling happens at training time

**View Handling Strategy**:
- Preprocessor stores all available views per sample (up to `max_views_stored`)
- Training config specifies `num_views_to_use` 
- If sample has more views than `num_views_to_use`: randomly sample at load time
- If sample has fewer views than `num_views_to_use`: use all available views with masking

**Data Structure in HDF5**:

```
/multiview_images/
    image_jpeg_view_0: (N_samples,) vlen uint8     # JPEG bytes for view 0
    image_jpeg_view_1: (N_samples,) vlen uint8     # JPEG bytes for view 1
    ...
    image_jpeg_view_k: (N_samples,) vlen uint8     # JPEG bytes for view k
    view_mask: (N_samples, max_views) bool         # Which views are available

/multiview_keypoints/
    keypoints_2d: (N_samples, max_views, N_joints, 2) float32
    keypoint_visibility: (N_samples, max_views, N_joints) float32
    view_valid: (N_samples, max_views) bool        # Valid view mask

    # Optional calibrated-camera + 3D supervision (if available in source SLEAP/Anipose export)
    camera_intrinsics: (N_samples, max_views, 3, 3) float32
    camera_extrinsics_R: (N_samples, max_views, 3, 3) float32
    camera_extrinsics_t: (N_samples, max_views, 3) float32
    image_sizes: (N_samples, max_views, 2) int32          # (width, height) per view
    keypoints_3d: (N_samples, N_joints, 3) float32        # 3D joints in world coordinates (scaled by world_scale at load time)

/parameters/
    # Shared body parameters (placeholders for SLEAP data)
    global_rot: (N_samples, 3) float32
    joint_rot: (N_samples, N_POSE+1, 3) float32
    betas: (N_samples, N_BETAS) float32            # Ground truth if available
    trans: (N_samples, 3) float32

/auxiliary/
    session_name: (N_samples,) string
    frame_idx: (N_samples,) int32
    camera_names: (N_samples, max_views) string    # Camera name for each view slot
    num_views: (N_samples,) int32                  # Actual number of views per sample
    has_ground_truth_betas: (N_samples,) bool
    has_3d_data: (N_samples,) bool                 # Per-sample indicator for 3D labels

/metadata/
    attrs:
        dataset_type: 'sleap_multiview'
        is_multiview: True
        max_views: int                             # Maximum views across all samples
        canonical_camera_order: list[str]          # Ordered list of camera names
        target_resolution: int
        has_camera_parameters: bool
        has_3d_keypoints: bool
        world_scale: float                         # Scales 3D points + camera translations into SMILify units
        ...
```

**Key Methods**:

```python
class SLEAPMultiViewPreprocessor:
    def __init__(self, max_views: int = 8, ...):
        """
        Args:
            max_views: Maximum number of views to support (pad/truncate)
        """
    
    def discover_multiview_samples(self, sessions_dir: str) -> List[Dict]:
        """
        Discover all multi-view samples grouped by (session, frame).
        
        Returns:
            List of dicts: [{
                'session_path': str,
                'frame_idx': int,
                'camera_views': List[str],
                'num_views': int
            }, ...]
        """
    
    def process_multiview_sample(self, sample_info: Dict) -> Optional[Dict]:
        """
        Process a single multi-view sample (all views for one frame).
        
        Returns:
            Dict containing aligned images and keypoints for all views
        """
    
    def _align_views_to_canonical_order(self, views_data: Dict, 
                                         canonical_order: List[str]) -> Dict:
        """
        Reorder/pad view data to match canonical camera order.
        """
```

#### `smal_fitter/sleap_data/sleap_multiview_dataset.py`

**Purpose**: PyTorch dataset class for loading preprocessed multi-view SLEAP datasets.

**Key Design Decisions**:

1. **Return Format**: Return tuple of (images_list, keypoints_dict, metadata) where images_list is ordered by canonical camera order
2. **Compatibility**: Provide a method to return single-view format for backward compatibility
3. **View Masking**: Include boolean mask indicating which views are valid

**Interface**:

```python
class SLEAPMultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path: str, rotation_representation: str = '6d', 
                 num_views_to_use: int = None,  # None = use all available
                 return_single_view: bool = False, 
                 preferred_view: int = 0,
                 random_view_sampling: bool = True,  # Sample views randomly during training
                 ...):
        """
        Args:
            hdf5_path: Path to preprocessed multi-view HDF5 file
            rotation_representation: '6d' or 'axis_angle'
            num_views_to_use: Maximum views to use per sample (None = all available)
                             If sample has more views, randomly sample this many
                             If sample has fewer views, use all available (no padding)
            return_single_view: If True, return only one view (backward compatible)
            preferred_view: Which view to return if return_single_view=True
            random_view_sampling: If True, randomly sample views when num_views_to_use 
                                 is less than available views (for training augmentation)
        """
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        """
        Returns:
            x_data: {
                'images': List[np.ndarray],           # (N_active_views,) each (H, W, 3)
                'view_mask': np.ndarray,              # (num_views_to_use,) bool - which slots have valid data
                'camera_names': List[str],            # (N_active_views,) camera identifiers
                'camera_indices': np.ndarray,         # (N_active_views,) canonical indices for view embeddings
                'num_active_views': int,              # Actual number of views returned
                'session_name': str,
                'frame_idx': int,
                'is_multiview': True,
                'available_labels': {...}
            }
            y_data: {
                'keypoints_2d': np.ndarray,           # (N_active_views, N_joints, 2)
                'keypoint_visibility': np.ndarray,   # (N_active_views, N_joints) - per-keypoint visibility
                'view_valid': np.ndarray,             # (N_active_views,) bool - all True for active views
                'betas': np.ndarray or None,          # (N_BETAS,) shared across all views
                ...
            }
        """
    
    def _sample_views(self, available_views: List[int], num_available: int) -> List[int]:
        """
        Sample views based on configuration.
        
        If num_available > num_views_to_use: randomly sample num_views_to_use
        If num_available <= num_views_to_use: return all available (no padding)
        """
    
    def get_max_views_in_dataset(self) -> int:
        """Return the maximum number of views across all samples."""
    
    def get_canonical_camera_order(self) -> List[str]:
        """Return the canonical ordering of cameras."""
```

### 2.2 Modifications to Existing Files

#### `sleap_data_loader.py` - Minor Modifications

Add method to load all camera views for a single frame:

```python
def load_all_cameras_for_frame(self, frame_idx: int) -> Dict[str, Dict]:
    """
    Load data from all cameras for a specific frame.
    
    Args:
        frame_idx: Frame index
        
    Returns:
        Dict mapping camera_name -> camera_data for all available cameras
    """
```

### 2.3 New File: `smal_fitter/sleap_data/sleap_3d_loader.py`

**Purpose**: Load 3D keypoints and calibrated camera parameters from SLEAP/Anipose exports and provide robust conversion to the PyTorch3D camera convention used by SMILify.

**Key Responsibilities**:
- Load per-camera intrinsics `K`, extrinsics `(R, t)`, and image sizes
- Convert OpenCV/SLEAP camera parameters to PyTorch3D (`R_p3d`, `T_p3d`, `fov_y_degrees`, `aspect_ratio`)
- Provide visualization utilities for verifying 3D→2D projection alignment

**Key Design Detail (Aspect Ratio)**:
PyTorch3D’s `FoVPerspectiveCameras` requires explicit `aspect_ratio` for correct projection when `W!=H` and/or `fx!=fy`. The conversion derives:
\[
\text{aspect\_ratio}=\frac{W\cdot f_y}{H\cdot f_x}
\]

---

## 3. Model Architecture Changes

### 3.1 New File: `smal_fitter/neuralSMIL/multiview_smil_regressor.py`

**Purpose**: Multi-view variant of SMILImageRegressor that processes multiple views and predicts shared body parameters with per-view camera parameters.

**Key Design Decisions**:

1. **Feature Fusion Strategy**: Simple concatenation of features from all views
   - Alternative considered: Cross-attention between views (more complex, defer to future work)
2. **Camera Head Design**: Options (to be decided):
   - **Option A**: Single camera head that outputs all camera parameters (N_views * cam_params_dim)
   - **Option B**: N separate camera heads (one per canonical view position)
   - **Option C**: Single camera head + view embedding (most flexible)
3. **Inheritance**: Inherit from SMILImageRegressor to reuse rendering/loss code

**Interface**:

```python
class MultiViewSMILImageRegressor(SMILImageRegressor):
    def __init__(self, device, data_batch, batch_size, shape_family, use_unity_prior,
                 num_views: int,
                 feature_fusion: str = 'concat',  # 'concat', 'mean', 'attention'
                 camera_head_type: str = 'per_view',  # 'per_view', 'shared_with_embedding'
                 **kwargs):
        """
        Args:
            num_views: Number of input views
            feature_fusion: How to combine features from multiple views
            camera_head_type: Architecture for camera parameter prediction
        """
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-view input.
        
        Args:
            images: (batch_size, num_views, 3, H, W) or (batch_size * num_views, 3, H, W)
            
        Returns:
            Dict containing:
                - 'global_rot': (batch_size, 6) - shared
                - 'joint_rot': (batch_size, N_POSE, 6) - shared
                - 'betas': (batch_size, N_BETAS) - shared
                - 'trans': (batch_size, 3) - shared
                - 'fov': (batch_size, num_views, 1) - per-view
                - 'cam_rot': (batch_size, num_views, 3, 3) - per-view
                - 'cam_trans': (batch_size, num_views, 3) - per-view
        """
    
    def forward_single_view(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible single-view forward pass.
        Treats input as view 0 and returns single-view format.
        """
    
    def predict_from_multiview_batch(self, x_data_batch, y_data_batch) -> Tuple:
        """
        Process a batch of multi-view samples.
        
        Similar to predict_from_batch but handles multi-view data format.
        """
```

### 3.2 Architecture Details

#### Feature Extraction (Shared Backbone)

```python
def _extract_multiview_features(self, images: torch.Tensor) -> torch.Tensor:
    """
    Extract features from all views using shared backbone.
    
    Args:
        images: (batch_size, num_views, 3, H, W)
        
    Returns:
        features: (batch_size, num_views, feature_dim)
    """
    batch_size, num_views, C, H, W = images.shape
    
    # Reshape for batch processing through backbone
    images_flat = images.view(batch_size * num_views, C, H, W)
    
    # Extract features (shared backbone weights)
    features_flat = self.backbone(images_flat)  # (B*V, feature_dim)
    
    # Reshape back
    features = features_flat.view(batch_size, num_views, -1)
    
    return features
```

#### Feature Fusion

```python
def _fuse_features(self, features: torch.Tensor, view_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Fuse features from multiple views.
    
    Args:
        features: (batch_size, num_views, feature_dim)
        view_mask: (batch_size, num_views) bool mask of valid views
        
    Returns:
        fused: (batch_size, fused_feature_dim)
    """
    if self.feature_fusion == 'concat':
        # Simple concatenation
        return features.view(features.size(0), -1)  # (B, V * feature_dim)
    
    elif self.feature_fusion == 'mean':
        # Mean pooling (handles missing views via mask)
        if view_mask is not None:
            mask_expanded = view_mask.unsqueeze(-1).float()
            features_masked = features * mask_expanded
            return features_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return features.mean(dim=1)
    
    elif self.feature_fusion == 'attention':
        # Cross-attention between views (future work)
        raise NotImplementedError("Attention fusion not yet implemented")
```

#### Body Parameter Head

```python
def _predict_body_params(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Predict shared body parameters from fused features.
    
    Uses same architecture as single-view but with larger input dimension.
    """
    # Similar to existing MLP/Transformer head
    # Output: global_rot, joint_rot, betas, trans
```

#### Camera Parameter Head (Per-View)

**Option A: Separate heads per view position**

```python
def __init__(self, ...):
    # Create camera head for each view position
    self.camera_heads = nn.ModuleList([
        nn.Sequential(
            nn.Linear(fused_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cam_params_dim)  # fov + cam_rot + cam_trans
        )
        for _ in range(num_views)
    ])

def _predict_camera_params(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Predict camera parameters for each view.
    
    Returns:
        Dict with 'fov', 'cam_rot', 'cam_trans' each of shape (B, num_views, ...)
    """
    all_cam_params = []
    for i, head in enumerate(self.camera_heads):
        cam_params_i = head(fused_features)
        all_cam_params.append(cam_params_i)
    
    # Stack and parse into fov, cam_rot, cam_trans
    ...
```

**Option B: Shared head with view embedding (preferred for permutation handling)**

```python
def __init__(self, ...):
    # View embedding
    self.view_embedding = nn.Embedding(max_views, view_embed_dim)
    
    # Shared camera head
    self.camera_head = nn.Sequential(
        nn.Linear(fused_feature_dim + view_embed_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, cam_params_dim)
    )

def _predict_camera_params(self, fused_features: torch.Tensor, 
                           view_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Predict camera parameters using shared head with view embeddings.
    
    Args:
        fused_features: (batch_size, fused_feature_dim)
        view_indices: (batch_size, num_views) indices into view embedding
    """
    batch_size = fused_features.size(0)
    
    all_cam_params = []
    for v in range(self.num_views):
        # Get view embedding
        view_embed = self.view_embedding(view_indices[:, v])
        
        # Concatenate with fused features
        cam_input = torch.cat([fused_features, view_embed], dim=-1)
        
        # Predict camera params for this view
        cam_params_v = self.camera_head(cam_input)
        all_cam_params.append(cam_params_v)
    
    # Stack: (B, num_views, cam_params_dim)
    all_cam_params = torch.stack(all_cam_params, dim=1)
    
    # Parse into components
    ...
```

### 3.3 Modifications to Existing Files

#### `smil_image_regressor.py` - Minor Modifications

Add class method to create multi-view variant:

```python
@classmethod
def create_multiview(cls, num_views: int, **kwargs) -> 'MultiViewSMILImageRegressor':
    """Factory method to create multi-view regressor."""
    from multiview_smil_regressor import MultiViewSMILImageRegressor
    return MultiViewSMILImageRegressor(num_views=num_views, **kwargs)
```

---

## 4. Loss Computation Changes

### 4.1 Multi-View Loss Strategy

The key insight is that **body parameters are shared** across views, but **2D keypoint loss must be computed per-view** using each view's camera parameters.

```
                         ┌─────────────────────┐
                         │  Predicted Body     │
                         │  Parameters         │
                         │  (shared)           │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
     ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
     │ Camera Params  │   │ Camera Params  │   │ Camera Params  │
     │    View 0      │   │    View 1      │   │    View 2      │
     └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
             │                    │                    │
             ▼                    ▼                    ▼
     ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
     │ Render 2D      │   │ Render 2D      │   │ Render 2D      │
     │ Keypoints      │   │ Keypoints      │   │ Keypoints      │
     │ (View 0)       │   │ (View 1)       │   │ (View 2)       │
     └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
             │                    │                    │
             ▼                    ▼                    ▼
     ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
     │ Loss vs GT     │   │ Loss vs GT     │   │ Loss vs GT     │
     │ Keypoints      │   │ Keypoints      │   │ Keypoints      │
     │ (View 0)       │   │ (View 1)       │   │ (View 2)       │
     └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                                  ▼
                          ┌───────────────┐
                          │  Total Loss   │
                          │  (averaged    │
                          │  over views)  │
                          └───────────────┘
```

### 4.2 Loss Computation Implementation

**Visibility-Weighted Loss Strategy**:
- Each keypoint has a visibility score (0.0 to 1.0)
- Loss for each keypoint is weighted by its visibility
- This naturally handles partial occlusions: more confident keypoints contribute more
- Per-view loss is normalized by total visibility weight (not count) to avoid penalizing views with fewer visible keypoints

**Joint Angle Regularization**:
- Added penalty for joint angles that deviate from default (0,0,0) angles
- Excludes root joint (which is required for global rotation)
- Uses L2 penalty (squared norm) to encourage natural poses and prevent extreme joint angles
- Configurable weight via `joint_angle_regularization` in loss curriculum
- Default weight: 0.001 (can be adjusted per training stage)
- This is especially useful for early training stages, before halfway decent camera predictions are produced which can otherwise lead the model to contort violently,

```python
def compute_multiview_batch_loss(self, predicted_params: Dict[str, torch.Tensor],
                                  target_params_batch: Dict[str, torch.Tensor],
                                  auxiliary_data: Dict,
                                  loss_weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict]:
    """
    Compute loss for multi-view batch.
    
    Key differences from single-view:
    1. Body parameter losses computed once (shared params from ALL views)
    2. 2D keypoint loss computed per-view with VISIBILITY WEIGHTING
    3. View validity mask applied to exclude invalid views
    4. Final loss is weighted average across views
    """
    total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
    loss_components = {}
    
    batch_size = predicted_params['global_rot'].shape[0]
    num_views = predicted_params['cam_rot'].shape[1]  # (B, V, 3, 3)
    
    # 1. Body parameter losses (same as single-view, but params derived from ALL views)
    if loss_weights.get('betas', 0) > 0:
        betas_loss = self._compute_betas_loss(predicted_params, target_params_batch)
        loss_components['betas'] = betas_loss
        total_loss = total_loss + loss_weights['betas'] * betas_loss
    
    # Similar for global_rot, joint_rot, trans (if ground truth available)
    ...
    
    # Joint angle regularization: penalize deviations from default (0,0,0) angles
    # Excludes root joint (which is required for global rotation)
    if loss_weights.get('joint_angle_regularization', 0) > 0:
        joint_rot_pred = predicted_params['joint_rot']  # (batch_size, N_POSE, 6) or (batch_size, N_POSE, 3)
        
        # Convert to axis-angle if needed
        if self.rotation_representation == '6d':
            joint_rot_aa = rotation_6d_to_axis_angle(joint_rot_pred)  # (batch_size, N_POSE, 3)
        else:
            joint_rot_aa = joint_rot_pred  # Already in axis-angle format
        
        # Compute L2 norm of joint angles (N_POSE excludes root joint)
        joint_angle_norms = torch.norm(joint_rot_aa, dim=-1)  # (batch_size, N_POSE)
        
        # Regularization: penalize large joint angles (L2 penalty)
        joint_angle_reg = torch.mean(joint_angle_norms ** 2)
        
        loss_components['joint_angle_regularization'] = joint_angle_reg
        total_loss = total_loss + loss_weights['joint_angle_regularization'] * joint_angle_reg
    
    # 2. Per-view 2D keypoint loss WITH VISIBILITY WEIGHTING
    if loss_weights.get('keypoint_2d', 0) > 0:
        view_keypoint_losses = []
        view_visibility_weights = []  # Track total visibility per view for weighted average
        
        for v in range(num_views):
            # Get view validity mask
            view_valid = auxiliary_data['view_valid'][:, v]  # (B,)
            
            if not view_valid.any():
                continue
            
            # Extract camera params for this view
            cam_params_v = {
                'fov': predicted_params['fov'][:, v],
                'cam_rot': predicted_params['cam_rot'][:, v],
                'cam_trans': predicted_params['cam_trans'][:, v]
            }
            
            # Render 2D keypoints using UNIFIED body params + view-specific camera
            rendered_joints_v = self._render_keypoints_with_camera(
                predicted_params,  # Body params (SHARED across all views)
                cam_params_v       # Camera params (UNIQUE to this view)
            )
            
            # Get ground truth keypoints and visibility for this view
            gt_keypoints_v = auxiliary_data['keypoints_2d'][:, v]  # (B, N_joints, 2)
            gt_visibility_v = auxiliary_data['keypoint_visibility'][:, v]  # (B, N_joints)
            
            # Compute VISIBILITY-WEIGHTED loss for valid samples in this view
            loss_v, total_vis_weight_v = self._compute_visibility_weighted_keypoint_loss(
                rendered_joints_v[view_valid],
                gt_keypoints_v[view_valid],
                gt_visibility_v[view_valid]
            )
            
            view_keypoint_losses.append(loss_v)
            view_visibility_weights.append(total_vis_weight_v)
        
        # Weighted average over views (weight by total visibility in each view)
        if view_keypoint_losses:
            losses_tensor = torch.stack(view_keypoint_losses)
            weights_tensor = torch.stack(view_visibility_weights)
            
            # Normalize weights
            weights_normalized = weights_tensor / (weights_tensor.sum() + 1e-8)
            
            # Weighted average
            keypoint_2d_loss = (losses_tensor * weights_normalized).sum()
            
            loss_components['keypoint_2d'] = keypoint_2d_loss
            total_loss = total_loss + loss_weights['keypoint_2d'] * keypoint_2d_loss
    
    return total_loss, loss_components


def _compute_visibility_weighted_keypoint_loss(self, 
                                                rendered_joints: torch.Tensor,
                                                gt_keypoints: torch.Tensor,
                                                visibility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute visibility-weighted 2D keypoint loss.
    
    Args:
        rendered_joints: (B, N_joints, 2) - rendered 2D keypoints
        gt_keypoints: (B, N_joints, 2) - ground truth 2D keypoints
        visibility: (B, N_joints) - per-keypoint visibility weights (0.0 to 1.0)
        
    Returns:
        Tuple of (weighted_loss, total_visibility_weight)
    """
    # Per-keypoint squared error
    diff = rendered_joints - gt_keypoints  # (B, N_joints, 2)
    per_keypoint_loss = (diff ** 2).sum(dim=-1)  # (B, N_joints)
    
    # Apply visibility weighting
    weighted_loss = per_keypoint_loss * visibility  # (B, N_joints)
    
    # Sum over keypoints, mean over batch
    total_vis_weight = visibility.sum()  # Total visibility weight for this view
    
    if total_vis_weight > 0:
        loss = weighted_loss.sum() / total_vis_weight  # Normalize by visibility
    else:
        loss = torch.tensor(0.0, device=rendered_joints.device, requires_grad=True)
    
    return loss, total_vis_weight
```

### 4.3 Rendering with Per-View Camera

Need to modify `_compute_rendered_outputs` to accept camera parameters:

```python
def _render_keypoints_with_camera(self, body_params: Dict, 
                                   cam_params: Dict) -> torch.Tensor:
    """
    Render 2D keypoints using specified camera parameters.
    
    Args:
        body_params: Dict with global_rot, joint_rot, betas, trans
        cam_params: Dict with fov, cam_rot, cam_trans
        
    Returns:
        rendered_joints: (batch_size, n_joints, 2)
    """
    # Run SMAL model with body params
    verts, joints, Rs, v_shaped = self.smal_model(
        body_params['betas'],
        torch.cat([body_params['global_rot'].unsqueeze(1),
                   body_params['joint_rot']], dim=1),
        ...
    )
    
    # Apply transformation
    ...
    
    # Set camera and render
    self.renderer.set_camera_parameters(
        R=cam_params['cam_rot'],
        T=cam_params['cam_trans'],
        fov=cam_params['fov']
    )
    
    _, rendered_joints = self.renderer(verts, joints, faces)
    
    return rendered_joints
```

---

## 5. Training Pipeline Changes

### 5.1 New File: `smal_fitter/neuralSMIL/train_multiview_regressor.py`

**Purpose**: Training script for multi-view model. Can be standalone or integrated into existing `train_smil_regressor.py`.

**Key Changes**:

1. **Data Loading**: Use `SLEAPMultiViewDataset` with custom collate function
2. **Model Creation**: Instantiate `MultiViewSMILImageRegressor`
3. **Loss Computation**: Call `compute_multiview_batch_loss`
4. **Visualization**: Update to show all views

### 5.2 Custom Collate Function for Multi-View

```python
def multiview_collate_fn(batch):
    """
    Collate function for multi-view batches.
    
    Args:
        batch: List of (x_data, y_data) where each x_data contains 'images' list
        
    Returns:
        Batched x_data and y_data with proper stacking
    """
    x_data_batch = []
    y_data_batch = []
    
    for x_data, y_data in batch:
        # Stack images: List[np.ndarray] -> (num_views, H, W, C)
        if 'images' in x_data and isinstance(x_data['images'], list):
            # Validate all views have same shape
            shapes = [img.shape for img in x_data['images'] if img is not None]
            if shapes and all(s == shapes[0] for s in shapes):
                x_data['images_stacked'] = np.stack(x_data['images'], axis=0)
        
        x_data_batch.append(x_data)
        y_data_batch.append(y_data)
    
    return x_data_batch, y_data_batch
```

### 5.3 Training Configuration Options

The following configuration options control multi-view behavior:

```python
# Training config (YAML or argparse)
multiview_config = {
    # Enable/disable multi-view mode
    'multiview_enabled': True,
    
    # Maximum views to use per sample during training
    # If a sample has more views, randomly sample this many
    # If a sample has fewer views, use all available (no padding)
    'num_views_to_use': 4,  # e.g., 2, 3, 4, 8, or None for all available
    
    # Whether to randomly sample views when more are available
    # If False, use first N views in canonical order
    'random_view_sampling': True,
    
    # View permutation augmentation (for learning view-invariant features)
    'enable_view_permutation': True,
    'view_permutation_prob': 0.5,  # Probability of shuffling view order
    
    # Feature fusion strategy
    'feature_fusion': 'concat',  # 'concat', 'mean', 'attention' (future)
    
    # Camera head architecture
    'camera_head_type': 'per_view',  # 'per_view', 'shared_with_embedding'
}
```

**Command Line Interface**:

```bash
# Multi-view training with 4 views
python train_multiview_regressor.py \
    --dataset /path/to/multiview_sleap.h5 \
    --multiview_enabled \
    --num_views_to_use 4 \
    --random_view_sampling \
    --feature_fusion concat

# Use all available views
python train_multiview_regressor.py \
    --dataset /path/to/multiview_sleap.h5 \
    --multiview_enabled \
    --num_views_to_use -1  # -1 or None means use all
```

### 5.4 Integration with Existing Training

**Option A: Separate Training Script**
- Pros: Clean separation, no risk of breaking existing code
- Cons: Code duplication

**Option B: Flag-Based Integration**
- Add `--multiview` flag to existing `train_smil_regressor.py`
- Conditionally load multi-view dataset and model

**Recommendation**: Start with Option A, then refactor to Option B once stable.

---

## 6. Permutation Equivariance

### 6.1 Problem Statement

If we shuffle the order of input images, we want the model to still predict the correct camera parameters for each view. This requires the model to either:

1. Learn to identify cameras from image content (implicit), or
2. Be explicitly told which camera each image comes from (explicit)

### 6.2 Proposed Solution: View Embeddings + Data Augmentation

**During Training**:

1. Randomly shuffle the view order with probability `p_shuffle` (e.g., 0.5)
2. Apply the same permutation to:
   - Input images
   - Ground truth keypoints
   - View embeddings (used in camera head)
3. Camera head uses view embeddings to know which camera position each input corresponds to

**Implementation**:

```python
def _apply_view_permutation(self, x_data: Dict, y_data: Dict, 
                            permutation: torch.Tensor) -> Tuple[Dict, Dict]:
    """
    Apply a permutation to multi-view data.
    
    Args:
        x_data: Multi-view x_data dict
        y_data: Multi-view y_data dict  
        permutation: (num_views,) tensor of view indices
        
    Returns:
        Permuted x_data and y_data
    """
    # Permute images
    x_data['images'] = [x_data['images'][i] for i in permutation]
    
    # Permute keypoints
    y_data['keypoints_2d'] = y_data['keypoints_2d'][permutation]
    y_data['keypoint_visibility'] = y_data['keypoint_visibility'][permutation]
    
    # Store permutation for camera head
    x_data['view_permutation'] = permutation
    
    return x_data, y_data
```

**In Forward Pass**:

```python
def forward(self, images: torch.Tensor, 
            view_permutation: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """
    Args:
        images: (B, V, 3, H, W) - views may be permuted
        view_permutation: (B, V) - indices mapping image order to canonical order
                         If None, assume canonical order
    """
    # Extract and fuse features
    features = self._extract_multiview_features(images)
    fused = self._fuse_features(features)
    
    # Predict body params (permutation-invariant)
    body_params = self._predict_body_params(fused)
    
    # Predict camera params with view information
    if view_permutation is None:
        # Use canonical order
        view_indices = torch.arange(self.num_views, device=images.device)
        view_indices = view_indices.unsqueeze(0).expand(images.size(0), -1)
    else:
        # Use provided permutation to get correct view embeddings
        view_indices = view_permutation
    
    cam_params = self._predict_camera_params(fused, view_indices)
    
    return {**body_params, **cam_params}
```

### 6.3 Alternative: Implicit Camera Identification

Let the model learn to identify cameras purely from image content:
- Different cameras have different viewpoints, lens characteristics, backgrounds
- Model may learn these patterns implicitly

**Trade-offs**:
- Pros: Simpler implementation, more general
- Cons: May not generalize to new camera setups, requires more training data

---

## 7. File Structure

```
SMILify/
├── smal_fitter/
│   ├── sleap_data/
│   │   ├── sleap_dataset.py                      # Existing (unchanged)
│   │   ├── preprocess_sleap_dataset.py           # Existing (unchanged)
│   │   ├── sleap_multiview_dataset.py            # NEW
│   │   └── preprocess_sleap_multiview_dataset.py # NEW
│   │
│   └── neuralSMIL/
│       ├── smil_image_regressor.py               # Minor additions
│       ├── multiview_smil_regressor.py           # NEW
│       ├── train_smil_regressor.py               # Optional: add --multiview flag
│       └── train_multiview_regressor.py          # NEW (optional standalone)
│
├── sleap_data_loader.py                          # Minor additions
└── MULTI_VIEW_IMPLEMENTATION_PLAN.md             # This document
```

---

## 8. Implementation Phases

### Phase 1: Data Pipeline (Week 1)

**Tasks**:
1. [ ] Implement `preprocess_sleap_multiview_dataset.py`
   - Sample discovery and grouping by frame
   - Multi-view sample processing
   - HDF5 storage with multi-view structure
2. [ ] Implement `sleap_multiview_dataset.py`
   - Loading multi-view samples
   - Backward compatibility mode (single-view)
3. [ ] Add `load_all_cameras_for_frame()` to `sleap_data_loader.py`
4. [ ] Unit tests for data pipeline

**Deliverables**:
- Working preprocessor that creates multi-view HDF5
- Working dataset class that loads multi-view samples
- Tests verifying data integrity

### Phase 2: Model Architecture (Week 2)

**Tasks**:
1. [ ] Implement `multiview_smil_regressor.py`
   - Feature extraction with shared backbone
   - Feature fusion (concatenation)
   - Body parameter head
   - Camera parameter head (start with per-view heads)
2. [ ] Implement view embedding mechanism
3. [ ] Verify forward pass produces correct output shapes
4. [ ] Unit tests for model components

**Deliverables**:
- Working multi-view model that produces predictions
- Tests verifying output shapes and gradient flow

### Phase 3: Loss Computation (Week 3)

**Tasks**:
1. [ ] Implement `compute_multiview_batch_loss()`
2. [ ] Implement `_render_keypoints_with_camera()`
3. [ ] Verify gradients flow correctly through per-view rendering
4. [ ] Unit tests for loss computation

**Deliverables**:
- Working loss computation that handles multiple views
- Tests verifying gradient flow and loss values

### Phase 4: Training Integration (Week 4)

**Tasks**:
1. [ ] Implement `train_multiview_regressor.py` or add flag to existing trainer
2. [ ] Implement multi-view collate function
3. [ ] Implement view permutation augmentation
4. [ ] Add multi-view visualization
5. [ ] Integration tests

**Deliverables**:
- Working training pipeline for multi-view model
- Visualization of multi-view predictions

### Phase 5: Testing and Validation (Week 5)

**Tasks**:
1. [ ] End-to-end testing with real SLEAP data
2. [ ] Performance benchmarking (single-view vs multi-view)
3. [ ] Hyperparameter tuning
4. [ ] Documentation

**Deliverables**:
- Validated multi-view pipeline
- Performance comparison results
- Complete documentation

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# test_multiview_preprocessor.py
def test_sample_grouping():
    """Test that samples are correctly grouped by (session, frame)."""

def test_view_alignment():
    """Test that views are correctly aligned to canonical order."""

def test_missing_views():
    """Test handling of frames with missing views."""

# test_multiview_dataset.py
def test_load_multiview_sample():
    """Test loading a multi-view sample."""

def test_backward_compatible_mode():
    """Test single-view return mode."""

def test_view_mask():
    """Test that view mask correctly indicates valid views."""

# test_multiview_model.py
def test_forward_shape():
    """Test output shapes of multi-view model."""

def test_gradient_flow():
    """Test that gradients flow through all components."""

def test_camera_head_isolation():
    """Test that each view's camera params are predicted independently."""

# test_multiview_loss.py
def test_per_view_loss():
    """Test that 2D keypoint loss is computed per-view."""

def test_shared_body_loss():
    """Test that body param loss uses shared params."""

def test_view_masking_in_loss():
    """Test that invalid views are excluded from loss."""
```

### 9.2 Integration Tests

```python
def test_end_to_end_training():
    """Test complete training loop with multi-view data."""

def test_view_permutation_equivariance():
    """Test that shuffling views produces consistent predictions."""

def test_mixed_batch():
    """Test batches with different numbers of valid views."""
```

### 9.3 Regression Tests

Ensure single-view workflow remains unchanged:

```python
def test_single_view_unchanged():
    """Verify single-view model produces identical results after changes."""
```

---

## 10. Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gradient vanishing through per-view rendering | Training fails | Add skip connections, monitor gradients |
| Memory explosion with multiple views | OOM errors | Implement gradient checkpointing, batch processing |
| Breaking existing single-view workflow | Major regression | Comprehensive regression tests, feature flags |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Camera identification fails (permutation) | Poor camera param estimation | Start with canonical ordering, add view embeddings |
| View synchronization errors in data | Inconsistent training | Validate frame timestamps, add consistency checks |
| Performance regression | Slower training | Profile and optimize, consider view subsampling |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| HDF5 schema changes break compatibility | Need to regenerate data | Version schema, provide migration tools |
| Feature fusion strategy suboptimal | Suboptimal accuracy | Design for easy swapping of fusion methods |

---

## 11. Design Decisions (Resolved)

The following decisions have been made based on team discussion:

### Data & View Handling

| Decision | Resolution |
|----------|------------|
| **Maximum Views** | User-configurable via training config (`num_views_to_use`) |
| **More views than configured** | Randomly sample `num_views_to_use` views during training |
| **Fewer views than configured** | Use masking (no padding) - model handles variable view counts |
| **Body param prediction** | Uses ALL available views (features concatenated from all valid views) |

### Loss & Training

| Decision | Resolution |
|----------|------------|
| **Loss Weighting** | Weighted by keypoint visibility (more visible keypoints = higher contribution) |
| **Curriculum Learning** | NO - use all available views from the start |
| **View availability** | Use all views when available; only mask when views are genuinely missing |

### Architecture (Still Open)

| Question | Status |
|----------|--------|
| **Feature Fusion Strategy** | Start with concatenation, design for swappable fusion methods |
| **Camera Head Design** | Per-view heads vs. shared head with embedding - TBD during implementation |
| **Permutation Handling** | View embeddings + random shuffling during training |

---

## Appendix A: Data Format Examples

### Multi-View Sample Structure (with View Sampling)

```python
# Example: Dataset has 5 cameras, but num_views_to_use=3
# Dataset returns randomly sampled 3 views (no padding)

# x_data for a sampled 3-view sample (from 5 available)
x_data = {
    'images': [
        np.array((224, 224, 3), dtype=float32),  # Sampled view (e.g., originally view 0 = top)
        np.array((224, 224, 3), dtype=float32),  # Sampled view (e.g., originally view 2 = back)  
        np.array((224, 224, 3), dtype=float32),  # Sampled view (e.g., originally view 4 = side)
    ],
    'view_mask': np.array([True, True, True]),   # All returned views are valid (no padding)
    'camera_names': ['top', 'back', 'side'],     # Actual camera names for sampled views
    'camera_indices': np.array([0, 2, 4]),       # Canonical indices for view embeddings
    'num_active_views': 3,                       # Actual number of views returned
    'session_name': 'session_001',
    'frame_idx': 42,
    'is_multiview': True,
    'available_labels': {
        'betas': True,
        'keypoint_2d': True,
        'joint_rot': False,
        ...
    }
}

# y_data for sampled 3-view sample
y_data = {
    'keypoints_2d': np.array((3, 25, 2), dtype=float32),      # (active_views, joints, xy)
    'keypoint_visibility': np.array((3, 25), dtype=float32),  # (active_views, joints) - 0.0 to 1.0
    'view_valid': np.array([True, True, True]),               # All active views are valid
    'betas': np.array((20,), dtype=float32),                  # SHARED shape across all views
    # Placeholder params (no ground truth for SLEAP)
    'global_rot': None,
    'joint_angles': None,
    ...
}

# Example: Sample only has 2 views available, but num_views_to_use=3
# Dataset returns all 2 available views (no padding to 3)

x_data_sparse = {
    'images': [
        np.array((224, 224, 3), dtype=float32),  # View 0
        np.array((224, 224, 3), dtype=float32),  # View 1
        # NO view 2 - we don't pad!
    ],
    'view_mask': np.array([True, True]),        # Only 2 valid views
    'camera_names': ['top', 'side'],
    'camera_indices': np.array([0, 1]),
    'num_active_views': 2,                       # Only 2 views available
    ...
}
```

### Predicted Parameters Structure

```python
# Model outputs for a batch with variable views
# Note: camera params dimension matches num_active_views for each sample

predicted_params = {
    # UNIFIED body parameters (predicted from ALL view features combined)
    'global_rot': torch.Tensor((batch_size, 6)),               # 6D rotation - SINGLE per sample
    'joint_rot': torch.Tensor((batch_size, N_POSE, 6)),        # 6D rotations - SINGLE per sample
    'betas': torch.Tensor((batch_size, N_BETAS)),              # Shape params - SINGLE per sample
    'trans': torch.Tensor((batch_size, 3)),                    # Translation - SINGLE per sample
    'log_beta_scales': torch.Tensor((batch_size, N_JOINTS, 3)), # Scale PCA - SINGLE per sample
    'betas_trans': torch.Tensor((batch_size, N_JOINTS, 3)),    # Trans PCA - SINGLE per sample
    
    # Per-view camera parameters (SEPARATE for each view)
    'fov': torch.Tensor((batch_size, num_active_views, 1)),           # FOV per view
    'cam_rot': torch.Tensor((batch_size, num_active_views, 3, 3)),    # Rotation per view
    'cam_trans': torch.Tensor((batch_size, num_active_views, 3)),     # Translation per view
}
```

### Loss Computation Flow Example

```python
# For a sample with 3 active views:

# Step 1: Model predicts UNIFIED body params from concatenated features
body_params = model.predict_body(concat([feat_view0, feat_view1, feat_view2]))
# body_params = {global_rot, joint_rot, betas, trans, scales, trans_pca}

# Step 2: Model predicts SEPARATE camera params for each view
cam_params_0 = model.predict_camera(features, view_idx=0)
cam_params_1 = model.predict_camera(features, view_idx=1)
cam_params_2 = model.predict_camera(features, view_idx=2)

# Step 3: Compute loss for each view
loss_view_0 = visibility_weighted_keypoint_loss(
    render(body_params, cam_params_0),  # Unified body + camera 0
    gt_keypoints_view_0,
    visibility_view_0  # Per-keypoint visibility weights
)

loss_view_1 = visibility_weighted_keypoint_loss(
    render(body_params, cam_params_1),  # Same unified body + camera 1
    gt_keypoints_view_1,
    visibility_view_1
)

loss_view_2 = visibility_weighted_keypoint_loss(
    render(body_params, cam_params_2),  # Same unified body + camera 2
    gt_keypoints_view_2,
    visibility_view_2
)

# Step 4: Combine losses (weighted by total visibility per view)
total_2d_loss = weighted_average([loss_view_0, loss_view_1, loss_view_2],
                                  weights=[vis_sum_0, vis_sum_1, vis_sum_2])
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **View** | A single camera perspective/image |
| **Active views** | Views that are actually present for a given sample (not padded/masked) |
| **Multi-view sample** | A set of synchronized views from the same time instant |
| **Canonical order** | The fixed ordering of cameras used for training |
| **View embedding** | Learned representation of camera identity |
| **View mask** | Boolean mask indicating which view slots contain valid data |
| **View sampling** | Random selection of N views when more than N are available |
| **Feature fusion** | Method of combining features from multiple views |
| **Body parameters** | SMIL parameters describing animal pose and shape (UNIFIED across views) |
| **Camera parameters** | Parameters describing camera pose and intrinsics (SEPARATE per view) |
| **Visibility weighting** | Weighting per-keypoint loss by keypoint visibility confidence |
| **Unified body params** | Single set of body parameters predicted from ALL view features |
| **num_views_to_use** | Training config option specifying max views to use per sample |
| **Joint angle regularization** | L2 penalty on joint angles to encourage natural poses and prevent extreme angles (excludes root joint) |

---

## Appendix C: Key Implementation Details

### Handling Variable View Counts in Batches

Since different samples may have different numbers of active views, the collate function needs special handling:

```python
def multiview_collate_fn(batch):
    """
    Collate function that handles variable view counts.
    
    Strategy: Pad camera params to max_views_in_batch, use view_mask to track valid views.
    Body params remain unchanged (single set per sample).
    """
    max_views_in_batch = max(x['num_active_views'] for x, y in batch)
    
    # For each sample, pad camera-related tensors to max_views_in_batch
    # Body params need no padding (single set per sample)
    ...
```

### Gradient Flow Verification

Critical to verify gradients flow correctly:

1. **Body params ← All views**: Gradients from each view's 2D loss should backprop through the unified body params
2. **Camera params ← Single view**: Each view's camera params should only receive gradients from that view's loss
3. **Feature fusion**: Gradients should flow back through fusion to all backbone outputs

```python
# Test gradient flow
def test_gradient_flow():
    # Forward pass
    pred = model(images)  # images: (B, V, 3, H, W)
    
    # Compute loss for view 0 only
    loss_v0 = compute_loss_single_view(pred, targets, view_idx=0)
    loss_v0.backward()
    
    # Verify:
    # - body_head.weight.grad is NOT None (body params affected)
    # - camera_head_0.weight.grad is NOT None (camera 0 affected)
    # - camera_head_1.weight.grad IS None (camera 1 NOT affected by view 0 loss)
```

### Device Placement

**Critical**: All tensors must be on the same device before operations like `torch.stack()`.

**Implementation**:
- `preprocess_image()` returns tensors on CPU (created via `torch.from_numpy()`)
- Dummy padding images are created directly on the target device
- **Fix**: Explicitly move preprocessed images to device with `.to(self.device)` before adding to batch lists

```python
# In predict_from_multiview_batch:
img_tensor = self.preprocess_image(img).squeeze(0)
img_tensor = img_tensor.to(self.device)  # Ensure device consistency
all_images_per_view[v].append(img_tensor)
```

---

## Appendix D: Implementation Status

### ✅ Completed Implementation

The following files have been created implementing the multi-view system:

#### Phase 1: Data Pipeline

| File | Status | Description |
|------|--------|-------------|
| `smal_fitter/sleap_data/preprocess_sleap_multiview_dataset.py` | ✅ Complete | Multi-view SLEAP preprocessor that groups frames by (session, frame_idx) and stores all camera views together |
| `smal_fitter/sleap_data/sleap_multiview_dataset.py` | ✅ Complete | PyTorch dataset for loading multi-view samples with view sampling and masking support |
| `smal_fitter/sleap_data/sleap_3d_loader.py` | ✅ Complete | Loader + conversion utilities for calibrated cameras + 3D keypoints from SLEAP/Anipose |

#### Phase 2: Model Architecture

| File | Status | Description |
|------|--------|-------------|
| `smal_fitter/neuralSMIL/multiview_smil_regressor.py` | ✅ Complete | Multi-view regressor with cross-attention feature fusion and per-view camera heads |

**Key Implementation Details**:
- **Feature Fusion**: Cross-attention between views (not simple concatenation)
- **Camera Heads**: N separate heads, one per canonical view position
- **View Embeddings**: Learned embeddings help identify which camera a view belongs to
- **Global mesh scaling (optional)**: model can predict a single scalar `mesh_scale` used consistently during training + visualization (no direct GT)

#### Phase 3: Training Pipeline

| File | Status | Description |
|------|--------|-------------|
| `smal_fitter/neuralSMIL/train_multiview_regressor.py` | ✅ Complete | Standalone training script with DDP support, mixed precision, and comprehensive configuration |

#### Phase 4: Visualization

| File | Status | Description |
|------|--------|-------------|
| `train_multiview_regressor.py::visualize_multiview_training_progress()` | ✅ Complete | Multi-view visualization showing all camera views in a grid |
| `train_multiview_regressor.py::visualize_singleview_renders()` | ✅ Complete | SMALFitter-style mesh rendering for each view separately |

**Multi-View Grid Visualization** (`visualize_multiview_training_progress`):
- Creates a grid visualization showing all camera views side-by-side
- Top row: Input images from each camera
- Bottom row: Rendered mesh using unified body params + per-view camera params
- Overlays GT keypoints (circles) and predicted keypoints (crosses) on each view

**Single-View Mesh Renders** (`visualize_singleview_renders`):
- Uses SMALFitter to generate full mesh visualization for each camera view
- Shows complete collage including: input image, rendered mesh, keypoint overlay
- Uses unified body parameters (pose, shape) with per-view camera parameters
- Produces one visualization file per view per sample
- Output directory: `multiview_singleview_renders/epoch_XXX/`
- Filename format: `sample_XXX_view_XX_epoch_XXX.png`
- Enables visual verification that the same body produces correct 2D projections across all views

**GT Cam + GT 3D Projection Sanity Overlay**:
- When `has_3d_data` and calibrated cameras are present, visualization projects **GT 3D keypoints** using **GT camera parameters** and overlays them on the image.
- This catches convention mismatches (R/T), unit scale issues (mm vs m), and aspect ratio errors early.

**Mesh Geometry Export** (`export_mesh_as_obj`):
- Exports mesh vertices and faces as OBJ files for external inspection
- One OBJ file per view per sample: `sample_XXX_view_XX_epoch_XXX.obj` (this is stupid because only the camera rotation differs between meshes, so this is just here for debugging. Later, we will generate only one mesh for each multi-view rendering and export camera parameters alongside.)
- Uses the same predicted parameters as visualization (no UE scaling)
- Enables debugging mesh geometry, pose, and scale in external 3D software (Blender, MeshLab, etc.)
- Saved alongside visualization images in the same output directory

### Design Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Feature Fusion | **Cross-Attention** | Allows views to share information and understand spatial relationships |
| Camera Head Design | **N Separate Heads** | One dedicated head per canonical view position for specialization |
| Training Integration | **Separate Script** | Start with standalone script, later combine via flag-based system |
| Joint Angle Regularization | **L2 Penalty on Squared Norm** | Encourages natural poses, prevents extreme joint angles, excludes root joint |

### Recent Improvements & Bug Fixes

#### Joint Angle Regularization (2026-01-06)
- **Added**: Regularization loss to penalize large joint angles (excluding root joint)
- **Implementation**: Computes L2 norm of axis-angle representation for all non-root joints
- **Configuration**: Configurable weight in training curriculum (default: 0.001, can be reduced/disabled in later stages)
- **Files Modified**: 
  - `multiview_smil_regressor.py`: Added `joint_angle_regularization` loss computation
  - `training_config.py`: Added default weight and curriculum stages

#### Device Mismatch Fix (2026-01-06)
- **Issue**: Preprocessed images were on CPU while dummy padding images were on GPU, causing `RuntimeError` during `torch.stack()`
- **Fix**: Added `.to(self.device)` after preprocessing each image in `predict_from_multiview_batch`
- **Files Modified**: `multiview_smil_regressor.py`

#### Mesh Geometry Export (2026-01-06)
- **Added**: OBJ file export functionality for debugging mesh geometry
- **Purpose**: Enable external inspection of predicted mesh in 3D software
- **Implementation**: `export_mesh_as_obj()` function exports vertices and faces using same parameters as visualization
- **Files Modified**: `train_multiview_regressor.py`

#### GT 3D + Calibrated Camera Integration (2026-01-14)
- **Added**: Optional loading of 3D keypoints + calibrated camera parameters into the multi-view dataset.
- **Added**: `world_scale` handling so GT 3D keypoints and GT camera translations stay consistent and in the expected numerical range.
- **Added**: Camera conversion returns **vertical FOV** and **aspect ratio**; aspect ratio is threaded into the renderer (`FoVPerspectiveCameras`).
- **Added**: Multi-view losses for `keypoint_3d`, `fov`, `cam_rot`, `cam_trans`, and printing of all components in the epoch summary.
- **Files Modified/Added**:
  - `smal_fitter/sleap_data/sleap_3d_loader.py`
  - `smal_fitter/sleap_data/preprocess_sleap_multiview_dataset.py`
  - `smal_fitter/sleap_data/sleap_multiview_dataset.py`
  - `smal_fitter/neuralSMIL/multiview_smil_regressor.py`
  - `smal_fitter/neuralSMIL/train_multiview_regressor.py`

#### Critical Bugfix: Transformer Body Head Ignored Inputs (2026-01-14)
- **Issue**: In multi-view mode with `head_type='transformer_decoder'`, the body head was called as `transformer_head(features, None)`. The decoder uses `spatial_features` for cross-attention; passing `None` meant the body prediction was effectively **input-independent**, causing identical body predictions across samples.
- **Fix**: Pass the aggregated body feature as a one-token `spatial_features` sequence: `spatial_feats = features.unsqueeze(1)` and call `transformer_head(features, spatial_feats)`.
- **Files Modified**: `smal_fitter/neuralSMIL/multiview_smil_regressor.py`

#### Camera/Rendering Robustness Fixes (2026-01-14)
- **Aspect ratio propagation**:
  - **Issue**: PyTorch3D projections misaligned when `aspect_ratio` wasn’t provided (defaulted to 1).
  - **Fix**: Compute/store per-view `aspect_ratio` and pass through dataset → loss/render → renderer.
- **`aspect_ratio=None` crash**:
  - **Issue**: `FoVPerspectiveCameras` can error if `aspect_ratio=None`.
  - **Fix**: Default to ones tensor and broadcast scalars to batch shape.
- **Rasterization overflow warning**:
  - **Issue**: PyTorch3D “Bin size was too small…” coarse rasterization overflow.
  - **Fix**: Set `bin_size=0` (naive rasterization) for stability.
- **Clipping planes**:
  - **Issue**: Mesh/keypoints could be clipped by narrow `znear/zfar`.
  - **Fix**: Use wider clipping bounds (`znear=1e-3`, `zfar=1e3`) in renderer camera setup.
- **Dtype consistency**:
  - **Issue**: runtime errors from `Double` vs `Float` during rendering and camera ops.
  - **Fix**: Enforce `float32` for camera tensors, renderer inputs, and initialized tensors (`torch.zeros/ones(..., dtype=torch.float32)`), and cast faces to `long`.
- **Files Modified**:
  - `smal_fitter/p3d_renderer.py`
  - `smal_fitter/neuralSMIL/multiview_smil_regressor.py`
  - `smal_fitter/smal_fitter.py`

#### Preprocessing/CLI Fixes (2026-01-14)
- **Cropping bug**:
  - **Issue**: Preprocessor defaulted to always cropping.
  - **Fix**: Default `crop_mode` changed to `'default'` so cropping is opt-in.
- **CLI cleanup**:
  - **Issue**: Redundant flags `--load_3d_data` and `--no_3d_data`.
  - **Fix**: Keep a single disabling flag (`--no_3d_data`) while default is to load when available.
- **All numeric outputs float32**:
  - **Fix**: Ensure HDF5 saved arrays are consistently `float32` to avoid downstream dtype mismatches.
- **Files Modified**:
  - `smal_fitter/sleap_data/preprocess_sleap_multiview_dataset.py`

### Usage

```bash
# Preprocess multi-view SLEAP data
python smal_fitter/sleap_data/preprocess_sleap_multiview_dataset.py \
    /path/to/sleap/sessions multiview_sleap.h5 \
    --min_views 2

# Train multi-view model
python smal_fitter/neuralSMIL/train_multiview_regressor.py \
    --dataset_path multiview_sleap.h5 \
    --batch_size 8 \
    --num_epochs 100

# Distributed training (multi-GPU)
torchrun --nproc_per_node=4 smal_fitter/neuralSMIL/train_multiview_regressor.py \
    --dataset_path multiview_sleap.h5
```

---

*Document Version: 2.3*  
*Last Updated: 2026-01-14*  
*Author: Fabian Plum x Claude (AI Assistant)*  
*Status: IMPLEMENTED - Multi-view system with optional GT 3D + calibrated camera supervision*

**Changelog**:
- v2.3: Added GT 3D + calibrated camera supervision (world_scale + aspect_ratio), fixed transformer body-head conditioning bug, and hardened rendering (dtype, rasterization, clipping planes)
- v2.2: Added joint angle regularization, device mismatch fix, and OBJ mesh export functionality
- v2.1: Added Phase 4 (Visualization) - multi-view training progress visualization
- v2.0: Implementation complete with cross-attention fusion and separate camera heads
- v1.1: Resolved design decisions for view handling, loss weighting, and training strategy
- v1.0: Initial draft
