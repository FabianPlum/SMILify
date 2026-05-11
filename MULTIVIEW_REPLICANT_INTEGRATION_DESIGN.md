# Multi-Camera replicAnt Dataset Integration Design

## Overview

This document outlines the architecture for integrating multi-camera replicAnt datasets into SMILify's training pipeline while maintaining backward compatibility with existing single-view replicAnt dataset handling.

**Goal**: Enable loading multi-camera replicAnt data that can be:
1. Preprocessed into HDF5 format usable by multi-view training (`train_multiview_regressor.py`)
2. Sampled as single-view instances for single-view training (`train_smil_regressor.py`)
3. Stored with per-view metadata (camera parameters, view indices, frame synchronization)

---

## Current Architecture Analysis

### Single-View Data Pipeline

**Data Loading Path**: 
```
replicAnt JSON files 
  → load_SMIL_Unreal_sample() [Unreal2Pytorch3D.py]
  → replicAntSMILDataset [smil_datasets.py]
  → UnifiedSMILDataset [smil_datasets.py]
  → HDF5 preprocessing [dataset_preprocessing.py]
  → OptimizedSMILDataset [optimized_dataset.py]
  → train_smil_regressor.py
```

**Key File**: `Unreal2Pytorch3D.py`
- Parses Unreal Engine JSON format (single instance per file)
- Extracts: camera extrinsics (R, T), intrinsics (fx, fy, cx, cy), joint angles, shape betas, keypoints
- Transforms to PyTorch3D convention (mirroring, coordinate system changes)
- Currently **only handles single-view per JSON file**

**replicAntSMILDataset**:
- Inherits from `torch.utils.data.Dataset`
- `__getitem__` calls `load_SMIL_Unreal_sample()` directly
- Returns `(x_input, y_output)` tuple with image, pose, and camera data
- Supports rotation representation switching (axis_angle ↔ 6d)

### Multi-View Architecture (SLEAP)

**Multi-View Data Pipeline**:
```
SLEAP sessions (multiple cameras per frame)
  → sleap_3d_loader.py (3D ground truth)
  → preprocess_sleap_multiview_dataset.py
  → HDF5 with multiview structure
  → SLEAPMultiViewDataset [sleap_multiview_dataset.py]
  → train_multiview_regressor.py
```

**SLEAPMultiViewDataset Key Features**:
- Stores multiple views per sample in HDF5
- Variable number of views per sample (padded to `max_views`, masked via `view_mask` + `camera_indices`)
- Canonical camera ordering tracked in metadata as a JSON-encoded list of camera names
- Optional backward compatibility: `return_single_view=True` extracts one view
- Per-view: JPEG-encoded image blobs, K matrix, R, t, image size, per-keypoint visibility
- View masking: `view_mask` (boolean per slot) + `camera_indices` (which canonical camera lives in each slot)

**HDF5 Structure** (verified against `preprocess_sleap_multiview_dataset.py` + `sleap_multiview_dataset.py`):
```
/metadata/                                                  # attrs only
  canonical_camera_order: JSON-encoded list[str]            # e.g. '["CAM1", "CAM2", ...]'
  num_samples, max_views, n_joints, n_pose, n_betas
  target_resolution, backbone_name
  is_multiview=True
  has_camera_parameters=True
  world_scale=1.0

/multiview_images/
  image_jpeg_view_{v}[num_samples] (vlen uint8)             # JPEG-encoded per view slot (v=0..max_views-1)
  view_mask[num_samples, max_views]                         # bool, 1 if slot is populated

/multiview_keypoints/
  keypoints_2d[num_samples, max_views, n_joints, 2]         # Normalised [0, 1], (x, y) per existing convention
  keypoint_visibility[num_samples, max_views, n_joints]     # 1.0 if in-bounds AND ID mask > 0
  camera_indices[num_samples, max_views]                    # index into canonical_camera_order, or -1 if padded
  camera_intrinsics[num_samples, max_views, 3, 3]           # K matrix
  camera_extrinsics_R[num_samples, max_views, 3, 3]         # rotation (note: capital R)
  camera_extrinsics_t[num_samples, max_views, 3]            # translation
  image_sizes[num_samples, max_views, 2]                    # (W, H) per view
  keypoints_3d[num_samples, n_joints, 3]                    # Shared 3D keypoints in model-centred coords

/parameters/                                                # shared across views (per sample)
  global_rot[num_samples, 3]                                # axis-angle, 0 after reparameterisation
  joint_rot[num_samples, n_pose, 3]                         # axis-angle per joint
  betas[num_samples, n_betas]
  trans[num_samples, 3]                                     # 0 after reparameterisation

/auxiliary/
  has_3d_data[num_samples]                                  # bool
  has_ground_truth_betas[num_samples]                       # bool
  num_views[num_samples]                                    # int, count of valid slots
  frame_idx[num_samples]
  session_name[num_samples] (vlen str)                      # replicAnt: dataset name
  camera_names[num_samples, max_views] (vlen str)           # canonical camera name per slot
  canonical_to_world_R[num_samples, 3, 3]                   # inverse of per-frame canonical-camera-frame
  canonical_to_world_t[num_samples, 3]                      # ↳ enables round-trip to raw world coords
```

**Note on the frame convention**: All `R`, `t`, `trans`, `global_rot`, and
`keypoints_3d` arrays above are stored in the **per-frame canonical-camera
frame**, not the raw rig world frame. See "Frame Convention" section near the
bottom for the rationale and the inverse-transform fields under `/auxiliary/`.

**Critical per-camera visibility computation**:
```python
# For each camera and joint:
# 1. Check if 2D keypoint is within image bounds [0, 1]
# 2. Check if corresponding ID mask pixel (at projected location) is > 0
# 3. Set visibility[cam_idx, joint_idx] = 1.0 if both true, else 0.0

# This allows:
# - Training loss to only count visible keypoints per camera
# - Network to learn occlusion-aware predictions
# - Multi-view training to weight views based on visibility
```

---

## Multi-Camera replicAnt Data Structure

**ACTUAL Directory Layout** (C:\replicAnt-dataset-multi-cam-mice):

```
replicAnt-dataset-multi-cam-mice/
  ├── _BatchData_replicAnt-dataset-multi-cam-mice.json    # Global metadata (one per dataset)
  │   - Name: dataset name
  │   - Iterations: 10000 (number of frames)
  │   - Number of Cameras: 12
  │   - Image Resolution: {"x": 512, "y": 512}
  │   - Subject Variations: {shape betas info}
  │
  ├── replicAnt-dataset-multi-cam-mice_00000_CAM1.json       # Camera 1, Frame 0 (metadata)
  ├── replicAnt-dataset-multi-cam-mice_00000_CAM1.JPG        # Image
  ├── replicAnt-dataset-multi-cam-mice_00000_ID_CAM1.png     # ID mask (always present)
  ├── replicAnt-dataset-multi-cam-mice_00000_Depth_CAM1.png  # Depth (optional)
  │
  ├── replicAnt-dataset-multi-cam-mice_00000_CAM2.json       # Camera 2, Frame 0
  ├── replicAnt-dataset-multi-cam-mice_00000_CAM2.JPG
  ├── replicAnt-dataset-multi-cam-mice_00000_ID_CAM2.png
  ├── replicAnt-dataset-multi-cam-mice_00000_Depth_CAM2.png  # optional
  │
  ├── ... (CAM3 through CAM12)
  │
  ├── replicAnt-dataset-multi-cam-mice_00001_CAM1.json       # Camera 1, Frame 1
  ├── replicAnt-dataset-multi-cam-mice_00001_CAM1.JPG
  ├── replicAnt-dataset-multi-cam-mice_00001_ID_CAM1.png
  │
  └── ... (more frames, continuing for all 10000 iterations)
```

**Key Features**:
1. **Flat directory structure** - all files in one directory, no subdirectories
2. **Naming convention**: `{dataset_name}_{frame_idx:05d}_{tag}_CAM{camera_id}.{ext}` (tag empty for `.JPG`/`.json`)
   - Frame index: 0-padded 5-digit number (00000, 00001, ..., 09999)
   - Camera ID: 1-N (not 0-indexed)
   - File set per (frame, camera):
     - `.JPG` — RGB image (always present)
     - `.json` — per-camera metadata (always present)
     - `_ID_CAM{id}.png` — ID/visibility mask (always present)
     - `_Depth_CAM{id}.png` — depth map (optional; loaded only when `--include_depth_map` is set)
3. **Per-camera metadata** (in `.json` files):
   - Same format as single-view replicAnt JSON
   - Contains camera parameters (view matrix, FOV, location, rotation)
   - Contains subject data with keypoints (same structure as single-view)
   - Shape betas, joint angles, and root pose are **per-frame, identical across all cameras**
4. **Batch metadata** (`_BatchData_*.json`):
   - Global image resolution
   - Number of frames and cameras
   - Subject variations (for loading shape betas from file if needed)

---

## Implementation Strategy

### Phase 1: Extend Unreal2Pytorch3D.py

**New Functions** (add without breaking existing ones):

```python
def load_SMIL_Unreal_multiview_sample(
    data_path,                # Path to dataset directory
    frame_index,              # Frame number (0-padded as needed)
    camera_indices=None,      # List of camera indices to load (None = all)
    plot_tests=False,
    propagate_scaling=True,
    translation_factor=0.01,
    load_images=True,
    verbose=False
) -> Tuple[Dict, Dict]:
    """
    Load multi-view replicAnt data from a flat-directory dataset for a specific frame.
    
    Dataset structure:
    - Flat directory: {dataset_name}_{frame_idx:05d}_CAM{cam_id}.{ext}
    - Per-camera files: .json (metadata), .JPG (image), _ID_CAM{id}.png (visibility mask)
    - Synchronized frames: all cameras for same frame_idx are temporally aligned
    - Shared pose/shape: identical across all cameras (only camera params differ)
    
    Args:
        data_path: Path to dataset directory
        frame_index: Frame number to load (e.g., 0, 1, 2, ...)
        camera_indices: List of camera IDs to load (e.g., [1, 2, 3] or None for all)
        
    Returns:
        x_output: Dictionary with per-view data
            - image_paths: List[str] - paths to images per camera
            - image_data: List[np.ndarray] - loaded images, shape (H, W, 3)
            - mask_data: List[np.ndarray] - per-camera ID/visibility masks, shape (H, W)
            - mask_paths: List[str] - paths to ID map files
            - num_views: int - number of views loaded
            - camera_ids: List[int] - camera IDs in order
            
        y_output: Dictionary with shared + per-view data
            - pose_data: dict - raw pose data (from first camera)
            - joint_angles: np.ndarray[n_joints, 3] - shared joint rotations
            - joint_names: list - joint names
            - shape_betas: np.ndarray[n_betas] - shared shape
            
            - cam_rot_per_view: List[np.ndarray] - rotation matrices per camera [3, 3]
            - cam_trans_per_view: List[np.ndarray] - translation vectors per camera [3]
            - fx_per_view: List[float] - focal length X per camera
            - fy_per_view: List[float] - focal length Y per camera
            - cx_per_view: List[float] - principal point X per camera
            - cy_per_view: List[float] - principal point Y per camera
            
            - keypoints_2d_per_view: List[np.ndarray] - normalized 2D keypoints per camera
            - keypoint_visibility_per_view: List[np.ndarray] - **PER-CAMERA VISIBILITY**
                                           computed as: in_bounds AND (id_mask_pixel > 0)
    """
    # 1. Detect camera count and dataset name from _BatchData_*.json
    # 2. Determine which camera indices to load (all or subset)
    # 3. Build the per-frame canonical camera list:
    #    a. Sort the resolved camera indices ascending.
    #    b. The lowest-index resolved camera becomes canonical slot 0
    #       (defines the per-frame world frame: R=I, t=0).
    #    c. Subsequent resolved cameras fill slots 1..N-1 in order.
    # 4. From the canonical-slot-0 camera's JSON:
    #    a. Extract shared pose, shape, joint_angles via load_SMIL_Unreal_sample()
    #       (or its inner helpers) to get raw world-frame values.
    #    b. Capture (R_0, t_0) as canonical_to_world for the inverse transform.
    # 5. For each canonical slot v:
    #    a. Parse that camera's JSON (extrinsics + intrinsics).
    #    b. Load image and ID mask.
    #    c. Compute per-view keypoint visibility (see below).
    # 6. Re-express in canonical-camera frame (see "Frame Convention" section):
    #    a. Per-view extrinsics: (R_v, t_v) → relative to (R_0, t_0).
    #    b. Model trans, global_rot, keypoints_3d → re-expressed in canonical frame.
    # 7. Return consolidated multi-view data + canonical_to_world for round-trip.
```

**Key Implementation Details**:
- All output is in **canonical-camera frame** (see Frame Convention section below).
  The raw world-frame values are computed internally then transformed; only the
  transformed values plus `canonical_to_world = (R_0, t_0)` are returned.
- Per-camera parameters extracted independently from each camera's JSON.
- Per-camera ID masks loaded from `_ID_CAM{id}.png` files.
- Per-camera visibility uses the existing `compute_keypoint_visibility()` logic:
  ```python
  visibility[cam] = 1.0 if (
      keypoint within image bounds [0, 1] AND
      id_mask[int(norm_y * H), int(norm_x * W)] > 0
  ) else 0.0
  ```
- ID mask files follow pattern: `{dataset_name}_{frame_idx:05d}_ID_CAM{cam_id}.png`.

**Backward Compatibility**: 
- Keep `load_SMIL_Unreal_sample()` unchanged
- Add `load_SMIL_Unreal_multiview_sample()` as new function
- Both work with existing code paths

---

### Phase 2: Create MultiViewreplicAntDataset

**New Class** in `smil_datasets.py`:

```python
class MultiViewreplicAntSMILDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for multi-camera replicAnt SMIL data.
    
    Loads all synchronized views for each frame from a flat-directory replicAnt dataset.
    Handles variable number of cameras and optional camera subsetting.
    """
    
    def __init__(self, 
                 data_path,                    # Path to replicAnt-dataset-multi-cam-mice
                 use_ue_scaling=True,
                 rotation_representation='axis_angle',
                 backbone_name='resnet152',
                 camera_subset=None,           # List of camera IDs to load (e.g., [1,2,3])
                 min_views_per_sample=2):
        """
        Initialize multi-view replicAnt dataset.
        
        Args:
            data_path: Path to flat-directory replicAnt dataset
            camera_subset: None (load all) or list of camera IDs (1-based)
        """
        self.data_path = data_path
        self.use_ue_scaling = use_ue_scaling
        self.rotation_representation = rotation_representation
        self.backbone_name = backbone_name
        self.camera_subset = camera_subset
        self.min_views_per_sample = min_views_per_sample
        
        # Load batch metadata to determine frame count and camera count
        self._detect_dataset_structure()
        
        # Detect input resolution from batch data file
        self.original_resolution = self._detect_input_resolution()
        
        # Determine target resolution based on backbone
        from backbone_factory import BackboneFactory
        self.target_resolution = BackboneFactory.get_default_input_resolution(backbone_name)
    
    def _detect_dataset_structure(self):
        """
        Scan for batch metadata file to determine frame count and camera count.
        Also verify that all cameras exist for all frames.
        """
        # Find _BatchData_*.json file
        # Extract: num_iterations (frames), num_cameras
        # Verify camera IDs 1-N exist for frame 0
        
    def _detect_input_resolution(self):
        """Detect input resolution from batch metadata."""
        # Load _BatchData_*.json and extract Image Resolution
        
    def __getitem__(self, idx):
        """
        Returns:
            x_output: Dict with per-view images and masks
            y_output: Dict with shared pose + per-view camera parameters
        """
        x, y = load_SMIL_Unreal_multiview_sample(
            data_path=self.data_path,
            frame_index=idx,
            camera_indices=self.camera_subset,
            propagate_scaling=True,
            translation_factor=0.01,
            load_images=True,
            verbose=False
        )
        
        # Convert rotation representations if needed
        if self.rotation_representation == '6d':
            # Convert root rotation and joint angles
            if 'root_rot' in y:
                y['root_rot'] = axis_angle_to_rotation_6d(y['root_rot'])
            if 'joint_angles' in y:
                y['joint_angles'] = axis_angle_to_rotation_6d(y['joint_angles'])
        
        return x, y
        
    def __len__(self):
        return self.num_frames
```

---

### Phase 3: Preprocessing to HDF5

**New module**: `smal_fitter/sleap_data/preprocess_replicant_multiview_dataset.py` —
sibling of `preprocess_sleap_multiview_dataset.py`, with the same output schema so
`SLEAPMultiViewDataset` and `train_multiview_regressor.py` consume it unchanged.

**Class**: `replicAntMultiViewPreprocessor`

```python
class replicAntMultiViewPreprocessor:
    def __init__(
        self,
        target_resolution: int = 224,
        backbone_name: str = 'vit_large_patch16_224',
        jpeg_quality: int = 95,
        chunk_size: int = 8,
        compression: str = 'gzip',
        compression_level: int = 6,
        frame_skip: int = 1,
        camera_subset: Optional[List[int]] = None,
        include_depth_map: bool = False,        # gate for /multiview_depth/ group
        propagate_scaling: bool = True,
        translation_factor: float = 0.01,
        debug: bool = False,
    ):
        ...

    def preprocess(self, input_dir: str, output_hdf5: str, num_workers: int = 8):
        """Produce an HDF5 file that matches the SLEAPMultiViewDataset schema."""
```

**Output schema**: identical to the schema documented in the SLEAP overview above
(`/metadata`, `/multiview_images`, `/multiview_keypoints`, `/parameters`,
`/auxiliary`). `canonical_camera_order` is `["CAM1", "CAM2", ...]` derived from
the first scanned frame.

**Image storage** (matches SLEAP preprocessor):
- Per-view images are **JPEG-encoded in the HDF5** as variable-length `uint8` arrays
  under `/multiview_images/image_jpeg_view_{v}` (one dataset per view slot).
- `jpeg_quality` defaults to 95 (same as SLEAP path).
- This keeps the 12-cam × 10k-frame replicAnt clip in the ~5–10 GB range rather
  than ~90 GB raw.

**Failure handling**:
- Frames that fail to load any view (corrupt JSON, missing files, projection
  failure, etc.) are **flagged and skipped**: the preprocessor logs the
  `(frame_idx, reason)` and continues with the next `idx`. A summary count is
  printed at the end and stored in `/metadata` attrs (`num_skipped_frames`,
  `skipped_frame_indices`).
- Per-camera failures within an otherwise-loadable frame mark just that view's
  `view_mask[i, v] = False` and `camera_indices[i, v] = -1`; the frame is kept
  as long as ≥1 view loads.

**Optional depth (`--include_depth_map`)**:
- When set, an additional `/multiview_depth/` group is written:
  ```
  /multiview_depth/
    depth_png_view_{v}[num_samples] (vlen uint8)    # PNG-encoded raw depth
    depth_mask[num_samples, max_views]              # 1 if depth present, 0 otherwise
  ```
- When unset (default), depth files are ignored even if present on disk.
- Future self-occlusion visibility refinement will read this group when
  available; the trainer does not require it.

---

### Phase 4: Unified Dataset Loading

**Update**: `UnifiedSMILDataset.from_path()` in `smil_datasets.py`

```python
@staticmethod
def from_path(data_path, **kwargs):
    """
    Auto-detect dataset type from a path.

    Detection (in order):
    1. .h5 / .hdf5  → inspect /metadata.is_multiview
                       True  → SLEAPMultiViewDataset
                       False → OptimizedSMILDataset
    2. Directory containing _BatchData_*.json and at least one
       {dataset}_00000_CAM*.json → MultiViewreplicAntSMILDataset
    3. Directory containing _BatchData_*.json with {dataset}_*.json
       (no _CAM suffix) → replicAntSMILDataset (single-view, current behaviour)
    """
```

The multiview replicAnt PyTorch `Dataset` is only used directly for tests and
ad-hoc validation; **production training reads the HDF5** produced by Phase 3
through `SLEAPMultiViewDataset`. The PyTorch class is therefore primarily a
test seam, not a hot path.

---

### Phase 5: Single-View Sampling — Deferred to Training Time

**Decision**: The multiview HDF5 is **dataset-shape-agnostic** — it always stores
all available views per frame. Whether downstream training is single-view or
multi-view is a *training-time* concern, not a dataset concern.

For multi-view training: feed the HDF5 directly to `SLEAPMultiViewDataset`
(via the `UnifiedSMILDataset` factory) → `train_multiview_regressor.py`.

For single-view training: a thin sampler will be added later that, per sample,
draws one view slot at random from the populated ones and emits the same dict
shape that `replicAntSMILDataset.__getitem__` produces today. **This sampler
is out of scope for the initial PR**; it can be implemented incrementally on
top of `SLEAPMultiViewDataset.return_single_view`.

---

## Data Format Compatibility

### Single-View Training Compatibility

The future single-view sampler must emit the return contract produced today by
`load_SMIL_Unreal_sample()` in [Unreal2Pytorch3D.py:694](smal_fitter/Unreal2Pytorch3D.py#L694):

```python
x_output = {
    'input_image': str,                 # Path to image file
    'input_image_data': np.ndarray,     # Image data (H, W, 3) uint8
    'input_image_mask': np.ndarray,     # Binary mask (H, W) or None
}

y_output = {
    'pose_data': dict,                  # Raw pose dict
    'joint_angles': np.ndarray,         # (n_joints, 3) or (n_joints, 6) after 6D
    'joint_names': list,                # config.dd["J_names"]
    'cam_rot': torch.Tensor,            # (3, 3) — reparameterised (see Phase 1)
    'cam_trans': torch.Tensor,          # (3,)  — reparameterised
    'cam_rot_orig': np.ndarray,         # (3, 3) raw Unreal extrinsics
    'cam_trans_orig': np.ndarray,       # (3,)  raw Unreal extrinsics
    'cx': float, 'cy': float,           # principal point (px)
    'fx': float, 'fy': float,           # focal length  (px)
    'cam_fov': list[float],             # [FOV degrees]
    'scale_weights': np.ndarray|None,
    'trans_weights': np.ndarray|None,
    'shape_betas': np.ndarray,          # (n_betas,)
    'root_loc': np.ndarray,             # (3,) — zero after reparam
    'root_rot': np.ndarray,             # (3,) — zero after reparam
    'keypoints_2d': np.ndarray,         # (n_joints, 2) normalised [0, 1]
    'keypoint_visibility': np.ndarray,  # (n_joints,) {0.0, 1.0}
    'keypoints_3d': np.ndarray,         # (n_joints, 3) model-centred
    'keypoints_3d_original': np.ndarray,# (n_joints, 3) world frame, debug
    'translation_factor': float,
    'propagate_scaling': bool,
}
```

### Multi-View Training Compatibility

Output must match the dict returned by `SLEAPMultiViewDataset.__getitem__`
(verified at [sleap_multiview_dataset.py:257](smal_fitter/sleap_data/sleap_multiview_dataset.py#L257)):

```python
y_data = {
    'keypoints_2d': np.ndarray,         # (num_views, n_joints, 2)
    'keypoint_visibility': np.ndarray,  # (num_views, n_joints)
    'camera_intrinsics': np.ndarray,    # (num_views, 3, 3) — K
    'camera_extrinsics_R': np.ndarray,  # (num_views, 3, 3) — capital R
    'camera_extrinsics_t': np.ndarray,  # (num_views, 3)    — t (world_scale applied)
    'keypoints_3d': np.ndarray,         # (n_joints, 3) when has_3d_data
    # shared pose/shape (from /parameters/):
    'global_rot': np.ndarray,           # (3,)   axis-angle (0 after reparam)
    'joint_rot': np.ndarray,            # (n_pose, 3) or (n_pose, 6)
    'betas': np.ndarray,                # (n_betas,)
    'trans': np.ndarray,                # (3,)   (0 after reparam)
}
```

---

## Key Design Decisions

### 1. **Dataset Layout Detection**
- replicAnt multi-camera datasets are **flat directories** (no per-frame subdirs);
  files follow `{dataset_name}_{frame:05d}_(ID_|Depth_)?CAM{id}.{ext}`.
- Detection: presence of `_BatchData_*.json` plus at least one
  `{dataset_name}_00000_CAM*.json` distinguishes from single-view replicAnt data
  (which has only `{dataset_name}_*.json` without the `_CAM` suffix).
- **Decision**: Flat-directory, name-pattern detection.

### 1b. **Per-Frame Failure Handling**
- If any frame index fails to load (corrupt JSON, missing camera file, etc.),
  log it and continue with the next `idx`. Do not abort the whole preprocessing
  run. The skipped indices are recorded in `/metadata` attrs.
- Per-camera failures within a frame mark just that view slot
  (`view_mask=False`); the frame is still emitted as long as ≥1 view loads.

### 2. **Camera Indexing**
- Store as list of views in HDF5 (order preserved)
- Track canonical camera order in metadata for consistent training
- Support optional subset selection during loading (camera_subset parameter)
- **Decision**: Preserve order, allow optional filtering

### 3. **Shared vs. Per-View Data**
- **Shared** (same for all cameras): pose (joint angles, global rotation), shape (betas)
- **Per-View**: images, masks, camera parameters (R, T, K), keypoints (projected from 3D)
- **Decision**: Store separately in HDF5, merge in dataset `__getitem__`

### 4. **Variable View Count Handling**
- No padding; store variable number of views per sample
- Use `valid_views_mask` to track which views are real vs. padding (if needed)
- Training code uses masking to ignore invalid views
- **Decision**: Match SLEAP multiview approach (no padding, masking in loss)

### 5. **Backward Compatibility**
- Ensure single-view loading still works for existing replicAnt JSON data
- Make multiview optional (can load as single-view if needed)
- **Decision**: Keep all existing code unchanged, add new functions/classes

---

## Validation Checklist

✅ **VERIFIED** - Actual dataset structure confirmed:

- ✅ Flat directory structure with {dataset_name}_{frame_idx:05d}_CAM{id}.{ext} naming
- ✅ 12 cameras (CAM1 through CAM12) in dataset
- ✅ 10000 frames (iterations) in dataset  
- ✅ Each camera has `.json` metadata file (Unreal format)
- ✅ Each camera has `.JPG` image file
- ✅ Depth maps exist as `_Depth_CAM{id}.png` files
- ✅ Batch metadata: `_BatchData_replicAnt-dataset-multi-cam-mice.json`
  - Image resolution: 512x512
  - Number of cameras: 12
  - Subject variations with shape betas
- ✅ JSON files contain same structure as single-view replicAnt:
  - Camera parameters (view matrix, FOV, location, rotation)
  - Subject data with keypoints (2D and 3D positions)
  - Shape betas (same across all cameras per frame)
  - Quaternion-based rotations

---

## Files to Create/Modify

### New Files:
1. `smal_fitter/sleap_data/preprocess_replicant_multiview_dataset.py` —
   `replicAntMultiViewPreprocessor`, writes the SLEAP-compatible HDF5 schema.
2. `smal_fitter/neuralSMIL/multiview_replicant_dataset.py` —
   `MultiViewreplicAntSMILDataset` (test seam; thin wrapper around
   `load_SMIL_Unreal_multiview_sample`).
3. CLI entry, either:
   - extend an existing `preprocess_dataset.py`, or
   - add `smal_fitter/sleap_data/preprocess_replicant_multiview_dataset.py`'s
     own `__main__` block (matches the SLEAP equivalent).

### Modify Existing Files:
1. `smal_fitter/Unreal2Pytorch3D.py` — already contains the Phase 1 draft of
   `load_SMIL_Unreal_multiview_sample()` (uncommitted). Needs cleanup +
   reparameterisation parity with `load_SMIL_Unreal_sample()` (see Open Items).
2. `smal_fitter/neuralSMIL/smil_datasets.py` — extend
   `UnifiedSMILDataset.from_path()` with the flat-directory `_CAM*.json`
   detection branch.

### No Changes Needed:
- `train_multiview_regressor.py`, `multiview_smil_regressor.py`,
  `sleap_multiview_dataset.py` — Phase 3 produces an HDF5 that already
  matches the SLEAP schema.
- `train_smil_regressor.py`, `smil_image_regressor.py` — touched only when the
  single-view sampler is added later.

---

## Implementation Order

1. **Step 1**: Clean up the Phase 1 draft in `Unreal2Pytorch3D.py`
   (deduplicate imports, add coord-swap comment, add reparameterisation —
   see Open Items).
2. **Step 2**: Add `MultiViewreplicAntSMILDataset` (test seam).
3. **Step 3**: Implement `replicAntMultiViewPreprocessor` + CLI (matches
   SLEAP HDF5 schema; JPEG-encoded images; optional `--include_depth_map`).
4. **Step 4**: Extend `UnifiedSMILDataset.from_path()` auto-detection.
5. **Step 5**: Round-trip tests
   (load → preprocess → SLEAPMultiViewDataset → assert tensor shapes/contents).
6. **Step 6**: Example multiview config under
   `smal_fitter/neuralSMIL/configs/examples/`.

---

## Frame Convention: Canonical-Camera-Frame Storage

Decision (informed by the SLEAP path's choice + the procedural nature of
replicAnt multi-cam data):

- **Do NOT use the existing single-view "model-at-origin / camera-absorbs-everything"
  reparameterisation** for multi-view storage.
- **Do NOT use the raw rig world frame** either — procedural generation places
  the rig at arbitrary world positions per frame, so absolute world coordinates
  carry only procgen noise and the empirical distribution of `cam_trans`,
  `cam_rot`, and model `trans` becomes too wide for stable regression.
- **Use the canonical camera's frame as the per-frame world origin**:
  - For each frame, identify the lowest-index camera that successfully resolves
    on disk. That camera becomes **canonical index 0** for that frame, and
    becomes the world frame origin (`R = I`, `t = 0`).
  - All other cameras are renumbered into the canonical order they appear
    (the next-lowest available index becomes 1, etc.). The mapping is recorded
    in `/auxiliary/camera_names[i, v]` and `/multiview_keypoints/camera_indices`.
  - Express **all** other quantities (other cameras' extrinsics, model
    `global_rot`, model `trans`, `keypoints_3d`) relative to the canonical
    camera's frame.

This convention:
- Bounds the empirical distributions across procedural variation (every frame
  "looks at the rig" from the same notional viewpoint at the input distribution
  level).
- Keeps `trans` and `global_rot` as learnable, bounded signals (unlike convention
  C where `trans` is degenerate-zero).
- Preserves multi-view relative geometry exactly — the inter-camera transforms
  and 2D reprojections are byte-equivalent to the raw world-frame version.
- Has a clean bridge to single-view sampling: one composition folds canonical
  cam → selected cam → model-at-origin, producing samples byte-equivalent to
  the existing single-view `load_SMIL_Unreal_sample()` output. (Out of scope
  for this PR, but the design supports it.)

### Inverse transform stored for validation

The canonical-camera-frame conversion is reversible. Store the per-frame
canonical-camera-to-world rigid transform under `/auxiliary/canonical_to_world`
(`R: (num_samples, 3, 3)`, `t: (num_samples, 3)`) so:
- Tests can validate that round-tripping reproduces the raw extrinsics.
- Downstream tools that *do* want absolute world coords (rendering at a
  specific scene location, debugging procgen, etc.) can recover them.
- Training pipeline ignores this field; it's metadata only.

For procedural replicAnt scenes the absolute world coordinates carry no
meaningful information — they're arbitrary draws from the procgen randomiser
— so the trainer not using this field is the desired behaviour.

### SLEAP path: apply the same normalisation

The SLEAP preprocessor currently stores camera extrinsics in the rig's
calibration-tool-defined world frame. To keep both paths producing
byte-equivalent on-disk conventions:

- **TODO (separate change, not in this PR)**: apply canonical-camera-frame
  normalisation in `preprocess_sleap_multiview_dataset.py` too. For SLEAP
  this is a no-op for learning (rig is fixed, so it's a constant shift) but
  makes the data format symmetric across the two paths.

### Open empirical question (not blocking implementation)

Past SLEAP multiview training did **not** apply direct supervision to the
`trans` head. The `/parameters/trans = 0` placeholder in the HDF5 was never
consumed: the dataset returns `root_loc = None`
([sleap_multiview_dataset.py:324](smal_fitter/sleap_data/sleap_multiview_dataset.py#L324)),
which sets `trans_mask = False`
([multiview_smil_regressor.py:2002-2011](smal_fitter/neuralSMIL/multiview_smil_regressor.py#L2002-L2011)),
which short-circuits the loss to an `eps` placeholder
([multiview_smil_regressor.py:892-902](smal_fitter/neuralSMIL/multiview_smil_regressor.py#L892-L902)).
The head has therefore only ever received **implicit gradient** through the
2D reprojection loss (which depends on `trans`).

For replicAnt multi-cam with canonical-camera-frame storage, `root_loc` will
be a real value (model's position relative to canonical camera), so direct
MSE supervision on the `trans` head turns on for the first time. This is
purely **additive** supervision on a head that already had implicit
gradient — not a "relearn" situation — so the risk is low.

- **Step 0 before any full preprocess run**: preprocess a small subset
  (~100–500 frames) of the replicAnt multi-cam dataset in canonical-cam-frame
  convention, run a short training smoke test. Primary validation goal: confirm
  the canonical-camera-frame transformation produces geometrically consistent
  reprojections (i.e., per-view reprojection loss converges normally —
  this catches frame-convention bugs more reliably than testing the trans
  head specifically). Secondary: confirm direct `trans` loss converges
  without destabilising the rest of training.
- Default `trans` loss weight (0.001) is a reasonable starting point. If
  needed, options are: bump the weight, freeze the head briefly, or zero
  `trans` in the stored data (degrading to no direct supervision, matching
  past SLEAP behaviour).

## Phase 1 cleanup (separate from frame convention)

- Remove duplicate `from pathlib import Path` / `from typing import ...` /
  `import re` lines inside the function body.
- Add a comment near the `norm_x = ...['y'] / image_height` block flagging
  that the x↔y swap matches the existing single-view convention.
- Replace the current raw-extrinsics emission with canonical-camera-frame
  output per the convention above. The function will compute the canonical
  camera's `(R_0, t_0)` from the first resolved-on-disk camera index, then
  re-express every per-view `(R_v, t_v)`, model `trans`, model `global_rot`,
  and `keypoints_3d` relative to that frame. Store `canonical_to_world` as
  `(R_0, t_0)` for round-trip validation.
