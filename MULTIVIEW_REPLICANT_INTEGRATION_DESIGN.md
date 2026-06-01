# Multi-Camera replicAnt Dataset Integration Design

## Overview

This document outlines the architecture for integrating multi-camera replicAnt datasets into SMILify's training pipeline while maintaining backward compatibility with existing single-view replicAnt dataset handling.

**Goal**: Enable loading multi-camera replicAnt data that can be:
1. Preprocessed into HDF5 format usable by multi-view training (`train_multiview_regressor.py`)
2. Sampled as single-view instances for single-view training (`train_smil_regressor.py`)
3. Stored with per-view metadata (camera parameters, view indices, frame synchronization)

---

## Status (2026-06-01)

| Phase | What | State |
|---|---|---|
| 1 | `load_SMIL_Unreal_multiview_sample` loader | **COMPLETE** |
| 3 | HDF5 preprocessor (`replicAntMultiViewPreprocessor`) | **NEXT** |
| 4 | `UnifiedSMILDataset.from_path()` auto-detection | pending |
| 2 | `MultiViewreplicAntSMILDataset` PyTorch test seam | pending, low priority |
| 5 | Single-view sampler | deferred per design |
| — | Step-0 smoke test (preprocess + short training) | follows Phase 3 |

**Phase 1 highlights** (see commit history on `multiview-replicant-integration`):
- Canonical-camera-frame storage (lowest CAM ID → R=I, t=0), forward-transform `canonical_to_world` for round-trip
- Depth-buffer self-occlusion visibility refinement at load time (commit `eb5b5f8`)
- Explicit `keypoint_in_dataset_per_view` bitmap so model-only joints stay invisible regardless of where the `[0, 0]` sentinel lands (commit `600f913`)
- Diagnostics: `tests/validate_multiview_replicant_loader.py` (`--compare_frames`, `--named_overlay`) and `smal_fitter/replicAnt_data/visualize_multiview_depth_occlusion.py`
- Empirical: 0.00 px reprojection error on 12 cameras (frame 0), 7.6e-06 max abs round-trip on 3D keypoints, canonical-cam extrinsics R=I and t=0 to numerical precision, 328 → 259 keypoints kept after depth refinement on frame 100 (mouse model)

**Important architectural shift from the original draft** — depth handling: the depth-buffer self-occlusion check runs at load time inside the loader, not as a post-process on stored bytes. The HDF5 schema therefore does NOT include a `/multiview_depth/` group, and the preprocessor does NOT take an `--include_depth_map` flag. Depth parameters are pass-through CLI args, recorded in `/metadata` attrs for reproducibility. See §Phase 3 below.

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
  canonical_to_world_R[num_samples, 3, 3]                   # = R_0; FORWARD world->canonical transform
  canonical_to_world_t[num_samples, 3]                      # = t_0; invert as (x_can - t_0) @ R_0.T to recover raw world
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
     - `_Depth_CAM{id}.png` — depth map (loaded by the multi-view loader when `depth_occlusion_check=True`; falls back silently to ID-mask-only visibility if missing)
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

### Phase 1: `load_SMIL_Unreal_multiview_sample` — COMPLETE

Implemented in `smal_fitter/Unreal2Pytorch3D.py`. The original draft is gone; this section documents the **landed API** for downstream consumers (Phase 3 preprocessor, validation diagnostics).

**Signature**:

```python
def load_SMIL_Unreal_multiview_sample(
    data_path: str,
    frame_index: int,
    camera_indices: list = None,           # None -> all cameras present for that frame
    propagate_scaling: bool = True,
    translation_factor: float = 0.01,
    load_images: bool = True,
    canonical_frame: bool = True,          # canonical-camera-frame storage on by default
    depth_occlusion_check: bool = True,    # depth-buffer self-occlusion refinement
    depth_max_cm: float = 1000.0,
    depth_tolerance_cm: float = 5.0,
    depth_neighborhood: int = 1,           # half-window, so 1 => 3x3 patch
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
```

**Behaviour**:
- Auto-detects `dataset_name` and frame count from `_BatchData_*.json`.
- Builds canonical camera ordering from `sorted(camera_indices)`; the lowest-numbered camera present this frame becomes canonical slot 0.
- Maps each camera's keypoints into model `J_names` order; tracks an in-dataset bitmap so model-only joints (no dataset GT) stay invisible regardless of mask content.
- Per-view visibility = `in_dataset AND in_bounds AND id_mask_pass AND depth_pass` (depth term skipped when `depth_occlusion_check=False` or the `_Depth_CAM{id}.png` file is missing).
- When `canonical_frame=True` (default), re-expresses all camera extrinsics, model `root_loc`, `root_rot`, and `keypoints_3d` relative to canonical cam 0. Emits `canonical_to_world = (R_0, t_0)` as the forward transform; the inverse `x_world = (x_can - t_0) @ R_0.T` recovers the original world frame.

**`x_output` schema**:
```python
{
    "image_data":         List[np.ndarray | None],  # V x (H, W, 3) uint8 (None when load_images=False)
    "image_paths":        List[str],                 # V
    "input_image_mask":   List[np.ndarray | None],   # V x (H, W) binary id mask
    "mask_paths":         List[str],                 # V
    "depth_paths":        List[str],                 # V — file paths only, no in-memory depth
    "num_views":          int,
    "camera_ids":         List[int],                 # sorted ascending; index 0 == canonical
}
```

**`y_output` schema** (shared fields first, then per-view):
```python
{
    # Shared across all cameras
    "shape_betas":        np.ndarray,                # (n_betas,)
    "joint_angles":       np.ndarray,                # (n_joints, 3) axis-angle, J_names order
    "joint_names":        list,                      # config.dd["J_names"]
    "root_loc":           np.ndarray,                # (3,)   canonical-frame translation
    "root_rot":           np.ndarray,                # (3,)   canonical-frame axis-angle
    "scale_weights":      np.ndarray | None,         # PCA scale weights (if present in JSON)
    "trans_weights":      np.ndarray | None,         # PCA trans weights (if present in JSON)
    "translation_factor": float,
    "propagate_scaling":  bool,
    "pose_data":          dict,                      # raw cam-0 keypoints dict (back-compat)
    "keypoints_3d":       np.ndarray,                # (n_joints, 3) canonical frame, J_names order
    "keypoints_3d_world": np.ndarray,                # (n_joints, 3) raw PyTorch3D-mirrored world frame
    "canonical_to_world": (np.ndarray, np.ndarray),  # (R_0 (3,3), t_0 (3,)) forward transform
    "canonical_cam_id":   int,                       # canonical cam ID, or -1 when canonical_frame=False

    # Per-view (V entries each)
    "keypoints_2d_per_view":        List[np.ndarray],   # V x (n_joints, 2) [y/H, x/W] axis-swapped
    "keypoint_visibility_per_view": List[np.ndarray],   # V x (n_joints,) float {0, 1}
    "keypoint_in_dataset_per_view": List[np.ndarray],   # V x (n_joints,) bool — has dataset GT
    "cam_rot_per_view":             List[torch.Tensor], # V x (3, 3) row-vector world->cam
    "cam_trans_per_view":           List[torch.Tensor], # V x (3,)
    "fx_per_view", "fy_per_view":   List[float],        # V
    "cx_per_view", "cy_per_view":   List[float],        # V
    "fov_per_view":                 List[float],        # V (degrees, from camera JSON)
}
```

**Conventions to be aware of when consuming this**:
- 2D keypoints are stored with an **intentional axis swap**: `kp[:, 0] = 2DPos.y / H`, `kp[:, 1] = 2DPos.x / W`. Downstream ID-mask and depth lookups depend on this. Don't "fix" without updating both sides.
- All extrinsics and 3D points are in **PyTorch3D-mirrored** convention (x-axis negated vs raw Unreal). Projection: `u = -fx * X_cam.x / X_cam.z + cx`, `v = -fy * X_cam.y / X_cam.z + cy`.
- `keypoint_in_dataset_per_view` is True iff the joint name appears in that camera's `keypoints` dict. Use it to distinguish "missing from dataset" from "present but culled by ID/depth".

**Backward compatibility**: `load_SMIL_Unreal_sample()` (single-view) is unchanged.

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
        # The loader defaults (canonical_frame=True, depth_occlusion_check=True)
        # match what the preprocessor uses, so production-side training stays
        # byte-equivalent to direct-loader sampling for tests.
        x, y = load_SMIL_Unreal_multiview_sample(
            data_path=self.data_path,
            frame_index=idx,
            camera_indices=self.camera_subset,
            propagate_scaling=True,
            translation_factor=0.01,
            load_images=True,
        )

        if self.rotation_representation == '6d':
            if 'root_rot' in y:
                y['root_rot'] = axis_angle_to_rotation_6d(y['root_rot'])
            if 'joint_angles' in y:
                y['joint_angles'] = axis_angle_to_rotation_6d(y['joint_angles'])

        return x, y

    def __len__(self):
        return self.num_frames
```

---

### Phase 3: HDF5 preprocessor — NEXT

**New module**: `smal_fitter/replicAnt_data/preprocess_replicant_multiview_dataset.py`. Lives alongside `visualize_multiview_depth_occlusion.py` in the `replicAnt_data` package. (NOT in `sleap_data/` — the SLEAP and replicAnt code paths are siblings, not nested.)

**Output schema**: identical to `SLEAPMultiViewDataset`'s expected HDF5 layout (`/metadata`, `/multiview_images`, `/multiview_keypoints`, `/parameters`, `/auxiliary` — see the SLEAP overview section above for the full table). The trainer (`train_multiview_regressor.py`) consumes both paths interchangeably.

`canonical_camera_order` in `/metadata` attrs is `["CAM1", "CAM2", ...]` derived from `sorted(camera_indices)` on the first scanned frame.

**Class**:

```python
class replicAntMultiViewPreprocessor:
    def __init__(
        self,
        target_resolution: int = 512,         # store at native; dataset class resizes at train time
        backbone_name: str = "vit_large_patch16_224",
        jpeg_quality: int = 95,
        chunk_size: int = 8,
        compression: str = "gzip",
        compression_level: int = 6,
        frame_skip: int = 1,
        camera_subset: Optional[List[int]] = None,
        # Depth visibility-refinement passthrough to the loader.
        # Defaults match load_SMIL_Unreal_multiview_sample.
        depth_occlusion_check: bool = True,
        depth_max_cm: float = 1000.0,
        depth_tolerance_cm: float = 5.0,
        depth_neighborhood: int = 1,
        propagate_scaling: bool = True,
        translation_factor: float = 0.01,
        debug: bool = False,
    ):
        ...

    def preprocess(self, input_dir: str, output_hdf5: str,
                   num_workers: int = 8,
                   max_frames: Optional[int] = None) -> None:
        """Produce an HDF5 matching the SLEAPMultiViewDataset schema."""
```

**Image storage**:
- `/multiview_images/image_jpeg_view_{v}` as `vlen uint8` (JPEG-encoded). `jpeg_quality=95`.
- Stored at the **native 512x512** of the replicAnt source. The dataset class resizes to backbone input size at training time. This keeps the HDF5 backbone-agnostic at the cost of ~2x disk versus pre-resizing to 224. Re-evaluate if disk pressure becomes real.
- 12 cams x 10k frames lands in the ~5-10 GB range vs ~90 GB raw.

**Depth handling (architectural shift vs original draft)**:
- Depth-buffer self-occlusion is applied at **load time** by `load_SMIL_Unreal_multiview_sample` (Phase 1). The HDF5 stores depth-refined visibility under `/multiview_keypoints/keypoint_visibility`.
- **No `/multiview_depth/` group is written. No `--include_depth_map` flag exists.** Raw depth PNGs are not persisted in the HDF5.
- The preprocessor passes the four depth knobs straight through to the loader and records the values used as `/metadata` attrs so the visibility computation is reproducible:
  ```
  /metadata attrs (depth additions vs SLEAP schema):
    depth_occlusion_check: bool
    depth_max_cm:          float
    depth_tolerance_cm:    float
    depth_neighborhood:    int
  ```
- **Implication**: changing the depth thresholds requires re-preprocessing. If a future caller needs runtime-tunable thresholds, the lowest-friction option is to add the `/multiview_depth/` group at that point and refresh from stored bytes. Keep the schema minimal until that need is proven.

**Failure handling**:
- Per-frame: load failures (corrupt JSON, missing image, projection failure, etc.) are caught, logged as `(frame_idx, reason)`, and skipped. A summary count plus the skipped index list are written to `/metadata` attrs (`num_skipped_frames`, `skipped_frame_indices`).
- Per-camera within a kept frame: failures mark just that view's `view_mask[i, v] = False` and `camera_indices[i, v] = -1`. Frame kept as long as >= 1 view loads.
- HDF5 uses **contiguous sample slots**: `num_samples = number of successfully preprocessed frames`. The mapping back to source frame numbers is via `/auxiliary/frame_idx[sample_idx]`. There are no gaps in the sample axis.

**Parallelism**:
- `concurrent.futures.ProcessPoolExecutor` with default 8 workers, one frame per task.
- Each worker calls `load_SMIL_Unreal_multiview_sample`, JPEG-encodes the per-view images, returns a typed dict to the main process.
- Main process drains the result queue in input order and writes to HDF5 sequentially (HDF5 is not thread-safe).
- Workers must call `apply_smal_file_override(smal_file)` and `import config` at init, then `from Unreal2Pytorch3D import load_SMIL_Unreal_multiview_sample`. Use a top-level worker `initializer` function passed to the ProcessPoolExecutor — this is how `apply_smal_file_override` flows into each child.

**Streaming HDF5 write**:
- Pre-allocate all fixed-shape datasets at `num_samples_alloc = ceil(num_input_frames / frame_skip)`; resize down at the end after counting actual successes.
- Apply `chunks=(chunk_size, ...)` and `compression="gzip"`, `compression_opts=6` on every dataset.
- The vlen `uint8` JPEG datasets use `dtype=h5py.vlen_dtype(np.uint8)` and grow by per-sample element assignment.

**CLI** (inside `__main__`):
```
--input_dir          (required)  flat-directory replicAnt dataset
--output_hdf5        (required)
--smal_file                       SMAL/SMIL .pkl matching the dataset's skeleton
--shape_family                    optional
--num_workers        8
--jpeg_quality       95
--chunk_size         8
--frame_skip         1
--camera_subset                   optional, list of ints
--max_frames                      optional; for Step-0 smoke testing
--depth_occlusion_check           default True; --no_depth_occlusion_check to disable
--depth_max_cm       1000.0
--depth_tolerance_cm 5.0
--depth_neighborhood 1
--debug              flag
```

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

### New Files (pending):
1. `smal_fitter/replicAnt_data/preprocess_replicant_multiview_dataset.py` — `replicAntMultiViewPreprocessor` + `__main__` CLI. **Phase 3, NEXT.**
2. `smal_fitter/neuralSMIL/multiview_replicant_dataset.py` — `MultiViewreplicAntSMILDataset` (Phase 2, test seam only; production reads HDF5).

### Modified Files:
1. `smal_fitter/Unreal2Pytorch3D.py` — **Phase 1 complete.** Multi-view loader with canonical-frame storage and depth visibility refinement. See §Status for commit references.
2. `smal_fitter/replicAnt_data/__init__.py` — created during Phase 1's depth work; advertises the planned preprocessor.
3. `smal_fitter/replicAnt_data/visualize_multiview_depth_occlusion.py` — created during Phase 1; per-view depth-occlusion diagnostic.
4. `smal_fitter/neuralSMIL/smil_datasets.py` — **Phase 4**: extend `UnifiedSMILDataset.from_path()` with the flat-directory `_CAM*.json` detection branch.

### No Changes Needed:
- `train_multiview_regressor.py`, `multiview_smil_regressor.py`, `sleap_multiview_dataset.py` — Phase 3 produces an HDF5 that already matches the SLEAP schema.
- `train_smil_regressor.py`, `smil_image_regressor.py` — single-view sampler is Phase 5 (deferred).

---

## Implementation Order

1. **Step 1** — Phase 1 loader: canonical-frame storage, depth visibility refinement, in-dataset bitmap. **DONE** (see §Status).
2. **Step 2** [NEXT] — Phase 3: implement `replicAntMultiViewPreprocessor` and `__main__` CLI.
3. **Step 3** — Step-0 smoke test (see §"Open empirical question" near the bottom):
   - Preprocess `--max_frames 500` to a small HDF5.
   - HDF5 round-trip check: open via `SLEAPMultiViewDataset`, take sample 0, apply inverse `canonical_to_world` (`R_0.T`, `-t_0 @ R_0.T`) to `keypoints_3d`, assert <= 1e-3 against the loader's `keypoints_3d_world` for the same source frame.
   - Short multi-view training run (~50 epochs on the 500-frame subset). Pass criterion: per-view reprojection loss converges normally; `trans` head behaves under direct supervision.
4. **Step 4** — Phase 4: extend `UnifiedSMILDataset.from_path()` auto-detection.
5. **Step 5** — Phase 2: add `MultiViewreplicAntSMILDataset` test seam (low priority; production reads HDF5).
6. **Step 6** — Example multi-view config JSON under `smal_fitter/neuralSMIL/configs/examples/`.

After Step 3 passes, the full preprocess (all 10k frames) and the production training run unblock; everything else above is wrap-up.

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
  the existing single-view `load_SMIL_Unreal_sample()` output. (Phase 5,
  deferred per §Status; the design supports it.)

### Inverse transform stored for validation

The canonical-camera-frame conversion is reversible. Store the per-frame
**forward** world->canonical transform under
`/auxiliary/canonical_to_world_R` and `/auxiliary/canonical_to_world_t`
(equal to the canonical camera's row-vector extrinsics `(R_0, t_0)`).
Consumers that need raw world coordinates apply the inverse explicitly:

```python
x_world = (x_canonical - t_0) @ R_0.T
```

We picked forward storage over storing the inverse because `(R_0, t_0)`
is the artifact computed at the canonical site — storing it avoids
drift from re-deriving, and the inverse is a one-liner at decode time.
The field is used for:
- Tests validating that round-tripping reproduces the raw extrinsics.
- Downstream tools that *do* want absolute world coords (rendering at a
  specific scene location, debugging procgen, etc.) recovering them.
- Training pipeline ignores this field; it's metadata only.

For procedural replicAnt scenes the absolute world coordinates carry no
meaningful information — they're arbitrary draws from the procgen randomiser
— so the trainer not using this field is the desired behaviour.

### SLEAP path: apply the same normalisation

The SLEAP preprocessor currently stores camera extrinsics in the rig's
calibration-tool-defined world frame. To keep both paths producing
byte-equivalent on-disk conventions:

- **TODO (out of scope for the replicAnt preprocessor work)**: apply
  canonical-camera-frame normalisation in `preprocess_sleap_multiview_dataset.py`
  too. For SLEAP this is a no-op for learning (rig is fixed, so it's a
  constant shift) but makes the data format symmetric across the two paths.

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

