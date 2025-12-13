# SLEAP Dataset Preprocessing Performance Analysis

## Current Performance Issues

### 1. **Video File I/O Bottleneck** (Major Issue)
**Location**: `sleap_data_loader.py:502-512`
```python
def _read_video_frame(self, video_file: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_file))  # Opens video file
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Seeks to frame
    ret, frame = cap.read()  # Reads frame
    cap.release()  # Closes video file
```

**Problem**: For each frame, the video file is:
- Opened
- Seeked to the target frame
- Read
- Closed

**Impact**: If processing 1000 frames from a video, this results in 1000 file open/close operations.

### 2. **Camera Data Reloading** (Major Issue)
**Location**: `preprocess_sleap_dataset.py:252`
```python
def _process_single_sample(self, loader, camera_name, frame_idx, session_path):
    camera_data = loader.load_camera_data(camera_name)  # Reloads HDF5 data
```

**Problem**: Camera data (HDF5 files) are reloaded for every single frame.

**Impact**: If processing 1000 frames from a camera, the HDF5 file is opened/parsed 1000 times.

### 3. **Inefficient Frame Processing Loop** (Moderate Issue)
**Location**: `preprocess_sleap_dataset.py:175-199`
```python
for camera_name in loader.camera_views:
    camera_data = loader.load_camera_data(camera_name)  # Load once per camera
    for frame_idx in range(num_frames):
        sample = self._process_single_sample(loader, camera_name, frame_idx, session_path)
        # Inside _process_single_sample: camera_data = loader.load_camera_data(camera_name)  # Reload again!
```

**Problem**: Camera data is loaded twice - once in the loop, once in `_process_single_sample`.

## Performance Improvements in Optimized Version

### 1. **Video File Optimization**
**Before**: Open/close video file for each frame
```python
# Original: 1000 frames = 1000 file operations
for frame_idx in range(1000):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    frame = cap.read()
    cap.release()
```

**After**: Open video file once per camera
```python
# Optimized: 1000 frames = 1 file operation
cap = cv2.VideoCapture(video_file)
for frame_idx in annotated_frames:  # Only process frames with annotations
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    frame = cap.read()
cap.release()
```

**Improvement**: ~99% reduction in video file I/O operations

### 2. **Camera Data Optimization**
**Before**: Reload camera data for each frame
```python
# Original: 1000 frames = 1000 HDF5 loads
for frame_idx in range(1000):
    camera_data = loader.load_camera_data(camera_name)
```

**After**: Load camera data once per camera
```python
# Optimized: 1000 frames = 1 HDF5 load
camera_data = loader.load_camera_data(camera_name)
for frame_idx in annotated_frames:
    # Use pre-loaded camera_data
```

**Improvement**: ~99% reduction in HDF5 file I/O operations

### 3. **Smart Frame Processing**
**Before**: Process all frames (including empty ones)
```python
# Original: Process all frames, even those without annotations
for frame_idx in range(num_frames):  # Could be 1000+ frames
    # Process frame (may have no keypoints)
```

**After**: Only process frames with annotations
```python
# Optimized: Only process frames with actual keypoint data
annotated_frames = self._get_annotated_frames(camera_data, data_structure_type)
for frame_idx in annotated_frames:  # Could be 100-200 frames
    # Process frame (guaranteed to have keypoints)
```

**Improvement**: ~80-90% reduction in frame processing (depends on annotation density)

### 4. **Batch Processing Optimization**
**Before**: Individual frame processing with repeated setup
```python
for frame_idx in range(num_frames):
    # Each frame: load data, setup, process, cleanup
    image_size = loader.get_camera_image_size(camera_name)  # Repeated
    ground_truth_betas = loader.get_ground_truth_shape_betas()  # Repeated
```

**After**: Batch processing with shared setup
```python
# Setup once per camera
image_size = loader.get_camera_image_size(camera_name)
ground_truth_betas = loader.get_ground_truth_shape_betas()
video_cap = self._open_video_capture(loader, camera_name)

for frame_idx in annotated_frames:
    # Each frame: just process (no setup/cleanup)
```

**Improvement**: ~50% reduction in setup overhead

## Expected Performance Gains

### I/O Operations Reduction
- **Video files**: 99% reduction (1 open/close per camera vs 1 per frame)
- **HDF5 files**: 99% reduction (1 load per camera vs 1 per frame)
- **Frame processing**: 80-90% reduction (only annotated frames)

### Overall Speed Improvement
**Conservative estimate**: 5-10x faster preprocessing
**Optimistic estimate**: 10-20x faster preprocessing

### Memory Usage
- **Slightly higher**: Keep video capture and camera data in memory
- **Better utilization**: Process frames in batches rather than individually

## Additional Optimization Opportunities

### 1. **Parallel Video Processing**
```python
# Process multiple cameras in parallel
with ThreadPoolExecutor(max_workers=num_cameras) as executor:
    futures = [executor.submit(process_camera, camera) for camera in cameras]
```

### 2. **Memory-Mapped HDF5 Access**
```python
# Use memory mapping for large HDF5 files
with h5py.File(hdf5_path, 'r', driver='core') as f:
    # Memory-mapped access
```

### 3. **Frame Caching**
```python
# Cache frequently accessed frames
frame_cache = {}
if frame_idx in frame_cache:
    frame = frame_cache[frame_idx]
else:
    frame = read_frame(frame_idx)
    frame_cache[frame_idx] = frame
```

### 4. **Batch Image Processing**
```python
# Process multiple images in batch
frames = [read_frame(i) for i in frame_indices]
processed_frames = batch_preprocess_images(frames)
```

## Implementation Priority

1. **High Priority**: Video file optimization (biggest impact)
2. **High Priority**: Camera data optimization (biggest impact)
3. **Medium Priority**: Smart frame processing (good impact)
4. **Low Priority**: Additional optimizations (diminishing returns)

## Usage

```bash
# Use optimized version
python optimized_sleap_preprocessor.py /path/to/sessions output.h5 \
    --joint_lookup_table lookup.csv \
    --shape_betas_table betas.csv \
    --num_workers 4
```

The optimized version maintains the same interface and output format as the original, ensuring compatibility with the existing training pipeline.
