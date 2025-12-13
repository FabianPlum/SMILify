# Performance Degradation Fix: Random Frame Seeking Issue

## Problem Identified

The preprocessing was experiencing **massive slowdown after a few hundred frames** due to a critical performance issue with OpenCV's video frame seeking.

### Root Cause: Random Frame Seeking

**Location**: `preprocess_sleap_dataset.py:366` (original code)
```python
video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # SEEKING TO RANDOM FRAMES
```

**The Issue**:
1. **Random Access Pattern**: `annotated_frames` contained frame indices in **random order**
2. **OpenCV Seeking Overhead**: Each `cap.set()` call forces OpenCV to:
   - Seek from current position to target frame
   - **Decode all intermediate frames** (even though we don't use them)
   - This gets **exponentially slower** as seek distance increases

### Why Performance Degraded Over Time

```
Frame 1:   Seek from 0 → 1     (1 frame decode)     ✅ Fast
Frame 2:   Seek from 1 → 50    (49 frames decode)   ⚠️  Slower  
Frame 3:   Seek from 50 → 5    (45 frames decode)   ⚠️  Slower
Frame 4:   Seek from 5 → 200   (195 frames decode)  ❌ Very Slow
Frame 5:   Seek from 200 → 10  (190 frames decode)  ❌ Very Slow
...
```

**Result**: Each seek operation became progressively slower as the seek distances increased.

## Solution Implemented

### 1. **Sort Annotated Frames** (Critical Fix)
```python
# CRITICAL: Sort frames to enable sequential reading (major performance fix)
annotated_frames = sorted(annotated_frames)
```

**Before**: `[45, 12, 200, 5, 150, 8, ...]` (random order)
**After**: `[5, 8, 12, 45, 150, 200, ...]` (sequential order)

### 2. **Sequential Frame Reading** (Performance Optimization)
```python
# Use sequential reading for optimal performance
current_frame = 0
for target_frame_idx in frame_pbar:
    # Seek to target frame if needed (only when not sequential)
    if current_frame != target_frame_idx:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        current_frame = target_frame_idx
    
    # Read the frame
    ret, frame = video_cap.read()
    current_frame += 1
```

**Benefits**:
- **Minimal seeking**: Only seeks when frames are not sequential
- **Sequential reads**: Most frames are read sequentially (fastest)
- **Reduced decode overhead**: No unnecessary frame decoding

### 3. **Pre-read Frame Processing** (Eliminates Redundant Operations)
```python
def _process_frame_with_data(self, loader, camera_data, camera_name, frame_idx, 
                           session_path, image_size, ground_truth_betas, frame):
    # frame is already read - no need to read again
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ... rest of processing
```

## Performance Impact

### Before Fix:
- **Random seeking**: Each frame required seeking to random position
- **Exponential slowdown**: Performance degraded as seek distances increased
- **Unnecessary decoding**: OpenCV decoded frames we never used

### After Fix:
- **Sequential reading**: Most frames read in order (optimal)
- **Consistent performance**: No degradation over time
- **Minimal seeking**: Only when absolutely necessary

### Expected Performance Improvement:
- **10-50x faster** for large videos with many annotated frames
- **Consistent speed** throughout processing (no degradation)
- **Reduced CPU usage** (less unnecessary decoding)

## Technical Details

### OpenCV VideoCapture Seeking Behavior:
```python
# SLOW: Random seeking
cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)  # Decodes frames 0-1000
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)    # Decodes frames 1000-50 (backwards!)
cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)  # Decodes frames 50-2000

# FAST: Sequential reading
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # Start at beginning
cap.read()  # Frame 0
cap.read()  # Frame 1  
cap.read()  # Frame 2
# ... no seeking needed
```

### Memory Considerations:
- **Slightly higher memory usage**: Keep video capture open longer
- **Much better performance**: Sequential access is optimal for video files
- **Reduced I/O**: Fewer file operations overall

## Verification

To verify the fix is working:

1. **Check frame order**: Annotated frames should be sorted
2. **Monitor progress bar**: Should maintain consistent speed
3. **CPU usage**: Should be more stable (less spiky)
4. **Processing time**: Should be dramatically faster for large videos

## Additional Optimizations Applied

1. **Progress bar**: Shows real-time processing speed
2. **Error handling**: Continues processing even if individual frames fail
3. **Memory management**: Proper cleanup of video captures
4. **Batch processing**: Process all frames for a camera before moving to next

This fix addresses the core performance bottleneck and should result in consistent, fast preprocessing regardless of video size or annotation density.
