# Data Processing Refactor: Move Image Preprocessing Out of Forward Pass

## Problem

The `predict_from_multiview_batch` method in `multiview_smil_regressor.py` (lines 2354-2438)
runs a **per-sample, per-view Python loop** that:

1. Calls `preprocess_image()` which converts data **back to numpy** (`image_data.cpu().numpy()`),
   runs format checks, `cv2.resize`, then converts back to a torch tensor.
2. Moves each individual image to GPU with `.to(self.device)` — one **synchronous** H2D copy per image.
3. Creates zero-padded dummy images one at a time on GPU.

With batch_size B and V views, that is **B×V sequential** CPU→GPU transfers and numpy↔torch
round-trips happening *inside the forward pass*, blocking the GPU between each one.
This completely serializes the data pipeline and prevents overlap between compute and data
transfer.  On A100s with 4 GPUs, the GPUs sit idle waiting for this loop.

The DataLoader prefetching / persistent worker improvements from the previous commit
cannot help because the heavy work happens **after** the DataLoader returns — inside
the model's own method.

## Goal

Move **all** image preprocessing (format normalization, resize, HWC→CHW, float conversion)
into the **dataset** (CPU worker side) and **collate function**, so that:

- The DataLoader yields **ready-to-use GPU tensors** (or pinned CPU tensors).
- `predict_from_multiview_batch` receives pre-batched, pre-resized tensors and only
  needs to stack, pad, and transfer in bulk.
- Visualization code paths that call `preprocess_image()` continue to work unchanged.

## Current Data Flow

```
Dataset.__getitem__
  → JPEG decode → float32 [0,1] numpy (H,W,3)  [CPU worker]
  → augmentation on numpy                       [CPU worker]
  → return x_data with list of numpy images

multiview_collate_fn
  → trivially zips x_data_batch, y_data_batch   [main process]

predict_from_multiview_batch                     [MAIN PROCESS, BLOCKING GPU]
  → per sample, per view:
      preprocess_image(numpy) → cv2.resize → torch tensor (1,3,H,W)
      .to(self.device)          ← synchronous H2D copy
  → torch.stack per view       ← redundant .to(device) again
  → forward_multiview(...)
```

## Target Data Flow

```
Dataset.__getitem__
  → JPEG decode → float32 [0,1] numpy (H,W,3)  [CPU worker]
  → augmentation on numpy                       [CPU worker]
  → cv2.resize to input_resolution              [CPU worker]  ← MOVED HERE
  → numpy HWC→CHW, torch.from_numpy            [CPU worker]  ← MOVED HERE
  → return x_data with list of (3,H,W) tensors

multiview_collate_fn                             [main process]
  → find max_views across batch
  → pad samples with fewer views (zero tensors + view_mask)
  → stack into (B, V, 3, H, W) tensor           ← BATCHED
  → stack camera_indices into (B, V) tensor
  → stack view_mask into (B, V) tensor
  → return structured batch dict

predict_from_multiview_batch                     [MAIN PROCESS]
  → single .to(device, non_blocking=True) for entire (B,V,3,H,W) tensor
  → forward_multiview(...)
```

## Implementation Plan

### Step 1: Add `input_resolution` to dataset — DONE
- [x] Add `input_resolution` parameter to `SLEAPMultiViewDataset.__init__`
- [x] Store as `self.input_resolution`

### Step 2: Move preprocessing into dataset `__getitem__` — DONE
- [x] After augmentation in `_get_multiview_sample`, resize each image to
      `(input_resolution, input_resolution)` using `cv2.resize`
- [x] Convert from numpy HWC float32 to torch CHW float32 via `_images_to_tensors()`
- [x] Replace `x_data['images']` list of numpy arrays with list of `(3,H,W)` tensors
- [x] Single-view path (`_get_single_view_sample`) left as-is — different format, not
      used in multiview training, and not performance-critical

### Step 3: Refactor `multiview_collate_fn` to batch tensors — DONE
- [x] Detect tensor vs numpy images (fast path vs legacy)
- [x] Find `max_views` across the batch
- [x] Pad samples with fewer views using zero tensors + view_mask
- [x] Stack images into `(B, V, 3, H, W)`, camera_indices `(B, V)`, view_mask `(B, V)`
- [x] Attach as `_batched_images`, `_batched_view_mask`, `_batched_camera_indices` on
      `x_data_batch[0]` to preserve list-of-dicts return type

### Step 4: Update `predict_from_multiview_batch` — DONE
- [x] Fast path: detect `_batched_images` key, single `.to(device, non_blocking=True)`
- [x] Legacy path: preserved for visualization / inference (per-image `preprocess_image()`)
- [x] Reshape `(B, V, 3, H, W)` → list of `(B, 3, H, W)` per view for `forward_multiview`

### Step 5: Keep `preprocess_image` working for visualization — DONE (no changes needed)
- [x] `preprocess_image()` in `smil_image_regressor.py` NOT modified
- [x] Visualization functions in `train_multiview_regressor.py` left unchanged

### Step 6: Pass `input_resolution` through config to dataset construction — DONE
- [x] Moved `input_resolution` resolution (from backbone name) before dataset construction
- [x] Removed duplicate resolution computation later in `main()`
- [x] Pass `input_resolution=input_resolution` to `SLEAPMultiViewDataset`

### Step 7: Validate and test — DONE
- [x] `pytest -m "not slow"` — 77 passed, 1 deselected
- [ ] Verify training starts and runs on HPC (pending deployment)

## Files Modified

| File | Change |
|------|--------|
| `smal_fitter/sleap_data/sleap_multiview_dataset.py` | Add `input_resolution`, resize+convert in `__getitem__` |
| `smal_fitter/sleap_data/sleap_multiview_dataset.py` | Refactor `multiview_collate_fn` to batch tensors |
| `smal_fitter/neuralSMIL/multiview_smil_regressor.py` | Simplify `predict_from_multiview_batch` |
| `smal_fitter/neuralSMIL/train_multiview_regressor.py` | Pass `input_resolution` to dataset |

## Files NOT Modified

| File | Reason |
|------|--------|
| `smal_fitter/neuralSMIL/smil_image_regressor.py` | `preprocess_image()` kept for visualization/inference |
| Visualization functions in `train_multiview_regressor.py` | Non-perf-critical, use `preprocess_image()` correctly |

## Risks and Mitigations

- **Augmentation correctness**: Augmentation happens on numpy *before* resize+convert.
  The existing augmentation code operates on raw-resolution images, which is correct.
  Resize happens after augmentation, same as before (it was in `preprocess_image`).
- **Variable view counts**: Handled by padding in collate with zero tensors + view_mask.
  This is the same logic currently in `predict_from_multiview_batch`, just moved earlier.
- **Single-view mode**: `_get_single_view_sample` also updated for consistency.
- **Backward compatibility**: `preprocess_image()` left intact for any code path that
  passes raw numpy/tensor images (visualization, inference scripts).
