# Multiview Transformer Decoder — Architecture Issues

Identified during deep review of `multiview_baseline.json` config and the
`MultiViewSMILImageRegressor` / `SMILTransformerDecoderHead` code paths.

---

## P0 — Critical

### 1. IEF Loop Has No Actual Feedback

**Files:** `transformer_decoder.py:330-348`

The Iterative Error Feedback loop assembles `param_tokens` from the current
predictions each iteration but **never feeds them back into the network**.
Instead, a fresh `torch.zeros(batch_size, 1, 1)` token is created every
iteration and embedded identically. Since `spatial_features` is also constant,
the transformer decoder receives identical input on every IEF iteration.

The residual updates (`pred_X = pred_X + head(token_out)`) simply accumulate
N copies of the same delta. Three IEF iterations cost 3x compute for
functionally 1x output.

**Fix:** Concatenate (or project) the current parameter estimates into the
token embedding so the decoder can condition on its own previous output, as in
HMR/SPIN/AniMer.

### 2. Transformer Decoder Cross-Attends to a Single Token in Multiview Mode

**Files:** `multiview_smil_regressor.py:594`

In single-view mode the transformer decoder cross-attends to 196 ViT patch
tokens — rich spatial context. In multiview mode, the mean-pooled body feature
vector is reshaped to `(B, 1, 1024)` and passed as `spatial_features`.

Six transformer decoder layers with 8-head cross-attention all attend to a
single key/value pair. This is mathematically equivalent to a linear projection
of that single vector; all representational capacity of the decoder is wasted.

**Fix:** Pass the full per-view fused features `(B, V, 1024)` as the
cross-attention context so the decoder can attend over views, mirroring the
single-view path's use of spatial patch tokens.

---

## P1 — High

### 3. Mean-Pool Bottleneck Before Body Prediction

**Files:** `multiview_smil_regressor.py:468-477`

After cross-view fusion produces per-view features `(B, V, 1024)`, they are
mean-pooled to `(B, 1024)` before entering the body aggregator and decoder.
This is a hard information bottleneck that discards view-specific cues the
decoder could leverage.

Closely related to issue #2. If #2 is fixed (decoder cross-attends to per-view
features), the mean-pool can be removed or made optional — the decoder's own
attention mechanism handles aggregation.

---

## P2 — Medium

### 4. NaN Clamping Breaks Gradient Flow

**Files:** `transformer_decoder.py:362-396`

When a prediction becomes non-finite mid-IEF, the code replaces it with
`torch.zeros_like(...)` (or identity for rotations). This:

- Silently masks numerical instability instead of surfacing it.
- Detaches the computational graph at the NaN point, zeroing gradients for
  the parameters that produced the failure.
- Corrupts the baseline for subsequent IEF iterations.

**Fix:** Address the root cause (likely unbounded outputs or loss spikes).
If a safety net is needed, use `torch.nan_to_num` with gradient-preserving
clamping or raise/log loudly so instability is caught during development.

### 5. No Multi-View Geometric Consistency for Camera Heads

**Files:** `multiview_smil_regressor.py:608-682`

Each `CameraHead` predicts independently from per-view fused features. There
is no explicit geometric consistency constraint (epipolar loss, triangulation
consistency, cross-view reprojection). The only coupling is indirect, through
the shared 3D keypoint loss.

Camera parameter convergence (especially translation and rotation) is likely
slow and may remain inconsistent across views.

**Fix (future):** Consider adding a triangulation consistency loss or
cross-view reprojection loss to directly enforce multi-view geometry.

---

## P3 — Low / Observations

### 6. Overly Complex Curriculum & LR Schedule

15 curriculum stages and 14 LR steps over 600 epochs with non-monotonic LR
warm restarts interleaved with curriculum weight jumps (e.g., `keypoint_3d`
jumps 10x at epoch 500). Fragile and hard to maintain. Consider cosine
annealing or a simpler stage-based schedule.

### 7. Batch Size 3 — Very Noisy Gradients

Each sample has up to 6 views, so memory pressure is real, but batch size 3
yields noisy gradient estimates. Gradient accumulation over multiple steps
could help without increasing memory.

### 8. `global_rot` Loss Weight is 0.0

Never overridden in any curriculum stage. Global orientation is only supervised
indirectly through keypoint losses, which can lead to rotation/translation
ambiguity.

---

## Fix Order

1. **#1 (IEF feedback)** and **#2 (single-token cross-attention)** — fix
   together since both concern the transformer decoder's input conditioning.
2. **#3 (mean-pool bottleneck)** — naturally resolves once #2 is addressed.
3. **#4 (NaN clamping)** — clean up after decoder changes stabilise training.
4. **#5–#8** — address incrementally as training matures.
