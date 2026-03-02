# Multiview Transformer Decoder — Architecture Issues

Identified during deep review of `multiview_baseline.json` config and the
`MultiViewSMILImageRegressor` / `SMILTransformerDecoderHead` code paths.

---

## P0 — Critical

### 1. IEF Loop Has No Actual Feedback — FIXED

**Files:** `transformer_decoder.py`

The Iterative Error Feedback loop assembled `param_tokens` from the current
predictions each iteration but **never fed them back into the network**.
Instead, a fresh `torch.zeros(batch_size, 1, 1)` token was created every
iteration and embedded identically. The transformer decoder received identical
input on every IEF iteration, so residual updates simply accumulated N copies
of the same delta.

**Resolution:** The concatenated parameter estimates are now projected through
`token_embedding` as the decoder query token each iteration, so the decoder
conditions on its own previous output. A `LayerNorm` (`param_norm`) normalises
the concatenated parameter vector before projection, handling the multi-scale
nature of the feedback adaptively. Currently disabled (`ief_iters=1`) pending
validation — see "Post-Fix Regression" below.

### 2. Transformer Decoder Cross-Attends to a Single Token in Multiview Mode — FIXED

**Files:** `multiview_smil_regressor.py`

In single-view mode the transformer decoder cross-attends to 196 ViT patch
tokens — rich spatial context. In multiview mode, the mean-pooled body feature
vector was reshaped to `(B, 1, 1024)` and passed as `spatial_features`,
making cross-attention mathematically equivalent to a linear projection.

**Resolution:** Per-view fused features `(B, V, 1024)` are now passed as the
cross-attention context so the decoder can attend over views. Note: with
V = 2–6 tokens this is still limited compared to single-view's 196 tokens
(see issue #11).

---

## P1 — High

### 3. Mean-Pool Bottleneck Before Body Prediction — FIXED

**Files:** `multiview_smil_regressor.py`

After cross-view fusion produces per-view features `(B, V, 1024)`, they were
mean-pooled to `(B, 1024)` before entering the body aggregator and decoder.

**Resolution:** Resolved by fix #2 — the transformer decoder now receives
per-view features directly as cross-attention context. The mean-pool is still
computed for the global feature vector injected into the decoder token (see
fix #10).

---

## P2 — Medium

### 4. NaN Clamping Breaks Gradient Flow — FIXED

**Files:** `transformer_decoder.py`

When a prediction became non-finite mid-IEF, `torch.zeros_like(...)` replacements
detached the computational graph, zeroing gradients for the parameters that
produced the failure.

**Resolution:** Replaced with gradient-preserving `torch.nan_to_num`. NaN → 0
and inf values are clamped, but the computational graph is preserved so gradients
still flow through non-NaN elements.

### 5. No Multi-View Geometric Consistency for Camera Heads — FIXED

**Files:** `multiview_smil_regressor.py` (`_triangulate_joints_dlt`,
`_compute_multiview_losses`)

**Resolution:** Implemented a differentiable **triangulation consistency loss**
that triangulates GT 2D keypoints using predicted cameras (via DLT with normal
equations and Tikhonov damping) and compares the result against detached body
model 3D predictions. Gradients flow through the differentiable triangulation
into the camera heads. The loss ramps up via curriculum as direct camera
supervision is phased out.

See [docs/triangulation_consistency_loss.md](docs/triangulation_consistency_loss.md)
for full details. Validated with 12 synthetic tests in `tests/test_triangulation_consistency.py`.

---

## P3 — Low / Observations

### 6. Overly Complex Curriculum & LR Schedule — FIXED

14 curriculum stages and 15 LR steps over 600 epochs with non-monotonic LR
warm restarts. Fragile and hard to maintain.

**Resolution:** Simplified to monotonically decreasing LR steps over 300 epochs.
Dead curriculum stages removed.

### 7. Batch Size 3 — Very Noisy Gradients — FIXED

**Resolution:** Larger batch sizes (8–16) now fit comfortably after IEF
restructuring eliminated redundant forward passes.

### 8. `global_rot` Loss Weight is 0.0 — BY DESIGN

**Resolution:** Intentional. The training paradigm assumes fixed, known cameras
with moving specimens; global rotation is observable only through 2D and 3D
keypoint reprojection losses. A direct regulariser would impose an orientation
prior that conflicts with arbitrary specimen poses and fixed-camera geometry.

### 9. `init_pose = zeros` Invalid for 6D Rotation Representation — FIXED

**Files:** `transformer_decoder.py` (`_initialize_prediction_buffers`)

`init_pose` was set to all-zeros regardless of rotation representation. For 6D,
the identity rotation is `[1,0,0,1,0,0]` (first two columns of I₃). All-zeros
is a degenerate rotation causing undefined gradients through Gram-Schmidt and
forcing the network to learn a constant offset per joint.

**Resolution:** When `rotation_representation == '6d'`, `init_pose` is now
initialised to `identity_6d.repeat(1 + N_POSE)`. Axis-angle path unchanged.

---

## Post-Fix Regression — Why the Fixed Model Performed Worse

Despite the fixes above being theoretically correct, training with IEF feedback
enabled (`ief_iters=3`) produced **slower convergence and lower accuracy** than
the pre-fix model. Root cause analysis identified the following issues:

### 10. `global_feats` computed but never used by the decoder — FIXED

**Files:** `transformer_decoder.py`, `multiview_smil_regressor.py`

The call site computed `global_feats` (mean-pooled view features, `(B, D)`) and
passed it as the first argument to `transformer_head(global_feats, spatial_feats)`.
The decoder's `forward()` only used this for `batch_size` and `device` — it
never injected the global feature vector into the computation.

**Resolution:** Added `global_feat_proj` (`nn.Linear(feature_dim, hidden_dim)`)
that projects the pooled image features and adds them to the decoder token.
Computed once outside the IEF loop (constant across iterations). Initialised
with `gain=0.1` to avoid disrupting pretrained decoder layers.

### 11. V-token cross-attention ≈ learned mean-pool — FIXED

**Files:** `multiview_smil_regressor.py`

The fix for #2 expanded cross-attention from 1 token to V = 2–6 view tokens.
However, cross-attention with 4 heads over 2–6 keys is a softmax-weighted
average of 2–6 values — barely more expressive than mean-pooling. The
transformer's representational advantage requires many tokens (e.g., 196 ViT
patches in single-view mode).

**Resolution:** `forward_multiview()` now calls `backbone.forward_with_spatial()`
(ViT only) to extract per-view patch tokens `(B, V, 196, D)` alongside CLS
tokens. Learned per-view positional embeddings (`patch_view_embed`) are added so
the decoder knows which view each patch belongs to, then patches are reshaped to
`(B, V*196, D)` and passed as `spatial_features` to the decoder. Cross-view
attention fusion still operates on CLS tokens only. VRAM impact is negligible
(Q=1 cross-attention, attention matrix is just `(B, heads, 1, V*196)`).

### 12. Parameter feedback scale mismatch — FIXED

**Files:** `transformer_decoder.py`

IEF concatenates all parameter predictions into a single vector. Components
span 3 orders of magnitude (pose ~1, cam_trans ~100). A hardcoded per-group
divisor was initially used but became stale as predictions diverged from init
(e.g. FOV converging to 60 still divided by init value 8).

**Resolution:** Replaced hardcoded `_param_norm_scale` buffer with
`nn.LayerNorm(_param_feedback_dim)` on the concatenated parameter vector.
LayerNorm learns per-component affine transform from data, adapting to actual
parameter distributions without hardcoded assumptions.

### 13. IEF intermediate supervision causes gradient interference — MITIGATED

**Files:** `multiview_smil_regressor.py`

Keypoint-level supervision on intermediate IEF iterations forced shared decoder
layers to optimise for both coarse (iter 1) and fine (iter 3) predictions
simultaneously. Additionally, intermediate 2D losses used **final** predicted
cameras (not per-iteration cameras), coupling intermediate body predictions to
final camera quality.

**Resolution:** IEF intermediate supervision disabled (`ief_intermediate: 0.0`)
and IEF iterations set to 1. To be re-evaluated once single-iteration baseline
is established.

---

## Current Status

IEF is disabled (`ief_iters=1`, `ief_intermediate=0.0`) to establish a
single-iteration baseline. The decoder fixes (#10 global feature injection,
#12 LayerNorm normalisation) are active even in single-iteration mode.

**Next steps to re-enable IEF:**

1. Validate single-iteration baseline matches or beats pre-fix accuracy
2. Re-enable with `ief_iters: 2`, monitor `ief_pose_delta_*` metrics
3. Re-enable intermediate supervision only after IEF itself converges

---

## Fix Order

1. ~~**#1 (IEF feedback)**~~ — FIXED. Currently disabled (`ief_iters=1`).
2. ~~**#2 (single-token cross-attention)**~~ — FIXED (V-token context).
3. ~~**#3 (mean-pool bottleneck)**~~ — FIXED (resolved by #2).
4. ~~**#4 (NaN clamping)**~~ — FIXED. `torch.nan_to_num`.
5. ~~**#5 (geometric consistency)**~~ — FIXED. Triangulation consistency loss.
6. ~~**#6 (curriculum complexity)**~~ — FIXED. Simplified schedule.
7. ~~**#7 (batch size)**~~ — FIXED. Larger batches now feasible.
8. **#8 (`global_rot` weight)** — BY DESIGN.
9. ~~**#9 (6D init_pose)**~~ — FIXED. Identity 6D initialisation.
10. ~~**#10 (global_feats unused)**~~ — FIXED. `global_feat_proj` injection.
11. ~~**#11 (V-token bottleneck)**~~ — FIXED. Per-view patch tokens `(B, V*196, D)`.
12. ~~**#12 (feedback scale mismatch)**~~ — FIXED. `LayerNorm` on param feedback.
13. ~~**#13 (IEF intermediate interference)**~~ — MITIGATED. Disabled for now.
