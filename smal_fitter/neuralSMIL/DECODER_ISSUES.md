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

### 4. NaN Clamping Breaks Gradient Flow — FIXED

**Files:** `transformer_decoder.py:362-396`

When a prediction becomes non-finite mid-IEF, the code replaced it with
`torch.zeros_like(...)` (or identity for rotations). This detached the
computational graph at the NaN point, zeroing gradients for the parameters
that produced the failure.

**Resolution:** Replaced graph-breaking `torch.zeros_like` clamping with
gradient-preserving `torch.nan_to_num`. NaN values are replaced with 0 and
inf values are clamped, but the operation preserves the computational graph
so gradients still flow through non-NaN elements.

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

### 9. `init_pose = zeros` Invalid for 6D Rotation Representation — FIXED

**Files:** `transformer_decoder.py` (`_initialize_prediction_buffers`)

`init_pose` was set to all-zeros regardless of rotation representation.
For axis-angle this is correct (zero vector = identity), but for 6D the
identity rotation is `[1,0,0,1,0,0]` (first two columns of I₃). All-zeros
in 6D is a degenerate, invalid rotation with three consequences:

- Gram-Schmidt in `rotation_6d_to_matrix` on a near-zero input produces
  undefined/huge gradients through `joint_angle_regularization`, silently
  masked by the `nan_to_num` fix — explaining why large angles were hard to
  reach after that fix landed together with increased batch size.
- IEF feedback at iteration 2 encodes a near-zero 6D vector the network
  cannot interpret as "I predict zero rotation".
- The network must learn a constant `[1,0,0,1,0,0]` offset per joint just to
  represent the identity pose, adding an implicit bias against any rotation.

**Resolution:** When `rotation_representation == '6d'`, `init_pose` is now
initialised to `identity_6d.repeat(1 + N_POSE)`.  Axis-angle path unchanged.

---

## Fix Order

1. **#1 (IEF feedback)** and **#2 (single-token cross-attention)** — fix
   together since both concern the transformer decoder's input conditioning.
2. **#3 (mean-pool bottleneck)** — naturally resolves once #2 is addressed.
3. ~~**#4 (NaN clamping)**~~ — **DONE.** Replaced with `torch.nan_to_num`.
4. ~~**#5 (geometric consistency)**~~ — **DONE.** Triangulation consistency
   loss implemented, tested, and integrated into the training curriculum.
5. ~~**#9 (6D init_pose)**~~ — **DONE.** `init_pose` now initialised to 6D
   identity `[1,0,0,1,0,0]` repeated per joint.
6. **#6–#8** — address incrementally as training matures.
