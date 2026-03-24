<!-- Created: 2026-02-27 | Last modified: 2026-02-27 -->

# Triangulation Consistency Loss

## Motivation

In the multi-view regressor, each view has its own **camera head** that independently predicts FOV, rotation, and translation. While the 2D keypoint reprojection loss (`keypoint_2d`) and optional direct camera supervision (`cam_rot`, `cam_trans`, `fov`) provide per-view camera learning signals, neither explicitly enforces that the cameras from different views are **geometrically consistent with each other**.

Without an explicit multi-view geometric constraint, the camera heads can drift into configurations where each camera independently explains its own 2D observations well, but the cameras collectively do not agree on a coherent 3D scene. The `triangulation_consistency` loss closes this gap.

## Conceptual Overview

The idea is simple:

1. Take the **ground-truth 2D keypoints** observed in each view.
2. Use the **predicted camera parameters** from each view to triangulate those 2D observations into 3D.
3. Compare the triangulated 3D points against the model's **predicted 3D joints** (from the body model).

If the predicted cameras are geometrically consistent, the triangulated 3D points will agree with the body model's 3D prediction. If the cameras are inconsistent, triangulation will produce points that diverge from the body model — and the loss penalises this.

### Gradient flow

The gradient flows through the **triangulation** into the **camera heads** — this is what makes the loss effective. The body model's 3D joint predictions (`joints_3d`) are **detached** and serve as a stable target, since they are already well-supervised by the `keypoint_3d` loss.

```
GT 2D keypoints ─┐
                  ├──▶ Differentiable DLT ──▶ triangulated_3d ──┐
Predicted cameras ┘        (grad flows)                          ├──▶ MSE loss
                                                                 │
Predicted 3D joints (detached, no grad) ────────────────────────┘
```

This design avoids conflicting gradient signals: the body model doesn't get pulled in two directions (by both `keypoint_3d` and `triangulation_consistency`), and the cameras get direct geometric feedback.

## Mathematical Formulation

### DLT (Direct Linear Transform) with w=1 normalisation

Given a 3D point `X = [X, Y, Z, 1]^T` in homogeneous coordinates and a 4x4 projection matrix `P`, the projection into normalised device coordinates (NDC) is:

```
[u*w, v*w, z*w, w] = [X, Y, Z, 1] @ P      (PyTorch3D row-vector convention)
```

After perspective divide: `u = (x @ P[:,0]) / (x @ P[:,3])`, `v = (x @ P[:,1]) / (x @ P[:,3])`.

The DLT constraint for each view is:

```
u * P[:,3] - P[:,0] = 0    (dot product with [X,Y,Z,1])
v * P[:,3] - P[:,1] = 0
```

This gives 2 linear equations per view, forming a system `A @ [X, Y, Z, 1]^T = 0` with `A` of shape `(2V, 4)` for `V` views.

**w=1 normalisation:** Rather than solving the homogeneous system via SVD (which has numerically unstable gradients), we fix the homogeneous coordinate to 1 and rewrite:

```
A[:, :3] @ [X, Y, Z]^T = -A[:, 3]
```

This is an overdetermined linear system solved via the **normal equations**:

```
(A_xyz^T @ A_xyz + λI) @ p = A_xyz^T @ (-a_w)
```

where `λ = 1e-6` is Tikhonov damping that ensures the system is always full-rank, even when some views are masked out (producing zero rows in `A`).

### Why normal equations instead of SVD or lstsq

| Method | Gradient stability | Rank-deficiency handling |
|--------|-------------------|------------------------|
| `torch.linalg.svd` | Unstable — backward has `1/(σ_i - σ_j)` terms that explode for near-degenerate systems | Handles naturally but gradients are NaN |
| `torch.linalg.lstsq` | Moderate — uses LAPACK `gelsd` internally | Fails hard with error when matrix is rank-deficient |
| **Normal equations + `torch.linalg.solve`** | **Stable** — backward is another linear solve (implicit function theorem) | **Always works** — Tikhonov damping ensures full rank |

### PyTorch3D projection matrix convention

PyTorch3D uses **row-vector convention**: `ndc_homo = world_homo @ P` where `P` is `(4, 4)`. The **columns** of `P` correspond to output dimensions:

- Column 0: `u * w` (x output)
- Column 1: `v * w` (y output)
- Column 2: `z * w` (depth output)
- Column 3: `w` (homogeneous weight)

This is transposed relative to the standard computer vision convention (column-vector, `P @ x`), so the DLT must index **columns** of `P`, not rows.

### NDC coordinate recovery

The forward projection in `_batch_project_joints_to_views` produces normalised [0,1] keypoints via:

```python
screen = transform_points_screen(joints, image_size=img_size)[:, :, [1, 0]]  # swap x,y
normalised = screen / img_size
```

The inverse (used in `_triangulate_joints_dlt`) recovers NDC from normalised keypoints:

```python
kp_screen = keypoints_2d * img_size      # un-normalise
kp_screen = kp_screen[..., [1, 0]]       # un-swap (y,x) -> (x,y)
ndc_xy = 1.0 - kp_screen / (img_size / 2)  # screen -> NDC
```

Note the sign: PyTorch3D's `ndc_to_screen` maps `screen = (W-1)/2 * (1 - ndc)`, so the inverse is `ndc = 1 - screen / (W/2)`, **not** `screen / (W/2) - 1`.

## Implementation Details

### Files

- **Loss computation:** `multiview_smil_regressor.py`, in `_compute_multiview_losses()`, starting at the `TRIANGULATION CONSISTENCY LOSS` section.
- **Triangulation solver:** `multiview_smil_regressor.py`, method `_triangulate_joints_dlt()`.
- **Configuration:** `configs/examples/multiview_sticks.json`, under `loss_curriculum.base_weights.triangulation_consistency` and `loss_curriculum.curriculum_stages`.
- **Tests:** `tests/test_triangulation_consistency.py` — 12 synthetic tests covering round-trip accuracy, gradient flow, loss computation, and edge cases.

### Loss computation (`_compute_multiview_losses`)

```python
# 1. Triangulate GT 2D keypoints using predicted cameras (differentiable)
triangulated, tri_valid = self._triangulate_joints_dlt(
    target_kps, visibility,
    predicted_params['fov_per_view'],
    predicted_params['cam_rot_per_view'],
    predicted_params['cam_trans_per_view'],
    ...
)

# 2. Reject outlier triangulations (no grad — just masking)
with torch.no_grad():
    tri_valid = tri_valid & (triangulated.norm(dim=-1) < 50.0)

# 3. Compute MSE against detached body model predictions
diff_sq = (triangulated - joints_3d.detach()) ** 2

# 4. Mask by validity and joint importance, normalise
masked_loss = diff_sq * mask_weights.unsqueeze(-1)
tri_loss = masked_loss.sum() / denom
```

Key design decisions:

- **`joints_3d.detach()`**: The body model's 3D predictions are the target. No gradient flows into the body model through this loss — it only supervises cameras.
- **`triangulated` retains grad**: The triangulation is differentiable w.r.t. the predicted camera parameters (via `torch.linalg.solve` backward). Gradients flow: `tri_loss → triangulated → AtA/Atb → projection matrices → camera head parameters`.
- **Outlier rejection at norm > 50**: Early in training, cameras may be poorly calibrated, producing nonsensical triangulations. These are masked out (the threshold is generous — the actual scene is much smaller).
- **Joint importance weights**: If configured, important joints (e.g., leg endpoints) receive higher weight in the loss, matching the weighting used by other losses.

### Triangulation solver (`_triangulate_joints_dlt`)

**Input:** GT 2D keypoints `(B, V, J, 2)`, per-joint visibility `(B, V, J)`, predicted camera parameters.

**Output:** Triangulated 3D joints `(B, J, 3)`, valid mask `(B, J)`.

Steps:

1. **Build projection matrices** from predicted FOV, rotation, translation using PyTorch3D's `FoVPerspectiveCameras.get_full_projection_transform()`.
2. **Convert [0,1]-normalised keypoints to NDC** by inverting the screen-space mapping.
3. **Construct the DLT constraint matrix** `A` of shape `(B, J, 2V, 4)`: two rows per view per joint, using columns of `P`. Zero out rows for invisible joints. The constraint rows are assembled as `(B, V, J, 2, 4)` then reshaped via `.permute(0, 2, 1, 3, 4).reshape(B, J, V*2, 4)` — the permute is critical to avoid scrambling joints with constraint rows in memory.
4. **Split** `A` into `A_xyz (B, J, 2V, 3)` and `a_w (B, J, 2V)` for the w=1 formulation.
5. **Form normal equations**: `AtA = A_xyz^T @ A_xyz + λI`, `Atb = A_xyz^T @ (-a_w)`.
6. **Solve** via `torch.linalg.solve(AtA, Atb)` → `(B, J, 3)`.
7. **Valid mask**: joints visible in >= 2 views.

## Configuration

The loss is controlled by the `triangulation_consistency` key in the loss curriculum. It starts at **0.0** (disabled) and ramps up via curriculum stages.

Example curriculum from `multiview_sticks.json`:

| Epoch | Weight | Rationale |
|-------|--------|-----------|
| 0     | 0.0    | Cameras still initialising; triangulation would be noise |
| 30    | 0.001  | Cameras have basic structure; introduce gentle geometric pressure |
| 45    | 0.005  | Increase as cameras improve |
| 50    | 0.01   | Moderate enforcement |
| 60    | 0.05   | Cameras should be reasonably calibrated |
| 100   | 0.1    | Strong multi-view consistency |
| 150   | 0.2    | Direct camera supervision (`cam_rot`, `cam_trans`) drops to ~0; triangulation consistency takes over as primary geometric constraint |
| 200   | 0.5    | Dominant geometric signal for fine-tuning camera heads |

The curriculum is designed so that as direct camera parameter supervision is reduced (the `cam_rot`/`cam_trans` weights drop to `1e-8` by epoch 150), the triangulation consistency loss ramps up to replace it with a purely geometric constraint. This transition encourages the cameras to be self-consistent rather than memorising GT parameters.
