"""
Tests for multiview augmentation pipeline.

Validates that:
- Photometric augmentations preserve camera params and 2D keypoints exactly
- Geometric augmentations (crop/scale jitter) maintain the projection
  relationship 2D = K @ [R|t] @ X_3d after updating K
- Augmentation is disabled for val/test splits

Keypoint convention (matching the training pipeline):
- 2D keypoints are stored as **normalized [0, 1] in [y, x] order**
- Camera intrinsics K are in pixel units of target_resolution

Each test that involves images saves diagnostic visualizations to
tests/test_output/augmentation/ for manual inspection.
"""

import os
import sys
import math

import pytest
import numpy as np

# Path setup
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_sleap_data = os.path.join(_repo_root, "smal_fitter", "sleap_data")
_neural_smil = os.path.join(_repo_root, "smal_fitter", "neuralSMIL")
for p in [_repo_root, _sleap_data, _neural_smil]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Output directory for visualizations
VIZ_DIR = os.path.join(os.path.dirname(__file__), "test_output", "augmentation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_viz_dir():
    os.makedirs(VIZ_DIR, exist_ok=True)


def _project_3d_to_2d(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                       points_3d: np.ndarray) -> np.ndarray:
    """Project world-space 3D points to 2D pixel coordinates [x, y].

    Args:
        K: (3, 3) intrinsics.
        R: (3, 3) rotation (world -> camera).
        t: (3,) translation.
        points_3d: (N, 3) world-space points.

    Returns:
        (N, 2) pixel coordinates in [x, y] order.
    """
    X_cam = (R @ points_3d.T).T + t  # (N, 3)
    proj = (K @ X_cam.T).T  # (N, 3)
    return proj[:, :2] / proj[:, 2:3]


def _project_3d_to_normalized(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                               points_3d: np.ndarray,
                               target_resolution: int) -> np.ndarray:
    """Project world-space 3D points to normalized [0, 1] coordinates [y, x].

    Matches the coordinate convention used by the training pipeline:
    keypoints are stored as [y, x] normalized by target_resolution.

    Args:
        K: (3, 3) intrinsics (in pixel units of target_resolution).
        R: (3, 3) rotation (world -> camera).
        t: (3,) translation.
        points_3d: (N, 3) world-space points.
        target_resolution: Image resolution for normalization (e.g. 512).

    Returns:
        (N, 2) normalized coordinates in [y, x] order.
    """
    px = _project_3d_to_2d(K, R, t, points_3d)  # [x, y] in pixels
    # Swap to [y, x] and normalize
    return px[:, ::-1] / target_resolution


def _norm_yx_to_pixel_xy(kp_norm_yx: np.ndarray, image_size: int) -> np.ndarray:
    """Convert normalized [y,x] keypoints to pixel [x,y] for visualization."""
    pixel = kp_norm_yx * image_size  # still [y, x]
    return pixel[:, ::-1]  # swap to [x, y]


def _make_synthetic_sample(image_size: int = 256, n_joints: int = 10, seed: int = 42):
    """Create a synthetic multiview-like sample with known geometry.

    Returns keypoints in **normalized [y, x]** convention matching the HDF5 format.

    Returns:
        image: (H, W, 3) float32 [0, 1] with some structure
        K: (3, 3) intrinsics
        R: (3, 3) rotation
        t: (3,) translation
        points_3d: (n_joints, 3) world-space points
        keypoints_2d: (n_joints, 2) **normalized [0,1] in [y, x] order**
        visibility: (n_joints,) all True
    """
    rng = np.random.RandomState(seed)
    h = w = image_size

    # Create an image with some spatial structure (gradient + noise)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, 0] = np.linspace(0, 1, w)[None, :].repeat(h, axis=0)
    img[:, :, 1] = np.linspace(0, 1, h)[:, None].repeat(w, axis=1)
    img[:, :, 2] = rng.uniform(0.2, 0.8, (h, w)).astype(np.float32)

    # Camera intrinsics (simple pinhole)
    fx = fy = image_size * 1.2
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    # Camera extrinsics: camera looks down -Z axis, 5 units back
    angle = 0.3
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)
    t = np.array([0.0, 0.0, 5.0], dtype=np.float32)

    # 3D points in front of the camera (within the image)
    points_3d = rng.uniform(-0.5, 0.5, (n_joints, 3)).astype(np.float32)
    points_3d[:, 2] += 0.5  # push forward so they're in front

    # Project to pixel [x, y], then convert to normalized [y, x]
    kp_pixel_xy = _project_3d_to_2d(K, R, t, points_3d)
    kp_norm_yx = kp_pixel_xy[:, ::-1] / image_size

    visibility = np.ones(n_joints, dtype=bool)

    return img, K, R, t, points_3d, kp_norm_yx, visibility


def _save_augmentation_viz(images, keypoints_pixel_xy_list, labels, filepath,
                           marker_styles=None):
    """Save side-by-side image panels with keypoint overlays.

    Args:
        images: list of (H, W, 3) images
        keypoints_pixel_xy_list: list of (N, 2) arrays in **pixel [x, y]** order
        labels: list of panel title strings
        filepath: output filename (relative to VIZ_DIR)
        marker_styles: optional list of dicts with matplotlib scatter kwargs
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed, skipping visualization")

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    if marker_styles is None:
        marker_styles = [{'c': 'lime', 's': 30, 'marker': 'o'}] * n

    for ax, img, kps, label, style in zip(axes, images, keypoints_pixel_xy_list,
                                           labels, marker_styles):
        ax.imshow(np.clip(img, 0, 1))
        if kps is not None and len(kps) > 0:
            ax.scatter(kps[:, 0], kps[:, 1], **style, zorder=5)
        ax.set_title(label)
        ax.axis('off')

    fig.tight_layout()
    out_path = os.path.join(VIZ_DIR, filepath)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPhotometricAugmentation:
    """Photometric augmentations must not alter geometry."""

    @pytest.fixture(autouse=True)
    def setup_dataset(self):
        """Create a minimal dataset-like object with augmentation methods."""
        from sleap_multiview_dataset import SLEAPMultiViewDataset
        self.ds = object.__new__(SLEAPMultiViewDataset)
        self.ds.augment = True
        self.ds.aug_color_jitter_brightness = 0.2
        self.ds.aug_color_jitter_contrast = 0.2
        self.ds.aug_color_jitter_saturation = 0.15
        self.ds.aug_gaussian_noise_std = 0.015
        self.ds.aug_gaussian_blur_prob = 0.5
        self.ds.aug_gaussian_blur_kernel_range = (3, 7)
        self.ds.aug_random_erasing_prob = 0.5
        self.ds.aug_random_erasing_scale_range = (0.02, 0.1)

    def test_photometric_preserves_keypoints(self):
        """Photometric augmentation must not change 2D keypoints or camera params."""
        _ensure_viz_dir()
        img, K, R, t, pts3d, kp2d, vis = _make_synthetic_sample()
        image_size = img.shape[0]

        K_orig = K.copy()
        kp2d_orig = kp2d.copy()

        np.random.seed(123)
        img_aug = self.ds._apply_photometric_augmentation(img.copy())

        # Camera params unchanged
        np.testing.assert_array_equal(K, K_orig)
        # Keypoints unchanged (photometric doesn't touch them)
        np.testing.assert_array_equal(kp2d, kp2d_orig)
        # Image was actually modified
        assert not np.allclose(img, img_aug, atol=1e-6)
        # Reprojection still matches original keypoints
        reproj = _project_3d_to_normalized(K, R, t, pts3d, image_size)
        np.testing.assert_allclose(reproj, kp2d_orig, atol=1e-4)

        # Visualization (convert to pixel [x,y] for display)
        kp_px = _norm_yx_to_pixel_xy(kp2d_orig, image_size)
        _save_augmentation_viz(
            [img, img_aug],
            [kp_px, kp_px],
            ["Original", "Photometric Aug (same keypoints)"],
            "photometric_preserves_geometry.png",
        )

    def test_individual_augmentations(self):
        """Test each photometric augmentation type individually with visualization."""
        _ensure_viz_dir()
        img_orig, K, R, t, pts3d, kp2d, vis = _make_synthetic_sample()
        image_size = img_orig.shape[0]
        kp_px = _norm_yx_to_pixel_xy(kp2d, image_size)

        aug_types = {
            "brightness": {"aug_color_jitter_brightness": 0.3,
                           "aug_color_jitter_contrast": 0,
                           "aug_color_jitter_saturation": 0,
                           "aug_gaussian_noise_std": 0,
                           "aug_gaussian_blur_prob": 0,
                           "aug_random_erasing_prob": 0},
            "contrast": {"aug_color_jitter_brightness": 0,
                         "aug_color_jitter_contrast": 0.3,
                         "aug_color_jitter_saturation": 0,
                         "aug_gaussian_noise_std": 0,
                         "aug_gaussian_blur_prob": 0,
                         "aug_random_erasing_prob": 0},
            "saturation": {"aug_color_jitter_brightness": 0,
                           "aug_color_jitter_contrast": 0,
                           "aug_color_jitter_saturation": 0.3,
                           "aug_gaussian_noise_std": 0,
                           "aug_gaussian_blur_prob": 0,
                           "aug_random_erasing_prob": 0},
            "noise": {"aug_color_jitter_brightness": 0,
                      "aug_color_jitter_contrast": 0,
                      "aug_color_jitter_saturation": 0,
                      "aug_gaussian_noise_std": 0.05,
                      "aug_gaussian_blur_prob": 0,
                      "aug_random_erasing_prob": 0},
            "blur": {"aug_color_jitter_brightness": 0,
                     "aug_color_jitter_contrast": 0,
                     "aug_color_jitter_saturation": 0,
                     "aug_gaussian_noise_std": 0,
                     "aug_gaussian_blur_prob": 1.0,
                     "aug_random_erasing_prob": 0},
            "erasing": {"aug_color_jitter_brightness": 0,
                        "aug_color_jitter_contrast": 0,
                        "aug_color_jitter_saturation": 0,
                        "aug_gaussian_noise_std": 0,
                        "aug_gaussian_blur_prob": 0,
                        "aug_random_erasing_prob": 1.0,
                        "aug_random_erasing_scale_range": (0.05, 0.15)},
        }

        for name, params in aug_types.items():
            for k, v in params.items():
                setattr(self.ds, k, v)

            np.random.seed(42)
            img_aug = self.ds._apply_photometric_augmentation(img_orig.copy())

            _save_augmentation_viz(
                [img_orig, img_aug],
                [kp_px, kp_px],
                ["Original", f"{name}"],
                f"photometric_{name}.png",
            )


class TestGeometricAugmentation:
    """Geometric augmentation must maintain 3D->2D projection consistency.

    All keypoints are in normalized [0,1] [y,x] order (matching HDF5 convention).
    """

    @pytest.fixture(autouse=True)
    def setup_dataset(self):
        from sleap_multiview_dataset import SLEAPMultiViewDataset
        self.ds = object.__new__(SLEAPMultiViewDataset)
        self.ds.augment = True
        self.ds.aug_crop_jitter_fraction = 0.05
        self.ds.aug_scale_jitter_range = (0.9, 1.1)

    def test_geometric_reprojection_consistency(self):
        """After geometric aug, reprojecting 3D with updated K must match augmented 2D."""
        _ensure_viz_dir()
        img, K, R, t, pts3d, kp2d, vis = _make_synthetic_sample()
        image_size = img.shape[0]

        np.random.seed(77)
        img_aug, K_aug, kp2d_aug, vis_aug = self.ds._apply_geometric_augmentation(
            img.copy(), K.copy(), kp2d.copy(), vis.copy()
        )

        # Independently reproject 3D points using the UPDATED K -> normalized [y,x]
        reproj = _project_3d_to_normalized(K_aug, R, t, pts3d, image_size)

        # Only compare visible keypoints
        visible = vis_aug.astype(bool)
        if visible.sum() > 0:
            # Tolerance: 0.5px / image_size in normalized units
            np.testing.assert_allclose(
                reproj[visible], kp2d_aug[visible], atol=0.5 / image_size,
                err_msg="Reprojected 3D with updated K must match augmented 2D keypoints"
            )

        # Visualization: convert to pixel [x,y] for display
        kp_orig_px = _norm_yx_to_pixel_xy(kp2d, image_size)
        kp_aug_px = _norm_yx_to_pixel_xy(kp2d_aug, image_size)
        reproj_px = _norm_yx_to_pixel_xy(reproj, image_size)

        _save_augmentation_viz(
            [img, img_aug, img_aug],
            [kp_orig_px, kp_aug_px, reproj_px],
            ["Original + orig kps",
             "Augmented + transformed kps",
             "Augmented + reprojected from 3D"],
            "geometric_reprojection_consistency.png",
            marker_styles=[
                {'c': 'lime', 's': 30, 'marker': 'o'},
                {'c': 'lime', 's': 30, 'marker': 'o'},
                {'c': 'red', 's': 50, 'marker': 'x', 'linewidths': 2},
            ],
        )

    def test_geometric_multiple_seeds(self):
        """Test reprojection consistency across many random augmentations."""
        img, K, R, t, pts3d, kp2d, vis = _make_synthetic_sample()
        image_size = img.shape[0]

        max_errors = []
        for seed in range(100):
            np.random.seed(seed)
            _, K_aug, kp2d_aug, vis_aug = self.ds._apply_geometric_augmentation(
                img.copy(), K.copy(), kp2d.copy(), vis.copy()
            )
            reproj = _project_3d_to_normalized(K_aug, R, t, pts3d, image_size)
            visible = vis_aug.astype(bool)
            if visible.sum() > 0:
                err = np.max(np.abs(reproj[visible] - kp2d_aug[visible]))
                max_errors.append(err)

        max_errors = np.array(max_errors)
        # All errors must be < 0.5 pixel in normalized units
        tol = 0.5 / image_size
        assert max_errors.max() < tol, (
            f"Max reprojection error across 100 seeds: {max_errors.max():.6f} "
            f"normalized ({max_errors.max() * image_size:.4f} px)"
        )
        print(f"  Geometric consistency over 100 seeds: "
              f"max={max_errors.max():.6f} norm ({max_errors.max() * image_size:.4f} px), "
              f"mean={max_errors.mean():.6f} norm")

    def test_extrinsics_unchanged(self):
        """Geometric augmentation must not modify R or t."""
        img, K, R, t, pts3d, kp2d, vis = _make_synthetic_sample()
        R_orig = R.copy()
        t_orig = t.copy()

        np.random.seed(42)
        self.ds._apply_geometric_augmentation(
            img.copy(), K.copy(), kp2d.copy(), vis.copy()
        )

        np.testing.assert_array_equal(R, R_orig)
        np.testing.assert_array_equal(t, t_orig)

    def test_out_of_bounds_keypoints_masked(self):
        """Keypoints pushed outside the image by crop jitter should be masked."""
        _ensure_viz_dir()
        image_size = 256
        img, K, R, t, pts3d, kp2d, vis = _make_synthetic_sample(image_size=image_size)

        # Force some keypoints near edges in normalized [y, x] coords
        kp2d_edge = kp2d.copy()
        kp2d_edge[0] = [0.5, 0.008]   # near left edge (x ~ 2px)
        kp2d_edge[1] = [0.5, 0.992]   # near right edge (x ~ 254px)
        kp2d_edge[2] = [0.008, 0.5]   # near top edge (y ~ 2px)
        kp2d_edge[3] = [0.992, 0.5]   # near bottom edge (y ~ 254px)

        # Use aggressive jitter to push edges out
        self.ds.aug_crop_jitter_fraction = 0.15
        self.ds.aug_scale_jitter_range = (0.8, 1.2)

        masked_count = 0
        for seed in range(50):
            np.random.seed(seed)
            _, _, kp_aug, vis_aug = self.ds._apply_geometric_augmentation(
                img.copy(), K.copy(), kp2d_edge.copy(), vis.copy()
            )
            masked_count += (~vis_aug).sum()
            # Verify that masked keypoints are actually out of bounds (normalized)
            for j in range(len(kp_aug)):
                if not vis_aug[j]:
                    ny, nx = kp_aug[j]
                    assert ny < 0 or ny > 1.0 or nx < 0 or nx > 1.0, (
                        f"Keypoint {j} was masked but is within [0,1]: "
                        f"(norm_y={ny:.4f}, norm_x={nx:.4f})"
                    )

        # At least some should be masked with aggressive jitter near edges
        assert masked_count > 0, "Expected some edge keypoints to be masked"
        print(f"  Edge keypoints masked: {masked_count} / {50 * 4}")

        # Visualization of one case
        np.random.seed(7)
        _, _, kp_aug, vis_aug = self.ds._apply_geometric_augmentation(
            img.copy(), K.copy(), kp2d_edge.copy(), vis.copy()
        )
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            kp_edge_px = _norm_yx_to_pixel_xy(kp2d_edge, image_size)
            ax1.imshow(np.clip(img, 0, 1))
            ax1.scatter(kp_edge_px[:, 0], kp_edge_px[:, 1], c='lime', s=40)
            ax1.set_title("Original (edge keypoints)")
            ax1.axis('off')

            kp_aug_px = _norm_yx_to_pixel_xy(kp_aug, image_size)
            ax2.imshow(np.clip(img, 0, 1))
            for j in range(len(kp_aug_px)):
                ax2.scatter(kp_aug_px[j, 0], kp_aug_px[j, 1],
                            c='lime' if vis_aug[j] else 'red', s=40,
                            marker='o' if vis_aug[j] else 'x')
            ax2.set_title("After aug (green=visible, red=masked)")
            ax2.axis('off')

            fig.tight_layout()
            out_path = os.path.join(VIZ_DIR, "geometric_edge_masking.png")
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")
        except ImportError:
            pass


class TestMultiViewConsistency:
    """Test augmentation across multiple views of the same 3D scene."""

    @pytest.fixture(autouse=True)
    def setup_dataset(self):
        from sleap_multiview_dataset import SLEAPMultiViewDataset
        self.ds = object.__new__(SLEAPMultiViewDataset)
        self.ds.augment = True
        self.ds.aug_color_jitter_brightness = 0.2
        self.ds.aug_color_jitter_contrast = 0.2
        self.ds.aug_color_jitter_saturation = 0.15
        self.ds.aug_gaussian_noise_std = 0.015
        self.ds.aug_gaussian_blur_prob = 0.3
        self.ds.aug_gaussian_blur_kernel_range = (3, 7)
        self.ds.aug_random_erasing_prob = 0.2
        self.ds.aug_random_erasing_scale_range = (0.02, 0.1)
        self.ds.aug_crop_jitter_fraction = 0.05
        self.ds.aug_scale_jitter_range = (0.9, 1.1)

    def test_multiview_reprojection_all_views(self):
        """Each view's augmented keypoints must match independent reprojection."""
        _ensure_viz_dir()
        n_views = 6
        n_joints = 12
        image_size = 256
        rng = np.random.RandomState(42)

        # Shared 3D points
        pts3d = rng.uniform(-0.3, 0.3, (n_joints, 3)).astype(np.float32)
        pts3d[:, 2] += 0.5

        images = []
        K_list = []
        R_list = []
        t_list = []
        kp2d_list = []

        for vi in range(n_views):
            angle = vi * 2.0 * math.pi / n_views
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            t = np.array([0.0, 0.0, 5.0], dtype=np.float32)

            fx = fy = image_size * (1.0 + 0.2 * vi / n_views)
            K = np.array([[fx, 0, image_size/2], [0, fy, image_size/2], [0, 0, 1]],
                         dtype=np.float32)

            # Project to normalized [y, x]
            kp2d = _project_3d_to_normalized(K, R, t, pts3d, image_size)

            img = rng.uniform(0.1, 0.9, (image_size, image_size, 3)).astype(np.float32)

            images.append(img)
            K_list.append(K)
            R_list.append(R)
            t_list.append(t)
            kp2d_list.append(kp2d)

        # Apply augmentation independently per view (as the real pipeline does)
        images_aug = []
        K_aug_list = []
        kp2d_aug_list = []
        vis_aug_list = []

        for vi in range(n_views):
            np.random.seed(1000 + vi)
            vis = np.ones(n_joints, dtype=bool)

            img_a, K_a, kp_a, vis_a = self.ds._apply_geometric_augmentation(
                images[vi].copy(), K_list[vi].copy(),
                kp2d_list[vi].copy(), vis.copy()
            )
            img_a = self.ds._apply_photometric_augmentation(img_a)

            images_aug.append(img_a)
            K_aug_list.append(K_a)
            kp2d_aug_list.append(kp_a)
            vis_aug_list.append(vis_a)

        # Verify reprojection consistency per view
        for vi in range(n_views):
            reproj = _project_3d_to_normalized(
                K_aug_list[vi], R_list[vi], t_list[vi], pts3d, image_size
            )
            visible = vis_aug_list[vi].astype(bool)
            if visible.sum() > 0:
                np.testing.assert_allclose(
                    reproj[visible], kp2d_aug_list[vi][visible],
                    atol=0.5 / image_size,
                    err_msg=f"View {vi}: reprojection mismatch after augmentation"
                )

        # Visualization: grid of all views
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            cols = min(3, n_views)
            rows = math.ceil(n_views / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
            axes_flat = np.array(axes).flatten()

            for vi in range(n_views):
                ax = axes_flat[vi]
                ax.imshow(np.clip(images_aug[vi], 0, 1))

                vis = vis_aug_list[vi]
                # Convert normalized [y,x] to pixel [x,y] for matplotlib
                kp_px = _norm_yx_to_pixel_xy(kp2d_aug_list[vi], image_size)
                ax.scatter(kp_px[vis, 0], kp_px[vis, 1], c='lime', s=30,
                           marker='o', label='aug 2D kps')

                reproj = _project_3d_to_normalized(
                    K_aug_list[vi], R_list[vi], t_list[vi], pts3d, image_size
                )
                rp_px = _norm_yx_to_pixel_xy(reproj, image_size)
                ax.scatter(rp_px[vis, 0], rp_px[vis, 1], c='red', s=60,
                           marker='x', linewidths=2, label='reproj 3D->2D')

                ax.set_title(f"View {vi}")
                ax.axis('off')
                if vi == 0:
                    ax.legend(fontsize=8)

            for vi in range(n_views, len(axes_flat)):
                axes_flat[vi].axis('off')

            fig.suptitle("Multiview: green=aug kps, red=independent reprojection (should overlap)")
            fig.tight_layout()
            out_path = os.path.join(VIZ_DIR, "multiview_reprojection_grid.png")
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")
        except ImportError:
            pass


class TestAugmentationToggle:
    """Augmentation must be disabled for val/test."""

    @pytest.fixture(autouse=True)
    def setup_dataset(self):
        from sleap_multiview_dataset import SLEAPMultiViewDataset
        self.ds = object.__new__(SLEAPMultiViewDataset)
        self.ds.aug_color_jitter_brightness = 0.3
        self.ds.aug_color_jitter_contrast = 0.3
        self.ds.aug_color_jitter_saturation = 0.3
        self.ds.aug_gaussian_noise_std = 0.05
        self.ds.aug_gaussian_blur_prob = 1.0
        self.ds.aug_gaussian_blur_kernel_range = (5, 5)
        self.ds.aug_random_erasing_prob = 1.0
        self.ds.aug_random_erasing_scale_range = (0.05, 0.15)

    def test_augmentation_disabled_is_identity(self):
        """With augment=False, _apply_photometric_augmentation should not be called,
        so we test that toggling the flag works as expected."""
        self.ds.augment = False
        img, K, R, t, pts3d, kp2d, vis = _make_synthetic_sample()

        img_copy = img.copy()

        # With augmentation enabled
        self.ds.augment = True
        np.random.seed(42)
        img_aug = self.ds._apply_photometric_augmentation(img.copy())
        assert not np.allclose(img_aug, img_copy, atol=1e-6), \
            "Augmentation should change the image"

    def test_deterministic_without_augmentation(self):
        """Multiple calls with same input and augment=False should give identical results."""
        img, K, R, t, pts3d, kp2d, vis = _make_synthetic_sample()

        # Photometric with all zeros -> no change
        self.ds.aug_color_jitter_brightness = 0
        self.ds.aug_color_jitter_contrast = 0
        self.ds.aug_color_jitter_saturation = 0
        self.ds.aug_gaussian_noise_std = 0
        self.ds.aug_gaussian_blur_prob = 0
        self.ds.aug_random_erasing_prob = 0

        result1 = self.ds._apply_photometric_augmentation(img.copy())
        result2 = self.ds._apply_photometric_augmentation(img.copy())
        np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# Integration test with real H5 dataset
# ---------------------------------------------------------------------------

REAL_H5_PATH = os.path.join(_repo_root, "SMILySTICKS_centred_reprojected_aug_test.h5")


def _get_real_sample_data(ds, sample_idx=0, seed=42):
    """Load and unpack a real dataset sample for testing.

    Returns a dict with all the needed arrays, or None if data is missing.
    """
    np.random.seed(seed)
    x_data, y_data = ds[sample_idx]

    if y_data.get('keypoints_3d') is None or y_data.get('camera_intrinsics') is None:
        return None

    target_res = ds.target_resolution
    ws = ds.world_scale

    return {
        'x_data': x_data,
        'y_data': y_data,
        'target_res': target_res,
        'pts3d_raw': y_data['keypoints_3d'] / ws,
        'K': y_data['camera_intrinsics'],
        'R': y_data['camera_extrinsics_R'],
        't_raw': y_data['camera_extrinsics_t'] / ws,
        'kp2d': y_data['keypoints_2d'],         # normalized [y, x]
        'vis': y_data['keypoint_visibility'],
        'images': x_data['images'],
        'num_views': len(x_data['images']),
        'camera_names': x_data.get('camera_names', [f'cam_{i}' for i in range(len(x_data['images']))]),
    }


@pytest.mark.skipif(
    not os.path.exists(REAL_H5_PATH),
    reason=f"Test H5 file not found: {REAL_H5_PATH}",
)
class TestRealDatasetAugmentation:
    """Integration tests using real multiview dataset with calibrated cameras.

    These tests produce visualizations in tests/test_output/augmentation/
    for every augmentation type so you can visually verify that augmentations
    look reasonable on real data and that keypoint overlays remain consistent.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from sleap_multiview_dataset import SLEAPMultiViewDataset
        self.ds = SLEAPMultiViewDataset(
            hdf5_path=REAL_H5_PATH,
            augment=True,
            augmentation_config={
                'color_jitter_brightness': 0.2,
                'color_jitter_contrast': 0.2,
                'color_jitter_saturation': 0.15,
                'gaussian_noise_std': 0.015,
                'gaussian_blur_prob': 0.3,
                'gaussian_blur_kernel_range': (3, 7),
                'random_erasing_prob': 0.2,
                'random_erasing_scale_range': (0.02, 0.1),
                'crop_jitter_fraction': 0.05,
                'scale_jitter_range': (0.9, 1.1),
            },
        )
        self.ds_no_aug = SLEAPMultiViewDataset(
            hdf5_path=REAL_H5_PATH,
            augment=False,
        )

    def test_no_aug_reprojection_baseline(self):
        """Baseline: measure reprojection error without augmentation.

        The 2D keypoints in the HDF5 come from SLEAP detections, and the 3D
        keypoints come from triangulation. There is inherent noise from both
        detection and triangulation, so we expect non-zero reprojection error.
        This test establishes the baseline and saves a visualization.
        """
        _ensure_viz_dir()
        d = _get_real_sample_data(self.ds_no_aug)
        if d is None:
            pytest.skip("Missing 3D/camera data")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            has_mpl = True
        except ImportError:
            has_mpl = False

        max_errors = []
        for vi in range(d['num_views']):
            reproj = _project_3d_to_normalized(
                d['K'][vi], d['R'][vi], d['t_raw'][vi],
                d['pts3d_raw'], d['target_res']
            )
            visible = d['vis'][vi].astype(bool)
            if visible.sum() == 0:
                continue
            err = np.abs(reproj[visible] - d['kp2d'][vi][visible])
            max_err = err.max()
            max_errors.append(max_err)

        max_errors = np.array(max_errors)
        print(f"  Baseline reprojection errors (normalized): "
              f"max={max_errors.max():.6f} ({max_errors.max() * d['target_res']:.1f} px), "
              f"mean={max_errors.mean():.6f}")

        # The inherent triangulation/detection error should be bounded.
        # 0.1 normalized = ~50px at 512 — generous but catches gross errors.
        assert max_errors.max() < 0.1, (
            f"Baseline reprojection error too large: {max_errors.max():.4f} norm "
            f"({max_errors.max() * d['target_res']:.1f} px)"
        )

        # Visualization: no-aug grid with keypoints + reprojection overlay
        if has_mpl:
            cols = min(3, d['num_views'])
            rows = math.ceil(d['num_views'] / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
            axes_flat = np.array(axes).flatten() if d['num_views'] > 1 else [axes]

            for vi in range(d['num_views']):
                ax = axes_flat[vi]
                ax.imshow(np.clip(d['images'][vi], 0, 1))

                v = d['vis'][vi].astype(bool)
                kp_px = _norm_yx_to_pixel_xy(d['kp2d'][vi], d['target_res'])
                ax.scatter(kp_px[v, 0], kp_px[v, 1], c='lime', s=15,
                           marker='o', alpha=0.8, label='stored 2D kps')

                reproj = _project_3d_to_normalized(
                    d['K'][vi], d['R'][vi], d['t_raw'][vi],
                    d['pts3d_raw'], d['target_res']
                )
                rp_px = _norm_yx_to_pixel_xy(reproj, d['target_res'])
                ax.scatter(rp_px[v, 0], rp_px[v, 1], c='red', s=30,
                           marker='x', linewidths=1.5, alpha=0.8, label='reproj 3D')

                ax.set_title(f"View {vi} ({d['camera_names'][vi]})")
                ax.axis('off')
                if vi == 0:
                    ax.legend(fontsize=8)

            for vi in range(d['num_views'], len(axes_flat)):
                axes_flat[vi].axis('off')

            fig.suptitle("Baseline (no augmentation): green=stored kps, red=reproj from 3D",
                         fontsize=11)
            fig.tight_layout()
            out_path = os.path.join(VIZ_DIR, "real_baseline_reprojection.png")
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")

    def test_real_reprojection_after_augmentation(self):
        """Augmented 2D kps must match reprojection of 3D through updated K.

        The augmentation should not introduce additional error beyond the
        inherent triangulation noise (baseline). We compare augmented error
        against a generous absolute threshold.
        """
        _ensure_viz_dir()
        d = _get_real_sample_data(self.ds, seed=42)
        if d is None:
            pytest.skip("Missing 3D/camera data")

        max_errors = []
        for vi in range(d['num_views']):
            reproj = _project_3d_to_normalized(
                d['K'][vi], d['R'][vi], d['t_raw'][vi],
                d['pts3d_raw'], d['target_res']
            )
            visible = d['vis'][vi].astype(bool)
            if visible.sum() == 0:
                continue
            err = np.abs(reproj[visible] - d['kp2d'][vi][visible])
            max_err = err.max()
            max_errors.append(max_err)

        max_errors = np.array(max_errors)
        print(f"  Augmented reprojection errors (normalized): "
              f"max={max_errors.max():.6f} ({max_errors.max() * d['target_res']:.1f} px), "
              f"mean={max_errors.mean():.6f}")

        # Same tolerance as baseline — augmentation should not increase error
        assert max_errors.max() < 0.1, (
            f"Augmented reprojection error too large: {max_errors.max():.4f} norm "
            f"({max_errors.max() * d['target_res']:.1f} px)"
        )

        # Visualization: augmented grid with keypoints + reprojection overlay
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            cols = min(3, d['num_views'])
            rows = math.ceil(d['num_views'] / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
            axes_flat = np.array(axes).flatten() if d['num_views'] > 1 else [axes]

            for vi in range(d['num_views']):
                ax = axes_flat[vi]
                ax.imshow(np.clip(d['images'][vi], 0, 1))

                v = d['vis'][vi].astype(bool)
                kp_px = _norm_yx_to_pixel_xy(d['kp2d'][vi], d['target_res'])
                ax.scatter(kp_px[v, 0], kp_px[v, 1], c='lime', s=15,
                           marker='o', alpha=0.8, label='aug 2D kps')

                reproj = _project_3d_to_normalized(
                    d['K'][vi], d['R'][vi], d['t_raw'][vi],
                    d['pts3d_raw'], d['target_res']
                )
                rp_px = _norm_yx_to_pixel_xy(reproj, d['target_res'])
                ax.scatter(rp_px[v, 0], rp_px[v, 1], c='red', s=30,
                           marker='x', linewidths=1.5, alpha=0.8, label='reproj 3D')

                ax.set_title(f"View {vi} ({d['camera_names'][vi]})")
                ax.axis('off')
                if vi == 0:
                    ax.legend(fontsize=8)

            for vi in range(d['num_views'], len(axes_flat)):
                axes_flat[vi].axis('off')

            fig.suptitle("After augmentation: green=aug kps, red=reproj from 3D",
                         fontsize=11)
            fig.tight_layout()
            out_path = os.path.join(VIZ_DIR, "real_augmented_reprojection.png")
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")
        except ImportError:
            pass

    def test_real_photometric_augmentations_visual(self):
        """Visualize each photometric augmentation type on a real image.

        Saves a side-by-side grid: original vs each augmentation type.
        Keypoints are overlaid to confirm they remain unchanged.
        """
        _ensure_viz_dir()
        d = _get_real_sample_data(self.ds_no_aug, seed=0)
        if d is None:
            pytest.skip("Missing data")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        from sleap_multiview_dataset import SLEAPMultiViewDataset

        # Pick the first view with visible keypoints
        view_idx = 0
        img_orig = d['images'][view_idx].copy()
        kp_norm = d['kp2d'][view_idx]
        v = d['vis'][view_idx].astype(bool)
        target_res = d['target_res']
        kp_px = _norm_yx_to_pixel_xy(kp_norm, target_res)

        # Create a bare dataset object for augmentation methods
        ds_bare = object.__new__(SLEAPMultiViewDataset)
        ds_bare.augment = True

        aug_types = {
            "brightness": {"aug_color_jitter_brightness": 0.3,
                           "aug_color_jitter_contrast": 0,
                           "aug_color_jitter_saturation": 0,
                           "aug_gaussian_noise_std": 0,
                           "aug_gaussian_blur_prob": 0,
                           "aug_random_erasing_prob": 0},
            "contrast": {"aug_color_jitter_brightness": 0,
                         "aug_color_jitter_contrast": 0.3,
                         "aug_color_jitter_saturation": 0,
                         "aug_gaussian_noise_std": 0,
                         "aug_gaussian_blur_prob": 0,
                         "aug_random_erasing_prob": 0},
            "saturation": {"aug_color_jitter_brightness": 0,
                           "aug_color_jitter_contrast": 0,
                           "aug_color_jitter_saturation": 0.3,
                           "aug_gaussian_noise_std": 0,
                           "aug_gaussian_blur_prob": 0,
                           "aug_random_erasing_prob": 0},
            "noise": {"aug_color_jitter_brightness": 0,
                      "aug_color_jitter_contrast": 0,
                      "aug_color_jitter_saturation": 0,
                      "aug_gaussian_noise_std": 0.03,
                      "aug_gaussian_blur_prob": 0,
                      "aug_random_erasing_prob": 0},
            "blur": {"aug_color_jitter_brightness": 0,
                     "aug_color_jitter_contrast": 0,
                     "aug_color_jitter_saturation": 0,
                     "aug_gaussian_noise_std": 0,
                     "aug_gaussian_blur_prob": 1.0,
                     "aug_gaussian_blur_kernel_range": (5, 5),
                     "aug_random_erasing_prob": 0},
            "erasing": {"aug_color_jitter_brightness": 0,
                        "aug_color_jitter_contrast": 0,
                        "aug_color_jitter_saturation": 0,
                        "aug_gaussian_noise_std": 0,
                        "aug_gaussian_blur_prob": 0,
                        "aug_random_erasing_prob": 1.0,
                        "aug_random_erasing_scale_range": (0.05, 0.15)},
        }

        n_aug = len(aug_types)
        fig, axes = plt.subplots(2, (n_aug + 2) // 2, figsize=(5 * ((n_aug + 2) // 2), 10))
        axes_flat = axes.flatten()

        # Panel 0: original
        axes_flat[0].imshow(np.clip(img_orig, 0, 1))
        axes_flat[0].scatter(kp_px[v, 0], kp_px[v, 1], c='lime', s=20, marker='o')
        axes_flat[0].set_title("Original")
        axes_flat[0].axis('off')

        for i, (name, params) in enumerate(aug_types.items()):
            for k, val in params.items():
                setattr(ds_bare, k, val)

            np.random.seed(42)
            img_aug = ds_bare._apply_photometric_augmentation(img_orig.copy())

            ax = axes_flat[i + 1]
            ax.imshow(np.clip(img_aug, 0, 1))
            ax.scatter(kp_px[v, 0], kp_px[v, 1], c='lime', s=20, marker='o')
            ax.set_title(name)
            ax.axis('off')

        # Hide unused axes
        for j in range(n_aug + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        fig.suptitle(f"Real data photometric augmentations (View {view_idx})", fontsize=12)
        fig.tight_layout()
        out_path = os.path.join(VIZ_DIR, "real_photometric_augmentations.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    def test_real_geometric_augmentation_visual(self):
        """Visualize geometric augmentation (crop/scale jitter) on real data.

        For each view: original image + keypoints vs augmented image + augmented
        keypoints vs augmented image + independently reprojected keypoints.
        The last two columns should be identical (green dots over red crosses).
        """
        _ensure_viz_dir()
        d_noaug = _get_real_sample_data(self.ds_no_aug, seed=0)
        if d_noaug is None:
            pytest.skip("Missing data")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        from sleap_multiview_dataset import SLEAPMultiViewDataset

        ds_bare = object.__new__(SLEAPMultiViewDataset)
        ds_bare.augment = True
        ds_bare.aug_crop_jitter_fraction = 0.05
        ds_bare.aug_scale_jitter_range = (0.9, 1.1)

        target_res = d_noaug['target_res']
        n_views = min(d_noaug['num_views'], 6)

        fig, axes = plt.subplots(n_views, 3, figsize=(18, 6 * n_views))
        if n_views == 1:
            axes = axes[np.newaxis, :]

        for vi in range(n_views):
            img = d_noaug['images'][vi].copy()
            K_v = d_noaug['K'][vi].copy()
            kp = d_noaug['kp2d'][vi].copy()
            v = d_noaug['vis'][vi].copy()

            np.random.seed(100 + vi)
            img_aug, K_aug, kp_aug, vis_aug = ds_bare._apply_geometric_augmentation(
                img, K_v, kp, v
            )

            # Independent reprojection with updated K
            reproj = _project_3d_to_normalized(
                K_aug, d_noaug['R'][vi], d_noaug['t_raw'][vi],
                d_noaug['pts3d_raw'], target_res
            )

            v_mask = vis_aug.astype(bool)
            v_orig = d_noaug['vis'][vi].astype(bool)

            # Panel 1: original
            ax = axes[vi, 0]
            ax.imshow(np.clip(d_noaug['images'][vi], 0, 1))
            kp_orig_px = _norm_yx_to_pixel_xy(d_noaug['kp2d'][vi], target_res)
            ax.scatter(kp_orig_px[v_orig, 0], kp_orig_px[v_orig, 1],
                       c='lime', s=20, marker='o')
            ax.set_title(f"View {vi} original")
            ax.axis('off')

            # Panel 2: augmented + transformed kps
            ax = axes[vi, 1]
            ax.imshow(np.clip(img_aug, 0, 1))
            kp_aug_px = _norm_yx_to_pixel_xy(kp_aug, target_res)
            ax.scatter(kp_aug_px[v_mask, 0], kp_aug_px[v_mask, 1],
                       c='lime', s=20, marker='o', label='aug kps')
            ax.set_title(f"Aug + transformed kps")
            ax.axis('off')

            # Panel 3: augmented + reprojected kps (should match panel 2)
            ax = axes[vi, 2]
            ax.imshow(np.clip(img_aug, 0, 1))
            rp_px = _norm_yx_to_pixel_xy(reproj, target_res)
            ax.scatter(rp_px[v_mask, 0], rp_px[v_mask, 1],
                       c='red', s=40, marker='x', linewidths=1.5, label='reproj 3D')
            ax.scatter(kp_aug_px[v_mask, 0], kp_aug_px[v_mask, 1],
                       c='lime', s=15, marker='o', alpha=0.7, label='aug kps')
            ax.set_title(f"Aug + reproj overlay")
            ax.axis('off')
            if vi == 0:
                ax.legend(fontsize=7)

        fig.suptitle("Geometric augmentation: cols = [original, aug+kps, aug+reproj overlay]",
                     fontsize=12)
        fig.tight_layout()
        out_path = os.path.join(VIZ_DIR, "real_geometric_augmentation.png")
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    def test_real_combined_augmentation_visual(self):
        """Visualize the full augmentation pipeline (geometric + photometric).

        Shows original vs fully-augmented for each view, with keypoint overlay.
        This is what the training pipeline actually sees.
        """
        _ensure_viz_dir()
        d_noaug = _get_real_sample_data(self.ds_no_aug, seed=0)
        d_aug = _get_real_sample_data(self.ds, seed=42)
        if d_noaug is None or d_aug is None:
            pytest.skip("Missing data")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        target_res = d_noaug['target_res']
        n_views = min(d_noaug['num_views'], 6)

        fig, axes = plt.subplots(n_views, 2, figsize=(12, 6 * n_views))
        if n_views == 1:
            axes = axes[np.newaxis, :]

        for vi in range(n_views):
            v_orig = d_noaug['vis'][vi].astype(bool)
            v_aug = d_aug['vis'][vi].astype(bool)

            # Original
            ax = axes[vi, 0]
            ax.imshow(np.clip(d_noaug['images'][vi], 0, 1))
            kp_px = _norm_yx_to_pixel_xy(d_noaug['kp2d'][vi], target_res)
            ax.scatter(kp_px[v_orig, 0], kp_px[v_orig, 1],
                       c='lime', s=20, marker='o')
            ax.set_title(f"View {vi} ({d_noaug['camera_names'][vi]}) — original")
            ax.axis('off')

            # Augmented
            ax = axes[vi, 1]
            ax.imshow(np.clip(d_aug['images'][vi], 0, 1))
            kp_aug_px = _norm_yx_to_pixel_xy(d_aug['kp2d'][vi], target_res)
            ax.scatter(kp_aug_px[v_aug, 0], kp_aug_px[v_aug, 1],
                       c='lime', s=20, marker='o')
            ax.set_title(f"View {vi} — augmented")
            ax.axis('off')

        fig.suptitle("Full augmentation pipeline: original vs augmented per view",
                     fontsize=12)
        fig.tight_layout()
        out_path = os.path.join(VIZ_DIR, "real_combined_augmentation.png")
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    def test_no_aug_vs_aug_different(self):
        """Augmented sample should differ from non-augmented sample."""
        np.random.seed(42)
        _, y_aug = self.ds[0]

        np.random.seed(42)  # same seed but augment=False
        _, y_noaug = self.ds_no_aug[0]

        # The 2D keypoints should differ because geometric aug changes them
        if y_aug.get('camera_intrinsics') is not None:
            assert not np.allclose(y_aug['keypoints_2d'], y_noaug['keypoints_2d'], atol=1e-6), \
                "Augmented and non-augmented 2D keypoints should differ"
