"""Visual validation for multi-view replicAnt camera geometry + model fit.

Two checks per frame:

1. **Camera geometry (always run)**: for each view, read raw 3D world
   keypoints, raw 2D keypoints, and raw extrinsics+intrinsics from that
   camera's JSON. Project the 3D through (R, t, K) and overlay against
   the raw 2D. Green (GT 2D) and red (projected 3D) should coincide
   pixel-for-pixel.

2. **Model-fit compatibility (only when --smal_file is passed)**: load
   the specified SMAL pickle via apply_smal_file_override and report
   how the model's ``J_names`` compare to the dataset's joint names —
   overlap count, dataset-only joints, model-only joints. Also runs
   ``load_SMIL_Unreal_multiview_sample`` end-to-end and reports root /
   shape / joint angle extraction.

Usage:
    # geometry only
    python tests/validate_multiview_replicant_loader.py \\
        --dataset_path /mnt/c/replicAnt-dataset-multi-cam-mice \\
        --frame_index 0

    # geometry + model-fit compatibility
    python tests/validate_multiview_replicant_loader.py \\
        --dataset_path /mnt/c/replicAnt-dataset-multi-cam-mice \\
        --frame_index 0 \\
        --smal_file 3D_model_prep/SMILy_Mouse_static_joints_Falkner_conv_repose_hind_legs_fix_eyes.pkl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")  # No display in WSL.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Parse --smal_file first so we can apply the override *before* importing
# Unreal2Pytorch3D (which pulls in `config.dd`).
def _early_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--smal_file", default=None)
    p.add_argument("--shape_family", type=int, default=None)
    known, _ = p.parse_known_args()
    return known


_early = _early_parse()
if _early.smal_file:
    from smal_fitter.neuralSMIL.configs.config_utils import apply_smal_file_override  # noqa: E402
    apply_smal_file_override(_early.smal_file, shape_family=_early.shape_family)

import config  # noqa: E402
from smal_fitter.Unreal2Pytorch3D import (  # noqa: E402
    load_SMIL_Unreal_multiview_sample,
    parse_camera_intrinsics,
    parse_projection_components,
    return_placeholder_data,
    sample_pca_transforms_from_dirs,
    set_axes_equal,
)

# Rendering deps (heavy — only used when --render is passed).
_TORCH = None
_SMALFitter = None
_ImageExporter = None
_R_scipy = None


def _lazy_render_imports():
    global _TORCH, _SMALFitter, _ImageExporter, _R_scipy
    if _TORCH is not None:
        return
    import torch as _t  # noqa: F401
    from smal_fitter.fitter import SMALFitter as _SF  # noqa: F401
    from smal_fitter.optimize_to_joints import ImageExporter as _IE  # noqa: F401
    from scipy.spatial.transform import Rotation as _R  # noqa: F401
    _TORCH = _t
    _SMALFitter = _SF
    _ImageExporter = _IE
    _R_scipy = _R


def project_row_vector(X_world, R, t, fx, fy, cx, cy):
    """Project N world-space points through (R, t, K).

    Tries a small grid of (R | R.T) × (depth axis: x | y | z) × (sign: +/-)
    conventions and picks the one minimising pixel error against ground-truth
    midway through the script's loop. For exploration we return all variants;
    the caller scores them externally.
    """
    X_world = np.asarray(X_world, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    variants = {}
    for r_name, R_use in (("R", R), ("Rt", R.T)):
        X_cam_base = X_world @ R_use + t
        for axis_name, axis_idx in (("x", 0), ("y", 1), ("z", 2)):
            for dsign in (1.0, -1.0):
                depth = dsign * X_cam_base[:, axis_idx]
                other = [i for i in (0, 1, 2) if i != axis_idx]
                for usign in (1.0, -1.0):
                    for vsign in (1.0, -1.0):
                        u_world = usign * X_cam_base[:, other[0]]
                        v_world = vsign * X_cam_base[:, other[1]]
                        safe = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
                        u = fx * u_world / safe + cx
                        v = fy * v_world / safe + cy
                        key = (
                            f"{r_name}_d={dsign:+.0f}{axis_name}"
                            f"_u={usign:+.0f}{'xyz'[other[0]]}"
                            f"_v={vsign:+.0f}{'xyz'[other[1]]}"
                        )
                        variants[key] = (u, v, depth)
    return variants


def list_camera_ids(dataset_path: Path, dataset_name: str, frame_index: int):
    pattern = re.compile(rf"^{re.escape(dataset_name)}_{frame_index:05d}_CAM(\d+)\.json$")
    ids = []
    for entry in dataset_path.iterdir():
        m = pattern.match(entry.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_per_camera(dataset_path: Path, dataset_name: str, frame_index: int, cam_id: int, batch_data_file):
    json_path = dataset_path / f"{dataset_name}_{frame_index:05d}_CAM{cam_id}.json"
    with open(json_path) as f:
        cam_data = json.load(f)
    pose_data = cam_data["iterationData"]["subject Data"][0]["1"]["keypoints"]
    names = list(pose_data.keys())

    kp3d = np.zeros((len(names), 3), dtype=np.float64)
    kp2d = np.zeros((len(names), 2), dtype=np.float64)
    for i, name in enumerate(names):
        p3 = pose_data[name].get("3DPos")
        p2 = pose_data[name].get("2DPos")
        if p3 is not None:
            kp3d[i] = [p3["x"], p3["y"], p3["z"]]
        if p2 is not None:
            kp2d[i] = [p2["x"], p2["y"]]

    R, t = parse_projection_components(cam_data)
    cx, cy, fx, fy = parse_camera_intrinsics(
        batch_data_file=batch_data_file, iteration_data_file=cam_data
    )

    image_path = json_path.with_suffix(".JPG")
    img = imageio.imread(image_path) if image_path.exists() else None

    return {
        "names": names,
        "kp3d_world": kp3d,
        "kp2d_pixels": kp2d,
        "R": R,
        "t": t,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image": img,
    }


def report_joint_name_overlap(dataset_names, model_names):
    ds = list(dict.fromkeys(dataset_names))  # preserve order, drop duplicates
    md = list(dict.fromkeys(model_names))
    ds_set, md_set = set(ds), set(md)
    overlap = [n for n in ds if n in md_set]
    ds_only = [n for n in ds if n not in md_set]
    md_only = [n for n in md if n not in ds_set]
    print("\n--- Joint name overlap (dataset vs. loaded SMAL model J_names) ---")
    print(f"  Dataset joints: {len(ds)}")
    print(f"  Model J_names:  {len(md)}")
    print(f"  Overlap:        {len(overlap)}/{max(len(ds), len(md))}  "
          f"(perfect-match needs both numbers to equal overlap)")
    if ds_only:
        print(f"  Dataset-only ({len(ds_only)}): {ds_only}")
    if md_only:
        print(f"  Model-only   ({len(md_only)}): {md_only}")
    if not ds_only and not md_only:
        print("  ✓ Joint sets are identical.")


MIRROR_MAT = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def _build_model_world_rotation(pose_data, root_key):
    """Compute R_model_p3d from the root joint's globalRotation, matching the
    single-view path at Unreal2Pytorch3D.py:998-1008.
    """
    grot = pose_data[root_key]["globalRotation"]
    rot_model_ue = _R_scipy.from_quat(
        [-grot["x"], -grot["y"], -grot["z"], grot["w"]], scalar_first=False
    )
    R_model_ue = rot_model_ue.as_matrix().astype(np.float32)
    return MIRROR_MAT @ R_model_ue @ MIRROR_MAT.T


def _reparameterize_view(R_model_p3d, t_model_p3d, R_v, T_v):
    """Per-view single-view-style reparameterisation:
        R_cam_new = Rz(180°) @ R_model @ R_v
        T_cam_new = t_model @ R_v + T_v
    Mirrors Unreal2Pytorch3D.py:1015-1026 (note: Rz applied to R only, not to T —
    matches the single-view convention).
    """
    Rz = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    R_new = (Rz @ R_model_p3d @ R_v).astype(np.float32)
    T_new = (t_model_p3d @ R_v + T_v).astype(np.float32)
    return R_new, T_new


def render_per_view(dataset_path, frame_index, output_dir,
                    canonical_frame=True, label="canonical"):
    """Mirror Render_SMAL_Model_from_Unreal_data for each view, write a grid PNG.

    Strategy: load multi-view data, derive the shared model-world transform,
    then for every view build a single-view-style (x_v, y_v) with the
    per-view reparameterisation applied and run the SMALFitter visualisation.
    Collect each view's collage from disk and stitch into one composite.

    canonical_frame controls the loader call. When True, R_model_p3d derived
    from raw pose_data quaternion is right-multiplied by R_0 to compose
    body->world with world->canonical — the renderer formula
    R_new = Rz @ R_model_p3d @ R_v, T_new = t_model @ R_v + T_v is then
    pixel-identical under the canonical transformation (see derivation
    in Unreal2Pytorch3D.py:load_SMIL_Unreal_multiview_sample).

    Returns the list of (cam_id, collage_path) so callers (e.g. the
    --compare_frames stitcher) can assemble combined plots.
    """
    _lazy_render_imports()
    torch = _TORCH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_mv, y_mv = load_SMIL_Unreal_multiview_sample(
        data_path=str(dataset_path),
        frame_index=frame_index,
        camera_indices=None,
        load_images=True,
        canonical_frame=canonical_frame,
        verbose=False,
    )
    pose_data = y_mv["pose_data"]
    root_key = config.ROOT_JOINT if config.ROOT_JOINT in pose_data else next(iter(pose_data))

    R_model_p3d = _build_model_world_rotation(pose_data, root_key)
    if canonical_frame:
        R_0_np, _ = y_mv["canonical_to_world"]
        R_model_p3d = (R_model_p3d @ R_0_np).astype(np.float32)
    t_model_p3d = np.asarray(y_mv["root_loc"], dtype=np.float32)

    print(f"\n--- Per-view rendering ({label}) ---")
    print(f"  root_key={root_key!r}  t_model={t_model_p3d}  canonical_cam={y_mv.get('canonical_cam_id')}")

    render_root = (Path(output_dir) / f"frame_{frame_index:05d}_renders_{label}").resolve()
    render_root.mkdir(parents=True, exist_ok=True)

    # Render_SMAL_Model_from_Unreal_data writes via ImageExporter("LOCAL_TEST", …)
    # to the *current* working directory. To redirect the output without
    # touching the Unreal2Pytorch3D module, we cd into render_root for the
    # duration of the renders. Critical: absolutise any relative paths in
    # config that the renderer will try to open (e.g. SMAL_FILE) before chdir.
    if not os.path.isabs(config.SMAL_FILE):
        config.SMAL_FILE = str((Path(prev_cwd_root := os.getcwd()) / config.SMAL_FILE).resolve())
    prev_cwd = os.getcwd()
    os.chdir(render_root)
    rendered_paths = []
    try:
        for v, cam_id in enumerate(x_mv["camera_ids"]):
            R_v = np.asarray(y_mv["cam_rot_per_view"][v], dtype=np.float32)
            T_v = np.asarray(y_mv["cam_trans_per_view"][v], dtype=np.float32)
            R_new, T_new = _reparameterize_view(R_model_p3d, t_model_p3d, R_v, T_v)

            x_v = {
                "input_image": x_mv["image_paths"][v],
                "input_image_data": x_mv["image_data"][v],
                "input_image_mask": x_mv["input_image_mask"][v],
            }
            y_v = dict(y_mv)  # shallow copy of shared fields
            y_v["cam_rot"] = torch.tensor(R_new, dtype=torch.float32)
            y_v["cam_trans"] = torch.tensor(T_new, dtype=torch.float32)
            y_v["root_loc"] = np.zeros(3, dtype=np.float32)
            y_v["root_rot"] = np.zeros(3, dtype=np.float32)
            y_v["cam_fov"] = [float(y_mv["fov_per_view"][v])]
            y_v["fx"] = float(y_mv["fx_per_view"][v])
            y_v["fy"] = float(y_mv["fy_per_view"][v])
            y_v["cx"] = float(y_mv["cx_per_view"][v])
            y_v["cy"] = float(y_mv["cy_per_view"][v])
            y_v["keypoints_2d"] = y_mv["keypoints_2d_per_view"][v]
            y_v["keypoint_visibility"] = y_mv["keypoint_visibility_per_view"][v]

            print(f"  CAM{cam_id}: rendering ...", flush=True)
            from smal_fitter.Unreal2Pytorch3D import Render_SMAL_Model_from_Unreal_data
            try:
                Render_SMAL_Model_from_Unreal_data(x_v, y_v, device, verbose=False)
            except Exception as e:
                print(f"  CAM{cam_id}: render FAILED — {e}")
                continue

            # ImageExporter writes LOCAL_TEST/<basename>/st0_ep0.png in current cwd.
            basename = os.path.splitext(os.path.basename(x_mv["image_paths"][v]))[0]
            collage_path = Path("LOCAL_TEST") / basename / "st0_ep0.png"
            if collage_path.exists():
                rendered_paths.append((cam_id, collage_path.resolve()))
            else:
                print(f"  CAM{cam_id}: expected collage at {collage_path} not found")
    finally:
        os.chdir(prev_cwd)

    if not rendered_paths:
        print("  No renders produced.")
        return []

    # Assemble the grid.
    n = len(rendered_paths)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for ax, (cam_id, path) in zip(axes.flatten(), rendered_paths):
        img = imageio.imread(path)
        ax.imshow(img)
        ax.set_title(f"CAM{cam_id}")
        ax.axis("off")
    for j in range(len(rendered_paths), n_rows * n_cols):
        axes.flatten()[j].axis("off")
    fig.suptitle(
        f"Frame {frame_index} — per-view SMAL renders ({label} frame)",
        fontsize=12,
    )
    fig.tight_layout()
    grid_path = Path(output_dir) / f"frame_{frame_index:05d}_render_grid_{label}.png"
    fig.savefig(grid_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Grid saved → {grid_path}")
    print(f"  Individual collages under: {render_root}/LOCAL_TEST/")
    return rendered_paths


def compare_frames_test(dataset_path, frame_index, output_dir, per_camera, cam_ids):
    """Run raw + canonical loader paths side-by-side and validate equivalence.

    Three checks:
      1. Numeric round-trip: |kp3d_world - (kp3d_can - t_0) @ R_0.T| ~ 0.
      2. Reprojection per camera in both frames; both should hit ~0 px GT error.
      3. Visual mesh render in both frames; pixel-identical implies the
         camera/extrinsics/model-rotation composition is correct.
    """
    # 1. Numeric round-trip on the canonical-frame loader output.
    _, y_can = load_SMIL_Unreal_multiview_sample(
        data_path=str(dataset_path),
        frame_index=frame_index,
        camera_indices=None,
        load_images=False,
        canonical_frame=True,
        verbose=False,
    )
    _, y_raw = load_SMIL_Unreal_multiview_sample(
        data_path=str(dataset_path),
        frame_index=frame_index,
        camera_indices=None,
        load_images=False,
        canonical_frame=False,
        verbose=False,
    )
    R_0, t_0 = y_can["canonical_to_world"]
    kp_world_from_can = (y_can["keypoints_3d"] - t_0) @ R_0.T
    rt_diff = np.max(np.abs(kp_world_from_can - y_can["keypoints_3d_world"]))
    print("\n--- Canonical frame: round-trip check ---")
    print(f"  canonical_cam_id: {y_can['canonical_cam_id']}")
    print(f"  max |kp3d_world - (kp3d_can - t_0) @ R_0.T| = {rt_diff:.3e}")
    if rt_diff > 1e-3:
        print("  WARN: round-trip residual above 1e-3 — math suspect.")
    # Also check raw == canonical-world-recovery on the camera extrinsics
    R_v0_raw = y_raw["cam_rot_per_view"][0].numpy()
    R_v0_can = y_can["cam_rot_per_view"][0].numpy()
    print(f"  cam[0] R should be identity in canonical frame: "
          f"max |R_v0_can - I| = {np.max(np.abs(R_v0_can - np.eye(3))):.3e}")
    t_v0_can = y_can["cam_trans_per_view"][0].numpy()
    print(f"  cam[0] t should be zero in canonical frame:     "
          f"max |t_v0_can| = {np.max(np.abs(t_v0_can)):.3e}")

    # 2. Reprojection check in both frames, using the loader's stored
    #    cam_rot_per_view / cam_trans_per_view (row-vector,
    #    PyTorch3D-mirrored) and the loader's keypoints_3d / _world.
    def _project_via_loader(kp3d, R_v, t_v, fx, fy, cx, cy):
        # All inputs are in PyTorch3D-mirrored frame (x-axis negated vs raw
        # Unreal). The raw-frame projection convention u = fx*x/z + cx then
        # flips to u = -fx*x/z + cx after the mirror; v keeps its sign.
        x_cam = kp3d @ R_v + t_v
        depth = x_cam[:, 2]
        safe = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
        u = -fx * x_cam[:, 0] / safe + cx
        v = -fy * x_cam[:, 1] / safe + cy
        return u, v, depth

    print("\n--- Loader-frame reprojection: raw vs canonical ---")
    print(f"{'CAM':>5}  {'raw_mean':>10}  {'raw_max':>10}  "
          f"{'can_mean':>10}  {'can_max':>10}")
    for v, cid in enumerate(cam_ids):
        gt = per_camera[cid]["kp2d_pixels"]
        # Build a 2D-GT array indexed the same way as keypoints_3d (model J_names order).
        gt_mapped = np.zeros((len(config.dd["J_names"]), 2))
        for o, jn in enumerate(config.dd["J_names"]):
            if jn in per_camera[cid]["names"]:
                src_idx = per_camera[cid]["names"].index(jn)
                gt_mapped[o] = gt[src_idx]
        valid = ~np.all(gt_mapped == 0, axis=1)
        fx, fy = y_raw["fx_per_view"][v], y_raw["fy_per_view"][v]
        cx, cy = y_raw["cx_per_view"][v], y_raw["cy_per_view"][v]

        u_r, v_r, d_r = _project_via_loader(
            y_raw["keypoints_3d_world"],
            y_raw["cam_rot_per_view"][v].numpy(),
            y_raw["cam_trans_per_view"][v].numpy(),
            fx, fy, cx, cy,
        )
        u_c, v_c, d_c = _project_via_loader(
            y_can["keypoints_3d"],
            y_can["cam_rot_per_view"][v].numpy(),
            y_can["cam_trans_per_view"][v].numpy(),
            fx, fy, cx, cy,
        )

        def _err(u, v, d):
            mask = valid & (d > 0)
            if not mask.any():
                return float("nan"), float("nan")
            e = np.sqrt((u[mask] - gt_mapped[mask, 0]) ** 2 +
                        (v[mask] - gt_mapped[mask, 1]) ** 2)
            return float(e.mean()), float(e.max())

        raw_mean, raw_max = _err(u_r, v_r, d_r)
        can_mean, can_max = _err(u_c, v_c, d_c)
        print(f"{cid:>5}  {raw_mean:>10.3f}  {raw_max:>10.3f}  "
              f"{can_mean:>10.3f}  {can_max:>10.3f}")

    # 3. Visual mesh renders, raw and canonical, side by side.
    raw_paths = render_per_view(
        dataset_path, frame_index, output_dir,
        canonical_frame=False, label="raw",
    )
    can_paths = render_per_view(
        dataset_path, frame_index, output_dir,
        canonical_frame=True, label="canonical",
    )

    if not raw_paths or not can_paths:
        print("  Skipping side-by-side stitch — one render set empty.")
        return

    raw_by_cam = dict(raw_paths)
    can_by_cam = dict(can_paths)
    common = [c for c, _ in raw_paths if c in can_by_cam]
    n_rows = len(common)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), squeeze=False)
    for r, cid in enumerate(common):
        axes[r, 0].imshow(imageio.imread(raw_by_cam[cid]))
        axes[r, 0].set_title(f"CAM{cid}  RAW (world frame)")
        axes[r, 0].axis("off")
        axes[r, 1].imshow(imageio.imread(can_by_cam[cid]))
        axes[r, 1].set_title(f"CAM{cid}  CANONICAL frame")
        axes[r, 1].axis("off")
    fig.suptitle(
        f"Frame {frame_index} — raw vs canonical SMAL render "
        "(should be pixel-identical)",
        fontsize=12,
    )
    fig.tight_layout()
    side_by_side = Path(output_dir) / f"frame_{frame_index:05d}_compare_frames.png"
    fig.savefig(side_by_side, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Side-by-side grid saved → {side_by_side}")


def _compute_posed_smal_joints(x_mv, y_mv, device):
    """Run SMAL forward with the loader's pose params and return (N_joints, 3)
    in canonical world frame.

    NOTE: this consumes the multi-view loader's scale-unified output
    (translation_factor=0.1 applied at load), so the model placement is the
    plain `(joints - root) + trans` branch — NO `*10` mesh expansion. That
    differs from Render_SMAL_Model_from_Unreal_data with apply_UE_transform=True,
    which is still used for the legacy single-view path.
    """
    torch = _TORCH
    SMALFitter = _SMALFitter

    data_json, _ = return_placeholder_data(
        input_image=x_mv["image_paths"][0],
        num_joints=len(y_mv["joint_angles"]),
        keypoints_2d=y_mv["keypoints_2d_per_view"][0],
        keypoint_visibility=y_mv["keypoint_visibility_per_view"][0],
        silhouette=x_mv["input_image_mask"][0],
    )

    model = SMALFitter(device, data_json, config.WINDOW_SIZE, config.SHAPE_FAMILY, False)
    model.betas = torch.nn.Parameter(torch.Tensor(y_mv["shape_betas"]).to(device))

    if ("scaledirs" in config.dd and "transdirs" in config.dd
            and y_mv.get("scale_weights") is not None
            and y_mv.get("trans_weights") is not None):
        trans_out, scale_out = sample_pca_transforms_from_dirs(
            config.dd, y_mv["scale_weights"], y_mv["trans_weights"]
        )
        model.log_beta_scales = torch.nn.Parameter(
            torch.from_numpy(np.log(scale_out))[None, ...].float().to(device)
        )
        model.betas_trans = torch.nn.Parameter(
            torch.from_numpy(trans_out * y_mv["translation_factor"])[None, ...].float().to(device)
        )
        model.propagate_scaling = y_mv["propagate_scaling"]

    model.joint_rotations = torch.nn.Parameter(
        torch.Tensor(y_mv["joint_angles"][1:])
        .reshape((1, y_mv["joint_angles"][1:].shape[0], 3))
        .to(device)
    )
    model.global_rotation = torch.nn.Parameter(
        torch.from_numpy(y_mv["root_rot"]).float().to(device).unsqueeze(0)
    )
    model.trans = torch.nn.Parameter(
        torch.Tensor(np.array([y_mv["root_loc"]])).to(device)
    )

    batch_params = {
        "global_rotation": model.global_rotation * model.global_mask,
        "joint_rotations": model.joint_rotations * model.rotation_mask,
        "betas": model.betas.expand(1, model.n_betas),
        "trans": model.trans,
    }
    if config.ignore_hardcoded_body:
        batch_params["log_betascale"] = model.log_beta_scales.expand(
            1, model.joint_rotations.shape[1] + 1, 3
        ).to(device)
        if hasattr(model, "betas_trans"):
            batch_params["betas_trans"] = model.betas_trans.expand(
                1, model.joint_rotations.shape[1] + 1, 3
            ).to(device)
    else:
        batch_params["log_betascale"] = model.log_beta_scales.expand(1, 6)

    with torch.no_grad():
        _, joints, _, _ = model.smal_model(
            batch_params["betas"],
            torch.cat([
                batch_params["global_rotation"].unsqueeze(1),
                batch_params["joint_rotations"]], dim=1),
            betas_logscale=batch_params.get("log_betascale", None),
            betas_trans=batch_params.get("betas_trans", None),
            propagate_scaling=getattr(model, "propagate_scaling", None),
        )
        # Scale-unified multi-view convention: data was already scaled by
        # translation_factor=0.1 at load time so mesh-native and trans-frame
        # units match. Recenter on the root and add trans directly — no *10.
        root_joint = joints[:, 0:1, :]
        joints = (joints - root_joint) + batch_params["trans"].unsqueeze(1)

    return joints.squeeze(0).cpu().numpy().astype(np.float32)


def _project_canonical(kp3d, R_v, t_v, fx, fy, cx, cy):
    """Project canonical-frame 3D points through a canonical-frame camera
    in the PyTorch3D-mirrored convention (see _project_via_loader)."""
    x_cam = kp3d @ R_v + t_v
    depth = x_cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
    u = -fx * x_cam[:, 0] / safe + cx
    v = -fy * x_cam[:, 1] / safe + cy
    return u, v, depth


def visualize_named_keypoints(dataset_path, frame_index, output_dir, cam_ids):
    """High-res per-camera 2D overlay + 3D side-by-side plot with joint
    names labelled. Compares loader GT (green) against SMAL forward
    posed joints (red) so we can spot per-joint indexing issues.
    """
    _lazy_render_imports()
    torch = _TORCH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_mv, y_mv = load_SMIL_Unreal_multiview_sample(
        data_path=str(dataset_path),
        frame_index=frame_index,
        camera_indices=cam_ids,
        load_images=True,
        canonical_frame=True,
        verbose=False,
    )

    j_names = list(y_mv["joint_names"])
    n_joints = len(j_names)
    gt_3d_can = np.asarray(y_mv["keypoints_3d"], dtype=np.float32)  # (N, 3)
    posed_3d_can = _compute_posed_smal_joints(x_mv, y_mv, device)  # (N, 3)
    if posed_3d_can.shape[0] != n_joints:
        print(f"  WARN: SMAL produced {posed_3d_can.shape[0]} joints, expected {n_joints}")
        n_joints = min(posed_3d_can.shape[0], n_joints)

    # --- Per-camera 2D overlay grid (high-res, joint names labelled) ---
    print(f"\n--- Named 2D overlay (frame {frame_index}) ---")
    n_views = len(cam_ids)
    n_cols = min(4, n_views)
    n_rows = (n_views + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()
    gt_present = ~np.all(gt_3d_can == 0, axis=1)  # joints the dataset actually has

    for v, cid in enumerate(cam_ids):
        ax = axes_flat[v]
        img = x_mv["image_data"][v]
        if img is None:
            ax.set_title(f"CAM{cid} - no image")
            ax.axis("off")
            continue
        H, W = img.shape[:2]
        ax.imshow(img)

        R_v = y_mv["cam_rot_per_view"][v].numpy()
        t_v = y_mv["cam_trans_per_view"][v].numpy()
        fx, fy = y_mv["fx_per_view"][v], y_mv["fy_per_view"][v]
        cx, cy = y_mv["cx_per_view"][v], y_mv["cy_per_view"][v]

        gt_u, gt_v_, gt_d = _project_canonical(gt_3d_can, R_v, t_v, fx, fy, cx, cy)
        ps_u, ps_v_, ps_d = _project_canonical(posed_3d_can[:n_joints],
                                               R_v, t_v, fx, fy, cx, cy)

        for i in range(n_joints):
            name = j_names[i]
            if gt_present[i] and gt_d[i] > 0:
                ax.plot(gt_u[i], gt_v_[i], marker="o", color="lime",
                        markersize=6, markerfacecolor="none", markeredgewidth=1.4)
                ax.annotate(name, (gt_u[i], gt_v_[i]), color="lime",
                            xytext=(4, -4), textcoords="offset points",
                            fontsize=5, alpha=0.9)
            if ps_d[i] > 0:
                ax.plot(ps_u[i], ps_v_[i], marker="+", color="red",
                        markersize=8, markeredgewidth=1.4)
                ax.annotate(name, (ps_u[i], ps_v_[i]), color="red",
                            xytext=(4, 8), textcoords="offset points",
                            fontsize=5, alpha=0.9)

        ax.set_title(f"CAM{cid}  GT (lime o) vs posed model (red +)", fontsize=10)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.axis("off")

    for j in range(n_views, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"Frame {frame_index} - per-view named GT vs posed-model joints "
        "(canonical frame). Same joint name on both colours should land on "
        "the same anatomy.",
        fontsize=12,
    )
    fig.tight_layout()
    overlay_path = Path(output_dir) / f"frame_{frame_index:05d}_named_overlay.png"
    fig.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  2D overlay grid (high-res) -> {overlay_path}")

    # --- 3D side-by-side ---
    print(f"\n--- Named 3D plot (canonical frame, frame {frame_index}) ---")
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    gt_pts = gt_3d_can[gt_present]
    ax.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2],
               c="green", s=50, label="GT (loader)", depthshade=False)
    ax.scatter(posed_3d_can[:n_joints, 0], posed_3d_can[:n_joints, 1],
               posed_3d_can[:n_joints, 2],
               c="red", s=50, marker="+", label="Posed model (SMAL fwd)",
               depthshade=False)

    for i in range(n_joints):
        name = j_names[i]
        if gt_present[i]:
            ax.text(gt_3d_can[i, 0], gt_3d_can[i, 1], gt_3d_can[i, 2],
                    f"  {name}", color="green", fontsize=6, alpha=0.85)
        ax.text(posed_3d_can[i, 0], posed_3d_can[i, 1], posed_3d_can[i, 2],
                f"  {name}", color="red", fontsize=6, alpha=0.85)

    # Equal axes for honest visual distance.
    all_pts = np.concatenate([gt_pts, posed_3d_can[:n_joints]], axis=0)
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    set_axes_equal(ax)
    ax.set_xlabel("X (canonical)")
    ax.set_ylabel("Y (canonical)")
    ax.set_zlabel("Z (canonical)")
    ax.set_title(f"Frame {frame_index} - 3D GT (green) vs posed model (red), "
                 "joint names labelled.")
    ax.legend()
    out_3d = Path(output_dir) / f"frame_{frame_index:05d}_named_3d.png"
    fig.savefig(out_3d, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  3D named plot -> {out_3d}")


def report_loader_smoke(dataset_path, frame_index):
    print("\n--- Loader smoke test: load_SMIL_Unreal_multiview_sample ---")
    x, y = load_SMIL_Unreal_multiview_sample(
        data_path=str(dataset_path),
        frame_index=frame_index,
        camera_indices=None,
        load_images=False,
        verbose=False,
    )
    print(f"  num_views:        {x['num_views']}")
    print(f"  camera_ids:       {x['camera_ids']}")
    print(f"  shape_betas[:5]:  {y['shape_betas'][:5]}")
    print(f"  joint_angles:     shape={y['joint_angles'].shape}")
    print(f"  root_loc:         {y['root_loc']}")
    print(f"  root_rot:         {y['root_rot']}")
    if y.get("scale_weights") is not None:
        print(f"  scale_weights:    present, shape={np.asarray(y['scale_weights']).shape}")
    else:
        print("  scale_weights:    None")
    # Verify joint_angles aren't all zero (would indicate mapping failure).
    nonzero = int(np.any(np.abs(y["joint_angles"]) > 1e-6, axis=1).sum())
    print(f"  joint_angles non-zero rows: {nonzero}/{y['joint_angles'].shape[0]}")
    if nonzero == 0:
        print("  ⚠ All joint_angles are zero — likely a model/dataset joint-name mismatch.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--output_dir", default="TEST_plots/multiview_loader")
    parser.add_argument("--camera_indices", type=int, nargs="*", default=None)
    parser.add_argument("--smal_file", default=None,
                        help="If given, run model-fit compatibility checks against this SMAL pickle.")
    parser.add_argument("--shape_family", type=int, default=None)
    parser.add_argument("--render", action="store_true",
                        help="Also render the posed SMAL mesh through each view's camera. "
                             "Requires --smal_file and GPU; mirrors the single-view render path.")
    parser.add_argument("--compare_frames", action="store_true",
                        help="Run the canonical-camera-frame validation: numeric "
                             "round-trip, side-by-side reprojection check, and raw-vs-"
                             "canonical mesh renders. Requires --smal_file and GPU.")
    parser.add_argument("--named_overlay", action="store_true",
                        help="High-res per-camera overlay + 3D side-by-side plot "
                             "with joint name labels. Compares loader GT against "
                             "SMAL forward posed joints in canonical frame. "
                             "Requires --smal_file and GPU.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_files = list(dataset_path.glob("_BatchData_*.json"))
    if not batch_files:
        sys.exit(f"No _BatchData_*.json in {dataset_path}")
    with open(batch_files[0]) as f:
        batch_data_file = json.load(f)
    dataset_name = batch_files[0].stem.replace("_BatchData_", "")

    cam_ids = args.camera_indices or list_camera_ids(dataset_path, dataset_name, args.frame_index)
    if not cam_ids:
        sys.exit(f"No camera JSONs found for frame {args.frame_index}")
    print(f"Frame {args.frame_index}: {len(cam_ids)} cameras: {cam_ids}")

    per_camera = {cid: load_per_camera(dataset_path, dataset_name, args.frame_index, cid, batch_data_file) for cid in cam_ids}

    n_views = len(cam_ids)
    n_cols = min(4, n_views)
    n_rows = (n_views + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    summary = []
    for view_idx, cid in enumerate(cam_ids):
        ax = axes_flat[view_idx]
        pc = per_camera[cid]
        img = pc["image"]
        if img is None:
            ax.set_title(f"CAM{cid} — no image")
            ax.axis("off")
            continue
        H, W = img.shape[:2]
        ax.imshow(img)

        # GT 2D from this camera's JSON, plotted as raw pixels (2DPos.x, 2DPos.y).
        gt = pc["kp2d_pixels"]
        valid_gt = ~np.all(gt == 0, axis=1)
        ax.scatter(gt[valid_gt, 0], gt[valid_gt, 1],
                   facecolors="none", edgecolors="lime", s=80, linewidths=1.5,
                   label="GT 2D (JSON 2DPos)")

        # Try every (R | R.T) × signed-depth-axis convention; pick the lowest-error one.
        variants = project_row_vector(
            pc["kp3d_world"], pc["R"], pc["t"], pc["fx"], pc["fy"], pc["cx"], pc["cy"]
        )
        best_conv, best_err, best_uv = None, float("inf"), None
        for name, (u, v, depth) in variants.items():
            in_front = depth > 0
            common = valid_gt & in_front
            if common.sum() < 4:
                continue
            err = np.sqrt((u[common] - gt[common, 0])**2 + (v[common] - gt[common, 1])**2).mean()
            if err < best_err:
                best_err = err
                best_conv = name
                best_uv = (u, v, depth)
        if best_uv is None:
            best_conv = "no-valid-conv"
            best_uv = (np.zeros(len(gt)), np.zeros(len(gt)), np.ones(len(gt)))
        u, v, depth = best_uv
        in_front = depth > 0
        ax.scatter(u[in_front], v[in_front],
                   marker="+", color="red", s=80, linewidths=1.5,
                   label=f"Projected 3D ({best_conv})")

        common = valid_gt & in_front
        if common.any():
            err = np.sqrt((u[common] - gt[common, 0])**2 + (v[common] - gt[common, 1])**2)
            mean_err = float(err.mean())
            max_err = float(err.max())
        else:
            mean_err = max_err = float("nan")
        summary.append((cid, best_conv, mean_err, max_err, int(in_front.sum()), int(valid_gt.sum())))

        ax.set_title(f"CAM{cid}  {best_conv}  mean={mean_err:.1f}px  max={max_err:.1f}px")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")

    for j in range(n_views, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"Frame {args.frame_index} — projection (red +) vs raw GT 2D (green circle)", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"frame_{args.frame_index:05d}_reprojection_check.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")

    print(f"\n{'CAM':>5}  {'conv':>30}  {'mean_px':>8}  {'max_px':>8}  {'in_front':>8}  {'gt_pts':>6}")
    for cid, conv, mean_err, max_err, in_front, gt_pts in summary:
        print(f"{cid:>5}  {conv:>30}  {mean_err:>8.2f}  {max_err:>8.2f}  {in_front:>8}  {gt_pts:>6}")

    # Optional model-fit compatibility report (requires --smal_file).
    if args.smal_file:
        first_cam = cam_ids[0]
        dataset_joint_names = per_camera[first_cam]["names"]
        model_joint_names = list(config.dd["J_names"])
        print(f"\nActive SMAL model: {config.SMAL_FILE}")
        print(f"Active ROOT_JOINT: {config.ROOT_JOINT!r}")
        report_joint_name_overlap(dataset_joint_names, model_joint_names)
        report_loader_smoke(dataset_path, args.frame_index)

        if args.render:
            render_per_view(dataset_path, args.frame_index, output_dir)
        if args.compare_frames:
            compare_frames_test(
                dataset_path, args.frame_index, output_dir, per_camera, cam_ids,
            )
        if args.named_overlay:
            visualize_named_keypoints(
                dataset_path, args.frame_index, output_dir, cam_ids,
            )
    elif args.render or args.compare_frames or args.named_overlay:
        print("\n⚠ --render / --compare_frames / --named_overlay requires --smal_file; skipping.")


if __name__ == "__main__":
    main()
