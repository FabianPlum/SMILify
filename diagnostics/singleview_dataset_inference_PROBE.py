#!/usr/bin/env python3
"""Static + behavioural probe for the unified single-view dataset inference path.

Verifies, WITHOUT importing torch/pytorch3d (so it runs anywhere):

  1. Structure — the shared convention helpers and the dataset pipeline exist in
     run_singleview_inference.py, and every render path routes joint scale /
     translation, mesh scale and camera through those helpers instead of
     writing raw predicted tensors onto a SMALFitter.

     This is the regression guard for the actual bug: in 'separate' +
     use_pca_transformation mode the network emits log_beta_scales as PCA
     weights of shape (B, N_BETAS), and the old code assigned them straight to
     temp_fitter.log_beta_scales, which is (B, n_joints, 3). They must go
     through model._transform_separate_pca_weights_to_joint_values() first,
     exactly as train_smil_regressor.py and run_multiview_inference.py do.

  2. CLI — --dataset is mutually exclusive with --input_folder / --input_video,
     and the multi-view-parity flags are present.

  3. Behaviour — compute_view_item_indices() and compute_subclip_ranges() are
     executed on synthetic inputs, including the case that matters most:
     in expand_all_views mode items are (sample, view) pairs enumerated
     sample-major, so filtering to one camera slot must recover a
     temporally-ordered clip and must skip frames where that camera is missing
     rather than padding them.

Usage:  python diagnostics/singleview_dataset_inference_PROBE.py
        (run from the repo root, or with the script's directory adjusted)
"""

import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smal_fitter", "neuralSMIL"))

import ast
import io
import sys

src = io.open("run_singleview_inference.py", encoding="utf-8").read()
tree = ast.parse(src)
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        ok = False


top = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
for name in [
    "apply_scale_trans_to_fitter",
    "resolve_mesh_scale",
    "apply_camera_to_fitter",
    "build_singleview_dataset",
    "compute_view_item_indices",
    "compute_subclip_ranges",
    "render_dataset_sample_collage",
    "run_dataset_inference_phase",
    "process_dataset",
    "PredictionSmoother",
    "_InMemoryImageExporter",
    "_pad_or_resize",
]:
    check(name in top, f"defined: {name}")

# No render path writes raw log_beta_scales onto a fitter any more
check(
    "temp_fitter.log_beta_scales.data = predicted_params[" not in src,
    "no render path writes raw predicted log_beta_scales to a fitter",
)
check(
    "temp_fitter.betas_trans.data = predicted_params[" not in src,
    "no render path writes raw predicted betas_trans to a fitter",
)

# Scales and translations are applied independently: a checkpoint with only one
# of the two heads must not lose the other (PR #101 review finding).
check(
    '"log_beta_scales" not in predicted_params or "betas_trans" not in predicted_params' not in src,
    "apply_scale_trans_to_fitter does not require both heads to be present",
)

# every render path routes scales through the helper
# 4 render paths (model_only, on_frame, generate_visualization, dataset collage) + 1 def
check(
    src.count("apply_scale_trans_to_fitter(") == 5 + 1,  # +1: extra call kwarg line in dataset path
    f"apply_scale_trans_to_fitter wired everywhere (got {src.count('apply_scale_trans_to_fitter(')})",
)
check(
    src.count("resolve_mesh_scale(") == 5,
    f"mesh_scale resolved in all 4 render paths + def (got {src.count('resolve_mesh_scale(')})",
)
check(
    src.count("apply_camera_to_fitter(") == 5,
    f"camera routed through helper in all 4 render paths + def (got {src.count('apply_camera_to_fitter(')})",
)

# argparse surface
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
opts = set()
for node in ast.walk(main):
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                opts.add(a.value)
for flag in [
    "-d",
    "--dataset",
    "--view_indices",
    "--smoothing_window",
    "--generate_num_subclips",
    "--disable_scaling",
    "--disable_translation",
    "--render_resolution",
    "--smal_file",
    "--shape_family",
    "--fov",
    "--export_animation",
]:
    check(flag in opts, f"CLI flag present: {flag}")

# mutually exclusive input group has 3 members
grp_adds = 0
for node in ast.walk(main):
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
        v = getattr(node.func, "value", None)
        if isinstance(v, ast.Name) and v.id == "input_group":
            grp_adds += 1
check(grp_adds == 3, f"input_folder/input_video/dataset are mutually exclusive ({grp_adds} members)")

# gt_fov threaded into run_inference_on_image
rioi = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_inference_on_image")
check("gt_fov" in [a.arg for a in rioi.args.args], "run_inference_on_image accepts gt_fov")
check("gt_fov=gt_fov" in src, "dataset inference passes calibrated gt_fov")

# camera-centric guard
check(
    "multi-view HDF5 (the camera is re-anchored per view)" in src, "camera_centric + non-multiview dataset is rejected"
)

# --- behavioural test of the pure helpers ---
import typing

ns = {"Dict": typing.Dict, "List": typing.List, "Tuple": typing.Tuple, "Optional": typing.Optional}
for fn in ("compute_view_item_indices", "compute_subclip_ranges"):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)


class FakeMV:
    # 3 frames x views [0,1,2], frame 1 missing view 1
    _sv_items = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    max_views = 3

    def __len__(self):
        return len(self._sv_items)


got = ns["compute_view_item_indices"](FakeMV(), [0, 2])
check(got == {0: [0, 3, 5], 2: [2, 4, 7]}, f"view filter yields per-camera, frame-ordered items: {got}")
check(all(FakeMV._sv_items[i][1] == 0 for i in got[0]), "view 0 items are all camera slot 0")
check([FakeMV._sv_items[i][0] for i in got[0]] == [0, 1, 2], "view 0 items are in frame order")

got2 = ns["compute_view_item_indices"](FakeMV(), [1])
check(got2 == {1: [1, 6]}, f"missing view slot in a frame is skipped, not padded: {got2}")


class FakeSV:
    def __len__(self):
        return 5


check(
    ns["compute_view_item_indices"](FakeSV(), [0]) == {0: [0, 1, 2, 3, 4]},
    "single-view dataset maps everything to slot 0",
)

csr = ns["compute_subclip_ranges"]
check(csr(100, None, 1) == [(0, 100)], "no max_frames -> full clip")
check(csr(100, 30, 1) == [(0, 30)], "max_frames caps a single clip")
check(csr(100, 20, 3) == [(0, 20), (33, 53), (66, 86)], f"3 subclips spaced evenly: {csr(100, 20, 3)}")
check(csr(50, 40, 3) == [(0, 50)], "subclips that don't fit fall back to one clip")
check(csr(10, 999, 1) == [(0, 10)], "max_frames beyond dataset is clamped")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
