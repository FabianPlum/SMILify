# SMIL Animation Export & Blender Import — Implementation Plan & Status

> ⚠️ **HISTORICAL PLAN** — tied to the `inference_animation_export` feature branch (not necessarily merged to `master`). Describes the planned/as-built animation `.npz`/`.json` export. For current behaviour see the `--export_animation` flag in [run_multiview_inference.py](../../smal_fitter/neuralSMIL/run_multiview_inference.py) / `run_singleview_inference.py`.

**Branch:** `inference_animation_export` (rebased onto `augmentation-robustness`)
**Last updated:** 2026-05-08

## Goal

`run_singleview_inference.py` and `run_multiview_inference.py` predict full SMIL
parameter trajectories (global rotation, per-joint axis-angle pose, translation,
betas, per-joint log-scales, per-joint translations, camera params) but
previously *discarded* them after rendering the MP4 preview. This work makes
those trajectories first-class, shareable outputs.

## Four-phase strategy

0. **Phase 0 — Branch setup.** ✅ Done — `inference_animation_export` created
   off `backbone_factory_overhaul`, then rebased onto `augmentation-robustness`
   (strict superset; 0 divergent commits, 36 commits ahead; no conflicts on
   rebase).
1. **Phase 1 — Lossless Python export.** ✅ Done — AMASS-compatible `.npz`
   + human-readable `.json` sidecar, written from both inference scripts behind
   a `--export_animation PATH` flag.
2. **Phase 2 — Blender addon importer.** ✅ Operator implemented and
   registered; awaiting end-to-end test in Blender 4.2 with a real `.npz`.
3. **Phase 3 — Addon-side export to shareable formats.** ⏳ Not started —
   wrapper around `bpy.ops.export_scene.gltf()` / `.fbx()` with sensible
   defaults for skeletal + morph-target animation + cameras.

## Phase 1 — format choice (AMASS `.npz` + JSON sidecar)

### Formats considered and rejected

- **BVH / FBX / Alembic** — lossy or toolchain-hostile for a parametric animal rig.
- **glTF 2.0** — strong baked-delivery path (morph targets + TRS animation) but
  bakes away the SMIL parameter semantics we want to preserve for research.
- **USD (UsdSkel + BlendShapeAnimation)** — best semantic match of any standard
  format, but Blender's UsdSkel importer still has rough edges as of 2025.
- **FBX first** — semantically capable (Shape Deformers + per-bone TRS +
  cameras), but Autodesk FBX SDK Python bindings are closed-source /
  version-locked; `pyfbx` / `pyfbx_jasper` are incomplete with spotty
  blend-shape support. Routing via `blender --background` requires the Phase 2
  importer to exist first, collapsing "FBX first" back into "addon first."
  FBX delivery therefore deferred to Phase 3.

### Chosen format

AMASS-compatible `.npz` with a documented SMIL extension schema, paired with a
small sidecar `.json` for human-readable metadata. De facto convention in the
parametric-body community (SMPL / SMPL-X / SMAL); lossless; diffable; trivially
loaded with `np.load`. We control both ends.

### On-disk schema

`<clip_name>.npz` (binary, primary):

```
poses            (F, N_JOINTS, 3)   float32  axis-angle per joint (index 0 = root / global_rot)
trans            (F, 3)             float32  world translation
betas            (N_BETAS,)         float32  clip-averaged shape vector (AMASS baseline)
betas_per_frame  (F, N_BETAS)       float32  per-frame shape vector (drives Blender morph animation)
log_beta_scales  (F, N_JOINTS, 3)   float32  per-joint log scale (when model predicts)
betas_trans      (F, N_JOINTS, 3)   float32  per-joint trans (when model predicts)
fps              scalar             float32
```

Why both `betas` and `betas_per_frame`: glTF morph targets, FBX Shape Deformers,
and USD BlendShapeAnimation all animate weights per-frame as a standard feature
(facial animation relies on this). AMASS's static-betas convention is a domain
assumption, not a format limitation — we store both and let the importer
choose.

`<clip_name>.json` (sidecar):

```json
{
  "schema_version": "1.0",
  "model_id": "...",
  "source_checkpoint": "...",
  "source_input": "...",
  "n_frames": ...,
  "n_joints": ...,
  "n_betas": ...,
  "joint_names": [...],
  "parents": [...],
  "rotation_representation": "axis_angle",
  "root_joint_index": 0,
  "static_joint_locs": true | false,
  "ignore_hardcoded_body": true | false,
  "fps": 30,
  "cameras": [{"view_name": "...", "R": 3x3, "t": 3, "fov": ...}, ...]
}
```

### Field sourcing (verified against code)

- `rotation_representation`: always the string `"axis_angle"` on output. The
  regressor may internally use `"6d"` or `"axis_angle"`
  ([smil_image_regressor.py:103,119,143](../../smal_fitter/neuralSMIL/smil_image_regressor.py));
  exporter normalises via the `rotation_6d_to_axis_angle` path
  (`pytorch3d.transforms.rotation_6d_to_matrix` → `matrix_to_axis_angle`)
  inside `animation_export.py` to avoid pulling the full regressor import
  chain.
- `root_joint_index`: fixed at 0; root identified as the entry where
  `dd["kintree_table"][0] == -1`
  ([config.py:96](../../config.py), [smal_torch.py:205](../../smal_model/smal_torch.py)).
- `static_joint_locs`: real flag (not invented). Derived from the pkl key
  `dd["static_joint_locs"]` and exposed as `config.STATIC_JOINT_LOCATIONS`
  ([config.py:82-89](../../config.py)). Controls joint computation inside
  `SMAL.__call__` ([smal_torch.py:254-263, 342-350](../../smal_model/smal_torch.py)):
  - `True`  → use stored `self.J` directly.
  - `False` → always recompute `J = J_regressor @ v_shaped` each forward pass.
- `ignore_hardcoded_body`: recorded for informational parity with
  [config.py:49](../../config.py); controls symmetry / joint-name loading, not
  joint recomputation.

### Correctness implication of `static_joint_locs`

- **`static_joint_locs == true`** — joint positions are fixed. Per-frame
  `betas_per_frame` shape animation is safe; importer keyframes shape keys per
  frame by default.
- **`static_joint_locs == false`** — joint positions depend on current shape
  (`J_regressor @ v_shaped`). Per-frame shape changes would require
  re-regressing joints every frame, which a static Blender armature can't
  represent. In this mode the importer must:
  1. Use the clip-averaged `betas` only (ignore `betas_per_frame`).
  2. Apply those betas to the shape keys once (static for the clip).
  3. Invoke the existing `SMPL_OT_RecomputeJointPositions` operator
     ([SMIL_processing_addon.py:3356](../../3D_model_prep/SMIL_processing_addon.py))
     once to re-regress the armature's rest pose to the averaged shape.
  4. Then keyframe rotation / bone scale / root trans as normal.
  Importer emits an INFO message explaining per-frame shape animation is
  disabled in this mode.

## Phase 1 — implementation (DONE)

### Files

- **NEW** [smal_fitter/neuralSMIL/animation_export.py](../../smal_fitter/neuralSMIL/animation_export.py)
  — `AnimationRecorder` class (per-frame accumulation: `record()` / `write()`
  / `set_cameras()` / `num_frames()`); `build_recorder_from_config` constructor
  that pulls joint metadata from the global `config`;
  `build_multiview_cameras` that averages `cam_{rot,trans,fov}_per_view` across
  frames into a static sidecar block; local `rotation_6d_to_axis_angle` with
  lazy pytorch3d import. Schema version `"1.0"`.

- **MODIFIED** [smal_fitter/neuralSMIL/run_singleview_inference.py](../../smal_fitter/neuralSMIL/run_singleview_inference.py)
  — added `--export_animation PATH` CLI flag; added `animation_recorder`
  optional parameter to `process_video`; calls
  `animation_recorder.record(predicted_params)` immediately after
  `run_inference_on_image` and *before* camera smoothing (captures raw
  pre-smoothing values); writes at end of `main()`.

- **MODIFIED** [smal_fitter/neuralSMIL/run_multiview_inference.py](../../smal_fitter/neuralSMIL/run_multiview_inference.py)
  — added `--export_animation` flag; new `_export_animation` helper that
  reuses existing DDP temp-dir machinery
  (`write_predictions_to_temp` / `load_all_predictions_from_temp`) with a
  dedicated `.animation_export_temp_*` directory so the gather doesn't
  interfere with the separate smoothing gather path; builds averaged
  multi-view cameras from `dataset.get_canonical_camera_order()`; writes on
  rank 0 only. Invoked immediately after `run_inference_phase` returns.

- **NEW** [tests/test_animation_export.py](../../tests/test_animation_export.py)
  — 5 round-trip tests, all passing:
  1. Round-trip axis-angle with full optional fields (poses, trans, betas,
     betas_per_frame, log_beta_scales, betas_trans, fps, sidecar schema).
  2. 6D rotations normalise to axis-angle (identity 6D → zero axis-angle).
  3. Optional fields absent when not recorded.
  4. Empty recorder raises RuntimeError.
  5. Invalid rotation_representation raises ValueError.

### Commits (on `inference_animation_export`, in order)

```
a122ab9  Add AnimationRecorder helper for SMIL inference export
e4f1bc9  Wire --export_animation into singleview inference
d416d5c  Wire --export_animation into multiview inference
93e0155  Add round-trip tests for AnimationRecorder
5c4a5db  Add SMPL_OT_ImportAnimation operator for SMIL animation playback
```

## Phase 2 — Blender addon importer (IMPLEMENTED, UNTESTED)

### Files

- **MODIFIED** [3D_model_prep/SMIL_processing_addon.py](../../3D_model_prep/SMIL_processing_addon.py)
  — new operator `SMPL_OT_ImportAnimation`
  (bl_idname: `smpl.import_animation`), registered in the `classes` tuple and
  surfaced as a panel button `"Import SMIL Animation (.npz)"` in
  `SMPL_PT_Panel.draw()` (below `smpl.apply_pose_correctives`).

### Helper functions (in addon)

- `_resolve_shape_key_for_beta(obj, beta_index)` — prefers `Shape_<i>`, falls
  back to `PC_<i+1>`, then positional (skipping `Basis`).
- `_apply_betas_to_shape_keys(obj, betas, frame=None)` — sets shape-key values
  with optional keyframing.
- `_load_animation_files(npz_path)` — loads `.npz` + sidecar `.json`.
- `_find_mesh_with_armature(context)` — resolves active object → (mesh,
  armature).

### Operator execute flow

1. Load `.npz` + `.json`.
2. Validate `joint_names` against armature bone names.
3. Branch on sidecar `static_joint_locs`:
   - `true` — use `betas_per_frame` (default) or clip-averaged `betas` if
     operator's `static_shape` property is set.
   - `false` — force clip-averaged `betas`, apply once to shape keys, call
     `bpy.ops.smpl.recompute_joint_positions()` to re-regress rest pose;
     emit INFO message explaining per-frame shape animation is disabled.
4. Set `scene.frame_start`, `scene.frame_end`, `scene.render.fps` from sidecar.
5. Set all bones to `rotation_mode = 'AXIS_ANGLE'`.
6. Per frame:
   - Keyframe bone `rotation_axis_angle` (normalise axis).
   - Keyframe bone `scale` from `exp(log_beta_scales)`.
   - Keyframe armature object `location` from `trans[f]`.
   - Keyframe shape keys from `betas_per_frame[f]` (static-skeleton mode only).
7. Create one Blender camera per sidecar view via `_create_cameras`, inverting
   the PyTorch3D world→view `R|t` into Blender's camera→world convention.

### Operator properties

- `static_shape: BoolProperty` — collapses per-frame shape to clip-averaged
  `betas` at frame 0 (for users who want a fixed body).
- `apply_joint_scales: BoolProperty` — toggle `log_beta_scales` keyframing.
- `create_cameras: BoolProperty` — toggle camera creation from sidecar.

### Outstanding items for Phase 2 (TODO — to be implemented)

Items surfaced from end-to-end testing in Blender; deferred for follow-up:

- **Imported model scaling.** Update the scaling applied to the SMIL model when
  it's brought into Blender so the imported armature/mesh lands at the expected
  world-space size (matches the inference rig / per-frame `mesh_scale`).
- **Camera parameters.** A few sidecar-driven camera parameters need
  adjustment on the importer side (specifics TBD during implementation).
- **Default range of shape parameters (blend-shape weights).** Adjust the
  default min/max range of the imported SMIL model's shape-key (blend-shape)
  weights so per-frame `betas_per_frame` values keyframe within the allowed
  range without being clamped.

### Out of scope for Phase 2

- Posedir correctives application (existing logic at
  [SMIL_processing_addon.py:705-760](../../3D_model_prep/SMIL_processing_addon.py)
  can be wired in later).
- Drivers-based live evaluation.
- Animated cameras.

## Phase 3 — shareable export (NOT STARTED)

### Planned

- New operator `SMPL_OT_ExportSharedAnimation`
  (bl_idname: `smpl.export_shared_animation`) with format enum
  (`GLTF` default, `FBX` optional).
- Thin wrapper around Blender's built-in exporters:
  - **glTF**: `bpy.ops.export_scene.gltf(filepath=..., export_format='GLB',
    export_animations=True, export_morph=True,
    export_morph_animation=True, export_cameras=True,
    export_apply=False)`.
  - **FBX**: `bpy.ops.export_scene.fbx(filepath=...,
    use_armature_deform_only=False, add_leaf_bones=False, bake_anim=True,
    bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False, bake_anim_step=1.0,
    mesh_smooth_type='FACE', use_mesh_modifiers=True)`.
- Pre-export sanity: ensure scene frame range matches imported clip; ensure
  shape-key fcurves exist (warn + offer static-shape fallback otherwise).
- No custom FBX/glTF writing — delegate entirely to Blender's exporters. Addon
  only curates the settings matrix.

## Verification

### Phase 1 — Python export ✅

- **Unit:** `tests/test_animation_export.py` — 5 tests, all passing in the
  `pytorch3d` conda env on Windows.
- **Integration:** pending — blocked by dataset/checkpoint mismatch (see
  Runtime notes below).
- **Regression:** without `--export_animation`, both inference scripts must
  produce byte-identical MP4s to prior master (guard rail for the default
  path). Not yet re-verified post-rebase.

### Phase 2 — Blender import ⏳

- **Manual:** launch Blender 4.2, import SMIL model via existing
  `smpl.import_model`, import `.npz` via new `smpl.import_animation`, scrub
  timeline, confirm bones rotate, shape keys drive, per-view cameras appear
  with correct framing.
- **Fidelity check:** in Blender, evaluate rig at frame F, read back bone
  rotations, compare to source `.npz` `poses[F]` — should be within float
  precision.

### Phase 3 — shareable export ⏳

- Open `.glb` in third-party glTF viewer (e.g.
  gltf-viewer.donmccurdy.com or Windows 3D Viewer); confirm skeletal +
  morph-target animation + camera are present.
- Repeat for `.fbx` in Unity or Maya.

## Runtime notes / current blockers

### Inference run needs an HF-weight cache

`timm.create_model(..., pretrained=True)` at
[backbone_factory.py:325](../../smal_fitter/neuralSMIL/backbone_factory.py#L325)
pulls ImageNet weights from HuggingFace. On machines without internet (or
where HF is unreachable — WSL2 networking frequently is), the backbone
constructor fails with `LocalEntryNotFoundError`.

Workarounds tried:

- **Download once, run offline.** Run
  `python hpc_files/download_backbone_weights.py` on a machine with internet
  access (populates `$HF_HOME/hub`), then on the inference host
  `export HF_HUB_OFFLINE=1` before running inference.
- **WSL ↔ Windows cache share.** Downloads on Windows land at
  `C:\Users\Fabian\.cache\huggingface` — WSL has its own cache at
  `/home/<user>/.cache/huggingface`. Point WSL at the Windows cache via
  `export HF_HOME=/mnt/c/Users/Fabian/.cache/huggingface` (plus
  `export HF_HUB_OFFLINE=1`).

Potential future fix: pass `pretrained=False` from
`load_multiview_model_from_checkpoint` since the checkpoint load supersedes
the ImageNet init. Not yet implemented.

### Checkpoint max_views vs dataset max_views mismatch

During the test run, the `multiview_checkpoints_MOUSE_Unet_512_EffB3_v3`
checkpoint has `view_embeddings.weight` of shape `(12, D)` (trained with
`max_views=12`), but `SMILymice_3D_only.h5` exposes 18 canonical cameras. When
a sample comes with view indices ≥ 12, the embedding lookup crashes with
`indexSelectLargeIndex` CUDA asserts (0/200 predictions completed).

User resolution: will provide a dataset matching the checkpoint's 12-view
range — no code change needed.

## Session continuation checklist

When resuming elsewhere:

1. `git fetch origin && git checkout inference_animation_export` (rebased onto
   `augmentation-robustness`; 5 commits ahead of that base — see list above).
2. Ensure conda env `pytorch3d` is active (Python 3.10, PyTorch 2.3.1,
   CUDA 11.8).
3. Populate HF cache (`python hpc_files/download_backbone_weights.py`) and
   `export HF_HUB_OFFLINE=1` if the inference host lacks reliable HF access.
4. Use a dataset whose view count ≤ checkpoint's `max_views`.
5. Integration test:
   ```bash
   python smal_fitter/neuralSMIL/run_multiview_inference.py \
       --dataset <h5> --smal_file <pkl> \
       --checkpoint <ckpt.pth> \
       --max_frames 20 \
       --export_animation True
   ```
   Expect `.npz` + `.json` next to the usual MP4 outputs.
6. Phase 2 test: open Blender 4.2, load SMIL model via existing addon
   operators, then the new **Import SMIL Animation (.npz)** button in the SMPL
   panel. Load the `.npz` from step 5.
7. Critical assessment of code so far (user's explicit next step).
8. Then Phase 3 (glTF/FBX export wrapper).
