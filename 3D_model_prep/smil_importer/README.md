# SMIL Model Importer — Blender add-on

Import, configure and export SMPL / SMIL parametric models from Blender.

This is the installable, modular successor to the old single-file
`SMIL_processing_addon.py` (kept under `3D_model_prep/legacy/` for reference).
The logic is split into focused modules:

| module | responsibility |
| --- | --- |
| `dependencies.py` | Detect `scipy` / `scikit-learn`; one-click installer in preferences |
| `state.py` | Shared Transformation-PCA components |
| `unpickler.py` | Legacy chumpy-aware `.pkl` loader |
| `core_mesh.py` | Leaf geometry / export utilities (uses `scipy`, lazily) |
| `pca.py` | PCA shape-space construction (uses `scikit-learn`, lazily) |
| `model_build.py` | Build Blender mesh / armature / shape-keys, export model |
| `measurements.py` | Joint distances, mesh measurements, SMPL data store |
| `properties.py` | Add-on `PropertyGroup` |
| `operators.py` | Import / export / generate / animation operators |
| `ui.py` | N-panel UI |

## Installing

1. Build the installable zip (from `3D_model_prep/`):

   ```bash
   python build_addon.py
   ```

   This produces `smil_importer.zip`.

2. In Blender: **Edit > Preferences > Add-ons > Install from Disk…**, pick
   `smil_importer.zip`, and enable **SMIL Model Importer**.

## Dependencies (scipy, scikit-learn)

These are **not** bundled with Blender. The add-on still registers without
them — the panels load, but shape-space (PCA) and joint-regressor features are
disabled until the packages are present.

To install them, open **Edit > Preferences > Add-ons > SMIL Model Importer**,
expand the add-on, and click **Install dependencies (scipy, scikit-learn)**.
This pip-installs them into Blender's bundled Python. **Restart Blender** once
it finishes. The panel shows a green check once both packages are importable.

## Usage

Open the 3D viewport sidebar (press **N**) and use the **SMIL** and
**Morphometry** tabs.

## Import options

- **Import Joint Limits** (default: on) — recreates a `.pkl`'s `joint_limits`
  as local-space **Limit Rotation** constraints (the exact inverse of the
  export remap), so the constraint panel, the overlay and a re-export match the
  file. Without it, an unchanged import → export opens every joint to ±180°
  (issue #116). Axes at the wide-open **Default Joint Limit Range** stay
  unticked. **Import Inference Animation** mutes these constraints on the rig
  so playback shows the predicted rotations unclamped; unmute them before
  exporting the model from that scene.

## Export options

The export section of the **SMIL** panel exposes, among others:

- **Export Joint Limits** (default: on) — reads per-bone rotation limits and
  stores them in the exported `.pkl` under the `joint_limits` key
  (shape `(J, 3, 2)`, radians, model frame). For each bone the limits come
  from an enabled (non-muted), local-space **Limit Rotation** bone constraint
  if present, otherwise from the bone's IK rotation limits/locks. See the
  [Joint Limits User Guide](../../docs/joint_limits_user_guide.md).
- **Default Joint Limit Range** — half-range used for axes with no explicit
  limit; such axes export as `[-value, +value]`. Displayed in degrees (like
  the Limit Rotation constraint fields), stored/exported in radians. The
  default (180°) is effectively unconstrained, so the fitter's limit prior
  stays inactive until real limits are authored.

## Visualization

The **Visualization** box of the SMIL panel has a **Show Joint Limit
Overlay** toggle that draws a translucent wedge per constrained axis of each
bone's Limit Rotation constraint — a faithful preview of what will be
exported. Also available as a standalone script:
`3D_model_prep/joint_rot_limit_vis.py`.

## Recommended Blender version

Export `.pkl` files with **Blender 4.2 LTS**. Blender versions that bundle
numpy >= 2 (4.5+/5.x) write pickles that environments running numpy < 2 —
including the SMILify `pytorch3d` environment — cannot load
(`ModuleNotFoundError: No module named 'numpy._core...'`). Importing and
editing models works on any Blender >= 4.2.
