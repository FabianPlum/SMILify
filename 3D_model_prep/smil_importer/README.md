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

Open the 3D viewport sidebar (press **N**) and use the **SMPL** and
**Morphometry** tabs.
