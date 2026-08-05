"""Headless SMIL import->export round-trip driver (runs inside Blender).

DIAGNOSTIC HARNESS for reviewing PR #88 — not part of the addon, never committed.

Usage:
  blender --background --factory-startup --python run_roundtrip.py -- \
      <addon: new|legacy> <input_pkl> <run_dir> <output_filename>

- "new"    registers 3D_model_prep/smil_importer (from the working tree, not the zip)
- "legacy" registers 3D_model_prep/legacy/SMIL_processing_addon.py

Both are driven with the identical, mutation-free option set:
  npz_filepath="", shapekeys_from_PCA=False, regress_joints=False,
  clean_mesh=False, symmetrise=False, force_static_joint_locs=False.

The input pkl is copied into run_dir first, so the exported pkl and all
test_*.npy debug files the exporter drops land in run_dir.
"""

import hashlib
import os
import shutil
import sys
import traceback

import bpy

REPO_PREP = "/home/fabi/dev/SMILify/3D_model_prep"

argv = sys.argv[sys.argv.index("--") + 1 :]
mode, input_pkl, run_dir, out_name = argv[0], argv[1], argv[2], argv[3]

os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)  # unsaved .blend => "//" paths resolve here

# Fresh empty scene BEFORE registering (read_factory_settings resets a lot).
bpy.ops.wm.read_factory_settings(use_empty=True)

if mode == "new":
    sys.path.insert(0, REPO_PREP)
    import smil_importer

    smil_importer.register()
elif mode == "legacy":
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "SMIL_processing_addon", os.path.join(REPO_PREP, "legacy", "SMIL_processing_addon.py")
    )
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy.register()
else:
    raise SystemExit(f"unknown mode {mode!r}")

local_input = os.path.join(run_dir, os.path.basename(input_pkl))
if os.path.abspath(local_input) != os.path.abspath(input_pkl):
    shutil.copy2(input_pkl, local_input)

tool = bpy.context.scene.smpl_tool
tool.pkl_filepath = local_input
tool.npz_filepath = ""
tool.shapekeys_from_PCA = False
tool.regress_joints = False
tool.clean_mesh = False
tool.symmetrise = False
tool.force_static_joint_locs = False
tool.output_filename = out_name

try:
    result = bpy.ops.smpl.import_model()
    print("IMPORT RESULT:", result)
except Exception:
    traceback.print_exc()
    raise SystemExit("IMPORT RAISED")

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if len(meshes) != 1:
    raise SystemExit(f"expected exactly 1 mesh after import, found {[o.name for o in meshes]}")
mesh = meshes[0]

# Export reads SMPL data from the ACTIVE object; import leaves the armature active.
bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh

try:
    result = bpy.ops.smpl.export_model()
    print("EXPORT RESULT:", result)
except Exception:
    traceback.print_exc()
    raise SystemExit("EXPORT RAISED")

out_path = os.path.join(run_dir, out_name)
if not os.path.isfile(out_path):
    raise SystemExit(f"export did not produce {out_path}")

sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
print(f"ROUNDTRIP OK mode={mode} out={out_path}")
print(f"SHA256 {sha}  {out_name}")
