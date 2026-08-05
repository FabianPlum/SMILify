"""Live test of the addon's dependency installer under Blender 5.2 (sandboxed).

DIAGNOSTIC HARNESS for reviewing PR #88 — not part of the addon, never committed.
Also probes the 'stale None binding' finding: core_mesh.KDTree should still be
None right after an in-session install (restart required), while
dependencies_installed() already flips to True.
"""

import os
import sys
import traceback

import bpy

sys.path.insert(0, "/home/fabi/dev/SMILify/3D_model_prep")
import smil_importer

smil_importer.register()

from smil_importer import core_mesh
from smil_importer import dependencies as deps

print("deps installed BEFORE:", deps.dependencies_installed())
print("missing:", deps.get_missing())
print("sys.executable:", sys.executable)

try:
    result = bpy.ops.smil.install_dependencies()
    print("INSTALL RESULT:", result)
except Exception:
    traceback.print_exc()
    raise SystemExit("INSTALL RAISED")

print("deps installed AFTER (find_spec):", deps.dependencies_installed())
print("core_mesh.KDTree binding after install (expect None pre-restart):", core_mesh.KDTree)

target = deps.user_modules_dir()
top = sorted(d for d in os.listdir(target) if not d.startswith("_"))
print("target dir:", target)
print("installed packages:", [d for d in top if not d.endswith((".dist-info", ".libs"))])
print("dist-infos:", [d for d in top if d.endswith(".dist-info")])
