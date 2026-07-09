#!/usr/bin/env python3
"""Build the installable Blender add-on zip for the SMIL Model Importer.

Produces ``smil_importer.zip`` containing the ``smil_importer/`` package, ready
to install via Blender's "Install from Disk…". Run from ``3D_model_prep/``:

    python build_addon.py
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = "smil_importer"
OUT = os.path.join(HERE, "smil_importer.zip")

SKIP_DIRS = {"__pycache__"}
SKIP_EXT = {".pyc", ".pyo"}


def main():
    pkg_dir = os.path.join(HERE, PKG)
    if not os.path.isdir(pkg_dir):
        raise SystemExit(f"Package folder not found: {pkg_dir}")
    if os.path.exists(OUT):
        try:
            os.remove(OUT)
        except OSError:
            pass  # some filesystems block delete; ZIP 'w' mode overwrites anyway
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(pkg_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                if os.path.splitext(name)[1] in SKIP_EXT:
                    continue
                abspath = os.path.join(root, name)
                arcname = os.path.relpath(abspath, HERE)
                zf.write(abspath, arcname)
                count += 1
    print(f"Wrote {OUT} ({count} files)")


if __name__ == "__main__":
    main()
