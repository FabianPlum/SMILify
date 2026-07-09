"""Clean, automatic handling of the add-on's external Python dependencies.

The SMIL importer needs ``scipy`` and ``scikit-learn``, which are *not* shipped
with Blender. Rather than crashing on import (as the old single-file script did),
this module lets the add-on register regardless, detects what is missing, and
offers a one-click installer in the add-on preferences.

Installation notes (Windows/macOS/Linux):

* Blender's bundled Python usually lives in a write-protected location (e.g.
  ``C:\\Program Files\\Blender Foundation\\...``). Installing there needs admin
  rights and fails otherwise. We therefore install into Blender's *user* modules
  directory with ``pip --target``, which is always writable and already on
  ``sys.path``.
* We do **not** try to upgrade pip itself: replacing a running ``pip.exe`` on
  Windows fails, and it is unnecessary.
"""

import importlib
import os
import subprocess
import sys

import bpy

# (import name, pip/distribution name)
DEPENDENCIES = [
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
]

MISSING_MESSAGE = (
    "Required packages (scipy, scikit-learn) are not installed. "
    "Open Edit > Preferences > Add-ons > 'SMIL Model Importer', click "
    "'Install dependencies', then restart Blender."
)


def user_modules_dir():
    """A user-writable directory that Blender already has on ``sys.path``.

    pip-installing here needs no admin rights and keeps Blender's own install
    untouched. Ensure it is importable in the current session too.
    """
    path = bpy.utils.user_resource("SCRIPTS", path="modules", create=True)
    if path and path not in sys.path:
        # Append (not insert) so Blender's bundled numpy keeps priority.
        sys.path.append(path)
    return path


def dependencies_installed():
    """Return True only if every required package can be imported."""
    return all(importlib.util.find_spec(mod) is not None for mod, _ in DEPENDENCIES)


def get_missing():
    """Return the pip names of the packages that are not currently importable."""
    return [pip_name for mod, pip_name in DEPENDENCIES if importlib.util.find_spec(mod) is None]


def _python_executable():
    """Path to the Python interpreter bundled with Blender.

    Since Blender 2.91 ``sys.executable`` points at Blender's own Python.
    """
    return sys.executable


def _run(cmd, env):
    """Run a subprocess, raising a RuntimeError with pip's real output on failure."""
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        output = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = "\n".join(output[-8:]) if output else "(no output)"
        raise RuntimeError(tail)


def install_dependencies():
    """pip-install the dependencies into Blender's user modules directory.

    Raises ``RuntimeError`` (with pip's stderr) if any step fails so the caller
    can surface a clean, actionable error to the user.
    """
    python_exe = _python_executable()
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    target = user_modules_dir()

    # Make sure a pip module exists in Blender's Python (only writes if missing).
    if importlib.util.find_spec("pip") is None:
        _run([python_exe, "-m", "ensurepip"], env)

    # Install into the user-writable target so we never touch Program Files and
    # never need administrator privileges.
    _run(
        [
            python_exe, "-m", "pip", "install",
            "--target", target,
            "--upgrade",
            *[pip_name for _, pip_name in DEPENDENCIES],
        ],
        env,
    )
    importlib.invalidate_caches()


class SMIL_OT_InstallDependencies(bpy.types.Operator):
    """Download and install the Python packages this add-on needs."""

    bl_idname = "smil.install_dependencies"
    bl_label = "Install dependencies (scipy, scikit-learn)"
    bl_description = (
        "Download and install scipy and scikit-learn into Blender's user "
        "modules folder (no admin required). Needs an internet connection and "
        "may take a minute"
    )
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        return not dependencies_installed()

    def execute(self, context):
        try:
            install_dependencies()
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to install dependencies: {exc}")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            "Dependencies installed. Please restart Blender to finish enabling all features.",
        )
        return {"FINISHED"}


class SMILAddonPreferences(bpy.types.AddonPreferences):
    """Preferences panel: shows dependency status and the install button."""

    bl_idname = __package__

    def draw(self, context):
        layout = self.layout
        if dependencies_installed():
            layout.label(text="All dependencies installed.", icon="CHECKMARK")
            return
        box = layout.box()
        box.label(text="Missing dependencies: " + ", ".join(get_missing()), icon="ERROR")
        box.label(text="Shape-space PCA and joint-regressor features need these packages.")
        box.operator(SMIL_OT_InstallDependencies.bl_idname, icon="IMPORT")
        box.label(text="After installing, restart Blender.", icon="INFO")
