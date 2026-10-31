"""Build (or round-trip) a SMIL mouse model through the smil_importer Blender add-on, headless.

  blender -b --python 3D_model_prep/mouse_limbfix/build_mouse_limbfix.py -- \
      --src <in.pkl> --out_dir <dir for .blend + debug sidecars> --export_pkl <out.pkl> \
      --mode {roundtrip,build} [--author_limits {none,from_pkl}] [--addon_dir <dir containing smil_importer/>] \
      [--knee_from <pkl whose tibia_L/R pivots to use>] [--knee_half_width 0.06] [--max_weights 3]
      [--pivot_plan <pivot plan JSON>] [--proxy_plan <proxy bone plan JSON>]

roundtrip: import -> export unchanged (proves the add-on round trip); build: import -> limb fix -> export;
proxies: import -> append the plan's proxy leaf bones (Spine/Trunk format) -> export.
--author_limits from_pkl recreates the pkl's joint_limits as LOCAL Limit Rotation constraints before export (the add-on's
import does not do this, so without it the exporter writes its +-default range instead of the source limits).
Import runs with clean_mesh / symmetrise / regress_joints off so the mesh topology and the static joints are untouched.
"""

import argparse
import importlib.util  # noqa: F401  smil_importer.dependencies uses importlib.util without importing it (fails headless)
import json
import os
import pickle
import sys

import addon_utils
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--export_pkl", required=True)
ap.add_argument("--mode", choices=["roundtrip", "build", "proxies"], required=True)
ap.add_argument("--author_limits", choices=["none", "from_pkl"], default="from_pkl")
ap.add_argument(
    "--addon_dir", default=None, help="load smil_importer from this directory instead of the installed add-on"
)
ap.add_argument(
    "--knee_from", default=None, help="build: take the tibia_L/R (knee) pivots from this pkl (anatomical knee)"
)
ap.add_argument(
    "--knee_half_width", type=float, default=0.06, help="build: knee weight-transition half-width (model units)"
)
ap.add_argument(
    "--max_weights",
    type=int,
    default=3,
    help="build: max non-zero weights per vertex in re-split regions (3 lets thigh/shank/pelvis skin blend)",
)
ap.add_argument(
    "--pivot_plan", default=None, help="build: JSON pivot plan; replaces the v3 knee/limbfix options"
)
ap.add_argument(
    "--proxy_plan", default=None, help="proxies: JSON plan of leaf bones (name, parent, head, vertex) to append"
)
a = ap.parse_args(argv)
src, out_dir, export_pkl = (os.path.abspath(p) for p in (a.src, a.out_dir, a.export_pkl))
os.makedirs(out_dir, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

bpy.ops.wm.read_homefile(use_empty=True)  # clean scene: the add-on resolves some armatures globally
if a.addon_dir:
    for mod in addon_utils.modules():
        if mod.__name__ == "smil_importer" and addon_utils.check(mod.__name__)[1]:
            addon_utils.disable(mod.__name__)
    sys.path.insert(0, os.path.abspath(a.addon_dir))
    for k in [k for k in sys.modules if k == "smil_importer" or k.startswith("smil_importer.")]:
        del sys.modules[k]
    import smil_importer

    smil_importer.register()
else:
    addon_utils.enable("smil_importer", default_set=False)
    import smil_importer
print(f"[build] smil_importer {smil_importer.bl_info['version']} from {os.path.dirname(smil_importer.__file__)}")
from smil_importer.axis_remap import is_signed_permutation, rot3  # noqa: E402

st = bpy.context.scene.smpl_tool
st.pkl_filepath = src
st.npz_filepath = ""
st.regress_joints = False
st.clean_mesh = False
st.symmetrise = False
st.force_static_joint_locs = False
st.export_joint_limits = True
if hasattr(st, "import_joint_limits"):  # add-on with the issue-#116 fix recreates the limits itself
    st.import_joint_limits = a.author_limits == "from_pkl"
assert bpy.ops.smpl.import_model() == {"FINISHED"}, "import failed"
mesh = next(o for o in bpy.data.objects if o.type == "MESH")
arm = mesh.find_armature()
print(
    f"[build] imported mesh {mesh.name} ({len(mesh.data.vertices)} verts), armature {arm.name} ({len(arm.data.bones)} bones)"
)

with open(src, "rb") as f:
    data = pickle.load(f, encoding="latin1")
names = list(data["J_names"])
assert [b.name for b in arm.data.bones] == names, "bone order differs from J_names"


def model_to_bone_bounds(B, lo, hi):
    """Inverse of axis_remap.remap_bounds_to_model_frame (exact for identity / signed-permutation B, verbatim otherwise)."""
    if np.allclose(B, np.eye(3), atol=1e-3) or not is_signed_permutation(B):
        return list(lo), list(hi)
    blo, bhi = [0.0] * 3, [0.0] * 3
    for m in range(3):
        k = int(np.argmax(np.abs(B[m])))
        if B[m, k] >= 0:
            blo[k], bhi[k] = lo[m], hi[m]
        else:
            blo[k], bhi[k] = -hi[m], -lo[m]
    return blo, bhi


def author_limits(limits):
    for i, nm in enumerate(names):
        if i == 0:
            continue  # the exporter pins the root to [0, 0]
        B = rot3(arm.data.bones[nm].matrix_local)
        lo, hi = model_to_bone_bounds(B, limits[i, :, 0].astype(float), limits[i, :, 1].astype(float))
        c = arm.pose.bones[nm].constraints.new("LIMIT_ROTATION")
        c.owner_space = "LOCAL"
        c.use_limit_x = c.use_limit_y = c.use_limit_z = True
        c.min_x, c.max_x, c.min_y, c.max_y, c.min_z, c.max_z = lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]


B_before = {b.name: rot3(b.matrix_local) for b in arm.data.bones}
prov = {
    "source_pkl": src,
    "mode": a.mode,
    "author_limits": a.author_limits,
    "addon_version": list(smil_importer.bl_info["version"]),
    "blender": bpy.app.version_string,
}

if a.mode == "build":
    from limbfix_geometry import apply_pivot_plan, limbfix

    V = np.array([v.co for v in mesh.data.vertices], dtype=np.float64)
    assert np.abs(V - np.asarray(data["v_template"])).max() < 1e-5, "Blender mesh != pkl v_template"
    knee_J = None
    if a.knee_from:
        with open(a.knee_from, "rb") as f:
            ref = pickle.load(f, encoding="latin1")
        assert list(ref["J_names"]) == names, "--knee_from model has different joints"
        knee_J = np.asarray(ref["J"])
        prov["knee_from"] = os.path.abspath(a.knee_from)
    if a.pivot_plan:
        plan = json.load(open(a.pivot_plan))
        J_new, W_new, geo = apply_pivot_plan(np.asarray(data["J"]), V, np.asarray(data["weights"]), names, plan)
        prov["pivot_plan"] = {"path": os.path.abspath(a.pivot_plan), **plan}
    else:
        J_new, W_new, geo = limbfix(
            np.asarray(data["J"]),
            V,
            np.asarray(data["weights"]),
            names,
            knee_ref_J=knee_J,
            knee_half_width=a.knee_half_width,
            max_weights=a.max_weights,
        )
    prov["limbfix"] = geo
    # 1. pivots: move bone heads; keep each bone's +z tail and zero roll so its rest frame (and the limit remap) is unchanged
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for i in np.flatnonzero(np.abs(J_new - np.asarray(data["J"])).max(1) > 0):
        eb = arm.data.edit_bones[names[i]]
        eb.head = J_new[i].tolist()
        eb.tail = (J_new[i] + np.array([0.0, 0.0, 0.1])).tolist()
        eb.roll = 0.0
        print(f"[build] moved {names[i]} -> {np.round(J_new[i], 4)}")
    bpy.ops.object.mode_set(mode="OBJECT")
    bad = [b.name for b in arm.data.bones if not np.allclose(rot3(b.matrix_local), B_before[b.name], atol=1e-6)]
    assert not bad, f"bone rest frames changed: {bad}"
    # 2. skinning: rewrite the vertex groups of the changed bones
    W_old = np.asarray(data["weights"])
    for j in np.flatnonzero(np.abs(W_new - W_old).max(0) > 1e-7):
        vg = mesh.vertex_groups[names[j]]
        idx = np.flatnonzero((W_new[:, j] > 0) | (W_old[:, j] > 0))
        for v in idx:
            if W_new[v, j] > 0:
                vg.add([int(v)], float(W_new[v, j]), "REPLACE")
            else:
                vg.remove([int(v)])
        print(
            f"[build] re-skinned {names[j]}: {int((W_new[:, j] > 0).sum())} verts (was {int((W_old[:, j] > 0).sum())})"
        )

if a.mode == "proxies":
    # Proxy joints in the Spine/Trunk format: a leaf bone on the skin at a template vertex, parented to the bone that
    # carries it, no children. It shares that one vertex with the bones skinning it (Spine: 0.60 proxy / 0.40 parent)
    # and its rotation is locked, so it drives nothing. Bones are appended after the existing ones, which keep their
    # indices, rest frames (+z tail, zero roll) and skinning.
    plan = json.load(open(a.proxy_plan))
    share = float(plan["share"])
    V = np.array([v.co for v in mesh.data.vertices], dtype=np.float64)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for p in plan["proxies"]:
        assert p["name"] not in arm.data.edit_bones, f"{p['name']} exists"
        head = np.asarray(p["head"], float)
        assert np.abs(V[p["vertex"]] - head).max() < 1e-5, f"{p['name']}: head is not on vertex {p['vertex']}"
        eb = arm.data.edit_bones.new(p["name"])
        eb.head = head.tolist()
        eb.tail = (head + np.array([0.0, 0.0, 0.05])).tolist()
        eb.roll = 0.0
        eb.parent = arm.data.edit_bones[p["parent"]]
        eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for p in plan["proxies"]:
        v = int(p["vertex"])
        for g in mesh.data.vertices[v].groups:
            vg = mesh.vertex_groups[g.group]
            vg.add([v], g.weight * (1.0 - share), "REPLACE")
        mesh.vertex_groups.new(name=p["name"]).add([v], share, "REPLACE")
        c = arm.pose.bones[p["name"]].constraints.new("LIMIT_ROTATION")
        c.owner_space = "LOCAL"
        c.use_limit_x = c.use_limit_y = c.use_limit_z = True
        c.min_x = c.max_x = c.min_y = c.max_y = c.min_z = c.max_z = 0.0
        print(f"[build] proxy {p['name']} <- {p['parent']} at vertex {v} {np.round(p['head'], 4)} (locked)")
    bad = [n for n in names if not np.allclose(rot3(arm.data.bones[n].matrix_local), B_before[n], atol=1e-6)]
    assert not bad, f"bone rest frames changed: {bad}"
    prov["proxy_plan"] = {"path": os.path.abspath(a.proxy_plan), **plan}

n_con = sum(c.type == "LIMIT_ROTATION" for n in names for c in arm.pose.bones[n].constraints)  # source bones only
prov["limit_constraints_from_addon_import"] = n_con
if a.author_limits == "from_pkl" and "joint_limits" in data and n_con == 0:
    author_limits(np.asarray(data["joint_limits"]))  # pre-#116 add-on: its import does not recreate them
    print("[build] authored Limit Rotation constraints from the pkl's joint_limits (add-on import created none)")
else:
    print(f"[build] {n_con} Limit Rotation constraints present from the add-on import")

bpy.ops.wm.save_as_mainfile(
    filepath=os.path.join(out_dir, os.path.splitext(os.path.basename(export_pkl))[0] + ".blend")
)
for o in bpy.context.view_layer.objects:
    o.select_set(False)
bpy.context.view_layer.objects.active = mesh
mesh.select_set(True)
st.pkl_filepath = export_pkl  # the exporter writes to dirname(pkl_filepath)/output_filename
st.output_filename = os.path.basename(export_pkl)
assert bpy.ops.smpl.export_model() == {"FINISHED"}, "export failed"
assert os.path.exists(export_pkl), f"export did not write {export_pkl}"
with open(os.path.splitext(export_pkl)[0] + "_build_provenance.json", "w") as f:
    json.dump(prov, f, indent=1)
print(f"[build] wrote {export_pkl}")
