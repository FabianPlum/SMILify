# DRAFT — follow-up issue (not yet posted)

**Proposed title:** Blender addon: inherited legacy bugs surfaced during the #88 review

**Proposed body:**

---

During the review of #88 (modular `smil_importer` port of the legacy
`SMIL_processing_addon.py`) we verified the port is faithful — a function-level
AST diff shows the ported logic is byte-identical to the legacy script, and
headless import→export round-trips produce **byte-identical pkls** (legacy vs.
new, Blender 4.2, dynamic- and static-joint models alike; see
`diagnostics/pr88_export_parity/` from the review session).

That faithfulness means the port also carries over a set of pre-existing bugs.
Per review policy these were deliberately **not** fixed in #88; they are
collected here. Line numbers refer to `3D_model_prep/smil_importer/` at the
#88 merge state; legacy line references give the provenance.

## A — Correctness bugs

1. **`export_joint_distances` crashes for every non-static model**
   ([measurements.py:115], legacy L1885). The `static_joint_locs` condition is
   inverted: `J_regressor` is *computed* only when static, but *used* only when
   non-static → `UnboundLocalError`. Fix: compute when `not static` (and
   initialise `J_regressor = None`). Flagged by Copilot on #88.

2. **`boundary_weights` J-regressor method can never activate**
   ([operators.py:1014-1017], legacy L3350-3354). `SMPL_OT_RecomputeJointPositions`
   reads `obj["kintree_table"]` / `obj["weights"]` custom properties that nothing
   ever writes → silent fallback to `inverse_distance`. Fix: source both arrays
   from the SMPL data embedded on the object (`get_smpl_data`), which always has
   them. Flagged by Copilot on #88.

3. **NaN joint positions when a vertex coincides with a joint**
   ([core_mesh.py:214], legacy L316). `find_nearest_neighbors` computes
   `1.0 / nearest_distances` without an epsilon → inf/inf → NaN row in the
   J_regressor → NaN bone heads. The sibling boundary-weights path already uses
   `epsilon = 1e-8` for exactly this.

4. **Multi-root armatures break `boundary_weights`**
   ([core_mesh.py:160-166, 263-266], legacy L262/365-368). The kintree builder
   hardcodes a single root at bone index 0; additional roots cause an
   IndexError on `parent_indices[0]`.

5. **`make_symmetrical` teleports unmatched vertices**
   ([core_mesh.py:626-634], legacy L1494-1509). A left-side vertex with no
   mirror partner within tolerance keeps `symIdx[i] == i` and gets "mirrored
   from itself" (y → −y). The left/right count check only prints and continues.

6. **`export_posedirs` is non-functional by design**
   ([core_mesh.py:462-475], legacy L564-587). It records `mesh.vertices[].co`
   per frame, which armature pose does not change → every frame is the rest
   shape. Legacy carried a warning comment ("disabled for now") that the port
   dropped; either implement via evaluated depsgraph or remove.

## B — Silent junk output

7. **Every J_regressor computation writes a stray `test_J_reg.csv` to the CWD**
   ([core_mesh.py:408, 455-457], legacy). `export_as_csv` defaults to `True`,
   no caller overrides it, and the filename is hard-coded. Fix: default
   `False`; when requested, write next to the provided `.npy` filepath.
   Flagged by Copilot on #88 (the default makes it worse than flagged).

8. **Every model export drops six `test_*.npy` debug files**
   ([model_build.py:411-416], legacy L1626). Written unconditionally next to
   the .blend (or CWD when unsaved). Should be opt-in debug output.

9. **Unconditional per-joint console print**
   ([core_mesh.py:271], legacy). One `print` in
   `J_regressor_from_boundary_weights` sits outside the `debug` gate and spams
   the console per joint. Flagged by Copilot on #88.

## C — Robustness / performance

10. **Armature created at the 3D-cursor position/orientation**
    ([model_build.py:213], legacy ~L807). `bpy.ops.object.add()` without
    explicit `location`/`rotation`/`align` inherits the cursor position and the
    user's add-align preference. Viewport-only effect (export reads local
    coordinates — round-trip parity confirmed unaffected), but confusing. Fix:
    create via `bpy.data.armatures.new()` + `bpy.data.objects.new()`.

11. **PCA component count never clamped** ([pca.py:30, 235];
    [properties.py:27-31]; legacy L945/1150). `number_of_PC` defaults to 20
    with no bound; fewer input scans than components → sklearn `ValueError`
    mid-operator. Clamp to `min(n_samples, n_features)`.

12. **`np.argpartition(distances, n)` crashes on meshes with ≤ n vertices**
    ([core_mesh.py:210], legacy L312) — toy/proxy meshes hit an opaque numpy
    error (all call sites pass n=10).

13. **Per-vertex `obj.data.update()` in `make_symmetrical`**
    ([core_mesh.py:660], legacy L1533) — one full mesh update per vertex;
    minutes on dense meshes. Hoist out of the loop.

## D — Nits

14. pca.py: `import csv` above the intended docstring makes it a no-op
    expression and shadows module-level csv ([pca.py:133-135], legacy
    L1048-1050); two dead no-op expressions ([pca.py:252-253], legacy
    L1167-1168); entangled-morph CSV block missing the `output_dir is not None`
    guard its sibling has ([pca.py:363], legacy L1278); inconsistent
    `overwrite_mesh` basis handling between plain and entangled PCA
    ([pca.py:47] vs [pca.py:273-277], legacy L962 vs L1188).

## E — Port-related items deferred to the #89 cycle

Not legacy bugs — these interact with the pose-correctives work and will be
handled when #89 is tackled:

15. **`Apply Pose Correctives` poll is permanently disabled for new imports**
    ([operators.py:1441]). #88's embedded data store writes `smpl_data_b64` +
    `has_smpl_data`, but the poll still requires the legacy `smpl_data_path`
    key. The fixed poll must also stay cheap (no full unpickle per UI redraw —
    e.g. a `has_posedirs` flag written at store time). Flagged by Copilot on #88.

16. **Features crash between "Install dependencies" and restart**
    (demonstrated during review on a fresh Blender 5.2 sandbox). After the
    in-session install, `dependencies_installed()` (find_spec-based) flips to
    True while the module-level `KDTree`/`PCA` bindings remain `None` →
    `TypeError` instead of the clean missing-deps message. Fix: resolve the
    imports lazily at call sites (retry once, then raise `MISSING_MESSAGE`).

## F — Pipeline-side enhancement

17. **numpy ≥ 2 Blender exports cannot be loaded by numpy < 2 environments**
    (verified empirically: Blender 5.2 exports fail in the `pytorch3d` env with
    `ModuleNotFoundError: numpy._core.numeric`). #88's README now recommends
    Blender 4.2 LTS for export; a proper fix is a `numpy._core → numpy.core`
    rename shim (custom `Unpickler.find_class`) in SMILify's pkl loaders, which
    already use a custom chumpy-aware unpickler.

---

Verification artifacts from the review session (parity matrix, installer live
test, comparison logs) are preserved locally under
`diagnostics/pr88_export_parity/`.
