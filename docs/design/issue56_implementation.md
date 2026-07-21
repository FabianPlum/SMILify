# Issue #56 — User-Defined Joint Limits: Implementation & Test Report

Companion doc: [Joint Limits User Guide](../joint_limits_user_guide.md) (novice-friendly, how to author limits in Blender).

**Status: DONE and validated (PASS).** All layers work — Blender authoring, `.pkl` storage, optimisation fitter, neural inference — with no regressions.

---

## 1. Objective

Before #56, a custom 3D model had no way to say "this joint can only bend this far." The code had a placeholder that clamped every non-root joint to ±0.01 rad (~0.57°), which effectively **froze new models solid** whenever the limit loss was on.

Issue #56 lets the user **author the limits in Blender** — e.g. "this knee rotates −30°…+45° on Z" — and have both fitting pipelines respect them automatically.

Design idea: **limits are data, not new math.** They travel along a pipe:

```
Blender bone limits → .pkl['joint_limits'] → LimitPrior → existing hinge loss
```

The loss functions were already correct. The new code is (a) reading limits off bones at export, (b) remapping them from the bone-local frame into the model frame, (c) reading them back in with validation, and (d) an optional limit penalty on the neural side.

---

## 2. Changes

### 2.1 Data format (schema)

New `.pkl` key `joint_limits`:

- `np.ndarray`, shape `(J, 3, 2)`, dtype `float32`; `J = len(dd["J_names"])`.
- Axis 1 = the three axis-angle components `(x, y, z)` in **radians**, in the **model frame** — the same space as `joint_rotations` in the SMAL forward pass, so limits compare directly against predicted/optimised rotations.
- Axis 2 = `[min, max]`.
- Ordering matches `dd["J_names"]` (`armature.data.bones` order) — automatic because the export helper iterates the same bones as the other exporters.
- Root joint (index 0) = all zeros (the fitter drops the root via `[3:]`).

Conventions:

1. **Unconstrained axis → wide-open** `[-π, π]` (replaces the old `±0.01` placeholder), so the limit prior is inactive until real limits are authored.
2. **Locked DOF → `min == max`** (e.g. `[0, 0]`); the existing hinge already penalises any deviation, no separate mask needed.
3. **Guaranteed `min ≤ max`**: the exporter swaps inverted user-authored bounds.

**Backward compatibility:** a `.pkl` without `joint_limits` falls back to wide-open, **not** the old freeze — an intentional behaviour change so unauthored custom models stop being pose-frozen. The legacy quadruped path (`ignore_hardcoded_body = False`) keeps its hard-coded ranges untouched.

### 2.2 Blender add-on — authoring & export

- `smil_importer/properties.py`: two panel props — `export_joint_limits` (toggle, default on) and `joint_limit_default_range` (half-range for unset axes, default π).
- `smil_importer/ui.py`: toggle + default-range field above the *Export SMIL Model* button.
- `smil_importer/core_mesh.py`: new `export_joint_limits_to_npy(...)`. Per bone, per axis, reads limits in priority order: a `LIMIT_ROTATION` pose-bone constraint first, otherwise the bone's IK limits/locks (`lock_ik_*` → pinned to 0, `use_ik_limit_*` → `ik_min_*`/`ik_max_*`). Unset axes → `[-default_range, +default_range]`; root → zeros.
- `smil_importer/model_build.py`: writes `pkl_data["joint_limits"]` in `export_smpl_model` when the toggle is on.

### 2.3 Bone-local → model-frame axis remap (`smil_importer/axis_remap.py`)

A Limit Rotation constraint is authored in the **bone-local** frame, but `LimitPrior` compares **model-frame** axis-angle components. For a bone whose rest orientation is tilted, a bound authored on bone-local Y would land on the wrong model axis.

New `bpy`-free module with three helpers, applied inside `export_joint_limits_to_npy`:

- `rot3(m4)` — upper-left 3×3 of `bone.matrix_local`; columns are the bone-local axes in model coordinates (`B`).
- `is_signed_permutation(B)` — true iff `B` only permutes/flips axes.
- `remap_bounds_to_model_frame(B, lo, hi)` — for a signed-permutation `B` (the common "clean axis" rig case) the axis-aligned box remaps **exactly**: permute rows, and a sign flip maps `[lo, hi] → [-hi, -lo]`. Identity `B` is a no-op. A genuinely mixed-axis `B` cannot be represented as a per-axis box, so bounds are exported verbatim with a warning (bounded #56 caveat — accurate for moderate per-axis limits, approximate for extreme combined-axis rotations).

Kept `bpy`-free deliberately so it is unit-testable without Blender (`tests/test_axis_remap.py`); `core_mesh` imports it under the old private aliases for backward compatibility.

### 2.4 Optimisation fitter — consuming the limits

- `smal_fitter/priors/joint_limits_prior.py`: shared helper `_ranges_from_joint_limits(dd, default_range=np.pi)`. If `dd['joint_limits']` is present it validates shape `(J, 3, 2)`, `min ≤ max`, and finiteness, then builds the ranges; otherwise wide-open `[-π, π]`. Root always forced to zero. Both module-level `Ranges` and `LimitPrior.__init__` call it — single source of truth (and `__init__` rebuilds from the current `config.dd`, so `apply_smal_file_override` is respected).
- **No change to the loss itself.** `fitter.py` still uses the same hinge `max(x − max, 0) + max(min − x, 0)`; it just receives better numbers.

### 2.5 Neural inference — optional limit penalty

`joint_angle_regularization` pulls angles toward **zero**, not toward authored **limits**, so a separate, **off-by-default** penalty was added to both `smal_fitter/neuralSMIL/multiview_smil_regressor.py` and `smal_fitter/neuralSMIL/smil_image_regressor.py`. It mirrors the fitter's hinge and:

- defaults to weight `0.0` → existing training is bit-for-bit unchanged;
- reuses `LimitPrior` → neural bounds identical to the fitter's;
- handles both 6D and axis-angle rotation representations (6D is converted before the hinge);
- caches bounds once on `self._joint_limit_bounds`;
- is wrapped in try/except → a shape mismatch can never crash a run.

Enable via `"joint_limit_regularization": <weight>` (start small, e.g. `1e-3`) in the training config's `loss_weights`.

### 2.6 Drive-by fix

`core_mesh.export_J_regressor_to_npy` wrote a debug CSV to a hard-coded bare path (`"test_J_reg.csv"`) in Blender's read-only CWD, crashing *any* export with a permission error. It now writes next to the `.npy` and is wrapped in try/except. Unrelated to #56, found and fixed along the way.

### 2.7 Files changed

| File | Change |
|---|---|
| `smil_importer/properties.py` | 2 new panel props |
| `smil_importer/ui.py` | panel toggle + default-range field |
| `smil_importer/core_mesh.py` | `export_joint_limits_to_npy` (with axis remap) + J_regressor CSV crash fix |
| `smil_importer/axis_remap.py` | **new** — bpy-free bone-local → model-frame remap helpers |
| `smil_importer/model_build.py` | write `joint_limits` in `export_smpl_model` |
| `smal_fitter/priors/joint_limits_prior.py` | read `joint_limits`, validation, wide-open fallback |
| `smal_fitter/neuralSMIL/multiview_smil_regressor.py` | optional limit penalty |
| `smal_fitter/neuralSMIL/smil_image_regressor.py` | optional limit penalty |
| `tests/test_axis_remap.py` | **new** — pytest suite for the remap (no Blender needed) |
| `diagnostics/check_remap_math.py` | randomized membership-equivalence property check |
| `diagnostics/probe_issue56.py` | Phase-0 ground-truth probe |
| `diagnostics/test_issue56.py` | consumer unit tests |
| `diagnostics/test_fitter_limit_loss.py` | fitter loss behaviour |
| `diagnostics/test_neural_limit_penalty.py` | neural penalty behaviour |

---

## 3. Tests done & results

| # | Test | How | Result |
|---|------|-----|--------|
| 0 | Phase-0 probe | `python -m diagnostics.probe_issue56` | PASS — confirmed the `±0.01` placeholder froze models; schema/ordering/loss inputs verified before any edit |
| 1 | Static compile | `py_compile` on all edited files | clean |
| 2 | Consumer unit tests | `python -m diagnostics.test_issue56` | **5/5 passed** |
| 3 | Blender round-trip | export in Blender, inspect `.pkl` | PASS |
| 4 | Fitter limit loss | `python -m diagnostics.test_fitter_limit_loss` | PASS |
| 5 | Neural penalty | `python -m diagnostics.test_neural_limit_penalty` | **3/3 passed** |
| 6 | Axis-remap unit tests | `pytest tests/test_axis_remap.py` | **8/8 passed** |
| 7 | Remap property check | `python diagnostics/check_remap_math.py` | PASS — **0 mismatches / 20 000 random trials** |
| 8 | Full regression suite | `pytest -m "not slow"` | **78 passed, 6 skipped** |

What each proves:

- **Consumer unit tests (5/5):** no-limits `.pkl` → every non-root joint `[-π, π]`, root `[0, 0]` (old `±0.01` gone); injected `joint_limits` reach the correct `(N_POSE, 3)` slot the fitter reads; hinge is zero inside the range and positive past it; validation rejects wrong shapes and `min > max`.
- **Blender round-trip:** exported a 55-joint model with a Limit Rotation constraint (Z, −30°/45°) on bone `l_3_pt_r`. Result: `joint_limits (55, 3, 2)`, root zeroed, exactly one constrained joint with `Z = [-0.5236, 0.7854]` (radians of −30°/45°), its unticked X/Y at `±π`. Only the ticked axis carried the authored value.
- **Fitter limit loss:** against that export, in-range loss `0.000000`; out-of-range loss `0.003086`; gradient at the violated entry `+0.006173` (descent pulls the joint back inside).
- **Neural penalty (3/3):** batched hinge legal → 0, violation → `0.003086` with corrective gradient; the real 6D→axis-angle path gives loss 0 for an identity pose; both regressors default the weight to `0.0`.
- **Axis-remap tests (8/8):** `rot3` extraction; signed-permutation detection; exact remap under the hand-analysed tilt `B = [[1,0,0],[0,0,−1],[0,1,0]]` (shared by bones `l_3_tr_r`, `b_a_1`); identity no-op; mixed-axis `B` → verbatim + warning; remapped bounds stay ordered.
- **Remap property check:** over 20 000 random (signed-permutation `B`, box, rotation) trials, a rotation inside the authored *bone-local* box is inside the exported *model-frame* box and vice versa — i.e. the fitter enforces exactly the box the user drew, 0 mismatches.

**Key finding:** the feature worked on the first try; every failure during manual testing was a Blender workflow trap (see the pitfalls section of the user guide), not a bug in the limits code.

---

## 4. Follow-ups (out of scope for #56)

- Optional `joint_limits_enabled` mask `(J, 3)` for explicit per-DOF disable (currently `min == max` covers locking). If added, the fitter's `splay` term — which hard-codes hinge axes `[0, 2]` — should read the mask instead of assuming a fixed axis set.
- Enable the neural `joint_limit_regularization` weight in a real training config and evaluate its effect on learned poses.
- Exact handling of mixed-axis rest orientations (currently verbatim + warning).
