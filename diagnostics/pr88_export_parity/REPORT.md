# PR #88 export-parity test report (2026-07-24)

Diagnostic artifacts for review — not part of the addon, never committed.
Scripts: `run_roundtrip.py` (Blender-side driver), `compare_exports.py` (comparator),
`install_deps_52.py` (live installer test). Raw logs sit next to each run dir.

## Setup

Identical, mutation-free import→export round-trip driven through the real operators
(`smpl.import_model` → `smpl.export_model`) with `regress_joints=False`,
`clean_mesh=False`, `symmetrise=False`, `shapekeys_from_PCA=False`, no npz.

Inputs:
- `SMILy_STICK.pkl` (dynamic joints → exercises J_regressor/KDTree path)
- `SMILy_Mouse_static_joints_Falkner_conv.pkl` (static joints path)

Environments:
- Blender 4.2.0 (Python 3.11.7, numpy 1.24.3) — scipy 1.14.0/sklearn 1.5.1 present in bundled site-packages
- Blender 5.2.0 (Python 3.13.13, numpy 2.3.4) — fresh; deps installed by the addon's own installer (with the numpy-pin fix)
- Consumer env: pytorch3d conda env (numpy 1.26.4)

## Results

### A — Port correctness (Blender 4.2, legacy vs new): PASS, byte-identical

| model | new run a | new run b | legacy | verdict |
|---|---|---|---|---|
| stick | `988ee402…` | `988ee402…` | `988ee402…` | deterministic + byte-identical |
| mouse | `4b1ff243…` | `4b1ff243…` | `4b1ff243…` | deterministic + byte-identical |

The modular port produces bit-exact the same pkl as the 4000-line legacy script.

### B — Environment sensitivity (Blender 5.2 vs 4.2): NOT byte-identical (expected), and worse

5.2 exports succeed but differ in bytes (numpy 2 pickles reference
`numpy._core.*` module paths; numpy 1 pickles reference `numpy.core.*` — no pin
strategy can change this).

### C — Consumer load test (pytorch3d env, numpy 1.26.4)

| export origin | load result |
|---|---|
| Blender 4.2 (all runs) | PASS |
| Blender 5.2 (both models) | **FAIL** — `ModuleNotFoundError: No module named 'numpy._core.numeric'` |

numpy 1.26.4's forward-compat stubs do not cover everything numpy 2.3 pickles
reference. **pkls exported from a numpy-2 Blender are unusable in the current
SMILify pipeline env.**

### Installer live test (Blender 5.2, fresh sandbox): PASS

- Addon registers cleanly on Blender 5.2 (API-compatible).
- `Install dependencies` detects missing scipy/scikit-learn, pip-installs into the
  user modules dir: scipy 1.18.0, scikit-learn 1.9.0, **numpy 2.3.4 = exactly the
  bundled version** (the numpy-pin fix works as intended; no shadowing hazard).
- Confirmed the "stale binding" finding empirically: right after the in-session
  install, `dependencies_installed()` returns True while `core_mesh.KDTree` is
  still `None` — a KDTree-dependent operator clicked before restarting Blender
  would raise TypeError instead of a clean message.

## Conclusions

1. The port itself is **correct**: byte-exact vs legacy under the same Blender.
2. Cross-Blender byte parity is structurally impossible (numpy pickle framing).
3. **Recommendation for docs:** bless Blender 4.2 LTS as the reference export
   environment while the SMILify consumer env runs numpy < 2. Exports from
   numpy-2 Blenders (4.5+/5.x) will not load there.
4. Optional repo-side follow-up: extend SMILify's pkl loaders with a
   `numpy._core → numpy.core` module-rename shim (custom `Unpickler.find_class`)
   so numpy-2 exports become loadable; the repo already uses a custom
   chumpy-aware unpickler, so this is a natural place.
