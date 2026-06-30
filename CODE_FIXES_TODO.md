# Code-side fixes discovered during doc ground-truthing

> **Temporary working doc** for the `ground_truthing_docs` branch (started 2026-06-30).
> While correcting the READMEs we keep hitting cases where **the code is wrong/inconsistent**,
> not the doc. Those are logged here so that once the doc pass is done we can tackle them as a
> second pass. **Delete this file before merging** (or fold remaining items into issues).

**Status legend:** 🔍 needs-verification · ✅ confirmed (ready to fix) · 🛠 fixed · ❎ won't-fix/not-a-bug

---

## Seeded from the README audit (2026-06-30)

### C1 — `config.CHECKPOINT_NAME` is undefined but read by `generate_video.py` 🔍
- **Source:** legacy/README.md audit (major).
- **Symptom:** [smal_fitter/generate_video.py](smal_fitter/generate_video.py) reads `config.CHECKPOINT_NAME` (lines ~36, ~70), but [config.py](config.py) never defines it (only `OUTPUT_DIR = "checkpoints/{timestamp}"`). Running `generate_video.py` as shipped → `AttributeError`.
- **Decision needed:** add `CHECKPOINT_NAME` to config.py, OR refactor generate_video.py to take a CLI arg / read `OUTPUT_DIR`.
- **Note:** legacy/optimization path — confirm it's still meant to work before fixing.

### C2 — `global_feats` may be computed but unused by the multi-view decoder 🔍
- **Source:** implementation_history/decoder_architecture_issues.md audit (major).
- **Symptom:** The doc claims issue #10 ("global_feats computed but never used") was FIXED via a `global_feat_proj` Linear — but no `global_feat_proj` symbol exists. `global_feats` is still computed and passed as the first positional arg to `self.transformer_head(global_feats, spatial_feats)` in [smal_fitter/neuralSMIL/multiview_smil_regressor.py](smal_fitter/neuralSMIL/multiview_smil_regressor.py) (~671-692). **Open question: does the transformer head actually consume `global_feats`, or is it dead compute?**
- **Action:** trace `transformer_head.forward` to confirm whether `global_feats` is used. If unused → either wire it in or drop the computation. Potential latent issue, not just a stale doc.

### C3 — Solver docstring says "via SVD" but code uses normal equations 🔍
- **Source:** implementation_history/triangulation_consistency_loss.md audit (minor).
- **Symptom:** Triangulation solver docstring at [multiview_smil_regressor.py:1682](smal_fitter/neuralSMIL/multiview_smil_regressor.py#L1682) says "via SVD" while the implementation uses normal equations + `torch.linalg.solve`.
- **Fix:** correct the in-code docstring (pure comment fix).

### C4 — Inconsistent CLI flag naming convention across scripts 🔍
- **Source:** neuralSMIL/README.md + configs/README.md audits (multiple).
- **Symptom:** Same concept spelled differently across entrypoints: `--smal_file` (run_multiview_inference.py, dataset_preprocessing.py) vs `--smal-file` (test_smil_regressor_ground_truth.py). Training scripts use underscores (`--batch_size`, `--num_epochs`) while several docs assumed hyphens. argparse does not treat them as aliases.
- **Decision needed:** standardize on underscores repo-wide (and optionally register hyphen aliases) so commands are predictable for students.

### C5 — `multiview_mouse_UNET_long.json` sets `use_mixed_precision: false` 🔍
- **Source:** configs/examples/README.md audit (major).
- **Symptom:** The "long/optimized" example config disables mixed precision ([configs/examples/multiview_mouse_UNET_long.json:194](smal_fitter/neuralSMIL/configs/examples/multiview_mouse_UNET_long.json#L194)). Doc described it as using mixed precision.
- **Decision needed:** is MP intentionally off here? If it was meant to be on, this is a config bug; otherwise just a doc fix.

### C6 — `fitter_3d` scheme names don't match their parameters 🔍
- **Source:** fitter_3d/README.md audit (major/minor).
- **Symptom:** In `SMALParamGroup.param_map` ([fitter_3d/trainer.py:238-249](fitter_3d/trainer.py#L238-L249)): `pose` includes `betas`/`log_beta_scales` (shape params) and adds `betas_trans`; `shape` drops `joint_rot` but also adds `betas_trans`. The names mislead ("pose" ≠ no-shape).
- **Decision needed:** rename/redefine schemes to match contents, or accept and document. (Low priority — mostly a clarity wart.)

### C7 — `--export_animation` silently treats any string as a path 🔍
- **Source:** docs/design/animation_export_plan.md audit (minor UX).
- **Symptom:** `--export_animation` is `type=str` and used as an output path; `--export_animation True` creates `True.npz`/`True.json` instead of erroring. Easy footgun.
- **Decision needed:** validate/normalize, or leave as-is and only document (likely just doc).

---

## Discovered while fixing (append below as we go)

<!-- ID — title — status — file:line — note -->

### C8 — Test suite does not run clean under any single pytest invocation ✅
- **Found:** while verifying tests/README (P0 #3) by *actually running* `pytest`.
- **Empirical state (env `pytorch3d`, 2026-06-30):**
  - `pytest tests/ -m "not slow"` → **70 passed**, but `test_triangulation_consistency.py` (12) errors and `test_animation_export.py` (5) errors on collection.
  - `PYTHONPATH=smal_fitter pytest tests/ -m "not slow"` → triangulation passes (76 total), **but `test_fitter_3d_optimise` then FAILS** (see C10).
  - bare `pytest` from repo root also collects `smal_fitter/sleap_data/test_sleap_preprocessing.py` (see C12), which errors.
- **Root cause:** no `tests/conftest.py` and no `testpaths` in `pytest.ini`, so each test module relies on its own ad-hoc `sys.path` hacks, which conflict.
- **Fix (proposed):** add a `conftest.py` that puts the right dirs on `sys.path` once, add `testpaths = tests` to `pytest.ini`, and resolve C9/C10/C11/C12 so `pytest tests/` just works.

### C9 — `smal_fitter` is ambiguous (dir vs module); no `smal_fitter/__init__.py` ✅
- **Symptom:** `from smal_fitter import SMALFitter` ([smal_fitter/neuralSMIL/smil_image_regressor.py:24](smal_fitter/neuralSMIL/smil_image_regressor.py#L24)) only resolves when `smal_fitter/` is on `sys.path` (so `smal_fitter` → `smal_fitter.py`). Under pytest, repo root is on the path first, so `smal_fitter` resolves to the **directory** (namespace package) and the import fails: `cannot import name 'SMALFitter' from 'smal_fitter'`.
- **Fix:** decide whether `smal_fitter` is a package (add `__init__.py`, make imports `from smal_fitter.smal_fitter import SMALFitter`) or keep it a sys.path dir — and make it consistent. Affects all of neuralSMIL.

### C10 — `utils` module name collision (`smal_fitter/utils.py` vs `fitter_3d/utils.py`) ✅
- **Symptom:** with `smal_fitter/` on `PYTHONPATH`, `from utils import perspective_proj_withz` ([smal_fitter/p3d_renderer.py:17](smal_fitter/p3d_renderer.py#L17)) resolves to `fitter_3d/utils.py` (wrong module, no such symbol) → `test_fitter_3d_optimise` fails. Without it, the triangulation tests fail instead.
- **Fix:** disambiguate the two `utils` modules (package-qualified imports, or rename one).

### C11 — `tests/test_animation_export.py` uses an import style that can't work ✅
- **Symptom:** `from smal_fitter.neuralSMIL.animation_export import ...` requires `smal_fitter` (and `neuralSMIL`) to be importable **packages**; they have no `__init__.py`, so this errors (`'smal_fitter' is not a package`) under every invocation tried. Inconsistent with the rest of the suite (which imports via `sys.path`).
- **Fix:** rewrite the import to match the suite's convention (depends on C9 decision).

### C12 — stray test file collected from repo root ✅
- **Symptom:** `smal_fitter/sleap_data/test_sleap_preprocessing.py` is picked up by bare `pytest` (no `testpaths`) and errors: `from preprocess_sleap_dataset import ...` (cwd-relative).
- **Fix:** add `testpaths = tests` to `pytest.ini` (and/or move/repair this file).
