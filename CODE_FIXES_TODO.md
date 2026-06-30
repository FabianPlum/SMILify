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
- **Source:** docs/animation_export_plan.md audit (minor UX).
- **Symptom:** `--export_animation` is `type=str` and used as an output path; `--export_animation True` creates `True.npz`/`True.json` instead of erroring. Easy footgun.
- **Decision needed:** validate/normalize, or leave as-is and only document (likely just doc).

---

## Discovered while fixing (append below as we go)

<!-- ID — title — status — file:line — note -->
