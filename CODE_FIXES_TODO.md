# Code-side fixes discovered during doc ground-truthing

> **Temporary working doc** for the `ground_truthing_docs` branch (started 2026-06-30).
> While correcting the READMEs we keep hitting cases where **the code is wrong/inconsistent**,
> not the doc. Those are logged here so that once the doc pass is done we can tackle them as a
> second pass. **Delete this file before merging** (or fold remaining items into issues).

**Status legend:** 🔍 needs-verification · ✅ confirmed (ready to fix) · 🛠 fixed · ❎ won't-fix/not-a-bug

> **Each open item has a GitHub issue** (linked in its header). Note: `C#` are this file's internal tracker ids; `#N` are GitHub issue/PR numbers (one shared sequence) — they are **not** the same. C14/C15 are fixed (no issue); C17 → existing #15; C18 → #46; C19 → #47; C20 → #48.

> **🛠 RESOLVED (2026-07-01) — import-structure refactor (branch `refactor/import-structure`):** **C8–C12 and the umbrella C16 are fixed.** Approach: make the tree a proper package (empty `__init__.py` in `smal_fitter/`, `smal_fitter/neuralSMIL/`, `smal_fitter/priors/`, `fitter_3d/`, `fitter_3d/pointcloud2smil/`, `custom_processing/`), rename `smal_fitter/smal_fitter.py` → `smal_fitter/fitter.py`, rewrite **all** cross-module imports to the absolute repo-root form, **remove all 47 `sys.path` hacks**, and launch entrypoints as `python -m <module>` from the repo root. **Deliberately NOT pip-installable** (rejected as bloat) — the goal was one consistent import convention, not packaging. Verified: `pytest` green (**83 passed** via `pytest` / `pytest tests/` / `python -m pytest`, no `PYTHONPATH`), 69/69 modules + 175 internal import targets resolve, and the benchmark / inference (incl. `mp.spawn`) / training / legacy `optimize_to_joints` flows all run under `-m`.

> **🛠 RESOLVED (2026-07-01) — post-refactor cleanup + CI:**
> - **C1** (#29) & **C3** (#31): fixed in code (`CHECKPOINT_NAME` defined + CLI; triangulation docstring corrected). Issues closed.
> - **C7** (#35): resolved doc-only (behaviour documented, community PR). Issue closed.
> - **C2** (#30), **C19** (#47), **C20** (#48): resolved in **PR #58** — C2 verified intentional (the pooled `global_feats` is deliberately not consumed by the decoder) and documented; C19/C20 dead modules deleted. Issues closed.
> - **CI groundwork for C17:** GitHub Actions added in **PR #59** (`.github/workflows/tests.yml`, CPU, `pytest -m "not slow"`).
> - **Still open:** **C4** (#32), **C5** (#33 — verdict: not-a-bug/doc), **C6** (#34), **C13** (#41 — partially addressed), **C17** (#15 — next), and new **C21** (dead `scipy.misc` import).

---

## Seeded from the README audit (2026-06-30)

### C1 — `config.CHECKPOINT_NAME` is undefined but read by `generate_video.py` 🛠 FIXED  · [#29](https://github.com/FabianPlum/SMILify/issues/29)
- **DONE (2026-07-01):** `config.py` now defines `CHECKPOINT_NAME = None`, and `generate_video.py` requires the checkpoint via CLI (`--checkpoint_name`) or config, erroring clearly if unset (commit 926d433). Issue closed.
- **Source:** legacy/README.md audit (major).
- **Symptom:** [smal_fitter/generate_video.py](smal_fitter/generate_video.py) reads `config.CHECKPOINT_NAME` (lines ~36, ~70), but [config.py](config.py) never defines it (only `OUTPUT_DIR = "checkpoints/{timestamp}"`). Running `generate_video.py` as shipped → `AttributeError`.
- **Decision needed:** add `CHECKPOINT_NAME` to config.py, OR refactor generate_video.py to take a CLI arg / read `OUTPUT_DIR`.
- **Note:** legacy/optimization path — confirm it's still meant to work before fixing.

### C2 — `global_feats` may be computed but unused by the multi-view decoder ❎ NOT-A-BUG (documented)  · [#30](https://github.com/FabianPlum/SMILify/issues/30)
- **DONE (2026-07-01, PR #58):** traced `transformer_head.forward` — the pooled `global_feats` is passed only as a **batch-size/device carrier**; its content is **deliberately not consumed** (an earlier pooled-global input into the decoder token gave the head a memorisable image-level fingerprint and drove train/val divergence on betas, so it was removed). No `global_feat_proj` ever existed (the doc's claimed fix was stale). Not dead-compute-worth-removing; documented at the call site + head docstring so it isn't re-wired in. Issue closed.
- **Source:** implementation_history/decoder_architecture_issues.md audit (major).
- **Symptom:** The doc claims issue #10 ("global_feats computed but never used") was FIXED via a `global_feat_proj` Linear — but no `global_feat_proj` symbol exists. `global_feats` is still computed and passed as the first positional arg to `self.transformer_head(global_feats, spatial_feats)` in [smal_fitter/neuralSMIL/multiview_smil_regressor.py](smal_fitter/neuralSMIL/multiview_smil_regressor.py) (~671-692). **Open question: does the transformer head actually consume `global_feats`, or is it dead compute?**
- **Action:** trace `transformer_head.forward` to confirm whether `global_feats` is used. If unused → either wire it in or drop the computation. Potential latent issue, not just a stale doc.

### C3 — Solver docstring says "via SVD" but code uses normal equations 🛠 FIXED  · [#31](https://github.com/FabianPlum/SMILify/issues/31)
- **DONE (2026-07-01):** docstring corrected to "damped normal equations (Tikhonov-regularized least squares)" (commit f2b4a42). Issue closed.
- **Source:** implementation_history/triangulation_consistency_loss.md audit (minor).
- **Symptom:** Triangulation solver docstring at [multiview_smil_regressor.py:1682](smal_fitter/neuralSMIL/multiview_smil_regressor.py#L1682) says "via SVD" while the implementation uses normal equations + `torch.linalg.solve`.
- **Fix:** correct the in-code docstring (pure comment fix).

### C4 — Repo-wide CLI flag inconsistencies ✅  · [#32](https://github.com/FabianPlum/SMILify/issues/32)
- **Source:** neuralSMIL/README.md + configs/README.md audits; **expanded 2026-07-01 via a full-repo argparse scan** (108 `.py` files, 41 define flags). argparse does NOT treat `-` and `_` as aliases, so a command copy-pasted between scripts errors out. Two classes:

**(a) Same flag, inconsistent hyphen vs underscore spelling:**

| Concept | `--hyphen-form` (files) | `--underscore_form` (files) |
|---|---|---|
| smal model file | benchmark_model, dataset_preprocessing, test_smil_regressor_ground_truth | run_multiview_inference, sleap_data/*, replicAnt_data/* |
| shape family | benchmark_model, dataset_preprocessing, test_smil_regressor_ground_truth, pointcloud2smil/sample_smil_model | run_multiview_inference, replicAnt_data/*, tests/validate_multiview_replicant_loader |
| batch size | fitter_3d/pointcloud2smil/{sample_smil_model, smil_pointnet} | benchmark_model, train_multiview_regressor, train_smil_regressor |
| num workers | fitter_3d/pointcloud2smil/smil_pointnet | benchmark_model, dataset_preprocessing, preprocess_dataset, sleap_data/*, replicAnt_data/* |
| output dir | fitter_3d/pointcloud2smil/sample_smil_model, plot_pca_data | custom_processing/batch_process_models, fitter_3d/SDF_batch, fitter_3d/SDF_tests, tests/validate_multiview_replicant_loader |
| rotation representation | fitter_3d/pointcloud2smil/smil_pointnet, test_smil_regressor_ground_truth | dataset_preprocessing, preprocess_dataset, train_smil_regressor, sleap_data/sleap_dataset |
| master port | run_multiview_inference, train_multiview_regressor | train_smil_regressor |

**(b) Same concept, different flag NAME (dataset input) — and one overloaded name:**

| Entrypoint | dataset-input flag | model-file flag |
|---|---|---|
| benchmark_model.py | `--dataset_path` | `--smal-file` |
| run_multiview_inference.py | `--dataset` | `--smal_file` |
| train_multiview_regressor.py | `--dataset_path` | (JSON only) |
| train_smil_regressor.py | `--data_path` | (JSON only) |

- The dataset input is spelled **three** ways: `--dataset_path` / `--dataset` / `--data_path`.
- **`--dataset` is overloaded:** an HDF5 path in `run_multiview_inference.py`, but a fixed-choice *legacy selector* (`masked_simple`/`pose_only_simple`/`test_textured`/`simple`) in `train_smil_regressor.py` — a genuine footgun.
- *Not* an inconsistency: `--num_epochs` is consistently underscore in code (the `--num-epochs` in old docs was a doc error, now fixed).

- **Fix (proposed):** standardize on underscores (the neuralSMIL majority); register the other spelling as an argparse alias per flag for back-compat; give the dataset input ONE name (`--dataset_path`) across benchmark/inference/train; and rename `train_smil_regressor.py`'s legacy `--dataset` selector to something unambiguous (e.g. `--builtin_dataset`).

### C5 — `multiview_mouse_UNET_long.json` sets `use_mixed_precision: false` ✅ VERDICT: not-a-bug (doc)  · [#33](https://github.com/FabianPlum/SMILify/issues/33)
- **VERDICT (2026-07-01):** `use_mixed_precision: false` is the **majority** across the example configs (7 of 10 set it false), so it is intentional/normal — **not** a config bug, just a doc mismatch. #33 kept open pending a doc fix + close as not-a-bug (deferred closing note).
- **Source:** configs/examples/README.md audit (major).
- **Symptom:** The "long/optimized" example config disables mixed precision ([configs/examples/multiview_mouse_UNET_long.json:194](smal_fitter/neuralSMIL/configs/examples/multiview_mouse_UNET_long.json#L194)). Doc described it as using mixed precision.
- **Decision needed:** is MP intentionally off here? If it was meant to be on, this is a config bug; otherwise just a doc fix.

### C6 — `fitter_3d` scheme names don't match their parameters 🔍  · [#34](https://github.com/FabianPlum/SMILify/issues/34)
- **Source:** fitter_3d/README.md audit (major/minor).
- **Symptom:** In `SMALParamGroup.param_map` ([fitter_3d/trainer.py:238-249](fitter_3d/trainer.py#L238-L249)): `pose` includes `betas`/`log_beta_scales` (shape params) and adds `betas_trans`; `shape` drops `joint_rot` but also adds `betas_trans`. The names mislead ("pose" ≠ no-shape).
- **Decision needed:** rename/redefine schemes to match contents, or accept and document. (Low priority — mostly a clarity wart.)

### C7 — `--export_animation` silently treats any string as a path 🛠 RESOLVED (doc-only)  · [#35](https://github.com/FabianPlum/SMILify/issues/35)
- **DONE (2026-07-01):** resolved doc-only — the flag's behaviour is now documented in the inference scripts (commit e92aeaf, via a community PR); the decision was to document rather than add validation. Issue closed.
- **Source:** docs/design/animation_export_plan.md audit (minor UX).
- **Symptom:** `--export_animation` is `type=str` and used as an output path; `--export_animation True` creates `True.npz`/`True.json` instead of erroring. Easy footgun.
- **Decision needed:** validate/normalize, or leave as-is and only document (likely just doc).

---

## Discovered while fixing (append below as we go)

<!-- ID — title — status — file:line — note -->

### C8 — Test suite does not run clean under any single pytest invocation 🛠 FIXED  · [#36](https://github.com/FabianPlum/SMILify/issues/36)
- **Found:** while verifying tests/README (P0 #3) by *actually running* `pytest`.
- **Empirical state (env `pytorch3d`, 2026-06-30):**
  - `pytest tests/ -m "not slow"` → **70 passed**, but `test_triangulation_consistency.py` (12) errors and `test_animation_export.py` (5) errors on collection.
  - `PYTHONPATH=smal_fitter pytest tests/ -m "not slow"` → triangulation passes (76 total), **but `test_fitter_3d_optimise` then FAILS** (see C10).
  - bare `pytest` from repo root also collects `smal_fitter/sleap_data/test_sleap_preprocessing.py` (see C12), which errors.
- **Root cause:** no `tests/conftest.py` and no `testpaths` in `pytest.ini`, so each test module relies on its own ad-hoc `sys.path` hacks, which conflict.
- **Fix (proposed):** add a `conftest.py` that puts the right dirs on `sys.path` once, add `testpaths = tests` to `pytest.ini`, and resolve C9/C10/C11/C12 so `pytest tests/` just works.

### C9 — `smal_fitter` is ambiguous (dir vs module); no `smal_fitter/__init__.py` 🛠 FIXED  · [#37](https://github.com/FabianPlum/SMILify/issues/37)
- **Symptom:** `from smal_fitter import SMALFitter` ([smal_fitter/neuralSMIL/smil_image_regressor.py:24](smal_fitter/neuralSMIL/smil_image_regressor.py#L24)) only resolves when `smal_fitter/` is on `sys.path` (so `smal_fitter` → `smal_fitter.py`). Under pytest, repo root is on the path first, so `smal_fitter` resolves to the **directory** (namespace package) and the import fails: `cannot import name 'SMALFitter' from 'smal_fitter'`.
- **Fix:** decide whether `smal_fitter` is a package (add `__init__.py`, make imports `from smal_fitter.smal_fitter import SMALFitter`) or keep it a sys.path dir — and make it consistent. Affects all of neuralSMIL.

### C10 — `utils` module name collision (`smal_fitter/utils.py` vs `fitter_3d/utils.py`) 🛠 FIXED  · [#38](https://github.com/FabianPlum/SMILify/issues/38)
- **Symptom:** with `smal_fitter/` on `PYTHONPATH`, `from utils import perspective_proj_withz` ([smal_fitter/p3d_renderer.py:17](smal_fitter/p3d_renderer.py#L17)) resolves to `fitter_3d/utils.py` (wrong module, no such symbol) → `test_fitter_3d_optimise` fails. Without it, the triangulation tests fail instead.
- **Fix:** disambiguate the two `utils` modules (package-qualified imports, or rename one).

### C11 — `tests/test_animation_export.py` uses an import style that can't work 🛠 FIXED  · [#39](https://github.com/FabianPlum/SMILify/issues/39)
- **Symptom:** `from smal_fitter.neuralSMIL.animation_export import ...` requires `smal_fitter` (and `neuralSMIL`) to be importable **packages**; they have no `__init__.py`, so this errors (`'smal_fitter' is not a package`) under every invocation tried. Inconsistent with the rest of the suite (which imports via `sys.path`).
- **Fix:** rewrite the import to match the suite's convention (depends on C9 decision).

### C12 — stray test file collected from repo root 🛠 FIXED  · [#40](https://github.com/FabianPlum/SMILify/issues/40)
- **Symptom:** `smal_fitter/sleap_data/test_sleap_preprocessing.py` is picked up by bare `pytest` (no `testpaths`) and errors: `from preprocess_sleap_dataset import ...` (cwd-relative).
- **Fix:** add `testpaths = tests` to `pytest.ini` (and/or move/repair this file).

### C13 — Inference uses a strict state_dict load; benchmark is tolerant ✅ (partially addressed)  · [#41](https://github.com/FabianPlum/SMILify/issues/41)
- **UPDATE (2026-07-01):** `run_multiview_inference.py` now loads with `strict=False` and re-derives `rotation_representation` from the checkpoint config ([run_multiview_inference.py:350](smal_fitter/neuralSMIL/run_multiview_inference.py#L350)), so **current-code checkpoints load fine**. Remaining gap: pre-6D-rotation (deprecated 9-dim cam-rot) checkpoints still hit a size mismatch — `strict=False` does **not** silence shape mismatches on matched keys. Low urgency (deprecated arch); #41 kept open for the tolerant-load / clear-error work.
- **Found:** verifying the Getting Started inference command (2026-07-01).
- **Symptom:** `run_multiview_inference.py` → `load_multiview_model_from_checkpoint` (~L430) does a **strict** `load_state_dict`, so it raises `RuntimeError: size mismatch` when the checkpoint's head dims differ from the reconstructed model — e.g. an older ViT stick checkpoint with a 9-dim camera-rotation head (`transformer_head.cam_rot_head` [9], `init_cam_rot` [1,9]) vs the current 6D-rotation model ([6] / [1,6]); `param_norm` / `token_embedding` differ by 3 accordingly. `benchmark_model.py` loads the **same** checkpoint fine (it re-infers dims from tensor shapes).
- **Note:** observed with an OLD substitute checkpoint (pre-6D-rotation refactor, `ViT_Large_Full_Stick_3D_post_decoder_fix`); a current-code checkpoint likely loads. But the asymmetry (benchmark tolerant, inference strict) is real and will bite anyone loading a slightly-older checkpoint for inference.
- **Fix (proposed):** make the inference loader as tolerant as benchmark's (re-infer head dims / handle 6D-vs-9D cam-rot), or at minimum emit a clear error pointing at the rotation-representation mismatch instead of a raw size-mismatch dump.

### C14 — CUDA initialized before `CUDA_VISIBLE_DEVICES` is set → breaks under torch 2.3.1 🛠 FIXED
- **Found:** verifying `environment.yml` in a fresh **torch 2.3.1** env (2026-07-01). `tests/test_pipeline.py::test_smal_fitter_optimize_to_joints` is the *only* test that regresses vs the local torch-2.1.1 env (69/70 pass); the env itself is valid (torch/CUDA/pytorch3d all work).
- **Symptom:** `optimize_to_joints.py` fails at first CUDA use — `RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED ... device=, num_gpus=` (wrapped in `torch.cuda.DeferredCudaCallError`). Reproduces standalone, not just under pytest.
- **Root cause (confirmed by minimal repro):** CUDA is touched (a `torch.cuda.*` call or a torch/torchvision import that inits CUDA) **before** `os.environ["CUDA_VISIBLE_DEVICES"]` is set; torch 2.3.1 then asserts on the stale device count after CVD changes, whereas torch 2.1.1 tolerated it. Minimal repro under 2.3.1: `import torch; torch.cuda.device_count(); os.environ['CUDA_VISIBLE_DEVICES']='0'; torch.zeros(1).cuda()` → same error (works if CVD is set *before* the first CUDA touch).
- **Affected (CVD set AFTER `import torch`):** [smal_fitter/optimize_to_joints.py:66-67](smal_fitter/optimize_to_joints.py#L66-L67), [smal_fitter/generate_video.py:39](smal_fitter/generate_video.py#L39), [smal_fitter/Unreal2Pytorch3D.py:2032](smal_fitter/Unreal2Pytorch3D.py#L2032). The neural path (train_multiview / benchmark / run_multiview_inference) does **not** set CVD, so it is unaffected.
- **Fix:** set `CUDA_DEVICE_ORDER` / `CUDA_VISIBLE_DEVICES` at the very TOP of each entrypoint — before any `import torch` / `import torchvision` (or any module that transitively imports them).
- **DONE (2026-07-01):** moved the CVD setup above the torch import in `optimize_to_joints.py` and `generate_video.py`, and into a guarded `if __name__ == "__main__"` block (before torch) in `Unreal2Pytorch3D.py` so importers (e.g. `dataset_preprocessing.py`) are unaffected. Verified: the full fast suite passes **70/70 in BOTH** torch 2.1.1 (`pytorch3d`) and torch 2.3.1 (`smilify_envtest`); `test_smal_fitter_optimize_to_joints` now passes on 2.3.1 (was the one failure).

### C15 — benchmark_model.py never sets CUDA_VISIBLE_DEVICES → fails under torch 2.3.1 🛠 FIXED
- **Found:** running the Getting Started benchmark on the matched checkpoint in a fresh torch 2.3.1 env (2026-07-01). Benchmark WORKS on torch 2.1.1 (PCK@5px 0.872, MPJPE 0.94 mm) but FAILED on 2.3.1 at `torch.load(map_location='cuda')` ([benchmark_model.py:733](smal_fitter/neuralSMIL/benchmark_model.py#L733)) with the C14 error (`device >= 0 && device < num_gpus`, num_gpus empty).
- **Root cause:** unlike the train/inference entrypoints, `benchmark_model.py` never set `CUDA_VISIBLE_DEVICES` at all; under torch 2.3.1 the CUDA init during `torch.load` then asserts. Confirmed by making it pass with `CUDA_VISIBLE_DEVICES=0` set externally.
- **FIXED (2026-07-01):** set `CUDA_DEVICE_ORDER` / `CUDA_VISIBLE_DEVICES` (via `os.environ.setdefault`, so an explicit external value still wins with `--device`) at the top of benchmark_model.py, before `import torch`. Verified benchmark now passes on BOTH torch 2.1.1 and 2.3.1 (PCK@5px 0.872, MPJPE 0.94 mm in both).
- **NOTE — C14's grep was incomplete:** it scoped to `smal_fitter/*.py` and missed `smal_fitter/neuralSMIL/`. Those entrypoints DO handle CVD but LATE (in `main()`, after `import torch`): `train_multiview_regressor.py:2227` (only CUDA_DEVICE_ORDER), `train_smil_regressor.py:1310-1311`, `test_smil_regressor_ground_truth.py:1159-1160`, `main.py:103-104`; `run_multiview_inference.py` only *prints* CVD, never sets it (why inference works on 2.3.1). They passed in testing, but the late-set pattern is fragile on torch ≥ 2.3 — normalize (set CVD before torch) in the code pass.
- **Also pre-existing (not from this branch):** importing `Unreal2Pytorch3D` (and its chain) sets `CUDA_VISIBLE_DEVICES` = `config.GPU_IDS` at **import time** — verified the same on `master`. Any importer (dataset_preprocessing, run_singleview_inference, train_smil_regressor, smil_datasets, …) inherits this side effect. Pin down the exact module and remove the import-time write when normalizing CVD handling.

### C16 — Unify import / package management across the repo 🛠 FIXED (umbrella)  · [#42](https://github.com/FabianPlum/SMILify/issues/42)
- **Symptom:** scripts depend on *where* they are run from and on ad-hoc `sys.path.append(...)` hacks, so imports and cross-module communication break depending on the working directory. There is no proper package: `smal_fitter/` (and `smal_fitter/neuralSMIL/`) have no `__init__.py`, so `smal_fitter` is ambiguous (the directory vs the `smal_fitter.py` module), `from smal_fitter import SMALFitter` only resolves from certain paths, and there are name collisions (`smal_fitter/utils.py` vs `fitter_3d/utils.py`). Most visible in the **test suite** (no single `pytest` invocation runs green) but it runs throughout — neuralSMIL scripts must be launched from specific directories, and the CVD import-order breakage in C14/C15 is downstream of the same fragility.
- **Umbrella / root cause for:** C8 (no `conftest.py` / `testpaths`), C9 (`smal_fitter` not a package), C10 (`utils` name collision), C11 (`test_animation_export` import style), C12 (stray collected test), and the import-order pieces of C14 / C15.
- **Fix direction:** make the repo a proper installable package — add `__init__.py` files + a `pyproject.toml`/setup so it can be `pip install -e .`; replace the ad-hoc `sys.path.append` hacks with package-qualified imports; resolve the `utils` and `smal_fitter` name collisions. Then a single `pytest` — and every entrypoint — works regardless of CWD. This is the foundational cleanup the other import-related items build on; do it before C17.

### C17 — Run `ruff` across the repo — SEPARATE PR, do LAST 📌 NEXT  · [#15](https://github.com/FabianPlum/SMILify/issues/15)
- **STATUS (2026-07-01):** the prerequisites are done — import/packaging (C16) landed, and CI is now in place (**PR #59**) to catch regressions from the ruff churn. This is the next task. Fold in **C21** (dead `scipy.misc` import, an F401) while here.
- Once the functional fixes above (especially **C16** import/packaging) are done, let `ruff` loose on the repo (lint + format + autofix) to clean up unused imports, dead code, style, etc.
- **Separate PR**, tracked by issue [#15](https://github.com/FabianPlum/SMILify/issues/15). Deliberately last, so `ruff` operates on the unified package structure rather than churning code that is about to be restructured by C16.

### C18 — `benchmark_model.py` reports PCK at only one resolution 🛠 FIXED  · [#46](https://github.com/FabianPlum/SMILify/issues/46)
- **FIXED (2026-07-01, branch `fix/benchmark-dual-resolution-pck`):** benchmark now reports PCK (@5 + full curve) at BOTH the model input resolution (224 ViT / 512 UNet) and the native resolution (override if given, else per-view `image_sizes`/`target_resolution`), in both single-view and multi-view paths. Shared valid-joint set; no redundant third report. Plots/arrays split per resolution (`pck_curve_{native,input}.png`, `error_histogram_{native,input}.png`, `errors_2d_px_{native,input}.npy`). Verified: multi-view end-to-end run (native PCK@5 0.8897 vs input-res 0.9903, same 5034-joint set); single-view dual-scale math unit-tested with a stub (no loadable SV checkpoint — all on-disk SV checkpoints use the deprecated 9-dim cam-rot arch, a C13-family issue).
- **Found:** running the neuralSMIL benchmark (2026-07-01).
- **Symptom:** `PCK@Npx` is resolution-dependent, but [benchmark_model.py](smal_fitter/neuralSMIL/benchmark_model.py) reports it at a **single** scale — `target_resolution`/`max(override_size)` for single-view ([_compute_pck_errors_singleview](smal_fitter/neuralSMIL/benchmark_model.py#L517-L536)), per-view native `image_sizes`/override for multi-view ([_get_original_image_size](smal_fitter/neuralSMIL/benchmark_model.py#L108-L121)). `input_resolution` (224 ViT / 512 UNet) is derived in both paths ([SV](smal_fitter/neuralSMIL/benchmark_model.py#L397), [MV](smal_fitter/neuralSMIL/benchmark_model.py#L932)) but only **logged**, never used for PCK. `PCK@5px` at 224 vs at a native 2048-frame are wildly different, so one number is ambiguous.
- **Fix:** report **two** PCKs (@5 + full curve, both paths, plots + `.npy`): (1) at the model **input resolution** (H=W=`input_resolution`), (2) at **native** resolution (`--orig_width`/`--orig_height` override if given, else dataset). If the override flag is passed, native = override — **no** redundant third PCK for the dataset's stored sizes. Exactly two reports.

### C19 — `smal_fitter/priors/shape_prior.py` imports nonexistent `global_utils` (dead module) 🛠 FIXED  · [#47](https://github.com/FabianPlum/SMILify/issues/47)
- **DONE (2026-07-01, PR #58):** deleted the dead module — imported by nothing, broken on import, and its shape-prior loading is already reimplemented inline in the live code (`smal_fitter/fitter.py`, `fitter_3d/trainer.py`). Issue closed.
- **Found:** import-health sweep during the import-structure refactor (2026-07-01).
- **Symptom:** [smal_fitter/priors/shape_prior.py](smal_fitter/priors/shape_prior.py) starts with `from global_utils import config`, but no `global_utils` module exists anywhere in the repo → `ModuleNotFoundError`.
- **Scope:** `shape_prior` is **imported by nothing** in the active tree (dead legacy module from upstream SMALify); **pre-existing** on `master`, not caused by the import refactor — so nothing currently breaks.
- **Fix:** delete the dead module, or (if the quadruped shape prior is still wanted) repair the import (`import config`) and wire it back in.

### C20 — `smal_fitter/outputs_imgs_to_video.py` runs file I/O at import time (no `__main__` guard) 🛠 FIXED  · [#48](https://github.com/FabianPlum/SMILify/issues/48)
- **DONE (2026-07-01, PR #58):** deleted — dead standalone scratch script (imported by nothing) that ran video-assembly at module top level against a hardcoded personal path. Issue closed.
- **Found:** import-health sweep during the import-structure refactor (2026-07-01).
- **Symptom:** [smal_fitter/outputs_imgs_to_video.py](smal_fitter/outputs_imgs_to_video.py) executes its video-assembly routine at **module top level** (no `if __name__ == "__main__":` guard) against a hardcoded path, so merely *importing* it runs that code and raises `FileNotFoundError`.
- **Scope:** **imported by nothing** (standalone legacy helper); **pre-existing** on `master`, not caused by the import refactor. Makes the file unsafe to import for reuse/tooling.
- **Fix:** move the top-level execution into an `if __name__ == "__main__":` block (and/or an `argparse` CLI) so importing only defines `create_video_from_images(...)`.

### C21 — `smal_fitter/fitter.py` imports removed `scipy.misc` (dead import, latent break) 🔍
- **Found:** surfaced as a `DeprecationWarning` in the first CI run (2026-07-01).
- **Symptom:** [smal_fitter/fitter.py:8](smal_fitter/fitter.py#L8) does `import scipy.misc`, which is deprecated and **removed in SciPy 2.0**. It is the module's **only** reference to `scipy.misc` (unused import), so it works today on SciPy 1.x but will hard-break with `ImportError` on a SciPy 2.0 bump.
- **Fix:** delete the unused `import scipy.misc`. Trivial — `ruff` (F401 unused-import) will catch it, so fold it into the **C17** pass rather than a standalone change. No GitHub issue yet.
