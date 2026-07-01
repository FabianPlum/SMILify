# SMILify Tests

`pytest` suite for the SMILify pipeline — the optimization fitters, the neural
single/multi-view path, the config system, augmentation, triangulation, and
animation export.

## Prerequisites

```bash
# 1. Activate the conda environment (see the root README "Installation")
conda activate pytorch3d
```

- **Run pytest from the repository root, scoped to `tests/`** (`pytest tests/`) so the `data/` relative paths resolve and pytest does not pick up the stray, currently-broken `smal_fitter/sleap_data/test_sleap_preprocessing.py`.
- The **slow** training test (`test_neural_smil_training_pipeline`) shells out to
  `train_smil_regressor.py` on the `test_textured` dataset, which must exist at
  `data/replicAnt_trials/replicAnt-x-SMIL-TEX/` (20 images + SMIL annotations).
- The fitter tests run the real optimization scripts, so they need PyTorch3D and a
  valid `config.SMAL_FILE` model (`test_fitter_3d_optimise` uses the bundled
  `fitter_3d/ATTA_BOI` mesh).

## Running tests

> **⚠️ Known issue — test-harness import paths are being repaired (not yet fixed).**
> The suite does **not** collect cleanly under a plain `pytest` run yet:
> `tests/test_animation_export.py` and `tests/test_triangulation_consistency.py` fail to import
> because `smal_fitter` is not a proper Python package (no `__init__.py`) and the test modules
> rely on conflicting `sys.path` hacks. There is currently **no single invocation that runs the
> whole suite green** — putting `smal_fitter/` on the path fixes the triangulation tests but
> breaks the `fitter_3d` subprocess test via a `utils` module-name collision. A fix
> (a `tests/conftest.py` + `testpaths = tests` in `pytest.ini`) is planned. Until then, use the
> commands below.

The only registered marker is `slow` (see `pytest.ini`); `test_neural_smil_training_pipeline`
is the single test carrying it.

```bash
# Fast tests, tolerating the known collection errors — ~70 pass today
pytest tests/ -m "not slow" --continue-on-collection-errors

# The modules that currently pass cleanly on their own
pytest tests/test_config_system.py tests/test_augmentation.py \
       tests/test_curriculum_sync.py tests/test_pipeline.py -m "not slow"

# The triangulation tests pass ONLY with smal_fitter on the path
# (this very requirement is what breaks test_fitter_3d_optimise — hence the planned fix)
PYTHONPATH=smal_fitter pytest tests/test_triangulation_consistency.py

# A single test function
pytest tests/test_pipeline.py::test_neural_smil_config_validation -v -s
```

> Use `-s` to see `print()` output from the integration tests (they are verbose).

## Test modules

| File | Tests | Covers |
|------|------:|--------|
| `test_pipeline.py` | 4 | Integration smoke tests: fitter_3d optimise, SMAL `optimize_to_joints`, neural config validation, and a slow end-to-end neural training run (details below). |
| `test_config_system.py` | 46 | The unified JSON/dataclass config system (`smal_fitter/neuralSMIL/configs/`): loading, dataclass merging, validation, legacy-dict conversion, curriculum application, CLI override precedence, round-trip serialization. |
| `test_augmentation.py` | 15 | Multi-view augmentation pipeline: photometric augmentations preserve camera params and 2D keypoints, geometric behavior, etc. |
| `test_triangulation_consistency.py` | 12 | The differentiable triangulation consistency loss: round-trip known 3D joints → project to 2D → `_triangulate_joints_dlt` → compare. |
| `test_animation_export.py` | 5 | Animation exporter round-trip: synthetic `predicted_params` → `AnimationRecorder` → reload the `.npz` + `.json`. |
| `test_curriculum_sync.py` | 1 | Verifies the loss curriculum from a JSON config is correctly synced into the legacy `TrainingConfig`. |

Two files in this directory are **not** pytest modules (no `test_*` functions) and are run directly:

- `config_test.py` — a fast-testing config-override helper (forces the OmniAnt `SMPL_fit.pkl` model); imported by manual test runs, not collected by pytest.
- `validate_multiview_replicant_loader.py` — a manual **visual** validation of multi-view replicAnt camera geometry + model fit; run it directly to produce diagnostic visualizations.

## `test_pipeline.py` in detail

| Test | Speed | What it does |
|------|-------|--------------|
| `test_fitter_3d_optimise` | medium | Runs `fitter_3d/optimise.py --mesh_dir fitter_3d/ATTA_BOI --scheme default --lr 1e-3 --nits 10` on the bundled Atta mesh. |
| `test_smal_fitter_optimize_to_joints` | medium | Runs the SMAL `optimize_to_joints` optimization pipeline. |
| `test_neural_smil_config_validation` | fast (<1 s, **pure logic, no filesystem**) | Asserts `get_data_path('test_textured')` resolves to `data/replicAnt_trials/replicAnt-x-SMIL-TEX`, the 85/5/10 train/val/test split, that the loss-curriculum weights are a dict containing `keypoint_2d`, and that the per-epoch learning rate is positive. |
| `test_neural_smil_training_pipeline` | **slow** (`@pytest.mark.slow`) | Shells out to `train_smil_regressor.py` for **2 epochs** on `test_textured` with `--batch_size 4 --checkpoint DISABLE_CHECKPOINT_LOADING --scale_trans_mode ignore`. Writes to a temporary directory and disables checkpoint saving, so it never overwrites trained models; artifacts are cleaned up afterwards. Asserts training completes. |

## CI

Once the import-path issue above is resolved, `pytest tests/ -m "not slow"` is the fast-feedback
signal on every change, and `pytest tests/ -m "slow"` exercises the full training subprocess on
scheduled builds (or before a release).
