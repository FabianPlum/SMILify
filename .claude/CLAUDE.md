# SMILify

3D (animal) model fitting and neural inference framework, based on SMALify. Supports arbitrary rigged 3D models (during developement focused on insects and mice) with single-view and multi-view reconstruction.

## Environment

- Ubuntu 24.04 as well as Windows 11, Python via conda (miniconda3 installed via scoop)
- Conda env: `pytorch3d` (Python 3.10, CUDA 11.8). Verified torch versions differ by OS: Linux = PyTorch 2.1.1 / pytorch3d 0.7.7; Windows 11 dev = PyTorch 2.3.1. See `environment.yml`.
- Conda executable: `C:/Users/Fabian/scoop/apps/miniconda3/current/Scripts/conda.exe`
- Cannot run conda/python directly from bash shell - shell hook setup is needed
- GitHub CLI (`gh`) is installed

## Project Structure

```
config.py                          # Legacy root config (still required by fitter_3d, optimize_to_joints.py)
smal_fitter/smal_fitter.py         # Optimization-based fitting (SMALFitter nn.Module)
smal_fitter/neuralSMIL/            # Neural inference module
  train_smil_regressor.py          # Single-view training
  train_multiview_regressor.py     # Multi-view training
  training_config.py               # Old training config (being deprecated)
  configs/                         # New dataclass-based config system with JSON support
fitter_3d/                         # 3D model fitting (optimization-based)
smal_model/                        # SMAL/SMIL parametric model
3D_model_prep/                     # Blender files to create and edit parametric models, stored as .pkl files
custom_processing/                 # 3D model preprocessing, primarily for mesh registration and beta regressors
tests/                             # Pytest test suite
utilities/                         # legacy utilities, ignore.
```

## Architecture

Two fitting approaches share the same `smal_model` (SMAL/SMIL parametric model):

- **Optimization-based** (`smal_fitter/smal_fitter.py`): `SMALFitter` is an `nn.Module` that optimizes per-frame pose, shape (betas), translation, and FOV via gradient descent. Losses: 2D joint reprojection, silhouette, beta prior, pose prior, joint limits, splay. Supports both legacy quadruped priors (`use_unity_prior`) and arbitrary rigged models (`config.ignore_hardcoded_body`) with per-joint scaling.
- **Neural inference** (`smal_fitter/neuralSMIL/`): Learned regressors that predict SMIL parameters from images (single-view or multi-view).

Key globals from `config.py` used throughout: `N_BETAS`, `N_POSE`, `CANONICAL_MODEL_JOINTS`, `ignore_hardcoded_body`, `dd` (model dict), `DEBUG`.

## Config System (config-refactor branch)

- Precedence: CLI args > JSON file > mode-specific defaults > base defaults > legacy config.py
- JSON configs must include `"mode": "singleview"` or `"mode": "multiview"`
- Curriculum learning dicts use string keys in JSON, auto-converted to int on load

## How to Build / Test

- Install deps: `conda env create -f environment.yml && conda activate pytorch3d` (there is no requirements.txt; the conda env is authoritative)
- Run all tests: `pytest`
- Run fast tests only: `pytest -m "not slow"`

## Conventions

- Use dataclasses for new configuration
- Keep legacy `config.py` working for backward compatibility with fitter_3d
- Unify functionality and flag repeated code when applicable, as duplicate functions exist from earlier developement stages
