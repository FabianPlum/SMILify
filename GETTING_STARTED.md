# Getting Started with SMILify

This guide takes you from a fresh clone to your first 3D reconstruction in four steps:
**install → get the example data → benchmark a trained model → train your own from scratch.**
It uses a small **multi-view stick-insect** example so the downloads stay light.

The rest of the [README](README.md) is the full reference — this is the fast path in.

> **You'll need:** an NVIDIA GPU with a CUDA 11.8-compatible driver, and `conda`
> (miniforge / miniconda / anaconda).

---

## 1. Install

```bash
# Clone (submodules are only needed for the legacy quadruped datasets — fine to skip for this guide)
git clone --recurse-submodules https://github.com/FabianPlum/SMILify
cd SMILify

# Create and activate the conda environment (named "pytorch3d", defined in environment.yml)
conda env create -f environment.yml
conda activate pytorch3d
```

On an HPC cluster, run [`hpc_files/install.sh`](hpc_files/install.sh) instead (supports `--skip-tests` and `ENV_NAME=...`).

**Verify it worked** with a quick, self-contained check:

```bash
pytest tests/test_pipeline.py::test_neural_smil_config_validation -v -s   # expect: 1 passed
```

> Don't run a bare `pytest` yet — a couple of test modules currently fail to import (a known, tracked issue). The single test above sidesteps that and confirms your install is sound. See [tests/README.md](tests/README.md).

More detail: [Installation](README.md#installation) · [environment.yml](environment.yml) · [tests/README.md](tests/README.md).

---

## 2. Download the example data + model

Two files (Google Drive), saved into the **repo root**:

```text
# ⬇️  Example dataset — preprocessed multi-view stick-insect HDF5
#     from:    https://drive.google.com/file/d/1wlVPe1ZwGmFkS9KhLODpIzvfi3DsqgQL/view?usp=drive_link
#     save as: SMILySTICKS_centred_reprojected_FIXED.h5

# ⬇️  Example checkpoint — a fully trained multi-view stick model (ViT-Large backbone)
#     from:    <GDRIVE_CHECKPOINT_LINK>         <!-- TODO: fill in -->
#     save as: SMILySTICKS_ViT_model.pth
```

The SMIL stick model — [`3D_model_prep/SMILy_STICK.pkl`](3D_model_prep/SMILy_STICK.pkl) — is already in the repo, so there's nothing else to download.

> The dataset is **already preprocessed** (the output of the SLEAP → HDF5 toolchain), so you can benchmark and train on it directly. To build your *own* dataset from raw footage later, see [Dataset preprocessing](README.md#dataset-preprocessing).

---

## 3. Benchmark the trained model

The example model is a **ViT-Large** multi-view stick model. Run from the repo root — note we pass `--smal-file` explicitly so you can see how a SMIL model is supplied to the run:

```bash
python smal_fitter/neuralSMIL/benchmark_model.py \
    --checkpoint SMILySTICKS_ViT_model.pth \
    --dataset_path SMILySTICKS_centred_reprojected_FIXED.h5 \
    --smal-file 3D_model_prep/SMILy_STICK.pkl
```

This evaluates the model on the dataset's held-out **test split** and writes a folder
`benchmark_multiview_SMILySTICKS_ViT_model_on_SMILySTICKS_centred_reprojected_FIXED/` containing:

| File | What |
|---|---|
| `benchmark_report.txt` | every metric, as text |
| `pck_curve.png` | PCK vs pixel-threshold curve |
| `error_histogram.png` | 2D keypoint-error distribution |
| `mpjpe_histogram.png` | 3D joint-error distribution (mm) |

On the console you'll see **PCK@5px** (2D accuracy) and — because this dataset carries 3D ground truth — **MPJPE in mm** (3D accuracy).

> **Notes:**
> - The script **auto-detects** single- vs multi-view from the checkpoint — there is no mode flag.
> - Watch the flag spelling: `--dataset_path` (underscore) but `--smal-file` (hyphen).
> - Always pass `--smal-file` — some checkpoints don't embed the model path and the run will abort without it.
> - Add `--max_batches 2` for a fast first-run sanity check.
>
> Full reference: [Benchmarking](README.md#benchmarking).

---

## 4. Train a model from scratch

The bundled [`getting_started.json`](smal_fitter/neuralSMIL/configs/examples/getting_started.json) config is set up for exactly this: multi-view with a **ViT-Large** backbone (the same architecture as the example checkpoint), pointed at the stick model, and — crucially — `"resume_checkpoint": null` so it genuinely starts fresh. It mirrors the production recipe in [`multiview_SMILySTICKS_3D_ViT_Large_AUG_FIXED.json`](smal_fitter/neuralSMIL/configs/examples/multiview_SMILySTICKS_3D_ViT_Large_AUG_FIXED.json), with resume cleared and isolated output dirs.

```bash
python smal_fitter/neuralSMIL/train_multiview_regressor.py \
    --config smal_fitter/neuralSMIL/configs/examples/getting_started.json \
    --dataset_path SMILySTICKS_centred_reprojected_FIXED.h5
```

What happens:

- Training runs single-GPU by default — add `--num_gpus N`, or launch with `torchrun --nproc_per_node=N` for multi-GPU.
- Checkpoints save every 5 epochs to `getting_started_checkpoints/` (`best_model.pth`, `checkpoint_epoch_*.pth`), next to a resolved `config.json` and validation visualizations.
- This is the **real ViT-Large recipe** used to train the example model (`batch_size 8`, augmentation on). If you hit out-of-memory, lower `training.batch_size` or `backbone_chunk_size` in the config.
- Stop anytime with `Ctrl-C`, then benchmark a saved checkpoint with the Step 3 command (point `--checkpoint` at `getting_started_checkpoints/best_model.pth`).

**To train on your own dataset:** pass `--dataset_path /path/to/your.h5`, or edit `dataset.data_path` in a copy of the config.

> ⚠️ **The "train from scratch" gotcha:** most of the *other* example configs hard-code a `training.resume_checkpoint` path — with those, a "fresh" run will either crash (single-view) or **silently resume** (multi-view). `getting_started.json` sets it to `null` so this just works. If you adapt a different config, clear `resume_checkpoint` first.
>
> Full reference: [Training](README.md#training) · [config fields](smal_fitter/neuralSMIL/configs/README.md) · [example configs](smal_fitter/neuralSMIL/configs/examples/README.md).

---

## Next steps

- **Render a video / run inference** from a checkpoint → [Inference](README.md#inference) (multi-view and single-view).
- **Use your own footage** → [Dataset preprocessing](README.md#dataset-preprocessing) (SLEAP → HDF5).
- **Build a new parametric model** for your own species/rig → [Adding new parametric models](README.md#adding-new-parametric-models) and [fitter_3d](fitter_3d/README.md).
- **Understand the codebase** → [Repository structure](README.md#repository-structure) and [Architecture: two fitting approaches](README.md#architecture-two-fitting-approaches).
- **Single-view** instead of multi-view → the same flow via `train_smil_regressor.py` / `run_singleview_inference.py`; see [smal_fitter/neuralSMIL/README.md](smal_fitter/neuralSMIL/README.md).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `conda activate pytorch3d` — env not found | The env is named **`pytorch3d`** (not `smilify`), regardless of the `environment.yml` filename. |
| Bare `pytest` shows import errors | Known and tracked — two modules fail to collect. Use `pytest tests/ -m "not slow" --continue-on-collection-errors`, or the single smoke test in Step 1. See [tests/README.md](tests/README.md). |
| Benchmark exits with a SMAL-file error | Pass `--smal-file 3D_model_prep/SMILy_STICK.pkl` — older checkpoints don't embed the model path. |
| `unrecognized arguments` on a flag | Flag spelling differs by script: benchmark uses `--dataset_path` / `--smal-file`; multi-view inference uses `--dataset` / `--smal_file`. |
| "Dataset path does not exist" when training | Pass `--dataset_path` to your `.h5`, or save the download as `SMILySTICKS_centred_reprojected_FIXED.h5` at the repo root. |
| Training resumes instead of starting fresh | A config's `training.resume_checkpoint` is set — use `getting_started.json`, or set it to `null`. |
