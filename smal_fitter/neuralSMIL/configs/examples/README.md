# Example Configuration Files

This folder contains example JSON configs for single-view and multi-view training. Use them as templates or run them directly with `--config configs/examples/<name>.json`.

## Required field: `mode`

Every config **must** include at the top level:

- `"mode": "singleview"` — for `train_smil_regressor.py`
- `"mode": "multiview"` — for `train_multiview_regressor.py`

The loader uses `mode` to choose the config class and to validate the file.

## File overview

| File | Mode | Description |
|------|------|-------------|
| `getting_started.json` | multiview | **Start here** — the config used by the repo-root Getting Started guide: multi-view stick insect, `resume_checkpoint: null` so it trains from scratch cleanly. |
| `singleview_baseline.json` | singleview | Full single-view config with ViT backbone, transformer decoder, and loss/LR curriculum. |
| `multiview_baseline.json` | multiview | Multi-view config with cross-attention and multi-view output directories. |
| `multiview_sticks.json` | multiview | Stick insect with UNet EfficientNet-B3 backbone (the dataset `STICKS_full_512_3D.h5` is 512px — resolution comes from the dataset, not a config key). |
| `multiview_sticks_UNET.json` | multiview | Stick insect UNet variant. |
| `multiview_sticks_UNET_continue.json` | multiview | Stick insect UNet, continuation from a checkpoint. |
| `multiview_sticks_UNET_optimal.json` | multiview | Tuned stick-insect config, UNet EfficientNet-B5 backbone (mixed precision on). |
| `multiview_mouse_UNET_long.json` | multiview | 18-camera mouse with UNet EfficientNet-B3 and backbone chunking for VRAM (mixed precision is **off** in this file). |
| `multiview_replicant_mice.json` | multiview | Mouse config (ViT-Large) on replicAnt-generated multi-view data. |
| `multiview_SMILymice_3D_COMBINED_ViT_Large.json` | multiview | Combined SMILy mouse 3D dataset with a ViT-Large backbone. |
| `multiview_SMILySTICKS_3D_ViT_Large_AUG_FIXED.json` | multiview | The production recipe that trained the Getting Started example stick model (ViT-Large, augmentation on). `getting_started.json` is this config with `resume_checkpoint` cleared and isolated output dirs. |

## Using the examples

```bash
# Single-view (set data_path in JSON or override)
python train_smil_regressor.py --config configs/examples/singleview_baseline.json

# Multi-view (often override dataset path)
python train_multiview_regressor.py --config configs/examples/multiview_baseline.json --dataset_path /path/to/multiview.h5

# Multi-view with mixed precision (CLI override)
python train_multiview_regressor.py --config configs/examples/multiview_sticks.json --use_mixed_precision
```

Copy an example to your project and edit; keep the `mode` field so the loader and legacy bridge continue to work.

**Before running an example as-is:**
- Run from `smal_fitter/neuralSMIL/` so the `configs/examples/<name>.json` path resolves. The `data_path` / `smal_file` paths *inside* each config are relative to your working directory.
- The referenced datasets (`*.h5`) and SMAL model files (`*.pkl`) are **not shipped** in the repo — point them at your local copies (in the JSON, or via `--dataset_path` / config `smal_model.smal_file`).
- Several examples (including `singleview_baseline.json` and `multiview_baseline.json`) set `training.resume_checkpoint` to a specific path — running them unedited will try to resume from a checkpoint that likely doesn't exist on your machine. Clear it for a fresh run.

For a complete field-by-field reference of all config sections, see the parent [README.md](../README.md).
