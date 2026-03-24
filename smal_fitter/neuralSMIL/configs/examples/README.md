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
| `singleview_baseline.json` | singleview | Full single-view config with ViT backbone, transformer decoder, and loss/LR curriculum. |
| `multiview_baseline.json` | multiview | Multi-view config with cross-attention and multi-view output directories. |
| `multiview_sticks.json` | multiview | Stick insect with UNet EfficientNet-B3 backbone, 512px resolution. |
| `multiview_sticks_UNET.json` | multiview | Stick insect UNet variant. |
| `multiview_mouse_UNET_long.json` | multiview | 18-camera mouse with UNet EfficientNet-B3, mixed precision, and backbone chunking for VRAM optimization. |

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

For a complete field-by-field reference of all config sections, see the parent [README.md](../README.md).
