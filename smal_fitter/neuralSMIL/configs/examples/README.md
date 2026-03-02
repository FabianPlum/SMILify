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
| `singleview_baseline.json` | singleview | Full single-view config with dataset path, ViT backbone, transformer decoder, and full loss/LR curriculum. |
| `multiview_baseline.json` | multiview | Multi-view config with cross-attention settings and multi-view output directories. |

## Top-level sections

### Shared (both modes)

- **`smal_model`** — Optional overrides for values otherwise sourced from `config.py`:
  - `smal_file`: path to the SMAL/SMIL model pickle (used by `config.py` to populate `dd`, `N_POSE`, `N_BETAS`, etc.). If set, training/inference should reload `config` so derived fields update.
  - `shape_family`: integer shape family passed into SMAL/SMIL code (overrides `config.SHAPE_FAMILY` for the run)
- **`dataset`** — `data_path`, `train_ratio`, `val_ratio`, `test_ratio`, `dataset_fraction`. For multiview, `data_path` is often set at runtime via CLI.
- **`model`** — `backbone_name`, `freeze_backbone`, `hidden_dim`, `head_type` (`mlp` or `transformer_decoder`), `use_unity_prior`, `rgb_only`, and transformer options (`transformer_depth`, `transformer_heads`, `transformer_dim_head`, `transformer_mlp_dim`, `transformer_dropout`, `transformer_ief_iters`, `transformer_trans_scale_factor`).
- **`optimizer`** — `learning_rate`, `weight_decay`, `gradient_clip_norm`, `optimizer_type`, and **`lr_schedule`**: a dict mapping epoch (as string) to learning rate, e.g. `"10": 3e-5`, `"100": 1e-5`.
- **`loss_curriculum`** — **`base_weights`**: dict of loss name → weight (e.g. `global_rot`, `joint_rot`, `keypoint_2d`, `keypoint_3d`, `silhouette`, regularizations). **`curriculum_stages`**: dict mapping epoch (as string) to partial weight updates applied from that epoch onward.
- **`training`** — `batch_size`, `num_epochs`, `seed`, `rotation_representation` (`6d` or `axis_angle`), `resume_checkpoint`, `num_workers`, `pin_memory`, `prefetch_factor`, `use_gt_camera_init`, `reset_ief_token_embedding`.
- **`output`** — `checkpoint_dir`, `plots_dir`, `visualizations_dir`, `train_visualizations_dir`, `save_checkpoint_every`, `generate_visualizations_every`, `plot_history_every`, `num_visualization_samples`.
- **`scale_trans_beta`** — `mode`: `ignore`, `separate`, or `entangled_with_betas`; optional nested options.
- **`mesh_scaling`** — Optional; `allow_mesh_scaling`, `init_mesh_scale`, `use_log_scale`.
- **`joint_importance`** — Per-joint importance weighting for keypoint losses: `enabled`, `important_joint_names` (list of joint name strings), `weight_multiplier`.
- **`ignored_joint_locations`** — Loss-level joint exclusion for 2D/3D keypoint supervision (joints stay in dataset but are not supervised): `enabled`, `ignored_joint_names`.
- **`ignored_joints`** — Data-preprocessing-level joint exclusion (joints removed from dataset entirely): `ignored_joint_names`, `verbose`.
- **`multi_dataset`** — Multi-dataset training: `enabled`, `datasets` (list of dataset entries with `name`, `path`, `type`, `weight`, `enabled`, `available_labels`), `validation_split_strategy` (`per_dataset` or `combined`).

### Multi-view only

In **`multiview_baseline.json`** (or any multiview config), you can set at the top level:

- **`num_views_to_use`** — Max views per sample (null = use all).
- **`min_views_per_sample`** — Minimum views required per sample.
- **`cross_attention_layers`**, **`cross_attention_heads`**, **`cross_attention_dropout`** — Cross-view attention in the multi-view regressor.
- **`output`** — Can override with multi-view dirs: `checkpoint_dir`, `visualizations_dir`, `singleview_visualizations_dir`, etc.

## Curriculum keys in JSON

JSON does not allow integer keys. Use **string** keys for epoch-based dicts; they are converted to integers on load:

- **`lr_schedule`**: `"0": 5e-5`, `"10": 3e-5`, …
- **`curriculum_stages`**: `"1": { "keypoint_2d": 0.1 }`, `"25": { ... }`, …

## Using the examples

```bash
# Single-view (set data_path in JSON or override)
python train_smil_regressor.py --config configs/examples/singleview_baseline.json

# Multi-view (often override dataset path)
python train_multiview_regressor.py --config configs/examples/multiview_baseline.json --dataset_path /path/to/multiview.h5
```

Copy an example to your project and edit; keep the `mode` field and the structure above so the loader and legacy bridge continue to work.
