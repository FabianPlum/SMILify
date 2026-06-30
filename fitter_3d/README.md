3D SMAL Fitter
------

This directory contains code for optimising the SMAL/SMBLD model to provided 3D models.

It optimises shape, pose, scale to allow for the production of new shape & pose priors based on the set of animal models provided.

![alt text](/docs/ant_registered_meshes.gif)

*Animation showing the registration of 3D ant models to an example template mesh, demonstrating the fitting results across different species. This is done with our [provided blender addon](/3D_model_prep/SMIL_processing_addon.py) and can produce shape keys for either registered meshes directly or shape principle components. The meshes registered in this example were provided by [tunosemi](https://sketchfab.com/tunosemi)*


## Input file format

Currently, this only supports .obj files.

To avoid errors in initialisation, you may need to manually align your model so that it faces the same direction as the SMAL model. This alignment is with tail to head going in the positive x direction, and z denoting vertical height.

## Quickstart

Two options to run an optimisation:

- Single stage. Pass args into optimise.py to start a single scheme of optimisation, for example:

`python fitter_3d/optimise.py --mesh_dir fitter_3d/ATTA_BOI --scheme default --lr 1e-3 --nits 100`

(`--mesh_dir` defaults to `fitter_3d/ATTA_BOI`, the bundled example mesh, so it can be omitted for a first run.)

- For a more complicated (eg multi-stage) and fine tuned optimisation, add a custom .yaml file. See example_cfg.yaml for how it must be organised. This can then be called in optimise.py using:

`python fitter_3d/optimise.py --mesh_dir fitter_3d/ATTA_BOI --yaml_src fitter_3d/example_cfg.yaml`

Note: only the top-level `args:` block in the YAML (e.g. `results_dir`, `shape_family_id`) overrides the matching command-line flags. The per-stage settings (`scheme`, `nits`, `lr`, `loss_weights`) are taken entirely from the YAML's `stages:` list — when a YAML is supplied, the CLI `--scheme`/`--lr`/`--nits` are ignored.

## Schemes

The `--scheme` choices are the keys of `SMALParamGroup.param_map` in [trainer.py](trainer.py) — **ten** schemes, each selecting which parameters are optimised:

| Scheme | Optimised parameters |
|---|---|
| `init` | `global_rot`, `trans` |
| `init_rot_lock` | `trans`, `log_beta_scales` |
| `init_rot_lock_trans` | `trans`, `betas_trans` |
| `init_rot_lock_trans_scale` | `trans`, `betas_trans`, `log_beta_scales` |
| `default` | `global_rot`, `joint_rot`, `trans`, `betas`, `log_beta_scales` |
| `default_with_betas_trans` | `default` + `betas_trans` |
| `shape` | `global_rot`, `trans`, `betas`, `log_beta_scales`, `betas_trans` — i.e. `default` minus `joint_rot`, **plus** `betas_trans` |
| `pose` | `global_rot`, `trans`, `joint_rot`, `betas`, `log_beta_scales`, `betas_trans` — i.e. `default` **plus** `betas_trans` (note: still **includes** the shape params) |
| `deform` | `deform_verts` (per-vertex deformations only) |
| `all` | everything above + `deform_verts` |

## Output

Results are written to `--results_dir` (default `fit3d_results/`) as per-stage `.npz` files (`<stage>.npz`, plus per-batch `<stage>_batch_<i>.npz`). Use [read_out_fitter_stages.py](read_out_fitter_stages.py) to load and inspect them. SDF-based registration is available via `--use_sdf` / `--sdf_dir`.



