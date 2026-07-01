# sc_venv_pytorch3d — HPC environment for SMILify (JURECA)

Forked from the Jülich Supercomputing Centre [`sc_venv_template`](https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template)
and adapted for SMILify: a self-contained Python venv for running the neural-inference pipeline on
JURECA-DC. Built on the `Stages/2024` stack (Python ≤ 3.11, required by the torch 2.3.1 + PyTorch3D
cu118 build); the venv lives in `./venv`.

> Migrating this environment to the newer `Stages/2026` stack is tracked in
> [#50](https://github.com/FabianPlum/SMILify/issues/50) (low priority; requires a JSC allocation to
> build/test, so not actionable by outside contributors).

Paths are derived relative to this folder — nothing here hardcodes a project allocation. `config.sh`
exports `ENV_NAME`, `ENV_DIR` (`./venv`), and `SMILIFY_DIR` (the repo root, two levels up), which the
batch scripts `cd` into before launching.

## One-time setup (login node)

```bash
bash hpc_files/sc_venv_pytorch3d/setup.sh
```

Loads the modules from `modules.sh`, creates `./venv`, and pip-installs `requirements.txt` then
`requirements_pt3d.txt` (PyTorch3D). The pip cache is redirected to `${SMILIFY_DIR}/.pip_cache` to
stay off the small home quota.

> Building PyTorch3D from source needs `nvcc`/`CUDA_HOME`. If setup can't find CUDA, add
> `module load CUDA` (the name matching the `Stages/2024` stack) to `modules.sh` and re-run.

Optional Jupyter kernel / VSCode interpreter: `bash create_kernel.sh` / `bash create_python_for_vscode.sh`
(then point VSCode at `./python`).

## Submitting jobs

Submit **from the repo root** so `${SLURM_SUBMIT_DIR}` resolves the activate script and the entrypoints
run as modules from the repo root:

```bash
sbatch hpc_files/sc_venv_pytorch3d/run_multiview_training_JURECA.sbatch     # multi-node DDP training (dc-gpu)
sbatch hpc_files/sc_venv_pytorch3d/run_multiview_inference_JURECA.sbatch    # multi-node DDP inference (dc-gpu)
sbatch hpc_files/sc_venv_pytorch3d/run_multiview_benchmark_JURECA.sbatch    # single-GPU benchmark (dc-gpu)
sbatch hpc_files/sc_venv_pytorch3d/run_multiview_preprocess_JURECA.sbatch   # CPU-only SLEAP preprocessing (dc-cpu)
```

Each script has a `# --- edit for your run ---` block for the dataset / checkpoint / config values and
declares `--account=ias-7` — change it to your allocation. Entrypoints launch with the module form
(`python -m smal_fitter.…`, via `torchrun_jsc -m …` for the distributed jobs), since SMILify uses
absolute-from-repo-root imports and is not pip-installable.

## Daily use (interactive)

```bash
source hpc_files/sc_venv_pytorch3d/activate.sh
```

Loads the modules, activates `./venv`, and prepends its site-packages to `PYTHONPATH`. Must be
**sourced**, not executed.
