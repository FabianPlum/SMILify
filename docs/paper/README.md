# Paper

Planning notes for the SMILify methods paper are kept in a **private** repository:

**<https://github.com/FabianPlum/SMILify-paper>**

Access on request — contact the maintainer.

They live outside this repository because pre-submission planning is candid by nature
(competitive assessment, reviewer pre-mortem, per-benchmark contingencies, and open
questions about which claims the current data support). None of it is needed to use,
build on, or reproduce the code here.

## What is in this repository instead

Everything required to run the pipeline is public and documented:

- [Getting Started](../../GETTING_STARTED.md) — fresh clone to trained model
- [README](../../README.md) — full reference for preprocessing, training, inference, benchmarking
- [`smal_fitter/neuralSMIL/`](../../smal_fitter/neuralSMIL/) — neural inference
- [`fitter_3d/`](../../fitter_3d/) — mesh registration
- [`3D_model_prep/smil_importer/`](../../3D_model_prep/smil_importer/) — the Blender add-on for authoring models
- [`docs/design/`](../design/) — design documents for landed features

## A note on benchmark numbers

The `benchmark_*/`, `TEST_*/`, `SMOKE_*/`, `multiview_checkpoints/` and `checkpoints/`
directories that may appear in a working tree are **scratch output from development
runs on non-final datasets**. They are not results and should not be cited. Published
numbers will come from runs on the final datasets, reported in the paper.
