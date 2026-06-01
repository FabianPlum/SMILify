"""replicAnt Data Integration Package

Tools for integrating replicAnt (procedural Unreal-Engine generated)
datasets into the SMILify pipeline. Sibling of `sleap_data/` for the
SLEAP-sourced multi-camera path.

Main components (current and planned):
- visualize_multiview_depth_occlusion.py: per-view diagnostic plots for
  the depth-buffer self-occlusion visibility refinement applied by the
  multi-camera replicAnt loader.
- preprocess_replicant_multiview_dataset.py (planned): converts a flat-
  directory replicAnt multi-cam dataset into the multiview HDF5 schema
  consumed by SLEAPMultiViewDataset and train_multiview_regressor.py.
"""
