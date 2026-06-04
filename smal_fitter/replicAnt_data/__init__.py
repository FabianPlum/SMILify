"""replicAnt Data Integration Package

Tools for integrating replicAnt (procedural Unreal-Engine generated)
datasets into the SMILify pipeline. Sibling of `sleap_data/` for the
SLEAP-sourced multi-camera path.

Main components:
- visualize_multiview_depth_occlusion.py: per-view diagnostic plots for
  the depth-buffer self-occlusion visibility refinement applied by the
  multi-camera replicAnt loader.
- preprocess_replicant_multiview_dataset.py: converts a flat-directory
  replicAnt multi-cam dataset into the multiview HDF5 schema consumed by
  SLEAPMultiViewDataset and train_multiview_regressor.py. Bakes the
  multi-view scale unification (translation_factor=0.1) into the stored
  data and writes camera extrinsics in OpenCV form so the existing
  SLEAPMultiViewDataset conversion lands on the loader's PyTorch3D
  values without any class changes.
"""
