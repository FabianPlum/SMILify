"""Shared utilities for multi-view dataset pipelines.

Houses code used across the SLEAP-side and replicAnt-side multi-view paths,
and by future heterogeneous-source tooling (the cross-source merger, the
SLEAP preprocessor's pending canonical-frame refactor, geometry validation).

Public API:

- canonical_frame.canonicalize_sample — column-vector OpenCV
  canonical-camera-frame transform, validated geometry-lossless on real
  SLEAP data (delta = 4e-6 px over a 6-cam mouse sample). The single source
  of truth for the convention; do NOT re-implement in callers.
"""
