"""
SLEAP Data Integration Package

This package provides tools for integrating SLEAP pose estimation datasets
into the SMILify training pipeline.

Main components:
- SLEAPDatasetPreprocessor: Preprocesses SLEAP sessions into HDF5 format
- SLEAPDataset: PyTorch dataset class for loading preprocessed SLEAP data
- preprocess_sleap_dataset.py: CLI script for dataset preprocessing
- test_sleap_preprocessing.py: Test script for validation
"""

from .sleap_dataset import SLEAPDataset
from .preprocess_sleap_dataset import SLEAPDatasetPreprocessor

__all__ = ['SLEAPDataset', 'SLEAPDatasetPreprocessor']
