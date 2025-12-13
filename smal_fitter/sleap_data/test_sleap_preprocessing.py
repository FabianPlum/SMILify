#!/usr/bin/env python3
"""
Test script for SLEAP dataset preprocessing and loading.

This script tests the complete SLEAP preprocessing pipeline:
1. Preprocess SLEAP sessions into HDF5 format
2. Load the preprocessed dataset
3. Validate compatibility with training pipeline
"""

import os
import sys
import tempfile
import argparse
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocess_sleap_dataset import SLEAPDatasetPreprocessor
from sleap_dataset import SLEAPDataset

# Import UnifiedSMILDataset with fallback
try:
    from smal_fitter.neuralSMIL.smil_datasets import UnifiedSMILDataset
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neuralSMIL'))
    from smil_datasets import UnifiedSMILDataset


def test_preprocessing(sessions_dir: str, output_path: str, 
                      joint_lookup_table: str = None, 
                      shape_betas_table: str = None,
                      num_workers: int = 1):
    """
    Test the SLEAP preprocessing pipeline.
    
    Args:
        sessions_dir: Directory containing SLEAP sessions
        output_path: Output HDF5 file path
        joint_lookup_table: Path to joint lookup table
        shape_betas_table: Path to shape betas table
        num_workers: Number of workers for preprocessing
    """
    print("="*60)
    print("TESTING SLEAP PREPROCESSING PIPELINE")
    print("="*60)
    
    # Create preprocessor
    preprocessor = SLEAPDatasetPreprocessor(
        joint_lookup_table_path=joint_lookup_table,
        shape_betas_table_path=shape_betas_table,
        target_resolution=224,
        backbone_name='vit_large_patch16_224',
        jpeg_quality=95,
        chunk_size=4
    )
    
    # Discover sessions
    sessions = preprocessor.discover_sleap_sessions(sessions_dir)
    print(f"Found {len(sessions)} SLEAP sessions:")
    for session in sessions:
        print(f"  - {Path(session).name}")
    
    if len(sessions) == 0:
        print("No sessions found, skipping preprocessing test")
        return False
    
    # Process dataset
    print(f"\nProcessing sessions...")
    try:
        stats = preprocessor.process_dataset(
            sessions_dir=sessions_dir,
            output_path=output_path,
            num_workers=num_workers,
            verbose=True
        )
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Statistics: {stats}")
        return True
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading(hdf5_path: str):
    """
    Test loading the preprocessed SLEAP dataset.
    
    Args:
        hdf5_path: Path to preprocessed HDF5 file
    """
    print("\n" + "="*60)
    print("TESTING SLEAP DATASET LOADING")
    print("="*60)
    
    try:
        # Load dataset directly
        print("Loading SLEAP dataset directly...")
        dataset = SLEAPDataset(hdf5_path)
        dataset.print_dataset_summary()
        
        # Test sample loading
        if len(dataset) > 0:
            print(f"\nTesting sample loading...")
            x_data, y_data = dataset[0]
            
            print(f"Sample 0:")
            print(f"  Image shape: {x_data['input_image_data'].shape}")
            print(f"  Image range: [{x_data['input_image_data'].min():.3f}, {x_data['input_image_data'].max():.3f}]")
            print(f"  Mask shape: {x_data['input_image_mask'].shape}")
            print(f"  Keypoints 2D shape: {y_data['keypoints_2d'].shape}")
            print(f"  Keypoint visibility shape: {y_data['keypoint_visibility'].shape}")
            print(f"  Visible keypoints: {y_data['visible_keypoints_count']}")
            print(f"  Has ground truth betas: {y_data['has_ground_truth_betas']}")
            print(f"  Session: {x_data['session_name']}")
            print(f"  Camera: {x_data['camera_name']}")
            print(f"  Frame: {x_data['frame_idx']}")
            
            # Test multiple samples
            print(f"\nTesting multiple samples...")
            for i in range(min(3, len(dataset))):
                x_data, y_data = dataset[i]
                print(f"  Sample {i}: {x_data['camera_name']}/{x_data['frame_idx']} - {y_data['visible_keypoints_count']} visible keypoints")
        
        return True
        
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_dataset_compatibility(hdf5_path: str):
    """
    Test compatibility with UnifiedSMILDataset.
    
    Args:
        hdf5_path: Path to preprocessed HDF5 file
    """
    print("\n" + "="*60)
    print("TESTING UNIFIED DATASET COMPATIBILITY")
    print("="*60)
    
    try:
        # Load via UnifiedSMILDataset
        print("Loading via UnifiedSMILDataset...")
        dataset = UnifiedSMILDataset.from_path(hdf5_path)
        
        print(f"Dataset type: {type(dataset).__name__}")
        print(f"Number of samples: {len(dataset)}")
        print(f"Input resolution: {dataset.get_input_resolution()}")
        print(f"Target resolution: {dataset.get_target_resolution()}")
        print(f"UE scaling flag: {dataset.get_ue_scaling_flag()}")
        
        # Test sample loading
        if len(dataset) > 0:
            print(f"\nTesting sample loading via UnifiedSMILDataset...")
            x_data, y_data = dataset[0]
            
            print(f"Sample 0:")
            print(f"  Image shape: {x_data['input_image_data'].shape}")
            print(f"  Keypoints 2D shape: {y_data['keypoints_2d'].shape}")
            print(f"  Keypoint visibility shape: {y_data['keypoint_visibility'].shape}")
            print(f"  Joint angles shape: {y_data['joint_angles'].shape}")
            print(f"  Shape betas shape: {y_data['shape_betas'].shape}")
            print(f"  Root rotation shape: {y_data['root_rot'].shape}")
            print(f"  Translation shape: {y_data['root_loc'].shape}")
            print(f"  Camera FOV: {y_data['cam_fov']}")
            print(f"  Camera rotation shape: {y_data['cam_rot'].shape}")
            print(f"  Camera translation shape: {y_data['cam_trans'].shape}")
        
        return True
        
    except Exception as e:
        print(f"Unified dataset compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SLEAP preprocessing pipeline")
    parser.add_argument("sessions_dir", help="Directory containing SLEAP sessions")
    parser.add_argument("--joint_lookup_table", help="Path to joint lookup table CSV")
    parser.add_argument("--shape_betas_table", help="Path to shape betas table CSV")
    parser.add_argument("--output_path", help="Output HDF5 file path (default: temp file)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--skip_preprocessing", action="store_true", 
                       help="Skip preprocessing and test existing HDF5 file")
    parser.add_argument("--hdf5_path", help="Path to existing HDF5 file for testing")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.skip_preprocessing:
        if not args.hdf5_path:
            print("Error: --hdf5_path required when skipping preprocessing")
            sys.exit(1)
        output_path = args.hdf5_path
    else:
        if args.output_path:
            output_path = args.output_path
        else:
            # Create temporary file
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "test_sleap_dataset.h5")
            print(f"Using temporary output file: {output_path}")
    
    success = True
    
    # Test preprocessing
    if not args.skip_preprocessing:
        success &= test_preprocessing(
            sessions_dir=args.sessions_dir,
            output_path=output_path,
            joint_lookup_table=args.joint_lookup_table,
            shape_betas_table=args.shape_betas_table,
            num_workers=args.num_workers
        )
    
    if success:
        # Test dataset loading
        success &= test_dataset_loading(output_path)
        
        # Test unified dataset compatibility
        success &= test_unified_dataset_compatibility(output_path)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    if success:
        print("✓ All tests passed!")
        print(f"✓ SLEAP dataset is ready for training: {output_path}")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
