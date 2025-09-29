#!/usr/bin/env python3
"""
CLI Script for Dataset Preprocessing

This script provides a command-line interface for preprocessing SMIL datasets
from JSON format to optimized HDF5 format.

Usage:
    python preprocess_dataset.py input_dir output.h5 [options]

Example:
    python preprocess_dataset.py data/replicAnt_trials/replicAnt-x-SMIL-TEX optimized_dataset.h5 --silhouette_threshold 0.15 --backbone vit_large_patch16_224
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smil_datasets import UnifiedSMILDataset
from training_config import TrainingConfig


def validate_input_directory(input_dir: str) -> bool:
    """
    Validate that input directory exists and contains JSON files.
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return False
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input path is not a directory: {input_dir}")
        return False
    
    # Check for JSON files
    json_files = list(Path(input_dir).glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith('_BatchData')]
    
    if len(json_files) == 0:
        print(f"Error: No JSON files found in directory: {input_dir}")
        return False
    
    print(f"Found {len(json_files)} JSON files in input directory")
    return True


def validate_output_path(output_path: str) -> bool:
    """
    Validate output path and create directory if needed.
    
    Args:
        output_path: Path to output HDF5 file
        
    Returns:
        True if valid, False otherwise
    """
    # Check file extension
    if not (output_path.endswith('.h5') or output_path.endswith('.hdf5')):
        print(f"Warning: Output file should have .h5 or .hdf5 extension: {output_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error: Could not create output directory {output_dir}: {e}")
            return False
    
    # Check if output file already exists - always overwrite
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path} - will overwrite")
    
    return True


def estimate_processing_time(num_samples: int, num_workers: int) -> str:
    """
    Estimate processing time based on number of samples.
    
    Args:
        num_samples: Number of samples to process
        num_workers: Number of parallel workers
        
    Returns:
        Estimated time string
    """
    # Rough estimate: 0.5 seconds per sample per worker
    seconds_per_sample = 0.5
    estimated_seconds = (num_samples * seconds_per_sample) / num_workers
    
    if estimated_seconds < 60:
        return f"{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"{estimated_seconds/60:.1f} minutes"
    else:
        return f"{estimated_seconds/3600:.1f} hours"


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SMIL dataset from JSON to optimized HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing with default settings
  python preprocess_dataset.py data/my_dataset optimized.h5
  
  # Custom settings for ViT backbone with higher quality threshold
  python preprocess_dataset.py data/my_dataset optimized.h5 \\
    --silhouette_threshold 0.15 \\
    --backbone vit_large_patch16_224 \\
    --min_visible_keypoints 10
  
  # High-quality preprocessing with more workers
  python preprocess_dataset.py data/my_dataset optimized.h5 \\
    --jpeg_quality 98 \\
    --num_workers 8 \\
    --chunk_size 16
        """
    )
    
    # Required arguments
    parser.add_argument("input_dir", 
                       help="Input dataset directory containing JSON files")
    parser.add_argument("output_path", 
                       help="Output HDF5 file path (e.g., optimized_dataset.h5)")
    
    # Filtering options
    parser.add_argument("--silhouette_threshold", type=float, default=0.1,
                       help="Minimum silhouette coverage fraction (0.0-1.0, default: 0.1)")
    parser.add_argument("--min_visible_keypoints", type=int, default=5,
                       help="Minimum number of visible keypoints required (default: 5)")
    
    # Image processing options
    parser.add_argument("--target_resolution", type=int, default=224,
                       help="Target image resolution in pixels (default: 224)")
    parser.add_argument("--backbone", dest="backbone_name", default='vit_large_patch16_224',
                       choices=['vit_large_patch16_224', 'vit_base_patch16_224', 'resnet152'],
                       help="Backbone network name (default: vit_large_patch16_224)")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                       help="JPEG compression quality 1-100 (default: 95)")
    
    # Model options
    parser.add_argument("--rotation_representation", choices=['6d', 'axis_angle'], default='6d',
                       help="Rotation representation (default: 6d)")
    
    # Performance options
    parser.add_argument("--chunk_size", type=int, default=8,
                       help="HDF5 chunk size - should match training batch size (default: 8)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel processing workers (default: 4)")
    
    # Output options
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the output dataset after preprocessing")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.silhouette_threshold < 0.0 or args.silhouette_threshold > 1.0:
        print("Error: silhouette_threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        print("Error: jpeg_quality must be between 1 and 100")
        sys.exit(1)
    
    if args.chunk_size < 1:
        print("Error: chunk_size must be at least 1")
        sys.exit(1)
    
    if args.num_workers < 1:
        print("Error: num_workers must be at least 1")
        sys.exit(1)
    
    # Validate input and output paths
    if not validate_input_directory(args.input_dir):
        sys.exit(1)
    
    if not validate_output_path(args.output_path):
        sys.exit(1)
    
    # Count samples for time estimation
    from pathlib import Path
    json_files = list(Path(args.input_dir).glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith('_BatchData')]
    num_samples = len(json_files)
    
    # Print configuration summary
    if not args.quiet:
        print("\n" + "="*60)
        print("DATASET PREPROCESSING CONFIGURATION")
        print("="*60)
        print(f"Input directory: {args.input_dir}")
        print(f"Output file: {args.output_path}")
        print(f"Number of samples: {num_samples}")
        print(f"Silhouette threshold: {args.silhouette_threshold}")
        print(f"Min visible keypoints: {args.min_visible_keypoints}")
        print(f"Target resolution: {args.target_resolution}x{args.target_resolution}")
        print(f"Backbone: {args.backbone_name}")
        print(f"Rotation representation: {args.rotation_representation}")
        print(f"JPEG quality: {args.jpeg_quality}")
        print(f"Chunk size: {args.chunk_size}")
        print(f"Workers: {args.num_workers}")
        print(f"Estimated time: {estimate_processing_time(num_samples, args.num_workers)}")
        print("="*60)
        
        print("\nStarting preprocessing...")
    
    # Start preprocessing
    start_time = time.time()
    
    try:
        # Load ignored joints configuration from training config
        ignored_joints_config = TrainingConfig.IGNORED_JOINTS_CONFIG
        
        stats = UnifiedSMILDataset.preprocess_dataset(
            input_dir=args.input_dir,
            output_path=args.output_path,
            silhouette_threshold=args.silhouette_threshold,
            target_resolution=args.target_resolution,
            backbone_name=args.backbone_name,
            rotation_representation=args.rotation_representation,
            min_visible_keypoints=args.min_visible_keypoints,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            jpeg_quality=args.jpeg_quality,
            ignored_joints_config=ignored_joints_config,
            verbose=not args.quiet
        )
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not args.quiet:
            print(f"\nProcessing completed in {processing_time:.1f} seconds")
            print(f"Average time per sample: {processing_time/num_samples:.2f} seconds")
            
            # Calculate file size
            if os.path.exists(args.output_path):
                file_size = os.path.getsize(args.output_path) / (1024 * 1024)  # MB
                print(f"Output file size: {file_size:.1f} MB")
                if stats['final_samples'] > 0:
                    print(f"Size per sample: {file_size/stats['final_samples']:.2f} MB")
        
        # Validate output if requested
        if args.validate:
            print("\nValidating output dataset...")
            from optimized_dataset import HDF5DatasetValidator
            
            validator = HDF5DatasetValidator(args.output_path)
            validation_summary = validator.validate_dataset(num_samples=min(10, stats['final_samples']))
            
            print("Validation Summary:")
            for key, value in validation_summary.items():
                print(f"  {key}: {value}")
        
        print(f"\nPreprocessing successful! Output saved to: {args.output_path}")
        
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
