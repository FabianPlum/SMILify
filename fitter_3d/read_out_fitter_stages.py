import numpy as np
import argparse
import os
from pathlib import Path

def print_npz_contents(npz_path):
    """Print the contents of an NPZ file."""
    print(f"\nReading {npz_path}:")
    print("-" * 50)
    
    try:
        with np.load(npz_path) as data:
            # Print all available keys
            print("Available arrays:")
            for key in data.files:
                array = data[key]
                print(f"\nKey: {key}")
                print(f"Shape: {array.shape}")
                print(f"Type: {array.dtype}")
                
                # Print first few values if array is not too large
                if array.size < 10:
                    print("Values:", array)
                else:
                    print("First few values:", array.flatten()[:5])
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Read and display contents of NPZ files from SMAL optimization')
    parser.add_argument('--results_dir', type=str, default='fit3d_results',
                        help='Directory containing the optimization results')
    parser.add_argument('--stage_name', type=str, default=None,
                        help='Specific stage to analyze (optional)')
    
    args = parser.parse_args()
    
    # Get all NPZ files in the results directory
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{args.results_dir}' not found")
        return
    
    npz_files = list(results_path.glob('**/*.npz'))
    
    if not npz_files:
        print(f"No NPZ files found in {args.results_dir}")
        return
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Filter by stage name if provided
    if args.stage_name:
        npz_files = [f for f in npz_files if args.stage_name in str(f)]
        if not npz_files:
            print(f"No NPZ files found for stage '{args.stage_name}'")
            return
    
    # Process each NPZ file
    for npz_file in npz_files:
        print_npz_contents(npz_file)

if __name__ == "__main__":
    main()
