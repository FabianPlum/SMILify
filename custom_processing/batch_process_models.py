import os
import subprocess
import multiprocessing
import argparse
from tqdm import tqdm

def find_stl_files(root_dir):
    """Find all STL files in the given directory and its subdirectories."""
    stl_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.stl'):
                stl_file = os.path.join(dirpath, filename)
                print(f"Found STL file: {stl_file}")
                stl_files.append(stl_file)
    print(f"Total STL files found: {len(stl_files)}")
    return stl_files

def process_stl(stl_path, output_dir, blender_path, script_path):
    """Process a single STL file using Blender."""
    cmd = [
        blender_path,
        "--background",
        "--python", script_path,
        "--",
        stl_path,
        output_dir
    ]
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {stl_path}:")
        print(f"Exit code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

def process_stl_wrapper(args):
    """Wrapper function to unpack arguments for process_stl."""
    return process_stl(*args)

def main(args):
    print(f"Searching for STL files in: {args.input_dir}")
    stl_files = find_stl_files(args.input_dir)
    if not stl_files:
        print("No STL files found. Exiting.")
        return

    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Preparing arguments for processing...")
    process_args = [
        (stl_file, args.output_dir, args.blender_path, args.script_path)
        for stl_file in stl_files
    ]

    print(f"Starting processing with {args.num_processes} processes...")
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        list(tqdm(pool.imap(process_stl_wrapper, process_args), total=len(stl_files)))

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process STL files using Blender")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing STL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed OBJ files")
    parser.add_argument("--blender_path", type=str, required=True, help="Path to Blender executable")
    parser.add_argument("--script_path", type=str, required=True, help="Path to Blender Python script")
    parser.add_argument("--num_processes", type=int, default=int(multiprocessing.cpu_count()/2), help="Number of parallel processes to use")

    args = parser.parse_args()
    main(args)

# example command
# python batch_process_models.py --input_dir /home/fabi/dev/SMILify/custom_processing/antscan_data/ --output_dir /home/fabi/dev/SMILify/custom_processing/antscan_processed --blender_path /home/fabi/blender/blender --script_path /home/fabi/dev/SMILify/custom_processing/prepare_antscan_data_for_mesh_fitting.py 
