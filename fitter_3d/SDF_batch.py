import os
import torch
import pickle
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import argparse
from tqdm import tqdm
from pathlib import Path
import glob

# Import functions from SDF_tests
from SDF_tests import (
    compute_sdf,
    smooth_distances,
    visualize_sdf,
    try_mkdir,
    PerformanceMonitor
)

def process_mesh_folder(
    input_dir: str,
    output_dir: str,
    num_samples: int = -1,
    num_rays: int = 30,
    k_smoothing: int = 50
) -> None:
    """
    Process all mesh files in a folder, computing their Spatial Diameter Function.
    
    Args:
        input_dir (str): Directory containing .obj files
        output_dir (str): Directory to save results
        num_samples (int): Number of faces to sample. If -1, samples all faces.
        num_rays (int): Number of rays to cast per sampled point
        k_smoothing (int): Number of neighbors for smoothing
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    plots_dir = os.path.join(output_dir, "plots")
    data_dir = os.path.join(output_dir, "data")
    try_mkdir(output_dir)
    try_mkdir(plots_dir)
    try_mkdir(data_dir)
    
    # Find all .obj files
    mesh_files = glob.glob(os.path.join(input_dir, "**/*.obj"), recursive=True)
    if not mesh_files:
        print(f"No .obj files found in {input_dir}")
        return
    
    print(f"\nFound {len(mesh_files)} mesh files to process")
    
    # Process each mesh
    results = {}
    monitor = PerformanceMonitor()
    
    for mesh_file in tqdm(mesh_files, desc="Processing meshes"):
        mesh_name = Path(mesh_file).stem
        print(f"\nProcessing mesh: {mesh_name}")
        
        try:
            # Load mesh
            monitor.start('mesh_loading')
            verts, faces, _ = load_obj(mesh_file)
            faces = faces.verts_idx
            
            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            
            # Create mesh object
            mesh = Meshes(verts=[verts], faces=[faces])
            monitor.end('mesh_loading')
            
            # Compute SDF
            print(f"Computing SDF with {num_rays} rays per sample...")
            monitor.start('sdf_computation')
            sample_points, diameters = compute_sdf(mesh, num_samples=num_samples, num_rays=num_rays)
            monitor.end('sdf_computation')
            
            # Smooth the distances
            print("Smoothing distances...")
            monitor.start('smoothing')
            smoothed_diameters = smooth_distances(sample_points, diameters, k=k_smoothing)
            monitor.end('smoothing')
            
            # Save visualization
            plot_path = os.path.join(plots_dir, f"{mesh_name}_sdf.png")
            print("Generating visualization...")
            monitor.start('visualization')
            visualize_sdf(mesh, sample_points, smoothed_diameters, plot_path,
                         title=f"Spatial Diameter Function - {mesh_name}")
            monitor.end('visualization')
            
            # Store results
            results[mesh_name] = {
                'sample_points': sample_points.cpu(),
                'smoothed_diameters': smoothed_diameters.cpu(),
                'mesh_file': mesh_file,
                'num_vertices': len(verts),
                'num_faces': len(faces)
            }
            
            # Save individual results
            data_path = os.path.join(data_dir, f"{mesh_name}_sdf.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(results[mesh_name], f)
            
            print(f"Results saved to {data_path}")
            
        except Exception as e:
            print(f"Error processing {mesh_name}: {str(e)}")
            continue
    
    # Save combined results
    combined_data_path = os.path.join(output_dir, "combined_sdf_results.pkl")
    with open(combined_data_path, 'wb') as f:
        pickle.dump(results, f)
    
    print("\nProcessing complete!")
    print(f"Combined results saved to: {combined_data_path}")
    print(f"Individual results and plots saved in: {output_dir}")
    
    # Print performance report
    monitor.report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and visualize Spatial Diameter Function for multiple meshes"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing input OBJ files (will be searched recursively)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdf_batch_output",
        help="Directory to save the visualizations and data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of faces to sample. If -1, samples all faces (default: -1)"
    )
    parser.add_argument(
        "--num_rays",
        type=int,
        default=30,
        help="Number of rays to cast per sampled point"
    )
    parser.add_argument(
        "--k_smoothing",
        type=int,
        default=50,
        help="Number of neighbors for smoothing"
    )
    
    args = parser.parse_args()
    
    process_mesh_folder(
        args.input_dir,
        args.output_dir,
        num_samples=args.num_samples,
        num_rays=args.num_rays,
        k_smoothing=args.k_smoothing
    ) 