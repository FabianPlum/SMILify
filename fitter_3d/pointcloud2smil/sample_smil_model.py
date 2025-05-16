"""Sample SMIL model and generate random parameters to visualize the model."""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import argparse

# Add the parent directory to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config
from fitter_3d.trainer import SMAL3DFitter
from fitter_3d.utils import plot_meshes


def load_smil_model(batch_size=1, device='cuda'):
    """
    Load the SMIL model using SMAL3DFitter.
    
    Args:
        batch_size: Number of models to generate
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        SMAL3DFitter object
    """
    print(f"Loading SMIL model with batch size {batch_size} on device {device}")
    
    # Get shape family from config (usually -1 for custom models)
    shape_family = config.SHAPE_FAMILY
    
    # Initialize the SMAL3DFitter with the specified batch size and device
    smal_fitter = SMAL3DFitter(batch_size=batch_size, device=device, shape_family=shape_family)
    
    print(f"SMIL model loaded successfully with {smal_fitter.n_joints} joints and {smal_fitter.n_betas} shape parameters")
    
    return smal_fitter


def generate_random_parameters(smal_fitter, seed=None):
    """
    Generate random parameters for joint rotations and shape parameters.
    
    Args:
        smal_fitter: SMAL3DFitter object
        seed: Random seed for reproducibility
        
    Returns:
        Updated SMAL3DFitter object with random parameters
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    device = smal_fitter.device
    batch_size = smal_fitter.batch_size
    
    # Generate random shape parameters (betas)
    # Sample from a normal distribution around the mean betas with a variance of 1
    random_betas = smal_fitter.mean_betas.unsqueeze(0) + 2.0 * torch.randn(batch_size, smal_fitter.n_betas, device=device)
    smal_fitter.betas.data = random_betas
    
    # Generate random joint rotations (in axis-angle representation)
    # Keep rotation small to avoid extreme poses
    # Each joint has 3 rotation parameters (axis-angle)
    random_joint_rot = 0.3 * torch.randn(batch_size, config.N_POSE, 3, device=device)
    smal_fitter.joint_rot.data = random_joint_rot
    
    # Set global rotation to zero (as the root joint can rotate and this would make the rotation ambiguous)
    # alternatively set the root rotation to 0,0,0 and we use global rotation relative to the scene camera
    random_global_rot = torch.zeros(batch_size, 3, device=device)
    smal_fitter.global_rot.data = random_global_rot
    
    # Optional: Generate random translation
    random_trans = 0.01 * torch.randn(batch_size, 3, device=device)
    smal_fitter.trans.data = random_trans
    
    # Optional: Generate random log_beta_scales for joint scaling
    if config.ALLOW_LIMB_SCALING:
        # Generate random scales for all joints except root (index 0)
        random_scales = torch.zeros(batch_size, smal_fitter.n_joints, 3, device=device)
        random_scales[:, 1:] = 0.2 * torch.randn(batch_size, smal_fitter.n_joints - 1, 3, device=device)
        smal_fitter.log_beta_scales.data = random_scales

    return smal_fitter


def export_mesh_to_obj(verts, faces, filepath):
    """
    Export mesh vertices and faces to an OBJ file.
    
    Args:
        verts: Tensor of shape (V, 3) containing vertex positions
        faces: Tensor of shape (B, F, 3) containing face indices for batch size B
        filepath: Path where to save the OBJ file
    """
    # Convert tensors to numpy arrays
    verts_np = verts.cpu().numpy()
    # Take the first batch of faces since we're only exporting one mesh
    faces_np = faces[0].cpu().numpy()
    
    with open(filepath, 'w') as file:
        # Write vertices
        for vert in verts_np:
            file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        
        # Write faces (add 1 to indices since OBJ files are 1-indexed)
        for face in faces_np:
            # Ensure we have 3 vertices per face
            if len(face) == 3:
                file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
            else:
                print(f"Warning: Skipping face with {len(face)} vertices (expected 3)")
    
    print(f"Mesh exported to {filepath} with {len(verts_np)} vertices and {len(faces_np)} faces")


def sample_and_plot_model(smal_fitter, output_dir="sample_output", num_points=5000, export_obj=True):
    """
    Sample points from the model and create a visualization.
    
    Args:
        smal_fitter: SMAL3DFitter object with parameters set
        output_dir: Directory to save the output files
        num_points: Number of points to sample from the mesh surface
        export_obj: Whether to export the mesh as an OBJ file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Forward pass through the SMAL model to get vertices
    with torch.no_grad():
        verts = smal_fitter()
    
    # Get faces from the model
    faces = smal_fitter.faces
    
    # Export mesh as OBJ if requested
    if export_obj:
        obj_path = os.path.join(output_dir, "smil_mesh.obj")
        # Export the first mesh in the batch (since we're only generating one at a time)
        export_mesh_to_obj(verts[0], faces, obj_path)
    
    # Create a Meshes object for visualization
    meshes = Meshes(verts=verts, faces=faces).to(smal_fitter.device)
    
    # Sample points from the mesh surface
    point_clouds, normals = sample_points_from_meshes(
        meshes, 
        num_samples=num_points, 
        return_normals=True
    )
    
    # Create a dummy target mesh (same as source for visualization)
    target_meshes = meshes.clone()
    
    # Plot the meshes
    plot_meshes(
        target_meshes=target_meshes,
        src_meshes=meshes,
        mesh_names=["SMIL Model"],
        title="Random SMIL Model",
        figtitle="Generated SMIL Model with Random Parameters",
        out_dir=output_dir,
        plot_normals=False
    )
    
    # Plot the point cloud
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get points for visualization
    points = point_clouds[0].cpu().numpy()
    
    # Plot the point cloud
    ax.scatter(
        points[:, 0], 
        points[:, 1], 
        points[:, 2], 
        c='blue', 
        s=1,
        alpha=0.5
    )
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Sampled Point Cloud")
    
    # Save the figure
    pointcloud_path = os.path.join(output_dir, "pointcloud.png")
    plt.savefig(pointcloud_path)
    plt.close()
    
    # Save the parameters and point cloud
    save_parameters(smal_fitter, output_dir, point_clouds, normals)
    
    print(f"Model and point cloud plotted and saved to {output_dir}")


def save_parameters(smal_fitter, output_dir, point_clouds=None, normals=None, include_mesh=False):
    """
    Save the model parameters to a file.
    
    Args:
        smal_fitter: SMAL3DFitter object
        output_dir: Directory to save the parameters
        point_clouds: Optional sampled point clouds
        normals: Optional point normals
    """
    # Create a dictionary with all the parameters
    params = {
        'betas': smal_fitter.betas.cpu().detach().numpy(),
        'joint_rot': smal_fitter.joint_rot.cpu().detach().numpy(),
        'global_rot': smal_fitter.global_rot.cpu().detach().numpy(),
        'trans': smal_fitter.trans.cpu().detach().numpy(),
        'log_beta_scales': smal_fitter.log_beta_scales.cpu().detach().numpy()
    }
    
    if include_mesh:
        # Generate vertices
        with torch.no_grad():
            verts = smal_fitter().cpu().detach().numpy()
            faces = smal_fitter.faces.cpu().detach().numpy()
    
        # Add vertices and faces to parameters
        params['verts'] = verts
        params['faces'] = faces

        # Add normals if available
        if normals is not None:
            params['normals'] = normals[0].cpu().detach().numpy()
    
    # Add point cloud if available
    if point_clouds is not None:
        params['point_cloud'] = point_clouds[0].cpu().detach().numpy()
    
    # Save parameters
    params_file = os.path.join(output_dir, 'random_smil_params.npz')
    np.savez(params_file, **params)
    
    print(f"Parameters saved to {params_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sample SMIL model with random parameters')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (number of models)')
    parser.add_argument('--output-dir', type=str, default='sample_output', help='Output directory')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--num-points', type=int, default=5000, help='Number of points to sample')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if CUDA is available')
    parser.add_argument('--shape-family', type=int, default=None, 
                        help='Shape family ID (overrides config.SHAPE_FAMILY if provided)')
    parser.add_argument('--no-obj-export', action='store_true', help='Disable OBJ file export')
    return parser.parse_args()


def main():
    """Main function to run the model sampling pipeline."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Check if CUDA is available, otherwise use CPU
        device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
        print(f"Using device: {device}")
        
        # Override shape_family in config if provided
        if args.shape_family is not None:
            config.SHAPE_FAMILY = args.shape_family
            print(f"Using shape family: {config.SHAPE_FAMILY}")
        else:
            print(f"Using default shape family from config: {config.SHAPE_FAMILY}")
        
        # Load the SMIL model
        smal_fitter = load_smil_model(batch_size=args.batch_size, device=device)

        smal_fitter = generate_random_parameters(smal_fitter, seed=args.seed)
        
        # Sample points and plot the model
        sample_and_plot_model(smal_fitter, output_dir=args.output_dir, num_points=args.num_points, 
                            export_obj=not args.no_obj_export)
        
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()