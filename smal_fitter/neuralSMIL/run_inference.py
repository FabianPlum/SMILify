#!/usr/bin/env python3
"""
SMIL Image Regressor Inference Script

This script loads a trained SMILImageRegressor model from a checkpoint and runs inference
on all image files in a specified folder. It generates visualizations using the existing
SMIL visualization functions and saves results to an output folder.

Usage:
    python run_inference.py --checkpoint path/to/checkpoint.pth --input-folder path/to/images --output-folder path/to/output

Features:
    - Loads trained model from checkpoint
    - Processes all common image formats (jpg, jpeg, png, bmp, tiff)
    - Generates SMIL model visualizations
    - Saves predicted parameters and visualizations
    - Supports batch processing for efficiency
    - Handles different image sizes and formats
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import pickle as pkl

import torch
import torch.nn as nn
import numpy as np
import cv2
import imageio
from tqdm import tqdm

# Set matplotlib backend BEFORE any other imports to prevent tkinter issues
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Add the parent directories to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smil_image_regressor import SMILImageRegressor, rotation_6d_to_axis_angle
from training_config import TrainingConfig
from smal_fitter import SMALFitter
from Unreal2Pytorch3D import return_placeholder_data
import config

class InferenceImageExporter:
    """Enhanced image exporter for inference results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the image exporter.
        
        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export(self, collage_np: np.ndarray, batch_id: int, global_id: int, 
               img_parameters: Dict[str, Any], vertices: torch.Tensor, faces: np.ndarray, 
               img_idx: int = 0, image_name: str = "image"):
        """
        Export visualization image and parameters.
        
        Args:
            collage_np: Visualization collage as numpy array
            batch_id: Batch ID
            global_id: Global ID  
            img_parameters: Dictionary of SMIL parameters
            vertices: Model vertices
            faces: Model faces
            img_idx: Image index
            image_name: Base name for the image
        """
        # Save visualization image
        vis_filename = f"{image_name}_visualization.png"
        vis_path = os.path.join(self.output_dir, vis_filename)
        imageio.imsave(vis_path, collage_np)
        
        # Save parameters as JSON (for human readability)
        params_filename = f"{image_name}_parameters.json"
        params_path = os.path.join(self.output_dir, params_filename)
        
        # Convert numpy arrays and tensors to lists for JSON serialization
        json_parameters = {}
        for key, value in img_parameters.items():
            if isinstance(value, np.ndarray):
                json_parameters[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                json_parameters[key] = value.detach().cpu().numpy().tolist()
            elif hasattr(value, 'numpy'):  # Handle other tensor-like objects
                json_parameters[key] = value.numpy().tolist()
            else:
                json_parameters[key] = value
        
        with open(params_path, 'w') as f:
            json.dump(json_parameters, f, indent=2)
        
        # Save parameters as pickle (for exact reproduction)
        pkl_filename = f"{image_name}_parameters.pkl"
        pkl_path = os.path.join(self.output_dir, pkl_filename)
        with open(pkl_path, 'wb') as f:
            pkl.dump(img_parameters, f)
        
        print(f"Saved results for {image_name}:")
        print(f"  Visualization: {vis_path}")
        print(f"  Parameters (JSON): {params_path}")
        print(f"  Parameters (PKL): {pkl_path}")


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> Tuple[SMILImageRegressor, Dict[str, Any]]:
    """
    Load a trained SMILImageRegressor model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: PyTorch device ('cuda' or 'cpu')
        
    Returns:
        Tuple of (loaded_model, model_config)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully")
        
        # Use training configuration as base - this contains the correct model settings
        print("Using training configuration from training_config.py")
        training_config = TrainingConfig.get_all_config()
        model_config = training_config['model_config'].copy()
        rotation_representation = training_config['training_params']['rotation_representation']
        
        print(f"Configuration from training_config.py:")
        print(f"  backbone_name: {model_config['backbone_name']}")
        print(f"  head_type: {model_config['head_type']}")
        print(f"  rotation_representation: {rotation_representation}")
        
        # Verify this matches the checkpoint by checking state dict keys
        state_dict = checkpoint['model_state_dict']
        
        # Verify backbone type matches
        has_vit = any('vit' in key.lower() for key in state_dict.keys())
        has_resnet = any('resnet' in key.lower() or 'layer' in key for key in state_dict.keys())
        has_transformer_head = any('transformer_head' in key for key in state_dict.keys())
        
        if model_config['backbone_name'].startswith('vit') and not has_vit:
            print("WARNING: Config specifies ViT but checkpoint appears to contain ResNet")
        elif model_config['backbone_name'].startswith('resnet') and not has_resnet:
            print("WARNING: Config specifies ResNet but checkpoint appears to contain ViT")
        
        if model_config['head_type'] == 'transformer_decoder' and not has_transformer_head:
            print("WARNING: Config specifies transformer_decoder but checkpoint doesn't contain transformer_head")
        
        print(f"Checkpoint verification:")
        print(f"  Contains ViT keys: {has_vit}")
        print(f"  Contains ResNet keys: {has_resnet}")
        print(f"  Contains transformer_head: {has_transformer_head}")
        
        print(f"Model configuration:")
        for key, value in model_config.items():
            if key != 'transformer_config':
                print(f"  {key}: {value}")
        
        print(f"Using rotation representation: {rotation_representation}")
        
        # Detect batch size from checkpoint state dict
        state_dict = checkpoint['model_state_dict']
        checkpoint_batch_size = 1  # default
        
        # Try to detect batch size from various tensors
        for key in ['global_rotation', 'trans', 'log_beta_scales', 'betas_trans']:
            if key in state_dict and len(state_dict[key].shape) > 0:
                checkpoint_batch_size = state_dict[key].shape[0]
                break
        
        print(f"Detected checkpoint batch size: {checkpoint_batch_size}")
        
        # Use the training configuration as-is (no overrides needed)
        
        # Create placeholder data for model initialization with checkpoint batch size
        batch_size = checkpoint_batch_size
        placeholder_data = torch.zeros((batch_size, 3, 512, 512))
        
        # Determine input resolution based on backbone
        if model_config['backbone_name'].startswith('vit'):
            input_resolution = 224
        else:
            input_resolution = 512
        
        print(f"Creating model with input resolution: {input_resolution}")
        
        # Initialize model with detected configuration
        model = SMILImageRegressor(
            device=device,
            data_batch=placeholder_data,
            batch_size=batch_size,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=model_config.get('use_unity_prior', False),
            rgb_only=model_config.get('rgb_only', True),
            freeze_backbone=model_config.get('freeze_backbone', True),
            hidden_dim=model_config.get('hidden_dim', 1024),
            use_ue_scaling=True,  # Default for replicAnt data
            rotation_representation=rotation_representation,
            input_resolution=input_resolution,
            backbone_name=model_config['backbone_name'],
            head_type=model_config.get('head_type', 'mlp'),
            transformer_config=model_config.get('transformer_config', {})
        ).to(device)
        
        # Load model state, handling batch size differences
        # For inference, we need the neural network weights, but skip SMAL optimization parameters
        state_dict = checkpoint['model_state_dict']
        
        # Filter out SMAL optimization parameters that have batch size dependencies
        # These are specific to the SMALFitter optimization process, not the neural network
        smal_optimization_params = [
            'global_rotation', 'joint_rotations', 'trans', 'log_beta_scales', 
            'betas_trans', 'betas', 'fov', 'target_joints', 'target_visibility'
        ]
        
        # Keep all neural network parameters (backbone, transformer_head, fc layers, etc.)
        nn_state_dict = {}
        skipped_params = []
        
        for k, v in state_dict.items():
            # Skip SMAL optimization parameters that are specific to the optimization process
            if any(k == param or k.startswith(param + '.') for param in smal_optimization_params):
                skipped_params.append(k)
            else:
                nn_state_dict[k] = v
        
        print(f"Loading {len(nn_state_dict)} neural network parameters")
        print(f"Skipping {len(skipped_params)} SMAL optimization parameters: {skipped_params[:5]}{'...' if len(skipped_params) > 5 else ''}")
        
        # Load the neural network weights
        missing_keys, unexpected_keys = model.load_state_dict(nn_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (will use random initialization): {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
        model.eval()
        
        print("Model loaded and set to evaluation mode")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Head type: {model.head_type}")
        print(f"  Backbone: {model.backbone_name}")
        print(f"  Input resolution: {input_resolution}")
        
        return model, model_config
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def find_image_files(input_folder: str, supported_extensions: List[str] = None) -> List[str]:
    """
    Find all image files in the input folder.
    
    Args:
        input_folder: Path to folder containing images
        supported_extensions: List of supported file extensions
        
    Returns:
        List of image file paths
    """
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF']
    
    image_files = []
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_folder}")
    
    # Find all image files
    for ext in supported_extensions:
        pattern = f"*{ext}"
        image_files.extend(input_path.glob(pattern))
    
    # Convert to strings and sort
    image_files = [str(f) for f in image_files]
    image_files.sort()
    
    print(f"Found {len(image_files)} image files in {input_folder}")
    
    return image_files


def load_and_preprocess_image(image_path: str, model: SMILImageRegressor) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path: Path to the image file
        model: SMILImageRegressor model for preprocessing
        
    Returns:
        Tuple of (original_image_array, preprocessed_tensor)
    """
    try:
        # Load image
        image_data = imageio.v2.imread(image_path)
        
        # Keep original for visualization
        original_image = image_data.copy()
        
        # Preprocess for model
        preprocessed_tensor = model.preprocess_image(image_data)
        
        return original_image, preprocessed_tensor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load/preprocess image {image_path}: {e}")


def run_inference_on_image(model: SMILImageRegressor, image_tensor: torch.Tensor, 
                          device: str) -> Dict[str, torch.Tensor]:
    """
    Run inference on a preprocessed image tensor.
    
    Args:
        model: SMILImageRegressor model
        image_tensor: Preprocessed image tensor
        device: PyTorch device
        
    Returns:
        Dictionary of predicted SMIL parameters
    """
    try:
        with torch.no_grad():
            # Move to device
            image_tensor = image_tensor.to(device)
            
            # Run inference
            predicted_params = model(image_tensor)
            
            # Move results back to CPU for visualization
            cpu_params = {}
            for key, value in predicted_params.items():
                if isinstance(value, torch.Tensor):
                    cpu_params[key] = value.cpu()
                else:
                    cpu_params[key] = value
            
            return cpu_params
            
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")


def generate_visualization(model: SMILImageRegressor, predicted_params: Dict[str, torch.Tensor],
                          original_image: np.ndarray, image_exporter: InferenceImageExporter,
                          image_name: str, device: str) -> None:
    """
    Generate visualization using the SMIL model and predicted parameters.
    
    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary of predicted SMIL parameters
        original_image: Original input image
        image_exporter: Image exporter for saving results
        image_name: Base name for the image
        device: PyTorch device
    """
    try:
        # Convert rotations to axis-angle if they're in 6D representation
        if model.rotation_representation == '6d':
            global_rot_aa = rotation_6d_to_axis_angle(predicted_params['global_rot'])
            joint_rot_aa = rotation_6d_to_axis_angle(predicted_params['joint_rot'])
        else:
            global_rot_aa = predicted_params['global_rot']
            joint_rot_aa = predicted_params['joint_rot']
        
        # Create a simplified SMALFitter for visualization
        # Use the original image as RGB input
        if original_image.max() > 1.0:
            rgb_image = original_image.astype(np.float32) / 255.0
        else:
            rgb_image = original_image.astype(np.float32)
        
        # Resize to model's expected input size
        if model.backbone_name.startswith('vit'):
            target_size = (224, 224)
        else:
            target_size = (512, 512)
        
        if rgb_image.shape[:2] != target_size:
            rgb_image = cv2.resize(rgb_image, target_size)
        
        # Convert to tensor format expected by SMALFitter
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Create temporary SMALFitter for visualization
        temp_fitter = SMALFitter(
            device=device,
            data_batch=rgb_tensor,
            batch_size=1,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=False,
            rgb_only=True
        )
        
        # Set the predicted parameters (ensure they're on the right device)
        temp_fitter.global_rotation.data = global_rot_aa.to(device)
        temp_fitter.joint_rotations.data = joint_rot_aa.to(device)
        temp_fitter.betas.data = predicted_params['betas'].to(device)
        temp_fitter.trans.data = predicted_params['trans'].to(device)
        temp_fitter.fov.data = predicted_params['fov'].to(device)
        
        # Set joint scales and translations if available
        if 'log_beta_scales' in predicted_params:
            temp_fitter.log_beta_scales.data = predicted_params['log_beta_scales'].to(device)
        if 'betas_trans' in predicted_params:
            temp_fitter.betas_trans.data = predicted_params['betas_trans'].to(device)
        
        # Set camera parameters using predicted values
        if 'cam_rot' in predicted_params and 'cam_trans' in predicted_params:
            temp_fitter.renderer.set_camera_parameters(
                R=predicted_params['cam_rot'].to(device),
                T=predicted_params['cam_trans'].to(device),
                fov=predicted_params['fov'].to(device)
            )
        
        # Set dummy target joints and visibility for visualization
        temp_fitter.target_joints = torch.zeros((1, config.N_POSE, 2), device=device)
        temp_fitter.target_visibility = torch.ones((1, config.N_POSE), device=device)
        
        # Generate visualization with custom image exporter wrapper
        class NamedImageExporter:
            def __init__(self, base_exporter, image_name):
                self.base_exporter = base_exporter
                self.image_name = image_name
            
            def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces, img_idx=0):
                # Call the base exporter with the specific image name
                self.base_exporter.export(
                    collage_np, batch_id, global_id, img_parameters, vertices, faces, 
                    img_idx=img_idx, image_name=self.image_name
                )
        
        named_exporter = NamedImageExporter(image_exporter, image_name)
        temp_fitter.generate_visualization(named_exporter, apply_UE_transform=True, img_idx=0)
        
        print(f"Generated visualization for {image_name}")
        
    except Exception as e:
        print(f"Warning: Failed to generate visualization for {image_name}: {e}")
        # Save just the parameters without visualization
        img_parameters = {k: v.cpu().data.numpy() if isinstance(v, torch.Tensor) else v 
                         for k, v in predicted_params.items()}
        
        # Create a simple visualization showing the original image
        simple_vis = original_image.copy()
        if simple_vis.max() > 1.0:
            simple_vis = (simple_vis).astype(np.uint8)
        else:
            simple_vis = (simple_vis * 255).astype(np.uint8)
        
        image_exporter.export(
            simple_vis, 0, 0, img_parameters, 
            torch.zeros(1, 1000, 3), np.zeros((1000, 3), dtype=int), 
            img_idx=0, image_name=image_name
        )


def process_images_batch(model: SMILImageRegressor, image_files: List[str], 
                        output_folder: str, device: str, batch_size: int = 1) -> None:
    """
    Process a batch of images for inference.
    
    Args:
        model: SMILImageRegressor model
        image_files: List of image file paths
        output_folder: Output folder for results
        device: PyTorch device
        batch_size: Batch size for processing (currently only supports 1)
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Create image exporter
    image_exporter = InferenceImageExporter(output_folder)
    
    print(f"Processing {len(image_files)} images...")
    
    # Process images with progress bar
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Get image name for output files
            image_name = Path(image_path).stem
            
            print(f"\nProcessing image {i+1}/{len(image_files)}: {image_name}")
            
            # Load and preprocess image
            original_image, preprocessed_tensor = load_and_preprocess_image(image_path, model)
            
            # Run inference
            predicted_params = run_inference_on_image(model, preprocessed_tensor, device)
            
            # Generate visualization with unique image name
            generate_visualization(
                model, predicted_params, original_image, 
                image_exporter, image_name, device
            )
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"\nProcessing complete! Results saved to: {output_folder}")


def main():
    """Main function for the inference script."""
    parser = argparse.ArgumentParser(
        description='Run SMIL inference on a folder of images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --checkpoint checkpoints/best_model.pth --input-folder test_images --output-folder results
  python run_inference.py -c model.pth -i images/ -o output/ --batch-size 4
  python run_inference.py --checkpoint model.pth --input-folder images --output-folder results --device cpu

Supported image formats: jpg, jpeg, png, bmp, tiff, tif (case-insensitive)
        """
    )
    
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                       help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('-i', '--input-folder', type=str, required=True,
                       help='Path to folder containing input images')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                       help='Path to folder for saving results')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for processing (default: 1, currently only 1 is supported)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference (default: auto)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SMIL Image Regressor - Inference Script")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Batch size: {args.batch_size}")
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Device: {device}")
    
    try:
        # Load model from checkpoint
        print("\n" + "="*40)
        print("Loading model...")
        model, model_config = load_model_from_checkpoint(args.checkpoint, device)
        
        # Find image files
        print("\n" + "="*40)
        print("Finding images...")
        image_files = find_image_files(args.input_folder)
        
        if len(image_files) == 0:
            print("No image files found in the input folder!")
            return 1
        
        # Process images
        print("\n" + "="*40)
        print("Running inference...")
        process_images_batch(
            model, image_files, args.output_folder, 
            device, args.batch_size
        )
        
        print("\n" + "="*60)
        print("Inference completed successfully!")
        print(f"Results saved to: {args.output_folder}")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
