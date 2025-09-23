#!/usr/bin/env python3
"""
Ground Truth Test for SMILImageRegressor

This script verifies that the SMILImageRegressor works correctly by:
1. Loading a sample from the dataset
2. Setting the predicted parameters to the ground truth values
3. Computing all loss components
4. Verifying that all losses are near zero (within tolerance)

This test ensures that:
- The loss computation is working correctly
- The rendering pipeline produces consistent results
- The parameter mapping is correct
- The optimization doesn't introduce errors
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add the parent directories to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smil_image_regressor import SMILImageRegressor, safe_to_tensor, axis_angle_to_rotation_6d, rotation_6d_to_axis_angle
from smil_datasets import replicAntSMILDataset
from smal_fitter import SMALFitter
from Unreal2Pytorch3D import load_SMIL_Unreal_sample
import config


def extract_target_parameters(y_data, device, rotation_representation='axis_angle'):
    """
    Extract target SMIL parameters from dataset y_data.
    (Copied from train_smil_regressor.py for consistency)
    
    Args:
        y_data: Dictionary containing SMIL data from dataset
        device: PyTorch device
        rotation_representation: '6d' or 'axis_angle' for joint rotations
    """
    targets = {}
    
    # Global rotation (root rotation) - format depends on dataset's rotation representation
    targets['global_rot'] = safe_to_tensor(y_data['root_rot'], device=device).unsqueeze(0)
    
    # Joint rotations (excluding root joint) - format depends on dataset's rotation representation
    joint_angles = safe_to_tensor(y_data['joint_angles'], device=device)
    targets['joint_rot'] = joint_angles[1:].unsqueeze(0)  # Exclude root joint
    
    # Shape parameters
    targets['betas'] = safe_to_tensor(y_data['shape_betas'], device=device).unsqueeze(0)
    
    # Translation (root location)
    targets['trans'] = safe_to_tensor(y_data['root_loc'], device=device).unsqueeze(0)
    
    # Camera FOV
    fov_value = y_data['cam_fov']
    if isinstance(fov_value, list):
        fov_value = fov_value[0]  # Take first element if it's a list
    targets['fov'] = torch.tensor([fov_value], dtype=torch.float32).to(device)
    
    # Camera rotation (in model space) - preserve as rotation matrix
    cam_rot_matrix = y_data['cam_rot']
    if hasattr(cam_rot_matrix, 'shape') and len(cam_rot_matrix.shape) == 2 and cam_rot_matrix.shape == (3, 3):
        targets['cam_rot'] = safe_to_tensor(cam_rot_matrix, device=device).unsqueeze(0)
    else:
        # Handle other formats
        if hasattr(cam_rot_matrix, 'shape') and cam_rot_matrix.shape == (3,):
            from scipy.spatial.transform import Rotation
            r = Rotation.from_rotvec(cam_rot_matrix)
            cam_rot_matrix = r.as_matrix()
            targets['cam_rot'] = safe_to_tensor(cam_rot_matrix, device=device).unsqueeze(0)
        else:
            targets['cam_rot'] = safe_to_tensor(cam_rot_matrix, device=device).unsqueeze(0)
    
    # Camera translation (in model space)
    targets['cam_trans'] = safe_to_tensor(y_data['cam_trans'], device=device).unsqueeze(0)
    
    # Joint scales and translations (if available)
    if y_data['scale_weights'] is not None and y_data['trans_weights'] is not None:
        # Import the PCA transformation function
        from Unreal2Pytorch3D import sample_pca_transforms_from_dirs
        
        # Compute actual scale and translation parameters from PCA weights
        translation_out, scale_out = sample_pca_transforms_from_dirs(
            config.dd, y_data['scale_weights'], y_data['trans_weights']
        )
        
        # Convert to log-space for scales and apply translation factor
        targets['log_beta_scales'] = torch.from_numpy(np.log(scale_out)).unsqueeze(0).float().to(device)
        targets['betas_trans'] = torch.from_numpy(translation_out * y_data['translation_factor']).unsqueeze(0).float().to(device)
    
    return targets


def create_ground_truth_predictions(target_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Create predicted parameters that exactly match the ground truth.
    """
    predictions = {}
    for key, value in target_params.items():
        predictions[key] = value.clone().detach()
    return predictions


def test_rotation_conversion_roundtrip(y_data: Dict[str, Any], device: str = 'cpu', tolerance: float = 1e-4):
    """
    Test that rotation conversion (axis-angle -> 6D -> axis-angle) preserves the same rotation.
    
    Note: The 6D representation is inherently ambiguous about the magnitude of the rotation.
    Multiple axis-angle representations can produce the same rotation matrix. This test
    verifies that the converted rotation is equivalent (not necessarily identical).
    
    Args:
        y_data: Dictionary containing SMIL data from dataset (in axis-angle format)
        device: PyTorch device
        tolerance: Maximum acceptable difference after round-trip conversion
        
    Returns:
        Dict with test results
    """
    print("\nTesting rotation conversion round-trip...")
    print("-" * 50)
    
    results = {}
    
    # Test global rotation conversion
    if 'root_rot' in y_data:
        original_global = safe_to_tensor(y_data['root_rot'], device=device)
        
        # Convert to 6D and back
        global_6d = axis_angle_to_rotation_6d(original_global)
        global_back = rotation_6d_to_axis_angle(global_6d)
        
        # Compute error
        global_error = torch.norm(original_global - global_back).item()
        global_passed = global_error < tolerance
        
        print(f"Global rotation round-trip error: {global_error:.8f} {'✓ PASS' if global_passed else '✗ FAIL'}")
        results['global_rot'] = {
            'error': global_error,
            'passed': global_passed,
            'original_shape': original_global.shape,
            '6d_shape': global_6d.shape
        }
    
    # Test joint rotations conversion
    if 'joint_angles' in y_data:
        original_joints = safe_to_tensor(y_data['joint_angles'], device=device)
        
        # Convert to 6D and back
        joints_6d = axis_angle_to_rotation_6d(original_joints)
        joints_back = rotation_6d_to_axis_angle(joints_6d)
        
        # Compute error - but handle the case where the error might be due to magnitude wrapping
        joints_error = torch.norm(original_joints - joints_back).item()
        
        # Check if the error is due to magnitude wrapping (error ≈ 2π)
        # If so, check if the rotations are actually equivalent by comparing rotation matrices
        if abs(joints_error - 2 * 3.14159) < 0.1:  # Error close to 2π
            print(f"Joint rotations round-trip error: {joints_error:.8f} (magnitude wrapping detected)")
            
            # Check if rotations are equivalent by comparing rotation matrices
            from pytorch3d.transforms import axis_angle_to_matrix
            
            original_matrices = axis_angle_to_matrix(original_joints)
            converted_matrices = axis_angle_to_matrix(joints_back)
            
            # Compute Frobenius norm of matrix differences
            matrix_errors = torch.norm(original_matrices - converted_matrices, p='fro', dim=(-2, -1))
            max_matrix_error = matrix_errors.max().item()
            
            if max_matrix_error < tolerance:
                print(f"  Matrix equivalence check: {max_matrix_error:.8f} ✓ PASS (rotations are equivalent)")
                joints_passed = True
                joints_error = max_matrix_error  # Use matrix error as the final error
            else:
                print(f"  Matrix equivalence check: {max_matrix_error:.8f} ✗ FAIL")
                joints_passed = False
        else:
            joints_passed = joints_error < tolerance
            print(f"Joint rotations round-trip error: {joints_error:.8f} {'✓ PASS' if joints_passed else '✗ FAIL'}")
        
        results['joint_rot'] = {
            'error': joints_error,
            'passed': joints_passed,
            'original_shape': original_joints.shape,
            '6d_shape': joints_6d.shape
        }
    
    # Overall result
    all_passed = all(result['passed'] for result in results.values())
    print(f"\nRotation conversion test: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    
    if not all_passed:
        print(f"Note: Round-trip errors should be < {tolerance} for equivalent rotations.")
        print("Large errors (≈2π) may indicate magnitude wrapping.")
    
    return {
        'success': all_passed,
        'components': results,
        'tolerance': tolerance
    }


def visualize_keypoints_comparison(model, predicted_params, y_data, x_data, backbone_name, dataset, output_dir="visualizations"):
    """
    Visualize ground truth vs rendered keypoints side by side.
    
    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary containing predicted SMIL parameters
        y_data: Ground truth data containing keypoints
        x_data: Input data containing the image
        output_dir: Directory to save visualization images
    """
    print("\nGenerating keypoint visualization...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the input image
    if x_data['input_image_data'] is not None:
        image = x_data['input_image_data'].copy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert from BGR to RGB for proper color display
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    height, width = image.shape[:2]
    
    # Get ground truth keypoints
    target_keypoints = y_data['keypoints_2d']  # Normalized [0, 1] format
    target_visibility = y_data['keypoint_visibility']
    
    # Get rendered keypoints using the optimized method
    rendered_joints, _ = model._compute_rendered_outputs(predicted_params, compute_joints=True, compute_silhouette=False)
    if rendered_joints is not None:
        rendered_keypoints = rendered_joints[0].cpu().numpy()  # First batch, convert to numpy
    else:
        print("ERROR: No rendered joints returned!")
        return
    
    # Create side-by-side visualization
    vis_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # Left side: Ground truth keypoints
    left_image = image.copy()
    for i, (kp, vis) in enumerate(zip(target_keypoints, target_visibility)):
        if vis > 0:  # Only draw visible keypoints
            # Convert normalized coordinates to pixel coordinates
            y_pixel = int(kp[0] * height)  # kp[0] is y_norm
            x_pixel = int(kp[1] * width)   # kp[1] is x_norm
            
            # Ensure coordinates are within image bounds
            if 0 <= x_pixel < width and 0 <= y_pixel < height:
                cv2.circle(left_image, (x_pixel, y_pixel), 3, (0, 255, 0), -1)  # Green for ground truth
                cv2.putText(left_image, str(i), (x_pixel + 5, y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # Right side: Rendered keypoints
    right_image = image.copy()
    for i, kp in enumerate(rendered_keypoints):
        # Convert normalized coordinates to pixel coordinates
        y_pixel = int(kp[0] * height)  # kp[0] is y_norm
        x_pixel = int(kp[1] * width)   # kp[1] is x_norm
        
        # Ensure coordinates are within image bounds
        if 0 <= x_pixel < width and 0 <= y_pixel < height:
            cv2.circle(right_image, (x_pixel, y_pixel), 3, (0, 0, 255), -1)  # Red for rendered
            cv2.putText(right_image, str(i), (x_pixel + 5, y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # Combine images
    vis_image[:, :width] = left_image
    vis_image[:, width:] = right_image
    
    # Add labels
    cv2.putText(vis_image, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_image, "Rendered", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save the visualization
    output_path = os.path.join(output_dir, "keypoints_comparison.png")
    cv2.imwrite(output_path, vis_image)
    print(f"Keypoint visualization saved to: {output_path}")
    
    # Print coordinate comparison for first few keypoints
    print(f"\nKeypoint coordinate comparison (normalized [0,1]):")
    print(f"{'Joint':<5} {'GT Y':<8} {'GT X':<8} {'Rend Y':<8} {'Rend X':<8} {'Diff Y':<8} {'Diff X':<8} {'Magnitude':<10}")
    print("-" * 75)
    for i in range(min(5, len(target_keypoints))):
        if target_visibility[i] > 0:
            gt_y, gt_x = target_keypoints[i]
            rend_y, rend_x = rendered_keypoints[i]
            diff_y = abs(gt_y - rend_y)
            diff_x = abs(gt_x - rend_x)
            magnitude = np.sqrt(diff_y**2 + diff_x**2)
            print(f"{i:<5} {gt_y:<8.4f} {gt_x:<8.4f} {rend_y:<8.4f} {rend_x:<8.4f} {diff_y:<8.4f} {diff_x:<8.4f} {magnitude:<10.4f}")


def visualize_silhouette_comparison(model, predicted_params, x_data, output_dir="visualizations"):
    """
    Visualize ground truth vs rendered silhouettes side by side.
    
    Args:
        model: SMILImageRegressor model
        predicted_params: Dictionary containing predicted SMIL parameters
        x_data: Input data containing the ground truth silhouette
        output_dir: Directory to save visualization images
    """
    print("\nGenerating silhouette visualization...")
    os.makedirs(output_dir, exist_ok=True)
    
    if x_data.get("input_image_mask") is None:
        print("No ground truth silhouette available for visualization")
        return
    
    # Get ground truth silhouette
    target_silhouette = x_data["input_image_mask"]
    if target_silhouette.max() > 1.0:
        target_silhouette = (target_silhouette / 255.0 * 255).astype(np.uint8)
    else:
        target_silhouette = (target_silhouette * 255).astype(np.uint8)
    
    height, width = target_silhouette.shape[:2]
    
    # Get rendered silhouette using the optimized method
    _, rendered_silhouette = model._compute_rendered_outputs(predicted_params, compute_joints=False, compute_silhouette=True)
    if rendered_silhouette is not None:
        rendered_sil_np = rendered_silhouette[0, 0].cpu().numpy()  # First batch, first channel
        rendered_sil_np = (rendered_sil_np * 255).astype(np.uint8)
        
        # Resize rendered silhouette to match ground truth size
        if rendered_sil_np.shape != target_silhouette.shape[:2]:
            rendered_sil_np = cv2.resize(rendered_sil_np, (width, height))
    else:
        print("ERROR: No rendered silhouette returned!")
        return
    
    # Create side-by-side visualization
    vis_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # Left side: Ground truth silhouette (in green)
    if len(target_silhouette.shape) == 2:
        left_image = cv2.cvtColor(target_silhouette, cv2.COLOR_GRAY2BGR)
    else:
        left_image = target_silhouette
    # Make it green-tinted
    left_image[:, :, 1] = np.maximum(left_image[:, :, 1], target_silhouette if len(target_silhouette.shape) == 2 else target_silhouette[:, :, 0])
    
    # Right side: Rendered silhouette (in red)
    right_image = cv2.cvtColor(rendered_sil_np, cv2.COLOR_GRAY2BGR)
    # Make it red-tinted
    right_image[:, :, 2] = np.maximum(right_image[:, :, 2], rendered_sil_np)
    
    # Combine images
    vis_image[:, :width] = left_image
    vis_image[:, width:] = right_image
    
    # Add labels
    cv2.putText(vis_image, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_image, "Rendered", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save the visualization
    output_path = os.path.join(output_dir, "silhouette_comparison.png")
    cv2.imwrite(output_path, vis_image)
    print(f"Silhouette visualization saved to: {output_path}")
    
    # Print statistics
    target_area = np.sum(target_silhouette > 127) / (height * width)
    rendered_area = np.sum(rendered_sil_np > 127) / (height * width)
    
    # Compute IoU
    target_binary = (target_silhouette > 127).astype(np.uint8)
    rendered_binary = (rendered_sil_np > 127).astype(np.uint8)
    intersection = np.sum(target_binary & rendered_binary)
    union = np.sum(target_binary | rendered_binary)
    iou = intersection / (union + 1e-8)
    
    print(f"\nSilhouette statistics:")
    print(f"Ground truth area: {target_area:.4f} ({target_area*100:.2f}% of image)")
    print(f"Rendered area: {rendered_area:.4f} ({rendered_area*100:.2f}% of image)")
    print(f"IoU (Intersection over Union): {iou:.4f}")
    print(f"Intersection: {intersection} pixels")
    print(f"Union: {union} pixels")


def test_ground_truth_loss(data_path: str = None, sample_idx: int = 0, 
                           tolerance_keypoints: float = 1e-2, 
                           tolerance_silhouette: float = 1e-2, 
                           device: str = 'cpu', 
                           enable_visualization: bool = True, 
                           override_ue_scaling: bool = None, 
                           rotation_representation: str = 'axis_angle', 
                           test_rotation_conversion: bool = True, 
                           backbone_name: str = 'resnet152'):
    """
    Test that ground truth parameters produce near-zero losses.
    
    Args:
        data_path: Path to the dataset (if None, tries default paths)
        sample_idx: Index of the sample to test
        tolerance: Maximum acceptable loss value
        device: Device to run on ('cpu' or 'cuda')
        enable_visualization: Whether to generate visual comparisons
        override_ue_scaling: Override dataset UE scaling setting (None uses dataset default)
        rotation_representation: '6d' or 'axis_angle' for joint rotations (default: 'axis_angle')
        test_rotation_conversion: Whether to test rotation conversion round-trip (default: True)
        backbone_name: Backbone name for keypoint scaling ('resnet152', 'vit_base_patch16_224', etc.)
    
    Returns:
        Dict with test results
    """
    print(f"Testing SMILImageRegressor Ground Truth Loss on {device}")
    print("=" * 60)
    
    # Try to find dataset to run testing, when no path is provided
    # Note that not all datasets have silhoutte masks, thus not all tests may be ran.
    if data_path is None:
        possible_paths = [
            "/media/fabi/Data/replicAnt-x-SMIL-OmniAnt-Masked",
            "data/replicAnt_trials/replicAnt-x-SMIL-TEX",
            "data/replicAnt_trials/replicAnt-x-SMIL-demo"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            print("ERROR: No dataset found. Please provide a valid data_path.")
            return {"success": False, "error": "Dataset not found"}
    
    print(f"Using dataset: {data_path}")
    
    try:
        # Load dataset with specified rotation representation
        print(f"Loading dataset with rotation representation: {rotation_representation}")
        dataset = replicAntSMILDataset(data_path, rotation_representation=rotation_representation, backbone_name=backbone_name)
        if len(dataset) == 0:
            print("ERROR: Dataset is empty")
            return {"success": False, "error": "Empty dataset"}
        
        # Use a sample that exists
        sample_idx = min(sample_idx, len(dataset) - 1)
        print(f"Loading sample {sample_idx} from {len(dataset)} available samples")
        
        # Load sample
        x_data, y_data = dataset[sample_idx]
        
        if x_data['input_image_data'] is None:
            print("ERROR: No image data in sample")
            return {"success": False, "error": "No image data"}
        
        print(f"Loaded sample with image shape: {x_data['input_image_data'].shape}")
        
        # Create placeholder data for model initialization
        batch_size = 1
        placeholder_data = torch.zeros((batch_size, 3, 512, 512))
        
        # Determine UE scaling setting
        use_ue_scaling = dataset.get_ue_scaling_flag() if override_ue_scaling is None else override_ue_scaling
        
        # Initialize model
        print("Initializing SMILImageRegressor...")
        print(f"Using UE scaling: {use_ue_scaling}")
        print(f"Using rotation representation: {rotation_representation}")
        model = SMILImageRegressor(
            device=device,
            data_batch=placeholder_data,
            batch_size=batch_size,
            shape_family=config.SHAPE_FAMILY,
            use_unity_prior=False,
            rgb_only=True,
            freeze_backbone=True,
            hidden_dim=512,
            use_ue_scaling=use_ue_scaling,
            rotation_representation=rotation_representation,
            input_resolution=224 if backbone_name.startswith('vit') else dataset.get_input_resolution()
        ).to(device)
        
        model.eval()  # Set to evaluation mode
        
        # Test rotation conversion round-trip if requested
        if test_rotation_conversion:
            # Load the same sample with axis-angle representation for conversion testing
            dataset_aa = replicAntSMILDataset(data_path, rotation_representation='axis_angle')
            x_data_aa, y_data_aa = dataset_aa[sample_idx]
            conversion_result = test_rotation_conversion_roundtrip(y_data_aa, device)
            
        # Extract target parameters
        target_params = extract_target_parameters(y_data, device, rotation_representation)
        print(f"Extracted target parameters: {list(target_params.keys())}")
        
        # Print shapes for debugging
        print(f"Target parameter shapes:")
        for key, value in target_params.items():
            print(f"  {key}: {value.shape}")
        
        # Create "predictions" that are identical to ground truth
        predicted_params = create_ground_truth_predictions(target_params)
        
        # Prepare additional data for loss computation
        keypoint_data = None
        if 'keypoints_2d' in y_data and 'keypoint_visibility' in y_data:
            keypoint_data = {
                'keypoints_2d': y_data['keypoints_2d'],
                'keypoint_visibility': y_data['keypoint_visibility']
            }
            print("Using keypoint data for loss computation")
        
        silhouette_data = None
        if x_data.get("input_image_mask") is not None:
            silhouette_data = x_data["input_image_mask"]
            print("Using silhouette data for loss computation")
        
        # Test different loss combinations
        test_results = {}
        
        print("\nTesting loss computations...")
        print("-" * 40)
        
        # Test 1: Parameter-only losses
        print("1. Testing parameter-only losses...")
        with torch.no_grad():
            total_loss, loss_components = model.compute_prediction_loss(
                predicted_params, target_params, return_components=True
            )
        
        param_losses = {k: v.item() for k, v in loss_components.items() 
                       if k not in ['keypoint_2d', 'silhouette']}
        
        print(f"   Total parameter loss: {total_loss.item():.8f}")
        for param_name, loss_val in param_losses.items():
            status = "✓ PASS" if loss_val < tolerance_keypoints else "✗ FAIL"
            print(f"   {param_name:15s}: {loss_val:.8f} {status}")
        
        test_results['parameter_only'] = {
            'total_loss': total_loss.item(),
            'components': param_losses,
            'passed': all(v < tolerance_keypoints for v in param_losses.values())
        }
        
        # Test 2: With keypoint loss (if available)
        if keypoint_data is not None:
            print("\n2. Testing with keypoint loss...")
            # Set loss weights to enable keypoint loss computation
            loss_weights = {
                'global_rot': 1.0,
                'joint_rot': 1.0,
                'betas': 1.0,
                'trans': 1.0,
                'fov': 1.0,
                'cam_rot': 1.0,
                'cam_trans': 1.0,
                'log_beta_scales': 1.0,
                'betas_trans': 1.0,
                'keypoint_2d': 1.0,  # Enable keypoint loss
                'silhouette': 0.0
            }
            with torch.no_grad():
                total_loss, loss_components = model.compute_prediction_loss(
                    predicted_params, target_params, pose_data=keypoint_data, return_components=True, loss_weights=loss_weights
                )
            
            print(f"   Total loss with keypoints: {total_loss.item():.8f}")
            if 'keypoint_2d' in loss_components:
                keypoint_loss = loss_components['keypoint_2d'].item()
                status = "✓ PASS" if keypoint_loss < tolerance_keypoints else "✗ FAIL"
                print(f"   keypoint_2d     : {keypoint_loss:.8f} {status}")
                
                test_results['with_keypoints'] = {
                    'total_loss': total_loss.item(),
                    'keypoint_loss': keypoint_loss,
                    'passed': keypoint_loss < tolerance_keypoints
                }
            else:
                print("   No keypoint loss computed")
                test_results['with_keypoints'] = {'passed': False, 'error': 'No keypoint loss'}
        
        # Test 3: With silhouette loss (if available)
        if silhouette_data is not None:
            print("\n3. Testing with silhouette loss...")
            # Set loss weights to enable silhouette loss computation
            loss_weights = {
                'global_rot': 1.0,
                'joint_rot': 1.0,
                'betas': 1.0,
                'trans': 1.0,
                'fov': 1.0,
                'cam_rot': 1.0,
                'cam_trans': 1.0,
                'log_beta_scales': 1.0,
                'betas_trans': 1.0,
                'keypoint_2d': 0.0,
                'silhouette': 1.0  # Enable silhouette loss
            }
            with torch.no_grad():
                total_loss, loss_components = model.compute_prediction_loss(
                    predicted_params, target_params, silhouette_data=silhouette_data, return_components=True, loss_weights=loss_weights
                )
            
            print(f"   Total loss with silhouette: {total_loss.item():.8f}")
            if 'silhouette' in loss_components:
                silhouette_loss = loss_components['silhouette'].item()
                status = "✓ PASS" if silhouette_loss < tolerance_silhouette else "✗ FAIL"
                print(f"   silhouette      : {silhouette_loss:.8f} {status}")
                
                test_results['with_silhouette'] = {
                    'total_loss': total_loss.item(),
                    'silhouette_loss': silhouette_loss,
                    'passed': silhouette_loss < tolerance_silhouette
                }
            else:
                print("   No silhouette loss computed")
                test_results['with_silhouette'] = {'passed': False, 'error': 'No silhouette loss'}
        
        # Test 4: Combined losses (if both available)
        if keypoint_data is not None and silhouette_data is not None:
            print("\n4. Testing with both keypoint and silhouette losses (optimized path)...")
            # Set loss weights to enable both keypoint and silhouette loss computation
            loss_weights = {
                'global_rot': 1.0,
                'joint_rot': 1.0,
                'betas': 1.0,
                'trans': 1.0,
                'fov': 1.0,
                'cam_rot': 1.0,
                'cam_trans': 1.0,
                'log_beta_scales': 1.0,
                'betas_trans': 1.0,
                'keypoint_2d': 1.0,  # Enable keypoint loss
                'silhouette': 1.0    # Enable silhouette loss
            }
            with torch.no_grad():
                total_loss, loss_components = model.compute_prediction_loss(
                    predicted_params, target_params, 
                    pose_data=keypoint_data, silhouette_data=silhouette_data, 
                    return_components=True, loss_weights=loss_weights
                )
            
            print(f"   Total combined loss: {total_loss.item():.8f}")
            
            combined_losses = {}
            for loss_name in ['keypoint_2d', 'silhouette']:
                if loss_name in loss_components:
                    loss_val = loss_components[loss_name].item()
                    status = "✓ PASS" if loss_val < tolerance_silhouette else "✗ FAIL"
                    print(f"   {loss_name:15s}: {loss_val:.8f} {status}")
                    combined_losses[loss_name] = loss_val
            
            test_results['combined'] = {
                'total_loss': total_loss.item(),
                'components': combined_losses,
                'passed': all(v < tolerance_silhouette for v in combined_losses.values())
            }
        
        # Generate visual comparisons if enabled
        if enable_visualization:
            print("\n" + "=" * 60)
            print("GENERATING VISUAL COMPARISONS")
            print("=" * 60)
            
            output_dir = "test_visualizations"
            
            # Generate keypoint visualization if keypoint data is available
            if 'keypoints_2d' in y_data and 'keypoint_visibility' in y_data:
                visualize_keypoints_comparison(model, predicted_params, y_data, x_data, backbone_name, dataset, output_dir)
            
            # Generate silhouette visualization if silhouette data is available
            if x_data.get("input_image_mask") is not None:
                visualize_silhouette_comparison(model, predicted_params, x_data, output_dir)
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        all_passed = True
        
        # Include rotation conversion test in summary if performed
        if test_rotation_conversion:
            if conversion_result.get('success', False):
                print(f"✓ {'rotation_conversion':20s}: PASSED")
            else:
                print(f"✗ {'rotation_conversion':20s}: FAILED")
                all_passed = False
        
        for test_name, result in test_results.items():
            if result.get('passed', False):
                print(f"✓ {test_name:20s}: PASSED")
            else:
                print(f"✗ {test_name:20s}: FAILED")
                if 'error' in result:
                    print(f"  Error: {result['error']}")
                all_passed = False
        
        print(f"\nOverall result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        if not all_passed:
            print(f"\nNote: Losses should be < {tolerance_keypoints} for ground truth parameters.")
            print("Higher losses may indicate:")
            print("- Numerical precision issues")
            print("- Coordinate system mismatches")
            print("- Rendering pipeline inconsistencies")
        
        if enable_visualization:
            print(f"\nVisualizations saved to: {os.path.abspath(output_dir)}")
        
        result_dict = {
            "success": all_passed,
            "results": test_results,
            "tolerance_keypoints": tolerance_keypoints,
            "tolerance_silhouette": tolerance_silhouette
        }
        
        # Include rotation conversion results if performed
        if test_rotation_conversion:
            result_dict["rotation_conversion"] = conversion_result
            
        return result_dict
        
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    """Main function to run the ground truth test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SMILImageRegressor with ground truth parameters')
    parser.add_argument('--data_path', type=str, default=None, 
                       help='Path to the dataset (optional, will try default paths)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of the sample to test')
    parser.add_argument('--tolerance_keypoints', type=float, default=1e-2,
                       help='Maximum acceptable loss value for keypoints')
    parser.add_argument('--tolerance_silhouette', type=float, default=0.05,
                       help='Maximum acceptable loss value for silhouette')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu or cuda)')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Disable visual comparisons generation')
    parser.add_argument('--no-ue-scaling', action='store_true',
                       help='Disable UE scaling (override dataset default)')
    parser.add_argument('--rotation-representation', type=str, default='axis_angle', choices=['6d', 'axis_angle'],
                       help='Rotation representation for joint rotations (default: axis_angle)')
    parser.add_argument('--no-rotation-test', action='store_true',
                       help='Disable rotation conversion round-trip test')
    parser.add_argument('--backbone', type=str, default='resnet152',
                       choices=['resnet50', 'resnet101', 'resnet152', 'vit_base_patch16_224', 'vit_large_patch16_224'],
                       help='Backbone network to use (default: resnet152)')
    
    args = parser.parse_args()
    
    # Set up environment
    if args.device == 'cuda':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("Warning: CUDA requested but not available, using CPU")
    else:
        device = "cpu"
    
    # Run test
    result = test_ground_truth_loss(
        data_path=args.data_path,
        sample_idx=args.sample_idx,
        tolerance_keypoints=args.tolerance_keypoints,
        tolerance_silhouette=args.tolerance_silhouette,
        device=device,
        enable_visualization=not args.no_visualization,
        override_ue_scaling=False if args.no_ue_scaling else None,
        rotation_representation=args.rotation_representation,
        test_rotation_conversion=not args.no_rotation_test,
        backbone_name=args.backbone
    )
    
    # Exit with appropriate code
    exit_code = 0 if result.get("success", False) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
