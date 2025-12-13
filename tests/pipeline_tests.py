import pytest
import subprocess
import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def run_script(script_path, args=[]):
    command = [sys.executable, script_path] + args
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"\nOutput from {script_path}:")
    print(result.stdout)
    if result.stderr:
        print(f"Output from {script_path}:")
        print(result.stderr)
    return result

def run_script_with_env(script_path, args=[], env=None):
    """Run script with custom environment variables."""
    command = [sys.executable, script_path] + args
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    print(f"\nOutput from {script_path}:")
    print(result.stdout)
    if result.stderr:
        print(f"Output from {script_path}:")
        print(result.stderr)
    return result

def test_fitter_3d_optimise(capsys):
    script_path = os.path.join(parent_dir, 'fitter_3d', 'optimise.py')
    result = run_script(script_path, ['--mesh_dir', 'fitter_3d/ATTA_BOI', '--scheme', 'default', '--lr', '1e-3', '--nits', '10'])
    assert result.returncode == 0, f"fitter_3d/optimise.py failed with error:\n{result.stderr}"
    
    # Capture and print the output
    captured = capsys.readouterr()
    print(captured.out)

def test_smal_fitter_optimize_to_joints(capsys):
    script_path = os.path.join(parent_dir, 'smal_fitter', 'optimize_to_joints.py')
    result = run_script(script_path, ['--test'])
    assert result.returncode == 0, f"smal_fitter/optimize_to_joints.py failed with error:\n{result.stderr}"
    
    # Capture and print the output
    captured = capsys.readouterr()
    print(captured.out)

def test_neural_smil_config_validation():
    """Test neural SMIL training configuration validation without running full training."""
    import sys
    sys.path.append(os.path.join(parent_dir, 'smal_fitter', 'neuralSMIL'))
    
    # Test that the configuration can be imported and used
    try:
        # Import from the correct path
        sys.path.insert(0, os.path.join(parent_dir, 'smal_fitter', 'neuralSMIL'))
        from training_config import TrainingConfig
        
        # Test dataset path resolution
        test_textured_path = TrainingConfig.get_data_path('test_textured')
        assert test_textured_path == "data/replicAnt_trials/replicAnt-x-SMIL-TEX"
        
        # Test that the dataset directory exists
        full_path = os.path.join(parent_dir, test_textured_path)
        assert os.path.exists(full_path), f"Test dataset directory not found: {full_path}"
        
   
        # Test train/val/test split calculation
        train_size, val_size, test_size = TrainingConfig.get_train_val_test_sizes(100)
        assert train_size == 85, f"Expected train size 85, got {train_size}"
        assert val_size == 5, f"Expected val size 5, got {val_size}"
        assert test_size == 10, f"Expected test size 10, got {test_size}"
        
        print("✅ Neural SMIL configuration validation passed")
        
    except ImportError as e:
        pytest.fail(f"Failed to import training configuration: {e}")
    except Exception as e:
        pytest.fail(f"Configuration validation failed: {e}")

@pytest.mark.slow
def test_neural_smil_training_pipeline(capsys):
    """Test neural SMIL training pipeline with minimal configuration.
    
    This test is marked as slow and requires PyTorch dependencies.
    Run with: pytest -m slow tests/pipeline_tests.py::test_neural_smil_training_pipeline
    """
    import tempfile
    import shutil
    
    # Create a temporary directory for test outputs to avoid overwriting real training data
    temp_dir = tempfile.mkdtemp(prefix='neural_smil_test_')
    
    try:
        # Create a temporary test configuration that overrides training parameters for quick testing
        script_path = os.path.join(parent_dir, 'smal_fitter', 'neuralSMIL', 'train_smil_regressor.py')
        
        # Run with test_textured dataset and minimal epochs for quick integration test
        # Use environment variable to override checkpoint directory for testing
        test_env = os.environ.copy()
        test_env['PYTEST_TEMP_DIR'] = temp_dir
        
        result = run_script_with_env(script_path, [
            '--dataset', 'test_textured',
            '--num-epochs', '2',  # Very minimal training for integration test
            '--batch-size', '4',  # Small batch size for testing
            '--checkpoint', 'DISABLE_CHECKPOINT_LOADING'  # Disable checkpoint loading for test
        ], test_env)
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # The training script should complete successfully (exit code 0)
    assert result.returncode == 0, f"neural SMIL training pipeline failed with error:\n{result.stderr}"
    
    # Check that key training outputs are present in the output
    assert "Loading dataset..." in result.stdout, "Dataset loading not detected in output"
    assert "Dataset size:" in result.stdout, "Dataset size info not found in output" 
    assert "Train set:" in result.stdout, "Train set info not found in output"
    assert "Epoch 0:" in result.stdout, "Training epoch 0 not detected"
    assert "Training completed!" in result.stdout, "Training completion message not found"
    
    # Capture and print the output
    captured = capsys.readouterr()
    print(captured.out)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
