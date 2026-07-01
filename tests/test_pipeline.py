import pytest
import subprocess
import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


# Entrypoints are launched as modules (`python -m pkg.mod`) from the repo root, so
# absolute imports resolve regardless of where pytest itself runs. Running a script
# by path would instead put the script's own dir on sys.path and break those imports.
def run_script(module, args=[], env=None):
    command = [sys.executable, "-m", module] + args
    result = subprocess.run(command, capture_output=True, text=True, cwd=parent_dir, env=env)
    print(f"\nOutput from -m {module}:")
    print(result.stdout)
    if result.stderr:
        print(f"Output from -m {module}:")
        print(result.stderr)
    return result


def run_script_with_env(module, args=[], env=None):
    """Run an entrypoint module (`python -m`) with custom environment variables."""
    return run_script(module, args, env=env)


# NOTE: marked `slow` — this launches the optimisation-based fitter as a subprocess,
# which does pytorch3d rasterisation/rendering. That is fine on a GPU but pathologically
# slow on CPU, so it is excluded from the CPU-only CI run (`pytest -m "not slow"`) and
# meant to be run on a GPU (`pytest -m slow` or the full `pytest`).
@pytest.mark.slow
def test_fitter_3d_optimise(capsys):
    result = run_script(
        "fitter_3d.optimise",
        ["--mesh_dir", "fitter_3d/ATTA_BOI", "--scheme", "default", "--lr", "1e-3", "--nits", "10"],
    )
    assert result.returncode == 0, f"fitter_3d.optimise failed with error:\n{result.stderr}"

    # Capture and print the output
    captured = capsys.readouterr()
    print(captured.out)


# NOTE: marked `slow` — same reason as test_fitter_3d_optimise: it runs the
# optimisation-based joint fitter (pytorch3d rendering) as a subprocess, which is
# GPU-appropriate and far too slow for the CPU-only CI run.
@pytest.mark.slow
def test_smal_fitter_optimize_to_joints(capsys):
    result = run_script("smal_fitter.optimize_to_joints", ["--test"])
    assert result.returncode == 0, f"smal_fitter.optimize_to_joints failed with error:\n{result.stderr}"

    # Capture and print the output
    captured = capsys.readouterr()
    print(captured.out)


def test_neural_smil_config_validation():
    """Test neural SMIL training configuration logic (pure, no filesystem dependencies)."""
    from smal_fitter.neuralSMIL.training_config import TrainingConfig

    # Test dataset path resolution
    test_textured_path = TrainingConfig.get_data_path("test_textured")
    assert test_textured_path == "data/replicAnt_trials/replicAnt-x-SMIL-TEX"

    # Test that requesting an unknown dataset raises ValueError
    with pytest.raises(ValueError):
        TrainingConfig.get_data_path("nonexistent_dataset")

    # Test train/val/test split calculation
    train_size, val_size, test_size = TrainingConfig.get_train_val_test_sizes(100)
    assert train_size == 85, f"Expected train size 85, got {train_size}"
    assert val_size == 5, f"Expected val size 5, got {val_size}"
    assert test_size == 10, f"Expected test size 10, got {test_size}"

    # Test loss curriculum returns weights dict
    weights = TrainingConfig.get_loss_weights_for_epoch(0)
    assert isinstance(weights, dict)
    assert "keypoint_2d" in weights

    # Test learning rate curriculum
    lr = TrainingConfig.get_learning_rate_for_epoch(0)
    assert lr > 0


@pytest.mark.slow
def test_neural_smil_training_pipeline(capsys):
    """Test neural SMIL training pipeline with minimal configuration.

    This test is marked as slow and requires PyTorch dependencies.
    Run with: pytest -m slow tests/test_pipeline.py::test_neural_smil_training_pipeline
    """
    import tempfile
    import shutil

    # Create a temporary directory for test outputs to avoid overwriting real training data
    temp_dir = tempfile.mkdtemp(prefix="neural_smil_test_")

    try:
        # Run with test_textured dataset and minimal epochs for quick integration test
        # Use environment variable to override checkpoint directory for testing
        test_env = os.environ.copy()
        test_env["PYTEST_TEMP_DIR"] = temp_dir

        result = run_script_with_env(
            "smal_fitter.neuralSMIL.train_smil_regressor",
            [
                "--dataset",
                "test_textured",
                "--num_epochs",
                "2",  # Very minimal training for integration test
                "--batch_size",
                "4",  # Small batch size for testing
                "--checkpoint",
                "DISABLE_CHECKPOINT_LOADING",  # Disable checkpoint loading for test
                "--scale_trans_mode",
                "ignore",  # Compatible with test_textured dataset
            ],
            test_env,
        )

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
