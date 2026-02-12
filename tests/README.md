# SMILify Tests

This directory contains integration tests for the SMILify pipeline components.

## Running Tests

### Quick Start (Recommended)
Use the provided test script which handles environment activation automatically:

```bash
# Fast configuration test only
./run_neural_smil_tests.sh config

# Full training pipeline test (slow)
./run_neural_smil_tests.sh training

# All neural SMIL tests
./run_neural_smil_tests.sh all
```

### Manual Test Execution
```bash
# Activate environment first
conda activate pytorch3d

# All tests (excluding slow tests)
pytest tests/test_pipeline.py -v -s

# Specific test functions
pytest tests/test_pipeline.py::test_neural_smil_config_validation -v -s
pytest tests/test_pipeline.py::test_neural_smil_training_pipeline -v -s
```

### Including Slow Tests
```bash
# Run all tests including slow ones
pytest tests/test_pipeline.py -v -s -m "slow or not slow"

# Run only slow tests
pytest tests/test_pipeline.py -v -s -m "slow"
```

## Test Descriptions

### `test_neural_smil_config_validation()`
- **Speed**: Fast (< 1 second)
- **Dependencies**: None (pure Python)
- **Purpose**: Validates that the neural SMIL training configuration system works correctly
- **Tests**:
  - Configuration import and initialization
  - Dataset path resolution for `test_textured` dataset
  - Loss curriculum computation across different epochs
  - Train/validation/test split calculations
  - Dataset directory existence

### `test_neural_smil_training_pipeline()` 
- **Speed**: Slow (~5 seconds)
- **Dependencies**: PyTorch, CUDA (optional)
- **Purpose**: Full integration test of the neural SMIL training pipeline
- **Configuration**:
  - Uses `test_textured` dataset (replicAnt-x-SMIL-TEX)
  - 2 epochs for quick validation
  - Batch size of 4 for low memory usage
  - Learning rate of 0.001 for faster convergence
  - **Checkpoint Protection**: Uses temporary directory, no overwriting of existing models
- **Validates**:
  - Dataset loading and preprocessing
  - Model initialization and training loop
  - Loss computation with curriculum
  - Training pipeline execution without data corruption
  - Training completion
- **Safety Features**:
  - Automatic temporary directory creation for test outputs
  - Disabled checkpoint saving to prevent overwriting trained models
  - Cleanup of all test artifacts after completion

### `test_fitter_3d_optimise()`
- Tests the 3D mesh fitting optimization pipeline

### `test_smal_fitter_optimize_to_joints()`
- Tests the SMAL model joint optimization pipeline

## Test Data

The neural SMIL tests use the `test_textured` dataset located at:
- `data/replicAnt_trials/replicAnt-x-SMIL-TEX/`

This dataset contains 20 images with corresponding SMIL parameter annotations, making it ideal for integration testing.

## CI/CD Integration

For continuous integration, it's recommended to:

1. **Run fast tests by default**:
   ```bash
   pytest tests/test_pipeline.py -v -m "not slow"
   ```

2. **Run slow tests on scheduled builds or manual triggers**:
   ```bash
   pytest tests/test_pipeline.py -v -m "slow"
   ```

This approach ensures fast feedback for regular development while still validating the full pipeline periodically.
